"""ONNX Runtime Execution Provider の解決とフォールバック（GPU 対応の中核）。

設計の背景は docs/260910_gpu_acceleration_options.md、実装計画は
docs/260910_gpu_acceleration_impl_plan.md を参照。

要点:
- 既定は CPU。GPU（CUDA EP）は「GPU コンポーネントが gpu_runtime/ に揃い、
  かつ ONNX Runtime が CUDAExecutionProvider を報告し、かつセッション生成に
  成功した」ときだけ有効化する。
- 上記のどれか一つでも欠ければ CPUExecutionProvider へ静かに落ちる（起動は止めない）。
- onnx_device="cpu"（既定の実効値）のときのセッション生成は、この機能が入る前の
  ``providers=["CPUExecutionProvider"]`` とバイト等価。
- gpu_runtime/ が唯一の持ち出し可能な source of truth。provider DLL は ONNX
  Runtime の制約で onnxruntime の capi/ にも要るが、それは `preload_gpu_dlls()` が
  毎起動 gpu_runtime/ から自動で複製する（`_mirror_capi_files`）ので、gpu_runtime/
  フォルダをコピーするだけで別ビルド/別マシンでも動く（バージョン不一致は
  `_read_ready_files` が検出して「未整備」扱いにする）。

このモジュールはプラットフォーム非依存のロジックのみ（実際のダウンロードは
gpu_runtime.GpuRuntimeInstaller、起動時の preload 呼び出しは image_tagger_gui.main）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils import log_dbg

if TYPE_CHECKING:
    import onnxruntime as ort  # type: ignore
else:
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        ort = None


# gpu_runtime/ は exe 隣（frozen）/ リポジトリルート（source）に置く。model.onnx と同じ層。
GPU_RUNTIME_DIRNAME = "gpu_runtime"
_MANIFEST_NAME = "manifest.json"

CPU_ONLY: list[str] = ["CPUExecutionProvider"]
CUDA_THEN_CPU: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]

_VALID_DEVICES = ("auto", "cpu", "cuda")

# ``ort_module`` 引数の「未指定」を表すセンチネル。None は「onnxruntime が無い」を
# 意味する正当な値なので区別する必要がある。
_ORT_DEFAULT = object()


def _resolve_ort(ort_module: Any):
    return ort if ort_module is _ORT_DEFAULT else ort_module


def capi_dir(ort_module: Any = _ORT_DEFAULT) -> Path | None:
    """onnxruntime パッケージの capi/ ディレクトリ（provider DLL の設置先）。

    frozen ビルドでも `onnxruntime/__file__` は `_internal/onnxruntime/__init__.py` を
    指すので、その隣の capi/ が得られる。onnxruntime が無い / 場所を特定できない
    ときは None。
    """
    mod = _resolve_ort(ort_module)
    path = getattr(mod, "__file__", None)
    if not path:
        return None
    try:
        return Path(path).resolve().parent / "capi"
    except (OSError, ValueError):  # pragma: no cover - defensive
        return None


def _base_dir(base_dir: Path | None) -> Path:
    if base_dir is not None:
        return Path(base_dir)
    # constants を import すると frozen 判定込みの BASE_DIR が得られる。循環 import を
    # 避けるため関数内 import。
    from constants import BASE_DIR

    return BASE_DIR


def gpu_runtime_dir(base_dir: Path | None = None) -> Path:
    """GPU コンポーネントの設置ディレクトリ（存在は保証しない）。"""
    return _base_dir(base_dir) / GPU_RUNTIME_DIRNAME


def _is_safe_filename(name: Any) -> bool:
    """A single path component, no separators, no `.`/`..`/empty/drive-qualified.

    Duplicated from `gpu_runtime._is_safe_filename` rather than imported: this
    module is the lower-level one (`gpu_runtime.py` imports `capi_dir`/
    `gpu_runtime_dir` from here), so importing back would be circular.

    Applied to every manifest-sourced filename here before it's joined onto
    `gpu_runtime_dir()` (`_read_ready_files`) or `capi_dir()` (`_mirror_capi_files`)
    (cubic review, PR #22, confidence 8): a tampered/corrupted `manifest.json` with
    a `"../../..."` or `"C:evil.dll"` name could otherwise make `_mirror_capi_files`
    read from and write to arbitrary paths outside `gpu_runtime/`/`capi/` - the same
    class of bug `gpu_runtime.uninstall()` was already hardened against.
    """
    return (isinstance(name, str) and name not in ("", ".", "..")
            and "/" not in name and "\\" not in name and ":" not in name)


def _read_ready_files(base_dir: Path | None = None, *, ort_module: Any = _ORT_DEFAULT) -> list[dict] | None:
    """gpu_runtime/ が「使える状態」か検証し、揃っているなら files のリストを返す。

    gpu_runtime/manifest.json を読み、記載された各ファイルが **gpu_runtime/ 直下に**
    実在すれば OK（`location` は「起動時に capi/ へも複製が要るか」のメタ情報でしか
    なく、実在チェック自体は gpu_runtime/ 単体で完結する — model.onnx の DL 判定と
    同じ「ファイルがあるか」チェック。Phase 4 で判明した「capi/ にしか無いファイルの
    存在まで要求すると、gpu_runtime/ を丸ごとコピーしただけでは『未整備』判定になる」
    問題への対応。capi/ への複製は preload_gpu_dlls() が毎起動時に自動でやる）。
    JSON パース失敗・キー欠け・型違い・ファイル欠けはすべて None
    （CLAUDE.md #2: is_file() だけで判断しない）。

    対応する形式:
      - {"files": [{"name": ..., "location": "gpu_runtime"|"capi"}, ...], "ort_version": ...}
        gpu_runtime.GpuRuntimeInstaller が書き出す正式形式。`ort_version` が実行中の
        onnxruntime と食い違う場合は None（provider DLL はビルド単位でバージョンロック
        されており、別バージョン向けの gpu_runtime/ を別マシン/別ビルドへコピーした
        ケースを安全に「未整備」扱いにする）。
      - {"required": ["a.dll", "b.dll", ...]}（旧形式・すべて gpu_runtime/ 直下、バージョン情報なし）
    起動ごとに走るので SHA-256 は取り直さない（存在確認のみ。ハッシュは
    インストール時に検証済み）。
    """
    root = gpu_runtime_dir(base_dir)
    manifest = root / _MANIFEST_NAME
    try:
        if not manifest.is_file():
            return None
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    files = data.get("files")
    if isinstance(files, list) and files:
        manifest_version = data.get("ort_version")
        if isinstance(manifest_version, str) and manifest_version:
            ort_mod = _resolve_ort(ort_module)
            running_version = getattr(ort_mod, "__version__", None) if ort_mod is not None else None
            if running_version and running_version != manifest_version:
                return None  # built for a different onnxruntime-gpu version
        normalized: list[dict] = []
        for entry in files:
            if not isinstance(entry, dict):
                return None
            name = entry.get("name")
            if not _is_safe_filename(name):
                return None
            try:
                if not (root / name).is_file():
                    return None
            except OSError:
                return None
            normalized.append({"name": name, "location": entry.get("location", "gpu_runtime")})
        return normalized

    required = data.get("required")
    if not isinstance(required, list) or not required:
        return None
    for rel in required:
        if not _is_safe_filename(rel):
            return None
        try:
            if not (root / rel).is_file():
                return None
        except OSError:
            return None
    return [{"name": rel, "location": "gpu_runtime"} for rel in required]


def gpu_runtime_ready(base_dir: Path | None = None, *, ort_module: Any = _ORT_DEFAULT) -> bool:
    """gpu_runtime/ が「使える状態」か（`_read_ready_files` の真偽版）。"""
    return _read_ready_files(base_dir, ort_module=ort_module) is not None


def _normalize_device(prefer: Any) -> str:
    value = str(prefer).strip().lower() if prefer is not None else ""
    return value if value in _VALID_DEVICES else "auto"


def resolve_providers(
    prefer: str,
    *,
    ort_module: Any = _ORT_DEFAULT,
    base_dir: Path | None = None,
) -> list[str]:
    """config.ini [Behavior] onnx_device の値から InferenceSession の providers を決める。

    prefer: "auto" | "cpu" | "cuda"（空文字・不正値・None は "auto" 扱い）。
      - "cpu": 常に CPU。
      - "auto": GPU が使えそうなら CUDA、駄目なら CPU（黙って）。
      - "cuda": GPU を積極的に使う。条件未達なら CPU に落ち、warning を残す
               （利用者が明示的に選んだのに効いていない、と分かるように）。
    """
    device = _normalize_device(prefer)
    if device == "cpu":
        return list(CPU_ONLY)

    ort_mod = _resolve_ort(ort_module)
    if ort_mod is None:
        if device == "cuda":
            log_dbg("resolve_providers: onnx_device=cuda but onnxruntime is unavailable; using CPU")
        return list(CPU_ONLY)

    try:
        available = set(ort_mod.get_available_providers())
    except Exception:  # pragma: no cover - defensive
        available = set()

    reasons: list[str] = []
    if "CUDAExecutionProvider" not in available:
        reasons.append("CUDAExecutionProvider not offered by this onnxruntime build")
    # gpu_runtime_ready() は「凍結 exe が自前 DL した CUDA/cuDNN が揃っているか」。
    # onnx_device="cuda" は利用者が自分で CUDA を用意した宣言（例: pip install
    # onnxruntime-gpu + nvidia-*）なので、このゲートは課さず素直に CUDA を試す
    # （駄目なら make_session が CPU へフォールバックし warning を残す）。
    # "auto" は既定なので保守的に、DL 済みのときだけ opt-in する。
    if device != "cuda" and not gpu_runtime_ready(base_dir, ort_module=ort_mod):
        reasons.append("gpu_runtime/ not present or incomplete")

    if not reasons:
        return list(CUDA_THEN_CPU)

    if device == "cuda":
        log_dbg("resolve_providers: onnx_device=cuda requested but falling back to CPU (" + "; ".join(reasons) + ")")
    return list(CPU_ONLY)


def make_session(
    model_path: Path | str,
    *,
    sess_options: Any = None,
    prefer: str = "auto",
    ort_module: Any = _ORT_DEFAULT,
    base_dir: Path | None = None,
    label: str = "session",
):
    """InferenceSession を生成する。GPU を要求して失敗したら CPU で作り直す。

    ``prefer`` が CPU 専用に解決された場合の呼び出しは、この機能が入る前の
    ``ort.InferenceSession(str(model_path), sess_options=sess_options,
    providers=["CPUExecutionProvider"])`` とバイト等価。
    """
    ort_mod = _resolve_ort(ort_module)
    if ort_mod is None:
        raise ImportError("onnxruntime is not available")

    providers = resolve_providers(prefer, ort_module=ort_mod, base_dir=base_dir)
    model_str = str(model_path)
    try:
        session = ort_mod.InferenceSession(model_str, sess_options=sess_options, providers=providers)
    except Exception as exc:
        if providers == list(CPU_ONLY):
            raise
        log_dbg(f"{label}: session creation with {providers} failed ({exc!r}); retrying on CPU")
        session = ort_mod.InferenceSession(model_str, sess_options=sess_options, providers=list(CPU_ONLY))

    try:
        active = list(session.get_providers())
    except Exception:  # pragma: no cover - defensive
        active = []
    if active:
        log_dbg(f"{label}: active execution provider = {active[0]}")
    return session


def preload_gpu_dlls(base_dir: Path | None = None, ort_module: Any = _ORT_DEFAULT) -> bool:
    """CUDA/cuDNN の DLL を最初の InferenceSession より前にロードする。

    最初の InferenceSession 生成より前に一度だけ呼ぶこと（順序を誤るとシステム
    PATH 上の別バージョン cuDNN を掴む）。`main()` の冒頭で呼ぶ。

    - `gpu_runtime/` が揃っていれば（＝自前 DL 済み、または他所からコピーされたもの）
      そこからロード。provider DLL（location="capi"）は onnxruntime の capi/ に
      複製されていないと ORT 自体が見つけられないので、ここで毎回ミラーする
      （`_mirror_capi_files`）。これにより gpu_runtime/ フォルダを別ビルド/別マシンの
      exe 隣にコピーするだけで動く（capi/ は `_internal/` 側にあり、アプリの
      再インストールで消えるので、gpu_runtime/ を単一の持ち出し可能な source of
      truth にしている）。
    - 揃っていなければ、pip で入れた `nvidia-*` wheel（ソース実行の開発者向け）を
      `ort.preload_dlls()` 既定探索で拾う。凍結 exe で未 DL の場合はどちらも
      no-op（`resolve_providers("auto")` が CPU を返すので実害なし）。
    例外はすべて握って返り値で表現（起動は止めない）。
    """
    ort_mod = _resolve_ort(ort_module)
    preload = getattr(ort_mod, "preload_dlls", None) if ort_mod is not None else None
    if not callable(preload):
        return False

    files = _read_ready_files(base_dir, ort_module=ort_mod)
    if files is not None:
        directory_path = gpu_runtime_dir(base_dir)
        directory = str(directory_path)
        _mirror_capi_files(files, directory_path, ort_mod)
        _prepend_dll_search([directory])
        try:
            preload(cuda=True, cudnn=True, directory=directory)
            log_dbg(f"preload_gpu_dlls: loaded GPU runtime DLLs from {directory}")
            return True
        except Exception as exc:  # noqa: BLE001 - ORT raises assorted types here
            log_dbg(f"preload_gpu_dlls: ort.preload_dlls(directory=...) failed ({exc!r})")
            return False

    # Only fall through to the default search when the pip nvidia-* wheels are
    # actually present (they create the `nvidia` namespace package). Otherwise
    # ort.preload_dlls() prints a wall of "Failed to load cudnn64_9.dll ... Please
    # follow ... install CUDA" to stderr on every launch of a plain
    # `pip install onnxruntime-gpu` env, which is just CPU-mode noise.
    nvidia_bin_dirs = _pip_nvidia_bin_dirs()
    if not nvidia_bin_dirs:
        return False
    _prepend_dll_search(nvidia_bin_dirs)
    try:
        preload(cuda=True, cudnn=True)
        log_dbg("preload_gpu_dlls: preloaded CUDA/cuDNN from the pip nvidia-* wheels")
        return True
    except Exception as exc:  # noqa: BLE001
        log_dbg(f"preload_gpu_dlls: default ort.preload_dlls() failed ({exc!r})")
        return False


def _prepend_dll_search(dirs: list[str]) -> None:
    """Make `dirs` searchable for DLL loads, including cuDNN 9's own runtime
    LoadLibrary of engine sublibraries (cudnn_engines_tensor_ir64_9.dll,
    cudnn_ext64_9.dll - added in 9.13+, not in ort.preload_dlls()'s fixed list).

    `os.add_dll_directory` alone is not enough on Windows: cuDNN loads those with
    the legacy search order, which only consults PATH / the app dir / system32.
    So prepend to `PATH` as well.
    """
    if not dirs:
        return
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        for d in dirs:
            try:
                add_dll_directory(d)
            except OSError:
                pass
    if sys.platform.startswith("win"):
        current = os.environ.get("PATH", "")
        have = current.split(os.pathsep)
        new = [d for d in dirs if d not in have]
        if new:
            os.environ["PATH"] = os.pathsep.join(new + ([current] if current else []))


def _mirror_capi_files(files: list[dict], root: Path, ort_module: Any) -> None:
    """gpu_runtime/ 内の location="capi" ファイルを、実行中の onnxruntime の capi/ に
    複製する。ONNX Runtime は provider DLL を自分（onnxruntime.dll / *.pyd）と同じ
    ディレクトリからしか探さないため、gpu_runtime/ に置くだけでは足りない。

    毎起動呼ぶことで、`_internal/` がアプリ更新で再生成されても、あるいは
    gpu_runtime/ フォルダを別ビルドの exe 隣にコピーしただけでも、自動で復元される。
    サイズ一致なら複製済みとみなしてスキップ（200MB 級を毎起動ハッシュ化しない）。
    capi/ の場所が特定できない・書き込めない環境では黙って諦める
    （CPU フォールバックで動くだけ）。
    """
    capi = capi_dir(ort_module)
    if capi is None:
        return
    for entry in files:
        if entry.get("location") != "capi":
            continue
        name = entry.get("name")
        if not _is_safe_filename(name):
            # Defense in depth: `_read_ready_files` already rejects unsafe names
            # before this is ever called with a live manifest, but this function
            # takes a bare `files` list and shouldn't trust it on its own (cubic
            # review, PR #22) - `capi / name` below is a write, not just a read.
            continue
        src = root / name
        dst = capi / name
        try:
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                continue
            capi.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        except OSError as exc:
            log_dbg(f"preload_gpu_dlls: could not mirror {name} into capi/ ({exc!r})")


def _pip_nvidia_bin_dirs() -> list[str]:
    """Directories of the shared libs shipped by the pip `nvidia-*-cu12` wheels, if
    installed: `nvidia/<lib>/bin` on Windows, `nvidia/<lib>/lib` on Linux.

    Empty list when no such wheel is present (plain `pip install onnxruntime-gpu`).
    """
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return []
    roots = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    out: list[str] = []
    for root in roots:
        for pat in ("*/bin", "*/lib"):
            try:
                out.extend(str(sub) for sub in Path(root).glob(pat) if sub.is_dir())
            except OSError:
                continue
    return out


def has_nvidia_gpu(*, run: Any = None) -> bool:
    """Best-effort check for an NVIDIA GPU via `nvidia-smi -L`.

    `nvidia-smi` ships with the NVIDIA driver on both Windows and Linux/WSL (not
    Windows-only). An `onnxruntime-gpu` build always reports
    `CUDAExecutionProvider` as *compiled in*, regardless of whether this machine
    actually has NVIDIA hardware - without this check, AMD/Intel/no-dGPU users
    would be offered a useless ~1.8GB download (cubic review, PR #21).

    Any failure - not installed, no driver, PATH doesn't have it, spawn error,
    times out - means "no NVIDIA GPU found": the safe default, since it just
    means the setup prompt doesn't show (the app keeps working on CPU either way).
    macOS has no NVIDIA GPU support to check for (and onnxruntime-gpu doesn't
    ship there either), so this is only ever meaningfully exercised on Windows/Linux.
    """
    import subprocess

    runner = run or subprocess.run
    try:
        kwargs: dict[str, Any] = {}
        if sys.platform.startswith("win"):
            # avoid a console window flashing in front of this windowed app
            # getattr, not a bare attribute access: the constant only exists on
            # the Windows build of the subprocess module, so even a test that
            # monkeypatches sys.platform to simulate Windows must not crash here
            # when actually running on a non-Windows interpreter.
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = runner(["nvidia-smi", "-L"], capture_output=True, text=True,
                        timeout=3, **kwargs)
    except Exception:
        return False
    return result.returncode == 0 and "GPU" in (result.stdout or "")
