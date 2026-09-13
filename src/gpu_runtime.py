"""GPU コンポーネント（CUDA provider DLL ＋ NVIDIA ランタイム DLL）の任意ダウンロード。

設計は docs/260910_gpu_acceleration_impl_plan.md の Phase 2。Qt 非依存。UI からは
workers.GpuRuntimeDownloadWorker が薄く包んで呼ぶ。

- 取得対象はビルドに同梱する `gpu_components.json`（RESOURCE_DIR 直下）で宣言する。
  アプリのバージョンとコンポーネントは 1:1 で紐付く（version-lock）。
- `direct`  … 単体ファイル（`onnxruntime_providers_cuda.dll` を GitHub Release から等）。
- `wheels`  … NVIDIA 公式 PyPI wheel（再ホストしない）。zip から必要な DLL だけ取り出す。
- すべて staging に落として SHA-256 検証してから **すべて gpu_runtime/ へ** `os.replace`
  （`location="capi"` のものも含む — gpu_runtime/ が唯一の持ち出し可能な source of
  truth。capi/ への複製は onnx_providers.preload_gpu_dlls() が毎起動やる）。最後に
  `gpu_runtime/manifest.json` を書く。途中で失敗/中断したら manifest を書かないので
  onnx_providers.gpu_runtime_ready() は False のまま（再試行で上書きされる）。
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from utils import calculate_sha256, log_dbg
from onnx_providers import capi_dir, gpu_runtime_dir

SCHEMA = 1
COMPONENT_SPEC_NAME = "gpu_components.json"
_MANIFEST_NAME = "manifest.json"
_STAGING_DIRNAME = ".staging"
# SHA-256 欄がこの値なら「未確定」（Windows 実機でリリース時に埋める）。検証をスキップする。
SHA_PLACEHOLDER_PREFIX = "TODO"

ProgressCb = Callable[[int, int], None]   # (done_bytes, total_bytes)
LogCb = Callable[[str, str], None]        # (message, level)  level: "info"|"warn"|"error"
StopCb = Callable[[], bool]


class GpuRuntimeError(RuntimeError):
    """インストール処理の想定内の失敗（ネットワーク・ハッシュ不一致・書き込み不可 等）。"""


class _Stopped(GpuRuntimeError):
    """stop_cb が True を返した（利用者がキャンセル）。失敗ではないので扱いを分ける。"""


# --- component spec --------------------------------------------------------

def component_spec_path(resource_dir: Path | None = None) -> Path:
    if resource_dir is None:
        from constants import RESOURCE_DIR

        resource_dir = RESOURCE_DIR
    return Path(resource_dir) / COMPONENT_SPEC_NAME


def _is_pinned_sha256(value: Any) -> bool:
    """64 桁 hex の SHA-256 か。プレースホルダ（"TODO..."）や欠落は False。

    このモジュールは DL した DLL を onnxruntime の capi/ に置いてプロセスにロードする。
    ハッシュ未確定の spec を配布物として受け入れると、MITM に対して丸腰になる。
    そのため load_component_spec は全エントリが pin 済みでなければ spec ごと拒否する
    （インストーラ側は防御多重化として TODO を許容し警告ログのみ）。
    """
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def load_component_spec(resource_dir: Path | None = None) -> dict | None:
    """同梱 gpu_components.json を読む。存在しない/壊れている/スキーマ不一致/
    ハッシュ未確定は None（＝ GPU プロンプトを出さない）。"""
    path = component_spec_path(resource_dir)
    try:
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    direct = data.get("direct", [])
    wheels = data.get("wheels", [])
    if not isinstance(direct, list) or not isinstance(wheels, list) or not (direct or wheels):
        return None
    if not all(isinstance(e, dict) and e.get("name") and e.get("url") and _is_pinned_sha256(e.get("sha256"))
               for e in direct):
        return None
    if not all(isinstance(e, dict) and e.get("url") and _is_pinned_sha256(e.get("sha256"))
               and isinstance(e.get("members"), list) and e["members"]
               for e in wheels):
        return None
    # onnx_providers._read_ready_files() version-locks gpu_runtime/ against the
    # running onnxruntime via this field; a spec that omits it would silently skip
    # that check (CodeRabbit review, PR #21) and let a mismatched provider DLL load.
    if not isinstance(data.get("ort_version"), str) or not data["ort_version"].strip():
        return None
    return data


def spec_total_bytes(spec: dict) -> int:
    """ダウンロード見込みバイト数（`bytes` 欄の合計。未記入は 0 扱い）。プロンプト表示用。"""
    total = 0
    for item in list(spec.get("direct", [])) + list(spec.get("wheels", [])):
        if isinstance(item, dict):
            try:
                total += int(item.get("bytes", 0) or 0)
            except (TypeError, ValueError):
                pass
    return total


# --- installer -----------------------------------------------------------

@dataclass
class _Planned:
    """staging に落とした後、所定位置へ配置する 1 ファイル分。"""
    name: str
    staged: Path
    location: str          # "gpu_runtime" | "capi"
    sha256: str


@dataclass
class GpuRuntimeInstaller:
    base_dir: Path | None = None
    ort_module: Any = field(default=None, repr=False)
    http_get: Callable[..., Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._root = gpu_runtime_dir(self.base_dir)
        self._staging = self._root / _STAGING_DIRNAME
        if self.http_get is None:
            self.http_get = _requests_get
        if self.ort_module is None:
            self._capi = capi_dir()
        else:
            self._capi = capi_dir(self.ort_module)

    # -- public -----------------------------------------------------------

    def install(self, spec: dict, *, progress_cb: ProgressCb | None = None,
                log_cb: LogCb | None = None, stop_cb: StopCb | None = None) -> bool:
        log = log_cb or (lambda m, lv="info": log_dbg(f"gpu_runtime[{lv}]: {m}"))
        stop = stop_cb or (lambda: False)
        total = spec_total_bytes(spec) or 0
        done = 0

        def bump(n: int) -> None:
            nonlocal done
            done += n
            if progress_cb:
                progress_cb(done, total)

        try:
            self._reset_staging()
            planned: list[_Planned] = []

            for item in spec.get("direct", []):
                if stop():
                    raise _Stopped("stopped")
                planned.append(self._fetch_direct(item, log, stop, bump))

            for item in spec.get("wheels", []):
                if stop():
                    raise _Stopped("stopped")
                planned.extend(self._fetch_wheel(item, log, stop, bump))

            if not planned:
                raise GpuRuntimeError("gpu_components.json produced no files")

            self._place(planned, log)
            self._write_manifest(spec, planned)
            log("GPU components installed; restart to enable GPU inference")
            return True
        except _Stopped:
            log("download cancelled", "warn")
            return False
        except GpuRuntimeError as exc:
            log(f"install aborted: {exc}", "error")
            return False
        except Exception as exc:  # noqa: BLE001 - network / zip / io
            log(f"install failed: {exc!r}", "error")
            return False
        finally:
            self._reset_staging(remove_only=True)

    def uninstall(self) -> None:
        """gpu_runtime/ と capi に置いた provider DLL を消す（破損時の作り直し用）。"""
        capi_names: list[str] = []
        manifest = self._root / _MANIFEST_NAME
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            capi_names = [e["name"] for e in data.get("files", [])
                          if isinstance(e, dict) and e.get("location") == "capi" and e.get("name")]
        except (OSError, ValueError, KeyError, TypeError):
            capi_names = ["onnxruntime_providers_cuda.dll"]
        if self._capi is not None:
            for name in capi_names:
                # CodeRabbit/cubic review (PR #21): a tampered or corrupted manifest
                # could otherwise carry a "../../..." name and unlink() outside capi/.
                # install() already rejects unsafe names before they ever reach a
                # manifest we wrote ourselves, but uninstall() must not trust an
                # on-disk manifest it didn't just validate.
                if not _is_safe_filename(name):
                    log_dbg(f"gpu_runtime: uninstall skipping unsafe capi entry {name!r}")
                    continue
                try:
                    (self._capi / name).unlink(missing_ok=True)
                except OSError:
                    pass
        shutil.rmtree(self._root, ignore_errors=True)

    # -- internals ------------------------------------------------------

    def _reset_staging(self, *, remove_only: bool = False) -> None:
        shutil.rmtree(self._staging, ignore_errors=True)
        if not remove_only:
            self._staging.mkdir(parents=True, exist_ok=True)

    def _fetch_direct(self, item: Any, log: LogCb, stop: StopCb,
                      bump: Callable[[int], None]) -> _Planned:
        if not isinstance(item, dict):
            raise GpuRuntimeError("invalid 'direct' entry")
        name = item.get("name")
        url = item.get("url")
        if not name or not url:
            raise GpuRuntimeError("'direct' entry needs name and url")
        if not _is_safe_filename(name):
            raise GpuRuntimeError(f"unsafe direct name {name!r}")
        location = item.get("location", "gpu_runtime")
        if location not in ("gpu_runtime", "capi"):
            raise GpuRuntimeError(f"unknown location {location!r}")
        dest = self._staging / name
        log(f"downloading {name}")
        self._download(url, dest, stop, bump)
        sha = self._verify(dest, item.get("sha256"), name)
        return _Planned(name=name, staged=dest, location=location, sha256=sha)

    def _fetch_wheel(self, item: Any, log: LogCb, stop: StopCb,
                     bump: Callable[[int], None]) -> list[_Planned]:
        if not isinstance(item, dict):
            raise GpuRuntimeError("invalid 'wheels' entry")
        url = item.get("url")
        members = item.get("members")
        if not url or not isinstance(members, list) or not members:
            raise GpuRuntimeError("'wheels' entry needs url and members[]")
        whl = self._staging / (_basename(url) or "component.whl")
        log(f"downloading {whl.name}")
        self._download(url, whl, stop, bump)
        self._verify(whl, item.get("sha256"), whl.name)

        out: list[_Planned] = []
        with zipfile.ZipFile(whl) as zf:
            names = set(zf.namelist())
            for member in members:
                if stop():
                    raise _Stopped("stopped")
                if not isinstance(member, dict):
                    raise GpuRuntimeError("invalid wheel member")
                arcname = member.get("arcname")
                out_name = member.get("name") or (_basename(arcname) if arcname else None)
                if not arcname or not out_name:
                    raise GpuRuntimeError("wheel member needs arcname")
                if not _is_safe_filename(out_name):
                    raise GpuRuntimeError(f"unsafe member name {out_name!r}")
                location = member.get("location", "gpu_runtime")
                if location not in ("gpu_runtime", "capi"):
                    raise GpuRuntimeError(f"unknown member location {location!r}")
                if arcname not in names:
                    raise GpuRuntimeError(f"{whl.name} has no member {arcname}")
                staged = self._staging / out_name
                with zf.open(arcname) as src, open(staged, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                out.append(_Planned(name=out_name, staged=staged, location=location,
                                    sha256=calculate_sha256(staged)))
        whl.unlink(missing_ok=True)
        return out

    def _download(self, url: str, dest: Path, stop: StopCb, bump: Callable[[int], None]) -> None:
        # staging は attempt ごとに作り直すので、常に頭から取得する（再開はしない）。
        part = dest.with_name(dest.name + ".part")
        resp = self.http_get(url, headers={}, stream=True, timeout=30)
        try:
            resp.raise_for_status()
            with open(part, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if stop():
                        raise _Stopped("stopped")
                    if not chunk:
                        continue
                    f.write(chunk)
                    bump(len(chunk))
        finally:
            close = getattr(resp, "close", None)
            if callable(close):
                close()
        os.replace(part, dest)

    def _verify(self, path: Path, expected: Any, label: str) -> str:
        actual = calculate_sha256(path)
        if isinstance(expected, str) and expected and not expected.upper().startswith(SHA_PLACEHOLDER_PREFIX):
            if actual.lower() != expected.lower():
                raise GpuRuntimeError(f"SHA-256 mismatch for {label}")
        else:
            log_dbg(f"gpu_runtime: {label} has no pinned SHA-256; skipping integrity check")
        return actual

    def _place(self, planned: list[_Planned], log: LogCb) -> None:
        # 全ファイルを常に gpu_runtime/ へ置く（location="capi" のものも含む）。
        # gpu_runtime/ が唯一の持ち出し可能な source of truth になり、コピーするだけで
        # 別ビルド/別マシンでも動くようにするため（Phase 4 で判明した「capi/ にしか
        # 無いファイルの存在まで求めると、コピーしただけでは『未整備』判定になる」
        # 問題への対応）。capi/ への複製は onnx_providers.preload_gpu_dlls() が
        # 毎起動 gpu_runtime/ を見て自動でやる（ONNX Runtime は provider DLL を自分と
        # 同じディレクトリからしか探さないため、capi/ への複製自体は避けられない）。
        for p in planned:
            try:
                self._root.mkdir(parents=True, exist_ok=True)
                os.replace(p.staged, self._root / p.name)
            except OSError as exc:
                raise GpuRuntimeError(f"cannot write {self._root / p.name}: {exc}") from exc
        log(f"placed {len(planned)} file(s)")

    def _write_manifest(self, spec: dict, planned: list[_Planned]) -> None:
        payload = {
            "schema": SCHEMA,
            "ort_version": spec.get("ort_version", ""),
            "files": [{"name": p.name, "location": p.location, "sha256": p.sha256} for p in planned],
        }
        self._root.mkdir(parents=True, exist_ok=True)
        tmp = self._root / (_MANIFEST_NAME + ".part")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self._root / _MANIFEST_NAME)


def _is_safe_filename(name: Any) -> bool:
    """A single path component, no separators, no `.`/`..`/empty. Used for every
    filename that comes from a manifest/spec before it's joined onto a real path
    (download destination, capi/ placement, uninstall cleanup).

    `:` is rejected too (cubic review, PR #21, P1): on Windows, `pathlib`'s `/`
    treats a drive-qualified name like "C:evil.dll" as anchored, so
    `Path(root) / "C:evil.dll"` silently discards `root` entirely and resolves to
    `C:evil.dll` relative to the current directory on C: - completely escaping
    staging/capi/gpu_runtime. NTFS Alternate Data Stream names ("legit.dll:hide")
    also contain `:` and are blocked the same way.
    """
    return (isinstance(name, str) and name not in ("", ".", "..")
            and "/" not in name and "\\" not in name and ":" not in name)


def _basename(url: str | None) -> str:
    if not url:
        return ""
    return url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]


def _requests_get(url: str, *, headers: dict | None = None, stream: bool = True, timeout: int = 30):
    import requests

    return requests.get(url, headers=headers or {}, stream=stream, timeout=timeout)


__all__ = [
    "COMPONENT_SPEC_NAME", "GpuRuntimeError", "GpuRuntimeInstaller",
    "component_spec_path", "load_component_spec", "spec_total_bytes",
]
