#!/usr/bin/env python3
"""Generate gpu_components.json for a release (Phase 3/4 of the GPU plan).

Cross-platform: it does NOT need onnxruntime-gpu installed - it pulls the exact
Windows wheels straight from PyPI, so it can run from WSL/Linux/macOS just as well
as from the Windows build box.

It writes a `wheels`-only manifest:

  * onnxruntime-gpu  <pinned>  win_amd64 cp310 wheel
       -> extracts onnxruntime/capi/onnxruntime_providers_cuda.dll  (location: capi)
  * nvidia-*-cu12 latest win_amd64 wheels (see NVIDIA_PACKAGES below for which,
    and why - it's evidence-based, not "grab everything")
       -> extracts every */bin/*.dll except _EXCLUDED_DLL_NAMES  (location: gpu_runtime)

Every wheel's SHA-256 + size is pinned. The app downloads these same wheels at
runtime, re-verifies, and extracts the same members. Nothing is self-hosted;
PyPI / files.pythonhosted.org stays the distributor.

Do NOT commit the generated gpu_components.json - keep it release-local.
Re-check the NVIDIA wheel versions against the onnxruntime-gpu CUDA/cuDNN
baseline (1.23.x == CUDA 12.x + cuDNN 9) before publishing.

Usage:
    python tools/gen_gpu_components.py [--ort-version 1.23.2] [--python cp310]
                                       [--out gpu_components.json] [--cache-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

SCHEMA = 1
_ROOT = Path(__file__).resolve().parents[1]

# cu12 runtime packages the ONNX Runtime CUDA EP actually needs. Evidence, not
# guesswork (2026-09-13):
#   - onnxruntime's own official extras_require for onnxruntime-gpu[cuda,cudnn]
#     (https://github.com/microsoft/onnxruntime/pull/23659) lists exactly:
#     cuda={nvrtc, cuda-runtime, cufft, curand}, cudnn={cudnn}. No cublas, no
#     cusparse there - cublas/nvjitlink come in transitively as nvidia-cudnn-cu12's
#     own dependencies (confirmed: a real `pip install` of the extras above pulled
#     in nvidia-nvjitlink-cu12 without it being requested).
#   - PyInstaller's dependency scan of the actual provider DLLs
#     (onnxruntime_providers_cuda.dll / onnxruntime_providers_tensorrt.dll) named
#     cudart64_12, cublasLt64_12, cublas64_12, cufft64_11, cudnn64_9 - never
#     cusparse.
#   - A real `pip install` of this exact set never pulled in nvidia-cusparse-cu12.
# So nvidia-cublas-cu12 stays here explicitly (this tool downloads each wheel
# independently rather than relying on pip's resolver to add it), and
# nvidia-cusparse-cu12 is dropped. Adjust if a future ORT/cuDNN release changes this.
NVIDIA_PACKAGES = [
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-nvjitlink-cu12",
]
_PROVIDER_ARCNAME = "onnxruntime/capi/onnxruntime_providers_cuda.dll"
# nvblas64_12.dll: NVBLAS is a standalone BLAS drop-in shim, not something
# onnxruntime/cuDNN load internally (absent from the PyInstaller dependency scan
# above). Excluded so it isn't downloaded/shipped for nothing.
_EXCLUDED_DLL_NAMES = {"nvblas64_12.dll"}


def _ort_version_from_requirements() -> str | None:
    try:
        txt = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^\s*onnxruntime-gpu==([0-9][0-9.]*)\s*$", txt, re.MULTILINE)
    return m.group(1) if m else None


def _pypi_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - trusted host
        return json.load(resp)


def _pick_wheel(files: list[dict], *, must_contain: tuple[str, ...] = ()) -> dict:
    for f in files:
        name = f.get("filename", "")
        if (f.get("packagetype") == "bdist_wheel" and name.endswith("-win_amd64.whl")
                and all(tok in name for tok in must_contain)):
            return f
    raise SystemExit(f"no matching win_amd64 wheel among {[f.get('filename') for f in files]}")


def _download_and_hash(url: str, cache_dir: Path | None, *,
                       expected_sha256: str | None = None) -> tuple[bytes, str, int]:
    """Return (wheel_bytes, sha256_hex, size). Caches by basename when cache_dir set.

    A cache hit is only trusted when its hash matches PyPI's published
    `expected_sha256` (cubic review, PR #21) - otherwise a stale/corrupted cached
    wheel would get re-recorded as "verified" in the generated manifest. On a
    mismatch the cache entry is refreshed from the network.
    """
    base = url.split("?", 1)[0].rsplit("/", 1)[-1]
    cached_path = cache_dir / base if cache_dir is not None else None
    if cached_path is not None and cached_path.is_file():
        data = cached_path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if expected_sha256 is None or sha.lower() == expected_sha256.lower():
            return data, sha, len(data)
        print(f"  (cache mismatch for {base}, re-downloading)")

    with urllib.request.urlopen(url, timeout=180) as resp:  # noqa: S310
        data = resp.read()
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and sha.lower() != expected_sha256.lower():
        raise SystemExit(f"SHA-256 mismatch downloading {url}: "
                         f"got {sha}, PyPI says {expected_sha256}")
    if cached_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(data)
    return data, sha, len(data)


def _bin_dll_members(whl_bytes: bytes) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with zipfile.ZipFile(io.BytesIO(whl_bytes)) as zf:
        for arc in zf.namelist():
            parts = arc.split("/")
            if (len(parts) >= 2 and parts[-2] == "bin" and arc.lower().endswith(".dll")
                    and parts[-1] not in _EXCLUDED_DLL_NAMES):
                out.append({"arcname": arc, "name": parts[-1]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ort-version", default=_ort_version_from_requirements(),
                    help="onnxruntime-gpu version (default: parsed from requirements.txt)")
    ap.add_argument("--python", default="cp310",
                    help="CPython tag of the frozen build (default cp310)")
    ap.add_argument("--out", default=str(_ROOT / "gpu_components.json"))
    ap.add_argument("--cache-dir", default=None,
                    help="keep downloaded wheels here (lets you validate install offline)")
    args = ap.parse_args()
    if not args.ort_version:
        ap.error("--ort-version is required (requirements.txt had no onnxruntime-gpu== pin)")
    cache = Path(args.cache_dir) if args.cache_dir else None

    wheels = []

    # 1) onnxruntime-gpu -> onnxruntime_providers_cuda.dll into capi/
    meta = _pypi_json(f"https://pypi.org/pypi/onnxruntime-gpu/{args.ort_version}/json")
    f = _pick_wheel(meta["urls"], must_contain=(f"-{args.python}-{args.python}-",))
    print(f"  onnxruntime-gpu {args.ort_version}: {f['filename']}")
    data, sha, size = _download_and_hash(f["url"], cache,
                                         expected_sha256=f.get("digests", {}).get("sha256"))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        if _PROVIDER_ARCNAME not in zf.namelist():
            raise SystemExit(f"{f['filename']} has no {_PROVIDER_ARCNAME}")
    wheels.append({
        "url": f["url"], "sha256": sha, "bytes": size,
        "members": [{"arcname": _PROVIDER_ARCNAME,
                     "name": "onnxruntime_providers_cuda.dll", "location": "capi"}],
    })

    # 2) NVIDIA cu12 runtime -> every */bin/*.dll into gpu_runtime/
    for pkg in NVIDIA_PACKAGES:
        meta = _pypi_json(f"https://pypi.org/pypi/{pkg}/json")
        version = meta["info"]["version"]
        f = _pick_wheel(meta.get("urls", []))
        print(f"  {pkg} {version}: {f['filename']}")
        data, sha, size = _download_and_hash(f["url"], cache,
                                             expected_sha256=f.get("digests", {}).get("sha256"))
        members = _bin_dll_members(data)
        if not members:
            raise SystemExit(f"{pkg}: no */bin/*.dll members in wheel")
        wheels.append({"url": f["url"], "sha256": sha, "bytes": size, "members": members})

    spec = {"schema": SCHEMA, "ort_version": args.ort_version, "direct": [], "wheels": wheels}
    Path(args.out).write_text(json.dumps(spec, indent=2), encoding="utf-8")
    total = sum(w["bytes"] for w in wheels)
    n_dll = sum(len(w["members"]) for w in wheels)
    print(f"\nwrote {args.out}: {len(wheels)} wheels, {n_dll} DLLs, "
          f"~{total / 1024 / 1024:.0f} MB total download")
    print("Do NOT commit gpu_components.json.")


if __name__ == "__main__":
    main()
