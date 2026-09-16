"""gpu_runtime のインストーラ / コンポーネント spec と、onnx_providers.gpu_runtime_ready
の "files" 形式（Phase 2: docs/260910_gpu_acceleration_impl_plan.md）。

Offline only - no network, no GUI. Fake http_get で完結。
Run:  rtk pytest tests/test_gpu_runtime.py -q
"""
import hashlib
import io
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gpu_runtime as GR
import onnx_providers as OP


PROV_BYTES = b"FAKE-onnxruntime_providers_cuda.dll-" + b"x" * 500
CUDNN_BYTES = b"FAKE-cudnn64_9.dll-" + b"y" * 400


def _wheel_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for arcname, data in members.items():
            zf.writestr(arcname, data)
    return buf.getvalue()


WHEEL_BYTES = _wheel_bytes({"nvidia/cudnn/bin/cudnn64_9.dll": CUDNN_BYTES,
                            "nvidia_cudnn_cu12-9.0.0.dist-info/METADATA": b"Name: x\n"})


class _Resp:
    def __init__(self, data: bytes):
        self._data = data
        self.status_code = 200
        self.headers = {"content-length": str(len(data))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i:i + chunk_size]

    def close(self):
        pass


def _http_get(mapping: dict[str, bytes]):
    def _get(url, *, headers=None, stream=True, timeout=30):
        if url not in mapping:
            raise RuntimeError(f"fake 404 for {url}")
        return _Resp(mapping[url])
    return _get


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fake_ort(tmp_path: Path, *, with_preload: bool = False, version: str | None = None):
    """capi_dir() が <tmp>/onnxruntime/capi を返すようにする onnxruntime 代役。"""
    capi_parent = tmp_path / "onnxruntime"
    capi_parent.mkdir(parents=True, exist_ok=True)
    ns = types.SimpleNamespace(__file__=str(capi_parent / "__init__.py"))
    if version is not None:
        ns.__version__ = version
    if with_preload:
        ns.preload_dlls = lambda cuda=False, cudnn=False, directory=None: None
    return ns


def _spec(prov_sha=None, whl_sha=None):
    return {
        "schema": 1,
        "ort_version": "1.23.1",
        "direct": [{
            "name": "onnxruntime_providers_cuda.dll",
            "url": "http://host/prov.dll",
            "sha256": _sha(PROV_BYTES) if prov_sha is None else prov_sha,
            "bytes": len(PROV_BYTES),
            "location": "capi",
        }],
        "wheels": [{
            "url": "http://host/nvidia_cudnn_cu12-9.0.0-py3-none-win_amd64.whl",
            "sha256": _sha(WHEEL_BYTES) if whl_sha is None else whl_sha,
            "bytes": len(WHEEL_BYTES),
            "members": [{"arcname": "nvidia/cudnn/bin/cudnn64_9.dll", "name": "cudnn64_9.dll"}],
        }],
    }


_URLS = {
    "http://host/prov.dll": PROV_BYTES,
    "http://host/nvidia_cudnn_cu12-9.0.0-py3-none-win_amd64.whl": WHEEL_BYTES,
}


def _installer(tmp_path, http=None):
    return GR.GpuRuntimeInstaller(base_dir=tmp_path, ort_module=_fake_ort(tmp_path),
                                  http_get=http or _http_get(_URLS))


# --- load_component_spec -------------------------------------------------

def test_spec_missing_file(tmp_path):
    assert GR.load_component_spec(tmp_path) is None


def test_spec_bad_json(tmp_path):
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text("{ nope", encoding="utf-8")
    assert GR.load_component_spec(tmp_path) is None


def test_spec_wrong_schema(tmp_path):
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(json.dumps({"schema": 99, "direct": [{}]}), encoding="utf-8")
    assert GR.load_component_spec(tmp_path) is None


def test_spec_empty(tmp_path):
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(
        json.dumps({"schema": 1, "direct": [], "wheels": []}), encoding="utf-8")
    assert GR.load_component_spec(tmp_path) is None


def test_spec_valid(tmp_path):
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(json.dumps(_spec()), encoding="utf-8")
    spec = GR.load_component_spec(tmp_path)
    assert spec is not None and spec["ort_version"] == "1.23.1"


def test_spec_direct_missing_url(tmp_path):
    bad = _spec()
    del bad["direct"][0]["url"]
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(json.dumps(bad), encoding="utf-8")
    assert GR.load_component_spec(tmp_path) is None


def test_spec_wheel_missing_members(tmp_path):
    bad = _spec()
    bad["wheels"][0]["members"] = []
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(json.dumps(bad), encoding="utf-8")
    assert GR.load_component_spec(tmp_path) is None


def test_spec_unpinned_sha_rejected(tmp_path):
    for bad_sha in ("TODO_FILL_ON_WINDOWS", "deadbeef", "", "z" * 64):
        bad = _spec(prov_sha=bad_sha)
        (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(json.dumps(bad), encoding="utf-8")
        assert GR.load_component_spec(tmp_path) is None, bad_sha


def test_spec_total_bytes_tolerates_garbage():
    spec = {"direct": [{"bytes": 10}, {"bytes": "x"}, {}], "wheels": [{"bytes": 5}]}
    assert GR.spec_total_bytes(spec) == 15


def test_spec_missing_ort_version_rejected(tmp_path):
    """Without ort_version, onnx_providers' version-lock check can't run at all
    and silently never rejects a mismatched gpu_runtime/ (CodeRabbit/cubic review,
    PR #21) - so the spec itself must require it."""
    bad = _spec()
    del bad["ort_version"]
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(json.dumps(bad), encoding="utf-8")
    assert GR.load_component_spec(tmp_path) is None

    blank = _spec()
    blank["ort_version"] = "   "
    (tmp_path / GR.COMPONENT_SPEC_NAME).write_text(json.dumps(blank), encoding="utf-8")
    assert GR.load_component_spec(tmp_path) is None


# --- install happy path ------------------------------------------------

def test_install_places_files_and_marks_ready(tmp_path):
    ok = _installer(tmp_path).install(_spec())
    assert ok is True

    # Everything lands in gpu_runtime/ - including the capi-tagged provider DLL.
    # It is NOT copied into onnxruntime/capi/ at install time (that only happens at
    # startup, via preload_gpu_dlls -> _mirror_capi_files; see below).
    prov = tmp_path / OP.GPU_RUNTIME_DIRNAME / "onnxruntime_providers_cuda.dll"
    cudnn = tmp_path / OP.GPU_RUNTIME_DIRNAME / "cudnn64_9.dll"
    assert prov.read_bytes() == PROV_BYTES
    assert cudnn.read_bytes() == CUDNN_BYTES
    assert not (tmp_path / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll").exists()

    manifest = json.loads((tmp_path / OP.GPU_RUNTIME_DIRNAME / "manifest.json").read_text())
    names = {f["name"]: f["location"] for f in manifest["files"]}
    assert names == {"onnxruntime_providers_cuda.dll": "capi", "cudnn64_9.dll": "gpu_runtime"}
    assert manifest["ort_version"] == "1.23.1"

    assert OP.gpu_runtime_ready(tmp_path, ort_module=_fake_ort(tmp_path)) is True
    # staging must be gone
    assert not (tmp_path / OP.GPU_RUNTIME_DIRNAME / ".staging").exists()


def test_preload_mirrors_capi_file_from_gpu_runtime(tmp_path):
    """The capi-tagged file gets copied into onnxruntime/capi/ at startup
    (preload_gpu_dlls), not at install time - this is what lets gpu_runtime/ be
    copied wholesale to a different build's exe directory and still work."""
    assert _installer(tmp_path).install(_spec()) is True
    capi_file = tmp_path / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll"
    assert not capi_file.exists()

    ort = _fake_ort(tmp_path, with_preload=True, version="1.23.1")
    assert OP.preload_gpu_dlls(base_dir=tmp_path, ort_module=ort) is True
    assert capi_file.read_bytes() == PROV_BYTES


def test_gpu_runtime_dir_is_portable_across_builds(tmp_path):
    """Copying only gpu_runtime/ (no capi file) to a fresh 'build' still works: the
    manifest + files under gpu_runtime/ are enough for gpu_runtime_ready(), and
    preload_gpu_dlls() restores the capi mirror for whatever onnxruntime is running
    there. This is the exact scenario a user hit copying gpu_runtime/ between a
    source run and a built exe."""
    src = tmp_path / "src_build"
    assert _installer(src).install(_spec()) is True

    dst = tmp_path / "other_build"
    dst.mkdir()
    shutil.copytree(src / OP.GPU_RUNTIME_DIRNAME, dst / OP.GPU_RUNTIME_DIRNAME)
    # deliberately do NOT copy anything under src/onnxruntime/capi/

    other_ort = _fake_ort(dst, with_preload=True, version="1.23.1")
    assert OP.gpu_runtime_ready(dst, ort_module=other_ort) is True
    assert OP.preload_gpu_dlls(base_dir=dst, ort_module=other_ort) is True
    assert (dst / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll").read_bytes() == PROV_BYTES


def test_install_skips_placeholder_sha(tmp_path):
    ok = _installer(tmp_path).install(_spec(prov_sha="TODO_FILL_ON_WINDOWS", whl_sha="TODO"))
    assert ok is True
    assert (tmp_path / OP.GPU_RUNTIME_DIRNAME / "onnxruntime_providers_cuda.dll").is_file()


# --- install failure modes -------------------------------------------

def _assert_not_ready(tmp_path):
    assert not (tmp_path / OP.GPU_RUNTIME_DIRNAME / "manifest.json").exists()
    assert OP.gpu_runtime_ready(tmp_path, ort_module=_fake_ort(tmp_path)) is False


def test_install_sha_mismatch_aborts(tmp_path):
    ok = _installer(tmp_path).install(_spec(prov_sha="deadbeef" * 8))
    assert ok is False
    _assert_not_ready(tmp_path)
    # Resume support (see test_install_retry_after_failure_resumes_download below)
    # means staging itself is no longer wiped on failure, but a file that failed
    # SHA-256 verification must still be gone - otherwise a retry would see it as
    # "already downloaded" and never re-fetch the corrupted bytes.
    assert not (tmp_path / OP.GPU_RUNTIME_DIRNAME / ".staging" / "onnxruntime_providers_cuda.dll").exists()


class _BreakingResp:
    """Yields `partial` then raises, simulating a connection drop mid-transfer."""
    def __init__(self, partial: bytes):
        self._partial = partial
        self.status_code = 200
        self.headers = {"content-length": str(len(self._partial))}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=65536):
        yield self._partial
        raise RuntimeError("simulated connection drop")

    def close(self):
        pass


class _FlakyThenResumableResp:
    """First call: breaks after `break_at` bytes (see _BreakingResp). Any later
    call: honors a Range header like a real HTTP server, returning 206 with only
    the remaining bytes - so `_download()`'s resume path gets tested honestly
    rather than by a fake that always just hands back the full file."""
    def __init__(self, data: bytes, break_at: int):
        self._data = data
        self._break_at = break_at
        self._used = False

    def __call__(self, url, *, headers=None, stream=True, timeout=30):
        headers = headers or {}
        if not self._used:
            self._used = True
            return _BreakingResp(self._data[:self._break_at])
        rng = headers.get("Range")
        assert rng, "resume attempt must send a Range header"
        start = int(rng.split("=", 1)[1].split("-", 1)[0])
        remainder = self._data[start:]
        resp = _Resp(remainder)
        resp.status_code = 206
        resp.headers["content-range"] = f"bytes {start}-{len(self._data) - 1}/{len(self._data)}"
        return resp


def test_install_retry_after_failure_resumes_download(tmp_path):
    """A connection drop mid-download must not force the whole ~2GB back to zero:
    the second install() attempt should resume the `.part` file via a Range
    request instead of re-fetching bytes already on disk (cubic/CodeRabbit never
    flagged this - restores intent dropped from docs/260910_gpu_acceleration_options.md
    during implementation, per user request)."""
    prov_url = "http://host/prov.dll"
    flaky = _FlakyThenResumableResp(PROV_BYTES, break_at=len(PROV_BYTES) // 2)

    def http(url, **kw):
        if url == prov_url:
            return flaky(url, **kw)
        return _http_get(_URLS)(url, **kw)

    inst = _installer(tmp_path, http=http)

    assert inst.install(_spec()) is False  # breaks mid-download of the direct item
    part = tmp_path / OP.GPU_RUNTIME_DIRNAME / ".staging" / "onnxruntime_providers_cuda.dll.part"
    assert part.is_file()
    assert part.stat().st_size == len(PROV_BYTES) // 2

    assert inst.install(_spec()) is True  # resumes, then finishes normally
    assert OP.gpu_runtime_ready(tmp_path, ort_module=_fake_ort(tmp_path, version="1.23.1")) is True
    placed = tmp_path / OP.GPU_RUNTIME_DIRNAME / "onnxruntime_providers_cuda.dll"
    assert placed.read_bytes() == PROV_BYTES


class _Resp416:
    """A server's real reply when a Range request's offset is already at (or past)
    the resource's full length: nothing left to send from there."""
    def __init__(self):
        self.status_code = 416
        self.headers: dict = {}

    def raise_for_status(self):
        raise RuntimeError("raise_for_status() must not be reached for this 416")

    def iter_content(self, chunk_size=65536):
        return iter(())

    def close(self):
        pass


def test_download_promotes_complete_part_on_416(tmp_path):
    """A `.part` file that already holds every byte (the server has nothing left
    to send from that offset, hence 416) must be promoted directly instead of
    treated as a hard failure - otherwise every retry would hit the same 416
    forever, since the Range request never changes (CodeRabbit review, PR #23,
    confirmed valid: this can happen when a prior attempt finished writing the
    file but was interrupted before the os.replace() that follows)."""
    inst = _installer(tmp_path)
    inst._staging.mkdir(parents=True, exist_ok=True)
    dest = inst._staging / "onnxruntime_providers_cuda.dll"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(PROV_BYTES)  # fully downloaded already, just never promoted

    def http_416(url, *, headers=None, stream=True, timeout=30):
        assert headers and headers.get("Range") == f"bytes={len(PROV_BYTES)}-"
        return _Resp416()

    inst.http_get = http_416
    bumped = []
    inst._download("http://host/prov.dll", dest, stop=lambda: False, bump=bumped.append)

    assert dest.read_bytes() == PROV_BYTES
    assert not part.exists()
    assert bumped == [len(PROV_BYTES)]  # progress still accounted for


def test_fetch_wheel_skips_network_when_whl_already_downloaded(tmp_path):
    """If the .whl itself is already sitting in staging (fully, matching bytes)
    from an earlier attempt, _fetch_wheel must not hit the network again -
    staging survives across attempts now (see install()), so this is the common
    case after e.g. a later item in the same spec fails. Extraction itself still
    runs fresh from the wheel (see the next test for why)."""
    def boom(*a, **k):
        raise AssertionError("must not re-download an already-staged .whl")

    inst = _installer(tmp_path, http=boom)
    inst._staging.mkdir(parents=True, exist_ok=True)
    whl_name = GR._basename("http://host/nvidia_cudnn_cu12-9.0.0-py3-none-win_amd64.whl")
    (inst._staging / whl_name).write_bytes(WHEEL_BYTES)

    item = _spec()["wheels"][0]
    planned = inst._fetch_wheel(item, log=lambda *a, **k: None, stop=lambda: False, bump=lambda n: None)

    assert len(planned) == 1
    assert planned[0].name == "cudnn64_9.dll"
    assert planned[0].staged.read_bytes() == CUDNN_BYTES


def test_fetch_wheel_overwrites_truncated_member_from_interrupted_extraction(tmp_path):
    """A member file already present in staging must NOT be trusted just because
    it exists (CodeRabbit review, PR #23, confirmed valid): if the process was
    killed mid-`shutil.copyfileobj` during an earlier attempt's extraction, the
    staged member file can be truncated/wrong even though it "exists". Since the
    .whl itself is re-verified (or, as here, already present and skip-downloaded)
    on every attempt, extraction must always re-run and overwrite any such stale
    bytes - never skip extraction based on the member file's mere presence."""
    inst = _installer(tmp_path)  # normal http_get; the .whl download itself is exercised too
    inst._staging.mkdir(parents=True, exist_ok=True)
    # Simulate a member left truncated/wrong by an interrupted prior extraction.
    (inst._staging / "cudnn64_9.dll").write_bytes(b"TRUNCATED-GARBAGE")

    item = _spec()["wheels"][0]
    planned = inst._fetch_wheel(item, log=lambda *a, **k: None, stop=lambda: False, bump=lambda n: None)

    assert len(planned) == 1
    assert planned[0].staged.read_bytes() == CUDNN_BYTES, \
        "stale/truncated bytes from an earlier interrupted extraction must be overwritten"


def test_install_stop_aborts(tmp_path):
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    ok = _installer(tmp_path).install(_spec(), stop_cb=stop)
    assert ok is False
    _assert_not_ready(tmp_path)


def test_install_stop_logs_warn_not_error(tmp_path):
    logs: list[tuple[str, str]] = []
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 2

    _installer(tmp_path).install(_spec(), stop_cb=stop,
                                 log_cb=lambda m, lv="info": logs.append((m, lv)))
    assert not any(lv == "error" for _, lv in logs)
    assert any(lv == "warn" for _, lv in logs)


def test_install_unsafe_member_name_aborts(tmp_path):
    spec = _spec()
    spec["wheels"][0]["members"] = [
        {"arcname": "nvidia/cudnn/bin/cudnn64_9.dll", "name": "../evil.dll"}]
    ok = _installer(tmp_path).install(spec)
    assert ok is False
    _assert_not_ready(tmp_path)


def test_install_unsafe_direct_name_aborts(tmp_path):
    spec = _spec()
    spec["direct"][0]["name"] = "../../evil.dll"
    ok = _installer(tmp_path).install(spec)
    assert ok is False
    _assert_not_ready(tmp_path)


def test_install_missing_wheel_member_aborts(tmp_path):
    spec = _spec()
    spec["wheels"][0]["members"] = [{"arcname": "nvidia/cudnn/bin/nonexistent.dll", "name": "nonexistent.dll"}]
    ok = _installer(tmp_path).install(spec)
    assert ok is False
    _assert_not_ready(tmp_path)


def test_install_succeeds_even_when_capi_dir_unresolvable(tmp_path):
    """Placement no longer needs onnxruntime's capi/ dir - only the startup mirror
    step (preload_gpu_dlls) does, and it degrades gracefully (see next test)."""
    inst = GR.GpuRuntimeInstaller(base_dir=tmp_path, ort_module=types.SimpleNamespace(),
                                  http_get=_http_get(_URLS))
    ok = inst.install(_spec())
    assert ok is True
    assert OP.gpu_runtime_ready(tmp_path, ort_module=types.SimpleNamespace()) is True


def test_preload_mirror_noop_when_capi_dir_unresolvable(tmp_path, monkeypatch):
    assert _installer(tmp_path).install(_spec()) is True
    unresolvable = types.SimpleNamespace(preload_dlls=lambda **kw: None)  # no __file__
    monkeypatch.setattr(OP, "log_dbg", lambda *a, **k: None)
    # preload_gpu_dlls still proceeds (PATH/preload_dlls), just can't mirror into capi.
    assert OP.preload_gpu_dlls(base_dir=tmp_path, ort_module=unresolvable) is True


def test_install_download_404_aborts(tmp_path):
    ok = _installer(tmp_path, http=_http_get({})).install(_spec())
    assert ok is False
    _assert_not_ready(tmp_path)


# --- uninstall -------------------------------------------------------

def test_uninstall_removes_everything(tmp_path):
    inst = _installer(tmp_path)
    assert inst.install(_spec()) is True
    # mirror the capi file first, so uninstall has something real to clean up there
    ort = _fake_ort(tmp_path, with_preload=True, version="1.23.1")
    assert OP.preload_gpu_dlls(base_dir=tmp_path, ort_module=ort) is True
    capi_file = tmp_path / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll"
    assert capi_file.is_file()

    inst.uninstall()
    assert not (tmp_path / OP.GPU_RUNTIME_DIRNAME).exists()
    assert not capi_file.exists()


def test_uninstall_rejects_path_traversal_in_manifest(tmp_path):
    """A tampered/corrupted manifest naming a capi entry like "../../evil.dll"
    must not let unlink() escape onnxruntime's capi/ directory (cubic review,
    PR #21, confidence 10).

    Geometry: _fake_ort(tmp_path) makes capi/ resolve to
    tmp_path/onnxruntime/capi/, so it takes TWO ".." to reach tmp_path/ itself
    (one ".." only reaches tmp_path/onnxruntime/) - a previous version of this
    test used one ".." and so victim.txt was never actually at the resolved
    path, meaning it would have passed even with the traversal guard removed
    (cubic review, PR #21, confidence 9)."""
    ort = _fake_ort(tmp_path)
    victim = tmp_path / "victim.txt"  # sits *outside* capi/, two levels up
    victim.write_text("do not delete me")

    root = tmp_path / OP.GPU_RUNTIME_DIRNAME
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "schema": 1, "ort_version": "1.23.1",
        "files": [{"name": "../../victim.txt", "location": "capi"}],
    }), encoding="utf-8")

    GR.GpuRuntimeInstaller(base_dir=tmp_path, ort_module=ort).uninstall()

    assert victim.is_file(), "path traversal must not delete files outside capi/"
    assert not root.exists()  # the rest of the (legitimate) cleanup still happens


def test_uninstall_leaves_capi_alone_when_manifest_unreadable(tmp_path):
    """No readable gpu_runtime/manifest.json must NOT fall back to guessing
    "onnxruntime_providers_cuda.dll" and deleting it from capi/ (cubic +
    CodeRabbit review, PR #21, confidence 8-9): in a source run, pip's
    onnxruntime-gpu wheel ships that exact file in capi/ on its own (confirmed
    against a real install - it's what makes CUDAExecutionProvider available
    before this app ever downloads anything), so a corrupted/missing local
    manifest must not be able to delete the package's own file. Only
    gpu_runtime/ itself - which this app owns outright - gets cleaned up."""
    ort = _fake_ort(tmp_path)
    capi_file = tmp_path / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll"
    capi_file.parent.mkdir(parents=True)
    capi_file.write_bytes(PROV_BYTES)  # stands in for the pip package's own file

    root = tmp_path / OP.GPU_RUNTIME_DIRNAME
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("not valid json", encoding="utf-8")

    GR.GpuRuntimeInstaller(base_dir=tmp_path, ort_module=ort).uninstall()

    assert capi_file.is_file(), "must not guess-delete a file it doesn't own"
    assert capi_file.read_bytes() == PROV_BYTES
    assert not root.exists()  # gpu_runtime/ itself is still cleaned up


# --- gpu_runtime_ready "files" form ---------------------------------

def test_ready_files_form_missing_from_gpu_runtime_root(tmp_path):
    """Readiness only looks at gpu_runtime/ root, regardless of a file's `location`
    tag - a capi-tagged file absent from onnxruntime/capi/ does NOT matter; absent
    from gpu_runtime/ itself does."""
    root = tmp_path / OP.GPU_RUNTIME_DIRNAME
    root.mkdir(parents=True)
    (root / "cudnn64_9.dll").write_bytes(b"x")
    # deliberately don't create onnxruntime_providers_cuda.dll anywhere
    (root / "manifest.json").write_text(json.dumps({
        "schema": 1, "ort_version": "1.23.1",
        "files": [{"name": "cudnn64_9.dll", "location": "gpu_runtime"},
                  {"name": "onnxruntime_providers_cuda.dll", "location": "capi"}],
    }), encoding="utf-8")
    assert OP.gpu_runtime_ready(tmp_path, ort_module=_fake_ort(tmp_path)) is False


def test_ready_files_ort_version_mismatch_is_not_ready(tmp_path):
    root = tmp_path / OP.GPU_RUNTIME_DIRNAME
    root.mkdir(parents=True)
    (root / "cudnn64_9.dll").write_bytes(b"x")
    (root / "manifest.json").write_text(json.dumps({
        "schema": 1, "ort_version": "1.23.1",
        "files": [{"name": "cudnn64_9.dll", "location": "gpu_runtime"}],
    }), encoding="utf-8")
    same = _fake_ort(tmp_path, version="1.23.1")
    other = _fake_ort(tmp_path, version="1.24.0")
    assert OP.gpu_runtime_ready(tmp_path, ort_module=same) is True
    assert OP.gpu_runtime_ready(tmp_path, ort_module=other) is False


def test_wheel_member_capi_location(tmp_path):
    """A wheels entry may route a member to capi/ (used for onnxruntime_providers_cuda.dll):
    it still lands in gpu_runtime/ at install time, tagged for a later capi mirror."""
    spec = _spec()
    spec["direct"] = []
    spec["wheels"][0]["members"] = [
        {"arcname": "nvidia/cudnn/bin/cudnn64_9.dll", "name": "onnxruntime_providers_cuda.dll",
         "location": "capi"}]
    assert _installer(tmp_path).install(spec) is True
    assert (tmp_path / OP.GPU_RUNTIME_DIRNAME / "onnxruntime_providers_cuda.dll").is_file()
    assert not (tmp_path / "onnxruntime" / "capi" / "onnxruntime_providers_cuda.dll").exists()
    m = json.loads((tmp_path / OP.GPU_RUNTIME_DIRNAME / "manifest.json").read_text())
    assert m["files"][0]["location"] == "capi"
    assert OP.gpu_runtime_ready(tmp_path, ort_module=_fake_ort(tmp_path)) is True


def test_wheel_member_bad_location_aborts(tmp_path):
    spec = _spec()
    spec["wheels"][0]["members"][0]["location"] = "somewhere_else"
    assert _installer(tmp_path).install(spec) is False
    _assert_not_ready(tmp_path)


def test_ready_files_form_all_present(tmp_path):
    """Both files only need to exist under gpu_runtime/ - the capi-tagged one does
    NOT need a copy under onnxruntime/capi/ for readiness (that's preload's job)."""
    root = tmp_path / OP.GPU_RUNTIME_DIRNAME
    root.mkdir(parents=True)
    (root / "cudnn64_9.dll").write_bytes(b"x")
    (root / "onnxruntime_providers_cuda.dll").write_bytes(b"x")
    (root / "manifest.json").write_text(json.dumps({
        "schema": 1, "files": [
            {"name": "cudnn64_9.dll", "location": "gpu_runtime"},
            {"name": "onnxruntime_providers_cuda.dll", "location": "capi"}],
    }), encoding="utf-8")
    assert OP.gpu_runtime_ready(tmp_path, ort_module=_fake_ort(tmp_path)) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
