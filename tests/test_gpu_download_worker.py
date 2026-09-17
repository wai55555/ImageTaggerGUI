"""GpuRuntimeDownloadWorker._run()'s uninstall-before-reinstall gating.

A bug fixed in this session: _run() used to call GpuRuntimeInstaller().uninstall()
whenever gpu_runtime/ existed at all, which wipes the whole directory including
.staging/ - silently defeating gpu_runtime.py's download-resume support (added the
same session) on *every* retry, since a prior cancelled/failed attempt always
leaves gpu_runtime/ (and .staging/ inside it) present with no manifest.json yet.

Run:  rtk pytest tests/test_gpu_download_worker.py -q
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])


def _worker():
    from workers import GpuRuntimeDownloadWorker
    return GpuRuntimeDownloadWorker(lambda section, key, **kw: key)


def test_run_does_not_uninstall_when_only_staging_exists(monkeypatch, tmp_path):
    """gpu_runtime/ present with only a .staging/ subdirectory (no manifest.json) -
    the exact state a resumable prior attempt leaves behind - must NOT be wiped."""
    import gpu_runtime as GR
    import onnx_providers as OP

    root = tmp_path / OP.GPU_RUNTIME_DIRNAME
    (root / GR._STAGING_DIRNAME).mkdir(parents=True)
    (root / GR._STAGING_DIRNAME / "partial.whl.part").write_bytes(b"resumable bytes")
    monkeypatch.setattr(OP, "gpu_runtime_dir", lambda base_dir=None: root)

    monkeypatch.setattr(GR, "load_component_spec", lambda *a, **k: {"schema": 1, "wheels": []})

    uninstall_calls = []
    monkeypatch.setattr(GR.GpuRuntimeInstaller, "uninstall",
                        lambda self: uninstall_calls.append(True))
    # install() itself is irrelevant to this test; stub it out so _run() doesn't
    # need a real network layer.
    monkeypatch.setattr(GR.GpuRuntimeInstaller, "install", lambda self, *a, **k: False)

    w = _worker()
    w._run()

    assert uninstall_calls == [], "must not uninstall() while only .staging/ exists"
    assert (root / GR._STAGING_DIRNAME / "partial.whl.part").is_file(), \
        "the resumable partial download must survive"


def test_run_uninstalls_stale_completed_install(monkeypatch, tmp_path):
    """gpu_runtime/ with an existing manifest.json means a full install already
    completed before (for a since-changed spec/ORT version) - that stale, fully
    installed state should still be cleared before a fresh install()."""
    import gpu_runtime as GR
    import onnx_providers as OP

    root = tmp_path / OP.GPU_RUNTIME_DIRNAME
    root.mkdir(parents=True)
    (root / GR._MANIFEST_NAME).write_text('{"schema": 1, "files": []}', encoding="utf-8")
    monkeypatch.setattr(OP, "gpu_runtime_dir", lambda base_dir=None: root)

    monkeypatch.setattr(GR, "load_component_spec", lambda *a, **k: {"schema": 1, "wheels": []})

    uninstall_calls = []
    monkeypatch.setattr(GR.GpuRuntimeInstaller, "uninstall",
                        lambda self: uninstall_calls.append(True))
    monkeypatch.setattr(GR.GpuRuntimeInstaller, "install", lambda self, *a, **k: False)

    w = _worker()
    w._run()

    assert uninstall_calls == [True]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
