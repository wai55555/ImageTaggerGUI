"""MainWindow GPU-download workflow fixes from the PR #21 review (offscreen).

Covers: the progress dialog's destroyed-signal bookkeeping (WA_DeleteOnClose can
free it out from under us) and cancellation being reported separately from a
genuine failure. Does not drive a real download (no network in tests).

Run:  rtk pytest tests/test_gpu_main_window.py -q
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog
from PySide6.QtCore import Qt

_APP = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Isolate config.ini so MainWindow.closeEvent's save doesn't pollute the
    real one. Scoped via monkeypatch (per test, auto-undone) rather than the
    former module-level `_A.CONFIG_PATH = ...` at import time, which permanently
    repointed the shared app_settings/constants modules for the whole pytest
    session regardless of collection order (cubic + CodeRabbit review, PR #21).
    """
    import app_settings as _A
    import constants as _C
    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(_A, "CONFIG_PATH", config_path)
    monkeypatch.setattr(_C, "CONFIG_PATH", config_path)


@pytest.fixture(autouse=True)
def _no_gpu_prompt(monkeypatch):
    """On a machine that actually has an NVIDIA GPU + CUDA-enabled onnxruntime,
    MainWindow()'s initial_load() would otherwise reach _maybe_prompt_gpu_setup()
    and pop a real, unpatched QMessageBox that hangs forever under the offscreen
    platform. Most tests here don't care about that prompt, so force it off by
    default; test_never_on_repair_prompt_clears_broken_gpu_runtime overrides this
    back to True for the one test that specifically exercises the prompt flow
    (cubic review, PR #21: was duplicated ad hoc in three tests before)."""
    import onnx_providers as OP
    monkeypatch.setattr(OP, "has_nvidia_gpu", lambda *a, **k: False)


def _mw_ready():
    import main_window
    w = main_window.MainWindow()
    _APP.processEvents()
    _APP.processEvents()  # let QTimer.singleShot(0, initial_load) run
    return w


def _fake_progress_dialog(mw):
    progress = QProgressDialog("x", "Cancel", 0, 100, mw)
    progress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    progress.canceled.connect(mw._cancel_gpu_runtime_download)
    progress.destroyed.connect(mw._on_gpu_dl_progress_destroyed)
    mw._gpu_dl_progress = progress
    return progress


def test_progress_dialog_destroyed_clears_reference():
    """WA_DeleteOnClose can free the dialog from a user close; the destroyed
    signal must be what keeps mw._gpu_dl_progress in sync (cubic review, PR #21)."""
    w = _mw_ready()
    progress = _fake_progress_dialog(w)
    assert w._gpu_dl_progress is progress

    progress.show()  # WA_DeleteOnClose only schedules deletion for a shown widget
    _APP.processEvents()
    progress.close()  # schedules WA_DeleteOnClose's deferred deletion
    _APP.processEvents()  # let the DeferredDelete event (and `destroyed`) run

    assert w._gpu_dl_progress is None
    w.close()


def test_finished_handler_survives_dialog_already_destroyed(monkeypatch):
    """_on_gpu_runtime_finished must not raise when the dialog died earlier
    (e.g. the user closed it) and _gpu_dl_progress is already None."""
    w = _mw_ready()
    w._gpu_dl_progress = None
    w._gpu_dl_thread = QThread()
    from workers import GpuRuntimeDownloadWorker
    worker = GpuRuntimeDownloadWorker(w.locale_manager.get_string)
    w._gpu_dl_worker = worker

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    w._on_gpu_runtime_finished(False)  # must not raise
    assert warned, "a genuine failure (not cancelled) should still warn"
    w.close()


def test_finished_handler_suppresses_popup_on_cancellation(monkeypatch):
    """cancelling a download must not show the generic 'download failed' dialog
    (CodeRabbit review, PR #21): install() collapses cancel and failure into the
    same `False`, so the handler must check the worker's own stop flag."""
    w = _mw_ready()
    w._gpu_dl_progress = None
    w._gpu_dl_thread = QThread()
    from workers import GpuRuntimeDownloadWorker
    worker = GpuRuntimeDownloadWorker(w.locale_manager.get_string)
    worker.stop()  # simulate: user cancelled
    w._gpu_dl_worker = worker

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(("warning", a)))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(("info", a)))
    w._on_gpu_runtime_finished(False)
    assert shown == [], "cancellation must not pop any dialog"
    w.close()


def test_never_on_repair_prompt_clears_broken_gpu_runtime(monkeypatch, tmp_path):
    """Selecting "Never" while the repair prompt is showing (gpu_runtime/ exists
    but is broken) must clear it - otherwise `partial` stays true forever and the
    same prompt re-appears every launch despite the user saying Never twice
    (CodeRabbit review, PR #21).

    All patches (including QMessageBox) must be in place *before* the MainWindow
    is built: initial_load() -> _maybe_prompt_gpu_setup() already fires once via
    the QTimer.singleShot(0, ...) that _mw_ready() pumps: an unpatched QMessageBox
    at that point would show a real modal and hang the (offscreen) test forever.
    """
    import onnxruntime
    import onnx_providers as OP
    import gpu_runtime as GR

    monkeypatch.setattr(onnxruntime, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    # this test needs the prompt path to actually run, unlike the _no_gpu_prompt
    # fixture's default (see that fixture's docstring)
    monkeypatch.setattr(OP, "has_nvidia_gpu", lambda *a, **k: True)
    # a broken gpu_runtime/: present, but not gpu_runtime_ready() (no manifest)
    broken = tmp_path / OP.GPU_RUNTIME_DIRNAME
    broken.mkdir(parents=True)
    (broken / "stray.dll").write_bytes(b"x")
    monkeypatch.setattr(OP, "gpu_runtime_dir", lambda base_dir=None: broken)
    monkeypatch.setattr(OP, "gpu_runtime_ready", lambda *a, **k: False)
    monkeypatch.setattr(GR, "load_component_spec",
                        lambda *a, **k: {"schema": 1, "wheels": [{"bytes": 100}]})

    uninstall_calls = []
    monkeypatch.setattr(GR.GpuRuntimeInstaller, "uninstall",
                        lambda self: uninstall_calls.append(True))

    class _FakeBox:
        """addButton() is called 3x (download/later/never, in that order); the
        real code compares clickedButton() by identity, so each must be distinct
        and clickedButton() must return the specific one "clicked" (here: Never,
        the 3rd, so this fake always simulates the user picking Never)."""
        Icon = QMessageBox.Icon
        ButtonRole = QMessageBox.ButtonRole

        def __init__(self, *a, **k):
            self._buttons = []
        def setIcon(self, *a, **k): pass
        def setWindowTitle(self, *a, **k): pass
        def setText(self, *a, **k): pass
        def addButton(self, label, role):
            btn = object()
            self._buttons.append(btn)
            return btn
        def exec(self):
            pass
        def clickedButton(self):
            return self._buttons[-1]

    import main_window as MW
    monkeypatch.setattr(MW, "QMessageBox", _FakeBox)

    w = _mw_ready()  # initial_load()'s automatic call already exercises the flow once
    w._maybe_prompt_gpu_setup()  # call again explicitly for a deterministic 2nd pass

    assert w.settings.behavior.gpu_setup_prompt == "dismissed"
    assert uninstall_calls, "uninstall() must run to clear the broken gpu_runtime/"
    w.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
