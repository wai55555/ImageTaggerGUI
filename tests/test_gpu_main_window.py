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

    `utils.py` and `tagging_core.py` each compute their own `CONFIG_PATH =
    BASE_DIR / "config.ini"` independently rather than importing it from
    `constants` (confirmed in source) - patching only `app_settings`/`constants`
    leaves both of those still pointed at the real config.ini, so
    `write_debug_log`/`DebugSettings` and any tagging_core config access during
    MainWindow init/closeEvent would silently touch the real file (cubic review,
    PR #22). Patch all four.
    """
    import app_settings as _A
    import constants as _C
    import utils as _U
    import tagging_core as _TC
    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(_A, "CONFIG_PATH", config_path)
    monkeypatch.setattr(_C, "CONFIG_PATH", config_path)
    monkeypatch.setattr(_U, "CONFIG_PATH", config_path)
    monkeypatch.setattr(_TC, "CONFIG_PATH", config_path)


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


class _DeadProgressStub:
    """Simulates a QProgressDialog whose C++ object is already deleted but whose
    Python wrapper is still referenced: any attribute access raises RuntimeError,
    matching PySide6's real behavior for a destroyed QObject ("wrapped C++ object
    ... has been deleted"). Used to actually exercise the
    `except (RuntimeError, TypeError)` guard in `_on_gpu_runtime_finished` (cubic
    review, PR #22) - a plain `_gpu_dl_progress = None` never reaches that branch
    at all, since `if self._gpu_dl_progress:` is already False in that case."""

    class _DeadSignal:
        def disconnect(self, *a, **k):
            raise RuntimeError("wrapped C++ object of type QProgressDialog has been deleted")

    canceled = _DeadSignal()

    def close(self):
        raise RuntimeError("wrapped C++ object of type QProgressDialog has been deleted")


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


def test_finished_handler_survives_dead_dialog_wrapper(monkeypatch):
    """_on_gpu_runtime_finished must not raise when `_gpu_dl_progress` still holds
    a reference but the underlying C++ object is already gone - unlike the
    `_gpu_dl_progress = None` case above, this actually drives `disconnect()`/
    `close()` into raising RuntimeError, exercising the `except (RuntimeError,
    TypeError)` guard the None-case test doesn't reach (cubic review, PR #22)."""
    w = _mw_ready()
    w._gpu_dl_progress = _DeadProgressStub()
    w._gpu_dl_thread = QThread()
    from workers import GpuRuntimeDownloadWorker
    worker = GpuRuntimeDownloadWorker(w.locale_manager.get_string)
    w._gpu_dl_worker = worker

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    w._on_gpu_runtime_finished(False)  # must not raise despite the dead wrapper
    assert warned, "a genuine failure should still warn even if progress cleanup raised"
    assert w._gpu_dl_progress is None  # cleared regardless of the RuntimeError
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


def _patch_gpu_usable(monkeypatch, *, usable: bool):
    """Controls the "is GPU actually usable" check ui_main_window._create_input_group
    runs once at construction time to decide use_gpu_check's visibility."""
    import onnxruntime
    import onnx_providers as OP
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if usable else ["CPUExecutionProvider"]
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: providers)
    monkeypatch.setattr(OP, "gpu_runtime_ready", lambda *a, **k: usable)


def test_gpu_checkbox_hidden_when_cuda_provider_unavailable(monkeypatch):
    """No CUDAExecutionProvider (CPU-only onnxruntime build) -> the checkbox must
    not be offered at all, matching _maybe_prompt_gpu_setup's own gating.

    Uses isHidden() rather than isVisible(): the MainWindow in these tests is
    never .show()n, so isVisible() (which also checks the whole ancestor chain)
    would be False for every child widget regardless of setVisible(), making
    that assertion pass even if the gating logic were deleted entirely.
    isHidden() reflects only this widget's own explicit hidden/shown state."""
    _patch_gpu_usable(monkeypatch, usable=False)
    w = _mw_ready()
    assert w.use_gpu_check.isHidden() is True
    w.close()


def test_gpu_checkbox_hidden_when_components_not_downloaded(monkeypatch):
    """CUDAExecutionProvider is compiled in, but GPU components haven't been
    downloaded yet -> still hidden (there is nothing to actually run on)."""
    import onnxruntime
    monkeypatch.setattr(onnxruntime, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    import onnx_providers as OP
    monkeypatch.setattr(OP, "gpu_runtime_ready", lambda *a, **k: False)
    w = _mw_ready()
    assert w.use_gpu_check.isHidden() is True
    w.close()


def test_gpu_checkbox_visible_and_checked_reflects_onnx_device(monkeypatch, tmp_path):
    """When GPU is actually usable, the checkbox is shown and its initial checked
    state mirrors the *persisted* onnx_device at construction time (cuda ->
    checked, cpu -> unchecked) - written to config.ini before the window is built,
    not set on the widget after the fact, so this actually exercises
    ui_main_window's `setChecked(... != "cpu")` line."""
    _patch_gpu_usable(monkeypatch, usable=True)
    # _isolated_config (autouse) already pointed CONFIG_PATH at tmp_path/config.ini;
    # write real content there before MainWindow() reads it via load_config().
    (tmp_path / "config.ini").write_text("[Behavior]\nonnx_device = cpu\n", encoding="utf-8")
    w = _mw_ready()
    assert w.settings.behavior.onnx_device == "cpu"
    assert w.use_gpu_check.isHidden() is False
    assert w.use_gpu_check.isChecked() is False
    w.close()


def test_gpu_checkbox_toggle_writes_onnx_device_and_saves(monkeypatch, tmp_path):
    """Toggling the checkbox after construction must write cuda/cpu to
    settings.behavior.onnx_device and persist it - this is what actually switches
    the execution provider for the *next* tagging run (no app restart needed,
    unlike the initial GPU-component download).

    Starts from a config.ini with onnx_device=cpu (so the checkbox actually
    constructs unchecked) rather than overwriting settings.behavior directly
    after construction: setChecked(True) on an already-checked box is a no-op
    that never fires `toggled` at all, which would make this test pass
    regardless of whether the handler does anything."""
    _patch_gpu_usable(monkeypatch, usable=True)
    (tmp_path / "config.ini").write_text("[Behavior]\nonnx_device = cpu\n", encoding="utf-8")
    w = _mw_ready()
    assert w.use_gpu_check.isChecked() is False

    saved = []
    monkeypatch.setattr(w, "save_current_config", lambda: saved.append(True))

    w.use_gpu_check.setChecked(True)
    assert w.settings.behavior.onnx_device == "cuda"
    assert saved == [True]

    w.use_gpu_check.setChecked(False)
    assert w.settings.behavior.onnx_device == "cpu"
    assert saved == [True, True]
    w.close()


def test_gpu_checkbox_construction_does_not_overwrite_auto(monkeypatch):
    """A user who has never touched the checkbox keeps onnx_device="auto" as-is:
    setChecked() during widget construction must not itself fire the toggled
    handler and collapse "auto" into "cuda"/"cpu" unasked (this is exactly why
    ui_main_window calls setChecked() before connecting toggled - see that
    comment)."""
    _patch_gpu_usable(monkeypatch, usable=True)
    w = _mw_ready()
    assert w.settings.behavior.onnx_device == "auto"
    assert w.use_gpu_check.isChecked() is True  # displayed as ON (auto uses GPU when available)
    w.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_detached_running_thread_is_deleted_only_after_it_finishes():
    """停止要求に応じなかったスレッドを、稼働中に破棄しないこと。

    260922 PR#27 レビュー指摘: _cleanup_tagger_thread() はタイムアウト後に
    deleteLater() を呼んでいたが、DeferredDelete は QThread *オブジェクト* の所属
    スレッド（=生成元のメインスレッド）へ post されるため、ワーカーの終了を待たず
    次のイベントループ巡目で破棄され、Qt の言う
    "Deleting a running QThread will probably result in a program crash" に当たる。
    当時のコメントは「deleteLater は稼働中でも安全」と逆のことを書いていた。
    """
    import main_window as MW
    from main_window import MainWindow

    MW._detached_threads.clear()
    thread = QThread()
    # self からはモジュール変数しか触らないので、MainWindow を丸ごと作らずに済む。
    MainWindow._detach_running_thread(None, thread, None)
    assert thread in MW._detached_threads
    assert MW.detached_thread_count() == 1

    thread.start()
    assert thread.isRunning()
    for _ in range(3):
        _APP.processEvents()
    # ここで破棄されていると、以降の属性アクセスが RuntimeError になる。
    assert thread.isRunning(), "稼働中のスレッドを破棄してはいけない"

    thread.quit()
    assert thread.wait(5000)
    for _ in range(3):
        _APP.processEvents()
    with pytest.raises(RuntimeError):
        thread.isFinished()  # finished 後に初めて破棄される


def test_detach_prunes_already_finished_threads():
    """detach 置き場が溜まり続けないこと（次の detach 時に掃除される）。"""
    import main_window as MW
    from main_window import MainWindow

    MW._detached_threads.clear()
    first = QThread()
    MainWindow._detach_running_thread(None, first, None)
    first.start()
    first.quit()
    assert first.wait(5000)
    for _ in range(3):
        _APP.processEvents()

    second = QThread()
    MainWindow._detach_running_thread(None, second, None)
    assert MW._detached_threads == [second]
    second.start()
    second.quit()
    assert second.wait(5000)
    for _ in range(3):
        _APP.processEvents()
    MW._detached_threads.clear()
