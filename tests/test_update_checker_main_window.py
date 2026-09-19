"""MainWindow's notification-only updater UI (offscreen).

Two halves, mirroring the code:
  * _maybe_prompt_update_check()  - guards (dismissed / 24h throttle / re-entrancy),
    timestamp persistence, and starting the UpdateCheckWorker QThread.
  * _on_update_check_finished()   - thread teardown, then the 3-button dialog
    (open / later / skip) and the skip-version persistence.

Does not hit the real network (update_checker.check_for_update is monkeypatched
in every test) and does not drive a real GPU prompt.

Run:  rtk pytest tests/test_update_checker_main_window.py -q
"""
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

_APP = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Isolate config.ini (same rationale/pattern as test_gpu_main_window.py's
    fixture of the same name - utils.py/tagging_core.py each compute their own
    CONFIG_PATH independently rather than importing it from constants)."""
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
    """This file only cares about the update-check prompt; keep the unrelated GPU
    prompt from ever firing a real QMessageBox (same trick as
    test_gpu_main_window.py's fixture of the same name)."""
    import onnx_providers as OP
    monkeypatch.setattr(OP, "has_nvidia_gpu", lambda *a, **k: False)


@pytest.fixture(autouse=True)
def _no_real_update_check(monkeypatch):
    """Default to "no update found" so the automatic initial_load() check never
    hits the real network. Individual tests override this with their own
    monkeypatch (the worker looks the function up on the module at call time,
    so the patch is seen from the worker thread too)."""
    import update_checker as UC
    monkeypatch.setattr(UC, "check_for_update", lambda *a, **k: None)


def _drain_update_check(w, timeout=10.0):
    """Pump the GUI event loop until the UpdateCheckWorker thread (if any) has
    delivered its result and been torn down by _on_update_check_finished."""
    deadline = time.monotonic() + timeout
    while w._update_check_thread is not None and time.monotonic() < deadline:
        _APP.processEvents()
        time.sleep(0.01)
    assert w._update_check_thread is None, "update-check thread did not finish in time"
    assert w._update_check_worker is None


def _mw_ready():
    import main_window
    w = main_window.MainWindow()
    _APP.processEvents()
    _APP.processEvents()  # let QTimer.singleShot(0, initial_load) run
    # initial_load() already ran _maybe_prompt_update_check() once automatically
    # (default state: update_check="ask", update_check_last="" - it fires for
    # real, unlike the GPU prompt which _no_gpu_prompt neutralizes). Let that
    # thread finish, then reset the Behavior fields it touched so each test's
    # own explicit call starts from a clean, deterministic slate. Tests that
    # track side effects via an external list (boxes/opened/calls) must
    # additionally `.clear()` it after calling this.
    _drain_update_check(w)
    w.settings.behavior.update_check = "ask"
    w.settings.behavior.update_check_last = ""
    w.settings.behavior.update_skip_version = ""
    return w


class _FakeBox:
    """addButton() is called 3x (open/later/skip, in that order); the real code
    compares clickedButton() by identity, so each must be distinct and
    clickedButton() must return whichever one this fake is told to simulate."""
    Icon = QMessageBox.Icon
    ButtonRole = QMessageBox.ButtonRole

    #: class-level; reset per-test via the fixture below.
    clicked_index = 1  # default: "Later" (the middle button), i.e. do nothing
    #: every instance constructed, so tests can assert on what was shown
    instances: list = []

    def __init__(self, *a, **k):
        self._buttons = []
        self.title = None
        self.text = None
        _FakeBox.instances.append(self)

    def setIcon(self, *a, **k):
        pass

    def setWindowTitle(self, title):
        self.title = title

    def setText(self, text):
        self.text = text

    def addButton(self, label, role):
        btn = object()
        self._buttons.append(btn)
        return btn

    def exec(self):
        pass

    def clickedButton(self):
        return self._buttons[_FakeBox.clicked_index]


@pytest.fixture(autouse=True)
def _reset_fake_box():
    _FakeBox.clicked_index = 1
    _FakeBox.instances = []
    yield
    _FakeBox.clicked_index = 1
    _FakeBox.instances = []


def _install_fake_box(monkeypatch):
    import main_window as MW
    monkeypatch.setattr(MW, "QMessageBox", _FakeBox)


def _fake_update_info(version="v9.9.9", html_url=None):
    import update_checker as UC
    if html_url is None:
        html_url = UC.RELEASE_URL_PREFIX + f"tag/{version}"
    return UC.UpdateInfo(version=version, html_url=html_url, published_at="2026-01-01T00:00:00Z")


# --------------------------------------------------------------------------
# _maybe_prompt_update_check(): guards, throttle, thread start
# --------------------------------------------------------------------------

def test_no_thread_and_no_network_when_dismissed(monkeypatch):
    calls = []
    import update_checker as UC
    monkeypatch.setattr(UC, "check_for_update", lambda *a, **k: calls.append(True) or None)
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    calls.clear()  # discard the automatic initial_load() call's own side effect
    w.settings.behavior.update_check = "dismissed"
    w._maybe_prompt_update_check()

    assert w._update_check_thread is None, "dismissed must not start the worker thread"
    assert calls == [], "dismissed must skip the network check entirely"
    assert w.settings.behavior.update_check_last == "", "dismissed must not touch the timestamp"
    w.close()


def test_no_thread_and_no_network_within_24h_throttle(monkeypatch):
    calls = []
    import update_checker as UC
    monkeypatch.setattr(UC, "check_for_update", lambda *a, **k: calls.append(True) or None)
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    calls.clear()
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    w.settings.behavior.update_check_last = recent.isoformat()
    w._maybe_prompt_update_check()

    assert w._update_check_thread is None
    assert calls == [], "a check within the last 24h must not hit the network again"
    assert w.settings.behavior.update_check_last == recent.isoformat(), "throttled: timestamp untouched"
    w.close()


def test_checks_again_after_24h_and_updates_timestamp(monkeypatch):
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    old = datetime.now(timezone.utc) - timedelta(hours=25)
    w.settings.behavior.update_check_last = old.isoformat()
    w._maybe_prompt_update_check()

    assert w._update_check_thread is not None, "after 24h the check must run"
    new_last = datetime.fromisoformat(w.settings.behavior.update_check_last)
    assert new_last > old
    _drain_update_check(w)
    w.close()


@pytest.mark.parametrize("bad_last", [
    "not-a-timestamp",
    (datetime.now(timezone.utc) + timedelta(days=3)).isoformat(),   # clock was set back
    "2099-01-01T00:00:00",                                          # naive + future
])
def test_unparseable_or_future_timestamp_is_treated_as_never_checked(monkeypatch, bad_last):
    """A garbage or future update_check_last must not silence the check (a future
    value would otherwise suppress it until the clock catches up)."""
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    w.settings.behavior.update_check_last = bad_last
    w._maybe_prompt_update_check()

    assert w._update_check_thread is not None
    new_last = datetime.fromisoformat(w.settings.behavior.update_check_last)
    assert new_last <= datetime.now(timezone.utc)
    _drain_update_check(w)
    w.close()


def test_naive_timestamp_within_24h_is_still_throttled(monkeypatch):
    """A naive ISO timestamp (no tz suffix) is read as UTC, so a recent one
    throttles just like an aware one."""
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    recent_naive = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None)
    w.settings.behavior.update_check_last = recent_naive.isoformat()
    w._maybe_prompt_update_check()

    assert w._update_check_thread is None
    w.close()


def test_update_check_last_is_set_even_when_check_fails(monkeypatch):
    """A network failure (check_for_update returning None) must still update
    update_check_last - otherwise a persistently-unreachable GitHub would be
    retried on every single launch instead of throttled (design doc 6章)."""
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    assert w.settings.behavior.update_check_last == ""
    w._maybe_prompt_update_check()
    _drain_update_check(w)

    assert w.settings.behavior.update_check_last != ""
    assert _FakeBox.instances == []
    w.close()


def test_timestamp_is_persisted_to_config_file(monkeypatch):
    """The throttle only works across launches if it actually reaches config.ini."""
    import app_settings as _A
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    w._maybe_prompt_update_check()
    _drain_update_check(w)

    text = Path(_A.CONFIG_PATH).read_text(encoding="utf-8")
    assert "update_check_last = " in text
    assert w.settings.behavior.update_check_last in text
    w.close()


def test_second_call_while_check_is_running_is_ignored(monkeypatch):
    """Re-entrancy: a second call while the worker thread is still running must
    neither start another thread nor bump the timestamp again."""
    gate = threading.Event()
    import update_checker as UC
    _install_fake_box(monkeypatch)

    w = _mw_ready()  # (patched after: the automatic startup check must not block on the gate)
    monkeypatch.setattr(UC, "check_for_update", lambda *a, **k: gate.wait(10) and None)
    w._maybe_prompt_update_check()
    first_thread = w._update_check_thread
    first_last = w.settings.behavior.update_check_last
    assert first_thread is not None and first_thread.isRunning()
    time.sleep(0.02)  # make a second timestamp distinguishable if it were written

    w._maybe_prompt_update_check()

    assert w._update_check_thread is first_thread
    assert w.settings.behavior.update_check_last == first_last
    gate.set()
    _drain_update_check(w)
    w.close()


def test_check_runs_off_the_gui_thread_end_to_end(monkeypatch):
    """The whole point of the worker: the HTTPS GET must not run on the GUI
    thread, and its result must still surface as a dialog on the GUI thread."""
    seen_threads = []
    import update_checker as UC

    def _fake_check(*a, **k):
        seen_threads.append(threading.current_thread())
        return _fake_update_info(version="v9.9.9")
    monkeypatch.setattr(UC, "check_for_update", _fake_check)
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    seen_threads.clear()
    _FakeBox.instances.clear()
    w._maybe_prompt_update_check()
    _drain_update_check(w)

    assert seen_threads and all(t is not threading.main_thread() for t in seen_threads)
    assert len(_FakeBox.instances) == 1
    assert "v9.9.9" in _FakeBox.instances[0].text
    w.close()


def test_close_while_check_is_running_does_not_hang_or_prompt(monkeypatch):
    """closeEvent must join the worker thread (it is in threads_to_stop) and the
    late result must not pop a dialog on a window that is shutting down."""
    gate = threading.Event()
    import update_checker as UC
    _install_fake_box(monkeypatch)

    w = _mw_ready()  # (patched after: the automatic startup check must not block on the gate)
    monkeypatch.setattr(UC, "check_for_update",
                        lambda *a, **k: (gate.wait(10) and None) or _fake_update_info())
    _FakeBox.instances.clear()
    w._maybe_prompt_update_check()
    thread = w._update_check_thread
    assert thread is not None and thread.isRunning()

    gate.set()
    w.close()  # closeEvent: stop() + quit() + wait()
    assert not thread.isRunning()
    _APP.processEvents()
    _APP.processEvents()  # deliver the queued check_finished, if any
    assert _FakeBox.instances == []


# --------------------------------------------------------------------------
# _on_update_check_finished(): the dialog and its three buttons
# --------------------------------------------------------------------------

def test_prompt_shown_with_both_versions_when_update_available(monkeypatch):
    import constants
    _install_fake_box(monkeypatch)
    _FakeBox.clicked_index = 1  # "Later" - don't act, just inspect what was shown

    w = _mw_ready()
    w._on_update_check_finished(_fake_update_info(version="v9.9.9"))

    assert len(_FakeBox.instances) == 1, "a dialog should have been constructed"
    text = _FakeBox.instances[0].text
    assert "v9.9.9" in text
    # Tags are "vX.Y.Z"; the running version is shown the same way for consistency.
    assert f"v{constants.APP_VERSION}" in text
    assert _FakeBox.instances[0].title
    w.close()


def test_none_result_shows_nothing(monkeypatch):
    _install_fake_box(monkeypatch)
    w = _mw_ready()
    w._on_update_check_finished(None)
    assert _FakeBox.instances == []
    w.close()


def test_open_button_opens_browser_to_release_page(monkeypatch):
    _install_fake_box(monkeypatch)
    _FakeBox.clicked_index = 0  # "Open download page"

    opened = []
    import main_window as MW
    monkeypatch.setattr(MW.webbrowser, "open", lambda url: opened.append(url))

    w = _mw_ready()
    info = _fake_update_info(version="v9.9.9")
    w._on_update_check_finished(info)

    assert opened == [info.html_url]
    assert w.settings.behavior.update_skip_version == ""
    w.close()


def test_later_button_does_not_persist_skip_version(monkeypatch):
    _install_fake_box(monkeypatch)
    _FakeBox.clicked_index = 1  # "Not now"

    opened = []
    import main_window as MW
    monkeypatch.setattr(MW.webbrowser, "open", lambda url: opened.append(url))

    w = _mw_ready()
    w._on_update_check_finished(_fake_update_info(version="v9.9.9"))

    assert opened == []
    assert w.settings.behavior.update_skip_version == ""
    w.close()


def test_skip_button_persists_skip_version(monkeypatch):
    import app_settings as _A
    _install_fake_box(monkeypatch)
    _FakeBox.clicked_index = 2  # "Skip this version"

    w = _mw_ready()
    w._on_update_check_finished(_fake_update_info(version="v9.9.9"))

    assert w.settings.behavior.update_skip_version == "v9.9.9"
    assert "update_skip_version = v9.9.9" in Path(_A.CONFIG_PATH).read_text(encoding="utf-8")
    w.close()


def test_skipped_version_is_not_reprompted(monkeypatch):
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    w.settings.behavior.update_skip_version = "v9.9.9"
    w._on_update_check_finished(_fake_update_info(version="v9.9.9"))

    assert _FakeBox.instances == [], "a version the user already skipped must not prompt again"
    w.close()


def test_older_than_skipped_version_is_not_reprompted(monkeypatch):
    """Skip is "this version and below" - if GitHub's latest somehow regresses
    (release deleted) we must not nag about an even older one."""
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    w.settings.behavior.update_skip_version = "v9.9.9"
    w._on_update_check_finished(_fake_update_info(version="v9.9.8"))

    assert _FakeBox.instances == []
    w.close()


def test_newer_version_is_reprompted_after_a_skip(monkeypatch):
    """Skipping v9.9.9 must not silently suppress a later v10.0.0."""
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    w.settings.behavior.update_skip_version = "v9.9.9"
    w._on_update_check_finished(_fake_update_info(version="v10.0.0"))

    assert len(_FakeBox.instances) == 1
    w.close()


def test_unparseable_skip_version_does_not_suppress_prompt(monkeypatch):
    """A hand-edited garbage update_skip_version must fail open (prompt), not
    silently disable notifications forever."""
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    w.settings.behavior.update_skip_version = "whatever"
    w._on_update_check_finished(_fake_update_info(version="v9.9.9"))

    assert len(_FakeBox.instances) == 1
    w.close()


def test_no_prompt_when_shutting_down(monkeypatch):
    _install_fake_box(monkeypatch)

    w = _mw_ready()
    w._is_shutting_down = True
    w._on_update_check_finished(_fake_update_info())

    assert _FakeBox.instances == []
    w.close()


def test_prompt_is_deferred_while_another_modal_is_open(monkeypatch):
    """The GPU-setup prompt / progress dialog may still be up when the result
    arrives; the update dialog must wait for it rather than stack on top."""
    _install_fake_box(monkeypatch)
    import main_window as MW
    w = _mw_ready()  # construct first: __init__ / initial_load use the real QApplication & QTimer

    real_app, real_timer = MW.QApplication, MW.QTimer
    modal = [object()]  # non-None => "a modal is active"

    class _FakeApp:
        @staticmethod
        def activeModalWidget():
            return modal[0]
    monkeypatch.setattr(MW, "QApplication", _FakeApp)

    deferred = []

    class _FakeTimer:
        @staticmethod
        def singleShot(ms, fn):
            deferred.append((ms, fn))
    monkeypatch.setattr(MW, "QTimer", _FakeTimer)

    w._on_update_check_finished(_fake_update_info(version="v9.9.9"))

    assert _FakeBox.instances == [], "must not stack on an open modal"
    assert len(deferred) == 1 and deferred[0][0] > 0

    # Modal still open on the retry -> defer again, still no dialog.
    deferred[0][1]()
    assert _FakeBox.instances == []
    assert len(deferred) == 2

    # Modal gone -> the deferred retry finally shows it.
    modal[0] = None
    deferred[1][1]()
    assert len(_FakeBox.instances) == 1
    assert "v9.9.9" in _FakeBox.instances[0].text
    # Restore just these two before closeEvent (monkeypatch.undo() would also
    # revert CONFIG_PATH and make close() write to the real config.ini).
    monkeypatch.setattr(MW, "QApplication", real_app)
    monkeypatch.setattr(MW, "QTimer", real_timer)
    w.close()


def test_gpu_prompt_runs_before_update_check_in_initial_load(monkeypatch):
    """initial_load() starts the GPU prompt first (it is synchronous) and only
    then kicks off the update check, so the update dialog can never be the one
    that pops up first / underneath."""
    order = []
    import main_window as MW
    monkeypatch.setattr(MW.MainWindow, "_maybe_prompt_gpu_setup", lambda self: order.append("gpu"))
    monkeypatch.setattr(MW.MainWindow, "_maybe_prompt_update_check", lambda self: order.append("update"))

    w = _mw_ready()  # initial_load() runs automatically via the singleShot timer

    assert order == ["gpu", "update"]
    w.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
