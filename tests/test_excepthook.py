"""Uncaught-exception handling (`utils.install_excepthook` / `utils._handle_uncaught`).

Covers the post-release hardening plan item 1: a windowed PyInstaller build
(console=False) has sys.stderr/sys.stdout == None, so the default sys.excepthook
writes nothing when a Qt slot lets a Python exception through - the exception
just vanishes and the GUI keeps running. `_handle_uncaught()` fixes that by always
writing to `utils.ERROR_LOG_PATH` (regardless of the [Debug] debug_log setting),
mirroring the same text into write_debug_log(), and showing one dialog per
process.

Offscreen only - no network. Run:  rtk pytest tests/test_excepthook.py -q
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

import utils

_APP = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_logs(monkeypatch, tmp_path):
    """Point both log files at tmp_path and reset the process-lifetime dialog flag.

    `utils._uncaught_dialog_shown` is a module-level "have we shown the dialog
    yet" flag by design (once per process) - tests must reset it themselves or a
    later test would inherit an earlier test's "already shown" state.

    Also stubs out QMessageBox.critical unconditionally: a real QMessageBox.exec()
    opens a modal event loop that blocks forever with no one to click a button
    (even under QT_QPA_PLATFORM=offscreen), so every test - not just the ones that
    specifically assert on dialog behavior - must never call the real one. Tests
    that care about the dialog override this again locally via their own
    monkeypatch.setattr(QMessageBox, "critical", ...).
    """
    monkeypatch.setattr(utils, "ERROR_LOG_PATH", tmp_path / "error_log.txt")
    monkeypatch.setattr(utils, "LOG_FILE_PATH", tmp_path / "debug_log.txt")
    monkeypatch.setattr(utils, "_uncaught_dialog_shown", False)
    monkeypatch.setattr(utils, "_LOG_FH", None)
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    # Debug logging is gated by [Debug] debug_log in config.ini; force it on so
    # write_debug_log() actually writes instead of early-returning.
    settings = utils.get_debug_settings()
    monkeypatch.setattr(settings, "debug_log_enabled", True)


def _make_exc():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        return sys.exc_info()


def test_uncaught_exception_logs_traceback_and_version():
    exc_type, exc, tb = _make_exc()
    utils._handle_uncaught(exc_type, exc, tb)

    assert utils.ERROR_LOG_PATH.is_file()
    text = utils.ERROR_LOG_PATH.read_text(encoding="utf-8")
    assert "RuntimeError" in text
    assert "boom" in text

    from constants import APP_VERSION
    assert APP_VERSION in text


def test_dialog_shown_only_once(monkeypatch):
    calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: calls.append(a))

    exc_type, exc, tb = _make_exc()
    utils._handle_uncaught(exc_type, exc, tb)
    utils._handle_uncaught(exc_type, exc, tb)

    assert len(calls) == 1
    # Both calls still logged, even though only the first showed a dialog. Each
    # traceback block contains "RuntimeError" twice (the `raise` line and the
    # final "RuntimeError: boom" summary line), so two calls means four hits of
    # "RuntimeError" but only two of the summary line.
    text = utils.ERROR_LOG_PATH.read_text(encoding="utf-8")
    assert text.count("RuntimeError: boom") == 2


def test_dialog_skipped_without_qapplication(monkeypatch):
    calls = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    exc_type, exc, tb = _make_exc()
    utils._handle_uncaught(exc_type, exc, tb)

    assert calls == []
    # Still logged - only the dialog is skipped.
    assert utils.ERROR_LOG_PATH.is_file()


def test_error_log_truncated_past_1mb():
    utils.ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    utils.ERROR_LOG_PATH.write_text("x" * (utils._ERROR_LOG_MAX_BYTES + 1), encoding="utf-8")

    exc_type, exc, tb = _make_exc()
    utils._handle_uncaught(exc_type, exc, tb)

    size = utils.ERROR_LOG_PATH.stat().st_size
    assert size < utils._ERROR_LOG_MAX_BYTES
    text = utils.ERROR_LOG_PATH.read_text(encoding="utf-8")
    assert "x" * 100 not in text  # old padding is gone, not just appended-to
    assert "RuntimeError" in text


def test_keyboard_interrupt_delegates_to_default_hook(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "__excepthook__", lambda *a: calls.append(a))

    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_type, exc, tb = sys.exc_info()

    utils._handle_uncaught(exc_type, exc, tb)

    assert len(calls) == 1
    assert calls[0][0] is KeyboardInterrupt
    assert not utils.ERROR_LOG_PATH.exists()


def test_handler_survives_log_write_failure(monkeypatch):
    def _raise_open(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(utils, "_write_error_log", _raise_open)

    exc_type, exc, tb = _make_exc()
    # Must not raise.
    utils._handle_uncaught(exc_type, exc, tb)


def test_install_excepthook_wires_both_hooks(monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    import threading as _threading
    monkeypatch.setattr(_threading, "excepthook", _threading.__excepthook__)

    utils.install_excepthook()

    assert sys.excepthook is utils._handle_uncaught
    assert _threading.excepthook is utils._threading_excepthook_adapter


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_uncaught_dialog_is_skipped_off_the_gui_thread(monkeypatch):
    """ワーカースレッドからの未捕捉例外で QMessageBox を作らない。

    260922 PR#27 レビュー指摘: sys.excepthook / threading.excepthook はどちらも
    失敗したスレッドで走るので、QApplication があるだけで QMessageBox を作ると
    GUIスレッド外での Qt ウィジェット操作になり、クラッシュやハングになりうる。
    docstring は元から「非GUIスレッドなら出さない」と書いていたが、実装は
    QApplication の有無しか見ていなかった。
    """
    import threading

    from PySide6.QtWidgets import QApplication, QMessageBox

    assert QApplication.instance() is not None, "このテストは QApplication 前提"
    monkeypatch.setattr(utils, "_uncaught_dialog_shown", False)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(QMessageBox, "critical",
                        classmethod(lambda cls, *a, **k: shown.append(a[1:3])))

    def _worker():
        utils._maybe_show_uncaught_dialog("t", "b")

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert shown == [], "非GUIスレッドからダイアログを作ってはいけない"
    # ワーカー側で「1回しか出さない」フラグを消費していないこと。消費していると、
    # 後続のメインスレッド例外が無言でログのみになってしまう。
    assert utils._uncaught_dialog_shown is False

    utils._maybe_show_uncaught_dialog("t", "b")
    assert shown == [("t", "b")], "GUIスレッドからは従来どおり出す"
    assert utils._uncaught_dialog_shown is True
