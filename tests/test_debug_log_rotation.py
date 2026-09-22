"""debug_log.txt rotation (`utils._rotate_log_if_needed`).

Covers the post-release hardening plan item 2: with [Debug] debug_log on and no
cap, debug_log.txt grows without bound (seen at 20MB+ on a real dev machine),
which makes "please send me your log" turn into a huge attachment. On the first
write_debug_log() call of a process, if the existing file is already past
LOG_MAX_BYTES it gets moved aside to debug_log.1.txt (single generation, no
history) via os.replace() before the new file is opened - rotation never runs
again for the rest of the process, since it happens only in the `_LOG_FH is
None` branch.

Offline only - no GUI, no network. Run:  rtk pytest tests/test_debug_log_rotation.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import utils


@pytest.fixture(autouse=True)
def _isolated_log(monkeypatch, tmp_path):
    """Point LOG_FILE_PATH/LOG_ROTATED_PATH at tmp_path and reset the module-level
    open file handle, which otherwise persists across tests (it is only opened
    once per process, in the `_LOG_FH is None` branch this file is testing)."""
    monkeypatch.setattr(utils, "LOG_FILE_PATH", tmp_path / "debug_log.txt")
    monkeypatch.setattr(utils, "LOG_ROTATED_PATH", tmp_path / "debug_log.1.txt")
    monkeypatch.setattr(utils, "_LOG_FH", None)
    settings = utils.get_debug_settings()
    monkeypatch.setattr(settings, "debug_log_enabled", True)
    yield
    # write_debug_log() may have left a handle open on a tmp_path file; close it
    # so later tests (and pytest's own tmp_path cleanup) don't see it as busy.
    if utils._LOG_FH is not None:
        try:
            utils._LOG_FH.close()
        except Exception:
            pass
        monkeypatch.setattr(utils, "_LOG_FH", None)


def _write_and_flush(message: str) -> None:
    """write_debug_log() only flushes to disk once per second (or at exit) to
    avoid disk churn during a large tagging run - so a bare write_debug_log()
    call here would leave the write sitting in Python's buffer, invisible to a
    read_text() from a separate file handle. Force the flush so tests can
    assert on file contents immediately."""
    utils.write_debug_log(message)
    utils._flush_debug_log()


def test_oversized_log_is_rotated_before_first_write():
    utils.LOG_FILE_PATH.write_text("x" * (utils.LOG_MAX_BYTES + 1), encoding="utf-8")

    _write_and_flush("hello")

    assert utils.LOG_ROTATED_PATH.is_file()
    assert utils.LOG_ROTATED_PATH.stat().st_size == utils.LOG_MAX_BYTES + 1
    assert utils.LOG_FILE_PATH.stat().st_size < utils.LOG_MAX_BYTES
    assert "hello" in utils.LOG_FILE_PATH.read_text(encoding="utf-8")


def test_undersized_log_is_not_rotated():
    utils.LOG_FILE_PATH.write_text("small", encoding="utf-8")

    _write_and_flush("hello")

    assert not utils.LOG_ROTATED_PATH.exists()
    text = utils.LOG_FILE_PATH.read_text(encoding="utf-8")
    assert "small" in text
    assert "hello" in text


def test_existing_rotated_file_is_overwritten_not_accumulated():
    utils.LOG_ROTATED_PATH.write_text("stale generation from a previous rotation", encoding="utf-8")
    utils.LOG_FILE_PATH.write_text("y" * (utils.LOG_MAX_BYTES + 1), encoding="utf-8")

    _write_and_flush("hello")

    rotated_text = utils.LOG_ROTATED_PATH.read_text(encoding="utf-8")
    assert "stale generation" not in rotated_text
    assert rotated_text == "y" * (utils.LOG_MAX_BYTES + 1)


def test_rotation_failure_does_not_block_logging(monkeypatch):
    utils.LOG_FILE_PATH.write_text("x" * (utils.LOG_MAX_BYTES + 1), encoding="utf-8")

    def _raise_replace(*a, **k):
        raise OSError("file is in use by another process")

    monkeypatch.setattr(utils.os, "replace", _raise_replace)

    # Must not raise, and must still log by appending to the (still oversized)
    # existing file rather than losing the write.
    _write_and_flush("hello")

    assert not utils.LOG_ROTATED_PATH.exists()
    assert "hello" in utils.LOG_FILE_PATH.read_text(encoding="utf-8")


def test_rotation_only_happens_once_per_process():
    """Rotation is checked only in the `_LOG_FH is None` branch, so once the
    handle is open for this process, growing the file past LOG_MAX_BYTES again
    must not trigger a second rotation (mid-run rotation with a held handle would
    corrupt on Windows - see _rotate_log_if_needed's docstring)."""
    _write_and_flush("first")
    assert utils._LOG_FH is not None

    # Manually inflate the file the handle is still writing to.
    with open(utils.LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("z" * (utils.LOG_MAX_BYTES + 1))

    _write_and_flush("second")

    assert not utils.LOG_ROTATED_PATH.exists()
    assert "second" in utils.LOG_FILE_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
