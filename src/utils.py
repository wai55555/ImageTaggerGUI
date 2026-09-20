from __future__ import annotations
import atexit
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Protocol
from datetime import datetime
import hashlib
import configparser

from constants import BASE_DIR

LOG_FILE_PATH = BASE_DIR / "debug_log.txt"
# 未捕捉例外は debug_log.txt とは別の error_log.txt に書く。write_debug_log() は
# [Debug] debug_log = False（既定）だと早期 return してしまうため、Debug 設定に関係
# なく必ず残したい未捕捉例外はここに書く（後方の「未捕捉例外のログ」セクション参照）。
ERROR_LOG_PATH = BASE_DIR / "error_log.txt"
CONFIG_PATH = BASE_DIR / "config.ini"

class GetString(Protocol):
    def __call__(self, section: str, key: str, **kwargs: Any) -> str: ...

def default_get_string_fallback(section: str, key: str, **kwargs: Any) -> str:
    """Default fallback for get_string if no localization function is provided."""
    return key

class DebugSettings:
    _instance: DebugSettings | None = None

    def __init__(self):
        self.debug_log_enabled: bool = False
        try:
            if CONFIG_PATH.is_file():
                config = configparser.ConfigParser()
                config.read(CONFIG_PATH, encoding='utf-8')
                self.debug_log_enabled = config.getboolean('Debug', 'debug_log', fallback=False)
        except Exception:
            self.debug_log_enabled = False

    @classmethod
    def get_instance(cls) -> 'DebugSettings':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_debug_settings() -> DebugSettings:
    return DebugSettings.get_instance()

def nowtag() -> str:
    """Return the current time as a string in the format [YYYY-MM-DD HH:MM:SS]."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")

# --- Buffered debug log --------------------------------------------------------
# Previously every write_debug_log() call did open()/write()/flush()/close(). A
# tagging run logs one or two lines per image, so a 20k-image batch became ~40k
# synchronous flushed appends - enough disk churn to make the app look frozen
# (issue #10). Keep one handle open and flush at most once per second (and at exit),
# so routine logging costs almost nothing while a crash still keeps ~1s of history.
_LOG_LOCK = threading.Lock()
_LOG_FH = None
_LOG_LAST_FLUSH = 0.0
_LOG_FLUSH_INTERVAL_S = 1.0


_LOG_ACQUIRE_TIMEOUT_S = 0.5

# --- debug_log.txt rotation -----------------------------------------------------
# Debug ユーザーは無期限に書き続けると際限なく育つ（開発機で既に20MB超）。世代管理は
# せず、起動時に一度だけ既存ファイルを .1 に退避する「単純な切り詰め」で十分とする
# （バグ報告で送ってもらうログが巨大にならないことが目的）。
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_ROTATED_PATH = BASE_DIR / "debug_log.1.txt"


def _rotate_log_if_needed() -> None:
    """`LOG_FILE_PATH` が `LOG_MAX_BYTES` を超えていたら `LOG_ROTATED_PATH` に退避する。

    呼び出しは `write_debug_log()` が `_LOG_FH` を初めて開く直前（プロセスにつき1回）
    に限る。開いたハンドルを持ったまま実体を差し替えると Windows では失敗するため、
    ハンドルを持つ前のこのタイミングでしか安全に呼べない。`os.rename` ではなく
    `os.replace` を使う（Windows では既存の宛先があると `os.rename` が失敗する）。
    失敗（別プロセスがファイルを掴んでいる等）してもログ機能自体は止めず、そのまま
    既存ファイルへの追記を継続する。
    """
    try:
        if LOG_FILE_PATH.is_file() and LOG_FILE_PATH.stat().st_size > LOG_MAX_BYTES:
            os.replace(LOG_FILE_PATH, LOG_ROTATED_PATH)
    except Exception:
        pass


def _flush_debug_log() -> None:
    global _LOG_LAST_FLUSH
    # タイムアウト付き取得: QThread.terminate()（closeEvent の最終手段）がこのロックを
    # 保持中のスレッドを任意の場所で強制終了しうる。解放されないロックを永久に待つと、
    # atexit で呼ばれるこの関数自身、ひいてはプロセス終了が固まる。取れなければ諦める。
    if not _LOG_LOCK.acquire(timeout=_LOG_ACQUIRE_TIMEOUT_S):
        return
    try:
        if _LOG_FH is not None:
            try:
                _LOG_FH.flush()
            except Exception:
                pass
        _LOG_LAST_FLUSH = time.monotonic()
    finally:
        _LOG_LOCK.release()


atexit.register(_flush_debug_log)


def write_debug_log(message: str, get_string: GetString | None = None):
    global _LOG_FH, _LOG_LAST_FLUSH
    _get_string: GetString = get_string if get_string else default_get_string_fallback
    if not get_debug_settings().debug_log_enabled:
        return
    if not message.strip():
        return

    lines = [ln for ln in message.split('\n') if ln.strip()]
    if not lines:
        return

    # タイムアウト付き取得（PR#16 レビュー指摘）: 別スレッドが terminate() で
    # このロックを保持したまま死んでいても、ここで永久ブロックしない。GUI スレッドの
    # closeEvent からもこの関数は呼ばれるので、取れないなら黙ってこの1行を諦める
    # （デバッグ用途であり、アプリが閉じられなくなる方が遥かに悪い）。
    if not _LOG_LOCK.acquire(timeout=_LOG_ACQUIRE_TIMEOUT_S):
        return
    try:
        if _LOG_FH is None:
            # プロセスにつき1回、実際に最初の1行を書く直前だけ呼ぶ（_LOG_FH を持った
            # まま呼ぶとハンドルの下でファイルを差し替えることになり Windows で壊れる）。
            _rotate_log_if_needed()
            _LOG_FH = open(LOG_FILE_PATH, 'a', encoding='utf-8')
        tag = nowtag()
        for line in lines:
            _LOG_FH.write(tag + line.strip() + "\n")
        now = time.monotonic()
        if now - _LOG_LAST_FLUSH >= _LOG_FLUSH_INTERVAL_S:
            _LOG_FH.flush()
            _LOG_LAST_FLUSH = now
    except Exception:
        # 書き込み/flush に失敗したハンドルを使い続けると以後ずっと書けなくなるので、
        # 閉じてリセットし次回呼び出しで開き直す。
        try:
            if _LOG_FH is not None:
                _LOG_FH.close()
        except Exception:
            pass
        _LOG_FH = None
        print(f"{_get_string('Utils', 'Log_Write_Failed', message=message)}", file=sys.stderr)
    finally:
        _LOG_LOCK.release()

def log_dbg(msg: str, get_string: GetString | None = None):
    write_debug_log(msg, get_string)

# --- Uncaught exception logging -------------------------------------------------
# The app ships as a windowed PyInstaller build (console=False), so a frozen exe
# has sys.stderr/sys.stdout == None. A Python exception raised inside a Qt slot is
# handed to sys.excepthook by PySide6, but the default excepthook writes nothing
# when stderr is missing - the exception vanishes and the GUI just keeps running
# as if nothing happened. install_excepthook() below replaces sys.excepthook (and,
# for completeness, threading.excepthook - QThread does not actually go through
# it, but a stray threading.Thread would) so every uncaught exception leaves a
# trace in ERROR_LOG_PATH regardless of the [Debug] debug_log setting, and shows a
# single "something went wrong" dialog per process instead of silently vanishing.

_ERROR_LOG_MAX_BYTES = 1 * 1024 * 1024
_uncaught_dialog_shown = False


def _write_error_log(text: str) -> None:
    """Append `text` to ERROR_LOG_PATH, truncating first if it has grown past 1MB.

    This is a simple truncate, not generation-based rotation like debug_log.txt -
    "the most recent crash is readable" is the whole goal here, not a full history.
    """
    try:
        if ERROR_LOG_PATH.is_file() and ERROR_LOG_PATH.stat().st_size > _ERROR_LOG_MAX_BYTES:
            ERROR_LOG_PATH.write_text("", encoding="utf-8")
    except Exception:
        pass
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text)


def _uncaught_error_strings() -> tuple[str, str]:
    """Returns (title, body) for the uncaught-exception dialog.

    Tries the app's live get_string, wired up by MainWindow via
    app_settings.set_get_string_func() once its LocaleManager exists (same
    convention every other module that needs a module-level get_string follows -
    see app_settings._get_string). An exception can happen before that wiring runs
    (e.g. during early startup), so any failure here - including the lazy import
    itself - falls back to a hardcoded English message instead of raising.
    """
    fallback_title = "Unexpected Error"
    fallback_body = (
        "An unexpected error occurred, but the app should still be usable. "
        f"Please report this issue and attach the log file:\n{ERROR_LOG_PATH}"
    )
    try:
        from app_settings import _get_string as _app_get_string
        title = _app_get_string("Error", "Uncaught_Title")
        body = _app_get_string("Error", "Uncaught_Body", path=str(ERROR_LOG_PATH))
        # app_settings._get_string defaults to default_get_string_fallback (echoes
        # the key back unchanged) until MainWindow's LocaleManager exists and
        # wires it up via set_get_string_func() - that happens without raising,
        # so an exception alone would not catch "too early to have a real
        # translation". Treat a get_string that just echoed the key back as the
        # same kind of failure as an exception.
        if title and body and title != "Uncaught_Title" and body != "Uncaught_Body":
            # configparser reads a literal "\n" in an .ini value as the two
            # characters backslash+n, not an actual newline (same quirk
            # api_key_dialog.py already works around for its own string) - undo
            # that so QMessageBox shows a real line break before the path.
            return title, body.replace("\\n", "\n")
    except Exception:
        pass
    return fallback_title, fallback_body


def _maybe_show_uncaught_dialog(title: str, body: str) -> None:
    """Shows the uncaught-exception dialog at most once per process.

    Skips the dialog entirely (log-only) if there is no QApplication instance -
    e.g. an exception before QApplication() is constructed, or an exception on a
    non-GUI thread. Qt/QMessageBox are imported lazily here so utils.py keeps its
    current no-Qt-dependency at module load time.
    """
    global _uncaught_dialog_shown
    if _uncaught_dialog_shown:
        return
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except Exception:
        return
    try:
        if QApplication.instance() is None:
            return
        _uncaught_dialog_shown = True
        QMessageBox.critical(None, title, body)
    except Exception:
        pass


def _handle_uncaught(exc_type, exc, tb) -> None:
    """Shared handler behind both sys.excepthook and threading.excepthook.

    KeyboardInterrupt is handed to the default hook untouched (no log, no dialog -
    it is normal, user-initiated termination, not a bug). Anything else is logged
    to ERROR_LOG_PATH and write_debug_log(), then surfaced once via a dialog. The
    whole body is wrapped so a failure in the handler itself (disk full, no
    permission, ...) can never crash the app further.
    """
    try:
        if exc_type is KeyboardInterrupt:
            sys.__excepthook__(exc_type, exc, tb)
            return

        try:
            from constants import APP_VERSION
        except Exception:
            APP_VERSION = "?"

        try:
            formatted = "".join(traceback.format_exception(exc_type, exc, tb))
        except Exception:
            formatted = f"{exc_type}: {exc}\n"

        block = f"{nowtag()}App version {APP_VERSION}\n{formatted}"
        if not block.endswith("\n"):
            block += "\n"

        try:
            _write_error_log(block)
        except Exception:
            pass

        try:
            write_debug_log(block)
        except Exception:
            pass

        try:
            title, body = _uncaught_error_strings()
            _maybe_show_uncaught_dialog(title, body)
        except Exception:
            pass
    except Exception:
        pass


def _threading_excepthook_adapter(args) -> None:
    """Adapts threading.excepthook's single-arg signature to _handle_uncaught()."""
    _handle_uncaught(args.exc_type, args.exc_value, args.exc_traceback)


def install_excepthook() -> None:
    """Routes uncaught exceptions on the main thread and threading.Thread threads
    to _handle_uncaught(). Call once, at startup, before QApplication is created."""
    sys.excepthook = _handle_uncaught
    threading.excepthook = _threading_excepthook_adapter


def calculate_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate the SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (FileNotFoundError, OSError):
        return ""


def config_mapping(config: Any, *keys: str) -> dict[str, Any]:
    """Walks `config` through `keys`, returning {} as soon as anything is not a mapping.

    model_config.json is hand-authored, so a key may be present but null (`"network":
    null`) - plain `cfg.get("network", {}).get("files", {})` raises AttributeError in
    that case, which would abort discover_models() entirely instead of skipping one
    bad manifest. Lives here (a leaf module both model_registry and tagging_core already
    import) so there is exactly one implementation to keep correct.
    """
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}
