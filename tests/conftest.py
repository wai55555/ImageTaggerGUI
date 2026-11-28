"""全テスト共通の隔離（二次改修 Phase 1 / REQ-1.1〜1.3）。

このファイルの目的は「テストが開発者の実環境を読まない・書かない」ことを、
テストモジュールごとの ad-hoc な差し替えではなく 1 か所で保証すること。

隔離するもの
------------
- ``config.ini``（app_settings / constants / utils / tagging_core が各自の定数として
  持っているため 4 か所すべて）
- debug / error ログ（``utils`` と ``tagging_core``）
- ``vlm_connections.json`` / ``vlm_profiles.json``（``vlm_config``）
- 既定 config の ``input_dir``（``app_settings.BASE_DIR`` をテストごとに tmp_path へ）

セッション開始時に 1 度だけ向け先を変えるもの（ログ・VLM JSON）と、テストごとに
``tmp_path`` へ向け直すもの（config.ini / BASE_DIR）を分けている。セッション終了時に
必ず元へ戻す。

実データの非破壊検査
--------------------
``real_state_fingerprint()`` が実 ``config.ini`` / ``.env`` / VLM JSON / ログ /
``inputs/`` / ``models/`` の指紋を返し、テストの前後で比較する。1 つでも変化したら
そのテストの責任として失敗させる（REQ-1.2 の「実環境で走らせても副作用が無いこと」を
テスト自体で保証する）。``inputs/`` と ``models/`` は再帰走査すると数 GB になるため、
直下エントリの名前・サイズ・mtime だけを見る（実データを触る事故は直下に現れる）。
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Qt は最初の import より前にプラットフォームを決める必要がある。pytest の
# ブートストラップである conftest で 1 度だけ既定値を入れる（テストごとの
# 差し替えではないので、他テストへ漏れる種類のグローバル変更ではない）。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import app_settings
import constants
import tagging_core
import utils
import vlm_config

_SESSION_TMP = Path(tempfile.mkdtemp(prefix="imagetaggergui-tests-"))

# (module, attribute) -> 元の値。セッション終了時に戻す。
_ORIGINAL: dict[tuple[object, str], object] = {}

_SESSION_PATCHES: dict[object, dict[str, object]] = {
    app_settings: {"CONFIG_PATH": _SESSION_TMP / "config.ini"},
    constants: {"CONFIG_PATH": _SESSION_TMP / "config.ini"},
    utils: {
        "CONFIG_PATH": _SESSION_TMP / "config.ini",
        "LOG_FILE_PATH": _SESSION_TMP / "debug_log.txt",
        "ERROR_LOG_PATH": _SESSION_TMP / "error_log.txt",
        "LOG_ROTATED_PATH": _SESSION_TMP / "debug_log.1.txt",
        # ロガーはプロセス内で 1 ハンドルを使い回す。開いたままだと差し替え先へ
        # 書かれないので、セッションの先頭で未オープンに戻す。
        "_LOG_FH": None,
    },
    tagging_core: {
        "CONFIG_PATH": _SESSION_TMP / "config.ini",
        "LOG_FILE_PATH": _SESSION_TMP / "debug_log.txt",
    },
    vlm_config: {
        "VLM_CONNECTIONS_PATH": _SESSION_TMP / "vlm_connections.json",
        "VLM_PROFILES_PATH": _SESSION_TMP / "vlm_profiles.json",
    },
}


def _install_session_isolation() -> None:
    for module, attrs in _SESSION_PATCHES.items():
        for name, value in attrs.items():
            _ORIGINAL[(module, name)] = getattr(module, name)
            setattr(module, name, value)


def _restore_session_isolation() -> None:
    handle = getattr(utils, "_LOG_FH", None)
    if handle is not None:
        try:
            handle.close()
        except OSError:  # pragma: no cover - ベストエフォートの後始末
            pass
    for (module, name), value in _ORIGINAL.items():
        setattr(module, name, value)


_install_session_isolation()


# --- 実データ指紋 ---------------------------------------------------------------

# 実ファイル。サイズが大きいものは mtime/size だけ見る（毎テスト 2 回走るため）。
_REAL_FILES = (
    "config.ini",
    ".env",
    "vlm_connections.json",
    "vlm_profiles.json",
    "debug_log.txt",
    "debug_log.1.txt",
    "error_log.txt",
)
_REAL_DIRS = ("inputs", "models")
_HASH_LIMIT_BYTES = 256 * 1024


def _file_fingerprint(path: Path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    digest = None
    if path.is_file() and stat.st_size <= _HASH_LIMIT_BYTES:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (stat.st_size, stat.st_mtime_ns, digest)


def _dir_fingerprint(path: Path):
    if not path.is_dir():
        return None
    entries = []
    for entry in sorted(path.iterdir(), key=lambda p: p.name):
        try:
            stat = entry.stat()
        except OSError:
            entries.append((entry.name, None, None))
            continue
        entries.append((entry.name, stat.st_size, stat.st_mtime_ns))
    return tuple(entries)


def real_state_fingerprint() -> dict[str, object]:
    """実データの指紋。テスト前後で一致することを要求する。"""
    state: dict[str, object] = {}
    for name in _REAL_FILES:
        state[name] = _file_fingerprint(REPO_ROOT / name)
    for name in _REAL_DIRS:
        state[name] = _dir_fingerprint(REPO_ROOT / name)
    return state


def is_inside_repo(path: Path | str) -> bool:
    """path がリポジトリ（実データの置き場）の中を指しているか。"""
    try:
        Path(path).resolve().relative_to(REPO_ROOT.resolve())
    except (ValueError, OSError):
        return False
    return True


def _changed_keys(before: dict[str, object], after: dict[str, object]) -> list[str]:
    return sorted(k for k in before if before[k] != after.get(k))


# --- fixtures -------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _session_isolation():
    """セッション全体の隔離を終了時に戻し、実データが変わっていないことを確認する。"""
    before = real_state_fingerprint()
    try:
        yield _SESSION_TMP
    finally:
        _restore_session_isolation()
        after = real_state_fingerprint()
        shutil.rmtree(_SESSION_TMP, ignore_errors=True)
    changed = _changed_keys(before, after)
    if changed:
        pytest.fail(
            "テスト実行で実データが変化した（テストの隔離漏れ）: " + ", ".join(changed)
        )


@pytest.fixture(scope="session")
def session_isolation_dir(_session_isolation) -> Path:
    """セッション共通の隔離先（ログ・VLM JSON）。"""
    return _SESSION_TMP


@pytest.fixture(scope="session")
def real_state(_session_isolation):
    """実データ指紋を返す関数（テストからの確認・記録用）。"""
    return real_state_fingerprint


@pytest.fixture(autouse=True)
def isolated_app_config(tmp_path, monkeypatch):
    """テストごとに専用の config.ini と input_dir を与える（REQ-1.1〜1.2）。

    テストモジュール側で import 時に ``CONFIG_PATH`` を書き換える必要は無くなった。
    ``monkeypatch`` なのでテスト終了時に必ずセッション既定へ戻る。

    名前を ``_isolated_config`` にしないのは、テストモジュール側に同名の
    ローカル fixture があるとそちらが優先されてこの隔離が効かなくなるため。
    """
    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", config_path)
    monkeypatch.setattr(constants, "CONFIG_PATH", config_path)
    monkeypatch.setattr(utils, "CONFIG_PATH", config_path)
    monkeypatch.setattr(tagging_core, "CONFIG_PATH", config_path)
    # get_default_config()/load_settings() の input_dir 既定値だけを tmp へ向ける。
    # constants.BASE_DIR は lang/・models/ の所在なので触らない。
    monkeypatch.setattr(app_settings, "BASE_DIR", tmp_path)
    assert not is_inside_repo(config_path), config_path
    return config_path


@pytest.fixture(autouse=True)
def _no_real_state_changes():
    """テスト 1 件ごとに実データが変化していないことを確認する。"""
    before = real_state_fingerprint()
    yield
    after = real_state_fingerprint()
    changed = _changed_keys(before, after)
    if changed:
        pytest.fail(
            "このテストが実データを変更した（隔離漏れ）: " + ", ".join(changed)
        )


@pytest.fixture(autouse=True)
def _assert_write_targets_are_isolated(isolated_app_config):
    """書き込み先がリポジトリ内を指したままのテストを即座に失敗させる。"""
    targets = {
        "app_settings.CONFIG_PATH": app_settings.CONFIG_PATH,
        "constants.CONFIG_PATH": constants.CONFIG_PATH,
        "utils.CONFIG_PATH": utils.CONFIG_PATH,
        "utils.LOG_FILE_PATH": utils.LOG_FILE_PATH,
        "utils.ERROR_LOG_PATH": utils.ERROR_LOG_PATH,
        "utils.LOG_ROTATED_PATH": utils.LOG_ROTATED_PATH,
        "tagging_core.CONFIG_PATH": tagging_core.CONFIG_PATH,
        "vlm_config.VLM_CONNECTIONS_PATH": vlm_config.VLM_CONNECTIONS_PATH,
        "vlm_config.VLM_PROFILES_PATH": vlm_config.VLM_PROFILES_PATH,
    }
    leaked = sorted(name for name, path in targets.items() if is_inside_repo(path))
    assert not leaked, "実ファイルを指したままの書き込み先がある: " + ", ".join(leaked)
    app_base = app_settings.BASE_DIR
    assert not is_inside_repo(app_base), (
        "app_settings.BASE_DIR がリポジトリを指している（既定 input_dir が実フォルダになる）: "
        f"{app_base}"
    )


# --- 順序依存の検査用フック ------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """``PYTEST_SHUFFLE_SEED`` が設定されたときだけ収集順を決定的に並べ替える。

    順序依存（import 時のグローバル変更やテスト間の状態持ち越し）を、プラグインを
    追加せずに検査するための入口。既定では何もしない。
    """
    seed = os.environ.get("PYTEST_SHUFFLE_SEED")
    if not seed:
        return
    random.Random(seed).shuffle(items)
