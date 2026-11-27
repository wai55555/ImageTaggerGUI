"""Phase 4（細かい実バグ修正）のテスト。

BUG-1: `save_config()` が float を丸める／未知キーを落とす
BUG-2: `delete_secret()` の削除成否判定
BUG-3: `.env` の値のクォート除去
BUG-4: カスタム接続の read_timeout 既定値
BUG-5: 翻訳 CSV の行対応ズレ
PERF-1: モデル DL 進捗シグナルの間引き
"""
from __future__ import annotations

import configparser
from pathlib import Path

import pytest

import utils


@pytest.fixture()
def debug_log_on(tmp_path, monkeypatch):
    log_path = tmp_path / "debug_log.txt"
    debug_settings = utils.get_debug_settings()
    original_enabled = debug_settings.debug_log_enabled
    debug_settings.debug_log_enabled = True
    monkeypatch.setattr(utils, "LOG_FILE_PATH", log_path)
    monkeypatch.setattr(utils, "_LOG_FH", None)
    monkeypatch.setattr(utils, "_LOG_LAST_FLUSH", 0.0)
    try:
        yield log_path
    finally:
        utils._flush_debug_log()
        handle = utils._LOG_FH
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        utils._LOG_FH = None
        debug_settings.debug_log_enabled = original_enabled


@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    import app_settings

    path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", path)
    return path


# ---------------------------------------------------------------------------
# BUG-1: save_config の丸め・未知キー消失（REQ-5.1〜5.4）
# ---------------------------------------------------------------------------


def test_float_thresholds_round_trip_without_rounding(config_path: Path):
    import app_settings

    settings = app_settings.load_settings(app_settings.load_config())
    settings.thresholds.general = 0.455
    settings.thresholds.character = 0.655
    assert app_settings.save_config(settings) is True

    text = config_path.read_text(encoding="utf-8")
    assert "general = 0.455" in text
    assert "character = 0.655" in text

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.thresholds.general == 0.455
    assert reloaded.thresholds.character == 0.655


def test_float_save_is_idempotent(config_path: Path):
    import app_settings

    settings = app_settings.load_settings(app_settings.load_config())
    settings.thresholds.general = 0.455
    app_settings.save_config(settings)
    first = config_path.read_text(encoding="utf-8")

    again = app_settings.load_settings(app_settings.load_config())
    app_settings.save_config(again)
    second = config_path.read_text(encoding="utf-8")
    assert first == second


def test_unknown_keys_and_sections_survive_save(config_path: Path):
    import app_settings

    config_path.write_text(
        "[Thresholds]\n"
        "general = 0.10\n"
        "hand_added = keep-me\n"
        "\n"
        "[MyOwnSection]\n"
        "custom_key = custom value\n"
        "\n"
        "[General]\n"
        "language_code = en\n",
        encoding="utf-8",
    )
    settings = app_settings.load_settings(app_settings.load_config())
    settings.thresholds.general = 0.4
    assert app_settings.save_config(settings) is True

    parsed = configparser.ConfigParser()
    parsed.read(config_path, encoding="utf-8")
    # 既知キーは dataclass 側の値で上書きされる
    assert parsed.getfloat("Thresholds", "general") == 0.4
    # 未知キー・未知セクションは保持される
    assert parsed.get("Thresholds", "hand_added") == "keep-me"
    assert parsed.get("MyOwnSection", "custom_key") == "custom value"


def test_legacy_model_verified_key_is_dropped_on_save(config_path: Path):
    import app_settings

    config_path.write_text(
        "[Model]\n"
        "model_id = pixai-tagger-v0.9\n"
        "verified = True\n",
        encoding="utf-8",
    )
    # 読み取り側は旧キーを最優先して移行する（現行挙動）
    settings = app_settings.load_settings(app_settings.load_config())
    assert settings.model.verified_models == {"pixai-tagger-v0.9": True}

    assert app_settings.save_config(settings) is True
    parsed = configparser.ConfigParser()
    parsed.read(config_path, encoding="utf-8")
    assert not parsed.has_option("Model", "verified"), "旧キーは保存時に落とす"
    assert parsed.get("Model", "verified_models") == "pixai-tagger-v0.9:True"

    # 次回読込でも verified_models 側が有効（旧キーが残っていないので上書きされない）
    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.model.verified_models == {"pixai-tagger-v0.9": True}


def test_update_model_verification_status_survives_reload(config_path: Path, monkeypatch):
    import app_settings

    config_path.write_text("[Model]\nmodel_id = pixai-tagger-v0.9\nverified = False\n",
                           encoding="utf-8")
    app_settings.update_model_verification_status(
        "pixai-tagger-v0.9", True, lambda section, key, **kwargs: key)
    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.model.verified_models == {"pixai-tagger-v0.9": True}


def test_broken_config_still_saves_and_logs_fallback(config_path: Path, debug_log_on: Path):
    import app_settings

    config_path.write_text("this is not a section header\nkey = value\n", encoding="utf-8")
    # 壊れた config は load_config() では読めない（現行どおり例外）。保存側だけが
    # 「読めなくても保存は成功させる」フォールバックを持つことを確認する。
    settings = app_settings.load_settings(app_settings.get_default_config())
    settings.thresholds.general = 0.42

    assert app_settings.save_config(settings) is True
    parsed = configparser.ConfigParser()
    parsed.read(config_path, encoding="utf-8")  # 読み直せる config になっている
    assert parsed.getfloat("Thresholds", "general") == 0.42

    utils._flush_debug_log()
    assert "could not be parsed" in debug_log_on.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# BUG-2: delete_secret の成否判定（REQ-5.5〜5.7）
# ---------------------------------------------------------------------------


class _FakeKeyring:
    """削除の読み直し挙動を差し替えられる最小の keyring 相当。"""

    def __init__(self, *, read_error: bool = False, sticky: dict | None = None):
        self.store: dict[tuple[str, str], str] = {}
        self.read_error = read_error
        self.sticky = sticky or {}

    def get_password(self, service, ref):
        if self.read_error:
            raise RuntimeError("backend cannot read")
        if (service, ref) in self.sticky:
            return self.sticky[(service, ref)]
        return self.store.get((service, ref))

    def set_password(self, service, ref, value):
        self.store[(service, ref)] = value

    def delete_password(self, service, ref):
        self.store.pop((service, ref), None)


@pytest.fixture()
def secrets_mod(monkeypatch):
    import vlm_secrets

    monkeypatch.setattr(vlm_secrets, "_session_store", {})
    return vlm_secrets


def test_delete_secret_treats_read_error_as_removed(secrets_mod, monkeypatch):
    """未登録時に例外を返すバックエンド: 実際は消えているので True（旧実装は False）。"""
    fake = _FakeKeyring(read_error=True)
    monkeypatch.setattr(secrets_mod, "keyring", fake)
    assert secrets_mod.delete_secret("vlm/gemini/api_key") is True


def test_delete_secret_reports_false_when_value_still_readable(secrets_mod, monkeypatch):
    fake = _FakeKeyring(sticky={(secrets_mod._SERVICE, "vlm/gemini/api_key"): "still-here"})
    monkeypatch.setattr(secrets_mod, "keyring", fake)
    assert secrets_mod.delete_secret("vlm/gemini/api_key") is False


def test_delete_secret_reports_false_when_only_legacy_remains(secrets_mod, monkeypatch):
    fake = _FakeKeyring(sticky={(secrets_mod._LEGACY_SERVICE, "vlm/gemini/api_key"): "old"})
    monkeypatch.setattr(secrets_mod, "keyring", fake)
    assert secrets_mod.delete_secret("vlm/gemini/api_key") is False


def test_delete_secret_without_keyring_is_true(secrets_mod, monkeypatch):
    monkeypatch.setattr(secrets_mod, "keyring", None)
    assert secrets_mod.delete_secret("vlm/gemini/api_key") is True


def test_delete_secret_clears_session_copy(secrets_mod, monkeypatch):
    monkeypatch.setattr(secrets_mod, "keyring", None)
    secrets_mod._session_store["vlm/gemini/api_key"] = "session-value"
    assert secrets_mod.delete_secret("vlm/gemini/api_key") is True
    assert "vlm/gemini/api_key" not in secrets_mod._session_store


# ---------------------------------------------------------------------------
# BUG-3: .env のクォート除去（REQ-5.8 / REQ-5.9）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("abc", "abc"),
    ('"abc"', "abc"),
    ("'abc'", "abc"),
    ('"abc\'', '"abc\''),      # ミスマッチはそのまま
    ("'abc\"", "'abc\""),
    ('""x""', '"x"'),           # 外側 1 組だけ
    ('"', '"'),                 # 1 文字はそのまま
    ("''", ""),                 # 空クォート
    ('""', ""),
    ("", ""),
    ("a=b", "a=b"),             # 値の途中の = は partition 側で処理済み
])
def test_unquote_dotenv_value(secrets_mod, raw, expected):
    assert secrets_mod._unquote_dotenv_value(raw) == expected


def test_dotenv_loading_keeps_unintentional_quotes(secrets_mod, monkeypatch, tmp_path):
    """クォートしていない値の末尾の `"` を壊さない（旧実装は剥がしていた）。"""
    import os

    (tmp_path / ".env").write_text(
        "PHASE4_BUG3_PLAIN=abc\"\n"
        "PHASE4_BUG3_QUOTED=\"def\"\n"
        "PHASE4_BUG3_MISMATCH=\"ghi'\n"
        "export PHASE4_BUG3_EXPORT='jkl'\n",
        encoding="utf-8",
    )
    for name in ("PHASE4_BUG3_PLAIN", "PHASE4_BUG3_QUOTED", "PHASE4_BUG3_MISMATCH",
                 "PHASE4_BUG3_EXPORT"):
        monkeypatch.delenv(name, raising=False)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        secrets_mod._load_dotenv()
        assert os.environ["PHASE4_BUG3_PLAIN"] == 'abc"'
        assert os.environ["PHASE4_BUG3_QUOTED"] == "def"
        assert os.environ["PHASE4_BUG3_MISMATCH"] == "\"ghi'"
        assert os.environ["PHASE4_BUG3_EXPORT"] == "jkl"
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# BUG-4: カスタム接続の read_timeout 既定値（REQ-5.10〜5.12）
# ---------------------------------------------------------------------------


def _qt_app():
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_read_timeout_default_comes_from_retry_policy():
    """数値の 60 をテスト側に直書きせず、RetryPolicy の既定と一致することだけを固定する。"""
    app = _qt_app()
    import vlm_connections
    from custom_connection_dialog import CustomConnectionDialog

    expected = int(vlm_connections.RetryPolicy().read_timeout_s)
    dialog = CustomConnectionDialog(lambda section, key, **kw: key)
    app.processEvents()
    try:
        assert dialog.read_timeout.value() == expected
    finally:
        dialog.close()


def test_missing_retry_block_falls_back_to_retry_policy_default():
    app = _qt_app()
    import vlm_connections
    from custom_connection_dialog import CustomConnectionDialog

    expected = int(vlm_connections.RetryPolicy().read_timeout_s)
    dialog = CustomConnectionDialog(
        lambda section, key, **kw: key,
        existing={"display_name": "x", "base_url": "https://example.test/v1",
                  "model_id": "m", "auth": {"type": "none"}},
    )
    app.processEvents()
    try:
        assert dialog.read_timeout.value() == expected
        # spin の範囲は現行のまま
        assert dialog.read_timeout.minimum() == 5
        assert dialog.read_timeout.maximum() == 900
    finally:
        dialog.close()


def test_explicit_read_timeout_is_preserved():
    """明示的に保存済みの値（例 120）は書き換えない（REQ-5.11）。"""
    app = _qt_app()
    from custom_connection_dialog import CustomConnectionDialog

    dialog = CustomConnectionDialog(
        lambda section, key, **kw: key,
        existing={"base_url": "https://example.test/v1", "model_id": "m",
                  "auth": {"type": "none"}, "retry": {"read_timeout_s": 120}},
    )
    app.processEvents()
    try:
        assert dialog.read_timeout.value() == 120
        dialog.name_edit.setText("kept")
        dialog._on_save()  # 何も触らず保存しても 120 のまま
        assert dialog._result is not None
        assert dialog._result["retry"]["read_timeout_s"] == 120
    finally:
        dialog.close()
