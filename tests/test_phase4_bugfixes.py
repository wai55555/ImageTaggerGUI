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
