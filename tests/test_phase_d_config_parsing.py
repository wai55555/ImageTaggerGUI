"""Phase D（設定・秘密情報のパース堅牢性）のテスト。

対象の指摘と要件:
  R1-1 / R9-6  -> REQ-D1  config.ini の読み側ガードと % の非対称
  R14-4        -> REQ-D2  [Thresholds] の nan／inf
  R29-2        -> REQ-D3  gpu_setup_prompt の「オフ意図」の別名
  R18-4        -> REQ-D4  PixAI v0.9 の character 既定しきい値の単一正本
  R14-3        -> REQ-D5  死んだ VLM 生成設定5項目のコア側配線
  R3-S1        -> REQ-D6  .env の取り込み許可リスト
  R9-9         -> REQ-D7  .env の BOM
  R32-1        -> REQ-D8  keyring プローブの負キャッシュと誤削除
  R9-7         -> REQ-D9  VLM 設定 JSON の version ガード
  R9-10        -> REQ-D10 aliases の文字列手編集

§15.2 の隔離原則に従い、実 config.ini / 実 .env / 実 keyring は触らない。
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import app_settings
import constants
import vlm_config
import vlm_secrets


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", path)
    return path


@pytest.fixture()
def app_log(monkeypatch):
    """app_settings の debug log を捕まえる（app_settings は名前を直接 import している）。"""
    lines: list[str] = []
    monkeypatch.setattr(app_settings, "write_debug_log", lambda msg, *a, **k: lines.append(str(msg)))
    return lines


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# REQ-D1: 読み側ガードと % の非対称
# ---------------------------------------------------------------------------


def test_invalid_numbers_fall_back_per_key_with_debug_log(config_path, app_log):
    """1つの不正値で起動不能にならず、キー単位で fallback へ倒れる（REQ-D1(a)）。"""
    _write(config_path, """[Thresholds]
general = abc
character = 0.75
[Limits]
general = 1.5
character = 2
[Behavior]
enable_solo_character_limit = maybe
convert_underscore_to_space = false
[Window]
tag_display_rows = x
tag_display_cols = 7
[Debug]
debug_log = on-and-off
""")

    settings = app_settings.load_settings(app_settings.load_config())

    assert settings.thresholds.general == 0.40          # 不正値 -> 既定
    assert settings.thresholds.character == 0.75       # 正常値はそのまま
    assert settings.limits.general == 55               # 小数は getint で不正 -> 既定
    assert settings.limits.character == 2
    assert settings.behavior.enable_solo_character_limit is True
    assert settings.behavior.convert_underscore_to_space is False
    assert settings.window.tag_display_rows == 6
    assert settings.window.tag_display_cols == 7
    assert settings.debug.debug_log is False
    # 不正値は debug log に残る（黙って無視しない）。
    joined = "\n".join(app_log)
    assert "[Thresholds] general" in joined
    assert "[Limits] general" in joined
    assert "[Behavior] enable_solo_character_limit" in joined
    assert "[Window] tag_display_rows" in joined
    assert "[Debug] debug_log" in joined


def test_broken_config_syntax_starts_with_defaults(config_path, app_log):
    """config.read 自体が失敗しても既定値で起動する（config.ini 削除を強制しない）。"""
    _write(config_path, "general = 0.40\n[Thresholds\ncharacter = 0.9\n")

    config = app_settings.load_config()
    settings = app_settings.load_settings(config)

    assert settings.thresholds.general == 0.40
    assert settings.thresholds.character == app_settings.DEFAULT_CHARACTER_THRESHOLD
    assert any("could not be parsed" in line for line in app_log)
    # 利用者が直せるよう、壊れたファイル自体は書き換えない。
    assert "general = 0.40" in config_path.read_text(encoding="utf-8")


def test_undecodable_config_starts_with_defaults(config_path, app_log):
    """UTF-8 でない config.ini でも既定値で起動する（REQ-D1(a)）。"""
    config_path.write_bytes(b"[Thresholds]\ngeneral = 0.40\n# \xff\xfe\n")

    settings = app_settings.load_settings(app_settings.load_config())

    assert settings.thresholds.general == 0.40


def test_percent_in_path_round_trips(config_path):
    """アプリ自身が書いた `%` を含むパスが次回読み込みで壊れない（REQ-D1(b)）。"""
    settings = app_settings.load_settings(app_settings.load_config())
    settings.paths.input_dir = r"C:\work\100% done\images"
    assert app_settings.save_config(settings) is True

    text = config_path.read_text(encoding="utf-8")
    assert "100% done" in text

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.paths.input_dir == r"C:\work\100% done\images"


def test_load_settings_receives_interpolation_free_parsers(config_path):
    """`load_settings` へ渡すパーサは補間なし（REQ-D1(b)）。

    app_settings 側の構築箇所（既定／保存／未知キー取り込み）と tagging_core の
    独自構築の両方が対象。ソース上の構築はタグ付け本体の引数なし ConfigParser() が
    残っていないことを静的に確認する。
    """
    import configparser as _cp

    # 補間なしを指定すると CPython は基底の Interpolation（括弧を展開しない）を入れる。
    assert type(app_settings.get_default_config()._interpolation) is _cp.Interpolation
    assert type(app_settings.load_config()._interpolation) is _cp.Interpolation

    # 実際の読みでも `%` が補間されない（既定の補間なら InterpolationSyntaxError）。
    config_path.write_text("[Paths]\ninput_dir = C:\\\\100% done\n", encoding="utf-8")
    assert app_settings.load_config().get("Paths", "input_dir") == r"C:\\100% done"

    source = (Path(__file__).resolve().parent.parent / "src" / "tagging_core.py").read_text(
        encoding="utf-8")
    assert "configparser.ConfigParser()" not in source
    assert "configparser.ConfigParser(interpolation=None)" in source


# ---------------------------------------------------------------------------
# REQ-D2: nan / inf は不正値
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "NaN", "1e999"])
def test_non_finite_thresholds_fall_back(config_path, app_log, raw):
    _write(config_path, f"[Thresholds]\ngeneral = {raw}\ncharacter = {raw}\n")

    settings = app_settings.load_settings(app_settings.load_config())

    assert settings.thresholds.general == 0.40
    assert settings.thresholds.character == app_settings.DEFAULT_CHARACTER_THRESHOLD
    assert any("non-finite" in line for line in app_log)


def test_non_finite_vlm_temperature_falls_back(config_path, app_log):
    _write(config_path, "[Vlm]\ntemperature = nan\ntop_p = inf\n")

    settings = app_settings.load_settings(app_settings.load_config())

    assert settings.vlm.temperature is None
    assert settings.vlm.top_p is None
    assert any("[Vlm] temperature" in line for line in app_log)


# ---------------------------------------------------------------------------
# REQ-D3: gpu_setup_prompt のオフ意図の別名
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["off", "OFF", "false", "no", "0", "never", "disabled", "none"])
def test_gpu_setup_prompt_accepts_off_aliases(raw):
    assert app_settings.parse_gpu_setup_prompt(raw) == "dismissed"


@pytest.mark.parametrize("raw,expected", [
    ("ask", "ask"),
    ("dismissed", "dismissed"),
    ("", "ask"),
    ("maybe", "ask"),
])
def test_gpu_setup_prompt_keeps_existing_values(raw, expected):
    assert app_settings.parse_gpu_setup_prompt(raw) == expected


def test_gpu_setup_prompt_off_alias_via_config(config_path):
    _write(config_path, "[Behavior]\ngpu_setup_prompt = off\n")
    settings = app_settings.load_settings(app_settings.load_config())
    assert settings.behavior.gpu_setup_prompt == "dismissed"


def test_off_aliases_are_shared_by_both_settings():
    """update_check の従来挙動が別名表の共用で変わっていないこと。"""
    for raw in ("false", "no", "off", "0", "never", "disabled", "disable", "none"):
        assert app_settings.parse_update_check(raw) == "dismissed"
    assert app_settings.parse_update_check("ask") == "ask"


# ---------------------------------------------------------------------------
# REQ-D4: character 既定しきい値の単一正本（0.85）
# ---------------------------------------------------------------------------


def test_character_threshold_has_a_single_source_of_truth(config_path):
    assert app_settings.DEFAULT_CHARACTER_THRESHOLD == 0.85

    # 既定 config.ini の値と、キー欠落時の fallback が同じ定数から引く。
    assert app_settings.get_default_config().getfloat("Thresholds", "character") == 0.85
    _write(config_path, "[Thresholds]\ngeneral = 0.40\n")
    settings = app_settings.load_settings(app_settings.load_config())
    assert settings.thresholds.character == app_settings.DEFAULT_CHARACTER_THRESHOLD

    # モデル側の既定（model_registry）と UI フォールバック（model_mode_controller）も同じ。
    import model_mode_controller
    import model_registry

    entry = model_registry.get_model_entry("pixai-tagger-v0.9")
    assert entry is not None
    assert entry.config["ui"]["default_character_threshold"] == pytest.approx(0.85)
    assert (model_mode_controller.ModelModeController._FALLBACK_THRESHOLDS["character"]
            == app_settings.DEFAULT_CHARACTER_THRESHOLD)

    controller = model_mode_controller.ModelModeController.__new__(
        model_mode_controller.ModelModeController)
    assert controller._model_default_threshold({}, "character") == pytest.approx(0.85)
    # general の 0.40 は対象外（全箇所一致したまま）。
    assert controller._model_default_threshold({}, "general") == pytest.approx(0.40)


def test_default_model_effective_character_threshold_stays_085():
    """既定モデルでの実効値が従来どおり 0.85 であること（R18-4 の意図的変更なし）。"""
    import model_registry

    ui = model_registry.get_model_entry("pixai-tagger-v0.9").config["ui"]
    assert float(ui["default_character_threshold"]) == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# REQ-D5: VLM 生成設定5項目のコア側配線
# ---------------------------------------------------------------------------


def test_handwritten_vlm_generation_keys_reach_the_profile(config_path):
    _write(config_path, """[Vlm]
custom_system_prompt = Describe only the character.
temperature = 0.4
top_p = 0.9
image_format = png
image_jpeg_quality = 77
""")

    settings = app_settings.load_settings(app_settings.load_config())
    vlm = settings.vlm
    assert vlm.custom_system_prompt == "Describe only the character."
    assert vlm.temperature == pytest.approx(0.4)
    assert vlm.top_p == pytest.approx(0.9)
    assert vlm.image_format == "png"
    assert vlm.image_jpeg_quality == 77

    profile = vlm_config.build_generation_profile(vlm)
    assert profile.custom_system_prompt == "Describe only the character."
    assert profile.temperature == pytest.approx(0.4)
    assert profile.top_p == pytest.approx(0.9)
    assert profile.image_format == "png"
    assert profile.image_jpeg_quality == 77


def test_unset_generation_keys_keep_the_old_defaults(config_path):
    settings = app_settings.load_settings(app_settings.load_config())
    vlm = settings.vlm
    assert vlm.custom_system_prompt == ""
    assert vlm.temperature is None
    assert vlm.top_p is None
    assert vlm.image_format == "auto"
    assert vlm.image_jpeg_quality == 90

    import vlm_profiles
    profile = vlm_config.build_generation_profile(vlm)
    default = vlm_profiles.GenerationProfile()
    assert profile.temperature is None and default.temperature is None
    assert profile.top_p is None and default.top_p is None
    assert profile.custom_system_prompt == ""
    assert profile.image_format == "auto"
    assert profile.image_jpeg_quality == 90


def test_none_temperature_round_trips_as_blank_not_none_text(config_path):
    """未設定（None）は "None" ではなく空文字で往復させる（REQ-D5）。"""
    settings = app_settings.load_settings(app_settings.load_config())
    assert app_settings.save_config(settings) is True

    text = config_path.read_text(encoding="utf-8")
    assert "temperature = None" not in text
    assert "top_p = None" not in text

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.vlm.temperature is None
    assert reloaded.vlm.top_p is None


# ---------------------------------------------------------------------------
# REQ-D6: .env の取り込み許可リスト
# ---------------------------------------------------------------------------


class _FakeKeyring:
    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service, ref):
        return self.store.get((service, ref))

    def set_password(self, service, ref, value):
        self.store[(service, ref)] = value


@pytest.fixture()
def isolated_dotenv(tmp_path, monkeypatch):
    """BASE_DIR/.env と cwd/.env の両候補を一時ディレクトリへ閉じ込める。"""
    fake = _FakeKeyring()
    monkeypatch.setattr(vlm_secrets, "keyring", fake)
    monkeypatch.setattr(vlm_secrets, "_session_store", {})
    monkeypatch.setattr(constants, "BASE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_dotenv_does_not_import_keys_the_app_never_reads(isolated_dotenv, monkeypatch):
    """TLS 系キーを含む .env で os.environ が汚染されないこと（R3-S1）。"""
    hostile = {
        "REQUESTS_CA_BUNDLE": "",
        "SSL_CERT_FILE": "",
        "CURL_CA_BUNDLE": "",
        "PYTHONHTTPSVERIFY": "0",
        "PHASE_D_UNRELATED": "x",
    }
    for name in hostile:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _write(isolated_dotenv / ".env",
           "".join(f"{k}={v}\n" for k, v in hostile.items())
           + "GEMINI_API_KEY=g_key\n")

    vlm_secrets._load_dotenv()

    for name in hostile:
        assert name not in __import__("os").environ
    # 正規の鍵は従来どおり解決される。
    assert vlm_secrets.get_secret("vlm/gemini/api_key") == "g_key"


def test_dotenv_imports_alias_and_generic_pixai_names(isolated_dotenv, monkeypatch):
    for name in ("CLOUDFLARE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID",
                 "PIXAI_VLM_CUSTOM_ABC"):
        monkeypatch.delenv(name, raising=False)
    _write(isolated_dotenv / ".env",
           "CLOUDFLARE_ACCOUNT_ID=acct\n"
           "ANTHROPIC_WORKSPACE_ID=ws\n"
           "PIXAI_VLM_CUSTOM_ABC=secret\n")

    vlm_secrets._load_dotenv()

    import os
    assert os.environ["CLOUDFLARE_ACCOUNT_ID"] == "acct"
    assert os.environ["ANTHROPIC_WORKSPACE_ID"] == "ws"
    assert vlm_secrets.get_secret("vlm/custom/abc") == "secret"


def test_dotenv_allowlist_is_derived_from_env_aliases(isolated_dotenv, monkeypatch):
    """REQ-D6: 許可リストは _ENV_ALIASES から生成され、別名追加に自動追随する。

    REQ-F8（xai の XAI_API_KEY 追加）と実施順序に依存しないことの確認。
    """
    monkeypatch.delenv("PHASE_D_NEW_ALIAS", raising=False)
    monkeypatch.setitem(vlm_secrets._ENV_ALIASES, "vlm/phase-d/api_key",
                        ("PHASE_D_NEW_ALIAS",))
    try:
        _write(isolated_dotenv / ".env", "PHASE_D_NEW_ALIAS=v\n")
        vlm_secrets._load_dotenv()
        import os
        assert os.environ["PHASE_D_NEW_ALIAS"] == "v"
    finally:
        monkeypatch.delenv("PHASE_D_NEW_ALIAS", raising=False)


# ---------------------------------------------------------------------------
# REQ-D7: .env の BOM
# ---------------------------------------------------------------------------


def test_dotenv_with_bom_reads_the_first_key(isolated_dotenv, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (isolated_dotenv / ".env").write_text("GEMINI_API_KEY=bom_key\n", encoding="utf-8-sig")

    vlm_secrets._load_dotenv()

    assert vlm_secrets.get_secret("vlm/gemini/api_key") == "bom_key"


# ---------------------------------------------------------------------------
# REQ-D8: keyring プローブの負キャッシュと誤削除
# ---------------------------------------------------------------------------


class _FlakyKeyring:
    """初回のプローブで失敗し、その後は使えるバックエンドを返すフェイク。"""

    priority = 5

    def __init__(self):
        self.probes = 0
        self.store: dict[tuple[str, str], str] = {}

    def get_keyring(self):
        self.probes += 1
        if self.probes == 1:
            raise RuntimeError("backend not ready")
        return self

    def get_password(self, service, ref):
        return self.store.get((service, ref))

    def set_password(self, service, ref, value):
        self.store[(service, ref)] = value


def test_keyring_probe_retries_after_a_failure(monkeypatch):
    """初回失敗を固定せず、保存操作の時点で再プローブする（REQ-D8(a)）。"""
    flaky = _FlakyKeyring()
    monkeypatch.setattr(vlm_secrets, "keyring", flaky)
    monkeypatch.setattr(vlm_secrets, "_session_store", {})
    monkeypatch.setattr(vlm_secrets, "_keyring_probe", None)

    assert vlm_secrets.keyring_available() is False   # 初回は失敗
    assert vlm_secrets.keyring_available() is True    # 回復後は再プローブで真
    assert vlm_secrets.set_secret("vlm/openai/api_key", "k", persist=True) is True
    assert vlm_secrets.get_secret("vlm/openai/api_key") == "k"
    assert flaky.store[(vlm_secrets._SERVICE, "vlm/openai/api_key")] == "k"


def test_persist_choice_ignores_a_probe_disabled_checkbox():
    """無効化されたチェックボックスの False を「利用者の切替」と解釈しない（REQ-D8(b)）。"""
    assert vlm_secrets.persist_choice_from_checkbox(False, enabled=False) == (False, False)
    assert vlm_secrets.persist_choice_from_checkbox(False, enabled=True) == (False, True)
    assert vlm_secrets.persist_choice_from_checkbox(True, enabled=True) == (True, False)


def test_disabled_persist_checkbox_does_not_delete_the_existing_key(monkeypatch):
    """プローブ由来で無効化されたチェックボックスでは既存キーを削除しない（R32-1）。"""
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication([])

    import custom_connection_dialog as ccd

    fake = _FakeKeyring()
    monkeypatch.setattr(vlm_secrets, "keyring", fake)
    monkeypatch.setattr(vlm_secrets, "_session_store", {})
    # keyring が使えない環境を模す（チェックボックスは未チェック＋無効になる）。
    monkeypatch.setattr(vlm_secrets, "keyring_available", lambda: False)
    existing_ref = "vlm/custom/existing"
    fake.set_password(vlm_secrets._SERVICE, existing_ref, "stored-secret")

    calls: list[str] = []
    monkeypatch.setattr(vlm_secrets, "delete_secret",
                        lambda ref: calls.append(ref) or True)

    existing = {
        "connection_id": "existing",
        "display_name": "Local",
        "kind": "custom_local",
        "protocol": "openai_chat_completions",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "m",
        "auth": {"type": "bearer", "secret_ref": existing_ref},
    }
    dialog = ccd.CustomConnectionDialog(lambda section, key, **kw: key, existing=existing)
    try:
        assert dialog.persist_key_check.isChecked() is False
        assert dialog.persist_key_check.isEnabled() is False
        dialog.api_key_edit.setText("new-secret")
        dialog._on_save()
    finally:
        dialog.deleteLater()

    assert calls == []                                        # 削除分岐に入らない
    assert fake.store[(vlm_secrets._SERVICE, existing_ref)] == "stored-secret"


def test_enabled_persist_checkbox_still_deletes_the_old_key(monkeypatch):
    """利用者が自分でチェックを外したときは従来どおり永続キーを消す（回帰防止）。"""
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    _ = QApplication.instance() or QApplication([])

    import custom_connection_dialog as ccd

    fake = _FakeKeyring()
    monkeypatch.setattr(vlm_secrets, "keyring", fake)
    monkeypatch.setattr(vlm_secrets, "_session_store", {})
    monkeypatch.setattr(vlm_secrets, "keyring_available", lambda: True)
    existing_ref = "vlm/custom/existing"
    fake.set_password(vlm_secrets._SERVICE, existing_ref, "stored-secret")

    calls: list[str] = []
    monkeypatch.setattr(vlm_secrets, "delete_secret",
                        lambda ref: calls.append(ref) or True)

    existing = {
        "connection_id": "existing",
        "display_name": "Local",
        "kind": "custom_local",
        "protocol": "openai_chat_completions",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "m",
        "auth": {"type": "bearer", "secret_ref": existing_ref},
    }
    dialog = ccd.CustomConnectionDialog(lambda section, key, **kw: key, existing=existing)
    try:
        assert dialog.persist_key_check.isEnabled() is True
        dialog.persist_key_check.setChecked(False)   # 利用者がセッション限定へ切替
        dialog.api_key_edit.setText("new-secret")
        dialog._on_save()
    finally:
        dialog.deleteLater()

    assert calls == [existing_ref]


# ---------------------------------------------------------------------------
# REQ-D9: VLM 設定 JSON の version ガード
# ---------------------------------------------------------------------------


@pytest.fixture()
def json_paths(tmp_path, monkeypatch):
    connections = tmp_path / "vlm_connections.json"
    profiles = tmp_path / "vlm_profiles.json"
    monkeypatch.setattr(vlm_config, "VLM_CONNECTIONS_PATH", connections)
    monkeypatch.setattr(vlm_config, "VLM_PROFILES_PATH", profiles)
    return connections, profiles


_VALID_CONNECTION = {
    "connection_id": "custom-x",
    "display_name": "X",
    "kind": "custom_local",
    "protocol": "openai_chat_completions",
    "base_url": "http://127.0.0.1:1234/v1",
    "model_id": "m",
    "auth": {"type": "none"},
}
_VALID_PROFILE = {
    "profile_id": "user-x-1",
    "bindings": {"openai": {"model_id": "gpt-x", "vlm_capable": True}},
}


def test_unknown_connection_version_is_rejected(json_paths, monkeypatch):
    connections, _ = json_paths
    logs: list[str] = []
    monkeypatch.setattr(vlm_config, "write_debug_log", lambda msg, *a, **k: logs.append(str(msg)))
    _write(connections, json.dumps({"version": 2, "connections": [_VALID_CONNECTION]}))

    assert vlm_config.load_custom_connections() == []
    assert any("unsupported version" in line for line in logs)


def test_supported_and_missing_connection_versions_load_as_before(json_paths):
    connections, _ = json_paths
    _write(connections, json.dumps({"version": 1, "connections": [_VALID_CONNECTION]}))
    assert vlm_config.load_custom_connections() == [_VALID_CONNECTION]

    _write(connections, json.dumps({"connections": [_VALID_CONNECTION]}))
    assert vlm_config.load_custom_connections() == [_VALID_CONNECTION]


def test_unknown_profile_version_is_rejected(json_paths):
    _, profiles = json_paths
    _write(profiles, json.dumps({"version": 2, "profiles": [_VALID_PROFILE]}))
    assert vlm_config.load_user_profiles() == []
    assert vlm_config.user_profile_objects() == []


def test_boolean_version_is_not_treated_as_version_1(json_paths):
    connections, _ = json_paths
    _write(connections, json.dumps({"version": True, "connections": [_VALID_CONNECTION]}))
    assert vlm_config.load_custom_connections() == []


def test_supported_and_missing_profile_versions_load_as_before(json_paths):
    _, profiles = json_paths
    _write(profiles, json.dumps({"version": 1, "profiles": [_VALID_PROFILE]}))
    assert vlm_config.load_user_profiles() == [_VALID_PROFILE]

    _write(profiles, json.dumps({"profiles": [_VALID_PROFILE]}))
    assert vlm_config.load_user_profiles() == [_VALID_PROFILE]
    objs = vlm_config.user_profile_objects()
    assert [p.profile_id for p in objs] == ["user-x-1"]


# ---------------------------------------------------------------------------
# REQ-D10: aliases の文字列手編集
# ---------------------------------------------------------------------------


def test_string_aliases_are_one_alias_not_one_per_character(json_paths):
    _, profiles = json_paths
    profile = dict(_VALID_PROFILE, aliases="solo-char")
    _write(profiles, json.dumps({"version": 1, "profiles": [profile]}))

    objs = vlm_config.user_profile_objects()

    assert [p.aliases for p in objs] == [("solo-char",)]
    assert "s" not in objs[0].aliases

    # リスト表記は従来どおり（空文字だけ捨てる）。
    profile = dict(_VALID_PROFILE, aliases=["a-1", "  ", "b-2"])
    _write(profiles, json.dumps({"version": 1, "profiles": [profile]}))
    assert vlm_config.user_profile_objects()[0].aliases == ("a-1", "b-2")


# ---------------------------------------------------------------------------
# §15.3 Phase D 追加検証: 正常な設定の読みが従来と一致する
# ---------------------------------------------------------------------------


def test_normal_config_round_trips_unchanged(config_path):
    """正常な config.ini の読みが従来と一致すること（互換の確認）。"""
    settings = app_settings.load_settings(app_settings.load_config())
    settings.thresholds.general = 0.45
    settings.limits.model = 5
    settings.behavior.existing_file_mode = "APPEND"
    settings.behavior.gpu_setup_prompt = "dismissed"
    settings.vlm.enabled = True
    settings.vlm.verified_bindings = "gemma-4-31b-it:gemini"
    settings.model.verified_models = {"pixai-tagger-v0.9": True}
    assert app_settings.save_config(settings) is True

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.thresholds == settings.thresholds
    assert reloaded.limits == settings.limits
    assert reloaded.behavior == settings.behavior
    assert reloaded.window == settings.window
    assert reloaded.model == settings.model
    assert reloaded.caption == settings.caption
    assert reloaded.vlm == settings.vlm
    assert reloaded.debug == settings.debug
    assert reloaded.paths == settings.paths


def test_normal_json_files_read_as_before(json_paths):
    connections, profiles = json_paths
    _write(connections, json.dumps({"version": 1, "connections": [_VALID_CONNECTION]}))
    _write(profiles, json.dumps({"version": 1, "profiles": [_VALID_PROFILE]}))

    assert vlm_config.load_custom_connections() == [_VALID_CONNECTION]
    assert vlm_config.load_user_profiles() == [_VALID_PROFILE]

    # 実行時に使う辞書にも従来どおり載る（内蔵 + カスタム）。
    connection_map = vlm_config.build_connection_map(
        SimpleNamespace(cloudflare_account_id="", anthropic_workspace_id="",
                        model_id_overrides="", model_profile_id="pixai-tagger-v0.9"))
    assert "custom-x" in connection_map
    assert "builtin-gemini" in connection_map
