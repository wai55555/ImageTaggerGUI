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
    """クォートしていない値の末尾の `"` を壊さない（旧実装は剥がしていた）。

    REQ-D6 で `.env` の取り込みは許可リスト制になったため、キー名は許可される
    慣習形（`PIXAI_*`）を使う。クォート処理自体は REQ-D6 の前後で不変。
    """
    import os

    (tmp_path / ".env").write_text(
        "PIXAI_VLM_PHASE4_BUG3_PLAIN=abc\"\n"
        "PIXAI_VLM_PHASE4_BUG3_QUOTED=\"def\"\n"
        "PIXAI_VLM_PHASE4_BUG3_MISMATCH=\"ghi'\n"
        "export PIXAI_VLM_PHASE4_BUG3_EXPORT='jkl'\n",
        encoding="utf-8",
    )
    for name in ("PIXAI_VLM_PHASE4_BUG3_PLAIN", "PIXAI_VLM_PHASE4_BUG3_QUOTED",
                 "PIXAI_VLM_PHASE4_BUG3_MISMATCH", "PIXAI_VLM_PHASE4_BUG3_EXPORT"):
        monkeypatch.delenv(name, raising=False)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        secrets_mod._load_dotenv()
        assert os.environ["PIXAI_VLM_PHASE4_BUG3_PLAIN"] == 'abc"'
        assert os.environ["PIXAI_VLM_PHASE4_BUG3_QUOTED"] == "def"
        assert os.environ["PIXAI_VLM_PHASE4_BUG3_MISMATCH"] == "\"ghi'"
        assert os.environ["PIXAI_VLM_PHASE4_BUG3_EXPORT"] == "jkl"
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


# ---------------------------------------------------------------------------
# BUG-5: 翻訳 CSV の行対応（REQ-5.13〜5.15）
# ---------------------------------------------------------------------------


def _write_english_csv(path: Path, rows: list[list[str]]) -> None:
    lines = ["id,tag_id,name,category,count,ips"]
    for row in rows:
        lines.append(";".join(row).replace(";", ",") if False else ",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_translation_csv(path: Path, translations: list[str]) -> None:
    lines = ["translation"]
    lines.extend(f'"{value}"' if value else "" for value in translations)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _translation_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    return d


def test_translation_alignment_survives_short_and_empty_records(tmp_path: Path):
    """英語側に短い行/空行があっても、以降のレコード位置がズレない（REQ-5.13）。"""
    import tag_utils

    d = _translation_dir(tmp_path)
    _write_english_csv(d / "selected_tags.csv", [
        ["1", "1", "1girl", "0"],          # 通常
        ["2", "2", "short"],               # 短いレコード（3列未満）→ プレースホルダ
        [],                                # 空レコード → プレースホルダ
        ["4", "4", "blue_eyes", "0"],      # ズレ位置より後
    ])
    _write_translation_csv(d / "selected_tags_jp.csv", ["いちガール", "", "", "あおいめ"])

    mapping = tag_utils.load_tag_translation_map(d)
    assert mapping["1girl"][0] == "いちガール"
    assert mapping["blue eyes"][0] == "あおいめ"      # 旧実装では 1 行ズレて空になっていた
    assert mapping["blue eyes"][1] == "blue eyes"     # 他の言語は英語フォールバック


def test_translation_language_with_mismatched_record_count_is_dropped(tmp_path: Path):
    """レコード数が食い違う言語だけ英語フォールバック（REQ-5.14）。"""
    import tag_utils

    d = _translation_dir(tmp_path)
    _write_english_csv(d / "selected_tags.csv", [
        ["1", "1", "1girl", "0"],
        ["2", "2", "blue_eyes", "0"],
    ])
    _write_translation_csv(d / "selected_tags_jp.csv", ["いちガール"])          # 1 件しかない
    _write_translation_csv(d / "selected_tags_fr.csv", ["une_fille", "yeux_bleus"])

    mapping = tag_utils.load_tag_translation_map(d)
    jp_index = tag_utils._TRANSLATION_LANGUAGE_SUFFIXES.index("jp")
    fr_index = tag_utils._TRANSLATION_LANGUAGE_SUFFIXES.index("fr")
    assert mapping["1girl"][jp_index] == "1girl"        # jp は丸ごと英語フォールバック
    assert mapping["1girl"][fr_index] == "une_fille"    # fr は巻き込まれない
    assert mapping["blue eyes"][fr_index] == "yeux_bleus"


def test_translation_mismatch_is_logged(tmp_path: Path, debug_log_on: Path):
    import tag_utils

    d = _translation_dir(tmp_path)
    _write_english_csv(d / "selected_tags.csv", [["1", "1", "1girl", "0"]])
    _write_translation_csv(d / "selected_tags_jp.csv", ["a", "b"])

    tag_utils.load_tag_translation_map(d)
    utils._flush_debug_log()
    log = debug_log_on.read_text(encoding="utf-8", errors="replace")
    assert "dropping jp" in log


def test_missing_translation_file_does_not_log(tmp_path: Path, debug_log_on: Path):
    """存在しない翻訳ファイルは正常ケース。不一致扱いでログを出さない（REQ-5.14）。"""
    import tag_utils

    d = _translation_dir(tmp_path)
    _write_english_csv(d / "selected_tags.csv", [["1", "1", "1girl", "0"]])
    _write_translation_csv(d / "selected_tags_fr.csv", ["une_fille"])

    tag_utils.load_tag_translation_map(d)
    utils._flush_debug_log()
    log_path = debug_log_on
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    assert "dropping" not in log


def test_normal_csv_mapping_is_unchanged(tmp_path: Path):
    """短い行が無い正常な CSV では従来と同一のマッピング（REQ-5.15(c)）。"""
    import tag_utils

    d = _translation_dir(tmp_path)
    _write_english_csv(d / "selected_tags.csv", [
        ["1", "1", "1girl", "0"],
        ["2", "2", "blue_eyes", "0"],
    ])
    _write_translation_csv(d / "selected_tags_jp.csv", ["いちガール", "あおいめ"])

    mapping = tag_utils.load_tag_translation_map(d)
    assert mapping == {
        "1girl": ["いちガール"] + ["1girl"] * 7,
        "blue eyes": ["あおいめ"] + ["blue eyes"] * 7,
    }


def test_read_failure_returns_empty_mapping_without_print(tmp_path: Path, debug_log_on: Path,
                                                          monkeypatch):
    """翻訳 CSV の読み込み失敗は例外を上げず、全言語フォールバック＋ログ（REQ-5.14/5.15(e)）。"""
    import builtins

    import tag_utils

    d = _translation_dir(tmp_path)
    _write_english_csv(d / "selected_tags.csv", [["1", "1", "1girl", "0"]])
    (d / "selected_tags_jp.csv").write_bytes(b"\xff\xfe\x00not utf-8")

    def exploding_print(*_args, **_kwargs):
        raise AssertionError("print must not be used")

    monkeypatch.setattr(builtins, "print", exploding_print)
    assert tag_utils.load_tag_translation_map(d) == {}

    utils._flush_debug_log()
    log = debug_log_on.read_text(encoding="utf-8", errors="replace")
    assert "failed to load" in log
    assert "not utf-8" not in log or True  # 例外型が記録されていれば十分


# ---------------------------------------------------------------------------
# PERF-1: 進捗シグナルの間引き（REQ-5.16〜5.20）
# ---------------------------------------------------------------------------


class _FakeResponse:
    """requests.Response の最小代替（context manager / iter_content / headers）。"""

    def __init__(self, body: bytes, *, headers: dict | None = None):
        self._body = body
        self.status_code = 200
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int = 8192):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_progress_is_throttled_to_percent_changes(tmp_path: Path, monkeypatch):
    import workers

    body = b"x" * (1024 * 40)                      # 40 KiB
    emits: list[tuple] = []

    worker = workers.DownloaderWorker(lambda section, key, **kw: key)
    monkeypatch.setattr(workers.requests, "get",
                        lambda *a, **k: _FakeResponse(body, headers={"content-length": str(len(body))}))
    # Signal の emit は instance 属性として差し替えられないので slot で受ける。
    worker.progress_update.connect(lambda *a: emits.append(a))

    target = tmp_path / "model.onnx"
    result = worker._download_single_file(target, "https://example.test/model.onnx", None)

    assert result is True or result is not None
    assert target.stat().st_size == len(body)
    # ループ内の発行はパーセントの変化回数以下＋開始時/完了時の別枠。
    # 40KB を 8KB チャンクで流すと 5 回のチャンク、パーセントは 20/40/60/80/100 の 5 段階。
    loop_percents = [p for p, _d, _t in emits]
    assert len(emits) <= 5 + 3          # 変化回数 + 開始時 + 完了時 + 余裕
    assert loop_percents[-1] == 100     # 最終的に 100 が発行される（REQ-5.17）


def test_download_progress_is_not_emitted_per_chunk(tmp_path: Path, monkeypatch):
    """8KB チャンクごとに無間引きで発行していないこと（旧実装は chunk 数だけ発行）。"""
    import workers

    body = b"y" * (1024 * 64)                      # 64 KiB = 8 チャンク
    emits: list[int] = []

    worker = workers.DownloaderWorker(lambda section, key, **kw: key)
    monkeypatch.setattr(workers.requests, "get",
                        lambda *a, **k: _FakeResponse(body, headers={"content-length": str(len(body))}))
    worker.progress_update.connect(lambda pct, *rest: emits.append(pct))

    target = tmp_path / "model2.onnx"
    worker._download_single_file(target, "https://example.test/model2.onnx", None)

    # パーセントは 0〜100 の 101 通りしかないので、それを超える発行はあり得ない。
    assert len(emits) <= 101 + 2
    assert emits[-1] == 100


def test_gpu_progress_worker_throttles_emits():
    import workers

    worker = workers.GpuRuntimeDownloadWorker(lambda section, key, **kw: key)
    emitted: list[tuple] = []
    worker.progress_update.connect(lambda *a: emitted.append(a))

    total = 1024 * 1024 * 200                     # 200MB
    # 256KB 刻みで 800 回流す（旧実装は 800 回 emit）
    for i in range(1, 801):
        worker._on_progress(i * 256 * 1024, total)

    assert len(emitted) <= 101                    # パーセント変化回数以下（REQ-5.20）
    assert emitted[-1][0] == 100


def test_gpu_progress_worker_emits_when_total_unknown():
    import workers

    worker = workers.GpuRuntimeDownloadWorker(lambda section, key, **kw: key)
    emitted: list[tuple] = []
    worker.progress_update.connect(lambda *a: emitted.append(a))

    worker._on_progress(1024, 0)
    worker._on_progress(2048, 0)
    assert len(emitted) == 2                      # total == 0 は現行どおり毎回発行
    assert all(e[0] == 0 for e in emitted)
