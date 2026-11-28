"""PR #38（dev → main）の AI レビュー指摘への回帰テスト。

指摘は 10 件。うち 9 件を修正し、1 件（delete_secret の三値化）は仕様判断が必要なため
`delete_secret` を「削除も確認もできなかった場合は False」に限定する形で吸収した。
"""
from __future__ import annotations

import configparser
import os
import stat
import sys
import time
from pathlib import Path

import pytest

import text_io
import utils
import vlm_scrub
from tagging_core import TagCategory


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


# ---------------------------------------------------------------------------
# 1. config.ini: 未知キーの値を補間せず保持する
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["100%", "%(other)s", "50%%done", "a % b"])
def test_unknown_config_values_with_percent_round_trip(tmp_path: Path, monkeypatch, value):
    import app_settings

    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", config_path)
    config_path.write_text(
        f"[Custom]\nprogress = {value}\n",
        encoding="utf-8",
    )
    settings = app_settings.load_settings(app_settings.get_default_config())

    # 例外を漏らさず True を返す（旧実装は items() の補間で例外になっていた）
    assert app_settings.save_config(settings) is True

    parsed = configparser.RawConfigParser()
    parsed.read(config_path, encoding="utf-8")
    assert parsed.get("Custom", "progress") == value


def test_save_config_returns_false_when_merge_fails(monkeypatch, tmp_path: Path):
    """取り込み中の予期しない失敗でも例外を漏らさず False を返す。"""
    import app_settings

    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", config_path)
    settings = app_settings.load_settings(app_settings.get_default_config())

    def boom(_config):
        raise RuntimeError("simulated merge failure")

    monkeypatch.setattr(app_settings, "_merge_unknown_config_entries", boom)
    assert app_settings.save_config(settings) is False


# ---------------------------------------------------------------------------
# 2. preprocessor_config.json の UTF-8 不正で Florence 初期化を落とさない
# ---------------------------------------------------------------------------


def test_preprocessor_config_with_broken_utf8_falls_back_to_bilinear(tmp_path: Path):
    import caption_core

    (tmp_path / "preprocessor_config.json").write_bytes(b'{"resample": "\xff\xfe"}')
    assert caption_core._read_preprocessor_resampling(tmp_path) == "bilinear"


def test_preprocessor_config_with_valid_resample_is_used(tmp_path: Path):
    import caption_core

    (tmp_path / "preprocessor_config.json").write_text('{"resample": 3}', encoding="utf-8")
    assert caption_core._read_preprocessor_resampling(tmp_path) == "bicubic"


# ---------------------------------------------------------------------------
# 3. 置換保存で既存ファイルのモードを広げない
# ---------------------------------------------------------------------------


def test_atomic_write_preserves_existing_mode_and_restricts_temp(tmp_path: Path, monkeypatch):
    chmod_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(text_io, "_existing_mode", lambda _p: 0o600)
    monkeypatch.setattr(text_io, "_apply_mode", lambda p, m: chmod_calls.append((Path(p).name, m)))

    target = tmp_path / "secret.txt"
    target.write_text("old", encoding="utf-8")
    atomic = text_io.atomic_write_text
    atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    # 書き込み前は 0600（本人だけ）、置換前に元のモードへ戻す
    modes = [mode for _name, mode in chmod_calls]
    assert modes == [0o600, 0o600]
    assert all(name.endswith(".vlmtmp") for name, _mode in chmod_calls)


def test_atomic_write_does_not_chmod_new_files(tmp_path: Path, monkeypatch):
    chmod_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(text_io, "_apply_mode", lambda p, m: chmod_calls.append((str(p), m)))

    target = tmp_path / "fresh.txt"
    text_io.atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert chmod_calls == []      # 新規作成は従来の open() と同じモードのまま


def test_existing_mode_helper_reads_mode(tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    os.chmod(target, 0o640)
    if sys.platform != "win32":
        assert text_io._existing_mode(str(target)) == 0o640
    assert text_io._existing_mode(str(tmp_path / "missing.txt")) is None


# ---------------------------------------------------------------------------
# 4. 区切り文字だけの CSV レコードを落とさない（スコア対応のズレ防止）
# ---------------------------------------------------------------------------


def test_is_blank_csv_row_only_matches_truly_empty_lines():
    import tagging_core as T

    assert T._is_blank_csv_row([]) is True
    assert T._is_blank_csv_row(["", "", ""]) is False       # `,,`
    assert T._is_blank_csv_row(["", "", "", "", "", ""]) is False  # `,,,,,`
    assert T._is_blank_csv_row(["   "]) is False


def test_simple3col_loader_keeps_alignment_across_delimiter_only_record(tmp_path: Path):
    import tagging_core as T

    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "id,tag,cat\n"
        "0_alpha,alpha,general\n"
        ",,\n"
        "2_beta,beta,general\n",
        encoding="utf-8",
    )
    tags = T._load_selected_tags_simple3col(csv_path)
    assert [tag.name for tag in tags] == ["alpha", "", "beta"]     # 行を落とさない
    assert tags[2].category == int(TagCategory.GENERAL)


def test_pixai6col_loader_keeps_delimiter_only_record(tmp_path: Path):
    import tagging_core as T

    csv_path = tmp_path / "selected_tags.csv"
    csv_path.write_text(
        "id,tag_id,name,category,count,ips\n"
        "1,1,alpha,0,10,\n"
        ",,,,,\n"
        "3,3,beta,0,20,\n",
        encoding="utf-8",
    )
    tags = T._load_selected_tags_pixai6col(csv_path)
    assert [tag.name for tag in tags] == ["alpha", "", "beta"]


# ---------------------------------------------------------------------------
# 5. 欠番診断で巨大な集合を作らない
# ---------------------------------------------------------------------------


def test_missing_index_diagnostics_do_not_allocate_by_index_value(tmp_path: Path):
    import tagging_core as T

    mapping = {"0": {"name": "a"}, "1000000000": {"name": "b"}}
    started = time.monotonic()
    with pytest.raises(ValueError) as excinfo:
        T._validated_numeric_keys(mapping, context="test", tags_path=tmp_path / "x.json")
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"took {elapsed:.2f}s (index-value-proportional work)"
    assert "missing indices" in str(excinfo.value)


def test_first_missing_indices_is_bounded():
    import tagging_core as T

    assert T._first_missing_indices([0, 1, 4]) == [2, 3]
    assert T._first_missing_indices([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]) == [1, 3, 5, 7, 9]
    assert T._first_missing_indices([0, 1, 2]) == []


# ---------------------------------------------------------------------------
# 6. 数値のプロバイダーコードを分類する
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code,expected", [
    ("429", "rate_limited"),
    ("401", "auth_error"),
    ("403", "auth_error"),
    ("404", "model_unsupported"),
    ("500", "server_error"),
    ("503", "server_error"),
])
def test_numeric_provider_codes_are_classified(code, expected):
    from vlm_errors import reason_from_provider_code

    reason = reason_from_provider_code(code)
    assert reason is not None and reason.value == expected


def test_unknown_numeric_code_still_falls_back_to_message():
    from vlm_errors import reason_from_provider_code

    assert reason_from_provider_code("418", "") is None
    assert reason_from_provider_code("418", "rate limit exceeded") is not None


# ---------------------------------------------------------------------------
# 7. Basic 認証の資格情報も伏せる
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,leaked", [
    ("Authorization: Basic dXNlcjpzZWNyZXQ=", "dXNlcjpzZWNyZXQ"),
    ("authorization=Basic dXNlcjpzZWNyZXQ=", "dXNlcjpzZWNyZXQ"),
    ("Basic dXNlcjpzZWNyZXQ=", "dXNlcjpzZWNyZXQ"),
    ("x-goog-api-key: AIzaSyTEST", "AIzaSyTEST"),
    ("Bearer sk-test-123", "sk-test-123"),
])
def test_scrub_hides_full_credentials(raw, leaked):
    scrubbed = vlm_scrub.scrub_text(raw)
    assert leaked not in scrubbed
    assert "<redacted>" in scrubbed


def test_scrub_keeps_paths_and_status_text():
    assert vlm_scrub.scrub_text("GET /v1/models -> 404") == "GET /v1/models -> 404"


# ---------------------------------------------------------------------------
# 8. 削除も確認もできなかった場合は成功扱いしない
# ---------------------------------------------------------------------------


class _LockedKeyring:
    """KeyringLocked 相当: 削除も読み直しも失敗する。"""

    def get_password(self, service, ref):
        raise RuntimeError("keyring is locked")

    def delete_password(self, service, ref):
        raise RuntimeError("keyring is locked")

    def set_password(self, service, ref, value):
        raise RuntimeError("keyring is locked")


class _MissingKeyring:
    """未登録エントリで例外を返す backend（削除は「無い」、読み直しも例外）。"""

    def get_password(self, service, ref):
        raise KeyError("no such entry")

    def delete_password(self, service, ref):
        raise KeyError("no such entry")

    def set_password(self, service, ref, value):
        pass


def test_delete_secret_with_locked_backend_reports_not_removed(monkeypatch):
    import vlm_secrets

    monkeypatch.setattr(vlm_secrets, "keyring", _LockedKeyring())
    monkeypatch.setattr(vlm_secrets, "_session_store", {})
    assert vlm_secrets.delete_secret("vlm/gemini/api_key") is False


def test_delete_secret_with_missing_entry_backend_reports_removed(monkeypatch):
    """未登録＝実際に消えているケースは True（REQ-5.5 の本来の修正）。"""
    import vlm_secrets

    monkeypatch.setattr(vlm_secrets, "keyring", _MissingKeyring())
    monkeypatch.setattr(vlm_secrets, "_session_store", {})
    assert vlm_secrets.delete_secret("vlm/gemini/api_key") is True


# ---------------------------------------------------------------------------
# 9. 診断の軽量プローブもリダイレクト時に認証ヘッダーを落とす
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    def __init__(self, body: bytes):
        self._body = body
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        self.encoding = "utf-8"
        self.text = body.decode()
        self.request = None

    def iter_content(self, chunk_size=512):
        yield self._body

    def json(self):
        import json

        return json.loads(self.text)

    def close(self):
        return None


class _RecordingSession:
    sensitive_headers: frozenset[str] = frozenset()
    used = 0

    def __init__(self, *args, **kwargs):
        self.requests: list[dict] = []
        type(self).used += 1

    def request(self, method, url, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        return _FakeHttpResponse(b'{"data": []}')


def test_execute_http_uses_guarded_session_by_default(monkeypatch):
    import vlm_transport
    from vlm_protocols import VlmHttpRequest
    from vlm_transport import execute_http

    created: list[_RecordingSession] = []

    def factory(*args, **kwargs):
        session = _RecordingSession(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(vlm_transport, "GuardedSession", factory)
    req = VlmHttpRequest(method="GET", url="https://example.test/v1/models",
                         headers={"x-goog-api-key": "AIza"}, params={}, json_body={})
    raw = execute_http(req, connect_timeout=1.0, read_timeout=1.0)
    assert created, "session を渡さない経路でも GuardedSession を使う"
    assert "x-goog-api-key" in created[0].sensitive_headers
    assert "authorization" in created[0].sensitive_headers
    assert raw.status == 200


def test_guarded_session_drops_custom_auth_header_on_host_change():
    from vlm_transport import GuardedSession

    session = GuardedSession()
    session.sensitive_headers = frozenset({"authorization", "x-goog-api-key"})

    import requests

    prepared = requests.Request("GET", "https://other.test/v1/models",
                                headers={"x-goog-api-key": "AIza"}).prepare()
    previous = requests.Request("GET", "https://example.test/v1/models").prepare()

    class _Resp:
        request = previous

    session.rebuild_auth(prepared, _Resp())
    assert "x-goog-api-key" not in prepared.headers


# ---------------------------------------------------------------------------
# 10. 全体締切をチャンク受信の途中でも守る
# ---------------------------------------------------------------------------


class _SlowResponse:
    """512 バイト貯まるまで返らない＝ゆっくり送るサーバーの模擬。"""

    def __init__(self, chunk_delay: float, chunks: int):
        self._chunk_delay = chunk_delay
        self._chunks = chunks
        self.closed = False
        self.status_code = 200
        self.headers = {"Content-Type": "text/plain"}
        self.encoding = "utf-8"
        self.text = ""
        self.request = None

    def iter_content(self, chunk_size=512):
        for _ in range(self._chunks):
            if self.closed:
                return
            time.sleep(self._chunk_delay)
            yield b"x" * 512

    def json(self):
        return None

    def close(self):
        self.closed = True


class _SlowSession:
    sensitive_headers: frozenset[str] = frozenset()

    def __init__(self, response):
        self._response = response

    def request(self, *args, **kwargs):
        return self._response


def test_total_deadline_is_enforced_during_slow_receive(monkeypatch):
    import vlm_transport
    from vlm_errors import VlmAttemptError, VlmErrorReason
    from vlm_protocols import VlmHttpRequest
    from vlm_transport import execute_http

    response = _SlowResponse(chunk_delay=0.2, chunks=50)   # 10 秒かかる想定
    session = _SlowSession(response)
    req = VlmHttpRequest(method="POST", url="https://example.test/v1/chat",
                         headers={}, params={}, json_body={"model": "m"})

    started = time.monotonic()
    result = execute_http(req, connect_timeout=1.0, read_timeout=5.0,
                          session=session, total_timeout_s=0.5)
    elapsed = time.monotonic() - started

    assert isinstance(result, VlmAttemptError)
    assert result.reason is VlmErrorReason.TIMEOUT
    assert elapsed < 2.0, f"deadline not enforced during receive (took {elapsed:.2f}s)"
    assert response.closed is True


def test_fast_receive_with_deadline_still_succeeds():
    from vlm_protocols import VlmHttpRequest
    from vlm_transport import execute_http

    session = _SlowSession(_FakeHttpResponse(b'{"ok": true}'))
    req = VlmHttpRequest(method="GET", url="https://example.test/v1/models",
                         headers={}, params={}, json_body={})
    raw = execute_http(req, connect_timeout=1.0, read_timeout=1.0,
                       session=session, total_timeout_s=5.0)
    assert raw.status == 200
    assert raw.json_body == {"ok": True}
