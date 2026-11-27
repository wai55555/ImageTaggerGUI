"""Phase G（261127 レビュー指摘の修正）の回帰テスト。

各テストは docs/261127_plan-review-findings-fix.md §13 の REQ-G* に対応する。
すべてオフラインで完結する（外部ネットワークへは出ない）。
"""
import json
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import requests

import vlm_diagnostics as D
import vlm_model_list as ML
import vlm_ratelimit as RL
import vlm_transport as T
from vlm_connections import (
    AuthSpec, ConnectionKind, RetryPolicy, VlmConnection, normalize_model_id,
)
from vlm_errors import (
    VlmAttemptError, VlmErrorClass, VlmErrorReason, reason_from_provider_code,
)
from vlm_image import PreparedImage
from vlm_profiles import GenerationProfile, build_system_prompt
from vlm_protocols import (
    AnthropicMessagesProtocol, GeminiGenerateContentProtocol,
    OpenAIChatCompletionsProtocol, OpenAIResponsesProtocol, VlmCallSpec,
)
from vlm_scrub import strip_url_userinfo
from vlm_transport import GuardedSession, RawHttpResponse, VlmExecutor, _enrich_parse_failure


def _conn(cid, protocol="openai_chat_completions", **kwargs):
    kwargs.setdefault("retry", RetryPolicy(retry_same_backoff_s=0.0))
    return VlmConnection(cid, cid, ConnectionKind.BUILTIN, protocol,
                         "https://x/v1", "m", provider_id=cid,
                         auth=AuthSpec(type="none"), **kwargs)


def _spec():
    return {
        "image": PreparedImage(b"\xff\xd8\xff", "image/jpeg"),
        "profile": GenerationProfile(),
        "system_prompt": "sys", "user_prompt": "u",
    }


def _call_spec(model_id="m"):
    return VlmCallSpec(
        model_id=model_id, system_prompt="sys", user_prompt="u",
        image=PreparedImage(b"\xff\xd8\xff", "image/jpeg"),
        profile=GenerationProfile(),
    )


def _patch_execute(monkey):
    old = T.execute_http
    T.execute_http = monkey
    return old


# ---------------------------------------------------------------- REQ-G1
def test_g1_chat_200_error_field_is_classified_like_non_200():
    """200 応答でも error フィールドがあれば非200 と同じ分類へ流す。"""
    body = {"error": {"code": "rate_limit_exceeded", "message": "slow down"}}
    parsed = OpenAIChatCompletionsProtocol().parse_response(200, body, json.dumps(body))
    assert parsed.error is not None
    assert parsed.error.reason is VlmErrorReason.RATE_LIMITED

    body = {"error": {"code": "model_not_found", "message": "no such model"}}
    parsed = OpenAIChatCompletionsProtocol().parse_response(200, body, json.dumps(body))
    assert parsed.error is not None
    assert parsed.error.reason is VlmErrorReason.MODEL_UNSUPPORTED
    # 恒久エラーは EXCLUDE（非200 の 404 と同じ）。
    assert parsed.error.classify() is VlmErrorClass.EXCLUDE


def test_g1_responses_200_failed_reads_error_code():
    """Responses API の 200 + status="failed" で実エラーコードを捨てない。"""
    body = {"status": "failed",
            "error": {"code": "rate_limit_exceeded", "message": "rate limited"}}
    parsed = OpenAIResponsesProtocol().parse_response(200, body, json.dumps(body))
    assert parsed.error is not None and parsed.error.reason is VlmErrorReason.RATE_LIMITED

    body = {"status": "failed", "error": {"code": "model_not_found", "message": "gone"}}
    parsed = OpenAIResponsesProtocol().parse_response(200, body, json.dumps(body))
    assert parsed.error is not None and parsed.error.reason is VlmErrorReason.MODEL_UNSUPPORTED


def test_g1_200_rate_limit_registers_cooldown_and_keeps_retry_semantics():
    """200 本文の rate limit でもクールダウン登録され、次画像で外れる。"""
    calls = []

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True, **kwargs):
        calls.append(req.url)
        if len(calls) == 1:
            body = {"error": {"code": "rate_limit_exceeded", "message": "slow down"}}
            return RawHttpResponse(200, {"Retry-After": "30"}, body, json.dumps(body))
        return RawHttpResponse(200, {}, {"choices": [{"message": {"content": "ok"},
                                                     "finish_reason": "stop"}]}, "")

    old = _patch_execute(responder)
    try:
        ex = VlmExecutor({"a": _conn("a"), "b": _conn("b")}, lambda ref: None)
        result = ex.caption_one(_spec(), ["a", "b"])
        assert result.ok and result.connection_id == "b"
        state = ex.runtime("a").rate_limit
        assert state is not None and state.in_cooldown()
        assert ex.live_candidates(["a", "b"]) == ["b"]
    finally:
        T.execute_http = old


def test_g1_provider_code_mapping_covers_known_codes():
    assert reason_from_provider_code("rate_limit_exceeded") is VlmErrorReason.RATE_LIMITED
    assert reason_from_provider_code("insufficient_quota") is VlmErrorReason.RATE_LIMITED
    assert reason_from_provider_code("model_not_found") is VlmErrorReason.MODEL_UNSUPPORTED
    assert reason_from_provider_code("invalid_api_key") is VlmErrorReason.AUTH_ERROR
    assert reason_from_provider_code("content_filter") is VlmErrorReason.CONTENT_POLICY
    assert reason_from_provider_code("weird_code", "plain message") is None


# ---------------------------------------------------------------- REQ-G2
def test_g2_anthropic_refusal_is_content_policy_and_fails_over():
    body = {"stop_reason": "refusal", "content": []}
    parsed = AnthropicMessagesProtocol().parse_response(200, body, json.dumps(body))
    assert parsed.error is not None
    assert parsed.error.reason is VlmErrorReason.CONTENT_POLICY
    # 他形式（content_filter / SAFETY）と同じく同一接続リトライをしない。
    assert parsed.error.classify() is VlmErrorClass.FAILOVER


# ---------------------------------------------------------------- REQ-G3
def test_g3_gemini_model_id_prefix_and_escaping():
    proto = GeminiGenerateContentProtocol()
    req = proto.build_request("https://generativelanguage.googleapis.com/v1beta", "k",
                              _call_spec("models/gemini-2.0-flash"))
    assert "/models/gemini-2.0-flash:generateContent" in req.url
    assert "models/models/" not in req.url

    req = proto.build_request("https://generativelanguage.googleapis.com/v1beta", "k",
                              _call_spec("a?b#c/d"))
    assert req.url.endswith(":generateContent")
    assert "a%3Fb%23c%2Fd" in req.url
    assert "?" not in req.url.split(":generateContent")[0].split("/v1beta")[-1]


def test_g3_normalization_is_shared_with_save_paths():
    assert normalize_model_id("models/gemini-2.0-flash") == "gemini-2.0-flash"
    assert normalize_model_id("  models/models/x ") == "x"
    assert normalize_model_id("plain") == "plain"
    # 保存経路（from_mapping）も同じ正規化を通る。
    conn = VlmConnection.from_mapping({
        "connection_id": "c", "base_url": "https://x", "model_id": "models/gemini-2.0-flash",
    })
    assert conn.model_id == "gemini-2.0-flash"


# ---------------------------------------------------------------- REQ-G4
def test_g4_naive_http_date_is_interpreted_as_utc():
    """タイムゾーン表記の無い HTTP 日付は UTC。JST でも正のクールダウンになる。"""
    reset_utc = datetime(2015, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
    naive = format_datetime(reset_utc.replace(tzinfo=None))  # "+0000" を付けない形
    naive = naive.rsplit(" ", 1)[0] if naive.endswith("0000") else naive
    now_wall = reset_utc.timestamp() - 60
    st = RL.RateLimitState("c")
    RL.update_from_429(st, {"Retry-After": naive}, now_wall=now_wall, now_mono=1000.0)
    remaining = st.remaining_s(1000.0)
    assert 55 <= remaining <= 65, remaining


def test_g4_gmt_date_is_unchanged():
    reset_utc = datetime(2015, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
    header = format_datetime(reset_utc)  # "+0000" 付き（GMT）
    now_wall = reset_utc.timestamp() - 30
    st = RL.RateLimitState("c")
    RL.update_from_429(st, {"Retry-After": header}, now_wall=now_wall, now_mono=5.0)
    assert 25 <= st.remaining_s(5.0) <= 35


# ---------------------------------------------------------------- REQ-G5
def test_g5_cooldown_uses_monotonic_clock_and_survives_wall_rollback():
    st = RL.RateLimitState("c")
    RL.update_from_429(st, {"Retry-After": "60"}, now_wall=1_700_000_000.0, now_mono=500.0)
    # 壁時計を大きく巻き戻しても残り時間は単調時計だけで決まる。
    assert abs(st.remaining_s(500.0) - 60.0) < 0.001
    assert st.in_cooldown(500.0) and not st.in_cooldown(561.0)


def test_g5_header_epoch_reset_does_not_become_permanent():
    """絶対 epoch を登録しても、単調 deadline へ換算されるので期限で解除される。"""
    now_wall = 1_700_000_000.0
    now_mono = 12_345.0
    st = RL.RateLimitState("c")
    RL.update_from_429(st, {"x-ratelimit-reset": str(now_wall + 45)},
                       now_wall=now_wall, now_mono=now_mono)
    assert abs(st.remaining_s(now_mono) - 45) < 1
    # 表示用の絶対時刻は壁時計のまま残る。
    assert st.reset_at_utc is not None and abs(st.reset_at_utc - (now_wall + 45)) < 1
    assert not st.in_cooldown(now_mono + 46)


def test_g5_live_candidates_uses_the_same_clock():
    def responder(req, *, connect_timeout, read_timeout, verify_tls=True, **kwargs):
        return RawHttpResponse(429, {"Retry-After": "60"}, {}, "rate limited")

    old = _patch_execute(responder)
    try:
        ex = VlmExecutor({"a": _conn("a"), "b": _conn("b")}, lambda ref: None)
        ex.caption_one(_spec(), ["a"])
        assert ex.live_candidates(["a", "b"]) == ["b"]
    finally:
        T.execute_http = old


# ---------------------------------------------------------------- REQ-G6
class _SlowDripHandler(BaseHTTPRequestHandler):
    """少しずつ送り続ける遅滴ピア（総量締切が無いと終わらない）。"""

    chunk_delay_s = 0.2
    chunks = 20

    def do_GET(self):  # noqa: N802 - http.server の命名
        payload = b'{"data": [' + b'{"id": "m"},' * 50 + b'{"id": "m"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload) * self.chunks))
        self.end_headers()
        try:
            for _ in range(self.chunks):
                time.sleep(self.chunk_delay_s)
                self.wfile.write(payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):  # テスト出力を汚さない
        pass


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_g6_catalog_fetch_has_a_total_deadline():
    server, thread = _serve(_SlowDripHandler)
    try:
        port = server.server_address[1]
        conn = VlmConnection("slow", "slow", ConnectionKind.CUSTOM_LOCAL,
                             "openai_chat_completions", f"http://127.0.0.1:{port}/v1", "m",
                             auth=AuthSpec(type="none"))
        old = ML.CATALOG_TOTAL_DEADLINE_S
        ML.CATALOG_TOTAL_DEADLINE_S = 0.6
        started = time.monotonic()
        try:
            result = ML.fetch_model_catalog(conn, None, connect_timeout=1.0, read_timeout=5.0)
        finally:
            ML.CATALOG_TOTAL_DEADLINE_S = old
        elapsed = time.monotonic() - started
        assert isinstance(result, VlmAttemptError), result
        assert result.reason is VlmErrorReason.TIMEOUT
        # サーバーの総量（0.2s × 20 = 4s）より十分早く打ち切る。
        assert elapsed < 2.0, elapsed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_g6_generation_path_keeps_no_total_deadline():
    """生成経路（executor）は総量締切を渡さない（長時間化が正常であり得る）。"""
    seen = {}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True, **kwargs):
        seen.update(kwargs)
        return RawHttpResponse(200, {}, {"choices": [{"message": {"content": "x"},
                                                     "finish_reason": "stop"}]}, "")

    old = _patch_execute(responder)
    try:
        ex = VlmExecutor({"a": _conn("a")}, lambda ref: None)
        assert ex.caption_one(_spec(), ["a"]).ok
        assert "total_timeout_s" not in seen
    finally:
        T.execute_http = old


# ---------------------------------------------------------------- REQ-G7
def test_g7_session_reused_and_api_key_resolved_once():
    seen_sessions = []

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True, **kwargs):
        seen_sessions.append(kwargs.get("session"))
        # 1回目は 5xx（同一接続リトライ）、2回目は成功。
        if len(seen_sessions) == 1:
            return RawHttpResponse(500, {}, {"error": {"message": "boom"}}, "boom")
        return RawHttpResponse(200, {}, {"choices": [{"message": {"content": "ok"},
                                                     "finish_reason": "stop"}]}, "")

    resolver_calls = []
    conn = _conn("a")
    conn.auth = AuthSpec(type="header_key", secret_ref="ref-1", header_name="x-api-key")
    old = _patch_execute(responder)
    try:
        ex = VlmExecutor({"a": conn}, lambda ref: resolver_calls.append(ref) or "KEY")
        assert ex.caption_one(_spec(), ["a"]).ok
        assert len(seen_sessions) == 2
        assert seen_sessions[0] is not None
        assert seen_sessions[0] is seen_sessions[1]
        assert resolver_calls == ["ref-1"]
        # 2 画像目でも解決は増えない（試行のまとまりごとに 1 回）。
        assert ex.caption_one(_spec(), ["a"]).ok
        assert resolver_calls == ["ref-1"]
    finally:
        T.execute_http = old
        ex.close()


# ---------------------------------------------------------------- REQ-G8
def test_g8_retry_same_waits_before_retrying():
    sleeps = []
    calls = []

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return RawHttpResponse(500, {}, {"error": {"message": "boom"}}, "boom")
        return RawHttpResponse(200, {}, {"choices": [{"message": {"content": "ok"},
                                                     "finish_reason": "stop"}]}, "")

    conn = _conn("a", retry=RetryPolicy(retry_same_backoff_s=0.5))
    old = _patch_execute(responder)
    try:
        ex = VlmExecutor({"a": conn}, lambda ref: None, sleep=sleeps.append)
        assert ex.caption_one(_spec(), ["a"]).ok
        assert len(calls) == 2
        assert sleeps == [0.5]
    finally:
        T.execute_http = old


def test_g8_success_path_never_waits():
    sleeps = []

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True, **kwargs):
        return RawHttpResponse(200, {}, {"choices": [{"message": {"content": "ok"},
                                                     "finish_reason": "stop"}]}, "")

    old = _patch_execute(responder)
    try:
        ex = VlmExecutor({"a": _conn("a")}, lambda ref: None, sleep=sleeps.append)
        assert ex.caption_one(_spec(), ["a"]).ok
        assert sleeps == []
    finally:
        T.execute_http = old


# ---------------------------------------------------------------- REQ-G9
def test_g9_custom_connection_cannot_shadow_a_builtin_id(monkeypatch):
    import vlm_config

    builtin = _conn("openai")
    monkeypatch.setattr(vlm_config, "default_builtin_connections", lambda: [builtin])
    monkeypatch.setattr(vlm_config, "load_custom_connections", lambda: [
        {"connection_id": "openai", "display_name": "evil", "kind": "custom_external",
         "protocol": "openai_chat_completions", "base_url": "https://evil.example/v1",
         "model_id": "m"},
        {"connection_id": "c1", "display_name": "first", "kind": "custom_external",
         "protocol": "openai_chat_completions", "base_url": "https://one.example/v1",
         "model_id": "m"},
        {"connection_id": "c1", "display_name": "second", "kind": "custom_external",
         "protocol": "openai_chat_completions", "base_url": "https://two.example/v1",
         "model_id": "m"},
    ])
    class _Settings:
        cloudflare_account_id = ""
        anthropic_workspace_id = ""
        model_profile_id = "p1"

        @staticmethod
        def model_id_override_map():
            return {}

    result = vlm_config.build_connection_map(_Settings(), None)
    assert result["openai"].base_url == "https://x/v1"     # 内蔵が勝つ
    assert result["c1"].base_url == "https://one.example/v1"  # 後勝ちで潰さない


# ---------------------------------------------------------------- REQ-G10
def test_g10_error_messages_are_masked_before_reaching_logs():
    """ボディをエラーへ入れる前に、transport 例外側と同じマスクを通す。

    マスク対象は文脈が明確な資格情報（URL クエリ・userinfo・Bearer・認証ヘッダー）。
    任意部分文字列はパス情報まで壊すため置換しない。
    """
    leaky = "request to https://api.example/v1/chat?key=sk-SECRET123 failed"
    parsed = OpenAIChatCompletionsProtocol().parse_response(500, {"error": {}}, leaky)
    assert parsed.error is not None
    assert "sk-SECRET123" not in parsed.error.message

    body = {"error": {"code": "invalid_api_key",
                      "message": "Bearer sk-SECRET999 is invalid"}}
    parsed = OpenAIChatCompletionsProtocol().parse_response(200, body, json.dumps(body))
    assert parsed.error is not None
    assert "sk-SECRET999" not in parsed.error.message


# ---------------------------------------------------------------- REQ-G11
def test_g11_cross_host_redirect_drops_auth_headers():
    session = GuardedSession()
    session.sensitive_headers = frozenset({"authorization", "x-api-key"})

    def prepared(url):
        req = requests.Request("GET", url,
                               headers={"x-api-key": "SECRET", "Authorization": "Bearer K",
                                        "Accept": "application/json"}).prepare()
        return req

    class _Resp:
        def __init__(self, url):
            self.request = prepared(url)

    same_host = prepared("https://api.example/v2/models")
    session.rebuild_auth(same_host, _Resp("https://api.example/v1/models"))
    assert same_host.headers.get("x-api-key") == "SECRET"

    crossed = prepared("https://evil.example/v2/models")
    session.rebuild_auth(crossed, _Resp("https://api.example/v1/models"))
    assert "x-api-key" not in crossed.headers
    assert "Authorization" not in crossed.headers
    assert crossed.headers.get("Accept") == "application/json"
    session.close()


# ---------------------------------------------------------------- REQ-G12
def test_g12_base_url_userinfo_is_stripped():
    assert strip_url_userinfo("https://user:pa%40ss@host/v1") == "https://host/v1"
    conn = VlmConnection.from_mapping({
        "connection_id": "c", "kind": "custom_external", "protocol": "openai_chat_completions",
        "base_url": "https://user:secret@api.example/v1", "model_id": "m",
    })
    assert conn.base_url == "https://api.example/v1"
    req = D._model_list_request(conn, None)
    assert "user" not in req.url and "secret" not in req.url


# ---------------------------------------------------------------- REQ-G13
def test_g13_language_label_separator_normalization():
    profile = GenerationProfile(language="zh-TW")
    assert "Traditional Chinese" in build_system_prompt(profile)
    for raw in ("zh-tw", "zh_TW", "zh-tw"):
        assert "Traditional Chinese" in build_system_prompt(GenerationProfile(language=raw))
    assert "Chinese" in build_system_prompt(GenerationProfile(language="zh-CN"))
    assert "Japanese" in build_system_prompt(GenerationProfile(language="ja"))


# ---------------------------------------------------------------- REQ-G14
def test_g14_output_limit_advice_is_larger_than_the_current_value():
    parsed = T.VlmParseResult(error=VlmAttemptError(VlmErrorReason.EMPTY_RESPONSE, 200, "empty"))
    body = {"choices": [{"finish_reason": "length"}]}
    conn = _conn("a")
    enriched = _enrich_parse_failure(parsed, protocol=OpenAIChatCompletionsProtocol(),
                                     body=body, conn=conn, max_output_tokens=8192)
    assert enriched.error is not None
    assert enriched.error.reason is VlmErrorReason.OUTPUT_LIMIT
    assert enriched.error.message_args["suggested"] > 8192
    # 現在値が既定以下のときは従来どおり既定値を勧める。
    low = _enrich_parse_failure(parsed, protocol=OpenAIChatCompletionsProtocol(),
                                body=body, conn=conn, max_output_tokens=512)
    assert low.error is not None
    assert low.error.message_args["suggested"] == T._SUGGESTED_MAX_OUTPUT_TOKENS


# ---------------------------------------------------------------- REQ-G15
def test_g15_diagnostics_uses_the_same_endpoint_as_the_real_path():
    xai = VlmConnection("x", "x", ConnectionKind.BUILTIN, "openai_chat_completions",
                        "https://api.x.ai/v1", "grok", provider_id="xai",
                        auth=AuthSpec(type="none"))
    assert ML.model_list_endpoint(xai)[0] == "https://api.x.ai/v1/language-models"
    assert D._model_list_request(xai, None).url == "https://api.x.ai/v1/language-models"

    cf = VlmConnection("cf", "cf", ConnectionKind.BUILTIN, "openai_chat_completions",
                       "https://api.cloudflare.com/client/v4/accounts/a/ai/v1", "m",
                       provider_id="cloudflare", auth=AuthSpec(type="none"))
    assert ML.model_list_endpoint(cf)[0].endswith("/ai/models/search")
    assert D._model_list_request(cf, None).url.endswith("/ai/models/search")
    assert ML.is_cloudflare_connection(cf)
    assert ML.is_cloudflare_connection(
        VlmConnection("c", "c", ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions",
                      "https://api.cloudflare.com/client/v4/x", "m",
                      auth=AuthSpec(type="none")))
    assert not ML.is_cloudflare_connection(xai)


# ---------------------------------------------------------------- U-G1
def test_ug1_measure_catalog_fetch_and_pin_the_total_deadline():
    """U-G1: 通常のカタログ取得時間を実測し、締切が十分な倍数であることを固定する。

    実測値はテスト出力に残す（Phase G の記録・コミットメッセージへ転記する）。
    """
    class _CatalogHandler(BaseHTTPRequestHandler):
        payload = json.dumps({"data": [
            {"id": f"model-{i}", "input_modalities": ["text", "image"],
             "output_modalities": ["text"]} for i in range(100)]}).encode()

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

        def log_message(self, *args):
            pass

    server, thread = _serve(_CatalogHandler)
    try:
        port = server.server_address[1]
        conn = VlmConnection("local", "local", ConnectionKind.CUSTOM_LOCAL,
                             "openai_chat_completions", f"http://127.0.0.1:{port}/v1", "m",
                             auth=AuthSpec(type="none"))
        samples = []
        for _ in range(5):
            started = time.monotonic()
            entries = ML.fetch_model_catalog(conn, None, connect_timeout=2.0, read_timeout=5.0)
            samples.append(time.monotonic() - started)
            assert isinstance(entries, list) and len(entries) == 100
        measured = max(samples)
        policy_floor = RetryPolicy().read_timeout_s
        print(f"\nU-G1 measured catalog fetch (100 entries, loopback): "
              f"min={min(samples):.4f}s max={measured:.4f}s; "
              f"policy floor={policy_floor:.1f}s; selected deadline="
              f"{ML.CATALOG_TOTAL_DEADLINE_S:.1f}s")
        assert ML.CATALOG_TOTAL_DEADLINE_S >= policy_floor
        assert ML.CATALOG_TOTAL_DEADLINE_S >= measured * 10
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
