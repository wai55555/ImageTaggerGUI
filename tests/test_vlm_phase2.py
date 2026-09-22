"""VLM Phase 2 tests: executor retry/failover/exclude, diagnostics static, worker with mock.

Offline only. Run:  rtk pytest tests/test_vlm_phase2.py -q
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vlm_transport as T
from vlm_transport import RawHttpResponse, VlmExecutor
from vlm_errors import VlmAttemptError, VlmErrorReason
from vlm_connections import VlmConnection, ConnectionKind, AuthSpec
from vlm_image import PreparedImage
from vlm_profiles import GenerationProfile


def _conn(cid, protocol="openai_chat_completions"):
    return VlmConnection(cid, cid, ConnectionKind.BUILTIN, protocol,
                         "https://x/v1", "m", provider_id=cid, auth=AuthSpec(type="none"))


def _spec():
    return {
        "image": PreparedImage(b"\xff\xd8\xff", "image/jpeg"),
        "profile": GenerationProfile(),
        "system_prompt": "sys", "user_prompt": "u",
    }


def _ok_body(text="a caption"):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


class _Responder:
    """execute_http の差し替え。connection_id ごとに応答スクリプトを返す。"""
    def __init__(self, script):
        self.script = script       # {url_substr: [resp, resp, ...]} ; resp は RawHttpResponse | VlmAttemptError
        self.calls = []

    def __call__(self, req, *, connect_timeout, read_timeout, verify_tls=True):
        self.calls.append(req.url)
        for key, seq in self.script.items():
            if key in req.url:
                return seq.pop(0) if len(seq) > 1 else seq[0]
        return RawHttpResponse(200, {}, _ok_body(), "")


def _patch(monkey):
    old = T.execute_http
    T.execute_http = monkey
    return old


def test_retry_policy_default_read_timeout_is_60s():
    """Pinned so nobody silently raises this back toward the old 120s default -
    with retry_same_max=1 and multiple candidate connections it multiplies into
    several minutes of a silent, apparently-frozen UI per image."""
    from vlm_connections import RetryPolicy
    assert RetryPolicy().read_timeout_s == 60.0


def test_adaptive_read_timeout_scales_with_max_output_tokens():
    """A real 11-image GUI batch with the default maximum_detail profile
    (max_output_tokens=3072) showed legitimate Gemini generations taking
    40-70s, some tripping the flat 60s timeout and wasting ~2min/image on a
    pointless retry+failover before an also-rate-limited fallback failed
    instantly. The timeout must scale up for large output budgets, but stay
    capped so a genuinely dead connection (the original NVIDIA problem) still
    fails in bounded time."""
    from vlm_transport import adaptive_read_timeout
    assert adaptive_read_timeout(60.0, 3072) == 90.0          # default profile: capped at 1.5x
    assert adaptive_read_timeout(60.0, 2048) == 60.0           # baseline: no change
    assert adaptive_read_timeout(60.0, 512) == 60.0            # small budget: never shrinks below base
    assert adaptive_read_timeout(60.0, 8192) == 90.0           # very large budget: still capped at 1.5x
    assert adaptive_read_timeout(60.0, 0) == 60.0              # no tokens configured: unchanged


def test_executor_passes_adaptive_timeout_to_execute_http():
    seen = {}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        seen["read_timeout"] = read_timeout
        return RawHttpResponse(200, {}, _ok_body(), "")

    old = _patch(responder)
    try:
        ex = VlmExecutor({"a": _conn("a")}, lambda ref: None)
        spec = _spec()
        spec["profile"] = GenerationProfile(max_output_tokens=3072)
        res = ex.caption_one(spec, ["a"])
        assert res.ok
        assert seen["read_timeout"] == 90.0
    finally:
        T.execute_http = old


def test_success_first_connection():
    conns = {"a": _conn("a"), "b": _conn("b")}
    r = _Responder({"x/v1": [RawHttpResponse(200, {}, _ok_body("first"), "")]})
    old = _patch(r)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.text == "first" and res.connection_id == "a"
        assert len(r.calls) == 1
    finally:
        T.execute_http = old
    print("  success on first connection: OK")


def test_output_limit_explains_generation_setting_and_skips_same_retry():
    conn = _conn("a")
    body = {"choices": [{
        "message": {"content": "", "reasoning_content": "long internal reasoning"},
        "finish_reason": "length",
    }], "usage": {"completion_tokens": 1024}}
    responder = _Responder({"x/v1": [RawHttpResponse(200, {}, body, "")]})
    old = _patch(responder)
    try:
        ex = VlmExecutor({"a": conn}, lambda ref: None)
        spec = _spec()
        spec["profile"] = GenerationProfile(max_output_tokens=1024)
        res = ex.caption_one(spec, ["a"])
        assert not res.ok
        assert res.error.reason is VlmErrorReason.OUTPUT_LIMIT
        assert "max_output_tokens=1024" in res.error.message
        assert "3072" in res.error.message
        assert len(res.attempts) == 1, "output-limit failure must not retry unchanged settings"
    finally:
        T.execute_http = old
    print("  output-limit failure explains max tokens and avoids duplicate retry: OK")


def test_empty_response_explains_custom_response_path():
    conn = _conn("a")
    conn.text_path = "wrong.path"
    responder = _Responder({"x/v1": [RawHttpResponse(200, {}, _ok_body("caption"), "")]})
    old = _patch(responder)
    try:
        ex = VlmExecutor({"a": conn}, lambda ref: None)
        res = ex.caption_one(_spec(), ["a"])
        assert not res.ok
        assert res.error.reason is VlmErrorReason.EMPTY_RESPONSE
        assert "wrong.path" in res.error.message
        assert "response extraction path" in res.error.message
    finally:
        T.execute_http = old
    print("  empty response explains custom response path: OK")


def test_429_failover():
    conns = {"a": _conn("a"), "b": _conn("b")}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        # both connections share a base_url; distinguish by call order.
        responder.n += 1
        if responder.n == 1:
            return RawHttpResponse(429, {"Retry-After": "30"}, {"error": {"code": "rate_limit"}}, "rate limited")
        return RawHttpResponse(200, {}, _ok_body("from b"), "")
    responder.n = 0
    old = _patch(responder)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.text == "from b" and res.connection_id == "b"
        assert ex.runtime("a").rate_limit is not None and ex.runtime("a").rate_limit.in_cooldown()
        # cooldown carries to next image
        live = ex.live_candidates(["a", "b"])
        assert live == ["b"]
    finally:
        T.execute_http = old
    print("  429 -> failover + cooldown carry-over: OK")


def test_timeout_retry_then_failover():
    conns = {"a": _conn("a"), "b": _conn("b")}
    seq = {"a": [VlmAttemptError(VlmErrorReason.TIMEOUT, None, "t1"),
                 VlmAttemptError(VlmErrorReason.TIMEOUT, None, "t2")]}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        # first two calls (connection a) time out, then b succeeds
        if responder.n < 2:
            responder.n += 1
            return VlmAttemptError(VlmErrorReason.TIMEOUT, None, f"timeout {responder.n}")
        return RawHttpResponse(200, {}, _ok_body("b ok"), "")
    responder.n = 0
    old = _patch(responder)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.connection_id == "b", res.connection_id
        assert responder.n == 2  # one retry_same on 'a', then failover
        reasons = [a.error_reason for a in res.attempts]
        assert reasons == ["timeout", "timeout"]
    finally:
        T.execute_http = old
    print("  timeout -> retry_same once -> failover: OK")


def test_chronically_timing_out_connection_gets_excluded_after_two_images():
    """A connection that times out on EVERY attempt across two whole images
    (2 x (initial + retry_same_max retries)) must be excluded for the rest of
    the session - but not after just one bad image, since a real connection
    (Gemini) was observed timing out on one image and then succeeding
    normally on the next (2026-09 VLM debugging, real 11-image batch). A
    too-eager 1-image threshold caused that still-working connection to be
    excluded, which combined with another connection's rate-limit cooldown
    left zero live candidates and silently abandoned 7 of 11 images."""
    conns = {"a": _conn("a"), "b": _conn("b")}
    # Both connections share a base_url in this test helper, so the responder
    # can't tell "a" from "b" by request content - script by call index
    # instead, matching the exact expected sequence: image1 tries a twice
    # (timeout, timeout) then falls over to b (success); image2 repeats that
    # on a (now excluded) then succeeds on b; image3 only ever reaches b.
    script = [
        VlmAttemptError(VlmErrorReason.TIMEOUT, None, "timeout 1"),   # image1: a attempt 1
        VlmAttemptError(VlmErrorReason.TIMEOUT, None, "timeout 2"),   # image1: a attempt 2 (retry)
        RawHttpResponse(200, {}, _ok_body("b ok"), ""),                # image1: b
        VlmAttemptError(VlmErrorReason.TIMEOUT, None, "timeout 3"),   # image2: a attempt 1
        VlmAttemptError(VlmErrorReason.TIMEOUT, None, "timeout 4"),   # image2: a attempt 2 (retry)
        RawHttpResponse(200, {}, _ok_body("b ok"), ""),                # image2: b
        RawHttpResponse(200, {}, _ok_body("b ok"), ""),                # image3: b only (a excluded)
    ]
    calls = {"n": 0}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        result = script[calls["n"]]
        calls["n"] += 1
        return result

    old = _patch(responder)
    try:
        ex = VlmExecutor(conns, lambda ref: None)

        # Image 1: "a" times out on both attempts, but that is only one
        # image's worth (consecutive_timeouts=2 < threshold 4) - failover to
        # "b", and "a" must remain a live candidate for the next image.
        res1 = ex.caption_one(_spec(), ["a", "b"])
        assert res1.ok and res1.connection_id == "b"
        assert calls["n"] == 3
        assert not ex.runtime("a").is_excluded, \
            "a must NOT be excluded after just one bad image - it may just be having a rough moment"
        assert ex.live_candidates(["a", "b"]) == ["a", "b"]

        # Image 2: "a" times out on both attempts again - now two whole
        # images' worth of all-timeout attempts (consecutive_timeouts=4) -
        # this time it gets excluded.
        res2 = ex.caption_one(_spec(), ex.live_candidates(["a", "b"]))
        assert res2.ok and res2.connection_id == "b"
        assert calls["n"] == 6
        assert ex.runtime("a").is_excluded, "a must be excluded after two whole images of all-timeout attempts"
        assert ex.runtime("a").excluded_reason == "timeout"

        # Image 3: "a" must no longer even be attempted.
        live = ex.live_candidates(["a", "b"])
        assert live == ["b"]
        res3 = ex.caption_one(_spec(), live)
        assert res3.ok and res3.connection_id == "b"
        assert calls["n"] == 7, "excluded connection must not be retried on later images"
    finally:
        T.execute_http = old
    print("  chronic timeout across 2 images -> excluded, not retried later; 1 bad image alone is not enough: OK")


def test_on_attempt_start_fires_before_each_http_call():
    """`on_attempt_start` must fire once per HTTP attempt, in order, with the
    connection object and a 1-based same-connection attempt counter - this is
    what lets the UI show "requesting X..." while a request (up to
    read_timeout_s, default 60s) is still in flight instead of going silent."""
    conns = {"a": _conn("a"), "b": _conn("b")}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        if responder.n < 2:
            responder.n += 1
            return VlmAttemptError(VlmErrorReason.TIMEOUT, None, f"timeout {responder.n}")
        return RawHttpResponse(200, {}, _ok_body("b ok"), "")
    responder.n = 0
    old = _patch(responder)
    calls = []
    try:
        ex = VlmExecutor(conns, lambda ref: None,
                         on_attempt_start=lambda conn, attempt: calls.append((conn.connection_id, attempt)))
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.connection_id == "b"
        assert calls == [("a", 1), ("a", 2), ("b", 1)]
    finally:
        T.execute_http = old
    print("  on_attempt_start fires per attempt with connection + attempt number: OK")


def test_on_attempt_start_exception_does_not_break_the_request():
    """A broken callback (UI-side bug) must not prevent the actual HTTP attempt
    from happening - it is a notification hook, not part of the request logic."""
    conn = _conn("a")
    responder = _Responder({"x/v1": [RawHttpResponse(200, {}, _ok_body("still works"), "")]})
    old = _patch(responder)
    try:
        ex = VlmExecutor({"a": conn}, lambda ref: None,
                         on_attempt_start=lambda conn, attempt: (_ for _ in ()).throw(RuntimeError("boom")))
        res = ex.caption_one(_spec(), ["a"])
        assert res.ok and res.text == "still works"
    finally:
        T.execute_http = old
    print("  on_attempt_start callback failure does not break the request: OK")


def test_auth_error_excludes_connection():
    conns = {"a": _conn("a"), "b": _conn("b")}

    def responder(req, *, connect_timeout, read_timeout, verify_tls=True):
        if responder.first:
            responder.first = False
            return RawHttpResponse(401, {}, {"error": {"message": "bad key"}}, "unauthorized")
        return RawHttpResponse(200, {}, _ok_body("b ok"), "")
    responder.first = True
    old = _patch(responder)
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.ok and res.connection_id == "b"
        assert ex.runtime("a").is_excluded and ex.runtime("a").excluded_reason == "auth_error"
        # excluded stays excluded for the next image
        res2 = ex.caption_one(_spec(), ["a", "b"])
        assert res2.connection_id == "b"
    finally:
        T.execute_http = old
    print("  auth error -> exclude connection (persists): OK")


def test_all_fail_returns_error():
    conns = {"a": _conn("a"), "b": _conn("b")}
    old = _patch(lambda req, **kw: RawHttpResponse(503, {}, {}, "server error"))
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert not res.ok and res.error is not None
        assert res.error.reason in (VlmErrorReason.SERVER_ERROR, VlmErrorReason.BAD_RESPONSE, VlmErrorReason.UNKNOWN)
    finally:
        T.execute_http = old
    print("  all candidates fail -> ImageResult.error set: OK")


def test_transport_applies_header_and_query_auth():
    from vlm_connections import AuthSpec
    seen = {}

    def responder(req, **kw):
        seen["headers"] = dict(req.headers)
        seen["params"] = dict(req.params)
        seen["body"] = dict(req.json_body)
        return RawHttpResponse(200, {}, _ok_body("ok"), "")

    # header_key auth -> key goes into the configured header, not Authorization
    c1 = VlmConnection("h", "h", ConnectionKind.CUSTOM_LOCAL, "openai_chat_completions",
                       "http://x/v1", "m", auth=AuthSpec(type="header_key", secret_ref="r",
                                                         header_name="X-Api-Key"))
    old = _patch(responder)
    try:
        ex = VlmExecutor({"h": c1}, lambda ref: "SECRET")
        ex.caption_one(_spec(), ["h"])
        assert seen["headers"].get("X-Api-Key") == "SECRET"
        assert "Authorization" not in seen["headers"]

        # query_key auth -> key goes into the query string
        c2 = VlmConnection("q", "q", ConnectionKind.CUSTOM_LOCAL, "openai_chat_completions",
                           "http://x/v1", "m", auth=AuthSpec(type="query_key", secret_ref="r",
                                                             query_param="api_key"))
        ex2 = VlmExecutor({"q": c2}, lambda ref: "SECRET")
        ex2.caption_one(_spec(), ["q"])
        assert seen["params"].get("api_key") == "SECRET"

        # bearer auth -> Authorization: Bearer
        c3 = VlmConnection("b", "b", ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions",
                           "http://x/v1", "m", auth=AuthSpec(type="bearer", secret_ref="r"),
                           request_headers={"anthropic-workspace-id": "wrkspc_test"})
        ex3 = VlmExecutor({"b": c3}, lambda ref: "SECRET")
        ex3.caption_one(_spec(), ["b"])
        assert seen["headers"].get("Authorization") == "Bearer SECRET"
        assert seen["headers"].get("anthropic-workspace-id") == "wrkspc_test"

        # provider-specific JSON options (Cloudflare's reasoning switch) are carried
        # through the normal executor as well as the diagnostic request.
        c4 = VlmConnection("cf", "cf", ConnectionKind.BUILTIN, "openai_chat_completions",
                           "http://x/v1", "m", provider_id="cloudflare",
                           request_body={"chat_template_kwargs": {"enable_thinking": False}})
        ex4 = VlmExecutor({"cf": c4}, lambda ref: None)
        ex4.caption_one(_spec(), ["cf"])
        assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        T.execute_http = old
    print("  transport auth routing (header_key / query_key / bearer): OK")


def test_scrub_exc_redacts_credentials_but_keeps_path():
    scrub = T._scrub_exc

    class E(Exception):
        pass

    # query-key credential in the URL (any param name), relative or absolute
    m = scrub(E("HTTPSConnectionPool(host='api.x.com') Max retries exceeded with "
                "url: /openai/v1/chat/completions?weirdkey=sk-SECRET123&x=1 (Caused by ...)"))
    assert "sk-SECRET123" not in m
    assert "/openai/v1/chat/completions" in m  # path kept for diagnostics

    # invalid / echoed bearer key in a generic RequestException message (any case)
    m2 = scrub(E("request failed: 401 for Authorization: Bearer xai-BADKEY-abcdef123"))
    assert "xai-BADKEY-abcdef123" not in m2
    m2u = scrub(E("rejected: BEARER xai-UPPER-abcdef123 is invalid"))
    assert "xai-UPPER-abcdef123" not in m2u

    # custom auth header value
    m3 = scrub(E("ConnectionError sending X-Api-Key: my-Sekret_Value.9 to host"))
    assert "my-Sekret_Value.9" not in m3

    # URL userinfo
    m4 = scrub(E("ProxyError: https://user:p%40ss@proxy.local:8080 unreachable"))
    assert "p%40ss" not in m4 and "user:" not in m4

    # nothing sensitive -> unchanged host/path text survives
    m5 = scrub(E("could not resolve host api.groq.com"))
    assert m5 == "could not resolve host api.groq.com"
    print("  _scrub_exc redaction (query / bearer / header / userinfo): OK")


def test_stop_job_on_prompt_format_error():
    conns = {"a": _conn("a"), "b": _conn("b")}
    # 400 with a prompt/format style error -> BAD_RESPONSE -> failover normally.
    # A dedicated PROMPT_FORMAT_ERROR reason -> stop_job. Simulate via a parser that
    # returns that reason: use a 200 body the OpenAI parser treats as empty is not it;
    # instead inject the reason directly through execute_http returning the error.
    from vlm_errors import VlmAttemptError, VlmErrorReason
    old = _patch(lambda req, **kw: VlmAttemptError(VlmErrorReason.PROMPT_FORMAT_ERROR, 400, "bad prompt shape"))
    try:
        ex = VlmExecutor(conns, lambda ref: None)
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.stop_job and not res.ok
        # only the first connection was tried (no failover on stop_job)
        assert len(res.attempts) == 1
    finally:
        T.execute_http = old
    print("  prompt-format error -> stop_job (no failover): OK")


def test_stop_mid_flight():
    conns = {"a": _conn("a"), "b": _conn("b")}
    stopped = {"v": False}
    old = _patch(lambda req, **kw: RawHttpResponse(503, {}, {}, "err"))
    try:
        ex = VlmExecutor(conns, lambda ref: None, stop_checker=lambda: stopped["v"])
        stopped["v"] = True
        res = ex.caption_one(_spec(), ["a", "b"])
        assert res.stopped
    finally:
        T.execute_http = old
    print("  stop checker aborts caption_one: OK")


def test_diagnostics_static():
    import vlm_diagnostics as D
    good = _conn("g")
    rep = D.diagnose(good, api_key=None, do_live_request=False)
    names = {i.name: i.status for i in rep.items}
    assert names["URL format"] is D.DiagStatus.PASS
    assert names["Model ID"] is D.DiagStatus.PASS
    assert names["HTTP response"] is D.DiagStatus.SKIP

    bad = VlmConnection("b", "b", ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions",
                        "not-a-url", "m", auth=AuthSpec(type="bearer", secret_ref="x"))
    rep2 = D.diagnose(bad, api_key=None, do_live_request=False)
    assert rep2.overall is D.DiagStatus.FAIL

    # a non-cloudflare template hole -> single URL-format FAIL, stop before any probe
    tmpl = VlmConnection("t", "t", ConnectionKind.CUSTOM_EXTERNAL, "openai_chat_completions",
                         "https://x.example/{region}/v1", "m",
                         auth=AuthSpec(type="bearer", secret_ref="x"))
    rep3 = D.diagnose(tmpl, api_key="k", do_live_request=True)
    assert [i.name for i in rep3.items] == ["URL format"]
    assert rep3.items[0].status is D.DiagStatus.FAIL and "region" in rep3.items[0].detail

    missing_auth = VlmConnection("missing", "missing", ConnectionKind.CUSTOM_EXTERNAL,
                                 "openai_chat_completions", "https://localhost/v1", "m",
                                 auth=AuthSpec(type="bearer", secret_ref="x"))
    rep_missing = D.diagnose(missing_auth, api_key=None, do_live_request=False)
    assert rep_missing.item("Auth").status is D.DiagStatus.FAIL
    assert rep_missing.item("HTTP response").status is D.DiagStatus.SKIP
    assert rep_missing.item("HTTP response").detail == "credential missing"

    for protocol in ("openai_responses", "anthropic_messages"):
        direct = _conn(f"direct-{protocol}", protocol)
        rep4 = D.diagnose(direct, api_key=None, do_live_request=False)
        assert rep4.item("Protocol").status is D.DiagStatus.PASS
    print("  diagnostics static checks: OK")


def test_diagnostics_worker_surfaces_internal_error():
    import vlm_diagnostics as D
    from vlm_worker import VlmDiagnosticsWorker
    old = D.diagnose
    reports = []
    D.diagnose = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("probe exploded"))
    try:
        worker = VlmDiagnosticsWorker(_conn("broken"), None)
        worker.report_ready.connect(reports.append)
        worker.run()
    finally:
        D.diagnose = old
    assert len(reports) == 1
    item = reports[0].item("Internal diagnostic error")
    assert item is not None and item.status is D.DiagStatus.FAIL
    assert "probe exploded" in item.detail
    print("  diagnostics worker exception -> visible failure report: OK")


def test_diagnostics_cloudflare_token_verify(monkeypatch):
    import vlm_diagnostics as D
    from vlm_transport import RawHttpResponse

    cf = VlmConnection("builtin-cloudflare", "Cloudflare", ConnectionKind.BUILTIN,
                       "openai_chat_completions",
                       "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
                       "@cf/x", auth=AuthSpec(type="bearer", secret_ref="x"),
                       provider_id="cloudflare",
                       request_body={"chat_template_kwargs": {"enable_thinking": False}})

    # valid + active token -> Auth PASS, HTTP response PASS, account-id only a WARN
    monkeypatch.setattr(D, "execute_http", lambda *a, **k: RawHttpResponse(
        200, {}, {"success": True, "result": {"status": "active"}}, ""))
    rep = D.diagnose(cf, api_key="cfut_ok", do_live_request=True)
    names = {i.name: i for i in rep.items}
    assert names["URL format"].status is D.DiagStatus.WARN
    assert names["Auth"].status is D.DiagStatus.PASS
    assert names["HTTP response"].status is D.DiagStatus.PASS
    assert rep.http_status == 200
    assert "Request build" not in names   # generation probe skipped for cloudflare

    # rejected token -> Auth FAIL, http_status 401 (api_key_dialog treats as "key wrong")
    monkeypatch.setattr(D, "execute_http", lambda *a, **k: RawHttpResponse(
        401, {}, {"success": False, "errors": [{"message": "Invalid API Token"}]}, ""))
    rep2 = D.diagnose(cf, api_key="cfut_bad", do_live_request=True)
    n2 = {i.name: i for i in rep2.items}
    assert n2["Auth"].status is D.DiagStatus.FAIL
    assert n2["HTTP response"].status is D.DiagStatus.FAIL
    assert rep2.http_status == 401

    # Account ID が埋まっていれば token/verify で終わらず、実際の画像生成・抽出まで進む。
    cf.base_url = "https://api.cloudflare.com/client/v4/accounts/0123456789abcdef0123456789abcdef/ai/v1"
    seen = {}
    def _cf_generation(req, **kwargs):
        seen["body"] = req.json_body
        return RawHttpResponse(200, {}, {"choices": [{"message": {"content": "cloudflare caption"}}]}, "")
    monkeypatch.setattr(D, "execute_http", _cf_generation)
    rep3 = D.diagnose(cf, api_key="cfut_ok", do_live_request=True)
    n3 = {i.name: i for i in rep3.items}
    assert n3["Request build"].status is D.DiagStatus.PASS
    assert n3["HTTP response"].status is D.DiagStatus.PASS
    assert n3["Caption extraction"].status is D.DiagStatus.PASS
    assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    print("  cloudflare: missing Account ID verifies token only; configured route generates: OK")


def test_diagnostics_live_extraction_branches():
    import vlm_diagnostics as D
    from vlm_protocols import get_protocol
    proto = get_protocol("gemini_generate_content")

    def _cls(body, text="{}"):
        return D._classify_extraction(RawHttpResponse(200, {}, body, text), proto)

    st, *_ = _cls({"candidates": [{"content": {"parts": [{"text": "a cat"}]}, "finishReason": "STOP"}]})
    assert st is D.DiagStatus.PASS

    # low token cap -> no parts, finishReason MAX_TOKENS: endpoint is fine -> WARN not FAIL
    st, *_ = _cls({"candidates": [{"finishReason": "MAX_TOKENS"}], "usageMetadata": {}})
    assert st is D.DiagStatus.WARN

    cf_error = RawHttpResponse(403, {}, {
        "success": False, "errors": [{"code": 10000, "message": "Authentication error"}]}, "")
    assert D._response_error_detail(cf_error) == "Authentication error"

    assert D.is_billing_or_credit_block(
        "AI Gateway requires a valid credit card on file to service requests")
    assert D.is_billing_or_credit_block(
        "Your credit balance is too low to access the Anthropic API")
    assert D.is_billing_or_credit_block("You have no credits remaining")
    assert not D.is_billing_or_credit_block("Invalid API key")

    # genuinely wrong shape -> FAIL with a body preview
    st, detail, key, _args, preview = _cls({"unexpected": "shape"}, '{"unexpected": "shape"}')
    assert st is D.DiagStatus.FAIL and "unexpected" in detail
    # 表示用は翻訳キー + 素のレスポンス本文プレビュー（本文は訳さない）。
    assert key == "Diag_D_Extract_Path_Mismatch"
    assert "unexpected" in preview

    # non-200 -> SKIP (nothing to extract)
    st, *_ = D._classify_extraction(RawHttpResponse(500, {}, {}, "boom"), proto)
    assert st is D.DiagStatus.SKIP

    responses = get_protocol("openai_responses")
    st, *_ = D._classify_extraction(RawHttpResponse(200, {}, {
        "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
        "output": []}, "{}"), responses)
    assert st is D.DiagStatus.WARN

    anthropic = get_protocol("anthropic_messages")
    st, *_ = D._classify_extraction(RawHttpResponse(200, {}, {
        "content": [], "stop_reason": "max_tokens"}, "{}"), anthropic)
    assert st is D.DiagStatus.WARN

    # A custom extraction path must not bypass protocol-level safety signals.
    custom_gemini = get_protocol("gemini_generate_content")
    custom_gemini.default_text_path = "custom.caption"
    st, *_ = D._classify_extraction(RawHttpResponse(200, {}, {
        "promptFeedback": {"blockReason": "SAFETY"},
        "custom": {"caption": "misleading text"},
    }, "{}"), custom_gemini, "custom.caption")
    assert st is D.DiagStatus.WARN
    print("  diagnostics extraction: text->PASS, MAX_TOKENS->WARN, bad shape->FAIL+preview, 500->SKIP: OK")


def test_diagnostics_rate_limit_can_mark_binding_verified():
    import vlm_diagnostics as D

    rep = D.DiagReport("builtin-openrouter")
    rep.add("URL format", D.DiagStatus.PASS, "https://openrouter.ai/api/v1")
    rep.add("Model ID", D.DiagStatus.PASS, "google/gemma-4-26b-a4b-it:free")
    rep.add("Protocol", D.DiagStatus.PASS, "openai_chat_completions")
    rep.add("Auth", D.DiagStatus.PASS, "accepted (server responded 429)")
    rep.add("Request build", D.DiagStatus.PASS, "POST https://openrouter.ai/api/v1/chat/completions")
    rep.add("Image input", D.DiagStatus.PASS, "image/jpeg, base64/data-url ready")
    rep.add("HTTP response", D.DiagStatus.WARN, "429 rate limited (endpoint reachable)")
    rep.add("Caption extraction", D.DiagStatus.SKIP, "no successful response to extract from")
    rep.http_status = 429
    assert rep.overall is D.DiagStatus.WARN
    assert rep.can_mark_binding_verified is True

    truncated = D.DiagReport("builtin-cloudflare")
    truncated.add("Auth", D.DiagStatus.PASS, "accepted (server responded 200)")
    truncated.add("Request build", D.DiagStatus.PASS, "POST https://api.cloudflare.com/client/v4/accounts/a/ai/v1/chat/completions")
    truncated.add("Image input", D.DiagStatus.PASS, "image/jpeg, base64/data-url ready")
    truncated.add("HTTP response", D.DiagStatus.PASS, "200 OK")
    truncated.add("Caption extraction", D.DiagStatus.WARN,
                  "response truncated at the diagnostic token cap (endpoint reachable)")
    truncated.http_status = 200
    assert truncated.can_mark_binding_verified is True

    billing = D.DiagReport("builtin-openai")
    billing.add("Auth", D.DiagStatus.PASS, "accepted; billing / credits unavailable")
    billing.add("Request build", D.DiagStatus.PASS, "POST https://api.openai.com/v1/responses")
    billing.add("Image input", D.DiagStatus.PASS, "image/jpeg, base64/data-url ready")
    billing.add("HTTP response", D.DiagStatus.WARN, "429 billing / credits unavailable")
    billing.add("Caption extraction", D.DiagStatus.SKIP, "no successful response to extract from")
    billing.http_status = 429
    assert billing.can_mark_binding_verified is False
    print("  diagnostics: authenticated 429 is reachable-confirmed; billing 429 is not: OK")


def test_diagnostics_lightweight_model_list_probe():
    import vlm_diagnostics as D

    conn = VlmConnection(
        "builtin-openrouter", "OpenRouter", ConnectionKind.BUILTIN,
        "openai_chat_completions", "http://localhost/v1", "google/gemma-4-31b-it:free",
        provider_id="openrouter", auth=AuthSpec(type="bearer", secret_ref="openrouter"))
    seen = []

    def _models(req, **kwargs):
        seen.append((req.method, req.url, dict(req.headers), dict(req.params), req.json_body))
        return RawHttpResponse(200, {}, {"data": [{"id": conn.model_id}]}, "")

    old_execute = D.execute_http
    D.execute_http = _models
    try:
        rep = D.diagnose(conn, api_key="router-key", do_live_request=True, lightweight=True)
    finally:
        D.execute_http = old_execute

    assert seen == [(
        "GET", "http://localhost/v1/models", {"Authorization": "Bearer router-key"}, {}, {})]
    assert rep.lightweight is True
    assert rep.item("Image input").status is D.DiagStatus.SKIP
    assert rep.item("HTTP response").status is D.DiagStatus.PASS
    assert rep.item("Caption extraction").status is D.DiagStatus.SKIP
    assert rep.can_mark_binding_verified is True

    # Rate limiting still proves authenticated endpoint reachability, while a billing
    # response is deliberately not promoted to confirmed.
    D.execute_http = lambda *a, **k: RawHttpResponse(
        429, {}, {"error": {"message": "rate limit exceeded"}}, "")
    try:
        limited = D.diagnose(conn, api_key="router-key", do_live_request=True, lightweight=True)
    finally:
        D.execute_http = old_execute
    assert limited.item("HTTP response").status is D.DiagStatus.WARN
    assert limited.can_mark_binding_verified is True

    D.execute_http = lambda *a, **k: RawHttpResponse(
        429, {}, {"error": {"message": "You have no credits remaining"}}, "")
    try:
        billing = D.diagnose(conn, api_key="router-key", do_live_request=True, lightweight=True)
    finally:
        D.execute_http = old_execute
    assert billing.item("Auth").status is D.DiagStatus.PASS
    assert billing.item("HTTP response").status is D.DiagStatus.WARN
    assert billing.can_mark_binding_verified is False

    cf = VlmConnection(
        "builtin-cloudflare", "Cloudflare", ConnectionKind.BUILTIN,
        "openai_chat_completions",
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1", "@cf/google/gemma-4-31b-it",
        provider_id="cloudflare", auth=AuthSpec(type="bearer", secret_ref="cloudflare"))
    cf_req = D._model_list_request(cf, "cf-key")
    assert cf_req.method == "GET"
    assert cf_req.url.endswith("/ai/models/search")
    assert cf_req.params == {"per_page": "100"}
    print("  lightweight diagnostics: model-list GET only; 429 WARN/reachability; billing not confirmed: OK")


def test_diagnostics_billing_block_is_not_auth_failure():
    import vlm_diagnostics as D

    conn = VlmConnection(
        "builtin-vercel", "Vercel", ConnectionKind.BUILTIN,
        "openai_chat_completions", "http://localhost/v1", "google/gemma-4-26b-a4b-it",
        provider_id="vercel", auth=AuthSpec(type="bearer", secret_ref="x"))
    old = D.execute_http
    try:
        cases = (
            (403, "AI Gateway requires a valid credit card on file to service requests"),
            (429, "You have no credits remaining"),
            (400, "Your credit balance is too low to access the Anthropic API"),
        )
        for status, message in cases:
            D.execute_http = lambda *a, _status=status, _message=message, **k: RawHttpResponse(
                _status, {}, {"error": {"message": _message}}, "")
            rep = D.diagnose(conn, api_key="valid-key", do_live_request=True)
            assert rep.http_status == status
            assert rep.item("Auth").status is D.DiagStatus.PASS
            assert "billing / credits unavailable" in rep.item("Auth").detail
            assert rep.item("HTTP response").status is D.DiagStatus.WARN
            assert rep.overall is D.DiagStatus.WARN
            assert message in rep.item("HTTP response").detail
            assert rep.can_mark_binding_verified is False
    finally:
        D.execute_http = old
    print("  diagnostics: billing/card blocks are HTTP WARN with auth PASS, never verified: OK")


def test_model_list_fetch():
    import vlm_model_list as ML
    from vlm_connections import VlmConnection, ConnectionKind, AuthSpec
    oai = VlmConnection("c", "c", ConnectionKind.BUILTIN, "openai_chat_completions",
                        "https://x/v1", "m", auth=AuthSpec(type="bearer", secret_ref="r"))
    gem = VlmConnection("g", "g", ConnectionKind.BUILTIN, "gemini_generate_content",
                        "https://y/v1beta", "m",
                        auth=AuthSpec(type="header_key", secret_ref="r", header_name="x-goog-api-key"))
    old = ML.execute_http
    try:
        ML.execute_http = lambda req, **kw: RawHttpResponse(200, {}, {"data": [{"id": "a"}, {"id": "b"}, {"id": "a"}]}, "")
        assert ML.fetch_model_ids(oai, "k") == ["a", "b"]                      # deduped

        ML.execute_http = lambda req, **kw: RawHttpResponse(200, {}, {"models": [
            {"name": "models/gemma-3-27b-it", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]}, "")
        assert ML.fetch_model_ids(gem, "k") == ["gemma-3-27b-it"]              # models/ stripped, non-vision filtered

        ML.execute_http = lambda req, **kw: RawHttpResponse(401, {}, {}, "unauthorized")
        assert ML.fetch_model_ids(oai, "bad").reason is VlmErrorReason.AUTH_ERROR

        ML.execute_http = lambda req, **kw: RawHttpResponse(404, {}, None, "nope")
        assert ML.fetch_model_ids(oai, "k").reason is VlmErrorReason.BAD_RESPONSE

        # Cloudflare: different endpoint + {"result":[{"name":...,"task":{"name":...}}]}
        cf = VlmConnection("cf", "cf", ConnectionKind.BUILTIN, "openai_chat_completions",
                           "https://api.cloudflare.com/client/v4/accounts/acct-1/ai/v1", "m",
                           provider_id="cloudflare", auth=AuthSpec(type="bearer", secret_ref="r"))
        seen = {}
        def _cf(req, **kw):
            seen["url"] = req.url
            return RawHttpResponse(200, {}, {"result": [
                {"name": "@cf/meta/llama-3.2-11b-vision", "task": {"name": "Image-to-Text"}},
                {"name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}}]}, "")
        ML.execute_http = _cf
        assert ML.fetch_model_ids(cf, "k") == ["@cf/meta/llama-3.2-11b-vision"]
        assert seen["url"].endswith("/ai/models/search")

        # Anthropic uses the same {data:[{id:...}]} shape but requires a version header.
        anthropic = VlmConnection("an", "an", ConnectionKind.BUILTIN, "anthropic_messages",
                                  "https://api.anthropic.com/v1", "m", provider_id="anthropic",
                                  auth=AuthSpec(type="header_key", secret_ref="r",
                                                header_name="x-api-key"),
                                  request_headers={"anthropic-workspace-id": "wrkspc_test"})
        def _anthropic(req, **kw):
            seen["anthropic_headers"] = dict(req.headers)
            return RawHttpResponse(200, {}, {"data": [{"id": "claude-haiku-4-5-20251001"}]}, "")
        ML.execute_http = _anthropic
        assert ML.fetch_model_ids(anthropic, "ak") == ["claude-haiku-4-5-20251001"]
        assert seen["anthropic_headers"]["anthropic-version"] == "2023-06-01"
        assert seen["anthropic_headers"]["x-api-key"] == "ak"
        assert seen["anthropic_headers"]["anthropic-workspace-id"] == "wrkspc_test"

        # New catalog path retains provider capability metadata and filters only after
        # classification, so a profile cannot hide unrelated but valid VLMs.
        catalog = ML._extract_catalog({"data": [
            {"id": "vendor/vision", "architecture": {
                "input_modalities": ["text", "image"], "output_modalities": ["text"]}},
            {"id": "vendor/text", "architecture": {
                "input_modalities": ["text"], "output_modalities": ["text"]}},
            {"id": "vendor/image-generator", "architecture": {
                "input_modalities": ["text", "image"], "output_modalities": ["text", "image"]},
             "description": "image generation model"},
        ]}, "openrouter")
        assert [e.model_id for e in ML.filter_vlm_catalog(catalog)] == ["vendor/vision"]
        assert catalog[0].capability_source == "model list input/output modalities"

        # llama.cpp returns the OpenAI data[].id beside a models[] row carrying
        # capabilities=multimodal; the parser must merge them before filtering.
        llama_id = "C:\\LLM\\models\\Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced\\model.gguf"
        llama_catalog = ML._extract_catalog({
            "data": [{"id": llama_id}],
            "models": [{"model": llama_id, "capabilities": ["completion", "multimodal"]}],
        }, "")
        assert len(ML.filter_vlm_catalog(llama_catalog)) == 1
        assert llama_catalog[0].supports_image_input is True
        assert "capability metadata" in llama_catalog[0].capability_source

        cf_catalog = ML._extract_catalog({"result": [
            {"name": "@cf/meta/llama-vision", "task": {"name": "Image-to-Text"}},
            {"name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}},
            {"name": "@cf/mistral/mistral-small-3.1-24b-instruct",
             "task": {"name": "Image-to-Text"}},
        ]}, "cloudflare")
        assert [e.model_id for e in ML.filter_vlm_catalog(cf_catalog)] == ["@cf/meta/llama-vision"]

    finally:
        ML.execute_http = old
    print("  model list fetch: IDs + capability metadata, Gemini/Cloudflare filters, auth: OK")


def test_worker_batch_with_mock(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PIL import Image
    app = QCoreApplication.instance() or QCoreApplication([])

    d = tmp_path
    for i in range(3):
        Image.new("RGB", (8, 8)).save(d / f"i{i}.png")
    (d / "i1.txt").write_text("1girl, solo", encoding="utf-8")

    import app_settings as A
    s = A.load_settings(A.get_default_config())
    s.paths.input_dir = str(d)
    s.vlm.enabled = True
    s.behavior.existing_file_mode = "APPEND"
    s.caption.placement = "APPEND"

    # make all three builtin bindings verified + provide fake auth + mock http.
    # This test relies on gemini failing to parse the OpenAI-shaped mock body
    # (gemini's real protocol expects a different JSON shape) and falling over to
    # openrouter, which does parse it - so it needs multiple candidates regardless
    # of the shipped DEFAULT_VLM_CONNECTION_ORDER (now just "gemini").
    s.vlm.connection_order = "gemini,openrouter,cloudflare"
    import vlm_models as M, dataclasses, vlm_secrets, vlm_config
    verified = {pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED, provider_constraint=None)
               for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()}
    monkeypatch.setattr(
        vlm_config, "resolve_model_profile",
        lambda v: dataclasses.replace(M.GEMMA_4_26B_A4B_IT, bindings=verified))
    monkeypatch.setattr(vlm_secrets, "get_secret", lambda ref: "FAKEKEY")

    http_calls = []

    def fake_http(req, **kw):
        http_calls.append(req.url)
        return RawHttpResponse(
            200, {}, _ok_body("a detailed natural language description of the scene"), "")
    monkeypatch.setattr(T, "execute_http", fake_http)

    from vlm_worker import VlmCaptionWorker
    logs = []
    prog = []
    batch = {"v": None}
    w = VlmCaptionWorker(s, decision_requester=None, get_string=lambda *a, **k: a[-1] if a else "")
    w.log_message.connect(lambda m, c: logs.append((m, c)))
    w.progress_update.connect(lambda a, b: prog.append((a, b)))
    w.batch_completed.connect(lambda lst: batch.__setitem__("v", lst))
    w.run_captioning()

    txt0 = (d / "i0.txt").read_text(encoding="utf-8")
    txt1 = (d / "i1.txt").read_text(encoding="utf-8")
    assert txt0 == "a detailed natural language description of the scene"
    assert txt1.startswith("1girl, solo\n") and "natural language description" in txt1
    assert batch["v"] is not None and len(batch["v"]) == 3
    assert prog and prog[-1] == (3, 3)
    # VlmExecutor.on_attempt_start must be wired to log_message: exactly one
    # "requesting/retrying X..." notification per actual HTTP attempt (some
    # bindings here parse-fail and fail over before one finally succeeds, so
    # this isn't simply 3 - it must track the real attempt count 1:1, which is
    # what lets the UI show something is happening for every attempt that can
    # take up to read_timeout_s, not just the first one per image).
    attempt_notifications = sum(1 for m, _ in logs if m in ("Attempt_Start", "Attempt_Retry"))
    assert attempt_notifications == len(http_calls) > 0
    print(f"  worker batch (mock http): OK  ({len(batch['v'])} files written, {len(prog)} progress)")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
