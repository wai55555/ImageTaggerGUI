"""VLM Phase 0/1 backend unit tests (260901_VLM_spec.md 12.1).

Offline only - no network, no API keys. Run:  python tests/test_vlm_phase1.py
"""
import io
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

import vlm_models as M
import vlm_profiles as P
import vlm_image as IMG
import vlm_errors as E
import vlm_protocols as PROTO
import vlm_connections as C
import vlm_persistence as PERS
import vlm_router as R
import vlm_ratelimit as RL


def test_model_alias_and_identity():
    reg = M.default_registry()
    assert reg.resolve_alias("gemma-4-26b-a4b-it") is M.GEMMA_4_26B_A4B_IT
    assert reg.resolve_alias("google/gemma-4-26b-a4b-it:free") is M.GEMMA_4_26B_A4B_IT
    assert reg.resolve_alias("@cf/google/gemma-4-26b-a4b-it") is M.GEMMA_4_26B_A4B_IT
    assert reg.resolve_alias("totally-unknown") is None

    verified = M.ModelBinding("x", "m", M.ModelIdentityStatus.VERIFIED)
    assert verified.is_strict_fallback_eligible()

    # provider_constraint present but not pinned -> VERIFIED downgraded to DECLARED
    loose = M.ModelBinding("x", "m", M.ModelIdentityStatus.VERIFIED,
                           M.ProviderConstraint(allowed_providers=(), allow_fallbacks=True))
    assert loose.effective_identity_status() is M.ModelIdentityStatus.DECLARED
    assert not loose.is_strict_fallback_eligible()

    pinned = M.ModelBinding("x", "m", M.ModelIdentityStatus.VERIFIED,
                            M.ProviderConstraint(allowed_providers=("prov-a",), allow_fallbacks=False))
    assert pinned.is_strict_fallback_eligible()

    assert M.parse_identity_status("garbage") is M.ModelIdentityStatus.UNKNOWN
    assert not M.VlmModelProfile("p", "P", "m", quantization="unknown").quantization_is_strict()
    assert M.GEMMA_4_26B_A4B_IT.quantization_is_strict()
    print("  model alias + identity: OK")


def test_prompt_building():
    base = P.GenerationProfile()
    sp = P.build_system_prompt(base)
    assert "highly detailed natural-language" in sp
    assert "Write the output in English." in sp
    assert "as many sentences as necessary" in sp          # automatic
    assert "no bullet lists" in sp.lower() or "no bullet" in sp.lower()  # markdown disabled
    assert "unambiguously clear" in sp                     # explicit_only

    three = P.GenerationProfile(sentence_mode=P.SentenceMode.S3)
    assert "approximately 3 sentences" in P.build_system_prompt(three)
    assert "exactly" not in P.build_system_prompt(three).lower()

    one = P.GenerationProfile(sentence_mode=P.SentenceMode.S1)
    assert "approximately 1 sentence." in P.build_system_prompt(one)

    md_ok = P.GenerationProfile(markdown=P.MarkdownMode.ALLOWED)
    assert "no bullet" not in P.build_system_prompt(md_ok).lower()

    no_id = P.GenerationProfile(character_name_mode=P.CharacterNameMode.DO_NOT_IDENTIFY)
    assert "Do not state any character name" in P.build_system_prompt(no_id)

    custom = P.GenerationProfile(custom_system_prompt="JUST DO IT")
    assert P.build_system_prompt(custom) == "JUST DO IT"

    # from_mapping clamps / falls back
    g = P.GenerationProfile.from_mapping({"detail_level": "bogus", "max_output_tokens": 999999,
                                          "sentence_mode": "3", "markdown": "allowed"})
    assert g.detail_level is P.DetailLevel.MAXIMUM_DETAIL
    assert g.max_output_tokens == 32768
    assert g.sentence_mode is P.SentenceMode.S3
    assert g.markdown is P.MarkdownMode.ALLOWED
    print("  prompt building: OK")


def test_prompt_modes_are_distinct_and_factual():
    dataset = P.GenerationProfile(prompt_mode=P.PromptMode.DATASET_LONG)
    dataset_system = P.build_system_prompt(dataset)
    assert "training caption" in dataset_system
    assert "Transcribe readable text exactly" in dataset_system
    assert P.build_user_prompt(dataset) == "Create a training caption for this image."
    assert "as many sentences as necessary" not in dataset_system

    tags = P.GenerationProfile(prompt_mode=P.PromptMode.SHORT_TAGS)
    tags_system = P.build_system_prompt(tags)
    assert "comma-separated visual tags" in tags_system
    assert "Output only tags separated by commas" in tags_system
    assert P.build_user_prompt(tags) == "Convert this image into concise comma-separated tags."
    assert "no explanation" in tags_system

    mapped = P.GenerationProfile.from_mapping({"prompt_mode": "short_tags"})
    assert mapped.prompt_mode is P.PromptMode.SHORT_TAGS
    assert P.GenerationProfile.from_mapping({"prompt_mode": "invalid"}).prompt_mode is P.PromptMode.STANDARD


def test_extract_by_path():
    obj = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 5}}
    assert PROTO.extract_by_path(obj, "choices[0].message.content") == "hello"
    assert PROTO.extract_by_path(obj, "choices[0].finish_reason") == "stop"
    assert PROTO.extract_by_path(obj, "usage.prompt_tokens") == 5
    assert PROTO.extract_by_path(obj, "choices[1].message.content") is None
    assert PROTO.extract_by_path(obj, "choices[0].missing") is None
    assert PROTO.extract_by_path(obj, "") is None
    nested = {"a": {"b": [[{"c": 1}]]}}
    assert PROTO.extract_by_path(nested, "a.b[0][0].c") == 1
    print("  extract_by_path: OK")


def _prepared():
    return IMG.PreparedImage(data=b"\xff\xd8\xff", mime_type="image/jpeg")


def test_protocol_parse():
    op = PROTO.OpenAIChatCompletionsProtocol()
    ok = op.parse_response(200, {"choices": [{"message": {"content": " a caption "},
                                              "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10, "completion_tokens": 20}}, "")
    assert ok.ok and ok.text == "a caption" and ok.completion_tokens == 20

    empty = op.parse_response(200, {"choices": [{"message": {"content": ""}}]}, "")
    assert not empty.ok and empty.error.reason is E.VlmErrorReason.EMPTY_RESPONSE

    filt = op.parse_response(200, {"choices": [{"finish_reason": "content_filter",
                                               "message": {"content": ""}}]}, "")
    assert filt.error.reason is E.VlmErrorReason.CONTENT_POLICY

    e429 = op.parse_response(429, {"error": {"code": "rate_limit"}}, "too many")
    assert e429.error.reason is E.VlmErrorReason.RATE_LIMITED

    e401 = op.parse_response(401, {}, "bad key")
    assert e401.error.reason is E.VlmErrorReason.AUTH_ERROR

    gm = PROTO.GeminiGenerateContentProtocol()
    gok = gm.parse_response(200, {"candidates": [{"content": {"parts": [{"text": "desc"}]},
                                                  "finishReason": "STOP"}],
                                  "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 7}}, "")
    assert gok.ok and gok.text == "desc" and gok.completion_tokens == 7

    gblock = gm.parse_response(200, {"promptFeedback": {"blockReason": "SAFETY"}}, "")
    assert gblock.error.reason is E.VlmErrorReason.CONTENT_POLICY

    # thinking model: parts[0] is an empty thought, the answer is in a later part
    gthink = gm.parse_response(200, {"candidates": [{"content": {"parts": [
        {"text": "", "thought": True},
        {"text": "a solid grey square"},
    ]}, "finishReason": "STOP"}]}, "")
    assert gthink.ok and gthink.text == "a solid grey square", gthink.text

    # only a thought part, no answer -> still EMPTY_RESPONSE
    gthink_only = gm.parse_response(200, {"candidates": [{"content": {"parts": [
        {"text": "", "thought": True}]}, "finishReason": "MAX_TOKENS"}]}, "")
    assert not gthink_only.ok and gthink_only.error.reason is E.VlmErrorReason.EMPTY_RESPONSE

    # A custom Gemini extraction path must not be masked by a normal candidate part.
    gm.default_text_path = "custom.caption"
    gcustom = gm.parse_response(200, {
        "candidates": [{"content": {"parts": [{"text": "normal candidate"}]}}],
        "custom": {"caption": "custom caption"},
    }, "")
    assert gcustom.ok and gcustom.text == "custom caption"
    gcustom_missing = gm.parse_response(200, {
        "candidates": [{"content": {"parts": [{"text": "normal candidate"}]}}],
    }, "")
    assert not gcustom_missing.ok and gcustom_missing.error.reason is E.VlmErrorReason.EMPTY_RESPONSE

    responses = PROTO.OpenAIResponsesProtocol()
    rok = responses.parse_response(200, {
        "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "first "},
            {"type": "output_text", "text": "caption"}]}],
        "usage": {"input_tokens": 11, "output_tokens": 22}}, "")
    assert rok.ok and rok.text == "first caption" and rok.completion_tokens == 22

    anthropic = PROTO.AnthropicMessagesProtocol()
    aok = anthropic.parse_response(200, {
        "content": [{"type": "text", "text": "Claude "},
                    {"type": "text", "text": "caption"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": 23}}, "")
    assert aok.ok and aok.text == "Claude caption" and aok.prompt_tokens == 12
    assert anthropic.parse_response(401, {"error": {"type": "authentication_error"}},
                                    "bad key").error.reason is E.VlmErrorReason.AUTH_ERROR
    print("  protocol parse (chat/responses/anthropic/gemini): OK")


def test_protocol_build_request():
    spec = PROTO.VlmCallSpec(model_id="m", system_prompt="sys", user_prompt="u",
                             image=_prepared(), profile=P.GenerationProfile(temperature=0.4))
    op = PROTO.OpenAIChatCompletionsProtocol()
    req = op.build_request("http://localhost:1234/v1", "KEY", spec)
    assert req.url.endswith("/v1/chat/completions")
    assert req.headers["Authorization"] == "Bearer KEY"
    assert req.json_body["model"] == "m"
    assert req.json_body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert req.json_body["temperature"] == 0.4
    assert req.json_body["max_tokens"] == 3072

    gm = PROTO.GeminiGenerateContentProtocol()
    greq = gm.build_request("https://x/v1beta", "GKEY", spec)
    assert greq.url == "https://x/v1beta/models/m:generateContent"
    assert greq.headers["x-goog-api-key"] == "GKEY"
    assert greq.json_body["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"

    responses = PROTO.OpenAIResponsesProtocol()
    rreq = responses.build_request("https://api.openai.com/v1", "OKEY", spec)
    assert rreq.url == "https://api.openai.com/v1/responses"
    assert rreq.headers["Authorization"] == "Bearer OKEY"
    assert rreq.json_body["store"] is False
    assert rreq.json_body["input"][0]["content"][1]["type"] == "input_image"

    anthropic = PROTO.AnthropicMessagesProtocol()
    areq = anthropic.build_request("https://api.anthropic.com/v1", "AKEY", spec)
    assert areq.url == "https://api.anthropic.com/v1/messages"
    assert areq.headers["x-api-key"] == "AKEY"
    assert areq.headers["anthropic-version"] == "2023-06-01"
    source = areq.json_body["messages"][0]["content"][0]["source"]
    assert source["type"] == "base64" and source["media_type"] == "image/jpeg"

    rendered = PROTO.render_template('{"m":"{{model}}","p":"{{system_prompt}}"}', spec)
    assert rendered == '{"m":"m","p":"sys"}'
    print("  protocol build_request + template: OK")


def test_error_classification():
    def mk(reason):
        return E.VlmAttemptError(reason)
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=1) is E.VlmErrorClass.RETRY_SAME
    # retry_same_max=1（既定）を使い切った直後（1画像分＝consecutive_timeouts=2）は
    # まだ FAILOVER。間欠的に遅い／たまたま1枚だけ運悪くタイムアウトしただけの
    # 健全な接続まで即座に締め出さないため、閾値は「2画像分」にしている
    # （2026-09 VLM デバッグ: 実機で Gemini が4枚中2枚成功していたのに、1画像分の
    # 失敗だけで除外され、後続7枚が一度も試行されず打ち切られた事故を踏まえた）。
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=2) is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=3) is E.VlmErrorClass.FAILOVER
    # 2画像分（(retry_same_max+1)×2 = 4）に達して初めて EXCLUDE。壊れた接続を
    # 毎画像フルに再試行し続ける無駄は、2画像分の犠牲で確実に止める。
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=4) is E.VlmErrorClass.EXCLUDE
    # 再試行は使い切っていても、セッション全体としてはまだ1回しかタイムアウトして
    # いない（consecutive_timeouts が閾値未満）場合は、まだ EXCLUDE せず FAILOVER。
    assert mk(E.VlmErrorReason.TIMEOUT).classify(consecutive_timeouts=1, already_retried_same=True) is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.TIMEOUT).classify(
        consecutive_timeouts=2, same_retries=1, retry_same_max=2) is E.VlmErrorClass.RETRY_SAME
    # retry_same_max=2 なら閾値は6（=2画像分＝(2+1)×2）。5ではまだ FAILOVER。
    assert mk(E.VlmErrorReason.TIMEOUT).classify(
        consecutive_timeouts=5, same_retries=2, retry_same_max=2) is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.TIMEOUT).classify(
        consecutive_timeouts=6, same_retries=2, retry_same_max=2) is E.VlmErrorClass.EXCLUDE
    assert mk(E.VlmErrorReason.RATE_LIMITED).classify() is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.SERVER_ERROR).classify() is E.VlmErrorClass.RETRY_SAME
    assert mk(E.VlmErrorReason.SERVER_ERROR).classify(already_retried_same=True) is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.AUTH_ERROR).classify() is E.VlmErrorClass.EXCLUDE
    assert mk(E.VlmErrorReason.MODEL_UNSUPPORTED).classify() is E.VlmErrorClass.EXCLUDE
    assert mk(E.VlmErrorReason.CONTENT_POLICY).classify() is E.VlmErrorClass.FAILOVER
    assert mk(E.VlmErrorReason.PROMPT_FORMAT_ERROR).classify() is E.VlmErrorClass.STOP_JOB
    assert mk(E.VlmErrorReason.EMPTY_RESPONSE).classify() is E.VlmErrorClass.RETRY_SAME
    assert E.reason_from_http_status(429) is E.VlmErrorReason.RATE_LIMITED
    assert E.reason_from_http_status(503) is E.VlmErrorReason.SERVER_ERROR
    assert E.reason_from_http_status(404) is E.VlmErrorReason.MODEL_UNSUPPORTED
    assert E.reason_from_http_status(
        400, "messages[1].content must be a string") is E.VlmErrorReason.PROMPT_FORMAT_ERROR
    assert E.reason_from_http_status(
        400, "invalid image data") is E.VlmErrorReason.BAD_RESPONSE
    print("  error classification: OK")


def test_newline_join_and_dedup():
    assert PERS.combine_caption("", "cap", "APPEND") == "cap"
    assert PERS.combine_caption("tags", "", "APPEND") == "tags"
    assert PERS.combine_caption("tags", "cap", "APPEND") == "tags\ncap"
    assert PERS.combine_caption("tags", "cap", "PREPEND") == "cap\ntags"
    assert PERS.combine_caption("tags", "cap", "OVERWRITE") == "cap"
    assert PERS.combine_caption("tags", "cap", "bogus") == "cap"  # -> OVERWRITE

    assert PERS.caption_already_present("a\nmy caption\nb", "my caption")
    assert PERS.caption_already_present("my caption", "my caption")
    assert not PERS.caption_already_present("a\nb", "my caption")
    # 改行コードは正規化して判定する（CRLF の既存 .txt / CRLF のキャプション両方向）。
    assert PERS.caption_already_present("a\r\nmy caption\r\nb", "my caption")   # CRLF file + LF caption
    assert PERS.caption_already_present("a\nline1\nline2\nb", "line1\r\nline2")  # LF file + CRLF caption
    assert PERS.caption_already_present("line1\r\nline2", "line1\nline2")        # whole-body, mixed
    print("  newline join + dedup: OK")


def test_atomic_save():
    d = Path(tempfile.mkdtemp())
    # new file
    p = d / "img1.txt"
    out = PERS.save_caption(p, "brand new caption", "OVERWRITE")
    assert out.written and out.previous_content is None and p.read_text(encoding="utf-8") == "brand new caption"

    # append to existing
    p2 = d / "img2.txt"
    p2.write_text("1girl, solo", encoding="utf-8")
    out2 = PERS.save_caption(p2, "a long natural caption", "APPEND")
    assert out2.written and p2.read_text(encoding="utf-8") == "1girl, solo\na long natural caption"

    # duplicate append -> skipped, file unchanged
    out3 = PERS.save_caption(p2, "a long natural caption", "APPEND")
    assert not out3.written and out3.skipped_reason == "duplicate"
    assert p2.read_text(encoding="utf-8") == "1girl, solo\na long natural caption"

    # overwrite identical -> no_change
    p3 = d / "img3.txt"
    p3.write_text("same", encoding="utf-8")
    out4 = PERS.save_caption(p3, "same", "OVERWRITE")
    assert not out4.written and out4.skipped_reason == "no_change"

    # CRLF の既存 .txt は APPEND しても改行規約を保持する（黙って LF 化しない）。
    # undo スナップショットは LF 正規化して返す。
    p5 = d / "img5.txt"
    p5.write_bytes(b"1girl, solo\r\ntag2\r\n")
    out5 = PERS.save_caption(p5, "a fresh natural caption", "APPEND")
    assert out5.written
    disk = p5.read_bytes()
    assert b"\r\n" in disk and b"\r\r\n" not in disk
    assert disk == b"1girl, solo\r\ntag2\r\na fresh natural caption"
    assert "\r" not in (out5.previous_content or "")  # LF snapshot for undo
    assert "\r" not in out5.new_content
    # 同じキャプションの再 APPEND は CRLF ファイルでも重複として弾く
    out5b = PERS.save_caption(p5, "a fresh natural caption", "APPEND")
    assert not out5b.written and out5b.skipped_reason == "duplicate"

    # 生成キャプションに \r\n が混じっていても保存が失敗しない（LF に正規化して扱う）。
    p6 = d / "img6.txt"
    out6 = PERS.save_caption(p6, "first line\r\nsecond line\r\n", "OVERWRITE")
    assert out6.written and p6.read_bytes() == b"first line\nsecond line"

    # no leftover temp files
    assert not list(d.glob("*.vlmtmp"))

    # unreadable existing file raises (caller must treat as error, not overwrite)
    p4 = d / "img4.txt"
    p4.write_bytes(b"\xff\xfe\x00bad")
    try:
        PERS.save_caption(p4, "cap", "APPEND")
        assert False, "expected read error"
    except UnicodeDecodeError:
        pass
    print("  atomic save (new/append/dup/no-change/unreadable): OK")


def test_router_builtin_fallback():
    profile = M.GEMMA_4_26B_A4B_IT
    # make all three bindings VERIFIED for this test
    import dataclasses
    verified_bindings = {pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED,
                                                  provider_constraint=None)
                         for pid, b in profile.bindings.items()}
    profile = dataclasses.replace(profile, bindings=verified_bindings)

    conns = {c.connection_id: c for c in C.default_builtin_connections()}
    has_auth = {cid: True for cid in conns}

    pol = R.RouterPolicy()
    cs = R.select_candidates(profile, conns, pol, has_auth=has_auth)
    assert cs.connection_ids == ["builtin-gemini", "builtin-openrouter", "builtin-cloudflare",
                                 "builtin-huggingface", "builtin-vercel"]

    # missing auth on gemini -> excluded
    cs3 = R.select_candidates(profile, conns, pol, has_auth={"builtin-openrouter": True})
    assert "builtin-gemini" not in cs3.connection_ids and cs3.excluded["builtin-gemini"] == "no_auth"

    # cooldown on openrouter
    cs4 = R.select_candidates(profile, conns, pol, has_auth=has_auth,
                              cooldown_until={"builtin-openrouter": time.time() + 999})
    assert cs4.connection_ids == ["builtin-gemini", "builtin-cloudflare",
                                  "builtin-huggingface", "builtin-vercel"]

    # shipped profile: bindings are DECLARED. Default policy allows DECLARED
    # (user picks the services + order), so every configured binding qualifies.
    cs5 = R.select_candidates(M.GEMMA_4_26B_A4B_IT, conns, pol, has_auth=has_auth)
    assert cs5.connection_ids == ["builtin-gemini", "builtin-openrouter", "builtin-cloudflare",
                                  "builtin-vercel"]

    # strict opt-out: allow_declared_identity=False -> DECLARED excluded as before
    strict = R.RouterPolicy(allow_declared_identity=False)
    cs6 = R.select_candidates(M.GEMMA_4_26B_A4B_IT, conns, strict, has_auth=has_auth)
    assert not cs6.has_candidates and cs6.rejected_reason == "no_verified_candidate"
    assert cs6.excluded.get("builtin-gemini") == "not_verified"

    # UNKNOWN identity is always excluded, even with allow_declared_identity=True
    import dataclasses as _dc
    unknown = _dc.replace(M.GEMMA_4_26B_A4B_IT, bindings={
        pid: _dc.replace(b, identity_status=M.ModelIdentityStatus.UNKNOWN, provider_constraint=None)
        for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()})
    cs7 = R.select_candidates(unknown, conns, pol, has_auth=has_auth)
    assert not cs7.has_candidates and cs7.excluded.get("builtin-gemini") == "identity_unknown"
    assert "builtin-gemini=not_verified" in R.explain_candidate_failure(
        "no_verified_candidate", {"builtin-gemini": "not_verified"})
    print("  router builtin_fallback: OK")


def test_router_gemma31_gemini_verified_candidate():
    import dataclasses

    profile = dataclasses.replace(M.GEMMA_4_31B_IT, bindings={
        pid: dataclasses.replace(binding,
                                 identity_status=M.ModelIdentityStatus.VERIFIED,
                                 provider_constraint=None)
        for pid, binding in M.GEMMA_4_31B_IT.bindings.items()})
    conns = {c.connection_id: c for c in C.default_builtin_connections()}
    has_auth = {cid: True for cid in conns}
    candidates = R.select_candidates(profile, conns, R.RouterPolicy(), has_auth=has_auth)
    assert "builtin-gemini" in candidates.connection_ids
    print("  verified Gemma 4 31B Gemini route remains eligible: OK")


def test_router_custom_single():
    ext = C.VlmConnection.from_mapping({
        "connection_id": "cust-ext", "display_name": "ext", "kind": "custom_external",
        "protocol": "openai_chat_completions", "base_url": "https://api.example.com/v1",
        "model_id": "m", "auth": {"type": "bearer", "secret_ref": "x"},
    })
    loc = C.VlmConnection.from_mapping({
        "connection_id": "cust-loc", "display_name": "loc", "kind": "custom_local",
        "protocol": "openai_chat_completions", "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "m", "auth": {"type": "none"},
    })
    conns = {c.connection_id: c for c in (ext, loc)}
    profile = M.GEMMA_4_26B_A4B_IT

    # External custom connections are allowed when selected and authenticated.
    pol = R.RouterPolicy(execution_mode=R.ExecutionMode.CUSTOM_SINGLE,
                         selected_connection_id="cust-ext")
    cs = R.select_candidates(profile, conns, pol, has_auth={"cust-ext": True})
    assert cs.connection_ids == ["cust-ext"]

    # Local custom connection remains available without credentials.
    pol2 = R.RouterPolicy(execution_mode=R.ExecutionMode.CUSTOM_SINGLE,
                          selected_connection_id="cust-loc")
    cs2 = R.select_candidates(profile, conns, pol2)
    assert cs2.connection_ids == ["cust-loc"]

    # custom is never mixed into builtin fallback
    pol3 = R.RouterPolicy(execution_mode=R.ExecutionMode.BUILTIN_FALLBACK)
    cs3 = R.select_candidates(profile, conns, pol3)
    assert not cs3.has_candidates  # no builtin connections present at all
    print("  router custom_single: OK")


def test_ratelimit():
    now = 1_000_000.0
    st = RL.RateLimitState("c1")
    RL.update_from_429(st, {"Retry-After": "30"}, now=now)
    assert st.cooldown_until_utc == now + 30 and st.source == "header"

    st2 = RL.RateLimitState("c2")
    RL.update_from_429(st2, {"x-ratelimit-reset": str(now + 45)}, now=now)
    assert abs(st2.cooldown_until_utc - (now + 45)) < 1 and st2.source == "header"

    st3 = RL.RateLimitState("c3")
    RL.update_from_429(st3, {}, now=now)
    assert st3.cooldown_until_utc == now + RL.ESTIMATED_COOLDOWN_DEFAULT_S and st3.source == "estimated"

    st4 = RL.RateLimitState("c4")
    RL.update_from_429(st4, {"Retry-After": "99999"}, now=now)
    assert st4.cooldown_until_utc == now + RL.ESTIMATED_COOLDOWN_CAP_S  # capped

    # 不正な x-ratelimit-reset（NaN / inf / 負値）は明示リセット扱いにせず、
    # 推測クールダウンへ倒す。in_cooldown() が True になること。
    for bad in ("nan", "inf", "-inf", "-100"):
        st_bad = RL.RateLimitState("cb")
        RL.update_from_429(st_bad, {"x-ratelimit-reset": bad}, now=now)
        assert st_bad.source == "estimated", bad
        assert st_bad.cooldown_until_utc == now + RL.ESTIMATED_COOLDOWN_DEFAULT_S, bad
        assert st_bad.in_cooldown(now + 1), bad

    # 既に過ぎた epoch のリセット値も無効。推測クールダウンへ倒す。
    big_now = 2_000_000_000.0
    st_exp = RL.RateLimitState("ce")
    RL.update_from_429(st_exp, {"x-ratelimit-reset": str(big_now - 500)}, now=big_now)
    assert st_exp.source == "estimated"
    assert st_exp.in_cooldown(big_now + 1)

    assert st.in_cooldown(now + 10) and not st.in_cooldown(now + 31)
    RL.clear(st)
    assert not st.in_cooldown(now + 10)
    print("  ratelimit cooldown: OK")


def _make_img(w, h, mode="RGB", color=(120, 60, 30)):
    return Image.new(mode, (w, h), color)


def test_image_preprocess():
    # landscape resize to long edge
    prepared = IMG.prepare_image(_make_img(4000, 2000),
                                 IMG.ImagePreprocessConfig(max_long_edge=1000, fmt="jpeg"))
    reopened = Image.open(io.BytesIO(prepared.data))
    assert reopened.size == (1000, 500)
    assert prepared.mime_type == "image/jpeg"
    assert prepared.data_url.startswith("data:image/jpeg;base64,")
    assert len(prepared.base64) > 0

    # small image is not upscaled
    small = IMG.prepare_image(_make_img(100, 80), IMG.ImagePreprocessConfig(max_long_edge=1000))
    assert Image.open(io.BytesIO(small.data)).size == (100, 80)

    # RGBA is flattened onto the configured color, output is RGB
    rgba = _make_img(50, 50, "RGBA", (0, 0, 0, 0))
    flat = IMG.prepare_image(rgba, IMG.ImagePreprocessConfig(fmt="png", flatten_rgba_color=(255, 255, 255)))
    out = Image.open(io.BytesIO(flat.data))
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255)
    assert flat.mime_type == "image/png"

    # bytes input
    buf = io.BytesIO()
    _make_img(30, 30).save(buf, format="PNG")
    b = IMG.prepare_image(buf.getvalue())
    assert b.mime_type in ("image/jpeg", "image/png")
    print("  image preprocess: OK")


def test_connection_locality():
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "http://localhost:1234/v1") is C.ConnectionKind.CUSTOM_LOCAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "http://192.168.1.9:8000") is C.ConnectionKind.CUSTOM_LOCAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "http://172.16.9.4:8000") is C.ConnectionKind.CUSTOM_LOCAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "http://[fe80::1]:8000") is C.ConnectionKind.CUSTOM_LOCAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "http://10.example.com/v1") is C.ConnectionKind.CUSTOM_EXTERNAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "https://api.openai.com/v1") is C.ConnectionKind.CUSTOM_EXTERNAL
    assert C.resolve_custom_kind(C.ConnectionLocality.AUTO, "") is C.ConnectionKind.CUSTOM_EXTERNAL  # unknown -> external
    assert C.resolve_custom_kind(C.ConnectionLocality.LOCAL, "https://api.openai.com/v1") is C.ConnectionKind.CUSTOM_LOCAL

    low = C.VlmConnection.from_mapping({
        "connection_id": "low", "image": {"max_long_edge": 1}})
    high = C.VlmConnection.from_mapping({
        "connection_id": "high", "image": {"max_long_edge": 99999}})
    unset = C.VlmConnection.from_mapping({"connection_id": "unset"})
    assert low.image_max_long_edge == 256
    assert high.image_max_long_edge == 8192
    assert unset.image_max_long_edge is None

    print("  connection locality: OK")


def test_captioner_manifest_controls_decoder_cache_shape():
    import caption_core

    cfg = caption_core.build_captioner_config(Path("custom-captioner"), {
        "captioner": {
            "decoder_layers": 8,
            "decoder_attention_heads": 16,
            "d_model": 1024,
        },
    })
    assert cfg.decoder_layers == 8
    assert cfg.decoder_attention_heads == 16
    assert cfg.decoder_head_dim == 64


def test_missing_provider_error_codes_stay_empty():
    protocols = (
        PROTO.OpenAIResponsesProtocol(),
        PROTO.AnthropicMessagesProtocol(),
        PROTO.GeminiGenerateContentProtocol(),
    )
    for protocol in protocols:
        parsed = protocol.parse_response(400, {"error": {}}, "bad request")
        assert parsed.error is not None
        assert parsed.error.provider_code == ""


def test_failed_undo_and_redo_remain_retryable():
    from undo_manager import UndoManager

    class FailingAction:
        def undo(self):
            return False

        def redo(self):
            return False

        def description(self):
            return "failing action"

    action = FailingAction()
    manager = UndoManager()
    manager.push(action)
    assert manager.undo() is False
    assert manager.undo_stack == [action]

    manager.undo_stack.clear()
    manager.redo_stack.append(action)
    assert manager.redo() is False
    assert manager.redo_stack == [action]


def test_verified_binding_promotion():
    import types
    import dataclasses
    import vlm_config as CFG

    prof_id = M.GEMMA_4_26B_A4B_IT.profile_id
    vlm = types.SimpleNamespace(model_profile_id=prof_id, verified_bindings="",
                                verified_set=lambda: {t.strip() for t in vlm.verified_bindings.split(",") if t.strip()})

    # mark: adds a token, idempotent, keyed by profile:provider
    assert CFG.mark_binding_verified(vlm, "gemini", profile_id=prof_id) is True
    assert CFG.mark_binding_verified(vlm, "gemini", profile_id=prof_id) is False
    assert vlm.verified_bindings == f"{prof_id}:gemini"

    # resolve_model_profile promotes only the listed binding; others keep shipped status
    prof = CFG.resolve_model_profile(vlm)
    assert prof.bindings["gemini"].identity_status is M.ModelIdentityStatus.VERIFIED
    assert prof.bindings["openrouter"].identity_status is M.ModelIdentityStatus.DECLARED

    # strict router (allow_declared_identity=False) now keeps the promoted gemini, drops the rest
    conns = {c.connection_id: c for c in C.default_builtin_connections()}
    strict = R.RouterPolicy(allow_declared_identity=False)
    cs = R.select_candidates(prof, conns, strict, has_auth={cid: True for cid in conns})
    assert cs.connection_ids == ["builtin-gemini"], cs.connection_ids
    print("  verified binding promotion: mark + resolve promote + strict router: OK")


def test_multi_provider_profiles():
    import types
    import vlm_config as CFG

    ids = {p.profile_id for p in M.default_registry().all_profiles()}
    assert "pixtral-12b" not in ids
    from vlm_connections import default_builtin_connections
    assert all(c.provider_id != "mistral" for c in default_builtin_connections())
    assert {"gemma-4-26b-a4b-it", "gemma-4-31b-it", "qwen3.8-27b", "qwen3.6-27b",
            "openai-gpt-4o", "openai-gpt-4o-mini",
            "openai-gpt-5.6-sol", "openai-gpt-5.6-terra",
            "openai-gpt-5.6-luna", "claude-fable-5-1", "claude-fable-5",
            "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
            "claude-opus-4-6", "claude-opus-4-5", "claude-sonnet-5",
            "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"} <= ids

    def _s(profile_id):
        s = types.SimpleNamespace(model_profile_id=profile_id,
                                  cloudflare_account_id="", verified_bindings="", model_id_overrides="")
        s.verified_set = lambda: set()
        s.order_list = lambda: ["gemini", "openrouter", "cloudflare"]

        def _ov():
            out = {}
            for tok in s.model_id_overrides.split(","):
                if "=" in tok:
                    k, _, v = tok.strip().partition("=")
                    if k.strip() and v.strip():
                        out[k.strip()] = v.strip()
            return out
        s.model_id_override_map = _ov
        return s

    s = _s("qwen3.8-27b")
    prof = CFG.resolve_model_profile(s)
    cm = CFG.build_connection_map(s, prof)
    on = {cid for cid, c in cm.items() if c.enabled and c.kind is C.ConnectionKind.BUILTIN}
    assert on == {"builtin-openrouter", "builtin-nvidia", "builtin-groq"}, on
    assert cm["builtin-openrouter"].model_id == "qwen/qwen3.8-27b"
    assert "builtin-ovhcloud" not in cm
    assert cm["builtin-gemini"].enabled is False
    # A profile binding alone must not re-enable a route unchecked in the UI.
    assert CFG.ordered_builtin_provider_ids(s, prof) == \
        ["gemini", "openrouter", "cloudflare"]

    gemma = CFG.resolve_model_profile(_s("gemma-4-26b-a4b-it"))
    gemma_cm = CFG.build_connection_map(_s("gemma-4-26b-a4b-it"), gemma)
    assert gemma_cm["builtin-huggingface"].model_id == "google/gemma-4-26B-A4B-it"
    assert gemma.bindings["huggingface"].identity_status is M.ModelIdentityStatus.UNKNOWN
    assert "huggingface" not in CFG.ordered_builtin_provider_ids(_s("gemma-4-26b-a4b-it"), gemma)
    enabled_hf = _s("gemma-4-26b-a4b-it")
    enabled_hf.order_list = lambda: ["gemini", "openrouter", "cloudflare", "huggingface"]
    assert CFG.ordered_builtin_provider_ids(enabled_hf, gemma)[-1] == "huggingface"

    assert gemma_cm["builtin-vercel"].model_id == "google/gemma-4-26b-a4b-it"
    gpt = CFG.resolve_model_profile(_s("openai-gpt-5.6-luna"))
    gpt_cm = CFG.build_connection_map(_s("openai-gpt-5.6-luna"), gpt)
    assert gpt_cm["builtin-openai"].model_id == "gpt-5.6-luna"
    assert gpt_cm["builtin-openai"].protocol == "openai_responses"
    assert gpt_cm["builtin-vercel"].model_id == "openai/gpt-5.6-luna"
    assert CFG.ordered_builtin_provider_ids(_s("openai-gpt-5.6-luna"), gpt) == [
        "openai", "vercel"]
    auth = {cid: True for cid in gpt_cm}
    paid_direct = R.select_candidates(gpt, gpt_cm, R.RouterPolicy(), has_auth=auth)
    assert paid_direct.connection_ids == ["builtin-openai", "builtin-vercel"]
    claude_settings = _s("claude-haiku-4-5")
    claude_settings.anthropic_workspace_id = "wrkspc_test123"
    claude = CFG.resolve_model_profile(claude_settings)
    claude_cm = CFG.build_connection_map(claude_settings, claude)
    assert claude_cm["builtin-anthropic"].model_id == "claude-haiku-4-5-20251001"
    assert claude_cm["builtin-anthropic"].protocol == "anthropic_messages"
    assert claude_cm["builtin-anthropic"].request_headers == {
        "anthropic-workspace-id": "wrkspc_test123"}
    assert claude_cm["builtin-vercel"].model_id == "anthropic/claude-haiku-4.5"
    assert CFG.ordered_builtin_provider_ids(claude_settings, claude) == [
        "anthropic", "vercel"]
    claude_paid = R.select_candidates(
        claude, claude_cm, R.RouterPolicy(),
        has_auth={cid: True for cid in claude_cm})
    assert claude_paid.connection_ids == ["builtin-anthropic", "builtin-vercel"]

    for profile_id, provider, model_id in (
        ("openai-gpt-5.6-sol", "openai", "gpt-5.6-sol"),
        ("openai-gpt-5.6-terra", "openai", "gpt-5.6-terra"),
        ("claude-fable-5-1", "anthropic", "claude-fable-5-1"),
        ("claude-fable-5", "anthropic", "claude-fable-5"),
        ("claude-opus-5", "anthropic", "claude-opus-5"),
        ("claude-opus-4-8", "anthropic", "claude-opus-4-8"),
        ("claude-opus-4-7", "anthropic", "claude-opus-4-7"),
        ("claude-opus-4-6", "anthropic", "claude-opus-4-6"),
        ("claude-opus-4-5", "anthropic", "claude-opus-4-5-20251101"),
        ("claude-sonnet-5", "anthropic", "claude-sonnet-5"),
        ("claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5-20250929"),
    ):
        selected = CFG.resolve_model_profile(_s(profile_id))
        mapped = CFG.build_connection_map(_s(profile_id), selected)
        assert mapped[f"builtin-{provider}"].model_id == model_id
        assert mapped[f"builtin-{provider}"].enabled is True
    CFG.set_model_id_override(s, "nvidia", "qwen/qwen3-vl-32b-instruct", profile_id="qwen3.8-27b")
    assert CFG.build_connection_map(s, prof)["builtin-nvidia"].model_id == "qwen/qwen3-vl-32b-instruct"
    CFG.set_model_id_override(s, "nvidia", "", profile_id="qwen3.8-27b")
    assert "qwen3.8-27b:nvidia" not in s.model_id_override_map()
    CFG.set_model_id_override(s, "groq", "groq/compound-mini", profile_id="qwen3.8-27b")
    assert CFG.build_connection_map(s, prof)["builtin-groq"].model_id == "qwen3.8-27b"
    CFG.set_model_id_override(s, "groq", "", profile_id="qwen3.8-27b")
    print("  multi-provider profiles: HF opt-in, OVH disabled, model-id override: OK")


def test_default_vlm_profile_and_fallback_order():
    import app_settings as A
    import vlm_config as CFG

    settings = A.load_settings(A.get_default_config())
    assert settings.vlm.model_profile_id == "gemma-4-31b-it"
    assert settings.vlm.order_list() == [
        "gemini", "openrouter", "cloudflare"]
    assert CFG.ordered_builtin_provider_ids(settings.vlm) == settings.vlm.order_list()
    profile = CFG.resolve_model_profile(settings.vlm)
    connections = CFG.build_connection_map(settings.vlm, profile)
    assert connections["builtin-gemini"].enabled is True
    candidates = R.select_candidates(
        profile, connections, R.RouterPolicy(),
        has_auth={cid: True for cid in connections},
    )
    assert candidates.connection_ids[0] == "builtin-gemini"
    print("  default VLM profile Gemma 4 31B IT; fallback order Gemini -> OpenRouter -> Cloudflare: OK")


def test_model_id_match_against_profile():
    reg = M.default_registry()
    gemma = reg.get("gemma-4-26b-a4b-it")
    qwen = reg.get("qwen3.8-27b")

    # exact alias in the list -> score 1.0
    bid, sc = M.match_model_id(gemma, "openrouter",
                               ["google/gemma-4-26b-a4b-it:free", "qwen/qwen-2.5-vl-72b", "gpt-4o"])
    assert bid == "google/gemma-4-26b-a4b-it:free" and sc == 1.0

    # a different size (27b vs 26b) is NOT the same model -> no confident match
    bid, sc = M.match_model_id(gemma, "gemini", ["gemma-3-27b-it", "gemini-2.0-flash"])
    assert bid is None

    # nothing gemma-ish at all -> no match
    assert M.match_model_id(gemma, "groq", ["llama-3.3-70b-versatile", "mixtral-8x7b"])[0] is None

    bid, sc = M.match_model_id(qwen, "openrouter", ["qwen/qwen3.8-27b", "qwen/qwen-2.5-72b-instruct"])
    assert bid == "qwen/qwen3.8-27b"

    assert M.looks_same_family(gemma, "google/gemma-4-26b-a4b-it:free") is True
    assert M.looks_same_family(gemma, "gpt-4o") is False
    assert M.looks_same_family(gemma, "gemma-3-27b-it") is False   # different size -> different model
    print("  model-id match: exact alias, size mismatch rejected, family check: OK")


def test_new_vlm_model_ids_flags_only_uncataloged_ids():
    """new_vlm_model_ids() is the detection primitive behind the settings-dialog
    'new model found, not auto-added' notice: a provider releasing a new vision
    model shouldn't require waiting for a code update before a user even learns
    it exists, but it also must never auto-register anything (that's how the
    stale Groq/Gemma binding happened before)."""
    # a model already bound in a shipped profile is not "new"
    assert M.new_vlm_model_ids("gemini", ["gemini-2.5-flash"]) == []
    # an entirely unseen provider id is flagged
    new = M.new_vlm_model_ids("gemini", ["gemini-2.5-flash", "gemini-9.9-astra-preview"])
    assert new == ["gemini-9.9-astra-preview"]
    # duplicates in the candidate list are not repeated in the output
    assert M.new_vlm_model_ids("gemini", ["gemini-9.9-astra", "gemini-9.9-astra"]) == ["gemini-9.9-astra"]
    # empty/blank candidates are ignored, not reported as "new"
    assert M.new_vlm_model_ids("gemini", ["", "   "]) == []


def test_vlm_only_model_guard():
    qwen = M.default_registry().get("qwen3.8-27b")
    gemma = M.default_registry().get("gemma-4-26b-a4b-it")
    assert qwen is not None
    assert gemma is not None
    assert M.is_known_non_vision_model("groq", "groq/compound-mini") is True
    assert M.is_vlm_model_id(qwen, "groq", "groq/compound-mini") is False
    # OpenRouter's provider-prefixed alias must not cross into Groq when that
    # profile already declares a different exact Groq binding.
    assert M.is_vlm_model_id(qwen, "groq", "qwen/qwen3.8-27b") is False
    assert M.is_vlm_model_id(gemma, "groq", "qwen/qwen3.8-27b") is True
    assert M.filter_vlm_model_ids(
        qwen, "groq",
        ["groq/compound-mini", "qwen/qwen3.8-27b", "qwen3.8-27b",
         "llama-3.3-70b-versatile"],
    ) == ["qwen3.8-27b"]
    for provider, model_ids in {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"],
        "anthropic": [
            "claude-fable-5-1", "claude-fable-5", "claude-opus-5",
            "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
            "claude-opus-4-5-20251101", "claude-sonnet-5", "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
        ],
    }.items():
        assert all(M.is_vlm_model_id(gemma, provider, model_id) for model_id in model_ids)
        assert M.filter_vlm_model_ids(gemma, provider, model_ids) == model_ids
    assert not M.is_vlm_model_id(
        M.default_registry().get("openai-gpt-5.6-luna"), "openai", "gpt-5.6-sol")
    assert not M.is_vlm_model_id(
        M.default_registry().get("claude-haiku-4-5"), "anthropic", "claude-opus-4-7")
    assert not M.is_vlm_model_id(
        M.default_registry().get("claude-haiku-4-5"), "anthropic", "claude-fable-5-1")
    assert not M.is_vlm_model_id(
        M.default_registry().get("claude-opus-5"), "anthropic", "claude-sonnet-5")
    assert not M.is_vlm_model_id(gemma, "groq", "gpt-4o")
    assert not M.is_vlm_model_id(gemma, "anthropic", "gpt-4o")
    # Provider-specific aliases must not cross a bound route.
    gemma31 = M.default_registry().get("gemma-4-31b-it")
    assert gemma31 is not None
    assert M.is_vlm_model_id(gemma31, "gemini", "gemma-4-31b-it")
    assert not M.is_vlm_model_id(gemma31, "gemini", "google/gemma-4-31b-it:free")
    # Groq has no Gemma vision model at all (confirmed live: "model gemma-4-31b-it
    # does not exist"), so gemma-4-31b-it deliberately has no groq binding - this
    # used to assert the opposite before that stale binding was removed.
    assert not M.is_vlm_model_id(gemma31, "groq", "gemma-4-31b-it")
    # Cloudflare has no gemma-4-31b-it binding either (removed: confirmed live
    # that Cloudflare does not host that exact model). Cloudflare DOES host
    # gemma-4-26b-a4b-it, a same-family ("Gemma 4") but different-sized sibling
    # already correctly bound to GEMMA_4_26B_A4B_IT. A manual override must not
    # silently accept that sibling's id for the 31b profile - build_connection_map
    # would then enable/wire the route under the 31b profile's name while actually
    # running the 26b model. A genuinely different family (e.g. a Claude id) is
    # still an acceptable manual override for an unbound route - see the
    # openai/anthropic cross-provider assertions above.
    assert not M.is_vlm_model_id(gemma31, "cloudflare", "@cf/google/gemma-4-26b-a4b-it")
    assert M.is_vlm_model_id(gemma31, "cloudflare", "@cf/meta/llama-4-scout-17b-16e-instruct")

    bad_profile = M.VlmModelProfile(
        profile_id="user-bad", display_name="bad", canonical_model_id="groq/compound-mini",
        bindings={"groq": M.ModelBinding("groq", "groq/compound-mini")})
    import types
    bad_settings = types.SimpleNamespace(
        model_profile_id="user-bad", cloudflare_account_id="",
        anthropic_workspace_id="", verified_bindings="", model_id_overrides="",
        model_id_override_map=lambda: {}, order_list=lambda: ["groq"])
    import vlm_config as CFG
    bad_map = CFG.build_connection_map(bad_settings, bad_profile)
    assert bad_map["builtin-groq"].enabled is False
    arbitrary_profile = M.VlmModelProfile(
        profile_id="user-arbitrary", display_name="arbitrary", canonical_model_id="vendor/unknown",
        bindings={"groq": M.ModelBinding("groq", "vendor/unknown")})
    arbitrary_map = CFG.build_connection_map(bad_settings, arbitrary_profile)
    assert arbitrary_map["builtin-groq"].enabled is False
    print("  VLM-only model guard: Groq Compound rejected, Qwen vision model retained: OK")


def test_catalog_capability_classification_covers_all_builtin_providers():
    """Static fallbacks cover metadata-less APIs; explicit metadata stays authoritative."""
    expected = {
        "gemini": ["gemini-2.5-pro", "gemini-3.8-flash", "gemma-4-31b-it"],
        "cloudflare": ["@cf/google/gemma-3-12b-it", "@cf/meta/llama-3.2-11b-vision-instruct"],
        "groq": ["qwen/qwen3.6-27b", "qwen/qwen3.8-27b"],
        "nvidia": ["google/gemma-4-26b-it", "google/gemma-4-31b-it",
                    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                    "nvidia/nemotron-nano-12b-v2-vl"],
        "openai": ["gpt-4.1", "gpt-5.5", "o4-mini"],
        "anthropic": ["claude-fable-5-1", "claude-opus-5", "claude-sonnet-5",
                      "claude-haiku-4-5-20251001"],
    }
    for provider, model_ids in expected.items():
        assert all(M.classify_model_capability(provider, mid)[0] is True
                   for mid in model_ids), (provider, model_ids)
    assert M.classify_model_capability("nvidia", "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1")[0] is False
    assert M.classify_model_capability("groq", "groq/compound-mini")[0] is False

    image_text = {"architecture": {"input_modalities": ["text", "image"],
                                    "output_modalities": ["text"]}}
    text_only = {"architecture": {"input_modalities": ["text"],
                                   "output_modalities": ["text"]}}
    empty_input = {"architecture": {"input_modalities": [],
                                     "output_modalities": ["text"]}}
    image_generation = {
        "architecture": {"input_modalities": ["text", "image"],
                          "output_modalities": ["text", "image"]},
        "description": "image generation model",
    }
    assert M.classify_model_capability("openrouter", "vendor/vlm", image_text)[0] is True
    assert M.classify_model_capability("openrouter", "vendor/text", text_only)[0] is False
    assert M.classify_model_capability("openrouter", "vendor/empty", empty_input)[0] is False
    assert M.classify_model_capability("openrouter", "vendor/image", image_generation)[0] is False
    assert M.is_vlm_model_id(None, "gemini", "gemini-2.0-flash") is True
    assert M.is_vlm_model_id(None, "nvidia", "nvidia/nemotron-nano-12b-v2-vl") is True
    assert M.is_vlm_model_id(None, "gemini", "gemini-2.5-flash-image") is False
    assert M.is_vlm_model_id(None, "cloudflare", "@cf/mistral/mistral-small-3.1-24b-instruct") is False
    print("  capability catalog: all enabled providers + metadata: OK")


def test_user_defined_profiles():
    import tempfile
    import vlm_config as CFG
    old = CFG.VLM_PROFILES_PATH
    CFG.VLM_PROFILES_PATH = Path(tempfile.mkdtemp()) / "vp.json"
    try:
        assert len(CFG.all_profiles()) == 22 and not CFG.is_user_profile("x")

        CFG.save_user_profiles([{
            "profile_id": "user-g3", "display_name": "My Gemma 3 27B",
            "canonical_model_id": "gemma-3-27b-it",
            "bindings": {
                "gemini": {"model_id": "gemma-3-27b-it"},
                "openrouter": {"model_id": "google/gemma-3-27b-it:free"},
            }}])
        aps = {p.profile_id: p for p in CFG.all_profiles()}
        assert "user-g3" in aps and CFG.is_user_profile("user-g3")
        p = aps["user-g3"]
        assert set(p.bindings) == {"gemini", "openrouter"}
        assert p.bindings["gemini"].identity_status is M.ModelIdentityStatus.UNKNOWN
        assert p.bindings["gemini"].vlm_capable is True
        # Capability persisted in a user profile remains usable even when the
        # process-local discovered-model cache is empty after restart.
        custom = M.VlmModelProfile(
            profile_id="user-live", display_name="Live", canonical_model_id="vendor/new-vlm",
            bindings={"groq": M.ModelBinding(
                "groq", "vendor/new-vlm", vlm_capable=True)})
        assert M.is_vlm_model_id(custom, "groq", "vendor/new-vlm")

        CFG.save_user_profiles([{
            "profile_id": "user-bad-flag", "display_name": "Bad flag",
            "canonical_model_id": "vendor/text-only",
            "bindings": {"groq": {
                "model_id": "vendor/text-only", "vlm_capable": "false"}},
        }])
        bad_flag = CFG.user_profile_objects()[0]
        assert bad_flag.bindings["groq"].vlm_capable is False
        assert not M.is_vlm_model_id(bad_flag, "groq", "vendor/text-only")

        batch = M.VlmModelProfile(
            profile_id="user-batch", display_name="Batch", canonical_model_id="vendor/vlm:batch",
            bindings={"openrouter": M.ModelBinding(
                "openrouter", "vendor/vlm:batch", vlm_capable=True)})
        assert not M.is_vlm_model_id(batch, "openrouter", "vendor/vlm:batch")

        # a user profile with the same id as a shipped one overrides it
        CFG.save_user_profiles([
            {"profile_id": "user-g3", "display_name": "My Gemma 3 27B",
             "canonical_model_id": "gemma-3-27b-it",
             "bindings": {
                 "gemini": {"model_id": "gemma-3-27b-it"},
                 "openrouter": {"model_id": "google/gemma-3-27b-it:free"}}},
            {"profile_id": "gemma-4-26b-a4b-it", "display_name": "Gemma (mine)",
             "canonical_model_id": "gemma-3-27b-it",
             "bindings": {"gemini": {"model_id": "gemma-3-27b-it"}}},
        ])
        aps = {p.profile_id: p for p in CFG.all_profiles()}
        assert aps["gemma-4-26b-a4b-it"].display_name == "Gemma (mine)"
        assert len(CFG.all_profiles()) == 23  # 22 shipped (one overridden in place) + user-g3
    finally:
        CFG.VLM_PROFILES_PATH = old
    print("  user-defined profiles: json round-trip, merge, same-id override: OK")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} VLM PHASE 0/1 TESTS PASSED")
