"""HTTP 実行層とフォールバック実行ループ（260901_VLM_spec.md 7・8章 / design.md 4.9節）。

`vlm_protocols` が作った VlmHttpRequest を requests で実行し、結果を
`VlmProtocol.parse_response` へ渡す。ここに「どの接続を、どの順で、どうリトライするか」
の実行ループ（VlmExecutor）も置く。UI スレッドからは呼ばない。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import requests

from utils import write_debug_log
from vlm_connections import VlmConnection
from vlm_errors import VlmAttemptError, VlmErrorClass, VlmErrorReason
from vlm_image import PreparedImage
from vlm_profiles import DEFAULT_MAX_OUTPUT_TOKENS, GenerationProfile
from vlm_protocols import (
    VlmCallSpec, VlmParseResult, apply_connection_auth, apply_request_body, apply_request_headers,
    default_auth_key, get_protocol,
)
from vlm_ratelimit import RateLimitState, update_from_429

StopChecker = Callable[[], bool]


@dataclass(frozen=True)
class RawHttpResponse:
    status: int
    headers: dict[str, str]
    json_body: Any
    text_body: str


# 推論系VLM向けに助言する max_output_tokens。GenerationProfile の既定値をそのまま
# 使う（助言だけ古い値を言い続ける、を防ぐ）。
_SUGGESTED_MAX_OUTPUT_TOKENS = DEFAULT_MAX_OUTPUT_TOKENS


def _output_limit_reason(protocol_name: str, body: Any) -> str:
    """空本文が生成上限到達によるものなら終了理由を返す。"""
    if not isinstance(body, dict):
        return ""
    if protocol_name == "openai_chat_completions":
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            reason = str(choices[0].get("finish_reason") or "").strip().lower()
            return reason if reason in ("length", "max_tokens") else ""
    elif protocol_name == "openai_responses":
        incomplete = body.get("incomplete_details")
        reason = incomplete.get("reason") if isinstance(incomplete, dict) else ""
        if str(reason).strip().lower() in ("max_output_tokens", "length"):
            return str(reason).strip()
        if str(body.get("status") or "").strip().lower() == "incomplete":
            return "incomplete"
    elif protocol_name == "anthropic_messages":
        reason = str(body.get("stop_reason") or "").strip().lower()
        if reason == "max_tokens":
            return reason
    elif protocol_name == "gemini_generate_content":
        candidates = body.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            reason = str(candidates[0].get("finishReason") or "").strip().upper()
            if reason in ("MAX_TOKENS", "LENGTH"):
                return reason
    return ""


def _enrich_parse_failure(parsed: VlmParseResult, *, protocol, body: Any,
                          conn: VlmConnection, max_output_tokens: int) -> VlmParseResult:
    """失敗理由へ設定確認に必要な情報を付ける。

    UI側が ``reason.value`` だけを表示しても原因が分かるよう、レスポンス終了理由、
    使用中の上限、抽出パス、接続設定の確認先を message に集約する。
    """
    error = parsed.error
    if error is None:
        return parsed
    path = conn.text_path or getattr(protocol, "default_text_path", "") or "(protocol default)"
    limit_reason = _output_limit_reason(protocol.name, body)
    if error.reason is VlmErrorReason.EMPTY_RESPONSE and limit_reason:
        return replace(parsed, error=VlmAttemptError(
            VlmErrorReason.OUTPUT_LIMIT, error.http_status,
            "output limit reached: max_output_tokens={} finish_reason={}; "
            "the model returned no final text. Increase VLM max tokens to {} or higher "
            "for reasoning VLMs.".format(max_output_tokens, limit_reason,
                                         _SUGGESTED_MAX_OUTPUT_TOKENS),
            error.provider_code,
            message_key="Advice_Output_Limit",
            message_args={"limit": max_output_tokens, "finish_reason": limit_reason,
                          "suggested": _SUGGESTED_MAX_OUTPUT_TOKENS}))

    # 対処案は英語の message へ足したまま（デバッグログ用）にし、表示用には
    # 翻訳キーを別に持たせる。以前は英語の対処案がそのまま翻訳済みメッセージの
    # {reason} へ差し込まれ、どの言語でも助言だけ英語になっていた。
    hints = {
        VlmErrorReason.EMPTY_RESPONSE: (
            "no text at response path {!r}; verify API protocol and custom response extraction path".format(path),
            "Advice_Empty_Response", {"path": path}),
        VlmErrorReason.BAD_RESPONSE: (
            "verify API protocol, model ID, request format, and response path {!r}".format(path),
            "Advice_Bad_Response", {"path": path}),
        VlmErrorReason.AUTH_ERROR: (
            "verify API key, authentication type, header name, and query parameter",
            "Advice_Auth_Error", {}),
        VlmErrorReason.MODEL_UNSUPPORTED: (
            "verify the model ID and select a model supported by this endpoint",
            "Advice_Model_Unsupported", {}),
        VlmErrorReason.PROMPT_FORMAT_ERROR: (
            "verify API protocol and image/message format for this endpoint",
            "Advice_Prompt_Format_Error", {}),
        VlmErrorReason.TIMEOUT: (
            "verify the server is running and increase connect/read timeout if needed",
            "Advice_Timeout", {}),
        VlmErrorReason.NETWORK: (
            "verify base URL, host/port, TLS settings, and server availability",
            "Advice_Network", {}),
    }
    entry = hints.get(error.reason)
    if entry is None:
        return parsed
    hint, hint_key, hint_args = entry
    detail = (error.message or "").strip()
    message = f"{detail}; {hint}" if detail else hint
    return replace(parsed, error=replace(error, message=message[:800],
                                         message_key=hint_key, message_args=hint_args))


# URL トークン（絶対 URL でも `url: /rel/path?...` の相対形でも）のクエリ文字列を丸ごと伏せる。
_URL_QUERY_RE = re.compile(r"((?:https?://|/)[^\s'\"]*?)\?[^\s'\"]*")
# URL の userinfo（http://user:pass@host）も伏せる。
_URL_USERINFO_RE = re.compile(r"(https?://)[^/\s'\"@]+@")
# `?foo=VALUE` / `&foo=VALUE` の値部分を伏せる（キー名は問わない。クエリ文脈のみ）。
_SECRET_PARAM_RE = re.compile(r"([?&][A-Za-z0-9_.\-]{1,40}=)[^\s&'\"]+")
# `Bearer <token>` のトークン部を伏せる（この接頭辞がある文脈だけ。パスは壊さない）。
_BEARER_RE = re.compile(r"(bearer\s+)[^\s'\",;)}\]]+", re.IGNORECASE)
# 認証系ヘッダー名 ": value" / "= value" の値を伏せる（ヘッダー文脈だけ）。
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|x-goog-api-key|api[-_]?key|api[-_]?token"
    r"|anthropic-workspace-id)(['\"]?\s*[:=]\s*['\"]?)([^\s'\",;)}\]]+)")


def _scrub_exc(exc: Exception) -> str:
    """requests 例外のテキストから資格情報を落とす。

    query_key 認証では API キーが URL のクエリ文字列に載り、requests の例外文言
    （"... with url: /v1/chat?key=SECRET ..." など。相対パスのこともある）へそのまま
    現れる。ホスト名・パスは診断に役立つので残し、クエリ文脈（`?`/`&` 以降）・URL の
    userinfo・`Bearer <token>`・認証系ヘッダーの値だけを伏せる（いずれも文脈が明確な
    箇所のみ。任意部分文字列の置換はパス情報まで壊すので行わない）。
    """
    text = " ".join(str(exc).split())
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _URL_QUERY_RE.sub(r"\1?<redacted>", text)
    text = _SECRET_PARAM_RE.sub(r"\1<redacted>", text)
    text = _BEARER_RE.sub(r"\1<redacted>", text)
    text = _AUTH_HEADER_RE.sub(r"\1\2<redacted>", text)
    return text[:300]


def adaptive_read_timeout(base_read_timeout_s: float, max_output_tokens: int) -> float:
    """max_output_tokens に応じて読み取りタイムアウトを引き上げる。

    既定プロファイル(maximum_detail, max_output_tokens=3072)の実機11枚バッチ検証で、
    正常に生成できている呼び出しが40〜70秒かかり、一部が既定の60秒タイムアウトに
    引っかかって不要な再試行・フェイルオーバー・失敗を引き起こしていた（デバッグログ上
    "timeout (http=None)" が繰り返し記録され、その後の再試行/フェイルオーバー先も
    無料枠のレート制限で即失敗し、画像1枚あたり約2分を浪費）。
    max_output_tokens=2048 を基準点とし、それ以上は最大1.5倍まで線形に引き上げる。
    上限を1.5倍に留めているのは、NVIDIA のように本当に壊れた接続を掴んだ場合の
    無駄待ちを際限なく増やさないため（元の問題は1接続あたり最大120秒の浪費だった）。
    """
    if max_output_tokens <= 0:
        return base_read_timeout_s
    ratio = min(1.5, max(1.0, max_output_tokens / 2048.0))
    return base_read_timeout_s * ratio


def execute_http(req, *, connect_timeout: float, read_timeout: float,
                 verify_tls: bool = True) -> RawHttpResponse | VlmAttemptError:
    """1回の HTTP リクエストを実行する。ネットワーク例外は VlmAttemptError にして返す。"""
    try:
        resp = requests.request(
            req.method, req.url,
            headers=req.headers or None,
            params=req.params or None,
            json=req.json_body or None,
            timeout=(connect_timeout, read_timeout),
            verify=verify_tls,
        )
    except requests.exceptions.Timeout:
        return VlmAttemptError(VlmErrorReason.TIMEOUT, None, "request timed out")
    except requests.exceptions.SSLError as e:
        return VlmAttemptError(VlmErrorReason.NETWORK, None, f"TLS error: {_scrub_exc(e)}")
    except requests.exceptions.ConnectionError as e:
        return VlmAttemptError(VlmErrorReason.NETWORK, None, f"connection error: {_scrub_exc(e)}")
    except requests.exceptions.RequestException as e:
        return VlmAttemptError(VlmErrorReason.UNKNOWN, None, f"request failed: {_scrub_exc(e)}")

    text_body = resp.text or ""
    body: Any = None
    ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype.lower() or (text_body[:1] in ("{", "[")):
        try:
            body = resp.json()
        except (ValueError, json.JSONDecodeError):
            body = None
    return RawHttpResponse(status=resp.status_code, headers=dict(resp.headers),
                           json_body=body, text_body=text_body)


@dataclass
class ConnectionRuntime:
    """1ジョブの実行中に接続ごとに変わる状態（スナップショットではない）。"""
    excluded_reason: str = ""            # 空なら候補として生きている
    consecutive_timeouts: int = 0
    retried_same_this_image: bool = False
    same_retries_this_image: int = 0
    rate_limit: RateLimitState | None = None

    @property
    def is_excluded(self) -> bool:
        return bool(self.excluded_reason)


@dataclass
class AttemptRecord:
    connection_id: str
    model_id: str
    http_status: int | None
    error_reason: str
    error_class: str
    at_utc: float


@dataclass
class ImageResult:
    text: str | None = None
    connection_id: str = ""
    model_id: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    error: VlmAttemptError | None = None
    stopped: bool = False
    # プロンプト形式エラーなど、続行しても全画像で同じ失敗になる致命的エラー。
    # 呼び出し側（Worker）はこれを見てバッチを打ち切る（spec.md 8.1 stop_job）。
    stop_job: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.stopped and bool((self.text or "").strip())


class VlmExecutor:
    """フォールバック順に接続を試し、最初に成功したキャプションを返す。

    接続ごとの ConnectionRuntime を持ち、EXCLUDE / cooldown を次画像へ引き継ぐ。
    別モデルへの切り替えは行わない（候補は全て同一モデルプロファイル）。
    """

    def __init__(self, connections: dict[str, VlmConnection],
                 secret_resolver: Callable[[str], str | None],
                 *, stop_checker: StopChecker | None = None,
                 on_attempt_start: Callable[[VlmConnection, int], None] | None = None):
        self._connections = connections
        self._resolve_secret = secret_resolver
        self._stop = stop_checker or (lambda: False)
        self._runtime: dict[str, ConnectionRuntime] = {}
        # UI 側で「今どの接続に問い合わせているか」を示すための通知フック。1リクエストが
        # 最大 read_timeout_s（既定60秒）かかりうる上、失敗時は同一接続で再試行もするため、
        # これが無いと成功/失敗が返るまで画面が完全に無反応に見える。呼び出し側
        # （_try_connection）はこのコールバック自体の失敗で実行ループを止めないよう
        # try/except で囲んで呼ぶ。
        self._on_attempt_start = on_attempt_start

    def runtime(self, cid: str) -> ConnectionRuntime:
        return self._runtime.setdefault(cid, ConnectionRuntime())

    def live_candidates(self, ordered_ids: list[str], now: float | None = None) -> list[str]:
        now = now if now is not None else time.time()
        out = []
        for cid in ordered_ids:
            rt = self.runtime(cid)
            if rt.is_excluded:
                continue
            if rt.rate_limit is not None and rt.rate_limit.in_cooldown(now):
                continue
            out.append(cid)
        return out

    def caption_one(self, spec_base: dict, ordered_connection_ids: list[str]) -> ImageResult:
        """1画像に対しフォールバック順で試行する。

        spec_base: {"image": PreparedImage, "profile": GenerationProfile,
                    "system_prompt": str, "user_prompt": str}
        """
        result = ImageResult()
        image: PreparedImage = spec_base["image"]
        profile: GenerationProfile = spec_base["profile"]

        for cid in list(ordered_connection_ids):
            if self._stop():
                result.stopped = True
                return result
            rt = self.runtime(cid)
            if rt.is_excluded:
                continue
            if rt.rate_limit is not None and rt.rate_limit.in_cooldown():
                continue
            conn = self._connections.get(cid)
            if conn is None:
                rt.excluded_reason = "missing_connection"
                continue

            rt.retried_same_this_image = False
            rt.same_retries_this_image = 0
            outcome = self._try_connection(conn, image, profile, spec_base, result)
            if outcome == "success":
                return result
            if outcome == "stopped":
                result.stopped = True
                return result
            if outcome == "stop_job":
                # プロンプト形式エラー等。呼び出し側でジョブ停止扱いにする。
                result.stop_job = True
                return result
            # failover / exclude はループ継続

        if result.error is None:
            result.error = VlmAttemptError(VlmErrorReason.UNKNOWN, None, "all candidates exhausted")
        return result

    def _try_connection(self, conn: VlmConnection, image: PreparedImage,
                        profile: GenerationProfile, spec_base: dict, result: ImageResult) -> str:
        protocol = get_protocol(conn.protocol)
        # カスタム接続のレスポンス抽出パス上書き（fresh なインスタンスなので安全）。
        if conn.text_path:
            protocol.default_text_path = conn.text_path
        api_key = self._resolve_secret(conn.auth.secret_ref) if conn.auth.type != "none" else None
        default_key = default_auth_key(conn.auth.type, api_key)
        call = VlmCallSpec(
            model_id=conn.model_id,
            system_prompt=spec_base["system_prompt"],
            user_prompt=spec_base["user_prompt"],
            image=image,
            profile=profile,
        )
        rt = self.runtime(conn.connection_id)
        # 同一接続での再試行回数の上限。分類ロジックが必ず有限回で FAILOVER へ倒す
        # 想定だが、将来の分類変更で無限ループにならないための保険も兼ねる。
        same_conn_attempts = 0
        max_same_conn_attempts = max(0, conn.retry.retry_same_max) + 1

        while True:
            if self._stop():
                return "stopped"
            same_conn_attempts += 1
            if self._on_attempt_start is not None:
                try:
                    self._on_attempt_start(conn, same_conn_attempts)
                except Exception:
                    pass
            req = protocol.build_request(conn.base_url, default_key, call)
            apply_connection_auth(req, conn.auth.type, api_key,
                                  conn.auth.header_name, conn.auth.query_param)
            apply_request_headers(req, conn.request_headers)
            apply_request_body(req, conn.request_body)
            raw = execute_http(req, connect_timeout=conn.retry.connect_timeout_s,
                               read_timeout=adaptive_read_timeout(
                                   conn.retry.read_timeout_s, profile.max_output_tokens),
                               verify_tls=conn.verify_tls)
            if isinstance(raw, VlmAttemptError):
                parsed = VlmParseResult(error=raw)
                response_body = None
            else:
                if raw.status == 429:
                    rt.rate_limit = update_from_429(rt.rate_limit or RateLimitState(conn.connection_id), raw.headers)
                parsed = protocol.parse_response(raw.status, raw.json_body, raw.text_body)
                response_body = raw.json_body
            parsed = _enrich_parse_failure(
                parsed, protocol=protocol, body=response_body, conn=conn,
                max_output_tokens=profile.max_output_tokens)

            if parsed.ok:
                rt.consecutive_timeouts = 0
                result.error = None          # 先行接続の失敗を引きずらない
                result.text = parsed.text
                result.connection_id = conn.connection_id
                result.model_id = conn.model_id
                result.finish_reason = parsed.finish_reason
                result.prompt_tokens = parsed.prompt_tokens
                result.completion_tokens = parsed.completion_tokens
                return "success"

            err = parsed.error or VlmAttemptError(VlmErrorReason.BAD_RESPONSE, None, "no text and no error")
            if err.reason is VlmErrorReason.TIMEOUT:
                rt.consecutive_timeouts += 1
            else:
                rt.consecutive_timeouts = 0

            cls = err.classify(
                consecutive_timeouts=rt.consecutive_timeouts,
                already_retried_same=rt.retried_same_this_image,
                same_retries=rt.same_retries_this_image,
                retry_same_max=conn.retry.retry_same_max,
                retry_5xx=conn.retry.retry_5xx,
            )
            result.attempts.append(AttemptRecord(
                connection_id=conn.connection_id, model_id=conn.model_id,
                http_status=err.http_status, error_reason=err.reason.value,
                error_class=cls.value, at_utc=time.time()))
            result.error = err
            write_debug_log(
                f"vlm: {conn.connection_id} attempt -> {err.reason.value} "
                f"(http={err.http_status}) => {cls.value}")

            if cls is VlmErrorClass.RETRY_SAME and same_conn_attempts < max_same_conn_attempts:
                rt.retried_same_this_image = True
                rt.same_retries_this_image += 1
                continue
            if cls is VlmErrorClass.EXCLUDE:
                rt.excluded_reason = err.reason.value
                return "failover"
            if cls is VlmErrorClass.STOP_JOB:
                return "stop_job"
            # FAILOVER / FAIL_IMAGE(=最終候補で処理) は次候補へ
            return "failover"
