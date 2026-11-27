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

from urllib.parse import urlparse

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
from vlm_scrub import AUTH_HEADER_NAMES, scrub_exception

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


def suggested_max_output_tokens(current: int | None) -> int:
    """OUTPUT_LIMIT で助言する上限値（REQ-G14）。

    既に既定より大きい値を設定している利用者へ「3072 へ増やせ」と言わないよう、
    現在値より必ず大きい値を返す（既定以下なら従来どおり既定値）。
    """
    base = _SUGGESTED_MAX_OUTPUT_TOKENS
    try:
        current_value = int(current or 0)
    except (TypeError, ValueError):
        return base
    if current_value > base:
        return max(current_value * 2, base)
    return base


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
        # REQ-G14: 現在値より大きい値だけを助言する（8192 設定済みに 3072 を勧めない）。
        suggested = suggested_max_output_tokens(max_output_tokens)
        return replace(parsed, error=VlmAttemptError(
            VlmErrorReason.OUTPUT_LIMIT, error.http_status,
            "output limit reached: max_output_tokens={} finish_reason={}; "
            "the model returned no final text. Increase VLM max tokens to {} or higher "
            "for reasoning VLMs.".format(max_output_tokens, limit_reason, suggested),
            error.provider_code,
            message_key="Advice_Output_Limit",
            message_args={"limit": max_output_tokens, "finish_reason": limit_reason,
                          "suggested": suggested}))

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


# 伏せ字処理は vlm_protocols（エラーボディ）と共有するため vlm_scrub に置いた。
# 既存の呼び出し・テスト互換のため、この名前は残す。
def _scrub_exc(exc: Exception) -> str:
    """例外のテキストから資格情報を落とす（実装は vlm_scrub.scrub_text）。"""
    return scrub_exception(exc)


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


class GuardedSession(requests.Session):
    """別ホストへのリダイレクトで認証ヘッダーを落とす Session（REQ-G11）。

    requests が自動で落とすのは Authorization だけなので、header_key 系
    （x-api-key 等）も同等に扱う。`sensitive_headers` はリクエスト直前に
    execute_http が設定する（Session は 1 スレッド内での逐次利用を前提とする）。
    """

    sensitive_headers: frozenset[str] = frozenset()

    def rebuild_auth(self, prepared_request, response) -> None:  # type: ignore[override]
        super().rebuild_auth(prepared_request, response)
        previous = response.request
        if previous is None:
            return
        old = urlparse(previous.url)
        new = urlparse(prepared_request.url)
        if (old.scheme, old.hostname, old.port) == (new.scheme, new.hostname, new.port):
            return
        for name in list(prepared_request.headers):
            if name.lower() in self.sensitive_headers:
                prepared_request.headers.pop(name, None)


def execute_http(req, *, connect_timeout: float, read_timeout: float,
                 verify_tls: bool = True,
                 session: requests.Session | None = None,
                 sensitive_headers: tuple[str, ...] = (),
                 total_timeout_s: float | None = None
                 ) -> RawHttpResponse | VlmAttemptError:
    """1回の HTTP リクエストを実行する。ネットワーク例外は VlmAttemptError にして返す。

    - `session` を渡すと接続（TLS/keep-alive）を再利用する（REQ-G7）。
    - `sensitive_headers` は別ホストへのリダイレクト時に落とす認証ヘッダー名（REQ-G11）。
    - `total_timeout_s` は受信全体の締切（REQ-G6）。read タイムアウトは recv 単位で
      総量ではないため、少しずつ送り続ける相手では終わらない。None なら従来どおり
      単発の read タイムアウトに任せる（生成経路はこちら）。
    """
    client = session if session is not None else requests
    if session is not None:
        try:
            session.sensitive_headers = frozenset(  # type: ignore[attr-defined]
                {name.lower() for name in AUTH_HEADER_NAMES}
                | {name.lower() for name in sensitive_headers if name})
        except AttributeError:  # 独自 Session 実装は属性を持てないことがある
            pass

    started = time.monotonic()
    timeout: tuple[float, float] | float
    if total_timeout_s is None:
        timeout = (connect_timeout, read_timeout)
    else:
        # 全体締切を read タイムアウトの上限としても使い、まず単発の受信で切る。
        timeout = (connect_timeout, max(0.001, min(read_timeout, total_timeout_s)))
    try:
        if total_timeout_s is None:
            resp = client.request(
                req.method, req.url,
                headers=req.headers or None,
                params=req.params or None,
                json=req.json_body or None,
                timeout=timeout,
                verify=verify_tls,
            )
        else:
            resp = client.request(
                req.method, req.url,
                headers=req.headers or None,
                params=req.params or None,
                json=req.json_body or None,
                timeout=timeout,
                verify=verify_tls,
                stream=True,
            )
    except requests.exceptions.Timeout:
        return VlmAttemptError(VlmErrorReason.TIMEOUT, None, "request timed out")
    except requests.exceptions.SSLError as e:
        return VlmAttemptError(VlmErrorReason.NETWORK, None, f"TLS error: {_scrub_exc(e)}")
    except requests.exceptions.ConnectionError as e:
        return VlmAttemptError(VlmErrorReason.NETWORK, None, f"connection error: {_scrub_exc(e)}")
    except requests.exceptions.RequestException as e:
        return VlmAttemptError(VlmErrorReason.UNKNOWN, None, f"request failed: {_scrub_exc(e)}")

    if total_timeout_s is not None:
        chunks: list[bytes] = []
        try:
            # チャンクを小さく保つ: chunk_size が大きいと「少しずつ送る相手」では
            # 次のチャンクが満たされるまで read が返らず、締切の判定が遅れる。
            for chunk in resp.iter_content(chunk_size=512):
                if chunk:
                    chunks.append(chunk)
                if time.monotonic() - started > total_timeout_s:
                    resp.close()
                    return VlmAttemptError(
                        VlmErrorReason.TIMEOUT, None,
                        f"total deadline {total_timeout_s:.0f}s exceeded while receiving")
        except requests.exceptions.Timeout:
            return VlmAttemptError(VlmErrorReason.TIMEOUT, None, "request timed out")
        except requests.exceptions.RequestException as e:
            return VlmAttemptError(VlmErrorReason.NETWORK, None,
                                   f"receive error: {_scrub_exc(e)}")
        raw_bytes = b"".join(chunks)
        encoding = resp.encoding or "utf-8"
        text_body = raw_bytes.decode(encoding, errors="replace")
    else:
        text_body = resp.text or ""

    body: Any = None
    ctype = resp.headers.get("Content-Type", "")
    if "json" in ctype.lower() or (text_body[:1] in ("{", "[")):
        try:
            body = json.loads(text_body) if total_timeout_s is not None else resp.json()
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


def _sensitive_header_names(conn: VlmConnection, default_key: str | None) -> tuple[str, ...]:
    """リダイレクト時に落とすべき認証ヘッダー名（REQ-G11）。"""
    names: list[str] = []
    if default_key:
        names.append("Authorization")
    if conn.auth.type in ("header_key", "query_key") and conn.auth.header_name:
        names.append(conn.auth.header_name)
    return tuple(names)


class VlmExecutor:
    """フォールバック順に接続を試し、最初に成功したキャプションを返す。

    接続ごとの ConnectionRuntime を持ち、EXCLUDE / cooldown を次画像へ引き継ぐ。
    別モデルへの切り替えは行わない（候補は全て同一モデルプロファイル）。
    """

    def __init__(self, connections: dict[str, VlmConnection],
                 secret_resolver: Callable[[str], str | None],
                 *, stop_checker: StopChecker | None = None,
                 on_attempt_start: Callable[[VlmConnection, int], None] | None = None,
                 sleep: Callable[[float], None] | None = None,
                 clock: Callable[[], float] | None = None):
        self._connections = connections
        self._resolve_secret = secret_resolver
        self._stop = stop_checker or (lambda: False)
        # REQ-G7: 接続（TLS/keep-alive）を試行ごとに張り直さず Session を使い回す。
        self._session: GuardedSession | None = None
        # REQ-G7: API キーは試行のまとまり（executor の寿命）ごとに 1 回だけ解決する。
        self._secret_cache: dict[str, str | None] = {}
        # REQ-G8: 同一接続リトライの待機。テストでは注入して実時間を待たない。
        self._sleep = sleep
        self._clock = clock or time.monotonic
        self._runtime: dict[str, ConnectionRuntime] = {}
        # UI 側で「今どの接続に問い合わせているか」を示すための通知フック。1リクエストが
        # 最大 read_timeout_s（既定60秒）かかりうる上、失敗時は同一接続で再試行もするため、
        # これが無いと成功/失敗が返るまで画面が完全に無反応に見える。呼び出し側
        # （_try_connection）はこのコールバック自体の失敗で実行ループを止めないよう
        # try/except で囲んで呼ぶ。
        self._on_attempt_start = on_attempt_start

    def close(self) -> None:
        """Session を閉じる（接続の解放）。ジョブ終了時に呼ぶ。"""
        session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - 解放失敗でジョブ結果を壊さない
                pass

    def _session_for_requests(self) -> GuardedSession:
        if self._session is None:
            self._session = GuardedSession()
        return self._session

    def _secret(self, secret_ref: str) -> str | None:
        """secret_ref ごとに 1 回だけ解決する（REQ-G7）。"""
        if not secret_ref:
            return None
        if secret_ref not in self._secret_cache:
            self._secret_cache[secret_ref] = self._resolve_secret(secret_ref)
        return self._secret_cache[secret_ref]

    def _wait_before_retry(self, seconds: float) -> bool:
        """RETRY_SAME のバックオフ待機（REQ-G8）。停止要求で False を返す。"""
        if seconds <= 0:
            return not self._stop()
        if self._sleep is not None:
            self._sleep(seconds)
            return not self._stop()
        # 実時間待機でも停止ボタンへ即応できるよう細かく分割する。
        deadline = self._clock() + seconds
        while True:
            if self._stop():
                return False
            remaining = deadline - self._clock()
            if remaining <= 0:
                return True
            time.sleep(min(0.1, remaining))

    def runtime(self, cid: str) -> ConnectionRuntime:
        return self._runtime.setdefault(cid, ConnectionRuntime())

    def live_candidates(self, ordered_ids: list[str],
                        now_mono: float | None = None) -> list[str]:
        # REQ-G5: クールダウン判定は単調時計。壁時計を渡すと期限が常に過去になり
        # フィルタが効かない（比較側だけ差し替えると壊れる典型）。
        now_mono = now_mono if now_mono is not None else self._clock()
        out = []
        for cid in ordered_ids:
            rt = self.runtime(cid)
            if rt.is_excluded:
                continue
            if rt.rate_limit is not None and rt.rate_limit.in_cooldown(now_mono):
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
        api_key = self._secret(conn.auth.secret_ref) if conn.auth.type != "none" else None
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
            session = self._session_for_requests()
            raw = execute_http(req, connect_timeout=conn.retry.connect_timeout_s,
                               read_timeout=adaptive_read_timeout(
                                   conn.retry.read_timeout_s, profile.max_output_tokens),
                               verify_tls=conn.verify_tls,
                               session=session,
                               sensitive_headers=_sensitive_header_names(conn, default_key))
            if isinstance(raw, VlmAttemptError):
                parsed = VlmParseResult(error=raw)
                response_body = None
            else:
                parsed = protocol.parse_response(raw.status, raw.json_body, raw.text_body)
                response_body = raw.json_body
            parsed = _enrich_parse_failure(
                parsed, protocol=protocol, body=response_body, conn=conn,
                max_output_tokens=profile.max_output_tokens)
            # REQ-G1: 200 の本文が rate limit を示す場合も、非200(429) と同じく
            # クールダウンへ登録する（ヘッダーは 200 応答にも付く）。
            if (isinstance(raw, RawHttpResponse)
                    and parsed.error is not None
                    and parsed.error.reason is VlmErrorReason.RATE_LIMITED):
                rt.rate_limit = update_from_429(
                    rt.rate_limit or RateLimitState(conn.connection_id), raw.headers)

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
                # REQ-G8: 同一接続リトライは即時再試行ではなく最小のバックオフを挟む。
                # クールダウン登録（429）とは独立の、同接続リトライ用の待機。
                base = max(0.0, float(getattr(conn.retry, "retry_same_backoff_s", 0.0) or 0.0))
                delay = base * (2 ** min(rt.same_retries_this_image, 6))
                rt.retried_same_this_image = True
                rt.same_retries_this_image += 1
                if delay > 0 and not self._wait_before_retry(delay):
                    return "stopped"
                continue
            if cls is VlmErrorClass.EXCLUDE:
                rt.excluded_reason = err.reason.value
                return "failover"
            if cls is VlmErrorClass.STOP_JOB:
                return "stop_job"
            # FAILOVER / FAIL_IMAGE(=最終候補で処理) は次候補へ
            return "failover"
