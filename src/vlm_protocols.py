"""通信形式ごとのリクエスト生成とレスポンス抽出（260901_VLM_spec.md 14章 / design.md 4.4節）。

対応形式: Google Gemini `generateContent`、OpenAI Chat Completions 互換、
OpenAI Responses API、Anthropic Messages API。
レスポンス抽出は JSONPath 全実装ではなく「ドット区切り + 配列インデックス」に限定する
（spec.md 8.5節）。カスタムテンプレートは定義済みプレースホルダー置換のみ。任意コードは
評価しない。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from vlm_connections import normalize_model_id
from vlm_errors import (
    VlmAttemptError, VlmErrorReason, reason_from_http_status, reason_from_provider_code,
)
from vlm_profiles import GenerationProfile
from vlm_scrub import scrub_text

if TYPE_CHECKING:
    from vlm_image import PreparedImage

_PLACEHOLDER_KEYS = (
    "model", "system_prompt", "user_prompt",
    "image_base64", "image_data_url", "image_mime_type",
)

_SEGMENT = re.compile(r"^([^.\[\]]*)((?:\[\d+\])*)$")


def extract_by_path(obj: Any, path: str) -> Any:
    """`choices[0].message.content` のような単純パスで値を取り出す。無ければ None。

    対応するのはドット区切りのキーと `[n]` 配列インデックスのみ（spec.md 8.5節）。
    ワイルドカードやフィルタなど JSONPath の高度な機能は実装しない。
    """
    if not path:
        return None
    cur = obj
    for seg in path.split("."):
        m = _SEGMENT.match(seg)
        if not m:
            return None
        key, idxs = m.group(1), m.group(2)
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        for idx in re.findall(r"\[(\d+)\]", idxs):
            i = int(idx)
            if not isinstance(cur, (list, tuple)) or i >= len(cur):
                return None
            cur = cur[i]
    return cur


@dataclass(frozen=True)
class VlmHttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    json_body: dict[str, Any] = field(default_factory=dict)


def default_auth_key(auth_type: str, api_key: str | None) -> str | None:
    """build_request にそのまま渡してよい鍵。bearer / none だけプロトコル既定の認証に任せ、
    header_key / query_key は下の apply_connection_auth で別に載せる。"""
    return api_key if auth_type in ("bearer", "none") else None


def apply_connection_auth(req: "VlmHttpRequest", auth_type: str, api_key: str | None,
                          header_name: str = "Authorization", query_param: str = "key") -> None:
    """header_key / query_key の認証を組み立て済みリクエストへ適用する（in-place）。

    vlm_transport（実行）と vlm_diagnostics（診断）の両方から呼び、両者で鍵の載せ方が
    ずれないようにする。frozen dataclass でも dict フィールドの中身は書き換えられる。
    """
    if not api_key:
        return
    if auth_type == "header_key":
        req.headers.pop("Authorization", None)
        req.headers[header_name or "Authorization"] = api_key
    elif auth_type == "query_key":
        req.headers.pop("Authorization", None)
        req.params[query_param or "key"] = api_key


def apply_request_headers(req: "VlmHttpRequest", headers: dict[str, str] | None) -> None:
    """秘密値ではない接続固有ヘッダーを適用する。空値は送らない。"""
    for name, value in (headers or {}).items():
        if name and value:
            req.headers[name] = value


def apply_request_body(req: "VlmHttpRequest", body: dict[str, Any] | None) -> None:
    """接続固有のJSONオプションを適用する（値はプロトコルが生成した本文へ上書き）。"""
    for name, value in (body or {}).items():
        if name:
            req.json_body[name] = value


@dataclass(frozen=True)
class VlmParseResult:
    """レスポンス解析の結果。text があれば成功、error があれば失敗（両立しない）。"""
    text: str | None = None
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: VlmAttemptError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool((self.text or "").strip())


@dataclass(frozen=True)
class VlmCallSpec:
    """1接続へ1回投げるための入力一式。"""
    model_id: str
    system_prompt: str
    user_prompt: str
    image: PreparedImage
    profile: GenerationProfile


class VlmProtocol:
    """通信形式の共通インターフェース。"""

    name: str = "base"

    def build_request(self, base_url: str, api_key: str | None, spec: VlmCallSpec) -> VlmHttpRequest:
        raise NotImplementedError

    def parse_response(self, status: int, body: Any, text_body: str) -> VlmParseResult:
        raise NotImplementedError

    # 共通ヘルパ: HTTP ステータスからの一次分類。各実装が body を見て上書きしてよい。
    @staticmethod
    def _error_from_status(status: int, text_body: str, provider_code: str = "") -> VlmAttemptError:
        # REQ-G10: 例外側（_scrub_exc）と同じマスクを通してからメッセージへ入れる。
        # 生のレスポンスボディには API キーがエコーされることがある。
        return VlmAttemptError(
            reason=reason_from_http_status(status, text_body, provider_code),
            http_status=status,
            message=scrub_text(text_body, limit=500),
            provider_code=provider_code,
        )


def _error_from_body_payload(payload: Any, http_status: int, text_body: str) -> VlmAttemptError:
    """200 応答の本文に含まれる error オブジェクトを通常の分類へ流す（REQ-G1）。"""
    code = ""
    message = ""
    if isinstance(payload, dict):
        code = str(payload.get("code") or payload.get("type") or payload.get("status") or "")
        message = str(payload.get("message") or "")
    reason = reason_from_provider_code(code, message) or VlmErrorReason.BAD_RESPONSE
    detail = scrub_text(message or text_body, limit=500)
    return VlmAttemptError(reason=reason, http_status=http_status,
                           message=detail, provider_code=code)


class OpenAIChatCompletionsProtocol(VlmProtocol):
    """OpenAI /v1/chat/completions 互換（ローカル VLM・OpenRouter 等）。"""

    name = "openai_chat_completions"
    default_text_path = "choices[0].message.content"
    system_prompt_field = "messages[0].content"
    user_prompt_field = "messages[1].content"
    max_tokens_field = "max_tokens"

    def build_request(self, base_url: str, api_key: str | None, spec: VlmCallSpec) -> VlmHttpRequest:
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        content = [
            {"type": "text", "text": spec.user_prompt},
            {"type": "image_url", "image_url": {"url": spec.image.data_url}},
        ]
        body: dict[str, Any] = {
            "model": spec.model_id,
            "messages": [
                {"role": "system", "content": spec.system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": spec.profile.max_output_tokens,
            "stream": False,
        }
        if spec.profile.temperature is not None:
            body["temperature"] = spec.profile.temperature
        if spec.profile.top_p is not None:
            body["top_p"] = spec.profile.top_p
        return VlmHttpRequest(method="POST", url=url, headers=headers, json_body=body)

    def parse_response(self, status: int, body: Any, text_body: str) -> VlmParseResult:
        if status != 200 or not isinstance(body, dict):
            code = ""
            if isinstance(body, dict):
                code = str(extract_by_path(body, "error.code")
                           or extract_by_path(body, "error.type") or "")
            err = self._error_from_status(status if status else 0, text_body, code)
            # コンテンツポリシー拒否の判定: 明示コード優先。本文の曖昧一致は 4xx のときだけ
            # （5xx 本文に "safety" 等が混ざっていても content_policy 扱いにしない）。
            is_policy = code in ("content_filter", "content_policy") or \
                (400 <= status < 500 and _looks_like_content_policy(text_body))
            if is_policy:
                err = VlmAttemptError(VlmErrorReason.CONTENT_POLICY, status or None, err.message, code)
            return VlmParseResult(error=err)

        # REQ-G1: 200 でも error フィールドを持つ応答は非200 と同じ分類へ流す。
        if body.get("error"):
            return VlmParseResult(error=_error_from_body_payload(body.get("error"), 200, text_body))
        text = extract_by_path(body, self.default_text_path)
        finish = str(extract_by_path(body, "choices[0].finish_reason") or "")
        pt = _as_int(extract_by_path(body, "usage.prompt_tokens"))
        ct = _as_int(extract_by_path(body, "usage.completion_tokens"))
        if finish == "content_filter":
            return VlmParseResult(error=VlmAttemptError(VlmErrorReason.CONTENT_POLICY, 200, "finish_reason=content_filter"))
        if not isinstance(text, str) or not text.strip():
            return VlmParseResult(error=VlmAttemptError(VlmErrorReason.EMPTY_RESPONSE, 200, "no text in choices[0].message.content"))
        return VlmParseResult(text=text.strip(), finish_reason=finish, prompt_tokens=pt, completion_tokens=ct)


class OpenAIResponsesProtocol(VlmProtocol):
    """OpenAI Responses API（画像入力対応）。"""

    name = "openai_responses"
    default_text_path = "output[0].content[0].text"
    system_prompt_field = "instructions"
    user_prompt_field = "input[0].content"
    max_tokens_field = "max_output_tokens"

    def build_request(self, base_url: str, api_key: str | None, spec: VlmCallSpec) -> VlmHttpRequest:
        url = base_url.rstrip("/") + "/responses"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body: dict[str, Any] = {
            "model": spec.model_id,
            "instructions": spec.system_prompt,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": spec.user_prompt},
                    {"type": "input_image", "image_url": spec.image.data_url},
                ],
            }],
            "max_output_tokens": spec.profile.max_output_tokens,
            # キャプション生成に会話履歴は不要。API 側への保存も行わない。
            "store": False,
        }
        if spec.profile.temperature is not None:
            body["temperature"] = spec.profile.temperature
        if spec.profile.top_p is not None:
            body["top_p"] = spec.profile.top_p
        return VlmHttpRequest(method="POST", url=url, headers=headers, json_body=body)

    def parse_response(self, status: int, body: Any, text_body: str) -> VlmParseResult:
        if status != 200 or not isinstance(body, dict):
            code = str((extract_by_path(body, "error.code") if isinstance(body, dict) else "") or "")
            err = self._error_from_status(status or 0, text_body, code)
            # openai_chat_completions と同じ判定にそろえる: 明示コードが無くても
            # 4xx 本文がポリシー拒否を示すなら AUTH_ERROR ではなく CONTENT_POLICY。
            is_policy = code in ("content_filter", "content_policy") or \
                (400 <= status < 500 and _looks_like_content_policy(text_body))
            if is_policy:
                err = VlmAttemptError(VlmErrorReason.CONTENT_POLICY, status or None, err.message, code)
            return VlmParseResult(error=err)
        # REQ-G1: Responses API は 200 + status="failed" + body["error"] を返す。
        # ここで error を読まないと実エラーコードを捨てて EMPTY_RESPONSE に潰れ、
        # rate_limit のクールダウン登録も model_not_found の EXCLUDE も働かない。
        if str(body.get("status") or "").strip().lower() == "failed" or body.get("error"):
            payload = body.get("error")
            if payload:
                return VlmParseResult(error=_error_from_body_payload(payload, 200, text_body))
        text = _openai_responses_text(body, self.default_text_path)
        status_text = str(body.get("status") or "")
        incomplete = str(extract_by_path(body, "incomplete_details.reason") or "")
        finish = incomplete or status_text
        pt = _as_int(extract_by_path(body, "usage.input_tokens"))
        ct = _as_int(extract_by_path(body, "usage.output_tokens"))
        if not text:
            return VlmParseResult(error=VlmAttemptError(
                VlmErrorReason.EMPTY_RESPONSE, 200, "no output_text in response output"))
        return VlmParseResult(text=text, finish_reason=finish,
                              prompt_tokens=pt, completion_tokens=ct)


class AnthropicMessagesProtocol(VlmProtocol):
    """Anthropic Messages API（base64 画像ブロック対応）。"""

    name = "anthropic_messages"
    default_text_path = "content[0].text"
    system_prompt_field = "system"
    user_prompt_field = "messages[0].content"
    max_tokens_field = "max_tokens"

    def build_request(self, base_url: str, api_key: str | None, spec: VlmCallSpec) -> VlmHttpRequest:
        url = base_url.rstrip("/") + "/messages"
        headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
        if api_key:
            headers["x-api-key"] = api_key
        body: dict[str, Any] = {
            "model": spec.model_id,
            "system": spec.system_prompt,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": spec.image.mime_type,
                        "data": spec.image.base64,
                    }},
                    {"type": "text", "text": spec.user_prompt},
                ],
            }],
            "max_tokens": spec.profile.max_output_tokens,
        }
        if spec.profile.temperature is not None:
            body["temperature"] = spec.profile.temperature
        if spec.profile.top_p is not None:
            body["top_p"] = spec.profile.top_p
        return VlmHttpRequest(method="POST", url=url, headers=headers, json_body=body)

    def parse_response(self, status: int, body: Any, text_body: str) -> VlmParseResult:
        if status != 200 or not isinstance(body, dict):
            code = str((extract_by_path(body, "error.type") if isinstance(body, dict) else "") or "")
            return VlmParseResult(error=self._error_from_status(status or 0, text_body, code))
        text = _anthropic_text(body, self.default_text_path)
        finish = str(body.get("stop_reason") or "")
        # REQ-G2: Anthropic の refusal は他形式の content_filter / SAFETY と同じ扱いにする。
        if finish.strip().lower() == "refusal":
            return VlmParseResult(error=VlmAttemptError(
                VlmErrorReason.CONTENT_POLICY, 200, "stop_reason=refusal"))
        pt = _as_int(extract_by_path(body, "usage.input_tokens"))
        ct = _as_int(extract_by_path(body, "usage.output_tokens"))
        if not text:
            return VlmParseResult(error=VlmAttemptError(
                VlmErrorReason.EMPTY_RESPONSE, 200, "no text in Anthropic content blocks"))
        return VlmParseResult(text=text, finish_reason=finish,
                              prompt_tokens=pt, completion_tokens=ct)


class GeminiGenerateContentProtocol(VlmProtocol):
    """Google Generative Language API `:generateContent`。"""

    name = "gemini_generate_content"
    default_text_path = "candidates[0].content.parts[0].text"
    system_prompt_field = "systemInstruction.parts[0].text"
    user_prompt_field = "contents[0].parts"
    max_tokens_field = "generationConfig.maxOutputTokens"

    def build_request(self, base_url: str, api_key: str | None, spec: VlmCallSpec) -> VlmHttpRequest:
        root = base_url.rstrip("/") if base_url else "https://generativelanguage.googleapis.com/v1beta"
        # REQ-G3: 公式表記 `models/gemini-...` を貼られても二重接頭辞にしない。
        # URL へ載せる前にセグメントとしてエスケープし、`?`／`#`／`/` でパスを壊させない。
        model_segment = quote(normalize_model_id(spec.model_id), safe="")
        url = f"{root}/models/{model_segment}:generateContent"
        headers = {"Content-Type": "application/json"}
        params: dict[str, str] = {}
        if api_key:
            # ヘッダー方式（x-goog-api-key）を優先。クエリキーは URL/ログに残りやすい。
            headers["x-goog-api-key"] = api_key
        gen_cfg: dict[str, Any] = {"maxOutputTokens": spec.profile.max_output_tokens}
        if spec.profile.temperature is not None:
            gen_cfg["temperature"] = spec.profile.temperature
        if spec.profile.top_p is not None:
            gen_cfg["topP"] = spec.profile.top_p
        body = {
            "systemInstruction": {"parts": [{"text": spec.system_prompt}]},
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": spec.user_prompt},
                    {"inlineData": {"mimeType": spec.image.mime_type, "data": spec.image.base64}},
                ],
            }],
            "generationConfig": gen_cfg,
        }
        return VlmHttpRequest(method="POST", url=url, headers=headers, params=params, json_body=body)

    def parse_response(self, status: int, body: Any, text_body: str) -> VlmParseResult:
        if status != 200 or not isinstance(body, dict):
            code = str((extract_by_path(body, "error.status") if isinstance(body, dict) else "") or "")
            return VlmParseResult(error=self._error_from_status(status or 0, text_body, code))

        block_reason = str(extract_by_path(body, "promptFeedback.blockReason") or "")
        if block_reason:
            return VlmParseResult(error=VlmAttemptError(VlmErrorReason.CONTENT_POLICY, 200, f"blockReason={block_reason}"))
        finish = str(extract_by_path(body, "candidates[0].finishReason") or "")
        if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"):
            return VlmParseResult(error=VlmAttemptError(VlmErrorReason.CONTENT_POLICY, 200, f"finishReason={finish}"))
        # A custom response path is an explicit contract.  Do not silently fall back
        # to Gemini's usual candidate-part scan, otherwise a response can appear
        # successful while the configured path is actually wrong.
        if self.default_text_path != type(self).default_text_path:
            text = extract_by_path(body, self.default_text_path)
        else:
            text = _gemini_answer_text(body) or extract_by_path(body, self.default_text_path)
        pt = _as_int(extract_by_path(body, "usageMetadata.promptTokenCount"))
        ct = _as_int(extract_by_path(body, "usageMetadata.candidatesTokenCount"))
        if not isinstance(text, str) or not text.strip():
            return VlmParseResult(error=VlmAttemptError(VlmErrorReason.EMPTY_RESPONSE, 200, "no text in candidates[0]"))
        return VlmParseResult(text=text.strip(), finish_reason=finish, prompt_tokens=pt, completion_tokens=ct)


def render_template(template: str, spec: VlmCallSpec) -> str:
    """カスタム接続の文字列テンプレートを、定義済みプレースホルダーだけで置換する。"""
    values = {
        "model": spec.model_id,
        "system_prompt": spec.system_prompt,
        "user_prompt": spec.user_prompt,
        "image_base64": spec.image.base64,
        "image_data_url": spec.image.data_url,
        "image_mime_type": spec.image.mime_type,
    }
    out = template
    for key in _PLACEHOLDER_KEYS:
        out = out.replace("{{" + key + "}}", values[key])
    return out


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _openai_responses_text(body: dict, path: str) -> str:
    if path != OpenAIResponsesProtocol.default_text_path:
        value = extract_by_path(body, path)
        return value.strip() if isinstance(value, str) else ""
    chunks: list[str] = []
    for output in body.get("output") or []:
        if not isinstance(output, dict):
            continue
        for part in output.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text" \
                    and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return "".join(chunks).strip()


def _anthropic_text(body: dict, path: str) -> str:
    if path != AnthropicMessagesProtocol.default_text_path:
        value = extract_by_path(body, path)
        return value.strip() if isinstance(value, str) else ""
    chunks = [part["text"] for part in body.get("content") or []
              if isinstance(part, dict) and part.get("type") == "text"
              and isinstance(part.get("text"), str)]
    return "".join(chunks).strip()


def _gemini_answer_text(body: Any) -> str:
    """candidates[0].content.parts から回答テキストだけを拾う。

    thinking 対応モデル（gemma-4-26b-a4b-it など）は最初の part を
    `{"text": "", "thought": true}` として返し、実際の回答は後続の part に入る。
    `parts[0].text` 固定だと空文字を掴んで「テキスト無し」になるので、thought part を
    飛ばして非 thought の text を連結する。
    """
    parts = extract_by_path(body, "candidates[0].content.parts")
    if not isinstance(parts, list):
        return ""
    chunks = [
        p["text"] for p in parts
        if isinstance(p, dict) and not p.get("thought") and isinstance(p.get("text"), str)
    ]
    return "".join(chunks).strip()


def _looks_like_content_policy(text_body: str) -> bool:
    low = (text_body or "").lower()
    return any(s in low for s in ("content policy", "safety", "content_filter", "flagged", "prohibited"))


PROTOCOLS: dict[str, type[VlmProtocol]] = {
    OpenAIChatCompletionsProtocol.name: OpenAIChatCompletionsProtocol,
    OpenAIResponsesProtocol.name: OpenAIResponsesProtocol,
    AnthropicMessagesProtocol.name: AnthropicMessagesProtocol,
    GeminiGenerateContentProtocol.name: GeminiGenerateContentProtocol,
}


def get_protocol(name: str) -> VlmProtocol:
    cls = PROTOCOLS.get(name, OpenAIChatCompletionsProtocol)
    return cls()
