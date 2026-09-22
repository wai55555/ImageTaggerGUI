"""内蔵接続とカスタム接続の定義（260901_VLM_spec.md 2.2・14章 / design.md 4.2・5.2節）。

- 内蔵接続: アプリが URL・プロトコル・プロバイダー固有の接続情報を持つ
  （Gemini / OpenRouter / Cloudflare / Groq / NVIDIA / Hugging Face /
   Vercel AI Gateway / OpenAI / Anthropic）
  ※ Mistral/Pixtral はキャプション用途として弱いため、内蔵経路をコメントアウト中。
- カスタム接続: 利用者が登録する外部 API / ローカル VLM。同一モデル判定は行わない。
  外部・ローカルの判定は安全側（不明なら外部扱い）。
"""
from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse


class ConnectionKind(str, Enum):
    BUILTIN = "builtin"
    CUSTOM_EXTERNAL = "custom_external"
    CUSTOM_LOCAL = "custom_local"


class ConnectionLocality(str, Enum):
    """カスタム接続編集画面での「接続先」選択。AUTO は URL から推測。"""
    AUTO = "auto"
    LOCAL = "local"
    EXTERNAL = "external"


_LOCAL_HOSTNAMES = {"localhost", "localhost.localdomain", "host.docker.internal"}
_PRIVATE_NETWORKS = tuple(ipaddress.ip_network(cidr) for cidr in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7",
))


def is_local_host(host: str) -> bool:
    """Return whether *host* names a loopback/private/link-local destination.

    Parse numeric addresses instead of relying on string prefixes: the latter both
    missed ranges such as 172.16/12 and treated names such as ``10.example.com`` as
    private.  ``.local`` remains supported for mDNS hosts used by LAN VLM servers.
    """
    normalized = (host or "").strip().rstrip(".").lower()
    if not normalized:
        return False
    if normalized in _LOCAL_HOSTNAMES or normalized.endswith(".local"):
        return True
    # urlparse removes IPv6 brackets but may leave a zone identifier (fe80::1%eth0).
    address = normalized.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return (ip.is_loopback or ip.is_link_local or ip.is_unspecified
            or any(ip.version == network.version and ip in network
                   for network in _PRIVATE_NETWORKS))


def _looks_local(base_url: str) -> bool:
    try:
        host = urlparse(base_url or "").hostname or ""
    except ValueError:
        return False
    return is_local_host(host)


def resolve_custom_kind(locality: ConnectionLocality, base_url: str) -> ConnectionKind:
    """接続先種別を確定する。AUTO かつローカルと判定できなければ外部扱い（安全側）。"""
    if locality is ConnectionLocality.LOCAL:
        return ConnectionKind.CUSTOM_LOCAL
    if locality is ConnectionLocality.EXTERNAL:
        return ConnectionKind.CUSTOM_EXTERNAL
    return ConnectionKind.CUSTOM_LOCAL if _looks_local(base_url) else ConnectionKind.CUSTOM_EXTERNAL


@dataclass
class RetryPolicy:
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0
    retry_same_max: int = 1
    retry_5xx: bool = True
    use_retry_after_on_429: bool = False   # spec: 429 は待たず failover

    @classmethod
    def from_mapping(cls, data: dict | None) -> "RetryPolicy":
        d = data or {}
        base = cls()
        return cls(
            connect_timeout_s=_timeout(d.get("connect_timeout_s"), base.connect_timeout_s),
            read_timeout_s=_timeout(d.get("read_timeout_s"), base.read_timeout_s),
            retry_same_max=_i(d.get("retry_same_max"), base.retry_same_max),
            retry_5xx=_b(d.get("retry_5xx"), base.retry_5xx),
            use_retry_after_on_429=_b(d.get("use_retry_after_on_429"), base.use_retry_after_on_429),
        )


@dataclass
class AuthSpec:
    """認証方式。秘密値そのものは持たず、`secret_ref` で vlm_secrets へ問い合わせる。"""
    type: str = "none"          # none | bearer | header_key | query_key
    secret_ref: str = ""        # keyring / env のキー名
    header_name: str = "Authorization"
    query_param: str = "key"

    @classmethod
    def from_mapping(cls, data: dict | None) -> "AuthSpec":
        d = data or {}
        base = cls()
        return cls(
            type=str(d.get("type", base.type) or base.type).lower(),
            secret_ref=str(d.get("secret_ref", "") or ""),
            header_name=str(d.get("header_name", base.header_name) or base.header_name),
            query_param=str(d.get("query_param", base.query_param) or base.query_param),
        )


@dataclass
class VlmConnection:
    connection_id: str
    display_name: str
    kind: ConnectionKind
    protocol: str                       # vlm_protocols.PROTOCOLS のキー
    base_url: str
    model_id: str
    provider_id: str = ""               # 内蔵の場合のみ（gemini / openrouter / cloudflare）
    enabled: bool = True
    auth: AuthSpec = field(default_factory=AuthSpec)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    verify_tls: bool = True
    concurrency: int = 1
    # レスポンス抽出パスの上書き（空ならプロトコル既定）。
    text_path: str = ""
    error_path: str = ""
    # APIキー以外のプロバイダー必須ヘッダー。現在は複数Workspace対象のAnthropic
    # personal/service-account keyで使う anthropic-workspace-id を保持する。
    request_headers: dict[str, str] = field(default_factory=dict)
    # プロバイダー固有のJSONオプション。Cloudflare Gemma 4では推論を無効化する。
    request_body: dict[str, Any] = field(default_factory=dict)
    # カスタム接続だけに適用する画像前処理上限。None は生成プロファイルの既定値。
    image_max_long_edge: int | None = None

    @property
    def is_custom(self) -> bool:
        return self.kind in (ConnectionKind.CUSTOM_EXTERNAL, ConnectionKind.CUSTOM_LOCAL)

    @property
    def is_local(self) -> bool:
        return self.kind is ConnectionKind.CUSTOM_LOCAL

    @classmethod
    def from_mapping(cls, data: dict) -> "VlmConnection":
        kind_raw = str(data.get("kind", "custom_external")).lower()
        try:
            kind = ConnectionKind(kind_raw)
        except ValueError:
            kind = ConnectionKind.CUSTOM_EXTERNAL
        raw_auth = data.get("auth")
        raw_retry = data.get("retry")
        raw_response = data.get("response")
        raw_headers = data.get("request_headers")
        raw_image = data.get("image")
        image_max = (raw_image.get("max_long_edge")
                     if isinstance(raw_image, dict) else data.get("image_max_long_edge"))
        parsed_image_max = _i(image_max, 0)
        return cls(
            connection_id=str(data["connection_id"]),
            display_name=str(data.get("display_name", data["connection_id"])),
            kind=kind,
            protocol=str(data.get("protocol", "openai_chat_completions")),
            base_url=str(data.get("base_url", "")),
            model_id=str(data.get("model_id", "")),
            provider_id=str(data.get("provider_id", "")),
            enabled=_b(data.get("enabled"), True),
            auth=AuthSpec.from_mapping(raw_auth if isinstance(raw_auth, dict) else None),
            retry=RetryPolicy.from_mapping(raw_retry if isinstance(raw_retry, dict) else None),
            verify_tls=_b(data.get("verify_tls"), True),
            concurrency=max(1, _i(data.get("concurrency"), 1)),
            text_path=str(raw_response.get("text_path", "") if isinstance(raw_response, dict) else data.get("text_path", "")),
            error_path=str(raw_response.get("error_path", "") if isinstance(raw_response, dict) else data.get("error_path", "")),
            request_headers={str(k): str(v) for k, v in raw_headers.items()
                             if str(k).strip() and v is not None}
            if isinstance(raw_headers, dict) else {},
            request_body=dict(data.get("request_body") or {}) if isinstance(data.get("request_body"), dict) else {},
            image_max_long_edge=max(256, min(8192, parsed_image_max))
            if parsed_image_max > 0 else None,
        )


def _f(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return default


def _timeout(v, default: float) -> float:
    """接続タイムアウト秒。非有限・非正値は requests が ValueError を投げ、
    execute_http のハンドラ外で落ちるため、ここで安全な範囲へ丸める。"""
    try:
        n = float(v)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(n) or n <= 0.0:
        return default
    return min(n, 900.0)


def _i(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError, OverflowError):
        return default


def _b(v, default: bool) -> bool:
    """JSON 由来の真偽値。実際の bool はそのまま、文字列は明示的な語のみ解釈し、
    それ以外（"false" を True と誤認する bool() の挙動含む）は既定値へ倒す。"""
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    return default


# --- 初期内蔵接続（spec.md 3.1・3.2節） ------------------------------------------------
# base_url と model_id 以外（APIキー等）は起動時にユーザー設定・秘密ストレージから補う。
BUILTIN_CONNECTION_TEMPLATES: list[dict] = [
    {
        "connection_id": "builtin-gemini",
        "display_name": "Gemini API",
        "kind": "builtin",
        "provider_id": "gemini",
        "protocol": "gemini_generate_content",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model_id": "gemma-4-26b-a4b-it",
        "auth": {"type": "header_key", "header_name": "x-goog-api-key", "secret_ref": "vlm/gemini/api_key"},
    },
    {
        "connection_id": "builtin-openrouter",
        "display_name": "OpenRouter",
        "kind": "builtin",
        "provider_id": "openrouter",
        "protocol": "openai_chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "model_id": "google/gemma-4-26b-a4b-it:free",
        "auth": {"type": "bearer", "secret_ref": "vlm/openrouter/api_key"},
    },
    {
        "connection_id": "builtin-cloudflare",
        "display_name": "Cloudflare",
        "kind": "builtin",
        "provider_id": "cloudflare",
        "protocol": "openai_chat_completions",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "model_id": "@cf/google/gemma-4-26b-a4b-it",
        "auth": {"type": "bearer", "secret_ref": "vlm/cloudflare/api_token"},
        # Gemma 4はReasoning対応。診断用の短いmax_tokensでも思考だけで上限に
        # 達して本文が出ないため、キャプション用途では推論を無効化する。
        "request_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    # --- implement_plan 2.2「後から追加する内蔵候補」。model_id は選択プロファイルの
    # binding から埋める（binding が無ければ build_connection_map が無効化する）。
    {
        "connection_id": "builtin-groq",
        "display_name": "Groq",
        "kind": "builtin",
        "provider_id": "groq",
        "protocol": "openai_chat_completions",
        "base_url": "https://api.groq.com/openai/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/groq/api_key"},
    },
    {
        "connection_id": "builtin-nvidia",
        "display_name": "NVIDIA NIM",
        "kind": "builtin",
        "provider_id": "nvidia",
        "protocol": "openai_chat_completions",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/nvidia/api_key"},
    },
    # Mistral/Pixtral VLM は対応しているが、キャプション経路としては弱いため
    # 内蔵プロバイダーから一時的にコメントアウト。必要になればこのブロックを戻す。
    # {
    #     "connection_id": "builtin-mistral",
    #     "display_name": "Mistral",
    #     "kind": "builtin",
    #     "provider_id": "mistral",
    #     "protocol": "openai_chat_completions",
    #     "base_url": "https://api.mistral.ai/v1",
    #     "model_id": "",
    #     "auth": {"type": "bearer", "secret_ref": "vlm/mistral/api_key"},
    # },
    {
        "connection_id": "builtin-huggingface",
        "display_name": "Hugging Face",
        "kind": "builtin",
        "provider_id": "huggingface",
        "protocol": "openai_chat_completions",
        "base_url": "https://router.huggingface.co/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/huggingface/api_token"},
        # Monthly credits exist, but routed inference is metered and can consume paid credits.
    },
    {
        "connection_id": "builtin-vercel",
        "display_name": "Vercel AI Gateway",
        "kind": "builtin",
        "provider_id": "vercel",
        "protocol": "openai_chat_completions",
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/vercel/api_key"},
        # 無料クレジットが付く場合もあるが、経路自体は従量課金。
    },
    {
        "connection_id": "builtin-openai",
        "display_name": "OpenAI",
        "kind": "builtin",
        "provider_id": "openai",
        "protocol": "openai_responses",
        "base_url": "https://api.openai.com/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/openai/api_key"},
    },
    {
        "connection_id": "builtin-anthropic",
        "display_name": "Anthropic Claude",
        "kind": "builtin",
        "provider_id": "anthropic",
        "protocol": "anthropic_messages",
        "base_url": "https://api.anthropic.com/v1",
        "model_id": "",
        "auth": {"type": "header_key", "header_name": "x-api-key",
                 "secret_ref": "vlm/anthropic/api_key"},
    },
    {
        # xAI は 2026 時点で /v1/chat/completions を非推奨化し Responses API へ寄せている。
        # OpenAI Responses と同形（instructions / input[].content の input_text・
        # input_image(文字列 data URL) / max_output_tokens / output[0].content[0].text）。
        "connection_id": "builtin-xai",
        "display_name": "xAI Grok",
        "kind": "builtin",
        "provider_id": "xai",
        "protocol": "openai_responses",
        "base_url": "https://api.x.ai/v1",
        "model_id": "",
        "auth": {"type": "bearer", "secret_ref": "vlm/xai/api_key"},
    },
    # OVHcloud は日本居住者によるアカウント作成・実機検証ができなかったため無効化。
    # 対応地域の利用者が接続確認できるまで、内蔵経路として UI へ公開しない。
    # {
    #     "connection_id": "builtin-ovhcloud",
    #     "display_name": "OVHcloud AI Endpoints",
    #     "kind": "builtin",
    #     "provider_id": "ovhcloud",
    #     "protocol": "openai_chat_completions",
    #     "base_url": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    #     "model_id": "",
    #     "auth": {"type": "bearer", "secret_ref": "vlm/ovhcloud/api_key"},
    # },
]


def default_builtin_connections() -> list[VlmConnection]:
    return [VlmConnection.from_mapping(t) for t in BUILTIN_CONNECTION_TEMPLATES]
