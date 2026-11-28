"""プロバイダーのモデル一覧と、画像入力可否を取得する層。

一覧 API の返す能力メタデータを捨てずに ``ModelCatalogEntry`` として UI へ渡す。
API が能力を返さないプロバイダーについては ``vlm_models`` の公式カタログ由来の
保守的な判定へフォールバックする。UI スレッドからは呼ばない（ワーカー経由）。
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from vlm_connections import VlmConnection
from vlm_errors import VlmAttemptError, VlmErrorReason
from vlm_models import classify_model_capability, _metadata_modalities
from vlm_protocols import (
    VlmHttpRequest, apply_connection_auth, apply_request_headers, default_auth_key,
)
from vlm_transport import RawHttpResponse, execute_http


# Mistral/PixtralはVLM対応でもキャプション品質が弱いため、内蔵プロバイダーの
# 動的モデル一覧からも選択肢に出さない。手入力のカスタム接続までは妨げない。
_DISABLED_BUILTIN_VLM_MARKERS = ("mistral", "pixtral")

# REQ-G6 / U-G1: モデルカタログ取得の総量締切（秒）。
# 実測（後述の tests/test_vlm_phase_g.py::test_ug1_*）ではローカル往復のカタログ取得が
# 0.1 秒未満で完了する。公衆回線・大型カタログ（Cloudflare per_page=100 等）でも
# 日常の取得を誤って切らないよう、RetryPolicy 既定の read_timeout（60 秒）を下限とした。
CATALOG_TOTAL_DEADLINE_S = 60.0


def is_cloudflare_connection(conn: VlmConnection, base_url: str | None = None) -> bool:
    """Cloudflare Workers AI の接続か（REQ-G15: 実経路と診断で同じ述語を使う）。"""
    if str(getattr(conn, "provider_id", "") or "") == "cloudflare":
        return True
    base = base_url if base_url is not None else (conn.base_url or "")
    try:
        host = urlparse(base).hostname or ""
    except ValueError:
        return False
    return host == "api.cloudflare.com" or host.endswith(".api.cloudflare.com")


def model_list_endpoint(conn: VlmConnection) -> tuple[str, dict[str, str]]:
    """モデル一覧の URL とクエリを返す（REQ-G15: 実経路・診断の共有定義）。

    - Cloudflare Workers AI: `/ai/v1` を `/ai` に落として `/models/search`
    - xAI: `/language-models`（`/models` は能力メタデータを返さない）
    - それ以外: `/models`
    """
    base = (conn.base_url or "").rstrip("/")
    params: dict[str, str] = {}
    if is_cloudflare_connection(conn) and base.endswith("/ai/v1"):
        return base[: -len("/v1")] + "/models/search", {"per_page": "100"}
    if str(getattr(conn, "provider_id", "") or "") == "xai":
        return f"{base}/language-models", params
    return f"{base}/models", params


@dataclass(frozen=True)
class ModelCatalogEntry:
    """モデル一覧の1行と、キャプション用途に必要な能力判定。"""

    model_id: str
    supports_image_input: bool | None = None
    supports_text_output: bool | None = None
    capability_source: str = ""

    @property
    def is_vlm(self) -> bool:
        """画像を入力し、テキストを返せることが確認できたモデルか。"""
        return self.supports_image_input is True and self.supports_text_output is not False


def fetch_model_catalog(conn: VlmConnection, api_key: str | None,
                       *, connect_timeout: float = 8.0, read_timeout: float = 20.0
                       ) -> list[ModelCatalogEntry] | VlmAttemptError:
    """接続先のモデル一覧を取得し、各モデルのVLM可否を付けて返す。"""
    raw = _fetch_model_body(conn, api_key, connect_timeout=connect_timeout,
                            read_timeout=read_timeout)
    if isinstance(raw, VlmAttemptError):
        return raw
    entries = _extract_catalog(raw, conn.provider_id)
    if not entries:
        return VlmAttemptError(VlmErrorReason.BAD_RESPONSE, 200,
                               "no model ids in response",
                               message_key="ModelList_No_Model_Ids")
    seen: set[str] = set()
    out: list[ModelCatalogEntry] = []
    for entry in entries:
        if entry.model_id and entry.model_id not in seen:
            seen.add(entry.model_id)
            out.append(entry)
    return out


def _fetch_model_body(conn: VlmConnection, api_key: str | None,
                      *, connect_timeout: float, read_timeout: float):
    base = (conn.base_url or "").rstrip("/")
    if not base or "{account_id}" in base:
        return VlmAttemptError(VlmErrorReason.UNKNOWN, None,
                               "connection base url is not ready (Cloudflare account id?)",
                               message_key="ModelList_Base_Url_Not_Ready")
    # Cloudflare の検索 API はデフォルトページサイズが小さいため、現行カタログを
    # 1回で取りこぼさない値を指定する（共有定義側で設定）。API が上限を下げても安全。
    url, params = model_list_endpoint(conn)
    req = VlmHttpRequest(method="GET", url=url, headers={}, params=params, json_body={})
    if conn.protocol == "anthropic_messages":
        req.headers["anthropic-version"] = "2023-06-01"
    default_key = default_auth_key(conn.auth.type, api_key)
    if default_key:
        req.headers["Authorization"] = f"Bearer {default_key}"
    apply_connection_auth(req, conn.auth.type, api_key, conn.auth.header_name, conn.auth.query_param)
    apply_request_headers(req, conn.request_headers)

    # REQ-G6: read タイムアウトは recv 単位で総量ではない。遅滴ピアで終わらないため、
    # カタログ取得には総量の締切を渡す（生成経路の execute_http は対象外）。
    raw = execute_http(req, connect_timeout=connect_timeout, read_timeout=read_timeout,
                       verify_tls=conn.verify_tls,
                       total_timeout_s=CATALOG_TOTAL_DEADLINE_S)
    if not isinstance(raw, RawHttpResponse):
        return raw
    if raw.status in (401, 403):
        return VlmAttemptError(VlmErrorReason.AUTH_ERROR, raw.status, f"{raw.status} auth rejected")
    if raw.status != 200 or not isinstance(raw.json_body, (dict, list)):
        return VlmAttemptError(VlmErrorReason.BAD_RESPONSE, raw.status,
                               f"HTTP {raw.status} (model list unavailable)",
                               message_key="ModelList_Unavailable",
                               message_args={"status": raw.status})
    return raw.json_body


def _entry_from_row(provider_id: str, model_id: str, row: dict) -> ModelCatalogEntry:
    inputs, outputs = _metadata_modalities(row)
    is_vlm, source = classify_model_capability(provider_id, model_id, row)
    # 能力を返す一覧では出力も明示的に保持し、画像生成など「画像を扱うが
    # キャプションを返さない」モデルをVLM一覧へ混ぜない。
    text_output = ("text" in outputs) if outputs is not None else (True if is_vlm else None)
    # metadata に画像入力があっても、description/task が画像生成・安全性モデルと
    # 明示する場合は caption VLM として扱わない（is_vlm=False を優先）。
    image_input = (("image" in inputs) and is_vlm is not False) if inputs is not None else is_vlm
    return ModelCatalogEntry(model_id=model_id, supports_image_input=image_input,
                             supports_text_output=text_output,
                             capability_source=source)


def _extract_catalog(body, provider_id: str) -> list[ModelCatalogEntry]:
    """プロバイダー別レスポンスを能力付きの共通行へ変換する。"""
    rows = body.get("data") if isinstance(body, dict) else body
    if isinstance(rows, list) and any(isinstance(row, dict) and "id" in row for row in rows):
        # llama.cpp は OpenAI 形式の data[].id と、能力情報を含む互換形式の
        # models[].model/capabilities を併せて返す。IDをキーに能力情報を補う。
        extras = {}
        model_rows = body.get("models") if isinstance(body, dict) else None
        if isinstance(model_rows, list):
            for extra in model_rows:
                if not isinstance(extra, dict):
                    continue
                extra_id = str(extra.get("id") or extra.get("model") or extra.get("name") or "").strip()
                if extra_id:
                    extras[extra_id] = extra
        out = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            mid = str(row["id"]).strip()
            if mid:
                metadata = dict(row)
                for key, value in extras.get(mid, {}).items():
                    metadata.setdefault(key, value)
                out.append(_entry_from_row(provider_id, mid, metadata))
        return out

    # xAI /v1/language-models: {"models": [{"id": "grok-4.6",
    #   "input_modalities": ["text","image"], "output_modalities": ["text"], ...}]}
    # （Gemini の models[].name とは別形式。id を持つ行だけをここで拾う。）
    id_rows = body.get("models") if isinstance(body, dict) else None
    if isinstance(id_rows, list) and any(
            isinstance(row, dict) and str(row.get("id", "")).strip() for row in id_rows):
        out = []
        for row in id_rows:
            if not isinstance(row, dict):
                continue
            mid = str(row.get("id", "")).strip()
            if mid:
                out.append(_entry_from_row(provider_id, mid, row))
        return out

    # Cloudflare Workers AI: {"result": [{"name": "@cf/...", "task": ...}]}
    cf = body.get("result") if isinstance(body, dict) else None
    if isinstance(cf, list):
        out = []
        for row in cf:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            mid = str(row["name"]).strip()
            if mid:
                out.append(_entry_from_row(provider_id, mid, row))
        return out

    # Gemini: {"models": [{"name": "models/gemma-...", ...}]}
    gm = body.get("models") if isinstance(body, dict) else None
    if isinstance(gm, list):
        out = []
        for row in gm:
            if not isinstance(row, dict):
                continue
            methods = row.get("supportedGenerationMethods") or row.get("supported_generation_methods") or []
            if methods and "generateContent" not in methods:
                continue
            name = str(row.get("name", "")).strip()
            mid = name[len("models/"):] if name.startswith("models/") else name
            if mid:
                out.append(_entry_from_row(provider_id, mid, row))
        return out
    return []


def filter_vlm_catalog(entries: list[ModelCatalogEntry], *,
                       exclude_disabled_markers: bool = True) -> list[ModelCatalogEntry]:
    """一覧から、画像→テキストが確認できた行だけを順序維持で返す。

    Mistral/Pixtral は内蔵候補からは除外するが、カスタム接続は利用者が明示的に
    登録した接続なので、そこで同モデルを選ぶ権利まで奪わない。
    """
    out: list[ModelCatalogEntry] = []
    seen: set[str] = set()
    for entry in entries:
        model_low = entry.model_id.lower()
        if (entry.is_vlm
                and (not exclude_disabled_markers
                     or not any(marker in model_low for marker in _DISABLED_BUILTIN_VLM_MARKERS))
                and entry.model_id not in seen):
            seen.add(entry.model_id)
            out.append(entry)
    return out


def catalog_entry_from_id(provider_id: str, model_id: str) -> ModelCatalogEntry:
    """旧形式のIDリストや手入力値を、静的カタログ判定付き行へ変換する。"""
    mid = str(model_id or "").strip()
    is_vlm, source = classify_model_capability(provider_id, mid)
    return ModelCatalogEntry(model_id=mid, supports_image_input=is_vlm,
                             supports_text_output=True if is_vlm else None,
                             capability_source=source)
