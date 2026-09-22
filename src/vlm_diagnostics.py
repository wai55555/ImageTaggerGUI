"""接続診断（260901_VLM_spec.md 12章 / design.md 4.6節）。

「その接続設定が実際に動くか」を一括で確認する。選択画像の品質を見る機能ではない。
APIキー登録時はモデル一覧GETだけの軽量確認も使い、推論レート制限を消費しない。
結果は接続定義に書き戻さず、状態キャッシュとして扱う。設定変更で無効化する。
"""
from __future__ import annotations

import io
import re
import socket
import ssl
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

from PIL import Image

from vlm_connections import VlmConnection, is_local_host
from vlm_errors import VlmErrorReason, reason_label_key
from vlm_image import ImagePreprocessConfig, prepare_image
from vlm_profiles import GenerationProfile, build_system_prompt, build_user_prompt
from vlm_protocols import (
    VlmCallSpec, VlmHttpRequest, apply_connection_auth, apply_request_body,
    default_auth_key, extract_by_path,
    get_protocol, apply_request_headers,
)
from vlm_transport import RawHttpResponse, execute_http
from vlm_model_list import _extract_catalog

# Cloudflare のトークン検証はアカウント ID やモデルに依存しない専用エンドポイント。
_CLOUDFLARE_TOKEN_VERIFY_URL = "https://api.cloudflare.com/client/v4/user/tokens/verify"


class DiagStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


# 診断項目の内部ID（DiagItem.name）→ 表示ラベルの翻訳キー。
# name は api_key_dialog / DiagReport.can_mark_binding_verified などの判定にも
# 使われる安定IDなので英語のまま固定し、翻訳は表示時にここを引いて行う。
# 診断の実リクエストで使う出力トークン上限。長文生成を待つ必要はないので絞るが、
# thinking 対応モデルは思考でトークンを食うため、回答が数トークン残る程度は確保する。
# 「打ち切られた」旨の詳細文にもこの値を差し込むので、名前付き定数にしておく。
_DIAG_MAX_OUTPUT_TOKENS = 128

DIAG_ITEM_LABEL_KEYS: dict[str, str] = {
    "URL format": "Diag_Item_Url_Format",
    "Model ID": "Diag_Item_Model_Id",
    "Protocol": "Diag_Item_Protocol",
    "DNS / TCP": "Diag_Item_Dns_Tcp",
    "TLS": "Diag_Item_Tls",
    "Auth": "Diag_Item_Auth",
    "Request build": "Diag_Item_Request_Build",
    "Image input": "Diag_Item_Image_Input",
    "HTTP response": "Diag_Item_Http_Response",
    "Caption extraction": "Diag_Item_Caption_Extraction",
    "Rate-limit info": "Diag_Item_Rate_Limit_Info",
}

DIAG_STATUS_LABEL_KEYS: dict[str, str] = {
    DiagStatus.PASS.value: "Diag_Status_Pass",
    DiagStatus.WARN.value: "Diag_Status_Warn",
    DiagStatus.FAIL.value: "Diag_Status_Fail",
    DiagStatus.SKIP.value: "Diag_Status_Skip",
}


@dataclass(frozen=True)
class DiagArgKey:
    """`detail_args` の値が、それ自体翻訳キーであることを示す包み。

    「トランスポート層の失敗理由」のように、差し込む値の側も訳したいものがある。
    表示時に item_detail() が先にこれを解決してから本文へ差し込む。
    """
    section: str
    key: str
    fallback: str = ""


@dataclass
class DiagItem:
    name: str
    status: DiagStatus
    detail: str = ""
    # 表示用の翻訳キーと差し込み値。`detail` は英語のまま残す:
    # デバッグログに出すのは英語が望ましく、api_key_dialog が
    # is_billing_or_credit_block() など文字列判定に使っているため、
    # ここを翻訳済み文字列に差し替えると判定が壊れる。
    detail_key: str = ""
    detail_args: dict = field(default_factory=dict)
    # 翻訳文の末尾に素のまま足す供給元のメッセージ（プロバイダーのエラー本文など）。
    detail_suffix: str = ""


@dataclass
class DiagReport:
    connection_id: str
    items: list[DiagItem] = field(default_factory=list)
    http_status: int | None = None   # live リクエストが返した HTTP ステータス（あれば）
    billing_blocked: bool = False    # 到達・認証後に課金／残高で生成を拒否された
    lightweight: bool = False        # 推論を行わず、モデル一覧GETだけで疎通確認した

    def add(self, name: str, status: DiagStatus, detail: str = "", *,
            detail_key: str = "", detail_args: dict | None = None,
            detail_suffix: str = "") -> None:
        self.items.append(DiagItem(name, status, detail, detail_key,
                                   dict(detail_args or {}), detail_suffix))

    def item(self, name: str) -> DiagItem | None:
        for i in self.items:
            if i.name == name:
                return i
        return None

    @property
    def overall(self) -> DiagStatus:
        if any(i.status is DiagStatus.FAIL for i in self.items):
            return DiagStatus.FAIL
        if any(i.status is DiagStatus.WARN for i in self.items):
            return DiagStatus.WARN
        return DiagStatus.PASS

    @property
    def can_mark_binding_verified(self) -> bool:
        """接続設定を「確認済み」として記録できるか。

        通常は HTTP 200 で本文抽出まで成功した場合だけ実証済みとする。ただし
        429 は認証済みのリクエストがエンドポイントへ到達したことを示すため、
        認証・リクエスト組み立て・画像入力が PASS なら到達確認済みとして扱う。
        この場合も ``overall`` は WARN のままなので、本文取得成功と混同しない。
        課金不足は HTTP を WARN にするが、課金不足だけではモデルの応答まで確認
        できないため、この例外には入れない。
        """
        http = self.item("HTTP response")
        if http is None:
            return False
        if self.billing_blocked or is_billing_or_credit_block(http.detail):
            return False
        if self.lightweight:
            # APIキー登録時の軽量確認は、認証付きのモデル一覧GETが通れば十分。
            # 429も認証済み到達として記録するが、請求／残高不足は上で除外する。
            auth = self.item("Auth")
            request = self.item("Request build")
            if (auth is None or auth.status is not DiagStatus.PASS
                    or request is None or request.status is not DiagStatus.PASS):
                return False
            return (http.status is DiagStatus.PASS
                    or (self.http_status == 429 and http.status is DiagStatus.WARN))
        extraction = self.item("Caption extraction")
        if http.status is DiagStatus.PASS and extraction is not None:
            if extraction.status is DiagStatus.PASS:
                return True
            # 200応答が診断用トークン上限で切れた場合も、モデルまで到達したことは
            # 確認できている。本文抽出成功とは区別するため、詳細に明示されたケースだけ
            # を到達確認として扱う（content policy 等の一般WARNは含めない）。
            if (extraction.status is DiagStatus.WARN
                    and "endpoint reachable" in (extraction.detail or "").lower()):
                return True
        if self.http_status != 429 or http.status is not DiagStatus.WARN:
            return False
        required = ("Auth", "Request build", "Image input")
        return all((item := self.item(name)) is not None
                   and item.status is DiagStatus.PASS for name in required)


def _tiny_test_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _first_cf_message(body: dict) -> str:
    for coll in (body.get("errors"), body.get("messages")):
        if isinstance(coll, list) and coll and isinstance(coll[0], dict):
            msg = coll[0].get("message")
            if msg:
                return str(msg)
    return ""


def _response_error_detail(raw: RawHttpResponse) -> str:
    """プロバイダーのJSONエラーを、キー値を含めず診断画面へ返す。"""
    body = raw.json_body if isinstance(raw.json_body, dict) else {}
    message = _first_cf_message(body)
    error = body.get("error")
    if not message and isinstance(error, dict):
        message = str(error.get("message") or error.get("type") or error.get("code") or "")
    if not message and isinstance(error, str):
        message = error
    if not message:
        message = (raw.text_body or "")[:300].replace("\n", " ")
    return message.strip()


def is_billing_or_credit_block(detail: str) -> bool:
    """認証後に返る請求設定・残高不足を、キー不正と区別する。"""
    low = (detail or "").lower()
    return any(marker in low for marker in (
        "credit card",
        "no credits remaining",
        "credit balance",
        "insufficient credit",
        "insufficient_quota",
        "billing",
        "payment method",
    ))


def item_label(name: str, get_string) -> str:
    """診断項目の表示ラベル。未知のIDは内部IDをそのまま返す。"""
    key = DIAG_ITEM_LABEL_KEYS.get(name)
    return get_string("Vlm", key) if key else name


def status_label(status: DiagStatus, get_string) -> str:
    """PASS / WARN / FAIL / SKIP の表示ラベル。"""
    key = DIAG_STATUS_LABEL_KEYS.get(status.value)
    return get_string("Vlm", key) if key else status.value


def item_detail(item: DiagItem, get_string) -> str:
    """診断項目の詳細を表示用に翻訳する。

    翻訳キーを持たない項目（URL・モデルID・生の例外文など、そもそも翻訳対象では
    ない値）は英語の `detail` をそのまま返す。
    """
    if not item.detail_key:
        return item.detail
    args = {}
    for name, value in item.detail_args.items():
        if isinstance(value, DiagArgKey):
            resolved = get_string(value.section, value.key) if value.key else ""
            args[name] = (value.fallback or value.key
                          if not resolved or resolved == value.key else resolved)
        else:
            args[name] = value
    text = get_string("Vlm", item.detail_key, **args)
    if not text or text == item.detail_key:
        # ini にキーが無い等で引けなかった場合は英語へ落とす（生キーを見せない）。
        text = item.detail
    if item.detail_suffix:
        text = f"{text}: {item.detail_suffix}" if text else item.detail_suffix
    return text


def format_report_lines(report: DiagReport, get_string) -> list[str]:
    """`[状態] 項目: 詳細` の表示行を組み立てる。"""
    return [f"[{status_label(i.status, get_string)}] "
            f"{item_label(i.name, get_string)}: {item_detail(i, get_string)}"
            for i in report.items]


def _cloudflare_token_probe(rep: DiagReport, api_key: str, *, verify_tls: bool = True) -> None:
    """Cloudflare API トークンを専用エンドポイントで検証し、結果を Auth / HTTP response
    項目へ反映する（api_key_dialog はこの2項目で保存可否を決める）。"""
    req = VlmHttpRequest(method="GET", url=_CLOUDFLARE_TOKEN_VERIFY_URL,
                         headers={"Authorization": f"Bearer {api_key}"})
    raw = execute_http(req, connect_timeout=10.0, read_timeout=15.0, verify_tls=verify_tls)
    if not isinstance(raw, RawHttpResponse):
        rep.add("HTTP response", DiagStatus.FAIL, f"{raw.reason.value}: {raw.message}",
                detail_key="Diag_D_Transport_Error",
                detail_args={"reason": DiagArgKey(
                    "Vlm", reason_label_key(raw.reason), str(raw.reason.value))},
                detail_suffix=raw.message)
        rep.add("Caption extraction", DiagStatus.SKIP, "no response to check",
                detail_key="Diag_D_No_Response_To_Check")
        return
    rep.http_status = raw.status
    body = raw.json_body if isinstance(raw.json_body, dict) else {}
    result = body.get("result") if isinstance(body.get("result"), dict) else {}
    token_status = str(result.get("status", "")).lower()
    auth_item = rep.item("Auth")
    if raw.status == 200 and body.get("success") is True and token_status in ("", "active"):
        rep.add("HTTP response", DiagStatus.PASS, "token valid and active",
                detail_key="Diag_D_Token_Valid")
        if auth_item is not None:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = "Cloudflare token verified"
            auth_item.detail_key = "Diag_D_Cf_Token_Verified"
    elif raw.status in (401, 403) or body.get("success") is False:
        cf_message = _first_cf_message(body)
        msg = cf_message or f"{raw.status} token rejected"
        rep.add("HTTP response", DiagStatus.FAIL, msg,
                detail_key="Diag_D_Token_Rejected",
                detail_args={"status": raw.status},
                detail_suffix=cf_message)
        if auth_item is not None:
            auth_item.status = DiagStatus.FAIL
            auth_item.detail = msg
            # cf_message の有無で分岐しない: 分岐していると、Cloudflare が
            # 具体的な拒否理由を返したときだけ detail_key の更新をスキップして
            # しまい、診断の早い段階で入った "credential present"（PASS寄りの
            # 文言）が残ったまま item_detail() が正しく解決してしまう。
            # ステータスは FAIL なのに表示だけ PASS 寄りのまま、という
            # 英語detailより始末が悪い状態になっていた（260923 PR#27 レビュー指摘、
            # 実機で再現確認）。
            auth_item.detail_key = "Diag_D_Token_Rejected"
            auth_item.detail_args = {"status": raw.status}
            auth_item.detail_suffix = cf_message
    elif raw.status == 200 and body.get("success") is True:
        rep.add("HTTP response", DiagStatus.FAIL, f"token is {token_status or 'not active'}",
                detail_key="Diag_D_Token_Not_Active",
                detail_args={"status": token_status or "not active"})
        if auth_item is not None:
            auth_item.status = DiagStatus.FAIL
            auth_item.detail = f"token is {token_status or 'not active'}"
            auth_item.detail_key = "Diag_D_Token_Not_Active"
            auth_item.detail_args = {"status": token_status or "not active"}
    else:
        rep.add("HTTP response", DiagStatus.WARN, f"HTTP {raw.status}",
                detail_key="Diag_D_Http_Status", detail_args={"status": raw.status})
    rep.add("Caption extraction", DiagStatus.SKIP, "Cloudflare token-verify check only",
            detail_key="Diag_D_Cf_Token_Only")


def _model_list_request(conn: VlmConnection, api_key: str | None) -> VlmHttpRequest:
    """推論を発生させない認証付きモデル一覧GETを組み立てる。"""
    base = (conn.base_url or "").rstrip("/")
    is_cloudflare = (conn.provider_id == "cloudflare"
                     or "api.cloudflare.com" in (urlparse(base).hostname or ""))
    if is_cloudflare and base.endswith("/ai/v1"):
        url = base[:-len("/v1")] + "/models/search"
        params = {"per_page": "100"}
    else:
        url = f"{base}/models"
        params = {}
    req = VlmHttpRequest(method="GET", url=url, headers={}, params=params, json_body={})
    if conn.protocol == "anthropic_messages":
        req.headers["anthropic-version"] = "2023-06-01"
    default_key = default_auth_key(conn.auth.type, api_key)
    if default_key:
        req.headers["Authorization"] = f"Bearer {default_key}"
    apply_connection_auth(req, conn.auth.type, api_key,
                          conn.auth.header_name, conn.auth.query_param)
    apply_request_headers(req, conn.request_headers)
    return req


def _run_lightweight_probe(rep: DiagReport, conn: VlmConnection,
                           api_key: str | None) -> None:
    """GET /models（Cloudflareは /models/search）だけで認証・到達性を確認する。"""
    try:
        req = _model_list_request(conn, api_key)
        rep.add("Request build", DiagStatus.PASS, f"{req.method} {req.url}")
    except Exception as e:  # noqa: BLE001 - 診断なので原因をレポートへ残す
        rep.add("Request build", DiagStatus.FAIL, f"{type(e).__name__}: {e}")
        return
    rep.add("Image input", DiagStatus.SKIP,
            "lightweight connectivity check; no inference request",
            detail_key="Diag_D_Lightweight_No_Inference")
    raw = execute_http(req, connect_timeout=min(conn.retry.connect_timeout_s, 10.0),
                       read_timeout=min(conn.retry.read_timeout_s, 15.0),
                       verify_tls=conn.verify_tls)
    if not isinstance(raw, RawHttpResponse):
        rep.add("HTTP response", DiagStatus.FAIL, f"{raw.reason.value}: {raw.message}",
                detail_key="Diag_D_Transport_Error",
                detail_args={"reason": DiagArgKey(
                    "Vlm", reason_label_key(raw.reason), str(raw.reason.value))},
                detail_suffix=raw.message)
        rep.add("Caption extraction", DiagStatus.SKIP, "no response to extract from",
                detail_key="Diag_D_No_Response_To_Extract")
        return

    rep.http_status = raw.status
    provider_detail = _response_error_detail(raw)
    rep.billing_blocked = is_billing_or_credit_block(provider_detail)
    if raw.status == 200 and isinstance(raw.json_body, (dict, list)):
        try:
            model_entries = _extract_catalog(raw.json_body, conn.provider_id)
        except (TypeError, ValueError, AttributeError):
            model_entries = []
        if model_entries:
            rep.add("HTTP response", DiagStatus.PASS,
                    f"200 OK (lightweight model-list check; {len(model_entries)} entries)",
                    detail_key="Diag_D_Model_List_Ok",
                    detail_args={"count": len(model_entries)})
        else:
            rep.add("HTTP response", DiagStatus.FAIL,
                    "200 OK but model-list response contained no model entries",
                    detail_key="Diag_D_Model_List_Empty")
    elif rep.billing_blocked:
        detail = f"{raw.status} billing / credits unavailable (endpoint reached; inference not verified)"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail,
                detail_key="Diag_D_Billing_Blocked",
                detail_args={"status": raw.status}, detail_suffix=provider_detail)
    elif raw.status in (401, 403):
        detail = f"{raw.status} auth rejected"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.FAIL, detail,
                detail_key="Diag_D_Auth_Rejected",
                detail_args={"status": raw.status}, detail_suffix=provider_detail)
    elif raw.status == 429:
        detail = "429 rate limited (lightweight endpoint reachable)"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail,
                detail_key="Diag_D_Rate_Limited",
                detail_suffix=provider_detail)
    elif raw.status == 200:
        rep.add("HTTP response", DiagStatus.WARN,
                "200 OK but model-list response was not JSON",
                detail_key="Diag_D_Model_List_Not_Json")
    else:
        detail = f"HTTP {raw.status}"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail,
                detail_key="Diag_D_Http_Status",
                detail_args={"status": raw.status}, detail_suffix=provider_detail)

    auth_item = rep.item("Auth")
    if auth_item is not None and conn.auth.type != "none":
        if rep.billing_blocked:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = f"accepted; billing / credits unavailable (server responded {raw.status})"
            auth_item.detail_key = "Diag_D_Auth_Accepted_Billing"
            auth_item.detail_args = {"status": raw.status}
        elif raw.status in (401, 403):
            auth_item.status = DiagStatus.FAIL
            auth_item.detail = f"rejected by the server ({raw.status})"
            auth_item.detail_key = "Diag_D_Auth_Rejected_By_Server"
            auth_item.detail_args = {"status": raw.status}
        else:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = f"accepted (server responded {raw.status})"
            auth_item.detail_key = "Diag_D_Auth_Accepted"
            auth_item.detail_args = {"status": raw.status}
    rep.add("Caption extraction", DiagStatus.SKIP,
            "lightweight connectivity check; inference skipped",
            detail_key="Diag_D_Lightweight_Inference_Skipped")
    rl_names = [k for k in raw.headers if k.lower().startswith(("x-ratelimit", "ratelimit", "retry-after"))]
    rep.add("Rate-limit info", DiagStatus.PASS,
            ", ".join(rl_names) if rl_names else "none exposed (falls back to 429 Retry-After)",
            detail_key="" if rl_names else "Diag_D_Rate_Limit_None")


def diagnose(conn: VlmConnection, api_key: str | None, *,
             do_live_request: bool = True, lightweight: bool = False) -> DiagReport:
    """接続を一括診断する。

    ``lightweight=True`` はAPIキー登録向けで、画像生成POSTを行わずモデル一覧GETのみを
    送る。``do_live_request=False`` なら、どちらの方式でもネットワークへ触れない。
    """
    rep = DiagReport(connection_id=conn.connection_id, lightweight=lightweight)

    # 1. URL / 設定値の形式
    try:
        parsed = urlparse(conn.base_url)
        # Accessing .port is itself validating for malformed ports and bracketed IPv6.
        parsed_port = parsed.port
    except ValueError as e:
        rep.add("URL format", DiagStatus.FAIL, f"invalid base_url: {e}",
                detail_key="Diag_D_Invalid_Base_Url", detail_suffix=str(e))
        return rep
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        rep.add("URL format", DiagStatus.FAIL, f"invalid base_url: {conn.base_url!r}",
                detail_key="Diag_D_Invalid_Base_Url", detail_suffix=repr(conn.base_url))
        return rep
    is_cloudflare = (conn.provider_id == "cloudflare"
                     or (parsed.hostname or "").endswith("api.cloudflare.com"))
    # base_url にテンプレート変数（`{account_id}` 等）が残っていると、そのまま実
    # リクエストして意味不明な 404 になる。Cloudflare のアカウント ID 未設定だけは
    # WARN 止まり（キー自体は下の専用エンドポイントで検証できる）。それ以外の
    # 未展開変数は原因を明示して打ち切る。
    unresolved = re.findall(r"\{[A-Za-z_][A-Za-z0-9_]*\}", conn.base_url)
    cf_missing_account = is_cloudflare and unresolved == ["{account_id}"]
    if unresolved and not cf_missing_account:
        rep.add("URL format", DiagStatus.FAIL,
                f"unresolved placeholder in base_url: {' '.join(unresolved)}",
                detail_key="Diag_D_Unresolved_Placeholder",
                detail_args={"placeholders": " ".join(unresolved)})
        return rep
    if cf_missing_account:
        rep.add("URL format", DiagStatus.WARN,
                "Cloudflare account ID is not set - the key can still be verified, "
                "but this route will not run until Register API key is opened and the "
                "Account ID is entered there",
                detail_key="Diag_D_Cf_Account_Missing")
    elif parsed.scheme == "http" and not _looks_localish(parsed.hostname or ""):
        rep.add("URL format", DiagStatus.WARN, "plain http to a non-local host",
                detail_key="Diag_D_Plain_Http_Remote")
    else:
        rep.add("URL format", DiagStatus.PASS, conn.base_url)
    if not conn.model_id:
        rep.add("Model ID", DiagStatus.FAIL, "model_id is empty",
                detail_key="Diag_D_Model_Id_Empty")
    else:
        rep.add("Model ID", DiagStatus.PASS, conn.model_id)
    if conn.protocol not in ("openai_chat_completions", "openai_responses",
                             "anthropic_messages", "gemini_generate_content"):
        rep.add("Protocol", DiagStatus.WARN,
                f"unknown protocol {conn.protocol!r}, treated as OpenAI compatible",
                detail_key="Diag_D_Unknown_Protocol",
                detail_args={"protocol": conn.protocol})
    else:
        rep.add("Protocol", DiagStatus.PASS, conn.protocol)

    host = parsed.hostname or ""
    port = parsed_port or (443 if parsed.scheme == "https" else 80)

    if not do_live_request:
        # 静的検査モード: ネットワークに触れない（DNS / TCP / TLS / 実リクエストを飛ばす）。
        rep.add("DNS / TCP", DiagStatus.SKIP, "static check only",
                detail_key="Diag_D_Static_Only")
        rep.add("TLS", DiagStatus.SKIP, "static check only",
                detail_key="Diag_D_Static_Only")
    else:
        # 2. DNS / TCP
        try:
            socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
            rep.add("DNS / TCP", DiagStatus.PASS, f"{host}:{port}")
        except OSError as e:
            rep.add("DNS / TCP", DiagStatus.FAIL, f"cannot resolve/connect {host}:{port}: {e}",
                    detail_key="Diag_D_Dns_Failed",
                    detail_args={"host": host, "port": port}, detail_suffix=str(e))
            return rep

        # 3. TLS
        if parsed.scheme == "https":
            try:
                ctx = ssl.create_default_context()
                if not conn.verify_tls:
                    ctx.check_hostname = False
                    ctx.verify_mode = ssl.CERT_NONE
                with socket.create_connection((host, port), timeout=conn.retry.connect_timeout_s) as sock:
                    with ctx.wrap_socket(sock, server_hostname=host):
                        pass
                rep.add("TLS", DiagStatus.PASS if conn.verify_tls else DiagStatus.WARN,
                        "verified" if conn.verify_tls else "verification disabled",
                        detail_key=("Diag_D_Tls_Verified" if conn.verify_tls
                                    else "Diag_D_Tls_Not_Verified"))
            except (ssl.SSLError, OSError) as e:
                rep.add("TLS", DiagStatus.FAIL, f"TLS handshake failed: {e}",
                        detail_key="Diag_D_Tls_Failed", detail_suffix=str(e))
        else:
            rep.add("TLS", DiagStatus.SKIP, "plain http",
                    detail_key="Diag_D_Plain_Http")

    # 4. Auth presence
    if conn.auth.type == "none":
        rep.add("Auth", DiagStatus.PASS, "no auth required",
                detail_key="Diag_D_No_Auth_Required")
    elif api_key:
        rep.add("Auth", DiagStatus.PASS, f"{conn.auth.type} credential present",
                detail_key="Diag_D_Credential_Present",
                detail_args={"type": conn.auth.type})
    else:
        rep.add("Auth", DiagStatus.FAIL, f"{conn.auth.type} required but no credential found",
                detail_key="Diag_D_Credential_Required",
                detail_args={"type": conn.auth.type})
        # 認証情報が無い状態で未認証リクエストを送ると、Vercel等の401本文だけが表示されて
        # 「キーが不正」と誤解しやすい。ネットワーク到達性は上で確認済みなので、ここで終了。
        rep.add("HTTP response", DiagStatus.SKIP, "credential missing",
                detail_key="Diag_D_Credential_Missing")
        rep.add("Caption extraction", DiagStatus.SKIP, "credential missing",
                detail_key="Diag_D_Credential_Missing")
        return rep

    # 4b. Account ID が未設定の Cloudflare だけは、専用エンドポイントでトークン単体を
    # 検証する。Account ID がある場合はこの先の chat/completions へ進み、アカウント・
    # Workers AI 権限・モデル・画像入力・応答抽出まで含めて接続を確認する。
    if cf_missing_account and conn.auth.type == "bearer" and api_key:
        if do_live_request:
            _cloudflare_token_probe(rep, api_key, verify_tls=conn.verify_tls)
        else:
            rep.add("HTTP response", DiagStatus.SKIP, "live request disabled",
                    detail_key="Diag_D_Live_Disabled")
            rep.add("Caption extraction", DiagStatus.SKIP, "live request disabled",
                    detail_key="Diag_D_Live_Disabled")
        return rep

    if lightweight:
        if do_live_request:
            _run_lightweight_probe(rep, conn, api_key)
        else:
            rep.add("Request build", DiagStatus.SKIP, "live request disabled",
                    detail_key="Diag_D_Live_Disabled")
            rep.add("Image input", DiagStatus.SKIP, "lightweight connectivity check",
                    detail_key="Diag_D_Lightweight_Check")
            rep.add("HTTP response", DiagStatus.SKIP, "live request disabled",
                    detail_key="Diag_D_Live_Disabled")
            rep.add("Caption extraction", DiagStatus.SKIP, "live request disabled",
                    detail_key="Diag_D_Live_Disabled")
        return rep

    # 5. Request build
    try:
        prepared = prepare_image(_tiny_test_image_bytes(), ImagePreprocessConfig(max_long_edge=64))
        # 診断は「200 が返り、テキストが取り出せるか」の確認。長文生成を待つ必要はないので
        # 出力トークンを絞る（既定 1024 のままだと Gemma 等で timeout する）。ただし thinking
        # 対応モデルは思考でトークンを食うので、回答が数トークンは残るよう 128 にする。
        profile = GenerationProfile(max_output_tokens=_DIAG_MAX_OUTPUT_TOKENS)
        call = VlmCallSpec(conn.model_id, build_system_prompt(profile), build_user_prompt(profile), prepared, profile)
        protocol = get_protocol(conn.protocol)
        if conn.text_path:
            protocol.default_text_path = conn.text_path
        req = protocol.build_request(conn.base_url, default_auth_key(conn.auth.type, api_key), call)
        apply_connection_auth(req, conn.auth.type, api_key, conn.auth.header_name, conn.auth.query_param)
        apply_request_headers(req, conn.request_headers)
        apply_request_body(req, conn.request_body)
        rep.add("Request build", DiagStatus.PASS, f"{req.method} {req.url}")
    except Exception as e:  # noqa: BLE001 - 診断なので全部拾う
        rep.add("Request build", DiagStatus.FAIL, f"{type(e).__name__}: {e}")
        return rep

    # 6. Image input format (静的確認のみ)
    rep.add("Image input", DiagStatus.PASS, f"{prepared.mime_type}, base64/data-url ready",
            detail_key="Diag_D_Image_Ready",
            detail_args={"mime": prepared.mime_type})

    if not do_live_request:
        rep.add("HTTP response", DiagStatus.SKIP, "live request disabled",
                detail_key="Diag_D_Live_Disabled")
        rep.add("Caption extraction", DiagStatus.SKIP, "live request disabled",
                detail_key="Diag_D_Live_Disabled")
        return rep

    # 7-11. 実リクエスト。診断は「疎通確認」なのでタイムアウトは短めに固定する
    # （設定ダイアログを閉じるときの待ち時間を抑える。実生成は本来の timeout を使う）。
    raw = execute_http(req, connect_timeout=min(conn.retry.connect_timeout_s, 10.0),
                       read_timeout=min(conn.retry.read_timeout_s, 30.0),
                       verify_tls=conn.verify_tls)
    if not isinstance(raw, RawHttpResponse):
        rep.add("HTTP response", DiagStatus.FAIL, f"{raw.reason.value}: {raw.message}",
                detail_key="Diag_D_Transport_Error",
                detail_args={"reason": DiagArgKey(
                    "Vlm", reason_label_key(raw.reason), str(raw.reason.value))},
                detail_suffix=raw.message)
        return rep

    rep.http_status = raw.status
    provider_detail = _response_error_detail(raw)
    billing_blocked = is_billing_or_credit_block(provider_detail)
    rep.billing_blocked = billing_blocked
    if raw.status == 200:
        rep.add("HTTP response", DiagStatus.PASS, "200 OK",
                detail_key="Diag_D_Http_Ok")
    elif billing_blocked:
        detail = f"{raw.status} billing / credits unavailable (endpoint reached; inference not verified)"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail,
                detail_key="Diag_D_Billing_Blocked",
                detail_args={"status": raw.status}, detail_suffix=provider_detail)
    elif raw.status in (401, 403):
        detail = f"{raw.status} auth rejected"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.FAIL, detail,
                detail_key="Diag_D_Auth_Rejected",
                detail_args={"status": raw.status}, detail_suffix=provider_detail)
    elif raw.status in (404, 400, 422):
        detail = f"{raw.status} model / request rejected (auth was accepted)"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.FAIL, detail,
                detail_key="Diag_D_Model_Request_Rejected",
                detail_args={"status": raw.status}, detail_suffix=provider_detail)
    elif raw.status == 429:
        detail = "429 rate limited (endpoint reachable)"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail,
                detail_key="Diag_D_Rate_Limited",
                detail_suffix=provider_detail)
    else:
        detail = f"HTTP {raw.status}"
        if provider_detail:
            detail += f": {provider_detail}"
        rep.add("HTTP response", DiagStatus.WARN, detail,
                detail_key="Diag_D_Http_Status",
                detail_args={"status": raw.status}, detail_suffix=provider_detail)

    # 4'. Auth の判定を実応答で上書きする。請求設定・残高不足が明記された403等は
    # 認証成功として扱い、それ以外の401/403だけを「キー不正」とする。
    auth_item = rep.item("Auth")
    if auth_item is not None and conn.auth.type != "none":
        if billing_blocked:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = f"accepted; billing / credits unavailable (server responded {raw.status})"
            auth_item.detail_key = "Diag_D_Auth_Accepted_Billing"
            auth_item.detail_args = {"status": raw.status}
        elif raw.status in (401, 403):
            auth_item.status = DiagStatus.FAIL
            auth_item.detail = f"rejected by the server ({raw.status})"
            auth_item.detail_key = "Diag_D_Auth_Rejected_By_Server"
            auth_item.detail_args = {"status": raw.status}
        else:
            auth_item.status = DiagStatus.PASS
            auth_item.detail = f"accepted (server responded {raw.status})"
            auth_item.detail_key = "Diag_D_Auth_Accepted"
            auth_item.detail_args = {"status": raw.status}

    ext_status, ext_detail, ext_key, ext_args, ext_suffix = _classify_extraction(
        raw, protocol, conn.text_path)
    rep.add("Caption extraction", ext_status, ext_detail,
            detail_key=ext_key, detail_args=ext_args, detail_suffix=ext_suffix)

    # 10. Rate-limit headers（情報表示のみ。無くても正常＝多くの API は付けない。
    # その場合は 429 応答の Retry-After を見て事後クールダウンする。WARN にしない）。
    rl_names = [k for k in raw.headers if k.lower().startswith(("x-ratelimit", "ratelimit", "retry-after"))]
    rep.add("Rate-limit info", DiagStatus.PASS,
            ", ".join(rl_names) if rl_names else "none exposed (falls back to 429 Retry-After)",
            detail_key="" if rl_names else "Diag_D_Rate_Limit_None")

    return rep


def _classify_extraction(raw: RawHttpResponse, protocol, configured_path: str = ""
                         ) -> tuple[DiagStatus, str, str, dict, str]:
    """live レスポンスからテキストが取り出せるかを判定する。

    戻り値は (状態, 英語の詳細, 表示用翻訳キー, 差し込み値, 素で末尾へ足す文字列)。
    英語の詳細はデバッグログ用にそのまま残す。

    診断は出力トークンを絞るので、テキストが出る前に打ち切られること（finishReason=
    MAX_TOKENS / length）がある。その場合はエンドポイント・認証・リクエスト形状は通って
    いるので WARN 止まり。真に形が違うときだけ FAIL（本文の頭を付ける）。
    """
    parsed = protocol.parse_response(raw.status, raw.json_body, raw.text_body)
    if parsed.ok:
        chars = len(parsed.text or "")
        if configured_path:
            return (DiagStatus.PASS, f"got {chars} chars via {configured_path}",
                    "Diag_D_Extract_Ok_Via",
                    {"chars": chars, "path": configured_path}, "")
        return (DiagStatus.PASS, f"got {chars} chars",
                "Diag_D_Extract_Ok", {"chars": chars}, "")
    if parsed.error and parsed.error.reason is VlmErrorReason.CONTENT_POLICY:
        return (DiagStatus.WARN,
                "content policy on the test image (extraction path unverified)",
                "Diag_D_Extract_Content_Policy", {}, "")
    if raw.status != 200:
        return (DiagStatus.SKIP, "no successful response to extract from",
                "Diag_D_Extract_No_Response", {}, "")
    finish_reason = str(
        extract_by_path(raw.json_body, "candidates[0].finishReason")
        or extract_by_path(raw.json_body, "choices[0].finish_reason")
        or extract_by_path(raw.json_body, "incomplete_details.reason")
        or extract_by_path(raw.json_body, "stop_reason") or ""
    ).upper()
    if finish_reason in ("MAX_TOKENS", "MAX_OUTPUT_TOKENS", "LENGTH"):
        return (DiagStatus.WARN,
                "response truncated at diagnostic max_output_tokens="
                f"{_DIAG_MAX_OUTPUT_TOKENS}; endpoint is reachable, "
                "but this VLM may need a larger generation budget",
                "Diag_D_Extract_Truncated", {"limit": _DIAG_MAX_OUTPUT_TOKENS}, "")
    preview = (raw.text_body or "")[:200].replace("\n", " ")
    path = configured_path or getattr(protocol, "default_text_path", "") or "(protocol default)"
    detail = f"200 OK but response text path {path!r} did not match; verify protocol/path settings"
    if preview:
        detail += f" — body starts: {preview}"
    return (DiagStatus.FAIL, detail, "Diag_D_Extract_Path_Mismatch",
            {"path": path}, preview)


def _looks_localish(host: str) -> bool:
    """Compatibility wrapper for diagnostics and its existing tests."""
    return is_local_host(host)
