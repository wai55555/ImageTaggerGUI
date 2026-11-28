"""資格情報・userinfo をログ／表示から伏せるための共有ヘルパー。

同じ伏せ字処理が vlm_transport（例外文言）と vlm_protocols（エラーボディ）で
必要になるため、循環 import を避けて独立モジュールに置く。
`vlm_transport._scrub_exc` はこの実装の薄いラッパーとして残す（既存テスト互換）。
"""
from __future__ import annotations

import re

# URL トークン（絶対 URL でも `url: /rel/path?...` の相対形でも）のクエリ文字列を丸ごと伏せる。
_URL_QUERY_RE = re.compile(r"((?:https?://|/)[^\s'\"]*?)\?[^\s'\"]*")
# URL の userinfo（http://user:pass@host）も伏せる。
_URL_USERINFO_RE = re.compile(r"(https?://)[^/\s'\"@]+@")
# `?foo=VALUE` / `&foo=VALUE` の値部分を伏せる（キー名は問わない。クエリ文脈のみ）。
_SECRET_PARAM_RE = re.compile(r"([?&][A-Za-z0-9_.\-]{1,40}=)[^\s&'\"]+")
# `Bearer <token>` / `Basic <credentials>` / `Token <token>` のトークン部を伏せる
# （この接頭辞がある文脈だけ。パスは壊さない）。
_BEARER_RE = re.compile(r"((?:bearer|basic|token)\s+)[^\s'\",;)}\]]+", re.IGNORECASE)
# 認証系ヘッダー名 ": value" / "= value" の値を伏せる（ヘッダー文脈だけ）。
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|x-goog-api-key|api[-_]?key|api[-_]?token"
    r"|anthropic-workspace-id)(['\"]?\s*[:=]\s*['\"]?)"
    # 値は `Basic dXNlcjpwYXNz` のように「スキーム＋資格情報」の 2 トークンになりうる。
    # 1 トークンだけ伏せると Base64 部分が残るため、スキームごと伏せる（PR#38 レビュー指摘）。
    r"((?:(?:bearer|basic|token)\s+)?[^\s'\",;)}\]]+)")

# 別ホストへのリダイレクトで落とす認証ヘッダー名（REQ-G11）。
AUTH_HEADER_NAMES = (
    "authorization",
    "x-api-key",
    "x-goog-api-key",
    "api-key",
    "api_key",
    "anthropic-api-key",
    "anthropic-workspace-id",
)


def scrub_text(text: str, *, limit: int = 300) -> str:
    """クエリ・userinfo・bearer・認証ヘッダーの値だけを伏せ、ホストとパスは残す。

    任意部分文字列の置換はパス情報まで壊すので行わない（文脈が明確な箇所のみ）。
    """
    text = " ".join(str(text or "").split())
    text = _URL_USERINFO_RE.sub(r"\1<redacted>@", text)
    text = _URL_QUERY_RE.sub(r"\1?<redacted>", text)
    text = _SECRET_PARAM_RE.sub(r"\1<redacted>", text)
    # ヘッダー文脈を先に処理する（`Authorization: Basic <b64>` を 1 回で伏せるため）。
    text = _AUTH_HEADER_RE.sub(r"\1\2<redacted>", text)
    text = _BEARER_RE.sub(r"\1<redacted>", text)
    return text[:limit]


def scrub_exception(exc: BaseException, *, limit: int = 300) -> str:
    """例外の文言を `scrub_text` に通す。"""
    return scrub_text(str(exc), limit=limit)


def strip_url_userinfo(url: str) -> str:
    """URL から userinfo（user:pass@）を除去する。無い場合はそのまま返す。"""
    return _URL_USERINFO_RE.sub(r"\1", str(url or ""))


def strip_auth_headers(headers: dict[str, str], extra: tuple[str, ...] = ()) -> dict[str, str]:
    """認証ヘッダーを取り除いた新しい辞書を返す（元は変更しない）。"""
    drop = {name.lower() for name in AUTH_HEADER_NAMES}
    drop.update(name.lower() for name in extra if name)
    return {k: v for k, v in headers.items() if k.lower() not in drop}
