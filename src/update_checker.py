"""起動時の新バージョン通知（通知型のみ。自己ダウンロード・自己更新は一切しない）。

設計の背景は docs/260917_updater_av_safety_design.md、実装計画は
docs/260917_updater_impl_plan.md を参照。要点:

- GitHub Releases の `/releases/latest` を HTTPS GET するだけ。実行ファイルの
  授受は一切無いため、自己更新特有のAV誤検知リスクはそもそも発生しない。
- 失敗はすべて握りつぶして None を返す（`onnx_providers.has_nvidia_gpu()`と同じ
  「失敗は機能オフとして安全側に倒す」方針）。呼び出し側は None を「新版なし」と
  同一に扱ってよい。

このモジュールはプラットフォーム非依存のロジックのみ（Qt非依存、UI結線は
main_window.py 側）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# GitHub の repo slug。`constants.APP_NAME` とは独立（たまたま同じ綴り）。
REPO = "wai55555/ImageTaggerGUI"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
# `html_url` はこのプレフィックスで始まるものだけを受け付ける。API 応答は TLS で
# 認証されているが、受け取った URL をそのまま `webbrowser.open()`（Windows では
# 実質 os.startfile）に渡すので、万一の改竄で任意スキーム/任意ホストを開かされない
# よう許可リストで閉じる（design doc 3.4.2 の「URL のホスト検査」を通知型にも適用）。
RELEASE_URL_PREFIX = f"https://github.com/{REPO}/releases/"


@dataclass
class UpdateInfo:
    """`current_version` より新しいリリースが見つかったときに返す情報。"""
    version: str        # tag_name そのまま（例 "v1.8.0"）
    html_url: str        # リリースページのブラウザURL（webbrowser.open 用）
    published_at: str


def _requests_get(url: str, *, headers: dict | None = None, timeout: int = 5):
    import requests

    return requests.get(url, headers=headers or {}, timeout=timeout)


def parse_version(tag_name: Any) -> tuple[int, ...] | None:
    """"v1.7.0" -> (1, 7, 0)。先頭の v/V を剥がし、"."区切りの数値タプルにする。

    数値でない要素（例 "1.7.0-beta"）が混じっていたら None（読めないバージョンは
    「新しいかどうか判定不能」として扱う——古いとも新しいとも断定しない）。
    """
    if not isinstance(tag_name, str) or not tag_name:
        return None
    raw = tag_name[1:] if tag_name[:1] in ("v", "V") else tag_name
    if not raw:
        return None
    parts = raw.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def check_for_update(current_version: str, *, http_get: Any = None, timeout: int = 5) -> UpdateInfo | None:
    """最新リリースを取得し、`current_version` より新しければ `UpdateInfo` を返す。

    新しくない・取得失敗・パース失敗はすべて None（例外を外に投げない）。
    `http_get` はテスト用の差し替え（gpu_runtime.py と同じ DI パターン）。
    """
    getter = http_get or _requests_get
    try:
        resp = getter(API_URL, headers={"Accept": "application/vnd.github+json"}, timeout=timeout)
        if getattr(resp, "status_code", None) != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    tag_name = data.get("tag_name")
    latest = parse_version(tag_name)
    current = parse_version(current_version)
    if latest is None or current is None:
        return None
    if latest <= current:
        return None

    html_url = data.get("html_url")
    if not isinstance(html_url, str) or not html_url.startswith(RELEASE_URL_PREFIX):
        return None
    published_at = data.get("published_at")

    return UpdateInfo(
        version=tag_name,
        html_url=html_url,
        published_at=published_at if isinstance(published_at, str) else "",
    )
