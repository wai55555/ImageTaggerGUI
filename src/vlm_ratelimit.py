"""接続・モデル単位のレート制限／クールダウン状態（260901_VLM_spec.md 9章）。

リセット情報が明示されない場合は推測値として扱い、「サービス停止」とは表示しない。
推測クールダウンには上限を設け、次回の接続診断で再確認できるようにする。

時刻源（REQ-G5）:
- 残り時間の比較は**単調時計**（`time.monotonic`）で行う。壁時計は、セッション中の
  システム時計の巻き戻しで残り時間が伸びてしまうため、判定には使わない。
- ヘッダー由来の絶対リセット時刻（サーバーの壁時計 epoch）は、登録時点で
  「残り時間」へ換算し、単調時計の deadline として保持する。比較側だけを単調時計へ
  差し替えると、epoch 由来の deadline が単調時計の現在値より常に大きくなり
  クールダウンが恒久化するので、必ず登録時に換算する。
- `reset_at_utc`（壁時計）は表示用にのみ残す。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime

# 推測クールダウンの上限（秒）。明示リセットが無い 429 でここまで自動で待つ。
ESTIMATED_COOLDOWN_CAP_S = 15 * 60
# 明示情報の無い 429 の初期推測クールダウン。
ESTIMATED_COOLDOWN_DEFAULT_S = 60


@dataclass
class RateLimitState:
    connection_id: str
    last_429_at_utc: float | None = None
    retry_after_s: float | None = None
    # 表示用の絶対時刻（壁時計）。判定には使わない。
    reset_at_utc: float | None = None
    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    # 判定用の deadline（単調時計）。None は「クールダウンしていない」。
    cooldown_deadline_mono: float | None = None
    source: str = ""  # header | response | estimated

    def in_cooldown(self, now_mono: float | None = None) -> bool:
        now_mono = now_mono if now_mono is not None else time.monotonic()
        return (self.cooldown_deadline_mono is not None
                and self.cooldown_deadline_mono > now_mono)

    def remaining_s(self, now_mono: float | None = None) -> float:
        now_mono = now_mono if now_mono is not None else time.monotonic()
        if self.cooldown_deadline_mono is None:
            return 0.0
        return max(0.0, self.cooldown_deadline_mono - now_mono)

    @property
    def cooldown_until_utc(self) -> float:
        """表示・互換用。壁時計の絶対時刻（無ければ 0）。判定には使わない。"""
        return self.reset_at_utc or 0.0


def _parse_retry_after(value: str, now: float) -> float | None:
    """Retry-After: 秒数 or HTTP-date。

    HTTP 日付は定義上 GMT。タイムゾーン表記の無い naive な日時は UTC として解釈する
    （ローカル時刻として解釈すると、JST では過去とみなされて実質ゼロに丸まり、
    429 を即座に踏み直す。REQ-G4）。
    """
    value = (value or "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, dt.timestamp() - now)


def _first_int(headers: dict, names: tuple[str, ...]) -> int | None:
    lower = {k.lower(): v for k, v in headers.items()}
    for n in names:
        v = lower.get(n.lower())
        if v is None:
            continue
        try:
            return int(float(v))
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def update_from_429(state: RateLimitState, headers: dict,
                    now_wall: float | None = None,
                    now_mono: float | None = None,
                    *, now: float | None = None) -> RateLimitState:
    """429 レスポンスのヘッダーからクールダウンを更新する。

    `now_wall` はヘッダー解析用の壁時計、`now_mono` は deadline 計算用の単調時計。
    `now` は両方へ同じ値を入れる旧来の便宜（テスト用）。実運用では省略する。
    """
    if now is not None:
        if now_wall is None:
            now_wall = now
        if now_mono is None:
            now_mono = now
    now_wall = now_wall if now_wall is not None else time.time()
    now_mono = now_mono if now_mono is not None else time.monotonic()
    lower = {k.lower(): v for k, v in (headers or {}).items()}
    state.last_429_at_utc = now_wall

    retry_after = _parse_retry_after(lower.get("retry-after", ""), now_wall)
    reset_epoch = None
    for name in ("x-ratelimit-reset", "x-ratelimit-reset-requests", "ratelimit-reset"):
        raw = lower.get(name)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        # NaN / inf / 負値は「ヘッダーで明示されたリセット」として扱わない。
        # 過去・非数のデッドラインは in_cooldown() を常に False にし、レート制限中の
        # 接続へ即再試行してしまうため、推測クールダウンへフォールバックさせる。
        if not math.isfinite(val) or val < 0:
            continue
        # 大きい値は epoch 秒、小さい値は「あと N 秒」とみなす。
        candidate = val if val > 1e6 else now_wall + val
        # 既に過ぎた（または現在時刻以前の）リセットは無効。推測クールダウンへ倒す。
        if candidate <= now_wall:
            continue
        reset_epoch = candidate
        break

    state.rpm = _first_int(headers or {}, ("x-ratelimit-limit-requests", "x-ratelimit-limit", "ratelimit-limit"))
    state.tpm = _first_int(headers or {}, ("x-ratelimit-limit-tokens",))

    if retry_after is not None:
        state.retry_after_s = retry_after
        remaining = min(retry_after, ESTIMATED_COOLDOWN_CAP_S)
        state.source = "header"
    elif reset_epoch is not None:
        state.reset_at_utc = reset_epoch
        remaining = min(max(0.0, reset_epoch - now_wall), ESTIMATED_COOLDOWN_CAP_S)
        state.source = "header"
    else:
        remaining = float(ESTIMATED_COOLDOWN_DEFAULT_S)
        state.source = "estimated"

    # 判定は単調時計の deadline で行う（REQ-G5）。壁時計の絶対値は表示用に持つ。
    state.cooldown_deadline_mono = now_mono + remaining
    if state.reset_at_utc is None:
        state.reset_at_utc = now_wall + remaining
    return state


def clear(state: RateLimitState) -> RateLimitState:
    state.cooldown_deadline_mono = None
    state.reset_at_utc = None
    state.source = ""
    return state
