"""同一モデルの接続候補選定とフォールバック順の決定（260901_VLM_spec.md 7・14.4章）。

ここが決めるのは「どの接続を、どの順で試すか」まで。実際のリトライ実行ループは
vlm_worker 側に置く（Phase 5）。別モデルへの自動切り替えは絶対に行わない。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from vlm_connections import ConnectionKind, VlmConnection
from vlm_models import ModelIdentityStatus, VlmModelProfile


class ExecutionMode(str, Enum):
    BUILTIN_FALLBACK = "builtin_fallback"
    CUSTOM_SINGLE = "custom_single"


def parse_execution_mode(raw: object) -> ExecutionMode:
    if isinstance(raw, ExecutionMode):
        return raw
    try:
        # str, Enum のメンバーでも str(member) は "ExecutionMode.X" になり得るため
        # .value を先に取る。文字列・その他はそのまま正規化して照合する。
        return ExecutionMode(str(getattr(raw, "value", raw)).strip().lower())
    except (ValueError, AttributeError):
        return ExecutionMode.BUILTIN_FALLBACK


_REJECTION_HINTS = {
    "custom_connection_not_found": "selected custom connection was not found; check the saved connection and selected ID",
    "selected_connection_is_not_custom": "selected route is not a custom connection; choose a custom route",
    "custom_connection_disabled": "custom connection is disabled; enable it in custom connection settings",
    "custom_connection_no_auth": "custom connection requires a credential; enter an API key or change auth type to None",
    "no_verified_candidate": "no eligible VLM route is available; check model IDs, API keys, identity, and enabled routes",
}

# 拒否理由コード → 表示用の翻訳キー（[Vlm] セクション）。理由コード自体は
# 保存・判定に使う安定IDなので英語のまま、画面へ出すときだけ訳す。
REJECTION_REASON_KEYS = {
    "custom_connection_not_found": "Reject_Custom_Connection_Not_Found",
    "selected_connection_is_not_custom": "Reject_Selected_Connection_Is_Not_Custom",
    "custom_connection_disabled": "Reject_Custom_Connection_Disabled",
    "custom_connection_no_auth": "Reject_Custom_Connection_No_Auth",
    "no_verified_candidate": "Reject_No_Verified_Candidate",
}
_REJECTION_FALLBACK_KEY = "Reject_Unknown"

# CandidateSet.excluded に入る除外理由コード → 表示用の翻訳キー。
EXCLUSION_REASON_KEYS = {
    "disabled": "Exclude_Disabled",
    "not_verified": "Exclude_Not_Verified",
    "identity_unknown": "Exclude_Identity_Unknown",
    "quantization_unknown": "Exclude_Quantization_Unknown",
    "no_auth": "Exclude_No_Auth",
    "no_image_support": "Exclude_No_Image_Support",
    "cooldown": "Exclude_Cooldown",
}


def explain_rejected_reason(reason: str) -> str:
    """候補選択で停止した理由へ、確認すべき設定を付ける（英語・ログ用）。"""
    raw = str(reason or "unknown")
    return f"{raw}: {_REJECTION_HINTS.get(raw, 'check VLM route settings and connection availability')}"


def explain_candidate_failure(reason: str, excluded: dict[str, str] | None = None) -> str:
    """候補が空になった理由と、接続ごとの除外理由をまとめて返す（英語・ログ用）。"""
    message = explain_rejected_reason(reason)
    if not excluded:
        return message
    details = ", ".join(f"{connection_id}={why}" for connection_id, why in excluded.items())
    return f"{message}; excluded candidates: {details}"


def _translated(get_string, key: str, fallback: str) -> str:
    text = get_string("Vlm", key)
    return fallback if not text or text == key else text


def localized_candidate_failure(reason: str, excluded: dict[str, str] | None,
                                get_string) -> str:
    """候補が空になった理由を、利用者の言語で1行にまとめる。

    従来は explain_candidate_failure() の英語をそのまま翻訳済みメッセージの
    `{reason}` へ差し込んでいたため、どの言語でも理由と対処案だけ英語だった。
    理由コード・除外理由コードは安定IDのまま保ち、ここで表示用に訳す
    （コード自体も併記する: 実機での問い合わせ時に言語を跨いで照合できる）。
    """
    raw = str(reason or "unknown")
    key = REJECTION_REASON_KEYS.get(raw, _REJECTION_FALLBACK_KEY)
    fallback = _REJECTION_HINTS.get(
        raw, "check VLM route settings and connection availability")
    message = f"{raw}: {_translated(get_string, key, fallback)}"
    if not excluded:
        return message
    details = ", ".join(
        f"{connection_id}="
        f"{_translated(get_string, EXCLUSION_REASON_KEYS.get(why, ''), why) if why in EXCLUSION_REASON_KEYS else why}"
        for connection_id, why in excluded.items())
    joined = _translated(get_string, "Exclude_List_Prefix", "excluded candidates")
    return f"{message}; {joined}: {details}"


@dataclass
class RouterPolicy:
    execution_mode: ExecutionMode = ExecutionMode.BUILTIN_FALLBACK
    selected_connection_id: str | None = None
    # 「サポート済みのVLMサービスを使う」経路で、identity が DECLARED（同一と宣言済みだが
    # 実測未確認）の binding も候補に含めるか。どのサービスをどの順で試すかは利用者が
    # 経路リストで決めるため、既定は True。UNKNOWN は常に除外する。
    allow_declared_identity: bool = True


@dataclass
class CandidateSet:
    """選定結果。connection_ids が空なら理由が rejected_reason に入る。"""
    connection_ids: list[str] = field(default_factory=list)
    # connection_id -> 除外理由（UI 表示・ログ用）
    excluded: dict[str, str] = field(default_factory=dict)
    rejected_reason: str = ""

    @property
    def has_candidates(self) -> bool:
        return bool(self.connection_ids)


def select_candidates(
    profile: VlmModelProfile,
    connections: dict[str, VlmConnection],
    policy: RouterPolicy,
    *,
    cooldown_until: dict[str, float] | None = None,
    has_auth: dict[str, bool] | None = None,
    supports_image: dict[str, bool] | None = None,
    now: float | None = None,
) -> CandidateSet:
    """処理開始時に一度だけ呼ぶ。結果はジョブスナップショットへ固定する。"""
    now = now if now is not None else time.time()
    cooldown_until = cooldown_until or {}
    has_auth = has_auth or {}
    supports_image = supports_image or {}
    result = CandidateSet()

    if policy.execution_mode is ExecutionMode.CUSTOM_SINGLE:
        return _select_custom_single(connections, policy, cooldown_until, has_auth, now)

    # --- builtin_fallback ---
    # プロファイルの bindings 挿入順 = 既定の優先順位。UI で並べ替えた順は
    # connections 側の順序で表現される想定なので、両方を尊重して交差を取る。
    ordered_provider_ids = [pid for pid in profile.bindings if _provider_has_connection(pid, connections)]
    for provider_id in ordered_provider_ids:
        conn = _builtin_connection_for(provider_id, connections)
        if conn is None:
            continue
        cid = conn.connection_id
        binding = profile.binding_for(provider_id)

        if not conn.enabled:
            result.excluded[cid] = "disabled"
            continue
        if binding is None:
            result.excluded[cid] = "not_verified"
            continue
        status = binding.effective_identity_status()
        if status is ModelIdentityStatus.UNKNOWN:
            result.excluded[cid] = "identity_unknown"
            continue
        if (not policy.allow_declared_identity
                and status is ModelIdentityStatus.VERIFIED
                and not profile.quantization_is_strict()):
            result.excluded[cid] = "quantization_unknown"
            continue
        if status is not ModelIdentityStatus.VERIFIED and not policy.allow_declared_identity:
            result.excluded[cid] = "not_verified"
            continue
        if not has_auth.get(cid, False):
            result.excluded[cid] = "no_auth"
            continue
        if not supports_image.get(cid, True):
            result.excluded[cid] = "no_image_support"
            continue
        if cooldown_until.get(cid, 0.0) > now:
            result.excluded[cid] = "cooldown"
            continue
        result.connection_ids.append(cid)

    if not result.connection_ids:
        # 量子化不明で厳格判定できないケースは、そもそも VERIFIED にならない前提。
        result.rejected_reason = "no_verified_candidate"
    return result


def _select_custom_single(connections, policy, cooldown_until, has_auth, now) -> CandidateSet:
    result = CandidateSet()
    cid = policy.selected_connection_id
    conn = connections.get(cid) if cid else None
    if conn is None:
        result.rejected_reason = "custom_connection_not_found"
        return result
    if not conn.is_custom:
        result.rejected_reason = "selected_connection_is_not_custom"
        return result
    if not conn.enabled:
        result.excluded[cid] = "disabled"
        result.rejected_reason = "custom_connection_disabled"
        return result
    if conn.auth.type != "none" and not has_auth.get(cid, False):
        result.excluded[cid] = "no_auth"
        result.rejected_reason = "custom_connection_no_auth"
        return result
    if cooldown_until.get(cid, 0.0) > now:
        result.excluded[cid] = "cooldown"
        result.rejected_reason = "custom_connection_cooldown"
        return result
    result.connection_ids.append(cid)
    return result


def _provider_has_connection(provider_id: str, connections: dict[str, VlmConnection]) -> bool:
    return _builtin_connection_for(provider_id, connections) is not None


def _builtin_connection_for(provider_id: str, connections: dict[str, VlmConnection]) -> VlmConnection | None:
    for conn in connections.values():
        if conn.kind is ConnectionKind.BUILTIN and conn.provider_id == provider_id:
            return conn
    return None
