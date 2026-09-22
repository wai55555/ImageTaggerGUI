"""VLM リクエスト結果のエラー分類（260901_VLM_spec.md 8章 / implement_plan 10.2節）。

「どのエラーで同一接続リトライ / 次の接続へ / 除外 / 画像失敗 / ジョブ停止 とするか」を
一箇所に集約する。判断はここだけで行い、Worker / Router には分岐を持ち込まない。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VlmErrorClass(str, Enum):
    RETRY_SAME = "retry_same"      # 同一接続で再試行
    FAILOVER = "failover"          # 次の同一モデル接続へ
    EXCLUDE = "exclude"            # 今回の処理からこの接続を除外
    FAIL_IMAGE = "fail_image"      # この画像を失敗として残す
    STOP_JOB = "stop_job"          # ジョブ全体を停止（設定ミスの可能性）


class VlmErrorReason(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"          # HTTP 429
    SERVER_ERROR = "server_error"          # HTTP 5xx
    AUTH_ERROR = "auth_error"              # 401 / 403（ポリシー拒否を除く）
    MODEL_UNSUPPORTED = "model_unsupported"
    IMAGE_FORMAT_ERROR = "image_format_error"
    PROMPT_FORMAT_ERROR = "prompt_format_error"
    CONTENT_POLICY = "content_policy"
    EMPTY_RESPONSE = "empty_response"
    OUTPUT_LIMIT = "output_limit"
    BAD_RESPONSE = "bad_response"          # JSON 不正 / 抽出パス不一致
    NETWORK = "network"                    # DNS / TCP / TLS
    UNKNOWN = "unknown"


# VlmErrorReason → 表示ラベルの翻訳キー（[Vlm] セクション）。
# enum の値（"timeout" 等）は保存・判定に使う安定IDなので英語のまま固定し、
# 画面へ出すときだけここを引いて訳す。
REASON_LABEL_KEYS: dict[str, str] = {
    VlmErrorReason.TIMEOUT.value: "Error_Reason_Timeout",
    VlmErrorReason.RATE_LIMITED.value: "Error_Reason_Rate_Limited",
    VlmErrorReason.SERVER_ERROR.value: "Error_Reason_Server_Error",
    VlmErrorReason.AUTH_ERROR.value: "Error_Reason_Auth_Error",
    VlmErrorReason.MODEL_UNSUPPORTED.value: "Error_Reason_Model_Unsupported",
    VlmErrorReason.IMAGE_FORMAT_ERROR.value: "Error_Reason_Image_Format_Error",
    VlmErrorReason.PROMPT_FORMAT_ERROR.value: "Error_Reason_Prompt_Format_Error",
    VlmErrorReason.CONTENT_POLICY.value: "Error_Reason_Content_Policy",
    VlmErrorReason.EMPTY_RESPONSE.value: "Error_Reason_Empty_Response",
    VlmErrorReason.OUTPUT_LIMIT.value: "Error_Reason_Output_Limit",
    VlmErrorReason.BAD_RESPONSE.value: "Error_Reason_Bad_Response",
    VlmErrorReason.NETWORK.value: "Error_Reason_Network",
    VlmErrorReason.UNKNOWN.value: "Error_Reason_Unknown",
}


def reason_label_key(reason: object) -> str:
    """この理由の表示ラベル翻訳キー。未知なら空文字。"""
    return REASON_LABEL_KEYS.get(str(getattr(reason, "value", reason)), "")


def reason_label(reason: object, get_string) -> str:
    """理由の表示ラベル。翻訳が引けなければ enum の値をそのまま返す。"""
    key = reason_label_key(reason)
    raw = str(getattr(reason, "value", reason))
    if not key:
        return raw
    text = get_string("Vlm", key)
    return raw if not text or text == key else text


@dataclass(frozen=True)
class VlmAttemptError:
    reason: VlmErrorReason
    http_status: int | None = None
    message: str = ""
    # サーバーが返したエラーコード文字列（あれば）。ログ用。
    provider_code: str = ""
    # 表示用の翻訳キーと差し込み値。`message` は英語のまま残す:
    # デバッグログは英語が望ましく、api_key_dialog / is_billing_or_credit_block
    # などが message を文字列判定に使っているため、ここを訳文に差し替えると壊れる。
    message_key: str = ""
    message_args: dict = field(default_factory=dict)

    def classify(self, *, consecutive_timeouts: int = 0,
                 already_retried_same: bool = False,
                 same_retries: int | None = None,
                 retry_same_max: int = 1,
                 retry_5xx: bool = True) -> VlmErrorClass:
        """このエラーに対する行動を返す（spec.md 8.2 の表）。

        - consecutive_timeouts: この接続で連続何回目のタイムアウトか（1 が初回）
        - already_retried_same: 同一接続でのリトライを今回すでに1回使ったか
        """
        r = self.reason
        explicit_retry_count = same_retries is not None
        retries = (int(same_retries) if explicit_retry_count
                   else (1 if already_retried_same else 0))
        can_retry_same = max(0, int(retry_same_max)) > retries
        if r is VlmErrorReason.TIMEOUT:
            # Preserve the legacy consecutive-timeout guard for callers that have
            # not migrated to an explicit retry counter.  New callers can opt into
            # retry_same_max > 1 by passing same_retries.
            if can_retry_same and (explicit_retry_count or consecutive_timeouts <= 1):
                return VlmErrorClass.RETRY_SAME
            # 2画像分の再試行（(retry_same_max+1)×2回）を使い切ってもなお毎回
            # タイムアウトしているなら、この接続はセッション中ずっと壊れている
            # 可能性が高い。auth_error 等と同様に EXCLUDE へ昇格し、以後の画像
            # から自動的に外す——さもないと、無反応な1接続だけで画像1枚あたり
            # 最大で (retry_same_max+1)×read_timeout_s を毎回無駄にし続ける
            # （2026-09 VLM デバッグ: 実機で NVIDIA 1接続が11枚中4枚で毎回
            # ちょうど120秒ずつ無駄にしていたのを確認）。
            #
            # 閾値は「1画像分」ではなく「2画像分」にしている: 実際の Gemini は
            # 完全には壊れておらず単に間欠的に遅い／タイムアウトすることがあり
            # （実機11枚バッチで4回中2回成功）、1画像分（consecutive_timeouts>=2）
            # で即除外すると、たまたま1枚だけ運悪くタイムアウトが重なっただけの
            # 健全な接続まで残りセッション全体から締め出してしまい、さらに他の
            # 候補が同時に一時的なレート制限中だと「使える接続が1つも無い」で
            # バッチ全体が中断する事故につながった（2026-09 実機11枚バッチで、
            # 4枚目でGeminiが誤って除外され、5枚目以降7枚が一度も試行されずに
            # 打ち切られたのを確認）。2画像分の全滅を要求することで、本当に
            # 恒常的に壊れている接続（NVIDIA相当）だけを狙って除外する。
            # 1画像あたりの試行回数は vlm_transport の
            # `max_same_conn_attempts = max(0, retry_same_max) + 1` が基準。
            # ここを max(1, ...) にしていると retry_same_max=0（カスタム接続の
            # スピンボックスは下限0なので設定できる）のとき閾値が2ではなく4に
            # なり、コメントの「2画像分」と食い違って除外が2枚ぶん遅れる。
            if consecutive_timeouts >= (max(0, int(retry_same_max)) + 1) * 2:
                return VlmErrorClass.EXCLUDE
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.RATE_LIMITED:
            return VlmErrorClass.FAILOVER          # 待機しない
        if r is VlmErrorReason.SERVER_ERROR:
            return (VlmErrorClass.RETRY_SAME
                    if retry_5xx and can_retry_same else VlmErrorClass.FAILOVER)
        if r in (VlmErrorReason.AUTH_ERROR, VlmErrorReason.MODEL_UNSUPPORTED):
            return VlmErrorClass.EXCLUDE
        if r is VlmErrorReason.CONTENT_POLICY:
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.EMPTY_RESPONSE:
            return VlmErrorClass.RETRY_SAME if can_retry_same else VlmErrorClass.FAILOVER
        if r is VlmErrorReason.OUTPUT_LIMIT:
            # 設定値を変えない限り同じ応答になるため、同一接続では無駄に再試行しない。
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.IMAGE_FORMAT_ERROR:
            # 呼び出し側が画像を作り直せたら retry、無理なら failover。ここでは failover を既定に。
            return VlmErrorClass.FAILOVER
        if r is VlmErrorReason.PROMPT_FORMAT_ERROR:
            return VlmErrorClass.STOP_JOB
        if r in (VlmErrorReason.BAD_RESPONSE, VlmErrorReason.NETWORK, VlmErrorReason.UNKNOWN):
            return VlmErrorClass.FAILOVER
        return VlmErrorClass.FAILOVER


def _looks_like_prompt_format(message: str, provider_code: str = "") -> bool:
    low = f"{provider_code} {message}".lower()
    # "request body" のような汎用句は含めない。ペイロード過大など画像単位で解消する
    # 400 まで PROMPT_FORMAT_ERROR (=STOP_JOB) 扱いになり、バッチ全体を止めてしまう。
    return any(marker in low for marker in (
        "messages[", "content must be", "content[", "image_url", "input_image",
        "inline_data", "inlineimage", "multimodal", "prompt format",
    ))


def reason_from_http_status(status: int, message: str = "",
                            provider_code: str = "") -> VlmErrorReason:
    if status == 408:
        return VlmErrorReason.TIMEOUT
    if status == 429:
        return VlmErrorReason.RATE_LIMITED
    if status in (401, 403):
        return VlmErrorReason.AUTH_ERROR
    if status == 404:
        return VlmErrorReason.MODEL_UNSUPPORTED
    if status == 400 and _looks_like_prompt_format(message, provider_code):
        return VlmErrorReason.PROMPT_FORMAT_ERROR
    if 500 <= status <= 599:
        return VlmErrorReason.SERVER_ERROR
    if 400 <= status <= 499:
        return VlmErrorReason.BAD_RESPONSE
    if status == 200:
        # 200 なのにここへ来る = 本文が JSON でない／抽出パスに合致しない（本文が壊れている）。
        return VlmErrorReason.BAD_RESPONSE
    return VlmErrorReason.UNKNOWN


def attempt_error_text(error: "VlmAttemptError", get_string) -> str:
    """VlmAttemptError を利用者向けの1行へ整える（理由ラベル + 詳細）。

    `message_key` を持つエラーはそれを訳し、持たないものは英語の `message` を
    そのまま出す（サーバーが返した本文など、そもそも翻訳対象ではないもの）。
    """
    reason = reason_label(error.reason, get_string)
    body = ""
    if error.message_key:
        body = get_string("Vlm", error.message_key, **error.message_args)
        if not body or body == error.message_key:
            body = error.message
    else:
        body = error.message
    return f"{reason}: {body}" if body else reason
