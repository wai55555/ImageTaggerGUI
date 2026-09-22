"""生成プロファイルとプロンプト組み立て（260901_VLM_spec.md 4・5章 / design.md 4.3節）。

生成プロファイルはモデルプロファイルとは独立。処理開始時に丸ごとスナップショット化して
読み取り専用にし、処理中の UI 変更に影響されないようにする（NFR-004 / spec 5.4節）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class DetailLevel(str, Enum):
    STANDARD = "standard"
    DETAILED = "detailed"
    MAXIMUM_DETAIL = "maximum_detail"


class SentenceMode(str, Enum):
    AUTOMATIC = "automatic_long_detailed"
    S1 = "1"
    S2 = "2"
    S3 = "3"
    S4 = "4"
    S5 = "5"


class CharacterNameMode(str, Enum):
    DO_NOT_IDENTIFY = "do_not_identify"
    EXPLICIT_ONLY = "explicit_only"
    ALLOW_GUESSING = "allow_guessing"


class MarkdownMode(str, Enum):
    DISABLED = "disabled"
    ALLOWED = "allowed"


class PromptMode(str, Enum):
    """Captioning prompt family selected by the user."""

    STANDARD = "standard"
    DATASET_LONG = "dataset_long"
    SHORT_TAGS = "short_tags"


# VLM の出力トークン上限。既定値・設定ファイルの初期値・UIの入力範囲・エラー時の
# 助言がバラバラに固定されていると、既定や最大を変えたときに保存値の補正・UI・
# 文言が食い違う。ここを唯一の定義にして app_settings / vlm_settings_dialog /
# vlm_transport から参照する。
#
# 既定 3072: 推論系VLMは最終キャプションを出す前に補完枠をかなり消費する。3072 は
# その余地を残しつつ、通常の画像プロンプト分を差し引いても 4096 コンテキストの
# ローカルサーバーに収まる。
DEFAULT_MAX_OUTPUT_TOKENS = 3072
# 下限 16: これ未満だと最終テキストが1文も出ない。上限 32768: プロバイダー側の
# 上限より大きく設定しても意味がないため、入力の暴走を止めるためだけの値。
MIN_MAX_OUTPUT_TOKENS = 16
MAX_MAX_OUTPUT_TOKENS = 32768


def _parse_enum(enum_cls, raw: object, default):
    try:
        return enum_cls(str(raw).strip().lower())
    except (ValueError, AttributeError):
        return default


@dataclass(frozen=True)
class GenerationProfile:
    """1回のキャプション生成に使う設定一式。frozen=スナップショットとして安全。"""
    profile_id: str = "default-caption-en"
    language: str = "en"
    detail_level: DetailLevel = DetailLevel.MAXIMUM_DETAIL
    sentence_mode: SentenceMode = SentenceMode.AUTOMATIC
    character_name_mode: CharacterNameMode = CharacterNameMode.EXPLICIT_ONLY
    markdown: MarkdownMode = MarkdownMode.DISABLED
    prompt_mode: PromptMode = PromptMode.STANDARD
    # provider 既定に任せる場合は None。
    temperature: float | None = None
    top_p: float | None = None
    # 既定値の根拠は DEFAULT_MAX_OUTPUT_TOKENS のコメントを参照。
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    # 上級者向け: これが空でなければ system プロンプトを完全に置き換える（spec 4.6節）。
    custom_system_prompt: str = ""
    # 画像前処理の設定（vlm_image.ImagePreprocessConfig をそのまま持たせず、値だけ）。
    image_max_long_edge: int = 1536
    image_format: str = "auto"  # auto | jpeg | png
    image_jpeg_quality: int = 90

    @classmethod
    def from_mapping(cls, data: dict) -> "GenerationProfile":
        """config / JSON からの復元。未知キー無視、型違いは既定へフォールバック。"""
        base = cls()
        return replace(
            base,
            profile_id=str(data.get("profile_id", base.profile_id)),
            language=str(data.get("language", base.language)) or base.language,
            detail_level=_parse_enum(DetailLevel, data.get("detail_level"), base.detail_level),
            sentence_mode=_parse_enum(SentenceMode, data.get("sentence_mode"), base.sentence_mode),
            character_name_mode=_parse_enum(CharacterNameMode, data.get("character_name_mode"), base.character_name_mode),
            markdown=_parse_enum(MarkdownMode, data.get("markdown"), base.markdown),
            prompt_mode=_parse_enum(PromptMode, data.get("prompt_mode"), base.prompt_mode),
            temperature=_opt_float(data.get("temperature")),
            top_p=_opt_float(data.get("top_p")),
            max_output_tokens=_clamp_int(data.get("max_output_tokens"), base.max_output_tokens,
                                         lo=MIN_MAX_OUTPUT_TOKENS, hi=MAX_MAX_OUTPUT_TOKENS),
            custom_system_prompt=str(data.get("custom_system_prompt", "") or ""),
            image_max_long_edge=_clamp_int(data.get("image_max_long_edge"), base.image_max_long_edge, lo=256, hi=8192),
            image_format=_norm_image_format(data.get("image_format", base.image_format)),
            image_jpeg_quality=_clamp_int(data.get("image_jpeg_quality"), base.image_jpeg_quality, lo=1, hi=100),
        )


def _opt_float(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _clamp_int(raw: object, default: int, *, lo: int, hi: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(n, hi))


_IMAGE_FORMATS = ("auto", "jpeg", "png")


def _norm_image_format(raw: object) -> str:
    """未対応の image_format（gif など）は auto へ寄せる。プレビュー表示と実際の
    送信形式が食い違わないよう、ここで既定値へ正規化しておく。"""
    v = str(raw or "auto").strip().lower()
    return v if v in _IMAGE_FORMATS else "auto"


# --- プロンプト組み立て（spec.md 5章） ----------------------------------------------

_BASE_INSTRUCTION = (
    "Write a highly detailed natural-language caption for this image.\n"
    "Describe only what is visibly supported by the image.\n"
    "Do not invent names, events, locations, or details that cannot be seen.\n"
    "Do not output an introductory explanation or a trailing summary."
)

_DETAIL_CLAUSE = {
    DetailLevel.STANDARD:
        "Cover the main subject, its appearance and clothing, the pose, and the "
        "overall setting in clear plain language.",
    DetailLevel.DETAILED:
        "Describe the subjects, their appearance, hair, face, clothing, pose, the "
        "composition, the background, notable objects, colors, and lighting.",
    DetailLevel.MAXIMUM_DETAIL:
        "Describe, as thoroughly as the image supports, every visible person, their "
        "clothing, hair, face, expression and pose, the composition and framing, the "
        "background, all notable objects, the colors, the lighting, and the spatial "
        "relationships between elements in the frame.",
}

_SENTENCE_CLAUSE_AUTOMATIC = (
    "Use as many sentences as necessary to provide the most detailed useful "
    "description possible within the output limit."
)

_CHARACTER_CLAUSE = {
    CharacterNameMode.DO_NOT_IDENTIFY:
        "Do not state any character name, person name, or franchise/series title.",
    CharacterNameMode.EXPLICIT_ONLY:
        "Only name a character, person, or franchise if it is unambiguously clear from "
        "the image itself or from information explicitly given to you; otherwise describe "
        "them without naming.",
    CharacterNameMode.ALLOW_GUESSING:
        "You may name a likely character, person, or franchise even if you are not "
        "certain, but keep the visual description accurate.",
}

_MARKDOWN_DISABLED_CLAUSE = (
    "Output a single plain-text paragraph block. Do not use Markdown: no headings, no "
    "bullet lists, no numbered lists, no code fences, no bold or italic markers."
)

# These are intentionally separate prompt families rather than a single prompt
# with a variable detail level.  The three families reflect three different
# downstream uses: ordinary annotation, dense training captions, and compact
# tag-like output.  Keep all three factual and image-grounded; the training
# caption mode must not turn visible uncertainty into invented metadata.
_DATASET_LONG_INSTRUCTION = (
    "Write a dense training caption for this image as one natural-language "
    "paragraph.\n"
    "Describe only what is visibly supported by the image, including the main "
    "subject(s), count, appearance, hair, face, expression, clothing, pose, action, "
    "composition, camera angle, framing, spatial relationships, background, notable "
    "objects, colors, materials, lighting, atmosphere, and visible artistic medium or "
    "style.\n"
    "Transcribe readable text exactly when it is legible; if text, a name, identity, "
    "location, event, or other detail is uncertain, do not guess it.\n"
    "Do not add a trigger word, dataset metadata, an introductory explanation, a "
    "trailing summary, or Markdown."
)

_SHORT_TAGS_INSTRUCTION = (
    "Convert this image into a concise list of comma-separated visual tags.\n"
    "Prioritize the main subject(s), count, visible appearance, hair, clothing, pose, "
    "action, setting, composition, colors, lighting, and clearly visible style or "
    "medium.\n"
    "Use short concrete lowercase phrases where natural. Include readable text only "
    "when it is legible. Do not guess names, identities, franchises, locations, or "
    "other unsupported details. Output only tags separated by commas, with no "
    "explanation, sentence, Markdown, weights, or quality judgment."
)


def _sentence_clause(mode: SentenceMode) -> str:
    if mode is SentenceMode.AUTOMATIC:
        return _SENTENCE_CLAUSE_AUTOMATIC
    n = int(mode.value)
    # 「exactly N」とは言わない: 後処理で機械的に分割・結合しないため（spec 5.3節）。
    return f"Write approximately {n} sentence{'s' if n != 1 else ''}."


def build_system_prompt(profile: GenerationProfile) -> str:
    """生成プロファイルから system プロンプトを組み立てる。custom があればそれを優先。"""
    if profile.custom_system_prompt.strip():
        return profile.custom_system_prompt.strip()

    if profile.prompt_mode is PromptMode.DATASET_LONG:
        parts = [_DATASET_LONG_INSTRUCTION, _language_clause(profile.language)]
        return "\n".join(parts)

    if profile.prompt_mode is PromptMode.SHORT_TAGS:
        parts = [_SHORT_TAGS_INSTRUCTION, _language_clause(profile.language)]
        return "\n".join(parts)

    parts = [_BASE_INSTRUCTION, _DETAIL_CLAUSE[profile.detail_level],
             _sentence_clause(profile.sentence_mode),
             _CHARACTER_CLAUSE[profile.character_name_mode]]
    if profile.markdown is MarkdownMode.DISABLED:
        parts.append(_MARKDOWN_DISABLED_CLAUSE)
    parts.append(_language_clause(profile.language))
    return "\n".join(parts)


def _language_clause(language: str) -> str:
    raw = str(language or "en").strip()
    labels = {
        "en": "English", "english": "English", "ja": "Japanese", "japanese": "Japanese",
        "zh": "Chinese", "zh-cn": "Chinese", "zh_tw": "Traditional Chinese",
        "ko": "Korean", "fr": "French", "de": "German", "es": "Spanish",
    }
    return f"Write the output in {labels.get(raw.lower(), raw or 'English')}."


def build_user_prompt(profile: GenerationProfile) -> str:
    """画像に添える user メッセージ本文。短く固定でよい（詳細指示は system 側）。"""
    if profile.prompt_mode is PromptMode.DATASET_LONG:
        return "Create a training caption for this image."
    if profile.prompt_mode is PromptMode.SHORT_TAGS:
        return "Convert this image into concise comma-separated tags."
    return "Caption this image."


DEFAULT_GENERATION_PROFILE = GenerationProfile()
