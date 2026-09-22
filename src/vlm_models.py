"""VLM モデルプロファイルと同一性判定（260901_VLM_spec.md 3章 / design.md 4.1・5.1節）。

内蔵サービス間のフォールバックは「サービス」ではなく「同一モデルプロファイル」を単位に
する。サービスごとの呼び名の違いは対応表（bindings）で吸収し、`vlm_protocols.py` などの
通信コードには分散させない。量子化方式・ベースモデル・リビジョンが違うものは同一モデルと
みなさない（別モデルへの自動切り替えは一切行わない）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

_TOKEN_SPLIT = re.compile(r"[/:@_.\-\s]+")
# 「同一モデル」を判定するのに効くトークン（サイズ・世代・ファミリー）。汎用語は除外。
_STOP_TOKENS = frozenset({"", "free", "instruct", "it", "chat", "latest", "preview",
                          "vision", "vl", "model", "google", "meta", "cf", "api"})


def _sig_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_SPLIT.split((text or "").lower()) if t and t not in _STOP_TOKENS}


class ModelIdentityStatus(str, Enum):
    """内蔵サービスが提供するモデルが、プロファイルの正規モデルと同一と言えるか。

    - VERIFIED: 厳格フォールバックで使用可能
    - DECLARED: 同一と宣言されているが、厳格フォールバックでは使わない（警告対象）
    - UNKNOWN : 判定不能。厳格フォールバックから除外
    """
    VERIFIED = "verified"
    DECLARED = "declared"
    UNKNOWN = "unknown"


def parse_identity_status(raw: object) -> ModelIdentityStatus:
    """手書き JSON からの値を検証する。不明・型違いは UNKNOWN（安全側）へ。"""
    try:
        return ModelIdentityStatus(str(raw).strip().lower())
    except (ValueError, AttributeError):
        return ModelIdentityStatus.UNKNOWN


@dataclass(frozen=True)
class ProviderConstraint:
    """OpenRouter のように内部プロバイダーが変わりうるサービスの固定条件。

    承認済みプロバイダーへ固定でき、かつ自動プロバイダーフォールバックを無効化できた
    場合だけ VERIFIED を名乗れる（spec.md 3.2節）。
    """
    allowed_providers: tuple[str, ...] = ()
    allow_fallbacks: bool = True

    @property
    def is_pinned(self) -> bool:
        return bool(self.allowed_providers) and not self.allow_fallbacks


@dataclass(frozen=True)
class ModelBinding:
    """1つの内蔵プロバイダーにおける、このモデルプロファイルの実体。"""
    provider_id: str
    model_id: str
    identity_status: ModelIdentityStatus = ModelIdentityStatus.UNKNOWN
    provider_constraint: ProviderConstraint | None = None
    # User profiles persist this only after the model list/static catalog has
    # confirmed image input + text output. It survives process-local discovery.
    vlm_capable: bool = False
    def effective_identity_status(self) -> ModelIdentityStatus:
        """provider_constraint を固定できていない場合、VERIFIED を格下げする。"""
        status = self.identity_status
        if status is ModelIdentityStatus.VERIFIED and self.provider_constraint is not None:
            if not self.provider_constraint.is_pinned:
                return ModelIdentityStatus.DECLARED
        return status

    def is_strict_fallback_eligible(self) -> bool:
        """通常（厳格）フォールバックの候補になれるか。VERIFIED のみ True。"""
        return self.effective_identity_status() is ModelIdentityStatus.VERIFIED


@dataclass(frozen=True)
class VlmModelProfile:
    """利用者が選ぶのはサービス名ではなくこのプロファイル（requirement FR-003）。"""
    profile_id: str
    display_name: str
    canonical_model_id: str
    family: str = ""
    base_model: str = ""
    revision: str = ""
    # unquantized_required / int8 / awq / gguf-q4 など。空・不明は厳格判定から除外する材料。
    quantization: str = "unknown"
    bindings: dict[str, ModelBinding] = field(default_factory=dict)
    # 別名（サービスの生モデルID等）から profile_id を引くための逆引き補助。
    aliases: tuple[str, ...] = ()

    def binding_for(self, provider_id: str) -> ModelBinding | None:
        return self.bindings.get(provider_id)

    def strict_fallback_providers(self) -> list[str]:
        """厳格フォールバックの対象になる provider_id を、bindings 挿入順で返す。"""
        return [pid for pid, b in self.bindings.items() if b.is_strict_fallback_eligible()]

    def quantization_is_strict(self) -> bool:
        """量子化方式が厳格判定に足るか（不明・空は不可。spec.md 3章 / plan 3.1節）。"""
        q = self.quantization.strip().lower()
        return bool(q) and q != "unknown"


class VlmModelRegistry:
    """内蔵モデルプロファイルの集合。model_registry.py（ローカル ONNX 用）とは混在させない。"""

    def __init__(self, profiles: list[VlmModelProfile] | None = None):
        self._profiles: dict[str, VlmModelProfile] = {}
        for p in profiles or []:
            self.add(p)

    def add(self, profile: VlmModelProfile) -> None:
        self._profiles[profile.profile_id] = profile

    def get(self, profile_id: str) -> VlmModelProfile | None:
        return self._profiles.get(profile_id)

    def all_profiles(self) -> list[VlmModelProfile]:
        return list(self._profiles.values())

    def resolve_alias(self, name: str) -> VlmModelProfile | None:
        """profile_id / canonical_model_id / alias / いずれかの binding.model_id で引く。"""
        key = name.strip()
        if key in self._profiles:
            return self._profiles[key]
        low = key.lower()
        for p in self._profiles.values():
            if p.canonical_model_id.lower() == low:
                return p
            if any(a.lower() == low for a in p.aliases):
                return p
            if any(b.model_id.lower() == low for b in p.bindings.values()):
                return p
        return None


# --- 内蔵プロファイル（spec.md 3章 / implement_plan 2.1・2.2・18章） ------------------
# identity_status は控えめに置く（多くは DECLARED、実在が未確認のものは UNKNOWN）。
# 各プロバイダーでの正確なモデル ID は「接続診断／1枚テスト」を通すまで確定しない前提。
# 診断のフル PASS、認証済み429／診断上限による到達確認、または生成成功で
# `[Vlm] verified_bindings` に載り VERIFIED 扱いになる（vlm_config._apply_verified_promotions）。
#
# providers: gemini / openrouter / cloudflare / groq / nvidia /
#            huggingface / vercel / openai / anthropic
#            （ovhcloud は実機検証できるまで無効）
#   - gemini      : Google Generative Language API（gemini_generate_content）
#   - openrouter  : OpenRouter（openai_chat_completions、`:free` はプロバイダー側のモデルID表記）
#   - cloudflare  : Cloudflare Workers AI（openai_chat_completions、要 account_id）
#   - groq        : Groq（openai_chat_completions、無料枠あり）
#   - nvidia      : NVIDIA NIM / build.nvidia.com（openai_chat_completions、無料クレジット）
#   - mistral     : 一時停止（Pixtral VLMはキャプション用途として弱いため）
#   - huggingface : Hugging Face Inference Providers（openai_chat_completions、従量課金）
#   - vercel      : Vercel AI Gateway（openai_chat_completions、従量課金）
#   - openai      : OpenAI Responses API（従量課金）
#   - anthropic   : Anthropic Messages API（従量課金）
#   - ovhcloud    : OVHcloud AI Endpoints（日本居住者環境で実機検証できるまで無効）

GEMMA_4_26B_A4B_IT = VlmModelProfile(
    profile_id="gemma-4-26b-a4b-it",
    display_name="Gemma 4 26B A4B IT",
    canonical_model_id="gemma-4-26b-a4b-it",
    family="Gemma 4",
    base_model="gemma-4-26b-a4b-it",
    revision="provider_verified",
    quantization="unquantized_required",
    aliases=(
        "google/gemma-4-26b-a4b-it",
        "google/gemma-4-26b-a4b-it:free",
        "@cf/google/gemma-4-26b-a4b-it",
    ),
    bindings={
        "gemini": ModelBinding("gemini", "gemma-4-26b-a4b-it",
                               ModelIdentityStatus.DECLARED),
        "openrouter": ModelBinding("openrouter", "google/gemma-4-26b-a4b-it:free",
                                   ModelIdentityStatus.DECLARED,
                                   ProviderConstraint(allowed_providers=(), allow_fallbacks=True)),
        "cloudflare": ModelBinding("cloudflare", "@cf/google/gemma-4-26b-a4b-it",
                                   ModelIdentityStatus.DECLARED),
        "huggingface": ModelBinding("huggingface", "google/gemma-4-26B-A4B-it",
                                     ModelIdentityStatus.UNKNOWN),
        "vercel": ModelBinding("vercel", "google/gemma-4-26b-a4b-it",
                                ModelIdentityStatus.DECLARED),
    },
)

# --- 2.2 Gemma 4 31B IT。各ホスト型経路の実体は DECLARED とし、軽量確認または
# 生成成功後に `[Vlm] verified_bindings` で VERIFIED へ昇格させる。
GEMMA_4_31B_IT = VlmModelProfile(
    profile_id="gemma-4-31b-it",
    display_name="Gemma 4 31B IT",
    canonical_model_id="gemma-4-31b-it",
    family="Gemma 4",
    base_model="gemma-4-31b-it",
    quantization="provider_managed",
    aliases=("google/gemma-4-31b-it", "google/gemma-4-31b-it:free"),
    bindings={
        "gemini": ModelBinding("gemini", "gemma-4-31b-it", ModelIdentityStatus.DECLARED),
        "openrouter": ModelBinding("openrouter", "google/gemma-4-31b-it:free",
                                   ModelIdentityStatus.DECLARED),
        # Cloudflare does not currently host @cf/google/gemma-4-31b-it (confirmed
        # live: absent from a real /ai/models/search listing of 65 models; a real
        # generation request gets 403 "This account is not allowed to access
        # @cf/google/gemma-4-31b-it" - worded like a permissions error but it is
        # actually an unknown-model error, since the account's API key passed a
        # lightweight auth-only check against the same endpoint). Cloudflare does
        # host @cf/google/gemma-4-26b-a4b-it, but that is a different-sized model
        # already correctly bound to GEMMA_4_26B_A4B_IT below, not a substitute
        # for this one (see looks_same_family()'s "different size -> different
        # model" rule). No cloudflare binding here until Cloudflare ships this
        # exact model.
        "nvidia": ModelBinding("nvidia", "google/gemma-4-31b-it", ModelIdentityStatus.DECLARED),
        # Groq does not host any Gemma vision model (confirmed live: "model
        # gemma-4-31b-it does not exist" from Groq's API, and it is absent
        # from _KNOWN_VISION_MODEL_IDS["groq"] below, which only lists
        # llama-4-scout and qwen3.x). GEMMA_4_26B_A4B_IT correctly has no groq
        # binding either - this one was a stale/incorrect copy-paste.
        "huggingface": ModelBinding("huggingface", "google/gemma-4-31B-it",
                                     ModelIdentityStatus.DECLARED),
        "vercel": ModelBinding("vercel", "google/gemma-4-31b-it",
                                ModelIdentityStatus.DECLARED),
    },
)

QWEN3_8_27B = VlmModelProfile(
    profile_id="qwen3.8-27b",
    display_name="Qwen3.8 27B",
    canonical_model_id="qwen3.8-27b",
    family="Qwen3.8",
    base_model="qwen3.8-27b",
    quantization="unknown",
    aliases=("qwen/qwen3.8-27b", "qwen/qwen3.8-27b-instruct", "qwen3.8-27b-instruct"),
    bindings={
        "openrouter": ModelBinding("openrouter", "qwen/qwen3.8-27b", ModelIdentityStatus.UNKNOWN),
        # NVIDIA binding removed: confirmed live (2026-09-22) that NVIDIA's real
        # /v1/models catalog (82 entries) has no Qwen model of any kind, so
        # "qwen/qwen3.8-27b-instruct" never existed there.
        # Groq requires the "qwen/" vendor prefix - confirmed live the real
        # catalog (13 entries) lists it as "qwen/qwen3.8-27b", not "qwen3.8-27b".
        "groq": ModelBinding("groq", "qwen/qwen3.8-27b", ModelIdentityStatus.UNKNOWN),
        # "ovhcloud": ModelBinding("ovhcloud", "Qwen3.8-27B",
        #                            ModelIdentityStatus.UNKNOWN),
    },
)

QWEN3_6_27B = VlmModelProfile(
    profile_id="qwen3.6-27b",
    display_name="Qwen3.6 27B",
    canonical_model_id="qwen3.6-27b",
    family="Qwen3.6",
    base_model="qwen3.6-27b",
    quantization="unknown",
    aliases=("qwen/qwen3.6-27b", "qwen/qwen3.6-27b-instruct", "qwen3.6-27b-instruct"),
    bindings={
        "openrouter": ModelBinding("openrouter", "qwen/qwen3.6-27b", ModelIdentityStatus.UNKNOWN),
        # NVIDIA and Groq bindings removed: confirmed live (2026-09-22) that
        # neither provider's real model catalog has any qwen3.6-27b model
        # (NVIDIA has no Qwen model at all; Groq's 13-entry catalog only has
        # qwen/qwen3.8-27b, not the 3.6 variant).
        # "ovhcloud": ModelBinding("ovhcloud", "Qwen3.6-27B",
        #                            ModelIdentityStatus.UNKNOWN),
    },
)

# Mistral/Pixtral VLM は対応しているが、キャプション用途では弱いため、内蔵の
# 出荷プロファイルから一時的にコメントアウト。OpenRouter等で同モデルを明示的に
# 使う場合の動的モデル一覧・カスタムプロファイルまでは禁止しない。
# PIXTRAL_12B = VlmModelProfile(
#     profile_id="pixtral-12b",
#     display_name="Pixtral 12B",
#     canonical_model_id="pixtral-12b",
#     family="Pixtral",
#     base_model="pixtral-12b",
#     quantization="unknown",
#     aliases=("mistralai/pixtral-12b", "pixtral-12b-2409"),
#     bindings={
#         "mistral": ModelBinding("mistral", "pixtral-12b-2409", ModelIdentityStatus.UNKNOWN,
#                                 ),
#         "openrouter": ModelBinding("openrouter", "mistralai/pixtral-12b", ModelIdentityStatus.UNKNOWN,
#                                    ),
#     },
# )

OPENAI_GPT_5_6_SOL = VlmModelProfile(
    profile_id="openai-gpt-5.6-sol",
    display_name="OpenAI GPT-5.6 Sol",
    canonical_model_id="gpt-5.6-sol",
    family="GPT-5.6 Sol",
    base_model="gpt-5.6-sol",
    revision="provider_managed",
    quantization="provider_managed",
    aliases=("gpt-5.6", "openai/gpt-5.6-sol"),
    bindings={
        "openai": ModelBinding("openai", "gpt-5.6-sol", ModelIdentityStatus.DECLARED),
        "vercel": ModelBinding("vercel", "openai/gpt-5.6-sol", ModelIdentityStatus.DECLARED),
        # 実機確認(2026-09-22): OpenRouterの実カタログに openai/gpt-5.6-sol が存在する。
        "openrouter": ModelBinding("openrouter", "openai/gpt-5.6-sol", ModelIdentityStatus.DECLARED),
    },
)

OPENAI_GPT_5_6_TERRA = VlmModelProfile(
    profile_id="openai-gpt-5.6-terra",
    display_name="OpenAI GPT-5.6 Terra",
    canonical_model_id="gpt-5.6-terra",
    family="GPT-5.6 Terra",
    base_model="gpt-5.6-terra",
    revision="provider_managed",
    quantization="provider_managed",
    aliases=("openai/gpt-5.6-terra",),
    bindings={
        "openai": ModelBinding("openai", "gpt-5.6-terra", ModelIdentityStatus.DECLARED),
        "vercel": ModelBinding("vercel", "openai/gpt-5.6-terra", ModelIdentityStatus.DECLARED),
        "openrouter": ModelBinding("openrouter", "openai/gpt-5.6-terra", ModelIdentityStatus.DECLARED),
    },
)

OPENAI_GPT_5_6_LUNA = VlmModelProfile(
    profile_id="openai-gpt-5.6-luna",
    display_name="OpenAI GPT-5.6 Luna",
    canonical_model_id="gpt-5.6-luna",
    family="GPT-5.6 Luna",
    base_model="gpt-5.6-luna",
    revision="provider_managed",
    quantization="provider_managed",
    aliases=("openai/gpt-5.6-luna",),
    bindings={
        "openai": ModelBinding("openai", "gpt-5.6-luna", ModelIdentityStatus.DECLARED),
        "vercel": ModelBinding("vercel", "openai/gpt-5.6-luna", ModelIdentityStatus.DECLARED),
        "openrouter": ModelBinding("openrouter", "openai/gpt-5.6-luna", ModelIdentityStatus.DECLARED),
    },
)


def _claude_profile(profile_id: str, display_name: str, model_id: str, *,
                    family: str, revision: str = "provider_managed",
                    aliases: tuple[str, ...] = (),
                    vercel_model_id: str | None = None) -> VlmModelProfile:
    """Anthropic の公式モデルIDを、直接APIとVercelの両経路へ束ねる。"""
    vercel_id = vercel_model_id or f"anthropic/{model_id}"
    return VlmModelProfile(
        profile_id=profile_id,
        display_name=display_name,
        canonical_model_id=model_id,
        family=family,
        base_model=model_id,
        revision=revision,
        quantization="provider_managed",
        aliases=aliases + (vercel_id,),
        bindings={
            "anthropic": ModelBinding("anthropic", model_id, ModelIdentityStatus.DECLARED),
            "vercel": ModelBinding("vercel", vercel_id, ModelIdentityStatus.DECLARED),
        },
    )


CLAUDE_FABLE_5_1 = _claude_profile(
    "claude-fable-5-1", "Claude Fable 5.1", "claude-fable-5-1",
    family="Claude Fable 5.1",
    # Vercel AI Gatewayは "claude-fable-5.1"(ドット表記)。ヘルパーの既定
    # (anthropicのダッシュ表記をそのまま流用)は2026-09-22の実機確認で誤りと判明。
    vercel_model_id="anthropic/claude-fable-5.1",
)

CLAUDE_FABLE_5 = _claude_profile(
    "claude-fable-5", "Claude Fable 5", "claude-fable-5",
    family="Claude Fable 5",
)

CLAUDE_OPUS_5 = _claude_profile(
    "claude-opus-5", "Claude Opus 5", "claude-opus-5",
    family="Claude Opus 5",
)

CLAUDE_OPUS_4_8 = VlmModelProfile(
    profile_id="claude-opus-4-8",
    display_name="Claude Opus 4.8",
    canonical_model_id="claude-opus-4-8",
    family="Claude Opus 4.8",
    base_model="claude-opus-4-8",
    revision="provider_managed",
    quantization="provider_managed",
    aliases=("anthropic/claude-opus-4-8",),
    bindings={
        "anthropic": ModelBinding("anthropic", "claude-opus-4-8", ModelIdentityStatus.DECLARED),
        # Vercelは "claude-opus-4.8"(ドット表記)。2026-09-22の実機確認で、
        # anthropicのダッシュ表記をそのまま使っていたのは誤りと判明。
        "vercel": ModelBinding("vercel", "anthropic/claude-opus-4.8",
                                ModelIdentityStatus.DECLARED),
    },
)

CLAUDE_OPUS_4_6 = _claude_profile(
    "claude-opus-4-6", "Claude Opus 4.6", "claude-opus-4-6",
    family="Claude Opus 4.6",
    vercel_model_id="anthropic/claude-opus-4.6",
)

CLAUDE_OPUS_4_5 = _claude_profile(
    "claude-opus-4-5", "Claude Opus 4.5", "claude-opus-4-5-20251101",
    family="Claude Opus 4.5", revision="20251101",
    aliases=("claude-opus-4-5",),
    vercel_model_id="anthropic/claude-opus-4.5",
)

CLAUDE_SONNET_5 = _claude_profile(
    "claude-sonnet-5", "Claude Sonnet 5", "claude-sonnet-5",
    family="Claude Sonnet 5",
)

CLAUDE_OPUS_4_7 = VlmModelProfile(
    profile_id="claude-opus-4-7",
    display_name="Claude Opus 4.7",
    canonical_model_id="claude-opus-4-7",
    family="Claude Opus 4.7",
    base_model="claude-opus-4-7",
    revision="provider_managed",
    quantization="provider_managed",
    aliases=("anthropic/claude-opus-4-7",),
    bindings={
        "anthropic": ModelBinding("anthropic", "claude-opus-4-7", ModelIdentityStatus.DECLARED),
        # Vercelは "claude-opus-4.7"(ドット表記)。2026-09-22の実機確認で、
        # anthropicのダッシュ表記をそのまま使っていたのは誤りと判明。
        "vercel": ModelBinding("vercel", "anthropic/claude-opus-4.7",
                                ModelIdentityStatus.DECLARED),
    },
)

CLAUDE_SONNET_4_6 = VlmModelProfile(
    profile_id="claude-sonnet-4-6",
    display_name="Claude Sonnet 4.6",
    canonical_model_id="claude-sonnet-4-6",
    family="Claude Sonnet 4.6",
    base_model="claude-sonnet-4-6",
    revision="provider_managed",
    quantization="provider_managed",
    aliases=("anthropic/claude-sonnet-4-6",),
    bindings={
        "anthropic": ModelBinding("anthropic", "claude-sonnet-4-6", ModelIdentityStatus.DECLARED),
        # Vercelは "claude-sonnet-4.6"(ドット表記)。2026-09-22の実機確認で、
        # anthropicのダッシュ表記をそのまま使っていたのは誤りと判明。
        "vercel": ModelBinding("vercel", "anthropic/claude-sonnet-4.6",
                                ModelIdentityStatus.DECLARED),
    },
)

CLAUDE_SONNET_4_5 = _claude_profile(
    "claude-sonnet-4-5", "Claude Sonnet 4.5", "claude-sonnet-4-5-20250929",
    family="Claude Sonnet 4.5", revision="20250929",
    aliases=("claude-sonnet-4-5",),
    vercel_model_id="anthropic/claude-sonnet-4.5",
)

CLAUDE_HAIKU_4_5 = VlmModelProfile(
    profile_id="claude-haiku-4-5",
    display_name="Claude Haiku 4.5",
    canonical_model_id="claude-haiku-4-5-20251001",
    family="Claude Haiku 4.5",
    base_model="claude-haiku-4-5-20251001",
    revision="20251001",
    quantization="provider_managed",
    aliases=("claude-haiku-4-5", "anthropic/claude-haiku-4.5"),
    bindings={
        "anthropic": ModelBinding("anthropic", "claude-haiku-4-5-20251001",
                                  ModelIdentityStatus.DECLARED),
        "vercel": ModelBinding("vercel", "anthropic/claude-haiku-4.5",
                                ModelIdentityStatus.DECLARED),
    },
)

def _grok_profile(profile_id: str, display_name: str, model_id: str, *,
                  family: str) -> VlmModelProfile:
    """xAI Grok の公式モデルIDを、xAI 直接APIの経路へ束ねる。

    Grok は現在 xAI 直販のみを内蔵経路にする（OpenRouter 等の同一モデル ID は
    利用者がプロファイルエディタで実在確認のうえ追加できる）。
    """
    return VlmModelProfile(
        profile_id=profile_id,
        display_name=display_name,
        canonical_model_id=model_id,
        family=family,
        base_model=model_id,
        revision="provider_managed",
        quantization="provider_managed",
        aliases=(profile_id, f"x-ai/{model_id}", f"xai/{model_id}"),
        bindings={
            "xai": ModelBinding("xai", model_id, ModelIdentityStatus.DECLARED),
        },
    )


GROK_4_6 = _grok_profile("grok-4-6", "Grok 4.6", "grok-4.6", family="Grok 4.6")

_ALL_PROFILES = [
    GEMMA_4_26B_A4B_IT, GEMMA_4_31B_IT, QWEN3_8_27B, QWEN3_6_27B,
    OPENAI_GPT_5_6_SOL, OPENAI_GPT_5_6_TERRA, OPENAI_GPT_5_6_LUNA,
    CLAUDE_FABLE_5_1, CLAUDE_FABLE_5, CLAUDE_OPUS_5,
    CLAUDE_OPUS_4_8, CLAUDE_OPUS_4_7, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_5,
    CLAUDE_SONNET_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_4_5, CLAUDE_HAIKU_4_5,
    GROK_4_6,
]


def _profile_reference_tokens(profile: VlmModelProfile) -> set[str]:
    """このプロファイルの「同一モデル」を表す代表トークン集合。"""
    toks: set[str] = set()
    for s in (profile.canonical_model_id, profile.base_model, profile.family):
        toks |= _sig_tokens(s)
    for a in profile.aliases:
        toks |= _sig_tokens(a)
    return toks


def match_model_id(profile: VlmModelProfile, provider_id: str,
                   candidates: list[str]) -> tuple[str | None, float]:
    """プロバイダーが返したモデル ID 一覧から、このプロファイルの正規モデルに
    もっとも合致するものを選ぶ。戻り値は (best_id | None, score[0..1])。

    フォールバックは「同一モデルを複数プロバイダーで回す」設計なので、ここでの狙いは
    「一覧の中でプロファイルのモデルはどれか」を当てること。任意モデルの自由選択ではない。
    """
    if not candidates:
        return None, 0.0
    binding = profile.binding_for(provider_id)
    lowered = {c.lower(): c for c in candidates}
    # 1) binding の既定 ID / alias が一覧にそのままあれば即決。
    exacts = [binding.model_id] if binding is not None else []
    exacts += list(profile.aliases) + [profile.canonical_model_id]
    for e in exacts:
        if e and e.lower() in lowered:
            return lowered[e.lower()], 1.0
    # 2) トークン一致で採点。
    ref = _profile_reference_tokens(profile)
    if not ref:
        return None, 0.0
    best_id, best_score = None, 0.0
    for c in candidates:
        ct = _sig_tokens(c)
        if not ct:
            continue
        inter = ref & ct
        # 数字トークン（27b / 12b など）が食い違うなら別サイズ＝別モデル。強く減点。
        ref_nums = {t for t in ref if any(ch.isdigit() for ch in t)}
        cand_nums = {t for t in ct if any(ch.isdigit() for ch in t)}
        size_ok = not (ref_nums and cand_nums) or bool(ref_nums & cand_nums)
        score = len(inter) / len(ref | ct)
        if not size_ok:
            score *= 0.3
        if score > best_score:
            best_id, best_score = c, score
    return (best_id, best_score) if best_score >= 0.34 else (None, best_score)


def looks_same_family(profile: VlmModelProfile, model_id: str) -> bool:
    """model_id がプロファイルと「だいたい同じモデル」に見えるか（緩いチェック）。"""
    if not model_id:
        return False
    ref = _profile_reference_tokens(profile)
    ct = _sig_tokens(model_id)
    if not ref or not ct:
        return False
    ref_nums = {t for t in ref if any(ch.isdigit() for ch in t)}
    cand_nums = {t for t in ct if any(ch.isdigit() for ch in t)}
    if ref_nums and cand_nums and not (ref_nums & cand_nums):
        return False
    return len(ref & ct) >= 1


# 現在のVLM UIで取得できても、画像入力を受け付けないことが公式仕様で明確な
# モデル群。GroqのCompoundはテキスト／ツール用で、画像キャプション経路には使えない。
# 将来のモデル追加で一覧から漏れても、既知の非VLMだけは安全側で弾く。
_KNOWN_NON_VISION_MODEL_IDS = {
    "groq": frozenset({
        "groq/compound",
        "groq/compound-mini",
        "compound",
        "compound-mini",
        "compound-beta",
        "compound-beta-mini",
        "groq/compound-beta",
        "groq/compound-beta-mini",
    }),
    # xAI: grok-3 / grok-2 の素体はテキスト専用。grok-build / grok-code は
    # コーディング特化でキャプション経路には使わない。
    "xai": frozenset({
        "grok-3", "grok-3-latest", "grok-3-fast", "grok-3-fast-latest",
        "grok-3-mini", "grok-3-mini-latest", "grok-3-mini-fast", "grok-3-mini-fast-latest",
        "grok-2", "grok-2-latest", "grok-2-1212",
        "grok-beta",
        "grok-build-0.1", "grok-code-fast-1",
    }),
}

# 選択プロファイルに binding がない場合でも、モデル一覧からVLMを探せるようにするための
# 既知の画像対応ID。通常のプロファイルでは binding のIDを優先して判定する。
# Groqの現行Visionドキュメントに掲載されている画像対応モデルもここへ明示する。
_KNOWN_VISION_MODEL_IDS = {
    "gemini": frozenset({
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview-customtools",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.5-pro-preview",
    }),
    "groq": frozenset({
        "qwen/qwen3.8-27b",
        # "meta-llama/llama-4-scout-17b-16e-instruct" と "qwen/qwen3.6-27b" は
        # 2026-09-22の実機 /models 一覧(13件)に無かったため削除。
    }),
    "openai": frozenset({
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4-turbo",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.4-pro",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "o1",
        "o1-pro",
        "o3",
        "o3-mini",
        "o3-pro",
        "o4-mini",
    }),
    "anthropic": frozenset({
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    }),
    "cloudflare": frozenset({
        "@cf/google/gemma-4-26b-a4b-it",
        # "@cf/google/gemma-4-31b-it" removed: confirmed live absent from a real
        # /ai/models/search listing, and a real generation request gets 403
        # "This account is not allowed to access @cf/google/gemma-4-31b-it"
        # (worded like a permissions error but is actually unknown-model - the
        # same account's key passes a lightweight auth-only check against the
        # same endpoint). See GEMMA_4_31B_IT's binding comment above.
        "@cf/meta/llama-3.2-11b-vision-instruct",
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        # "@cf/mistral/mistral-small-3.1-24b-instruct",  # Mistral系VLMは一時停止
        "@cf/moonshotai/kimi-k2.6",
        "@cf/moonshotai/kimi-k2.7-code",
        # 以下4件は2026-09-22の実機 /ai/models/search 一覧(65件)に無かったため削除:
        # "@cf/google/gemma-3-12b-it", "@cf/moonshotai/kimi-k2.5",
        # "@cf/moondream/moondream3.1-9b-a2b", "@cf/uform/uform-gen2-qwen-500m"。
        # いずれも実プロファイルのbindingには使われていない(静的カタログの
        # 未対応経路向け許可リストのみ)ため、削除の実害は無い。
    }),
    "nvidia": frozenset({
        "adept/fuyu-8b",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-4-31b-it",
        "meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-90b-vision-instruct",
        "meta/muse-glimmer-30b",
        "microsoft/kosmos-2",
        "microsoft/phi-3-vision-128k-instruct",
        "moonshotai/kimi-k2.6",
        "nvidia/cosmos-reason2-8b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "nvidia/neva-22b",
        "nvidia/vila",
    }),
    # xAI Grok。/v1/language-models の input_modalities が本来の判定材料で、
    # ここはメタデータを取れないとき（旧 /v1/models、カスタム接続等）の保険。
    "xai": frozenset({
        "grok-4.6", "grok-4.5", "grok-4.3",
        "grok-4.20-0309-reasoning", "grok-4.20-0309-non-reasoning",
        "grok-4.20-multi-agent-0309",
        "grok-4", "grok-4-latest", "grok-4-0709",
        "grok-4-fast", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
        "grok-4-1-fast", "grok-4.1-fast",
        "grok-2-vision", "grok-2-vision-latest", "grok-2-vision-1212",
        "grok-vision-beta",
        "latest",
    }),
    # Mistral/Pixtral: VLMは存在するが、内蔵キャプション経路としては一時停止。
    # "mistral": frozenset({
    #     "mistral-large-2512",
    #     "mistral-medium-2508",
    #     "mistral-small-2506",
    #     "mistral-small-3.1-24b-instruct",
    #     "mistral-small-3.2-24b-instruct",
    #     "ministral-3b-2512",
    #     "ministral-8b-2512",
    #     "ministral-14b-2512",
    #     "magistral-small-2509",
    #     "magistral-medium-2509",
    #     "pixtral-12b-2409",
    #     "pixtral-large-latest",
    # }),
}

# 実際の一覧APIが返した能力確認済みID。プロセス内だけで保持し、設定ファイルへは
# 保存しない。次回起動後も一覧を再取得すれば同じ判定になるため、古いカタログを
# 永続化して実行経路を誤って有効化することを避ける。
_DISCOVERED_VLM_MODEL_IDS: dict[str, set[str]] = {}


def register_discovered_vlm_ids(provider_id: str, model_ids: list[str] | set[str]) -> None:
    """一覧APIでVLMと確認したIDを、現在のプロセスの安全判定へ登録する。"""
    provider = (provider_id or "").strip().lower()
    if not provider:
        return
    bucket = _DISCOVERED_VLM_MODEL_IDS.setdefault(provider, set())
    for model_id in model_ids:
        low = _base_catalog_model_id(str(model_id or ""))
        if low and low not in _KNOWN_NON_VISION_MODEL_IDS.get(provider, ()):
            bucket.add(low)


def _base_catalog_model_id(model_id: str) -> str:
    """OpenRouter等の一覧に付く batch/free 修飾子を外した比較用IDを返す。"""
    low = (model_id or "").strip().lower()
    if low.endswith(":batch"):
        return low[:-len(":batch")]
    return low


def _modality_list(value: object) -> set[str] | None:
    if isinstance(value, (list, tuple, set)):
        return {str(v).strip().lower() for v in value if str(v).strip()}
    if isinstance(value, str) and value.strip():
        return {v.strip().lower() for v in re.split(r"[,+>]+", value) if v.strip()}
    return None


def _metadata_modalities(metadata: dict) -> tuple[set[str] | None, set[str] | None]:
    """OpenRouter/HF/Vercelの異なる能力フィールドを共通形へ読む。"""
    architecture = metadata.get("architecture")
    architecture = architecture if isinstance(architecture, dict) else {}
    modalities = metadata.get("modalities")
    modalities = modalities if isinstance(modalities, dict) else {}
    def first_declared(*values: object) -> set[str] | None:
        for value in values:
            parsed = _modality_list(value)
            if parsed is not None:
                return parsed
        return None

    inputs = first_declared(architecture.get("input_modalities"), modalities.get("input"),
                            metadata.get("input_modalities"))
    outputs = first_declared(architecture.get("output_modalities"), modalities.get("output"),
                             metadata.get("output_modalities"))
    # OpenRouterの旧形式などは配列ではなく ``text+image->text`` だけを返す。
    if inputs is None and outputs is None:
        modality = architecture.get("modality") or modalities.get("modality") \
            or metadata.get("modality")
        if isinstance(modality, str) and "->" in modality:
            left, right = modality.split("->", 1)
            inputs = _modality_list(left)
            outputs = _modality_list(right)
    return inputs, outputs


def _static_model_capability(provider_id: str, model_id: str) -> tuple[bool | None, str]:
    """能力メタデータを返さないAPI向けの公式カタログ由来の保守的判定。"""
    provider = (provider_id or "").strip().lower()
    low = _base_catalog_model_id(model_id)
    if not low:
        return None, "empty model id"
    if low in _KNOWN_NON_VISION_MODEL_IDS.get(provider, ()):
        return False, "known non-vision model"
    if any(x in low for x in ("embedding", "-embed", "/embed", "rerank", "moderation",
                              "content-safety", "guardrail")):
        return False, "non-caption model family"
    if low in _KNOWN_VISION_MODEL_IDS.get(provider, ()):
        return True, "provider catalog allowlist"

    if provider == "gemini":
        if low.startswith("gemini-"):
            if any(x in low for x in ("-image", "-live", "-tts", "-transcribe", "-audio")):
                return False, "Gemini media/audio model"
            if "embedding" not in low:
                return True, "Google Gemini model catalog"
        if low.startswith("gemma-") and low.startswith(("gemma-3-", "gemma-4-")):
            return True, "Google Gemma model documentation"
    elif provider == "anthropic":
        if low.startswith(("claude-fable-", "claude-opus-", "claude-sonnet-", "claude-haiku-")):
            return True, "Anthropic Claude model documentation"
    elif provider == "openai":
        if any(x in low for x in ("embedding", "whisper", "transcri", "tts", "realtime",
                                  "moderation", "audio", "image")):
            return False, "OpenAI non-caption model family"
        if low.startswith(("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-5", "o1", "o3", "o4-")):
            return True, "OpenAI model documentation"
    # elif provider == "mistral":
    #     if any(x in low for x in ("pixtral", "magistral", "ministral-3", "mistral-large-3",
    #                               "mistral-medium-3", "mistral-small-3.1", "mistral-small-3.2")):
    #         return True, "Mistral vision model documentation"
    elif provider == "nvidia":
        if (low.startswith(("google/gemma-3-", "google/gemma-4-"))
                or any(x in low for x in ("vision", "vlm", "-vl", "fuyu", "kosmos",
                                          "neva", "vila", "muse-glimmer", "omni",
                                          "cosmos-reason", "phi-3.5-vision", "kimi-k2.6"))):
            return True, "NVIDIA visual model catalog"
    elif provider == "cloudflare":
        if any(x in low for x in ("image-to-text", "moondream", "vision", "gemma-3-",
                                  "gemma-4-", "kimi-k2.5", "kimi-k2.6", "kimi-k2.7")):
            return True, "Cloudflare Workers AI model catalog"
    elif provider == "groq":
        if low in {"meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3.6-27b",
                   "qwen/qwen3.8-27b"}:
            return True, "Groq vision documentation"
    elif provider == "xai":
        if "vision" in low:
            return True, "xAI Grok vision model"
        # grok-4 系（grok-4.6 / grok-4.3 / grok-4-fast など）は入力マルチモーダル。
        # grok-3 / grok-2 素体、grok-build / grok-code はテキスト専用。
        if re.match(r"grok-4(?:[.\-]|$)", low) and not any(
                x in low for x in ("code", "build")):
            return True, "xAI Grok 4 family documentation"
    return None, "capability not declared by provider"


def classify_model_capability(provider_id: str, model_id: str,
                              metadata: dict | None = None) -> tuple[bool | None, str]:
    """モデルが画像を入力し、テキストを出力するVLMかを判定する。

    一覧APIの能力メタデータを最優先し、メタデータを持たないAPIだけ公式カタログの
    保守的な判定へフォールバックする。戻り値は (判定, 根拠) で、判定Noneは不明。
    """
    data = metadata if isinstance(metadata, dict) else {}
    provider = (provider_id or "").strip().lower()
    low_id = (model_id or "").strip().lower()
    if low_id.endswith(":batch"):
        return False, "batch-only route is not chat caption API"
    if provider == "openrouter" and low_id in {
        "openrouter/auto", "openrouter/auto-beta", "openrouter/free",
    }:
        return False, "router route is not a fixed caption model"
    if provider == "openrouter" and low_id.startswith("~"):
        return False, "router alias is not a fixed caption model"
    task = data.get("task") or data.get("task_name") or ""
    if isinstance(task, dict):
        task = task.get("name") or task.get("id") or ""
    task_text = str(task).strip().lower()
    flags = data.get("tags") or data.get("capabilities") or []
    flag_text = " ".join(str(x) for x in flags) if isinstance(flags, (list, tuple, set)) else str(flags)
    searchable = f"{task_text} {flag_text} {(data.get('description') or '')}".lower()
    if any(x in searchable for x in (
        "text-to-image", "text to image", "image generation", "image-generation",
        "text to video", "text-to-video", "music generation", "speech-to-text",
        "text-to-speech", "embedding", "content safety", "content-safety", "guardrail",
        "moderation model", "router that", "auto router",
    )):
        return False, "model list task is not caption VLM"
    inputs, outputs = _metadata_modalities(data)
    if inputs is not None:
        if "image" not in inputs:
            return False, "model list input_modalities excludes image"
        if outputs is not None and "text" not in outputs:
            return False, "model list output_modalities excludes text"
        return True, "model list input/output modalities"

    capabilities = data.get("capabilities")
    if isinstance(capabilities, dict) and "vision" in capabilities:
        if not bool(capabilities.get("vision")):
            return False, "model list capabilities.vision=false"
        return True, "model list capabilities.vision=true"

    # xAI のカタログ（/v1/models・/v1/language-models）は画像入力トークンの課金単価を
    # 持つ。input_modalities も明示的な capabilities.vision も無い応答で、これが正なら
    # 画像入力対応とみなす（明示的な否定より後に置く）。
    img_price = data.get("prompt_image_token_price")
    if isinstance(img_price, (int, float)) and not isinstance(img_price, bool) and img_price > 0:
        if outputs is not None and "text" not in outputs:
            return False, "image priced but output_modalities excludes text"
        return True, "model list prompt_image_token_price"

    if "image-to-text" in searchable or "vision" in searchable or "multimodal" in searchable:
        return True, "model list task/capability metadata"
    return _static_model_capability(provider_id, model_id)


def is_known_non_vision_model(provider_id: str, model_id: str) -> bool:
    """既知のテキスト専用モデルかを判定する。"""
    provider = (provider_id or "").strip().lower()
    model = (model_id or "").strip().lower()
    return model in _KNOWN_NON_VISION_MODEL_IDS.get(provider, ())


def _known_vision_model_ids(provider_id: str) -> set[str]:
    """内蔵プロファイルとプロバイダー公式掲載IDから既知のVLM IDを集める。"""
    provider = (provider_id or "").strip().lower()
    ids = set(_KNOWN_VISION_MODEL_IDS.get(provider, ()))
    for profile in _ALL_PROFILES:
        binding = profile.binding_for(provider)
        if binding is None:
            continue
        if binding.model_id:
            ids.add(binding.model_id.strip().lower())
    return ids


def _profile_model_ids(profile: VlmModelProfile, provider_id: str) -> set[str]:
    """プロファイルがこのプロバイダーで表す正規ID・別名を返す。"""
    ids = {profile.canonical_model_id.strip().lower()} if profile.canonical_model_id else set()
    ids.update(a.strip().lower() for a in profile.aliases if a.strip())
    binding = profile.binding_for(provider_id)
    if binding is not None and binding.model_id:
        ids.add(binding.model_id.strip().lower())
    return ids


def _known_profile_for_model(provider_id: str, model_id: str) -> VlmModelProfile | None:
    """既知IDを、同一IDを定義している出荷プロファイルへ結び付ける。"""
    low = (model_id or "").strip().lower()
    if not low:
        return None
    for candidate in _ALL_PROFILES:
        if candidate.binding_for(provider_id) is None:
            continue
        if low in _profile_model_ids(candidate, provider_id):
            return candidate
    return None


def is_vlm_model_id(profile: VlmModelProfile | None, provider_id: str,
                    model_id: str) -> bool:
    """選択中のVLMプロファイルで利用可能なモデルIDかを判定する。

    選択プロファイルに binding がある経路は、同じモデルに対応するIDだけを許可する。
    binding がない経路は、既知のVLM IDから選べるようにする。どちらの場合も、既知の
    テキスト専用モデルは許可しない。
    """
    if not (model_id or "").strip() or is_known_non_vision_model(provider_id, model_id):
        return False
    known_profile = _known_profile_for_model(provider_id, model_id)
    if profile is not None and profile.binding_for(provider_id) is not None:
        low = model_id.strip().lower()
        binding = profile.binding_for(provider_id)
        # A profile's aliases are shared metadata, but many aliases are provider
        # specific (for example OpenRouter's :free ID is not Gemini's ID). The
        # current provider's exact binding is authoritative; an exact binding from
        # another provider is not an acceptable substitute. Identical IDs on two
        # providers remain valid because the current binding matches first.
        if binding is not None and low == binding.model_id.strip().lower():
            static_capability, _ = classify_model_capability(provider_id, model_id)
            discovered = _DISCOVERED_VLM_MODEL_IDS.get(
                (provider_id or "").strip().lower(), set())
            # Explicit static rejection (for example a :batch route or known
            # text-only ID) always wins over persisted/discovered allowlists.
            return (static_capability is not False
                    and (binding.vlm_capable
                         or static_capability is True
                         or low in _known_vision_model_ids(provider_id)
                         or _base_catalog_model_id(model_id) in discovered))
        if any(other_provider != provider_id
               and other_binding.model_id.strip().lower() == low
               for other_provider, other_binding in profile.bindings.items()):
            return False
        if profile.canonical_model_id and low == profile.canonical_model_id.strip().lower():
            return True
        if low in {alias.strip().lower() for alias in profile.aliases if alias.strip()}:
            return False
        # 既知の別プロファイルIDを、同じ世代というだけで別モデルとして採用しない。
        if known_profile is not None and known_profile.profile_id != profile.profile_id:
            return False
        return looks_same_family(profile, model_id)
    # No binding for this provider on the current profile: any known-good VLM id
    # for the provider is normally an acceptable manual choice (a Gemma profile
    # picking a Claude model via a manual override is a deliberate, unambiguous
    # cross-vendor choice). But a same-family sibling profile's id (e.g.
    # gemma-4-31b-it accepting gemma-4-26b-a4b-it's cloudflare id) looks like it
    # could be "this same model, just this provider's exact ID" and would get
    # silently accepted, wired in, and shown as an enabled/verified route -
    # while actually running a different-sized model under the current
    # profile's name. Reject that specific case; a genuinely different family
    # from another profile is still fine (see is_vlm_model_id's docstring).
    if (profile is not None and known_profile is not None
            and known_profile.profile_id != profile.profile_id
            and known_profile.family == profile.family):
        return False
    low = _base_catalog_model_id(model_id)
    static_capability, _ = classify_model_capability(provider_id, model_id)
    return (low in _known_vision_model_ids(provider_id)
            or low in _DISCOVERED_VLM_MODEL_IDS.get((provider_id or "").strip().lower(), set())
            or static_capability is True)


def filter_vlm_model_ids(profile: VlmModelProfile | None, provider_id: str,
                         candidates: list[str]) -> list[str]:
    """モデル一覧から、選択中プロファイルで使えるVLM IDだけを返す。"""
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        model_id = str(candidate or "").strip()
        if model_id and model_id not in seen and is_vlm_model_id(profile, provider_id, model_id):
            seen.add(model_id)
            out.append(model_id)
    return out


def new_vlm_model_ids(provider_id: str, candidate_ids: list[str]) -> list[str]:
    """VLM対応が確認できた一覧のうち、出荷カタログに未登録のIDだけを返す。

    _ALL_PROFILES のbinding にも _KNOWN_VISION_MODEL_IDS にも無いIDを、
    プロバイダーが新しく公開した可能性のあるモデルとして検出する（自動で
    プロファイルへ追加はしない。UI側で「新モデルを検出」の通知材料として使う）。
    """
    known = _known_vision_model_ids(provider_id)
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidate_ids:
        model_id = str(candidate or "").strip()
        low = _base_catalog_model_id(model_id)
        if not low or low in known or low in seen:
            continue
        seen.add(low)
        out.append(model_id)
    return out


def default_registry() -> VlmModelRegistry:
    return VlmModelRegistry(_ALL_PROFILES)


DEFAULT_MODEL_PROFILE_ID = GEMMA_4_31B_IT.profile_id
