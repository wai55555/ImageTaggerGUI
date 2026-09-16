import configparser
import os
from typing import Any
from dataclasses import dataclass, field, is_dataclass, fields
from pathlib import Path

from utils import write_debug_log, GetString, default_get_string_fallback

_get_string: GetString = default_get_string_fallback

# VLM の既定値。新規設定だけでなく、欠落した [Vlm] キーのフォールバックにも
# 同じ値を使う。既存ユーザーが明示的に選んだモデル／順序は上書きしない。
DEFAULT_VLM_MODEL_PROFILE_ID = "gemma-4-31b-it"
DEFAULT_VLM_CONNECTION_ORDER = "gemini,nvidia,openrouter,cloudflare,groq"

def set_get_string_func(func: GetString):
    global _get_string
    _get_string = func

# --- AppSettings Models (from settings_model.py) ---
@dataclass
class Paths:
    input_dir: str
    model_dir: str
    model_filename: str

# The tag categories that can carry their own threshold / max-count. Order is the
# order rows appear in the "詳細" (per-category) dialog. "general" and "character"
# also have sliders in the main window; the rest are dialog-only.
TAG_CATEGORY_NAMES: tuple[str, ...] = (
    "general", "character", "rating", "copyright", "artist", "meta", "model", "quality", "year",
)


def parse_touched(raw: str) -> set[str]:
    """Splits a `Thresholds.touched` / `Limits.touched` comma list into a set."""
    return {c.strip() for c in raw.split(",") if c.strip()}


def add_touched(raw: str, category: str) -> str:
    """Returns `raw` with `category` added to its comma list (sorted, de-duped)."""
    marked = parse_touched(raw)
    marked.add(category)
    return ",".join(sorted(marked))

@dataclass
class Thresholds:
    general: float
    character: float
    rating: float = 0.50
    copyright: float = 0.50
    artist: float = 0.50
    meta: float = 0.50
    model: float = 0.50
    quality: float = 0.50
    year: float = 0.50
    # Comma-separated category names the user has adjusted by hand. On a model switch
    # a category NOT listed here is reset to the new model's recommended default; a
    # listed one is left as the user set it (design: 2026-08-31 user decision).
    touched: str = ""

@dataclass
class Limits:
    general: int
    character: int
    rating: int = 0
    copyright: int = 0
    artist: int = 0
    meta: int = 0
    model: int = 0
    quality: int = 0
    year: int = 0
    touched: str = ""

@dataclass
class Behavior:
    enable_solo_character_limit: bool
    convert_underscore_to_space: bool
    # 既存 .txt が存在する場合の処理方針: ASK / OVERWRITE / SKIP / APPEND。
    # 既定値により、このキーを持たない旧 config.ini でも従来どおり動作する。
    existing_file_mode: str = "ASK"
    # ONNX Runtime の intra-op スレッド数。0 = 既定（全物理コア＝このオプション追加前と同じ）。
    # 正の N を指定すると N に制限し、大量枚数のタグ付けでマシンが張り付くのを避けられる。
    # GUI には出さない隠し設定（config.ini を手で編集する人向け）。
    onnx_threads: int = 0
    # 一括処理の対象選択。既定 ALL によりこのキーを持たない旧 config.ini でも
    # 従来どおり全画像を処理する（260903_vlm-gap-fix.md todo 5）。
    target_mode: str = "ALL"
    # ローカル ONNX 推論のデバイス希望。auto / cpu / cuda。
    # 既定 auto は「GPU コンポーネント（gpu_runtime/）が揃っていれば CUDA、無ければ CPU」で、
    # 未整備の環境では従来どおり CPU 動作になる（docs/260910_gpu_acceleration_impl_plan.md）。
    onnx_device: str = "auto"
    # 初回に NVIDIA GPU を検出したときの「GPU コンポーネントをダウンロードしますか」
    # ダイアログを出すか。ask / dismissed。dismissed は「使わない・次回から聞かない」。
    # 設定画面のボタンからはいつでもダウンロードできる。
    gpu_setup_prompt: str = "ask"

@dataclass
class Window:
    geometry: str
    tag_display_rows: int = 6
    tag_display_cols: int = 5

@dataclass
class Model:
    model_id: str = "pixai-tagger-v0.9"
    verified_models: dict[str, bool] = field(default_factory=dict)

# 生成キャプションを既存 .txt の内容とどう組み合わせるか。
# danbooru タグ＋自然言語を1ファイルに同居させるモデルがあるため、単純な上書き以外に
# 前後への挿入が必要になる（2026-08-31 ユーザー要望）。
CAPTION_PLACEMENTS: tuple[str, ...] = ("PREPEND", "APPEND", "OVERWRITE")


def parse_caption_placement(raw: str) -> str:
    """config.ini の文字列を検証する。不正値は OVERWRITE（従来動作）へフォールバック。"""
    value = str(raw).strip().upper()
    return value if value in CAPTION_PLACEMENTS else "OVERWRITE"


# 一括処理の対象選択（すべて / 未生成のみ / 失敗のみ / 選択画像のみ）。
# tagging_core.TargetMode と同じ4値。ワーカーは tagging_core.parse_target_mode()
# で enum へ変換するため、ここでは文字列のまま保持・検証する。
TARGET_MODES: tuple[str, ...] = ("ALL", "UNPROCESSED", "FAILED", "SELECTED")


def parse_target_mode_setting(raw: str) -> str:
    """[Behavior] target_mode を検証する。空文字・不正値は ALL（全画像＝この項目を
    持たない旧 config.ini と同じ挙動）へフォールバックする。"""
    value = str(raw).strip().upper()
    return value if value in TARGET_MODES else "ALL"


ONNX_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
GPU_SETUP_PROMPT_STATES: tuple[str, ...] = ("ask", "dismissed")


def parse_onnx_device(raw: str) -> str:
    """[Behavior] onnx_device を検証する。空文字・不正値は auto（＝GPU コンポーネントが
    無ければ CPU、というこのキーを持たない旧 config.ini と同じ実効挙動）へ。"""
    value = str(raw).strip().lower()
    return value if value in ONNX_DEVICES else "auto"


def parse_gpu_setup_prompt(raw: str) -> str:
    """[Behavior] gpu_setup_prompt を検証する。空文字・不正値は ask へ。"""
    value = str(raw).strip().lower()
    return value if value in GPU_SETUP_PROMPT_STATES else "ask"


def _parse_onnx_threads(raw: str) -> int:
    """[Behavior] onnx_threads を検証する。手書きの想定なので空文字・非数値・小数・
    負値は全て 0（= ONNX Runtime 既定 = 全コア）へフォールバックする。

    実コア数を超える値をそのまま ONNX Runtime の SessionOptions.intra_op_num_threads に
    渡すとセッション生成に失敗しうる（タガー/キャプショナーが起動しなくなる）ので、
    実コア数（取得できなければ 32）でクランプする（PR#16 レビュー指摘）。"""
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    return min(n, os.cpu_count() or 32)


@dataclass
class Caption:
    task: str = "MORE_DETAILED_CAPTION"
    # PREPEND / APPEND / OVERWRITE。既定は OVERWRITE で、この項目を持たない
    # 旧 config.ini はこれまでと同じ挙動になる。
    placement: str = "OVERWRITE"

@dataclass
class Debug:
    debug_log: bool


@dataclass
class Vlm:
    """VLM 生成機能の基本設定（260901_VLM_spec.md 16章）。

    複雑なカスタム接続定義は config.ini ではなく vlm_connections.json 側に持つ。
    この項目を持たない旧 config.ini でも enabled=False の未設定状態として動く。
    """
    # True のとき、選択中のローカルモデルの代わりにネットワーク VLM で生成する。
    # 出力の形式（標準文 / モデル学習用長文 / 短いタグ）は VLM プロンプトモードで選ぶ。
    enabled: bool = False
    model_profile_id: str = DEFAULT_VLM_MODEL_PROFILE_ID
    generation_profile_id: str = "default-caption-en"
    # builtin_fallback / custom_single
    execution_mode: str = "builtin_fallback"
    selected_connection_id: str = ""
    connection_order: str = DEFAULT_VLM_CONNECTION_ORDER
    # Cloudflare Workers AI はアカウント ID を URL に含むため別途保持する。
    cloudflare_account_id: str = ""
    # 複数Workspace対象のAnthropic APIキーで必要。単一Workspaceキーなら空でよい。
    anthropic_workspace_id: str = ""
    # 生成プロファイルの内訳（フラットに保持。vlm_profiles.GenerationProfile へ写す）
    language: str = "en"
    detail_level: str = "maximum_detail"
    sentence_mode: str = "automatic_long_detailed"
    character_name_mode: str = "explicit_only"
    markdown: str = "disabled"
    # standard / dataset_long / short_tags
    prompt_mode: str = "standard"
    max_output_tokens: int = 3072
    image_max_long_edge: int = 1536
    # 接続確認済みの binding。`<profile_id>:<provider_id>` をカンマ区切りで保持する。
    # キー登録時の軽量モデル一覧GET、接続診断のフルPASS、または1枚テスト成功で追記される。
    # UNKNOWN 出荷でもここに載れば VERIFIED 扱いになり、「厳格」モードでも候補に残る。
    verified_bindings: str = ""
    # True のとき、内蔵フォールバックは VERIFIED（実証済み or verified_bindings 収録）
    # の接続だけを候補にする。既定 False（同一と宣言されていれば未実証でも使う）。
    strict_identity: bool = False
    # 内蔵経路のモデル ID 上書き。`<profile_id>:<provider_id>=<model_id>` をカンマ区切り。
    # 出荷時の推定 ID が実サービスと違うとき、接続診断で確認しつつここで直せる。
    model_id_overrides: str = ""

    def order_list(self) -> list[str]:
        return [p.strip() for p in self.connection_order.split(",") if p.strip()]

    def verified_set(self) -> set[str]:
        return {t.strip() for t in self.verified_bindings.split(",") if t.strip()}

    def model_id_override_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for tok in self.model_id_overrides.split(","):
            tok = tok.strip()
            if "=" in tok:
                k, _, v = tok.partition("=")
                if k.strip() and v.strip():
                    out[k.strip()] = v.strip()
        return out


@dataclass
class AppSettings:
    paths: Paths
    thresholds: Thresholds
    limits: Limits
    behavior: Behavior
    window: Window
    model: Model
    caption: Caption
    vlm: Vlm
    debug: Debug
    language_code: str

# --- Config Utilities (from config_utils.py) ---
from constants import BASE_DIR, CONFIG_PATH, MODEL_DIR_NAME

def get_default_config() -> configparser.ConfigParser:
    """Returns a ConfigParser object with default settings."""
    config = configparser.ConfigParser()
    DEFAULT_CONFIG = {
        'Paths': {'input_dir': str(BASE_DIR / "inputs"), 'model_dir': MODEL_DIR_NAME, 'model_filename': 'model.onnx'},
        'Thresholds': {'general': '0.40', 'character': '0.65', 'rating': '0.50', 'copyright': '0.50', 'artist': '0.50', 'meta': '0.50', 'model': '0.50', 'quality': '0.50', 'year': '0.50', 'touched': ''},
        'Limits': {'general': '55', 'character': '1', 'rating': '0', 'copyright': '0', 'artist': '0', 'meta': '0', 'model': '0', 'quality': '0', 'year': '0', 'touched': ''},
        'Behavior': {'enable_solo_character_limit': 'True', 'convert_underscore_to_space': 'True', 'existing_file_mode': 'ASK', 'onnx_threads': '0', 'target_mode': 'ALL', 'onnx_device': 'auto', 'gpu_setup_prompt': 'ask'},
        'Window': {'geometry': '986x976+50+50', 'tag_display_rows': '6', 'tag_display_cols': '5'},
        'Model': {'model_id': 'pixai-tagger-v0.9', 'verified_models': ''},
        'Caption': {'task': 'MORE_DETAILED_CAPTION', 'placement': 'OVERWRITE'},
        'Vlm': {
            'enabled': 'False',
            'model_profile_id': DEFAULT_VLM_MODEL_PROFILE_ID, 'generation_profile_id': 'default-caption-en',
            'execution_mode': 'builtin_fallback', 'selected_connection_id': '',
            'connection_order': DEFAULT_VLM_CONNECTION_ORDER,
            'cloudflare_account_id': '',
            'anthropic_workspace_id': '',
            'language': 'en', 'detail_level': 'maximum_detail',
            'sentence_mode': 'automatic_long_detailed', 'character_name_mode': 'explicit_only',
            'markdown': 'disabled', 'prompt_mode': 'standard',
            'max_output_tokens': '3072', 'image_max_long_edge': '1536',
            'verified_bindings': '', 'strict_identity': 'False', 'model_id_overrides': '',
        },
        'Debug': {'debug_log': 'False'},
        'General': {'language_code': ''}
    }
    config.read_dict(DEFAULT_CONFIG)
    return config

def load_config() -> configparser.ConfigParser:
    """Loads the config.ini file, creating it from defaults if it doesn't exist."""
    config = get_default_config()
    if CONFIG_PATH.is_file():
            
        config.read(CONFIG_PATH, encoding='utf-8')
        write_debug_log(_get_string("ConfigUtils", "Config_File_Load_Success", CONFIG_PATH=CONFIG_PATH), _get_string)
    else:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                config.write(f)
            write_debug_log(_get_string("ConfigUtils", "Config_File_NotFound_Create_Default", CONFIG_PATH=CONFIG_PATH), _get_string)
        except Exception as e:
            write_debug_log(_get_string("ConfigUtils", "Config_File_Creation_Failed", e=e), _get_string)
    return config

def _parse_verified_models(config: configparser.ConfigParser) -> dict[str, bool]:
    """
    Parses the `[Model] verified_models` field ("model_id:True,other_id:False").

    Back-compat: a config.ini written before multi-model support has a single
    `[Model] verified` bool for the (implicit) PixAI tagger, and no `verified_models`
    key at all. Migrate it into `{"pixai-tagger-v0.9": <that bool>}` in that case.

    Note: `config.has_option('Model', 'model_id')` is NOT a reliable way to detect
    this, because `get_default_config()` (called by `load_config()` before the file
    is read) already seeds a `model_id` default into the same ConfigParser object -
    so by the time we see it here, `model_id` is present whether or not the file on
    disk ever had it. `verified`, on the other hand, is never added by the current
    defaults, so its presence really does mean "this came from the file".
    """
    if config.has_option('Model', 'verified'):
        legacy_verified = config.getboolean('Model', 'verified', fallback=False)
        return {"pixai-tagger-v0.9": legacy_verified}

    raw = config.get('Model', 'verified_models', fallback='')
    result: dict[str, bool] = {}
    for entry in raw.split(','):
        entry = entry.strip()
        if not entry or ':' not in entry:
            continue
        model_id, _, value = entry.partition(':')
        model_id = model_id.strip()
        if model_id:
            result[model_id] = value.strip().lower() == 'true'
    return result

def load_settings(config: configparser.ConfigParser) -> AppSettings:
    """Loads settings from a ConfigParser object into an AppSettings dataclass."""

    return AppSettings(
        paths=Paths(
            input_dir=config.get('Paths', 'input_dir', fallback=str(BASE_DIR / "inputs")),
            model_dir=config.get('Paths', 'model_dir', fallback=MODEL_DIR_NAME),
            model_filename=config.get('Paths', 'model_filename', fallback='model.onnx')
        ),
        thresholds=Thresholds(
            general=config.getfloat('Thresholds', 'general', fallback=0.40),
            character=config.getfloat('Thresholds', 'character', fallback=0.65),
            rating=config.getfloat('Thresholds', 'rating', fallback=0.50),
            copyright=config.getfloat('Thresholds', 'copyright', fallback=0.50),
            artist=config.getfloat('Thresholds', 'artist', fallback=0.50),
            meta=config.getfloat('Thresholds', 'meta', fallback=0.50),
            model=config.getfloat('Thresholds', 'model', fallback=0.50),
            quality=config.getfloat('Thresholds', 'quality', fallback=0.50),
            year=config.getfloat('Thresholds', 'year', fallback=0.50),
            touched=config.get('Thresholds', 'touched', fallback='')
        ),
        limits=Limits(
            general=config.getint('Limits', 'general', fallback=55),
            character=config.getint('Limits', 'character', fallback=1),
            rating=config.getint('Limits', 'rating', fallback=0),
            copyright=config.getint('Limits', 'copyright', fallback=0),
            artist=config.getint('Limits', 'artist', fallback=0),
            meta=config.getint('Limits', 'meta', fallback=0),
            model=config.getint('Limits', 'model', fallback=0),
            quality=config.getint('Limits', 'quality', fallback=0),
            year=config.getint('Limits', 'year', fallback=0),
            touched=config.get('Limits', 'touched', fallback='')
        ),
        behavior=Behavior(
            enable_solo_character_limit=config.getboolean('Behavior', 'enable_solo_character_limit', fallback=True),
            convert_underscore_to_space=config.getboolean('Behavior', 'convert_underscore_to_space', fallback=True),
            existing_file_mode=config.get('Behavior', 'existing_file_mode', fallback='ASK'),
            onnx_threads=_parse_onnx_threads(config.get('Behavior', 'onnx_threads', fallback='0')),
            target_mode=parse_target_mode_setting(config.get('Behavior', 'target_mode', fallback='ALL')),
            onnx_device=parse_onnx_device(config.get('Behavior', 'onnx_device', fallback='auto')),
            gpu_setup_prompt=parse_gpu_setup_prompt(config.get('Behavior', 'gpu_setup_prompt', fallback='ask'))
        ),
        window=Window(
            geometry=config.get('Window', 'geometry', fallback='986x976+50+50'),
            tag_display_rows=config.getint('Window', 'tag_display_rows', fallback=6),
            tag_display_cols=config.getint('Window', 'tag_display_cols', fallback=5)
        ),
        model=Model(
            model_id=config.get('Model', 'model_id', fallback='pixai-tagger-v0.9'),
            verified_models=_parse_verified_models(config)
        ),
        caption=Caption(
            task=config.get('Caption', 'task', fallback='MORE_DETAILED_CAPTION'),
            placement=parse_caption_placement(config.get('Caption', 'placement', fallback='OVERWRITE'))
        ),
        vlm=_load_vlm(config),
        debug=Debug(
            debug_log=config.getboolean('Debug', 'debug_log', fallback=False) # Default is False for debug_log
        ),
        language_code=config.get('General', 'language_code', fallback="")
    )

def _load_vlm(config: configparser.ConfigParser) -> Vlm:
    """[Vlm] セクションを読み込む。欠けているキーは Vlm の既定値へフォールバック。"""
    d = Vlm()
    g = lambda k, fb: config.get('Vlm', k, fallback=fb)  # noqa: E731

    def gb(k: str, fb: bool) -> bool:
        # 手書き config.ini の [Vlm] enabled / strict_identity が bool として
        # 解釈できない値でも、gi と同様に既定値へ倒して起動を止めない。
        try:
            return config.getboolean('Vlm', k, fallback=fb)
        except (TypeError, ValueError):
            return fb

    def gi(k: str, fb: int) -> int:
        try:
            return config.getint('Vlm', k, fallback=fb)
        except (TypeError, ValueError):
            return fb

    return Vlm(
        enabled=gb('enabled', d.enabled),
        model_profile_id=g('model_profile_id', d.model_profile_id),
        generation_profile_id=g('generation_profile_id', d.generation_profile_id),
        execution_mode=g('execution_mode', d.execution_mode).strip().lower() or d.execution_mode,
        selected_connection_id=g('selected_connection_id', d.selected_connection_id),
        connection_order=g('connection_order', d.connection_order),
        cloudflare_account_id=g('cloudflare_account_id', d.cloudflare_account_id).strip(),
        anthropic_workspace_id=g('anthropic_workspace_id', d.anthropic_workspace_id).strip(),
        language=g('language', d.language),
        detail_level=g('detail_level', d.detail_level),
        sentence_mode=g('sentence_mode', d.sentence_mode),
        character_name_mode=g('character_name_mode', d.character_name_mode),
        markdown=g('markdown', d.markdown),
        prompt_mode=g('prompt_mode', d.prompt_mode),
        max_output_tokens=gi('max_output_tokens', d.max_output_tokens),
        image_max_long_edge=gi('image_max_long_edge', d.image_max_long_edge),
        verified_bindings=g('verified_bindings', d.verified_bindings),
        strict_identity=gb('strict_identity', d.strict_identity),
        model_id_overrides=g('model_id_overrides', d.model_id_overrides),
    )


def save_config(settings: AppSettings) -> bool:
    """Saves the AppSettings object to the config.ini file.

    Return the persistence result so dialogs can keep unsaved changes open when the
    configuration file cannot be written.
    """
    write_debug_log(_get_string("ConfigUtils", "Settings_Save_Start"), _get_string)
    config = configparser.ConfigParser()

    def _save_dataclass_to_config(config_parser: configparser.ConfigParser, dataclass_instance: Any, section_name: str):
        if not config_parser.has_section(section_name):
            config_parser.add_section(section_name)
        
        for field_info in fields(dataclass_instance):
            value = getattr(dataclass_instance, field_info.name)
            
            if is_dataclass(value):
                # Recursively handle nested dataclasses
                _save_dataclass_to_config(config_parser, value, field_info.name.capitalize())
            else:
                # Convert value to string for configparser
                if isinstance(value, float):
                    config_parser.set(section_name, field_info.name, f"{value:.2f}")
                elif isinstance(value, Path):
                    config_parser.set(section_name, field_info.name, str(value))
                elif isinstance(value, dict):
                    # e.g. Model.verified_models -> "model_id:True,other_id:False"
                    config_parser.set(section_name, field_info.name, ",".join(f"{k}:{v}" for k, v in value.items()))
                else:
                    config_parser.set(section_name, field_info.name, str(value))

    # Handle top-level fields of AppSettings
    for field_info in fields(settings):
        value = getattr(settings, field_info.name)
        if is_dataclass(value):
            _save_dataclass_to_config(config, value, field_info.name.capitalize())
        else:
            # Handle language_code which is directly in AppSettings but belongs to 'General' section
            if field_info.name == 'language_code':
                if not config.has_section('General'):
                    config.add_section('General')
                config.set('General', field_info.name, str(value))
            else:
                # This case should ideally not be hit if AppSettings only contains nested dataclasses
                # and language_code is handled specifically.
                pass # Or raise an error if unexpected direct fields exist


    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            config.write(f)
        write_debug_log(_get_string("ConfigUtils", "Config_File_Save_Success", CONFIG_PATH=CONFIG_PATH), _get_string)
        return True
    except Exception as e:
        write_debug_log(_get_string("ConfigUtils", "Config_File_Save_Failed", e=e), _get_string)
        return False
def update_model_verification_status(model_id: str, is_verified: bool, get_string: GetString):
    """
    Loads config, sets the verification status for a single model_id, and saves it.
    Used by worker threads to update model status without clobbering settings the
    GUI thread may have changed concurrently (config is reloaded fresh from disk).
    """
    try:
        config = load_config()
        settings = load_settings(config)
        if settings.model.verified_models.get(model_id) != is_verified:
            settings.model.verified_models[model_id] = is_verified
            save_config(settings)
            if is_verified:
                write_debug_log(get_string("ConfigUtils", "ModelVerified_Success_Debug"), get_string)
            else:
                write_debug_log(get_string("ConfigUtils", "ModelUnverified_Debug"), get_string)
    except Exception as e:
        write_debug_log(get_string("ConfigUtils", "ModelVerification_Update_Failed_Debug", e=e), get_string)
