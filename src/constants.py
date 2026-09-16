import os
import shutil
import sys
from pathlib import Path
from typing import Mapping


def _project_root_for_source() -> Path:
    """Return the repository/application root when source files live under ``src``.

    Frozen builds continue to use the executable directory.  Keeping this resolution in
    one place prevents moving Python modules into ``src`` from accidentally moving the
    user's config, language files, and model directories with them.
    """
    source_dir = Path(__file__).resolve().parent
    return source_dir.parent if source_dir.name.lower() == "src" else source_dir


def get_resource_dir() -> Path:
    """
    Determines the resource directory, handling PyInstaller's _internal folder.
    """
    if getattr(sys, "frozen", False):
        # For bundled apps, resources like icons might be in _internal
        exe_dir = Path(sys.executable).parent
        internal_dir = exe_dir / "_internal"
        return internal_dir if internal_dir.is_dir() else exe_dir
    return _project_root_for_source()

# --- Path Constants ---
BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else _project_root_for_source()

# RESOURCE_DIR is where bundled, non-user-editable resources are located.
# This handles PyInstaller's `_internal` folder structure.
RESOURCE_DIR = get_resource_dir()

# LANG_DIR is where user-editable translation files are located.
# It should be next to the executable.
LANG_DIR = BASE_DIR / "lang"
# When frozen, the bundled copy lives under _internal/. _seed_bundled_dir() copies it out
# to LANG_DIR on startup; this stays as the fallback if that copy could not be made.
LANG_RESOURCE_DIR = RESOURCE_DIR / "lang"

# User-facing paths are relative to BASE_DIR
CONFIG_PATH = BASE_DIR / "config.ini"
LOG_FILE_PATH = BASE_DIR / "debug_log.txt"

# --- Model-related constants ---
MODEL_SIZE_BYTES = 1271365853
# Directory for additional (non-PixAI) tagger/captioner models, one subdirectory per model_id.
# app_settings.Paths.model_dir/model_filename remain PixAI-only legacy fields (design.md 6.9節).
# MODELS_DIR is user-writable (next to the exe) - downloaded model.onnx files land here.
MODELS_DIR = BASE_DIR / "models"
# When frozen, the hand-authored model_config.json files are bundled under RESOURCE_DIR
# (_internal/), separate from the user-writable MODELS_DIR. Non-frozen: same directory.
MODELS_RESOURCE_DIR = RESOURCE_DIR / "models"


def _seed_bundled_dir(resource_dir: Path, dest_root: Path) -> None:
    """Copy bundled files out of `_internal/` into the user-visible directory next to
    the executable.

    PyInstaller puts bundled data under `_internal/`, which users are not expected to
    open. Both models/ and lang/ are meant to be user-visible - models/ is where people
    drop manually-downloaded model files and where downloads land, and lang/ is
    documented as user-editable - so the shipped copies have to end up there.

    Only files that do not already exist are copied, so a user's edits and the multi-GB
    downloaded model.onnx are never touched. No-op when not frozen (both paths resolve
    to the same directory).
    """
    if resource_dir.resolve() == dest_root.resolve():
        return
    if not resource_dir.is_dir():
        return
    for src in resource_dir.rglob("*"):
        if not src.is_file():
            continue
        dest = dest_root / src.relative_to(resource_dir)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Copy to a sibling temp file and rename into place: os.replace() is atomic on the
        # same volume, so an interrupted startup can never leave a truncated file behind.
        # A partial file would otherwise satisfy the `dest.exists()` check above forever.
        tmp = dest.with_name(dest.name + ".part")
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)


try:
    _seed_bundled_dir(MODELS_RESOURCE_DIR, MODELS_DIR)
    # lang/ has the same _internal-vs-exe-adjacent split as models/. Without this the
    # frozen app finds no .ini at all and every string falls back to its raw key, which
    # also makes the OS-language detection look broken.
    _seed_bundled_dir(LANG_RESOURCE_DIR, LANG_DIR)
except Exception:
    # A read-only install directory must not stop the app from starting; discover_models()
    # also scans MODELS_RESOURCE_DIR directly, and LocaleManager falls back to
    # LANG_RESOURCE_DIR, so the app still works.
    pass
# PixAI's own directory lives under MODELS_DIR like every other model (unified 2026-08-31),
# but it still has no model_config.json - it stays the hardcoded pseudo-entry in
# model_registry.py, so discover_models()'s directory scan skips it and there is no
# duplicate model_combo entry.
MODEL_DIR_NAME = "pixai-tagger-v0.9"
_NEW_PIXAI_DIR = MODELS_DIR / MODEL_DIR_NAME
_LEGACY_PIXAI_DIR = BASE_DIR / "pixai-tagger-v0.9-onnx"
if not (_NEW_PIXAI_DIR / "model.onnx").is_file() and (_LEGACY_PIXAI_DIR / "model.onnx").is_file():
    # An update replaces app code/assets in place but never touches a user's already-
    # downloaded 1.2GB model file sitting at the pre-migration path - silently moving it
    # would be an unnecessary risk (see task.md 2026-08-31 note), and a fresh redownload
    # is wasteful, so fall back to wherever the model actually is. Once found there,
    # everything derived from this directory (translation CSVs included) also comes
    # from there, so an existing install keeps working exactly as before until the user
    # deletes/redownloads it under the new path.
    _PIXAI_DIR = _LEGACY_PIXAI_DIR
else:
    _PIXAI_DIR = _NEW_PIXAI_DIR
MODEL_PATH = _PIXAI_DIR / "model.onnx"
MODEL_POINTER_PATH = _PIXAI_DIR / "model_pointer.txt"
TAGS_CSV_PATH = _PIXAI_DIR / "selected_tags.csv"
DOWNLOAD_URLS: Mapping[Path, str] = {
    MODEL_PATH: "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/model.onnx",
    MODEL_POINTER_PATH: "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/raw/main/model.onnx",
    TAGS_CSV_PATH: "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/selected_tags.csv",
}

# --- Application settings ---
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']
TAGS_PER_PAGE = 16
TAGS_PER_PAGE_FOR_IMAGE = 20
MAX_LOG_LINES = 1000
# 1バッチで書き換えたファイルがこの数を超えたら Undo スナップショットを作らない
# （issue #10: 全ファイルの旧内容＋新内容を1つの CompositeUndoAction に抱えるとメモリを圧迫する）。
UNDO_BATCH_SNAPSHOT_LIMIT = 500

# 右パネルの縦スプリッター [viewer, bulk_actions(section[1]), log] の初期サイズ。
# tagger は現行どおり。captioner / VLM では section[1] を shared_run_block の高さへ畳み、
# 余りを viewer（キャプション編集欄）へ回す（ModelModeController._apply_model_type_ui ->
# MainWindow._rebalance_right_splitter）。
RIGHT_SPLIT_TAGGER = [400, 200, 100]
RIGHT_SPLIT_TEXT = [560, 120, 100]

# --- UI TEXT ---
MSG_WINDOW_TITLE = "PixAI Tagger 0.9 onnx GUI (Viewer/Bulk Edit)"

# --- Style Sheet Colors ---
STYLE_BTN_GREEN = "QPushButton { font-size: 16pt; padding: 10px; background-color: #4CAF50; color: white; }"
STYLE_BTN_BLUE = "QPushButton { font-size: 16pt; padding: 10px; background-color: #2196F3; color: white; }"
STYLE_BTN_ORANGE = "QPushButton { font-size: 16pt; padding: 10px; background-color: #FF9800; color: white; }"
STYLE_BTN_RED = "QPushButton { font-size: 16pt; padding: 10px; background-color: #F44336; color: white; }"
STYLE_LIST_ITEM_SELECTED_DARK = "QListWidget::item:selected { background-color: #1a6b9a; color: #ffffff; }"

# Light Theme Colors (current colors)
COLOR_LOG_SUCCESS_LIGHT = "#00AA00"
COLOR_LOG_ERROR_LIGHT = "#FF0000"
COLOR_LOG_INFO_LIGHT = "#0000FF"
COLOR_LOG_WARN_LIGHT = "#FF8C00"
COLOR_LOG_DEFAULT_LIGHT = "#000000"

# Dark Theme Colors (adjusted for dark background)
COLOR_LOG_SUCCESS_DARK = "#90EE90" # Light green
COLOR_LOG_ERROR_DARK = "#FF6347"  # Tomato
COLOR_LOG_INFO_DARK = "#ADD8E6"   # Light blue
COLOR_LOG_WARN_DARK = "#FFD700"   # Gold
COLOR_LOG_DEFAULT_DARK = "#FFFFFF" # White
