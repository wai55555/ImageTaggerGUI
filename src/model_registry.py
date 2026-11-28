from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import app_settings
import constants
from utils import config_mapping, write_debug_log

ModelType = Literal["tagger", "captioner"]


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    model_name: str
    model_type: ModelType
    model_dir: Path
    config: dict[str, Any]


# PixAI has no model_config.json on disk (its directory predates this feature and is
# never touched). This dict is the synthesized equivalent, matching the current
# hardcoded OnnxTagger class-constant defaults exactly (design.md 6.1節).
_PIXAI_MODEL_ID = "pixai-tagger-v0.9"
_PIXAI_CONFIG: dict[str, Any] = {
    "model_id": _PIXAI_MODEL_ID,
    "model_name": "PixAI Tagger v0.9",
    "model_type": "tagger",
    "license": "Apache-2.0",
    "network": {
        "files": {
            "model.onnx": "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/model.onnx",
            "selected_tags.csv": "https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx/resolve/main/selected_tags.csv",
        },
    },
    "inference": {
        "input_size": 448,
        "layout": "nchw",
        "channel_order": "rgb",
        "rescale_to_unit": True,
        "pad_color_rgb": [0, 0, 0],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "extra_inputs": [],
        "output_name": None,
        "output_activation": "sigmoid",
    },
    "tags_csv": {
        "format": "pixai6col",
        "default_category": "general",
    },
    "ui": {
        "supports_character_tag": True,
        "supports_threshold_slider": True,
        "default_threshold": 0.40,
        # character しきい値の既定は app_settings.DEFAULT_CHARACTER_THRESHOLD が
        # 単一の正本（REQ-D4 / R18-4）。ここで数値を二重管理しない。
        "default_character_threshold": app_settings.DEFAULT_CHARACTER_THRESHOLD,
        "categories": ["general", "character"],
    },
}


# Curated model-picker order (user request 2026-08-31): roughly popularity order,
# with the natural-language captioner pinned last. Any model_id not listed here still
# loads, but sorts after these entries, alphabetically by model_id.
_MODEL_DISPLAY_ORDER: tuple[str, ...] = (
    "pixai-tagger-v0.9",
    "pixai-tagger-v1.0",
    "wd_eva02_large_v3",
    "oppaioracle_v1_1",
    "camie_tagger_v2",
    "cl_tagger_1_02",
    "cl_tagger_v2_01a",
    "wd14_convnextv2",
    "wd_eva02_2026_canary",
    "florence2_base_ft",
)


def _pixai_entry() -> ModelEntry:
    return ModelEntry(
        model_id=_PIXAI_MODEL_ID,
        model_name=_PIXAI_CONFIG["model_name"],
        model_type="tagger",
        model_dir=constants.MODEL_PATH.parent,
        config=_PIXAI_CONFIG,
    )


def _iter_model_config_dirs() -> "list[Path]":
    """Every directory that may hold a model_config.json, most-authoritative first:
    the user-writable MODELS_DIR (downloads land here), then, when frozen and different,
    the bundled MODELS_RESOURCE_DIR (_internal/models/ - config-only, no model.onnx)."""
    seen: set[str] = set()
    dirs: list[Path] = []
    for root in (constants.MODELS_DIR, constants.MODELS_RESOURCE_DIR):
        if not root.is_dir():
            continue
        for sub_dir in sorted(root.iterdir(), key=lambda p: p.name):
            if not sub_dir.is_dir() or sub_dir.name in seen:
                continue
            if not (sub_dir / "model_config.json").is_file():
                continue
            seen.add(sub_dir.name)
            dirs.append(sub_dir)
    return dirs


def discover_models() -> list[ModelEntry]:
    """
    Returns the PixAI pseudo-entry plus every model_config.json-bearing model directory
    (see _iter_model_config_dirs), ordered by _MODEL_DISPLAY_ORDER (unlisted model_ids
    sort last, by model_id).
    """
    entries: list[ModelEntry] = [_pixai_entry()]
    seen_model_ids: set[str] = {_PIXAI_MODEL_ID}

    for config_dir in _iter_model_config_dirs():
        config_path = config_dir / "model_config.json"
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg: Any = json.load(f)
        except Exception as e:
            write_debug_log(f"model_registry: failed to load {config_path}: {e}")
            continue

        if not isinstance(cfg, dict):
            write_debug_log(f"model_registry: {config_path} must contain a JSON object; skipping.")
            continue

        # model_id ends up as a dict key in _sort_by_display_order() and is compared all
        # over the app, so a non-string (or empty) value has to be rejected here rather
        # than blowing up discovery later with an unhashable-key TypeError.
        model_id = cfg.get("model_id", config_dir.name)
        if not isinstance(model_id, str) or not model_id.strip():
            write_debug_log(f"model_registry: {config_path} has an invalid model_id {model_id!r}; skipping.")
            continue
        model_id = model_id.strip()

        # 同じ model_id が2つ以上あると get_model_entry() が先頭しか返さず、後発の
        # モデルディレクトリを選択できなくなる。最初に見つかったものだけ採用する。
        if model_id in seen_model_ids:
            write_debug_log(f"model_registry: duplicate model_id '{model_id}' at {config_path}; skipping.")
            continue

        model_type = cfg.get("model_type", "tagger")
        if model_type not in ("tagger", "captioner"):
            write_debug_log(f"model_registry: {config_path} has unknown model_type '{model_type}', skipping.")
            continue

        model_name = cfg.get("model_name", model_id)
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = model_id

        # Downloaded model.onnx files always live under the user-writable MODELS_DIR, even
        # when the config was read from the bundled resource copy.
        model_dir = constants.MODELS_DIR / config_dir.name

        # "manual_download" models have no working in-app download (e.g. a gated HF repo):
        # stay hidden from the picker until the user has placed every required file here
        # by hand. Once complete, the entry appears and behaves like any other model.
        if cfg.get("manual_download", False):
            required_files = config_mapping(cfg, "network", "files")
            if not required_files or not all((model_dir / name).is_file() for name in required_files):
                write_debug_log(f"model_registry: '{model_id}' is manual_download and its files are not all present yet; hiding it.")
                continue

        entries.append(ModelEntry(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            model_dir=model_dir,
            config=cfg,
        ))
        seen_model_ids.add(model_id)

    return _sort_by_display_order(entries)


def _sort_by_display_order(entries: list[ModelEntry]) -> list[ModelEntry]:
    order = {model_id: i for i, model_id in enumerate(_MODEL_DISPLAY_ORDER)}
    return sorted(entries, key=lambda e: (order.get(e.model_id, len(order)), e.model_id))


def get_model_entry(model_id: str) -> ModelEntry | None:
    """Convenience lookup: returns the ModelEntry with the given model_id, or None."""
    for entry in discover_models():
        if entry.model_id == model_id:
            return entry
    return None


def primary_onnx_file_name(entry: ModelEntry) -> str:
    """Return the model's ONNX file name as declared by the manifest's `network.files`.

    REQ-A4: the download side already treats `network.files` as the source of truth, so
    inference must not hardcode a second copy of the name. Tagger manifests declare
    exactly one `.onnx` entry; `model.onnx` remains the fallback for manifests that
    declare none (the legacy PixAI pseudo-entry) or several (not expected for taggers).
    """
    files = config_mapping(entry.config, "network", "files")
    onnx_names = [str(name) for name in files if str(name).lower().endswith(".onnx")]
    if len(onnx_names) == 1:
        return onnx_names[0]
    if "model.onnx" in onnx_names:
        return "model.onnx"
    write_debug_log(
        f"model_registry: '{entry.model_id}' declares {len(onnx_names)} .onnx files "
        f"({onnx_names}); falling back to model.onnx for tagger inference."
    )
    return "model.onnx"

