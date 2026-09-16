from __future__ import annotations

import sys
import csv
import json
import configparser
import traceback
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Mapping, Sequence, Any, Callable, TYPE_CHECKING
from time import perf_counter

from constants import BASE_DIR

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image
    import onnxruntime as ort # type: ignore
else:
    try:
        import numpy as np
        from numpy.typing import NDArray
        from PIL import Image
        import onnxruntime as ort # type: ignore
    except ImportError:
        np = None
        NDArray = None
        Image = None
        ort = None

LOG_FILE_PATH = BASE_DIR / "debug_log.txt"
CONFIG_PATH = BASE_DIR / "config.ini"

from utils import config_mapping, log_dbg, GetString
from app_settings import AppSettings, load_settings
from onnx_providers import make_session


_get_string: GetString = lambda section, key, **kwargs: str(key)

if not TYPE_CHECKING:
    if np is None or Image is None or ort is None:
        log_dbg(_get_string("TaggerCore", "Info_Required_Libraries_Missing"))
        if __name__ != "__main__":
            raise ImportError(_get_string("TaggerCore", "Error_Required_Libraries_NotFound"))
        sys.exit(1)

@dataclass(frozen=True)
class FileChange:
    """1ファイル分の書き換え記録（design.md 2.2節）。

    previous_content が None なら「変更前にファイルが存在しなかった」ことを表し、
    Undo ではファイルを削除する。was_append は追記ログ / 説明用のメタ情報。
    """
    path: Path
    previous_content: str | None
    new_content: str
    was_append: bool
    added_tags: tuple[str, ...] = ()


class ExistingFileMode(Enum):
    """出力先の .txt が既に存在する場合の処理方針（spec.md 1.1節）。"""
    ASK = auto()        # 都度確認
    OVERWRITE = auto()  # 常に上書き
    SKIP = auto()       # 常にスキップ
    APPEND = auto()     # 常に追記


class OverwriteDecision(Enum):
    """ASK モードでユーザーが1ファイルに対して下した判断（spec.md 2.2節）。"""
    OVERWRITE = auto()
    SKIP = auto()
    APPEND = auto()


EXISTING_FILE_MODE_MAP: dict[str, ExistingFileMode] = {
    "ASK": ExistingFileMode.ASK,
    "OVERWRITE": ExistingFileMode.OVERWRITE,
    "SKIP": ExistingFileMode.SKIP,
    "APPEND": ExistingFileMode.APPEND,
}


def parse_existing_file_mode(raw: str, get_string: GetString | None = None) -> ExistingFileMode:
    """config.ini の文字列を ExistingFileMode に変換する。不正値は ASK にフォールバック
    し、警告をデバッグログに残す（spec.md 5.2節）。"""
    mode = EXISTING_FILE_MODE_MAP.get(str(raw).strip().upper())
    if mode is None:
        _get_string_internal = get_string if get_string else _get_string
        log_dbg(_get_string_internal("TaggerCore", "Invalid_Existing_Mode_Debug", raw=raw))
        return ExistingFileMode.ASK
    return mode


def merge_tags(existing_tags: list[str], generated_tags: list[str]) -> list[str]:
    """既存タグを先頭に保持し、含まれていない生成タグだけを末尾に追加する（spec.md 3.1節）。

    - 重複判定は大文字小文字を無視した完全一致
    - 既存タグ同士の重複や表記ゆれ（`long hair` vs `long_hair`）は正規化しない
    - 純粋関数。呼び出し側は戻り値が existing_tags と等しいかで書き込み要否を判断する
    """
    seen = {tag.lower() for tag in existing_tags}
    appended: list[str] = []
    for tag in generated_tags:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)          # 生成タグ側の重複も1回だけ採用する
        appended.append(tag)
    return existing_tags + appended


class TagCategory(IntEnum):
    GENERAL = 0
    ARTIST = 1
    RATING = 2
    COPYRIGHT = 3
    CHARACTER = 4
    META = 5
    # cl_tagger exposes "Model" and "Quality" as first-class categories (its defining
    # feature); camie-tagger-v2 exposes "Year". Kept separate from META so each can carry
    # its own threshold / max-count in the per-category dialog.
    MODEL = 6
    QUALITY = 7
    YEAR = 8

# Settings-side category name -> TagCategory. Mirrors app_settings.TAG_CATEGORY_NAMES;
# kept here as a plain dict so tagging_core stays free of an app_settings import.
CATEGORY_NAME_TO_ENUM: dict[str, "TagCategory"] = {
    "general": TagCategory.GENERAL,
    "character": TagCategory.CHARACTER,
    "rating": TagCategory.RATING,
    "copyright": TagCategory.COPYRIGHT,
    "artist": TagCategory.ARTIST,
    "meta": TagCategory.META,
    "model": TagCategory.MODEL,
    "quality": TagCategory.QUALITY,
    "year": TagCategory.YEAR,
}

@dataclass(frozen=True)
class TagPrediction:
    name: str
    score: float
    category: TagCategory

@dataclass
class TagResult:
    tags: list[TagPrediction] = field(default_factory=list)
    series_tags: tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class TagMeta:
    name: str
    category: int
    count: int | None = None
    ips: tuple[str, ...] = ()

_CATEGORY_LOOKUP: dict[str, int] = {
    "0": 0, "general": 0,
    "1": 1, "artist": 1,
    "2": 2, "rating": 2,
    "3": 3, "copyright": 3,
    "4": 4, "character": 4,
    "5": 5, "meta": 5,
    "9": 2,           # wd-14 / wd-eva02-large-v3 / wd-eva02-2026-canary use "9" for rating
    "6": 6, "model": 6,     # cl_tagger (Model tags - 25-26 of them)
    "7": 7, "quality": 7,   # cl_tagger / cl_tagger v2 (best/normal/bad/worst quality)
    "8": 8, "year": 8,      # camie-tagger-v2
}

TagsCsvFormat = str  # "pixai6col" | "simple3col" | "wd4col" | "idx_json" | "camie_metadata_json"

def _resolve_category(category_str: str) -> int | None:
    try:
        return _CATEGORY_LOOKUP[category_str.strip().lower()]
    except KeyError:
        return None


def _resolve_category_keep_row(category_str: str, *, context: str) -> int:
    """Like _resolve_category, but never drops the row: an unrecognized category is
    treated as GENERAL. Every tag loader MUST use this - infer_batch_prepared pairs
    labels with model scores by index (zip(self.tags, scores)), so a single skipped
    row shifts every later tag onto the wrong score."""
    resolved = _resolve_category(category_str)
    if resolved is None:
        log_dbg(f"load_selected_tags[{context}]: unknown category {category_str!r}; "
                f"mapping to GENERAL to keep index alignment")
        return int(TagCategory.GENERAL)
    return resolved

def _load_selected_tags_pixai6col(tags_path: Path) -> list[TagMeta]:
    labels: list[TagMeta] = []
    with tags_path.open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        try:
            next(reader)
        except StopIteration:
            return []
        for cells in reader:
            if len(cells) < 6:
                continue
            tag_name = cells[2]
            category = _resolve_category_keep_row(cells[3], context="pixai6col")
            count_str = cells[4]
            ips_json = cells[5]
            try:
                count = int(count_str) if count_str else None
            except ValueError:
                count = None
            ips: tuple[str, ...] = ()
            if ips_json:
                try:
                    parsed = json.loads(ips_json)
                    if isinstance(parsed, list):
                        ips = tuple(str(item) for item in parsed)
                except json.JSONDecodeError:
                    pass
            labels.append(TagMeta(name=tag_name, category=category, count=count, ips=ips))
    return labels

def _load_selected_tags_simple3col(tags_path: Path) -> list[TagMeta]:
    """OppaiOracle: tag_id,name,category (3 columns, no count/ips)."""
    labels: list[TagMeta] = []
    with tags_path.open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        try:
            next(reader)
        except StopIteration:
            return []
        for cells in reader:
            if len(cells) < 3:
                continue
            category = _resolve_category_keep_row(cells[2], context="simple3col")
            labels.append(TagMeta(name=cells[1], category=category, count=None, ips=()))
    return labels

def _load_selected_tags_wd4col(tags_path: Path) -> list[TagMeta]:
    """wd-14 / wd-eva02-large-v3 / wd-eva02-2026-canary: tag_id,name,category,count."""
    labels: list[TagMeta] = []
    with tags_path.open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        try:
            next(reader)
        except StopIteration:
            return []
        for cells in reader:
            if len(cells) < 4:
                continue
            category = _resolve_category_keep_row(cells[2], context="wd4col")
            try:
                count = int(cells[3]) if cells[3] else None
            except ValueError:
                count = None
            labels.append(TagMeta(name=cells[1], category=category, count=count, ips=()))
    return labels

def _numeric_keys_sorted(raw: dict[str, Any]) -> list[str]:
    """Keys that are integer strings, in numeric order. Non-numeric keys (a stray
    version/comment field in the JSON) are logged and skipped instead of raising."""
    numeric: list[str] = []
    for k in raw:
        try:
            int(k)
        except (TypeError, ValueError):
            log_dbg(f"load_selected_tags: ignoring non-numeric mapping key {k!r}")
            continue
        numeric.append(k)
    return sorted(numeric, key=int)


def _load_selected_tags_idx_json(tags_path: Path) -> list[TagMeta]:
    """Index -> tag/category mapping JSON. Handles the shapes seen across cl_tagger:

    * v1 tag_mapping.json:  {"<idx>": {"tag": "...", "category": "General"}}
    * v2 model_vocabulary.json (provisional - gated, not yet verified):
        {"idx_to_tag": {"<idx>": "tag"}, "tag_to_category": {"tag": "General"},
         [optional] "idx_to_category": {"<idx>": "General"}}
    * flat list:  [{"tag"/"name": "...", "category": "..."}, ...]  (order == index)
    """
    with tags_path.open(encoding="utf-8") as fp:
        raw: Any = json.load(fp)
    labels: list[TagMeta] = []

    if isinstance(raw, dict) and "idx_to_tag" in raw:
        idx_to_tag: dict[str, Any] = raw.get("idx_to_tag", {}) or {}
        tag_to_category: dict[str, Any] = raw.get("tag_to_category", {}) or {}
        idx_to_category: dict[str, Any] = raw.get("idx_to_category", {}) or {}
        for idx in _numeric_keys_sorted(idx_to_tag):
            tag_name = str(idx_to_tag[idx])
            cat_str = str(idx_to_category.get(idx, tag_to_category.get(tag_name, "")))
            category = _resolve_category_keep_row(cat_str, context="idx_json/v2")
            labels.append(TagMeta(name=tag_name, category=category, count=None, ips=()))
        return labels

    if isinstance(raw, dict) and "tags" in raw and isinstance(raw["tags"], list):
        raw = raw["tags"]

    if isinstance(raw, list):
        for entry in raw:
            entry = entry if isinstance(entry, dict) else {}
            name = str(entry.get("tag", entry.get("name", "")))
            category = _resolve_category_keep_row(str(entry.get("category", "")), context="idx_json/list")
            labels.append(TagMeta(name=name, category=category, count=None, ips=()))
        return labels

    # v1 shape: {"<idx>": {"tag": ..., "category": ...}}
    for idx in _numeric_keys_sorted(raw):
        entry = raw[idx] if isinstance(raw[idx], dict) else {}
        category = _resolve_category_keep_row(str(entry.get("category", "")), context="idx_json")
        labels.append(TagMeta(name=str(entry.get("tag", "")), category=category, count=None, ips=()))
    return labels

def _load_selected_tags_camie_metadata_json(tags_path: Path) -> list[TagMeta]:
    """camie-tagger-v2: camie-tagger-v2-metadata.json -> dataset_info.tag_mapping.{idx_to_tag,tag_to_category}."""
    with tags_path.open(encoding="utf-8") as fp:
        raw: dict[str, Any] = json.load(fp)
    tag_mapping = raw.get("dataset_info", {}).get("tag_mapping", {})
    idx_to_tag: dict[str, str] = tag_mapping.get("idx_to_tag", {})
    tag_to_category: dict[str, str] = tag_mapping.get("tag_to_category", {})
    labels: list[TagMeta] = []
    for idx in _numeric_keys_sorted(idx_to_tag):
        tag_name = idx_to_tag[idx]
        category = _resolve_category_keep_row(str(tag_to_category.get(tag_name, "")), context="camie_metadata_json")
        labels.append(TagMeta(name=tag_name, category=category, count=None, ips=()))
    return labels

_TAGS_CSV_LOADERS: dict[str, Callable[[Path], list[TagMeta]]] = {
    "pixai6col": _load_selected_tags_pixai6col,
    "simple3col": _load_selected_tags_simple3col,
    "wd4col": _load_selected_tags_wd4col,
    "idx_json": _load_selected_tags_idx_json,
    "camie_metadata_json": _load_selected_tags_camie_metadata_json,
}

def load_selected_tags(tags_csv: str | Path, csv_format: TagsCsvFormat = "pixai6col") -> list[TagMeta]:
    tags_path = Path(tags_csv)
    if not tags_path.is_file():
        log_dbg(_get_string("TaggerCore", "Tag_CSV_File_Not_Found", tags_csv=str(tags_csv)))
        raise FileNotFoundError(_get_string("TaggerCore", "Tag_CSV_File_Not_Found", tags_csv=str(tags_csv)))
    loader = _TAGS_CSV_LOADERS.get(csv_format, _load_selected_tags_pixai6col)
    return loader(tags_path)

def discover_labels_csv(model_dir: Path, tags_csv: str | Path | None) -> Path | None:
    if tags_csv:
        candidate = Path(tags_csv)
        return candidate if candidate.exists() else None
    search_dir = model_dir
    candidates: list[Path] = []
    default_names = ("selected_tags.csv", "selected_tags_v3.csv", "selected_tags_v3c.csv")
    for name in default_names:
        candidate = search_dir / name
        if candidate.is_file() and candidate not in candidates:
             candidates.append(candidate)
    for extra in sorted(search_dir.glob("selected_tags*.csv")):
        if extra not in candidates:
            candidates.append(extra)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

def _sigmoid(x: NDArray[np.float_]) -> NDArray[np.float_]:
    return 1 / (1 + np.exp(-x))

def _normalize_np_chw(x: NDArray[np.float32], mean: NDArray[np.float_], std: NDArray[np.float_]) -> NDArray[np.float32]:
    x = x.astype(np.float32, copy=False)
    for c in range(3):
        x[c] = (x[c] - mean[c]) / std[c]
    return x

def make_cpu_session_options(intra_op_num_threads: int):
    """Return an ort.SessionOptions that caps intra-op parallelism, or None.

    Driven by config.ini [Behavior] onnx_threads: 0 (the default) returns None, i.e.
    ONNX Runtime keeps its own default of one thread per physical core - byte-for-byte
    the same as before this option existed. A positive N caps intra_op_num_threads at
    N so a large batch does not pin every core (issue: "20k tagging makes the machine
    unusable"). Shared by the tagger and the Florence-2 captioner sessions.
    """
    try:
        n = int(intra_op_num_threads)
    except (TypeError, ValueError):
        return None
    if n <= 0 or ort is None:
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = n
    return so


@dataclass(frozen=True)
class InferenceConfig:
    """
    Per-model preprocessing/postprocessing description (design.md 6.1〜6.5節, 7章).
    Defaults match the original hardcoded PixAI OnnxTagger behavior exactly so that
    OnnxTagger(model_path, tags_csv) with no inference_config is byte-for-byte
    unchanged from before this class existed (NFR-3 regression guarantee).
    """
    input_size: int = 448
    layout: str = "nchw"                 # "nchw" | "nhwc"
    channel_order: str = "rgb"           # "rgb" | "bgr"
    rescale_to_unit: bool = True         # divide by 255 before normalizing
    pad_color_rgb: tuple[int, int, int] = (0, 0, 0)
    normalize_mean: tuple[float, float, float] | None = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] | None = (0.229, 0.224, 0.225)
    extra_inputs: tuple[str, ...] = ()   # e.g. ("padding_mask",)
    output_name_hint: str | None = None
    output_activation: str = "sigmoid"   # "sigmoid" | "none" (already applied in-graph)
    tags_csv_format: TagsCsvFormat = "pixai6col"

class OnnxTagger:
    # Kept as class constants (matching InferenceConfig's own defaults) purely so any
    # external code that still references OnnxTagger.INPUT_SIZE etc. keeps working.
    INPUT_SIZE = 448
    MODEL_MEAN: NDArray[np.float_] = np.array([0.485, 0.456, 0.406])
    MODEL_STD: NDArray[np.float_] = np.array([0.229, 0.224, 0.225])
    input_name: str
    output_name: str
    tags: list[TagMeta]
    tag_meta_lookup: dict[str, TagMeta]
    session: ort.InferenceSession
    get_string: GetString
    inference_config: InferenceConfig

    def __init__(
        self,
        model_path: Path,
        tags_csv: Path | None = None,
        get_string: GetString | None = None,
        inference_config: InferenceConfig | None = None,
        intra_op_num_threads: int = 0,
        onnx_device: str = "cpu",
    ):
        self.get_string = get_string if get_string else _get_string
        self.inference_config = inference_config if inference_config is not None else InferenceConfig()

        if not model_path.is_file():
            log_dbg(self.get_string("TaggerCore", "Model_File_Not_Found", model_path=str(model_path)))
            raise FileNotFoundError(self.get_string("TaggerCore", "Model_File_Not_Found", model_path=str(model_path)))
        if ort is None:
            log_dbg(self.get_string("TaggerCore", "Onnxruntime_Not_Installed"))
            raise ImportError(self.get_string("TaggerCore", "Onnxruntime_Not_Installed"))
        log_dbg(self.get_string("TaggerCore", "Info_ONNX_Session_Creation_Start", model_path=model_path.name))
        sess_options = make_cpu_session_options(intra_op_num_threads)
        if sess_options is not None:
            log_dbg(f"OnnxTagger: intra_op_num_threads capped at {sess_options.intra_op_num_threads} ([Behavior] onnx_threads)")
        self.session = make_session(model_path, sess_options=sess_options,
                                    prefer=onnx_device, label="OnnxTagger")
        log_dbg(self.get_string("TaggerCore", "Info_ONNX_Session_Created"))
        model_dir = model_path.parent
        tags_path = discover_labels_csv(model_dir, tags_csv)
        if not tags_path or not tags_path.is_file():
            log_dbg(self.get_string("TaggerCore", "Tag_CSV_File_Not_Found_Check_Dir", model_dir=str(model_dir)))
            raise FileNotFoundError(self.get_string("TaggerCore", "Tag_CSV_File_Not_Found_Check_Dir", model_dir=str(model_dir)))
        self.tags = load_selected_tags(tags_path, self.inference_config.tags_csv_format)
        self.tag_meta_lookup = {tag.name: tag for tag in self.tags}
        log_dbg(self.get_string("TaggerCore", "Loaded_Tags_Count", count=len(self.tags), tags_path=tags_path.name))

        inputs: list[Any] = list(self.session.get_inputs()) # type: ignore
        self.input_name = inputs[0].name

        outputs: list[Any] = list(self.session.get_outputs()) # type: ignore
        output_names: list[str] = [output.name for output in outputs]

        if self.inference_config.output_name_hint and self.inference_config.output_name_hint in output_names:
            self.output_name = self.inference_config.output_name_hint
        elif len(output_names) == 1:
            # A single-output classifier is unambiguous: use it even when the config's
            # output_name hint is absent or stale (e.g. a community re-export that renamed
            # the tensor). Models with several outputs still require an accurate hint.
            self.output_name = output_names[0]
        else:
            preferred_order = ("prediction", "logits")
            for name in preferred_order:
                if name in output_names:
                    self.output_name = name
                    break
            else:
                log_dbg(self.get_string("TaggerCore", "ONNX_Prediction_Tensor_NotFound", output_names=str(output_names)))
                raise RuntimeError(self.get_string("TaggerCore", "ONNX_Prediction_Tensor_NotFound", output_names=str(output_names)))

    def prepare_batch_from_rgb_np(self, images: Sequence[NDArray[np.uint8]]) -> tuple[NDArray[np.float32], dict[str, NDArray[Any]]]:
        cfg = self.inference_config
        target_size = cfg.input_size
        preprocessed_images: list[NDArray[np.float32]] = []
        padding_masks: list[NDArray[np.bool_]] = []
        needs_padding_mask = "padding_mask" in cfg.extra_inputs

        for img_array in images:
            image_pil = Image.fromarray(img_array)
            w, h = image_pil.size
            ratio = min(target_size / w, target_size / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            resized_image = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_size, target_size), cfg.pad_color_rgb)
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            canvas.paste(resized_image, (x_offset, y_offset))

            if needs_padding_mask:
                mask = np.ones((target_size, target_size), dtype=np.bool_)
                mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = False
                padding_masks.append(mask)

            img_np = np.asarray(canvas, dtype=np.float32)
            if cfg.channel_order == "bgr":
                img_np = img_np[:, :, ::-1]
            if cfg.rescale_to_unit:
                img_np = img_np / 255.0

            if cfg.layout == "nhwc":
                normalized = img_np
                if cfg.normalize_mean is not None and cfg.normalize_std is not None:
                    mean = np.asarray(cfg.normalize_mean, dtype=np.float32)
                    std = np.asarray(cfg.normalize_std, dtype=np.float32)
                    normalized = (normalized - mean) / std
                normalized = normalized.astype(np.float32, copy=False)
            else:
                img_chw = img_np.transpose((2, 0, 1))
                if cfg.normalize_mean is not None and cfg.normalize_std is not None:
                    mean = np.asarray(cfg.normalize_mean, dtype=np.float32)
                    std = np.asarray(cfg.normalize_std, dtype=np.float32)
                    normalized = _normalize_np_chw(img_chw, mean, std)
                else:
                    normalized = img_chw.astype(np.float32, copy=False)
            preprocessed_images.append(normalized)

        extra_input_batches: dict[str, NDArray[Any]] = {}
        if not preprocessed_images:
            empty_shape = (0, target_size, target_size, 3) if cfg.layout == "nhwc" else (0, 3, target_size, target_size)
            batch = np.empty(empty_shape, dtype=np.float32)
        else:
            batch = np.stack(preprocessed_images, axis=0)
        if needs_padding_mask:
            extra_input_batches["padding_mask"] = (
                np.stack(padding_masks, axis=0) if padding_masks else np.empty((0, target_size, target_size), dtype=np.bool_)
            )
        return batch, extra_input_batches

    def infer_batch_prepared(
        self,
        batch: NDArray[np.float32],
        extra_inputs: Mapping[str, NDArray[Any]] | None = None,
        *,
        thresholds: Mapping[TagCategory, float] | None = None,
        max_tags: Mapping[TagCategory, int] | None = None,
    ) -> list[TagResult]:
        if batch.size == 0:
            return []
        input_feed: dict[str, Any] = {self.input_name: batch}
        if extra_inputs:
            input_feed.update(extra_inputs)
        outputs = self.session.run([self.output_name], input_feed) # type: ignore
        output_array = np.asarray(outputs[0], dtype=np.float_)
        if self.inference_config.output_activation == "none":
            scores_batch = output_array
        else:
            scores_batch = _sigmoid(output_array)
        results: list[TagResult] = []
        
        cat_thresholds: Mapping[TagCategory, float] = thresholds if thresholds is not None else {}
        cat_limits: Mapping[TagCategory, int] = max_tags if max_tags is not None else {}

        if not self.tags:
             return [TagResult() for _ in scores_batch]
        hard_cap = sum(cat_limits.values()) if cat_limits else 100
        score_floor = 1e-4
        for scores in scores_batch:
            raw_predictions: list[TagPrediction] = []
            for tag_meta, score in zip(self.tags, scores):
                if score < score_floor:
                    continue
                category = TagCategory(tag_meta.category)
                threshold = cat_thresholds.get(category, cat_thresholds.get(TagCategory.GENERAL, 0.0))
                if float(score) < threshold:
                    continue
                raw_predictions.append(TagPrediction(
                    name=tag_meta.name,
                    score=float(score),
                    category=category
                ))
            ordered = sorted(raw_predictions, key=lambda pred: (-pred.score, pred.name))
            taken: list[TagPrediction] = []
            per_category: dict[TagCategory, int] = {}
            for prediction in ordered:
                if len(taken) >= hard_cap:
                    break
                
                category = prediction.category
                limit = cat_limits.get(category)

                current = per_category.get(category, 0)
                if limit is not None and current >= limit:
                    continue
                
                per_category[category] = current + 1
                taken.append(prediction)
            results.append(TagResult(tags=taken))
        return results

    def infer_batch(self, images: Sequence[Image.Image], *, thresholds: Mapping[TagCategory, float] | None = None, max_tags: Mapping[TagCategory, int] | None = None) -> list[TagResult]:
        rgb_arrays = [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images]
        batch, extra_inputs = self.prepare_batch_from_rgb_np(rgb_arrays)
        return self.infer_batch_prepared(batch, extra_inputs, thresholds=thresholds, max_tags=max_tags)

def get_image_paths_recursive(base_dir: Path) -> list[Path]:
    IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]
    image_paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(base_dir.rglob(f"*{ext}"))
    return sorted(image_paths)

def format_tags(tag_results: TagResult, convert_underscore: bool) -> str:
    # Output order: artist, copyright, character, <series tags>, general, meta, rating.
    # "character, <series>, general" is kept exactly as before, so models that only
    # emit general + character tags (which was every model until per-category limits
    # were added) produce byte-identical output. The other categories only appear
    # when the user raises their max-count above 0 in the per-category dialog.
    by_category: dict[TagCategory, list[TagPrediction]] = {}
    for pred in tag_results.tags:
        by_category.setdefault(TagCategory(pred.category), []).append(pred)

    def _emit(preds: list[TagPrediction]) -> list[str]:
        names: list[str] = []
        for pred in sorted(preds, key=lambda p: p.score, reverse=True):
            name = pred.name.replace("_", " ") if convert_underscore else pred.name
            names.append(name)
        return names

    output_tags: list[str] = []
    output_tags += _emit(by_category.get(TagCategory.ARTIST, []))
    output_tags += _emit(by_category.get(TagCategory.COPYRIGHT, []))
    output_tags += _emit(by_category.get(TagCategory.CHARACTER, []))
    for series_tag in tag_results.series_tags:
        output_tags.append(series_tag.replace("_", " ") if convert_underscore else series_tag)
    output_tags += _emit(by_category.get(TagCategory.GENERAL, []))
    output_tags += _emit(by_category.get(TagCategory.META, []))
    output_tags += _emit(by_category.get(TagCategory.MODEL, []))
    output_tags += _emit(by_category.get(TagCategory.QUALITY, []))
    output_tags += _emit(by_category.get(TagCategory.YEAR, []))
    output_tags += _emit(by_category.get(TagCategory.RATING, []))
    return ", ".join(output_tags)

def build_inference_config(model_config: dict[str, Any]) -> InferenceConfig:
    """
    Builds an InferenceConfig from a model_config.json dict's "inference"/"tags_csv"
    blocks. Any key missing from the dict falls back to InferenceConfig's own
    (PixAI-matching) default, so a partially-specified model_config.json is safe.
    """
    inference = config_mapping(model_config, "inference")
    tags_csv = config_mapping(model_config, "tags_csv")
    defaults = InferenceConfig()

    def _tuple3(value: Any) -> tuple[float, float, float] | None:
        # `value` has already had its default applied by inference.get(...); None here
        # means the model explicitly disables normalization.
        if value is None:
            return None
        return tuple(value)  # type: ignore[return-value]

    pad_color = inference.get("pad_color_rgb")
    normalize_mean = inference.get("normalize_mean", defaults.normalize_mean)
    normalize_std = inference.get("normalize_std", defaults.normalize_std)
    extra_inputs = inference.get("extra_inputs", list(defaults.extra_inputs))

    return InferenceConfig(
        input_size=inference.get("input_size", defaults.input_size),
        layout=inference.get("layout", defaults.layout),
        channel_order=inference.get("channel_order", defaults.channel_order),
        rescale_to_unit=inference.get("rescale_to_unit", defaults.rescale_to_unit),
        pad_color_rgb=tuple(pad_color) if pad_color is not None else defaults.pad_color_rgb,  # type: ignore[arg-type]
        normalize_mean=_tuple3(normalize_mean),
        normalize_std=_tuple3(normalize_std),
        extra_inputs=tuple(extra_inputs),
        output_name_hint=inference.get("output_name", defaults.output_name_hint),
        output_activation=inference.get("output_activation", defaults.output_activation),
        tags_csv_format=tags_csv.get("format", defaults.tags_csv_format),
    )

def setup_tagger_from_settings(app_settings: AppSettings, get_string: GetString | None) -> tuple[OnnxTagger | None, dict[str, Any]]:
    """Initializes the tagger and extracts settings from the AppSettings object."""
    _get_string_internal = get_string if get_string else _get_string
    try:
        import model_registry  # local import to avoid a hard import-time dependency for callers that never tag

        model_id = app_settings.model.model_id
        entry = model_registry.get_model_entry(model_id)

        tags_csv_path: Path | None = None
        if entry is not None and entry.model_id == "pixai-tagger-v0.9":
            # PixAI pseudo-entry: entry.model_dir is constants.MODEL_PATH.parent, which
            # already resolves the models/pixai-tagger-v0.9/ location with a legacy-path
            # fallback. Using app_settings.paths.model_dir here would point at a third,
            # non-existent path on a fresh install.
            import constants
            model_path = constants.MODEL_PATH
            inference_config = InferenceConfig()
        elif entry is None:
            # Truly unknown model_id (e.g. stale config) -> legacy PixAI path,
            # matching pre-multi-model behavior exactly (NFR-3).
            model_path = BASE_DIR / app_settings.paths.model_dir / app_settings.paths.model_filename
            inference_config = InferenceConfig()
        else:
            model_path = entry.model_dir / "model.onnx"
            inference_config = build_inference_config(entry.config)
            # Models whose tag-metadata file isn't named selected_tags*.csv (camie's
            # own metadata JSON, cl_tagger's tag_mapping.json) must say so explicitly -
            # discover_labels_csv()'s glob would never find them otherwise.
            tags_file_name = config_mapping(entry.config, "tags_csv").get("file_name")
            if tags_file_name:
                tags_csv_path = entry.model_dir / tags_file_name

        # Per-category thresholds / limits.
        #
        # EVERY category is seeded as "blocked" (threshold above any possible score,
        # limit 0) and only the categories this model declares in ui.categories are then
        # given the user's settings. Leaving a category out of these dicts is NOT safe:
        # infer_batch_prepared() falls back to the general threshold for an unlisted
        # category and treats a missing limit as unbounded, and filter_tags_by_solo_rule()
        # now passes non-general/character predictions straight through - so an undeclared
        # category would silently reach the output file with no user-visible control.
        ui_cfg = config_mapping(entry.config, "ui") if entry is not None else {}
        model_categories: list[str] = list(ui_cfg.get("categories", ["general", "character"]))
        tag_thresholds: dict[TagCategory, float] = {cat: 1.1 for cat in TagCategory}
        max_tags_per_category: dict[TagCategory, int] = {cat: 0 for cat in TagCategory}
        declared: set[TagCategory] = set()
        for name in model_categories:
            cat = CATEGORY_NAME_TO_ENUM.get(name)
            if cat is None:
                log_dbg(f"setup_tagger_from_settings: model '{model_id}' declares unknown category {name!r}; ignoring.")
                continue
            declared.add(cat)
            tag_thresholds[cat] = float(getattr(app_settings.thresholds, name, app_settings.thresholds.general))
            max_tags_per_category[cat] = int(getattr(app_settings.limits, name, 0))

        settings_dict: dict[str, Any] = {
            'INPUT_DIR': Path(app_settings.paths.input_dir),
            'MODEL_PATH': model_path,
            'TAG_THRESHOLDS': tag_thresholds,
            'MAX_TAGS_PER_CATEGORY': max_tags_per_category,
            'ENABLE_SOLO_LIMIT': app_settings.behavior.enable_solo_character_limit,
            'CONVERT_UNDERSCORE': app_settings.behavior.convert_underscore_to_space,
            'EXISTING_FILE_MODE': parse_existing_file_mode(app_settings.behavior.existing_file_mode, _get_string_internal),
        }
        tagger = OnnxTagger(model_path=settings_dict['MODEL_PATH'], tags_csv=tags_csv_path, get_string=_get_string_internal, inference_config=inference_config, intra_op_num_threads=app_settings.behavior.onnx_threads, onnx_device=app_settings.behavior.onnx_device)

        # Surface a model_config.json whose ui.categories does not cover what its tag file
        # actually contains: those tags are blocked above (limit 0), so without this log
        # the omission would be invisible.
        undeclared = {TagCategory(t.category) for t in tagger.tags} - declared
        if undeclared:
            log_dbg(f"setup_tagger_from_settings: model '{model_id}' has tags in "
                    f"{sorted(c.name for c in undeclared)} but does not declare them in "
                    f"ui.categories - those tags will not be emitted.")

        return tagger, settings_dict
    except Exception as e:
        log_dbg(f"Error during Tagger initialization: {type(e).__name__}: {e}")
        log_dbg(f"Traceback: {traceback.format_exc()}")
        return None, {}


def process_image_loop(
    tagger: OnnxTagger,
    settings: dict[str, Any],
    image_paths: list[Path],
    decision_resolver: Callable[[Path], OverwriteDecision] | None,
    log_gui: Callable[[str, str], None] | None,
    stop_checker: Callable[[], bool] | None,
    get_string: GetString | None,
    progress_cb: Callable[[int, int], None] | None = None,
    failed_paths: list[Path] | None = None,
) -> list[FileChange]:
    """設定に従って画像へタグを付け、既存 .txt の扱いを EXISTING_FILE_MODE で分岐する。

    `decision_resolver` は ASK モードのときだけ呼ばれる（spec.md 1.2節）。ASK 以外の
    モードでは、既存ファイル検出時にワーカースレッドから GUI への往復は一切発生しない。

    戻り値: 実際に書き換えたファイルの FileChange リスト。呼び出し側はこれを1つの
    Undo エントリ（CompositeUndoAction）にまとめる（design.md 4.4節）。

    `progress_cb(done, total)` は全体で最大 ~200 回＋最後の N/N（完了）だけ呼ぶ。
    issue #10: 1画像=1〜2回の GUI ログ発行だと高速ループ（SKIP など）で Qt イベント
    キューが飽和して固まるため、ルーチンの「処理中／出力成功」ログは GUI へ出さず、
    間引いた progress_cb へ回す（progress_cb 自体もクロススレッドの queued signal なので
    毎画像発行はしない）。スキップ・エラー・警告・追記件数など有界・低頻度のログは
    従来どおり GUI へ出す。
    """

    def core_log_gui(message: str, color: str = "black") -> None:
        if log_gui:
            log_gui(message, color)

    _get_string_internal = get_string if get_string else _get_string
    mode: ExistingFileMode = settings.get('EXISTING_FILE_MODE', ExistingFileMode.ASK)
    changed_files: list[FileChange] = []
    total = len(image_paths)
    # issue #10: 1画像=1 GUI ログだと全既存フォルダの SKIP で Qt キューが飽和する。
    # 個々の skip / 成功は debug log だけに残し、GUI へは末尾のサマリ1行だけ出す。
    n_skipped = 0
    n_errors = 0
    n_unchanged = 0
    failed_seen = set(failed_paths or ())
    # progress_cb 自体もクロススレッドの queued signal なので、毎画像発行すると
    # シグナルのキュー投入コストが積み上がる（PR#16 レビュー指摘）。全体で ~200 回に
    # 間引く。最後の1枚は必ず発行して N/N（完了）に到達させる。
    # 天井除算にする: total//200 だと 201〜399 枚で step=1 になり間引きが効かない。
    progress_step = max(1, (total + 199) // 200)

    def mark_failed(path: Path) -> None:
        if failed_paths is not None and path not in failed_seen:
            failed_seen.add(path)
            failed_paths.append(path)

    def mark_remaining_failed(start: int) -> None:
        """Record unprocessed work without labelling SKIP outputs as failures."""
        if failed_paths is None:
            return
        for pending in image_paths[start:]:
            if (mode is ExistingFileMode.SKIP
                    and pending.with_suffix(".txt").is_file()):
                continue
            mark_failed(pending)

    for i, image_path in enumerate(image_paths):
        if progress_cb and ((i + 1) % progress_step == 0 or i == total - 1):
            progress_cb(i + 1, total)
        if stop_checker and stop_checker():
            core_log_gui(_get_string_internal("TaggerCore", "Tagging_Process_Aborted_By_User"), "red")
            log_dbg(_get_string_internal("TaggerCore", "Tagging_Process_Aborted_By_User_Debug"))
            mark_remaining_failed(i)
            break

        # First, check if the output file exists and should be skipped.
        base_name, _ = os.path.splitext(str(image_path))
        output_path = Path(base_name + ".txt")
        relative_path = image_path.relative_to(settings['INPUT_DIR'])
        current_index_str = f"[{i+1}/{len(image_paths)}]"

        will_append = False
        if output_path.is_file():
            if mode is ExistingFileMode.SKIP:
                n_skipped += 1
                log_dbg(_get_string_internal("TaggerCore", "Tag_Output_Skipped_Existing_File", current_index_str=current_index_str, relative_path=str(relative_path)))
                continue
            if mode is ExistingFileMode.ASK:
                if decision_resolver is None:
                    # 防御: resolver 未設定なら既存ファイルには触らない
                    n_skipped += 1
                    log_dbg("process_image_loop: ASK モードだが decision_resolver が未設定のためスキップします")
                    continue
                if stop_checker and stop_checker():
                    # ここで停止要求が来ていたら resolver（GUI 往復しうる）を呼ばずに抜ける
                    mark_remaining_failed(i)
                    break
                decision = decision_resolver(output_path)
                if decision is OverwriteDecision.SKIP:
                    n_skipped += 1
                    log_dbg(_get_string_internal("TaggerCore", "Tag_Output_Skipped_Existing_File", current_index_str=current_index_str, relative_path=str(relative_path)))
                    continue
                will_append = decision is OverwriteDecision.APPEND
            else:
                will_append = mode is ExistingFileMode.APPEND

        # ルーチンの「処理中」表示は GUI へ流さず debug log のみ（issue #10）。GUI は progress_cb 経由。
        log_dbg(_get_string_internal("TaggerCore", "Processing_Image", current_index_str=current_index_str, relative_path=str(relative_path)))

        try:
            with open(image_path, 'rb') as f:
                image = Image.open(f).convert("RGB")
        except Exception as e:
            n_errors += 1
            mark_failed(image_path)
            log_dbg(_get_string_internal("TaggerCore", "Image_Load_Failed", current_index_str=current_index_str, relative_path=str(relative_path), type_e_name=type(e).__name__, e=str(e)))
            core_log_gui(_get_string_internal("TaggerCore", "Image_Load_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "red")
            continue

        try:
            results = tagger.infer_batch(
                images=[image],
                thresholds=settings['TAG_THRESHOLDS'],
                max_tags=settings['MAX_TAGS_PER_CATEGORY'],
            )
        except Exception as e:
            n_errors += 1
            mark_failed(image_path)
            log_dbg(_get_string_internal("TaggerCore", "Tag_Inference_Failed", current_index_str=current_index_str, relative_path=str(relative_path), type_e_name=type(e).__name__, e=str(e)))
            core_log_gui(_get_string_internal("TaggerCore", "Tag_Inference_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "red")
            continue

        if not results or not results[0].tags:
            n_errors += 1
            mark_failed(image_path)
            log_dbg(_get_string_internal("TaggerCore", "Tag_Acquisition_Failed", current_index_str=current_index_str, relative_path=str(relative_path)))
            core_log_gui(_get_string_internal("TaggerCore", "Tag_Acquisition_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "orange")
            continue

        tag_result = results[0]
        
        final_tags, all_series_tags = filter_tags_by_solo_rule(
            tag_result, tagger, settings['ENABLE_SOLO_LIMIT']
        )
        tag_result.tags = final_tags
        tag_result.series_tags = tuple(sorted(list(all_series_tags)))
        
        formatted_tags = format_tags(tag_result, settings['CONVERT_UNDERSCORE'])

        previous_content: str | None = None
        added_tags: tuple[str, ...] = ()
        if will_append:
            try:
                previous_content = output_path.read_text(encoding='utf-8')
            except Exception as e:
                # 既存内容が読めないファイルは触らずスキップする（spec.md 3.3節）
                n_errors += 1
                mark_failed(image_path)
                log_dbg(f"append: 既存ファイルの読み込みに失敗したためスキップします {output_path.name}: {type(e).__name__}: {e}")
                core_log_gui(_get_string_internal("TaggerCore", "Save_Failed_Short", current_index_str=current_index_str, output_path_name=output_path.name), "red")
                continue
            existing_tags = [t.strip() for t in previous_content.split(",") if t.strip()]
            generated_tags = [t.strip() for t in formatted_tags.split(",") if t.strip()]
            merged = merge_tags(existing_tags, generated_tags)
            if merged == existing_tags:
                # 追加できる新規タグが無い場合は mtime も変えない（spec.md 3.3節）
                # ルーチン結果なので GUI へは出さず debug log のみ（issue #10）。
                n_unchanged += 1
                log_dbg(_get_string_internal("TaggerCore", "Log_No_New_Tags", current_index_str=current_index_str, file_name=output_path.name))
                continue
            added_tags = tuple(merged[len(existing_tags):])
            new_content = ", ".join(merged)
        else:
            # 新規作成（previous_content=None）または上書き
            if output_path.is_file():
                try:
                    previous_content = output_path.read_text(encoding='utf-8')
                except Exception as e:
                    # 既存だが読めないファイル（非UTF-8等）はここで previous_content=None
                    # のまま書くと「元は存在しなかった」と undo に誤認され、undo で
                    # ファイルごと削除して原本を復元不能に破壊する（PR#16 レビュー指摘）。
                    # APPEND 分岐と同様、触らずスキップする。
                    n_errors += 1
                    mark_failed(image_path)
                    log_dbg(f"overwrite: 既存ファイルの読み込みに失敗したためスキップします {output_path.name}: {type(e).__name__}: {e}")
                    core_log_gui(_get_string_internal("TaggerCore", "Save_Failed_Short", current_index_str=current_index_str, output_path_name=output_path.name), "red")
                    continue
            new_content = formatted_tags

        try:
            resolved_path = output_path.resolve()
            if sys.platform == "win32":
                long_path_str = f"\\\\?\\{resolved_path}"
            else:
                long_path_str = str(resolved_path)

            with open(long_path_str, 'w', encoding='utf-8') as f:
                f.write(new_content)

            changed_files.append(FileChange(
                path=output_path, previous_content=previous_content,
                new_content=new_content, was_append=will_append, added_tags=added_tags))
            # 個々の追記／出力成功は debug log のみ。件数は末尾サマリで GUI へ（issue #10）。
            if will_append:
                log_dbg(_get_string_internal("TaggerCore", "Log_Appended", current_index_str=current_index_str, count=len(added_tags), file_name=output_path.name))
            else:
                log_dbg(_get_string_internal("TaggerCore", "Tag_Output_Success", current_index_str=current_index_str, output_path_name=output_path.name))
            log_dbg(_get_string_internal("TaggerCore", "Tagging_Result_Output", current_index_str=current_index_str, output_path_name=output_path.name))
        except Exception as e:
            n_errors += 1
            mark_failed(image_path)
            log_dbg(_get_string_internal("TaggerCore", "Save_Failed", current_index_str=current_index_str, relative_path=str(relative_path), type_e_name=type(e).__name__, e=str(e)))

            core_log_gui(_get_string_internal("TaggerCore", "Save_Failed_Short", current_index_str=current_index_str, output_path_name=output_path.name), "red")

    n_appended = sum(1 for c in changed_files if c.was_append)
    core_log_gui(_get_string_internal(
        "TaggerCore", "Batch_Summary",
        written=len(changed_files), appended=n_appended, skipped=n_skipped,
        unchanged=n_unchanged, errors=n_errors, total=total), "blue")
    return changed_files

def filter_tags_by_solo_rule(
    tag_result: TagResult,
    tagger: OnnxTagger,
    enable_solo_limit: bool
) -> tuple[list[TagPrediction], set[str]]:
    """
    if solo tag is available.    
    Returns:
        (final_tags, all_series_tags)
    """
    general_tags = [pred for pred in tag_result.tags if pred.category == TagCategory.GENERAL]
    character_tags = [pred for pred in tag_result.tags if pred.category == TagCategory.CHARACTER]
    # Everything else (rating / copyright / artist / meta / model / quality / year) is
    # untouched by the solo rule - it only ever trims character tags. Dropping these here
    # would make the per-category thresholds / max-tag limits have no effect on output.
    other_tags = [pred for pred in tag_result.tags
                  if pred.category not in (TagCategory.GENERAL, TagCategory.CHARACTER)]
    solo_tag_found = any(pred.name.lower() == "solo" for pred in general_tags)
    all_series_tags: set[str] = set()
    final_tags: list[TagPrediction]

    # if solo tag is available, and a character tag exists
    if enable_solo_limit and solo_tag_found and character_tags:
        # Only keep the character tag with the highest score.
        character_tags.sort(key=lambda pred: pred.score, reverse=True)
        best_character_tag = character_tags[0]

        # get series tags from the best character tag
        char_meta = tagger.tag_meta_lookup.get(best_character_tag.name)
        if char_meta and char_meta.ips:
            all_series_tags.update(char_meta.ips)

        final_tags = general_tags + [best_character_tag] + other_tags
    else:
        # if solo tag not found
        for char_pred in character_tags:
            char_meta = tagger.tag_meta_lookup.get(char_pred.name)
            if char_meta and char_meta.ips:
                all_series_tags.update(char_meta.ips)

        final_tags = general_tags + character_tags + other_tags

    return final_tags, all_series_tags

def main(decision_resolver: Callable[[Path], OverwriteDecision] | None = None, log_gui: Callable[[str, str], None] | None = None, stop_checker: Callable[[], bool] | None = None, get_string: GetString | None = None):
    start_time = perf_counter()
    
    def core_log_gui(message: str, color: str = "black") -> None:
        if log_gui:
            log_gui(message, color)

    _get_string_internal = get_string if get_string else _get_string
    log_dbg(_get_string_internal("TaggerCore", "Info_Tagging_Core_Main_Start"))

    try:
        assert CONFIG_PATH.is_file(), _get_string_internal("TaggerCore", "Config_File_NotFound", CONFIG_PATH=str(CONFIG_PATH))
        config = configparser.ConfigParser()
        config.read(CONFIG_PATH, encoding='utf-8')
    except Exception as e:
        log_dbg(_get_string_internal("TaggerCore", "Fatal_Error_Config_Load_Failed", type_e_name=type(e).__name__, e=str(e)))
        core_log_gui(_get_string_internal("TaggerCore", "Fatal_Error_Config_Load_Failed_GUI"), "red")
        return

    app_settings = load_settings(config)

    tagger, settings_dict = setup_tagger_from_settings(app_settings, _get_string_internal)
    if not tagger or not settings_dict:
        core_log_gui(_get_string_internal("TaggerCore", "Error_Tagger_Init_Failed_GUI"), "red")
        return

    image_paths = get_image_paths_recursive(settings_dict['INPUT_DIR'])
    if not image_paths:
        log_dbg(_get_string_internal("TaggerCore", "Warning_No_Image_Files_Found_In_Dir", INPUT_DIR=str(settings_dict['INPUT_DIR'])))
        core_log_gui(_get_string_internal("TaggerCore", "Warning_No_Image_Files_Found_GUI"), "orange")
        return

    core_log_gui(_get_string_internal("TaggerCore", "Total_Image_Files_Found", count=len(image_paths)), "blue")
    
    process_image_loop(tagger, settings_dict, image_paths, decision_resolver, log_gui, stop_checker, _get_string_internal)

    end_time = perf_counter()
    log_dbg(_get_string_internal("TaggerCore", "Total_Processing_Time", time=f"{end_time - start_time:.2f}"))
    core_log_gui(_get_string_internal("TaggerCore", "Tagging_Process_Complete"), "green")

if __name__ == "__main__":
    main()


class TargetMode(Enum):
    """タグ付け対象の選択方針。"""
    ALL = auto()          # 全画像
    UNPROCESSED = auto()  # .txt が存在しない画像のみ
    FAILED = auto()       # 失敗リストとの積集合
    SELECTED = auto()     # 選択中の1件のみ


TARGET_MODE_MAP: dict[str, TargetMode] = {
    "ALL": TargetMode.ALL,
    "UNPROCESSED": TargetMode.UNPROCESSED,
    "FAILED": TargetMode.FAILED,
    "SELECTED": TargetMode.SELECTED,
}


def parse_target_mode(raw: str, get_string: GetString | None = None) -> TargetMode:
    """文字列を TargetMode に変換する。不正値は ALL にフォールバックする。"""
    mode = TARGET_MODE_MAP.get(str(raw).strip().upper())
    if mode is None:
        _get_string_internal = get_string if get_string else _get_string
        log_dbg(_get_string_internal("TaggerCore", "Invalid_Target_Mode_Debug", raw=raw))
        return TargetMode.ALL
    return mode


def filter_target_images(
    paths: Sequence[Path],
    mode: TargetMode,
    failed_paths: Sequence[Path] = (),
    selected: Path | None = None,
) -> list[Path]:
    """TargetMode に従って対象画像を絞り込む純粋関数。

    - ALL: そのまま返す。
    - UNPROCESSED: `.txt` が存在しないものだけ返す（空ファイルも処理済み扱い）。
    - FAILED: failed_paths との積集合を入力順で返す。
    - SELECTED: selected が入力内にあれば `[selected]`、なければ `[]`。
    """
    if mode is TargetMode.UNPROCESSED:
        return [p for p in paths if not p.with_suffix(".txt").is_file()]
    if mode is TargetMode.FAILED:
        failed_set = set(failed_paths)
        return [p for p in paths if p in failed_set]
    if mode is TargetMode.SELECTED:
        if selected is not None and selected in paths:
            return [selected]
        return []
    return list(paths)
