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

import constants
from constants import BASE_DIR

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from PIL import Image, ImageOps
    import onnxruntime as ort # type: ignore
else:
    try:
        import numpy as np
        from numpy.typing import NDArray
        from PIL import Image, ImageOps
        import onnxruntime as ort # type: ignore
    except ImportError:
        np = None
        NDArray = None
        Image = None
        ImageOps = None
        ort = None

LOG_FILE_PATH = BASE_DIR / "debug_log.txt"
CONFIG_PATH = BASE_DIR / "config.ini"

from utils import config_mapping, log_dbg, GetString
from app_settings import AppSettings, load_settings
from onnx_providers import make_session


# カテゴリ別の最大タグ数が1つも指定されていない場合に適用する出力上限。
# 呼び出し側（GUI）は必ず設定値を渡すので、これが効くのは max_tags を省略して
# predict() を直接呼んだとき（単体テスト・スクリプト利用）だけ。出力仕様に影響
# するため、無名の 100 ではなく根拠付きの定数として置く。
DEFAULT_TAG_OUTPUT_CAP = 100

# 透過画像を推論へ渡すときの合成背景（REQ-A9）。マニフェストで
# inference.alpha_background_rgb を指定していないモデルにも同じ既定を適用する。
# 値は唯一の既存定義（pixai-tagger-v1.0 の [255, 255, 255]）と VLM 経路の
# 平坦化色に合わせた白。モデルごとに透明の扱いが変わるのを防ぐ。
DEFAULT_ALPHA_BACKGROUND_RGB = (255, 255, 255)


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
    STYLE = 9

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
    "style": TagCategory.STYLE,
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
    "style": 9,
}

TagsCsvFormat = str  # "pixai6col" | "simple3col" | "wd4col" | "idx_json" | "pixai_v1_config" | "camie_metadata_json"

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

def _category_slot(category_key: Any) -> int | None:
    """閾値／上限の Mapping のキーをカテゴリ ID（int）へ落とす。

    `TagCategory` は IntEnum なので int キーでも同じスロットを指す。解釈できない
    キーは呼び出し側で「使われないキー」として無視する（改修前の辞書引きと同じ挙動）。
    """
    try:
        return int(category_key)
    except (TypeError, ValueError):
        return None


def _cell(cells: list[str], index: int) -> str:
    """CSV 行から列を取り出す。短い行は空文字で位置を保持する（REQ-A1）。

    infer_batch_prepared はラベルとスコアを index で zip するため、短い行を
    `continue` で飛ばすと以降のタグがすべて別のスコアに結び付く。
    """
    return cells[index] if index < len(cells) else ""


def _is_blank_csv_row(cells: list[str]) -> bool:
    """空行（および空白だけの行）はデータ行ではないので読み飛ばす。

    末尾の空行を 1 レコードとして数えると、短い行の保持（REQ-A1）とは逆に
    ラベル数を増やしてずれの原因になる。
    """
    return not cells or all(not cell.strip() for cell in cells)


def _load_selected_tags_pixai6col(tags_path: Path) -> list[TagMeta]:
    labels: list[TagMeta] = []
    with tags_path.open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        try:
            next(reader)
        except StopIteration:
            return []
        for cells in reader:
            if _is_blank_csv_row(cells):
                continue
            tag_name = _cell(cells, 2)
            category = _resolve_category_keep_row(_cell(cells, 3), context="pixai6col")
            count_str = _cell(cells, 4)
            ips_json = _cell(cells, 5)
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
            if _is_blank_csv_row(cells):
                continue
            category = _resolve_category_keep_row(_cell(cells, 2), context="simple3col")
            labels.append(TagMeta(name=_cell(cells, 1), category=category, count=None, ips=()))
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
            if _is_blank_csv_row(cells):
                continue
            category = _resolve_category_keep_row(_cell(cells, 2), context="wd4col")
            count_str = _cell(cells, 3)
            try:
                count = int(count_str) if count_str else None
            except ValueError:
                count = None
            labels.append(TagMeta(name=_cell(cells, 1), category=category, count=count, ips=()))
    return labels

class _IndexPairsDict(dict):
    """`json.load(object_pairs_hook=...)` の結果。重複キーを記録しておく。

    json は重複キーを後勝ちで潰すため、parse 後の dict からは「重複していた」ことを
    検出できない。インデックスの重複はタグとスコアの対応ずれに直結するので、
    parse 時点で記録して REQ-A2 の検証へ渡す。
    """

    duplicate_keys: tuple[str, ...] = ()


def _index_pairs_hook(pairs: list[tuple[str, Any]]) -> _IndexPairsDict:
    result = _IndexPairsDict()
    duplicates: list[str] = []
    for key, value in pairs:
        if key in result:
            duplicates.append(key)
        result[key] = value
    result.duplicate_keys = tuple(duplicates)
    return result


def _load_index_json(tags_path: Path) -> Any:
    """重複キーを記録しつつ JSON を読む（REQ-A2）。"""
    with tags_path.open(encoding="utf-8") as fp:
        return json.load(fp, object_pairs_hook=_index_pairs_hook)


def _as_index_key(key: Any) -> int | None:
    try:
        return int(key)
    except (TypeError, ValueError):
        return None


def _validated_numeric_keys(mapping: Any, *, context: str, tags_path: Path) -> list[str]:
    """インデックス写像の連続性を検証し、数値キーを昇順で返す（REQ-A2）。

    欠け・重複・非連続は「タグとスコアの対応が黙ってずれる」ため、詰めて読まずに
    例外へ倒す（PixAI v1 のローダと同じ振る舞い）。
    """
    if not isinstance(mapping, dict):
        raise ValueError(
            f"load_selected_tags[{context}]: expected an object, got "
            f"{type(mapping).__name__} in {tags_path}"
        )
    numeric = _numeric_keys_sorted(mapping)
    duplicates = sorted({
        index for key in getattr(mapping, "duplicate_keys", ())
        if (index := _as_index_key(key)) is not None
    })
    indices = [int(key) for key in numeric]
    if duplicates or indices != list(range(len(indices))):
        problems: list[str] = []
        if duplicates:
            problems.append(f"duplicate indices {duplicates}")
        if indices:
            missing = sorted(set(range(indices[-1] + 1)) - set(indices))
            if missing:
                problems.append(f"missing indices {missing}")
        detail = "; ".join(problems) or f"non-contiguous indices {indices}"
        raise ValueError(f"load_selected_tags[{context}]: {detail} in {tags_path}")
    return numeric


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
    raw: Any = _load_index_json(tags_path)
    labels: list[TagMeta] = []

    if isinstance(raw, dict) and "idx_to_tag" in raw:
        idx_to_tag: dict[str, Any] = raw.get("idx_to_tag", {}) or {}
        tag_to_category: dict[str, Any] = raw.get("tag_to_category", {}) or {}
        idx_to_category: dict[str, Any] = raw.get("idx_to_category", {}) or {}
        for idx in _validated_numeric_keys(idx_to_tag, context="idx_json/v2", tags_path=tags_path):
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
    for idx in _validated_numeric_keys(raw, context="idx_json", tags_path=tags_path):
        entry = raw[idx] if isinstance(raw[idx], dict) else {}
        category = _resolve_category_keep_row(str(entry.get("category", "")), context="idx_json")
        labels.append(TagMeta(name=str(entry.get("tag", "")), category=category, count=None, ips=()))
    return labels


def _load_pixai_v1_tags_json(tags_path: Path) -> list[TagMeta]:
    """PixAI Tagger v1.0 config.json: ordered `tags` plus category/count pairs in
    `tags_split`. Keeping this as a dedicated format avoids interpreting the model's
    architecture config as a generic index map.
    """
    with tags_path.open(encoding="utf-8") as fp:
        raw: Any = json.load(fp)

    if not isinstance(raw, dict) or not isinstance(raw.get("tags"), list) or not isinstance(raw.get("tags_split"), list):
        raise ValueError(f"Invalid PixAI v1.0 tag metadata: {tags_path}")

    tags: list[Any] = raw["tags"]
    splits: list[Any] = raw["tags_split"]
    labels: list[TagMeta] = []
    offset = 0
    for split in splits:
        if (not isinstance(split, list) or len(split) != 2 or
                not isinstance(split[0], str) or isinstance(split[1], bool) or
                not isinstance(split[1], int) or split[1] < 0):
            raise ValueError(f"Invalid PixAI v1.0 category split in {tags_path}: {split!r}")
        category_name, count = split
        end = offset + count
        if end > len(tags):
            raise ValueError(f"PixAI v1.0 tag split exceeds tag list in {tags_path}")
        category = _resolve_category_keep_row(category_name, context="pixai_v1_config")
        labels.extend(TagMeta(name=str(name), category=category) for name in tags[offset:end])
        offset = end

    if offset != len(tags):
        raise ValueError(f"PixAI v1.0 tag split covers {offset} of {len(tags)} tags in {tags_path}")
    return labels

def _load_selected_tags_camie_metadata_json(tags_path: Path) -> list[TagMeta]:
    """camie-tagger-v2: camie-tagger-v2-metadata.json -> dataset_info.tag_mapping.{idx_to_tag,tag_to_category}."""
    raw: dict[str, Any] = _load_index_json(tags_path)
    tag_mapping = config_mapping(raw, "dataset_info", "tag_mapping")
    idx_to_tag: dict[str, str] = tag_mapping.get("idx_to_tag", {})
    tag_to_category: dict[str, str] = tag_mapping.get("tag_to_category", {})
    labels: list[TagMeta] = []
    for idx in _validated_numeric_keys(idx_to_tag, context="camie_metadata_json", tags_path=tags_path):
        tag_name = idx_to_tag[idx]
        category = _resolve_category_keep_row(str(tag_to_category.get(tag_name, "")), context="camie_metadata_json")
        labels.append(TagMeta(name=tag_name, category=category, count=None, ips=()))
    return labels

_TAGS_CSV_LOADERS: dict[str, Callable[[Path], list[TagMeta]]] = {
    "pixai6col": _load_selected_tags_pixai6col,
    "simple3col": _load_selected_tags_simple3col,
    "wd4col": _load_selected_tags_wd4col,
    "idx_json": _load_selected_tags_idx_json,
    "pixai_v1_config": _load_pixai_v1_tags_json,
    "camie_metadata_json": _load_selected_tags_camie_metadata_json,
}

# CSV 形式のタグファイルが持つ列数。DL 側（workers）の「取得済みファイルが完全か」の
# 判定に使う（REQ-E3）。ローダーと同じ表をここに置き、列数を二重管理しない。
# 列数が変わるとタグとスコアの対応も変わるので、形式を足すときは両方を確認すること。
TAGS_CSV_COLUMN_COUNTS: dict[str, int] = {
    "pixai6col": 6,
    "simple3col": 3,
    "wd4col": 4,
}

def load_selected_tags(tags_csv: str | Path, csv_format: TagsCsvFormat = "pixai6col") -> list[TagMeta]:
    tags_path = Path(tags_csv)
    if not tags_path.is_file():
        log_dbg(_get_string("TaggerCore", "Tag_CSV_File_Not_Found", tags_csv=str(tags_csv)))
        raise FileNotFoundError(_get_string("TaggerCore", "Tag_CSV_File_Not_Found", tags_csv=str(tags_csv)))
    loader = _TAGS_CSV_LOADERS.get(csv_format)
    if loader is None:
        # REQ-A3(a): 未知の csv_format を PixAI 6列として黙って読むと、別形式の
        # ファイルでタグとスコアの対応がずれる。fallback せず失敗させる。
        log_dbg(f"load_selected_tags: unknown csv_format {csv_format!r} for {tags_path}")
        raise ValueError(f"Unsupported tags csv_format: {csv_format!r}")
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


def resampling_filter(name: str, default: str = "lanczos") -> "Image.Resampling":
    """設定値のリサンプリング方式名を PIL のフィルタへ変換する。

    未知の名前は `default` に倒す（従来どおり黙って LANCZOS / BILINEAR で動く）。
    タガーの `inference.resampling` と Florence の preprocessor_config.json の
    `resample` が同じ表を使うように、変換をここへ一本化する。
    """
    filters = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "box": Image.Resampling.BOX,
        "hamming": Image.Resampling.HAMMING,
    }
    key = str(name).strip().lower()
    if key in filters:
        return filters[key]
    return filters.get(str(default).strip().lower(), Image.Resampling.LANCZOS)


def flatten_alpha_to_rgb(image: "Image.Image", background_rgb: tuple[int, int, int]) -> "Image.Image":
    """透過を指定色で平坦化して RGB 画像を返す（REQ-A9）。

    EXIF Orientation の適用は呼び出し側の責務（読み込み時 or 前処理時）。
    alpha を持たない画像は従来どおり `convert("RGB")` だけを行う。
    """
    has_alpha = "A" in image.getbands() or "transparency" in image.info
    if not has_alpha:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*background_rgb, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")

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
    resampling: str = "lanczos"       # "nearest" | "bilinear" | "bicubic" | "lanczos"
    alpha_background_rgb: tuple[int, int, int] | None = DEFAULT_ALPHA_BACKGROUND_RGB

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
    # REQ-1.2: 後処理をベクトル化するためのカテゴリ ID 列（TagMeta.category の int 配列）。
    # self.tags は tag_meta_lookup 等の既存参照のために残す。タグ名の配列は作らない
    # （並べ替えで同点が出た行だけタグ名を引けば足りるため。NFR-4）。
    _tag_category_ids: NDArray[np.intp]
    _category_slots: int
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
        # REQ-1.2: 後処理のフィルタ・上限判定を NumPy で回すための事前構築。
        self._build_tag_arrays()
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
        resampling = resampling_filter(cfg.resampling)

        for img_array in images:
            image_pil = Image.fromarray(img_array)
            w, h = image_pil.size
            ratio = min(target_size / w, target_size / h)
            # REQ-A6: int() 切り捨てだと極端なアスペクト比（長辺の 1/448 未満）で
            # 短辺が 0 になり PIL が例外を出す。VLM 側と同じ round()＋最小1で揃える。
            new_w = max(1, round(w * ratio))
            new_h = max(1, round(h * ratio))
            resized_image = image_pil.resize((new_w, new_h), resampling)
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
        cat_thresholds: Mapping[TagCategory, float] = thresholds if thresholds is not None else {}
        cat_limits: Mapping[TagCategory, int] = max_tags if max_tags is not None else {}

        if not self.tags:
            return [TagResult() for _ in scores_batch]
        hard_cap = (sum(cat_limits.values()) if cat_limits
                    else DEFAULT_TAG_OUTPUT_CAP)
        score_floor = 1e-4
        # REQ-1.3: 閾値は呼び出しごとに変わる引数なので、行ループの外で 1 回だけ
        # カテゴリ ID → 実効閾値のベクトルに落とす。
        threshold_by_category = self._threshold_vector(cat_thresholds)
        limit_by_category = self._limit_vector(cat_limits)
        return [
            TagResult(tags=self._predictions_for_row(
                scores,
                threshold_by_category,
                limit_by_category,
                hard_cap,
                score_floor,
            ))
            for scores in scores_batch
        ]

    def _build_tag_arrays(self) -> None:
        """タグメタデータから後処理用の NumPy 配列を構築する（REQ-1.2）。

        `self.tags` を読み終えた直後に 1 回だけ呼ぶ。13k〜31k 要素の int 配列 1 本だけで、
        タグ数に比例する追加コストは最小限に留める（NFR-4。タグ名の配列は作らない）。
        テストなどで `__init__` を通さずにタグを組み立てる場合も、これを呼べば揃う。
        """
        self._tag_category_ids = np.fromiter(
            (int(tag.category) for tag in self.tags), dtype=np.intp, count=len(self.tags)
        )
        # カテゴリ ID → 閾値／上限の配列長。全メンバーぶんを確保し、未知の ID が
        # 混ざっていても（あり得ないが）インデックスがずれないよう最大値まで広げる。
        max_category_id = int(self._tag_category_ids.max()) + 1 if self.tags else 0
        self._category_slots = max(len(TagCategory), max_category_id)

    def _threshold_vector(self, thresholds: Mapping[TagCategory, float]) -> NDArray[np.float64]:
        """カテゴリ ID → その呼び出しでの実効閾値（float64）を作る。

        フォールバック規則は改修前の `cat_thresholds.get(category,
        cat_thresholds.get(GENERAL, 0.0))` と同一: 指定の無いカテゴリは GENERAL の
        閾値、GENERAL も無ければ 0.0。dtype を float64 にするのは比較の等価性のためで、
        float32 に落とすと Python float の閾値と約 1e-8 ずれ、閾値近傍で採否が反転し得る。
        """
        general = float(thresholds.get(TagCategory.GENERAL, 0.0))
        vector = np.full(self._category_slots, general, dtype=np.float64)
        for category_key, value in thresholds.items():
            slot = _category_slot(category_key)
            if slot is not None and 0 <= slot < vector.size:
                # 改修前は辞書引きで「使われないキー」を無視していたので、範囲外・
                # 解釈できないキーも無視して同じ結果にする。
                vector[slot] = float(value)
        return vector

    def _limit_vector(self, limits: Mapping[TagCategory, int]) -> NDArray[np.float64]:
        """カテゴリ ID → カテゴリ別上限を作る（NaN = 上限なし）。

        改修前は `cat_limits.get(category)` が None のカテゴリを無制限として扱っていた。
        判定を `current >= limit` の 1 比較に統一できるよう、無制限は NaN で表す
        （NaN との比較は常に False になるので「上限に達していない」＝無制限になる）。
        """
        vector = np.full(self._category_slots, np.nan, dtype=np.float64)
        for category_key, value in limits.items():
            slot = _category_slot(category_key)
            if slot is not None and 0 <= slot < vector.size:
                vector[slot] = float(value)
        return vector

    def _predictions_for_row(
        self,
        scores: NDArray[np.float_],
        threshold_by_category: NDArray[np.float64],
        limit_by_category: NDArray[np.float64],
        hard_cap: int,
        score_floor: float,
    ) -> list[TagPrediction]:
        """1 画像ぶんのスコア行から採用タグを選ぶ（REQ-1.3〜1.5, REQ-1.9）。

        改修前（9e21884 の `infer_batch_prepared` 後段）と同一のセマンティクスを保つ:

        - スコア床・カテゴリ別閾値とも「未満が棄却」（ちょうどは残る）。
        - 並びはスコア降順 → タグ名昇順 → 元のタグ順（安定）。
        - カテゴリ別上限に達したものは飛ばしつつ全体上限 hard_cap で打ち切る。
        - zip と同じく min(len(tags), len(scores)) で打ち切る。

        走査と棄却だけを NumPy で行い、TagPrediction の生成は採用された分に限る。
        """
        count = min(self._tag_category_ids.size, int(scores.size))
        if count == 0 or hard_cap <= 0:
            return []
        row = scores[:count]
        category_ids = self._tag_category_ids[:count]

        # NaN は「床未満」「閾値未満」のどちらも False になるため改修前は通過していた。
        # 通過させる点までは再現できるが、NaN を含むキーの並びは Python の sorted() と
        # NumPy で一致しない（§6 U-1 / REQ-1.10 の判断対象）。勝手にどちらかへ倒さず、
        # NaN を含む行だけ改修前の実装そのものへ委譲して挙動を固定する。
        if np.isnan(row).any():
            return self._predictions_for_row_reference(
                row, threshold_by_category, limit_by_category, hard_cap, score_floor
            )

        keep = ~((row < score_floor) | (row < threshold_by_category[category_ids]))
        candidate_index = np.flatnonzero(keep)
        if candidate_index.size == 0:
            return []
        candidate_scores = row[candidate_index]
        full_index, full_scores = candidate_index, candidate_scores

        # 全体上限まで届く見込みがあるときは、先頭部分（真の並びの接頭辞）だけを確定
        # させる。同点が多いモデルでは候補が数千件になるが、必要なのは先頭 hard_cap 件
        # だけなので、これで並べ替えの対象を減らせる（§1.3 の計測に基づく）。
        # 接頭辞が埋め切れなかった場合は全候補でやり直すので、結果は常に同じになる。
        truncated = candidate_index.size > max(hard_cap * 4, 256)
        if truncated:
            top = np.argpartition(-candidate_scores, hard_cap - 1)[:hard_cap]
            # 境界と同点の候補はどちらが先か決まらないので全部含める。こうすると
            # 選んだ集合は必ず「真の並びの接頭辞」になる（外側はすべてスコアが低い）。
            boundary = float(candidate_scores[top].min())
            prefix_mask = candidate_scores >= boundary
            candidate_index, candidate_scores = self._sort_candidates(
                candidate_index[prefix_mask], candidate_scores[prefix_mask]
            )
        else:
            candidate_index, candidate_scores = self._sort_candidates(candidate_index, candidate_scores)

        taken = self._take_within_caps(
            candidate_index, candidate_scores, category_ids, limit_by_category, hard_cap
        )
        if not truncated or len(taken) >= hard_cap:
            return taken

        # 上限を埋め切れなかった（上限に達していないカテゴリの候補が接頭辞の外にあった）
        # → 全候補でやり直す。ここに来るのはカテゴリ別上限が厳しく効いている行だけ。
        full_index, full_scores = self._sort_candidates(full_index, full_scores)
        return self._take_within_caps(
            full_index, full_scores, category_ids, limit_by_category, hard_cap
        )

    def _sort_candidates(
        self,
        candidate_index: NDArray[np.intp],
        candidate_scores: NDArray[np.float_],
    ) -> tuple[NDArray[np.intp], NDArray[np.float_]]:
        """候補を「スコア降順 → タグ名昇順 → 元のタグ順」に並べる（REQ-1.4）。"""
        # スコア降順。同点は kind="stable" で元のタグ順が保たれる。
        order = np.argsort(-candidate_scores, kind="stable")
        candidate_index = candidate_index[order]
        candidate_scores = candidate_scores[order]
        if candidate_scores.size > 1 and np.any(candidate_scores[1:] == candidate_scores[:-1]):
            # 同点があるときだけタグ名昇順を足す（同点が無ければ argsort で確定）。
            # 第 3 キーの candidate_index が一意なので並びは完全に決まる。
            names = np.array([self.tags[int(tag_index)].name for tag_index in candidate_index])
            tie_order = np.lexsort((candidate_index, names, -candidate_scores))
            candidate_index = candidate_index[tie_order]
            candidate_scores = candidate_scores[tie_order]
        return candidate_index, candidate_scores

    def _take_within_caps(
        self,
        candidate_index: NDArray[np.intp],
        candidate_scores: NDArray[np.float_],
        category_ids: NDArray[np.intp],
        limit_by_category: NDArray[np.float64],
        hard_cap: int,
    ) -> list[TagPrediction]:
        """並び順に走査し、カテゴリ別上限を飛ばしつつ全体上限で打ち切る（§1.1-6）。"""
        taken: list[TagPrediction] = []
        per_category: dict[int, int] = {}
        for position in range(candidate_index.size):
            if len(taken) >= hard_cap:
                break
            tag_index = int(candidate_index[position])
            category_id = int(category_ids[tag_index])
            used = per_category.get(category_id, 0)
            # 無制限は NaN なので比較は False になり、上限判定が効かない。
            if used >= limit_by_category[category_id]:
                continue
            per_category[category_id] = used + 1
            taken.append(TagPrediction(
                name=self.tags[tag_index].name,
                score=float(candidate_scores[position]),
                category=TagCategory(category_id),
            ))
        return taken

    def _predictions_for_row_reference(
        self,
        scores: NDArray[np.float_],
        threshold_by_category: NDArray[np.float64],
        limit_by_category: NDArray[np.float64],
        hard_cap: int,
        score_floor: float,
    ) -> list[TagPrediction]:
        """改修前（9e21884）の後処理そのもの。NaN を含む行のためだけに残す。

        `sorted()` は NaN キーに対して「比較が常に False」という定義外の並びを作る。
        その並びは NumPy のソートと一致しないため、NaN を含む行ではこの実装へ委譲し、
        改修前後で観測可能な出力が変わる余地を残さない（REQ-1.10）。
        """
        raw_predictions: list[TagPrediction] = []
        for tag_meta, score in zip(self.tags, scores):
            if score < score_floor:
                continue
            category = TagCategory(tag_meta.category)
            if float(score) < threshold_by_category[int(category)]:
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
            current = per_category.get(category, 0)
            if current >= limit_by_category[int(category)]:
                continue
            per_category[category] = current + 1
            taken.append(prediction)
        return taken

    def infer_batch(self, images: Sequence[Image.Image], *, thresholds: Mapping[TagCategory, float] | None = None, max_tags: Mapping[TagCategory, int] | None = None) -> list[TagResult]:
        rgb_arrays: list[NDArray[np.uint8]] = []
        background_rgb = self.inference_config.alpha_background_rgb
        for image in images:
            # REQ-A9: 既定は白背景。マニフェストが明示的に null を書いた場合だけ
            # 従来の「alpha を捨てる」変換に倒す。
            if background_rgb is None:
                image = image.convert("RGB")
            else:
                image = flatten_alpha_to_rgb(image, background_rgb)
            rgb_arrays.append(np.asarray(image, dtype=np.uint8))
        batch, extra_inputs = self.prepare_batch_from_rgb_np(rgb_arrays)
        return self.infer_batch_prepared(batch, extra_inputs, thresholds=thresholds, max_tags=max_tags)

def get_image_paths_recursive(base_dir: Path) -> list[Path]:
    # 拡張子の定義は constants.IMAGE_EXTENSIONS が唯一。ここでローカルに
    # 再定義すると、対応拡張子を増やしたときに走査側だけ取り残される。
    image_paths: list[Path] = []
    for ext in constants.IMAGE_EXTENSIONS:
        image_paths.extend(base_dir.rglob(f"*{ext}"))
    return sorted(image_paths)

def format_tags(tag_results: TagResult, convert_underscore: bool) -> str:
    # Output order: artist, copyright, character, <series tags>, general, style, meta, rating.
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
    output_tags += _emit(by_category.get(TagCategory.STYLE, []))
    output_tags += _emit(by_category.get(TagCategory.META, []))
    output_tags += _emit(by_category.get(TagCategory.MODEL, []))
    output_tags += _emit(by_category.get(TagCategory.QUALITY, []))
    output_tags += _emit(by_category.get(TagCategory.YEAR, []))
    output_tags += _emit(by_category.get(TagCategory.RATING, []))
    return ", ".join(output_tags)

_INFERENCE_ACTIVATIONS = ("sigmoid", "none")
_INFERENCE_LAYOUTS = ("nchw", "nhwc")
_INFERENCE_CHANNEL_ORDERS = ("rgb", "bgr")
_NORMALIZE_VECTOR_LENGTH = 3


def _validated_choice(value: Any, allowed: tuple[str, ...], *, key: str) -> str:
    """model_config.json の列挙値を検証する（REQ-A3(b)）。

    activation のタイプミスで二重 sigmoid、未知 layout が nchw 扱い、未知 channel が
    rgb 扱いになるのを防ぐ。`build_captioner_config` の検証と同じ方針。
    """
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"inference.{key} must be one of {list(allowed)}, got {value!r}")
    return value


def _validated_normalize_vector(value: Any, *, key: str) -> tuple[float, float, float] | None:
    """正規化用の3要素ベクトルであることを検証する（REQ-A3(b)）。

    None は「このモデルは正規化しない」という明示指定として従来どおり許す。
    """
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"inference.{key} must be a list of 3 numbers, got {value!r}")
    if len(value) != _NORMALIZE_VECTOR_LENGTH:
        raise ValueError(
            f"inference.{key} must have exactly {_NORMALIZE_VECTOR_LENGTH} elements, got {len(value)}"
        )
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError(f"inference.{key} must contain only numbers, got {value!r}")
    return tuple(value)  # type: ignore[return-value]


def build_inference_config(model_config: dict[str, Any]) -> InferenceConfig:
    """
    Builds an InferenceConfig from a model_config.json dict's "inference"/"tags_csv"
    blocks. Any key missing from the dict falls back to InferenceConfig's own
    (PixAI-matching) default, so a partially-specified model_config.json is safe.
    """
    inference = config_mapping(model_config, "inference")
    tags_csv = config_mapping(model_config, "tags_csv")
    defaults = InferenceConfig()

    pad_color = inference.get("pad_color_rgb")
    alpha_background = inference.get("alpha_background_rgb", defaults.alpha_background_rgb)
    normalize_mean = inference.get("normalize_mean", defaults.normalize_mean)
    normalize_std = inference.get("normalize_std", defaults.normalize_std)
    extra_inputs = inference.get("extra_inputs", list(defaults.extra_inputs))

    return InferenceConfig(
        input_size=inference.get("input_size", defaults.input_size),
        layout=_validated_choice(inference.get("layout", defaults.layout), _INFERENCE_LAYOUTS, key="layout"),
        channel_order=_validated_choice(
            inference.get("channel_order", defaults.channel_order),
            _INFERENCE_CHANNEL_ORDERS,
            key="channel_order",
        ),
        rescale_to_unit=inference.get("rescale_to_unit", defaults.rescale_to_unit),
        pad_color_rgb=tuple(pad_color) if pad_color is not None else defaults.pad_color_rgb,  # type: ignore[arg-type]
        resampling=inference.get("resampling", defaults.resampling),
        alpha_background_rgb=tuple(alpha_background) if alpha_background is not None else None,  # type: ignore[arg-type]
        normalize_mean=_validated_normalize_vector(normalize_mean, key="normalize_mean"),
        normalize_std=_validated_normalize_vector(normalize_std, key="normalize_std"),
        extra_inputs=tuple(extra_inputs),
        output_name_hint=inference.get("output_name", defaults.output_name_hint),
        output_activation=_validated_choice(
            inference.get("output_activation", defaults.output_activation),
            _INFERENCE_ACTIVATIONS,
            key="output_activation",
        ),
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
            # REQ-A4: ONNX ファイル名の正本はマニフェストの network.files。
            # ここで "model.onnx" を二重管理しない。
            model_path = entry.model_dir / model_registry.primary_onnx_file_name(entry)
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

    `progress_cb(done, total)` は全体で最大 constants.PROGRESS_SIGNAL_BUDGET 回
    ＋最後の N/N（完了）だけ呼ぶ。
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
    # 全体で PROGRESS_SIGNAL_BUDGET 回に間引く（理由と天井除算の根拠は
    # constants.progress_step_for() を参照）。最後の1枚は必ず発行して
    # N/N（完了）に到達させる。
    progress_step = constants.progress_step_for(total)

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
            with Image.open(image_path) as opened_image:
                # REQ-A5: VLM 経路と同じく EXIF Orientation を適用してから推論する
                # （横向きのまま推論して黙って精度が落ちるのを防ぐ）。alpha は
                # exif_transpose 後も保持されるので、モデル側の合成にそのまま渡せる。
                image = ImageOps.exif_transpose(opened_image)
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
            # REQ-A8: 全カテゴリの上限が 0 のときは hard_cap=0 で結果が必ず空になる。
            # これを推論失敗として数えると「失敗分のみ」モードで同じ画像を無限に
            # 再試行してしまうため、上限0による空は「成功（タグ無し）」として扱う。
            # 判別はループ側が保持している設定値だけで行い、推論後処理の外部
            # シグネチャは変えない。
            limits = settings.get('MAX_TAGS_PER_CATEGORY') or {}
            limits_all_zero = bool(limits) and all(int(limit) == 0 for limit in limits.values())
            if not limits_all_zero:
                n_errors += 1
                mark_failed(image_path)
                log_dbg(_get_string_internal("TaggerCore", "Tag_Acquisition_Failed", current_index_str=current_index_str, relative_path=str(relative_path)))
                core_log_gui(_get_string_internal("TaggerCore", "Tag_Acquisition_Failed_Short", current_index_str=current_index_str, relative_path_name=relative_path.name), "orange")
                continue
            log_dbg(f"tags: all per-category limits are 0 for {relative_path}; treating the empty "
                    f"result as success (no tags) rather than a failure.")
            tag_result = results[0] if results else TagResult()
        else:
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
    # Everything else (rating / copyright / style / artist / meta / model / quality / year) is
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
