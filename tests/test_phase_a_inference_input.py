"""Phase A（レビューループ指摘の修正計画 §7）の回帰テスト。

対象: R1-2 / R3-F1 / R3-F2 / R3-F5 / R5-1 / R5-2 / R5-3 / R5-4 / R17-5 / R20-1
（REQ-A1〜A9）。実 models/ の重みファイルには触れず、同梱マニフェストの読み取りと
tmp_path のフィクスチャだけで検証する（計画 §15.2）。

Offline only - no network. Run:  pytest tests/test_phase_a_inference_input.py -q
"""
from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

import caption_core
import model_registry
import tagging_core

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


# --- helpers ----------------------------------------------------------------------

def _write_csv(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp).writerows(rows)
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _bundled_manifest_paths() -> list[Path]:
    return sorted(MODELS_DIR.glob("*/model_config.json"))


def _bundled_tagger_configs() -> list[tuple[Path, dict]]:
    configs = []
    for path in _bundled_manifest_paths():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("model_type", "tagger") == "tagger":
            configs.append((path, cfg))
    return configs


class _FakeTagger:
    """process_image_loop が使う最小のタガー。推論はせず入力画像を記録する。"""

    def __init__(self, tag_result: tagging_core.TagResult | None = None):
        self.tag_meta_lookup: dict[str, tagging_core.TagMeta] = {}
        self.seen: list[Image.Image] = []
        self._tag_result = tag_result

    def infer_batch(self, images, *, thresholds=None, max_tags=None):
        results = []
        for image in images:
            self.seen.append(image)
            if self._tag_result is None:
                results.append(tagging_core.TagResult(tags=[
                    tagging_core.TagPrediction("sample_tag", 0.9, tagging_core.TagCategory.GENERAL)
                ]))
            else:
                results.append(tagging_core.TagResult(tags=list(self._tag_result.tags)))
        return results


def _loop_settings(base_dir: Path, **overrides) -> dict:
    settings = {
        "INPUT_DIR": base_dir,
        "TAG_THRESHOLDS": {},
        "MAX_TAGS_PER_CATEGORY": {category: 5 for category in tagging_core.TagCategory},
        "ENABLE_SOLO_LIMIT": False,
        "CONVERT_UNDERSCORE": False,
        "EXISTING_FILE_MODE": tagging_core.ExistingFileMode.OVERWRITE,
    }
    settings.update(overrides)
    return settings


def _run_loop(tmp_path: Path, tagger: _FakeTagger, image_names: list[str], **overrides):
    image_paths = []
    for name in image_names:
        path = tmp_path / name
        Image.new("RGB", (32, 24), (10, 20, 30)).save(path)
        image_paths.append(path)
    failed: list[Path] = []
    changed = tagging_core.process_image_loop(
        tagger, _loop_settings(tmp_path, **overrides), image_paths,
        None, None, None, None, failed_paths=failed,
    )
    return changed, failed


# --- REQ-A1: CSV ローダの行保持 -----------------------------------------------------

def test_req_a1_pixai6col_keeps_short_rows_and_index_alignment(tmp_path):
    path = _write_csv(tmp_path / "tags.csv", [
        ["id", "tag_id", "name", "category", "count", "ips"],
        ["0", "100", "alpha", "general", "10", ""],
        ["1", "101", "short_tag", "character"],           # 6列未満
        ["2", "102", "beta", "general", "30", ""],
        [],                                                 # 末尾の空行はデータ行ではない
    ])
    tags = tagging_core.load_selected_tags(path, "pixai6col")
    # 短い行を飛ばすと beta が short_tag のスコアに結び付いてしまう。
    assert [tag.name for tag in tags] == ["alpha", "short_tag", "beta"]
    assert [tag.category for tag in tags] == [
        tagging_core.TagCategory.GENERAL,
        tagging_core.TagCategory.CHARACTER,
        tagging_core.TagCategory.GENERAL,
    ]
    assert tags[1].count is None


def test_req_a1_simple3col_keeps_short_rows(tmp_path):
    path = _write_csv(tmp_path / "tags.csv", [
        ["tag_id", "name", "category"],
        ["0", "alpha", "general"],
        ["1", "short_tag"],                                 # 3列未満
        ["2", "beta", "artist"],
    ])
    tags = tagging_core.load_selected_tags(path, "simple3col")
    assert [tag.name for tag in tags] == ["alpha", "short_tag", "beta"]
    assert tags[1].category == tagging_core.TagCategory.GENERAL   # 行は保持し既定カテゴリへ
    assert tags[2].category == tagging_core.TagCategory.ARTIST


def test_req_a1_wd4col_keeps_short_rows(tmp_path):
    path = _write_csv(tmp_path / "tags.csv", [
        ["tag_id", "name", "category", "count"],
        ["0", "alpha", "general", "10"],
        ["1", "short_tag", "character"],                    # 4列未満
        ["2", "beta", "general", "30"],
    ])
    tags = tagging_core.load_selected_tags(path, "wd4col")
    assert [tag.name for tag in tags] == ["alpha", "short_tag", "beta"]
    assert tags[1].count is None
    assert tags[2].count == 30


# --- REQ-A2: インデックスローダの連続性検証 -----------------------------------------

def test_req_a2_idx_json_v1_gap_fails_instead_of_shifting(tmp_path):
    path = _write_json(tmp_path / "tag_mapping.json", {
        "0": {"tag": "alpha", "category": "General"},
        "2": {"tag": "beta", "category": "General"},        # 1 が欠けている
    })
    with pytest.raises(ValueError, match="missing indices"):
        tagging_core.load_selected_tags(path, "idx_json")


def test_req_a2_idx_json_v2_gap_fails(tmp_path):
    path = _write_json(tmp_path / "model_vocabulary.json", {
        "version": 2,
        "idx_to_tag": {"0": "alpha", "1": "beta", "3": "gamma"},
        "tag_to_category": {"alpha": "General", "beta": "General"},
    })
    with pytest.raises(ValueError, match="missing indices"):
        tagging_core.load_selected_tags(path, "idx_json")


def test_req_a2_duplicate_index_keys_fail(tmp_path):
    # json.load は重複キーを後勝ちで潰すため、parse 時点の記録で検出する。
    path = tmp_path / "tag_mapping.json"
    path.write_text(
        '{"0": {"tag": "alpha"}, "0": {"tag": "beta"}, "1": {"tag": "gamma"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate indices"):
        tagging_core.load_selected_tags(path, "idx_json")


def test_req_a2_camie_metadata_gap_fails(tmp_path):
    path = _write_json(tmp_path / "camie.json", {
        "dataset_info": {"tag_mapping": {
            "idx_to_tag": {"0": "alpha", "2": "beta"},
            "tag_to_category": {"alpha": "General"},
        }},
    })
    with pytest.raises(ValueError, match="missing indices"):
        tagging_core.load_selected_tags(path, "camie_metadata_json")


def test_req_a2_contiguous_index_maps_still_load(tmp_path):
    v1 = _write_json(tmp_path / "tag_mapping.json", {
        "version": 1,
        "0": {"tag": "alpha", "category": "General"},
        "1": {"tag": "beta", "category": "Character"},
    })
    assert [tag.name for tag in tagging_core.load_selected_tags(v1, "idx_json")] == ["alpha", "beta"]

    v2 = _write_json(tmp_path / "model_vocabulary.json", {
        "version": 2,
        "idx_to_tag": {"0": "alpha", "1": "beta"},
        "tag_to_category": {"alpha": "General", "beta": "Artist"},
    })
    tags = tagging_core.load_selected_tags(v2, "idx_json")
    assert [tag.name for tag in tags] == ["alpha", "beta"]
    assert tags[1].category == tagging_core.TagCategory.ARTIST

    camie = _write_json(tmp_path / "camie.json", {
        "dataset_info": {"tag_mapping": {
            "idx_to_tag": {"0": "alpha", "1": "beta"},
            "tag_to_category": {"alpha": "General", "beta": "General"},
        }},
    })
    assert [tag.name for tag in tagging_core.load_selected_tags(camie, "camie_metadata_json")] == [
        "alpha", "beta",
    ]


# --- REQ-A3: 値検証 -----------------------------------------------------------------

def test_req_a3_unknown_csv_format_fails_without_fallback(tmp_path):
    path = _write_csv(tmp_path / "tags.csv", [
        ["tag_id", "name", "category"],
        ["0", "alpha", "general"],
    ])
    with pytest.raises(ValueError, match="Unsupported tags csv_format"):
        tagging_core.load_selected_tags(path, "wd14-someday")
    # 既知の形式では従来どおり読める。
    assert [tag.name for tag in tagging_core.load_selected_tags(path, "simple3col")] == ["alpha"]


@pytest.mark.parametrize("inference, match", [
    ({"layout": "nchw2"}, "inference.layout"),
    ({"channel_order": "rgba"}, "inference.channel_order"),
    ({"output_activation": "softmax"}, "inference.output_activation"),
    ({"normalize_mean": [0.5, 0.5]}, "normalize_mean"),
    ({"normalize_std": [0.5, 0.5, 0.5, 0.5]}, "normalize_std"),
])
def test_req_a3_build_inference_config_rejects_invalid_values(inference, match):
    with pytest.raises(ValueError, match=match):
        tagging_core.build_inference_config({"inference": inference})


def test_req_a3_bundled_manifests_all_build(tmp_path):
    """同梱10モデル相当（マニフェスト9件＋PixAI v0.9 疑似エントリ）が読めること。"""
    configs = _bundled_tagger_configs()
    assert configs, "同梱マニフェストが見つからない"
    for path, cfg in configs:
        built = tagging_core.build_inference_config(cfg)
        assert built.layout in ("nchw", "nhwc"), path
        assert built.channel_order in ("rgb", "bgr"), path
        assert built.output_activation in ("sigmoid", "none"), path
        assert built.normalize_mean is None or len(built.normalize_mean) == 3, path
        assert built.normalize_std is None or len(built.normalize_std) == 3, path

    # captioner（Florence）は専用のビルダーで検証する。実 models/ の
    # preprocessor_config.json を読まないよう、model_dir は空の tmp_path を渡す。
    for path in _bundled_manifest_paths():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("model_type") == "captioner":
            built = caption_core.build_captioner_config(tmp_path, cfg)
            assert built.image_size > 0, path

    # PixAI v0.9 はディスク上の model_config.json を持たない疑似エントリだが、
    # 同じ検証を通ること（同梱10モデル ＝ マニフェスト9件 ＋ 疑似エントリ1件）。
    assert tagging_core.build_inference_config(model_registry._PIXAI_CONFIG).layout == "nchw"
    assert len(_bundled_manifest_paths()) == 9
    assert len(configs) + 2 == 10


# --- REQ-A4: ONNX ファイル名の正本 ------------------------------------------------

def test_req_a4_primary_onnx_file_name_comes_from_the_manifest():
    for path, cfg in _bundled_tagger_configs():
        entry = model_registry.ModelEntry(
            model_id=cfg["model_id"],
            model_name=cfg.get("model_name", cfg["model_id"]),
            model_type="tagger",
            model_dir=path.parent,
            config=cfg,
        )
        name = model_registry.primary_onnx_file_name(entry)
        assert name in cfg["network"]["files"], path
        # 従来ハードコードしていた名前と一致すること（挙動不変）。
        assert name == "model.onnx", path
    assert model_registry.primary_onnx_file_name(model_registry._pixai_entry()) == "model.onnx"


def test_req_a4_no_hardcoded_model_onnx_in_tagger_setup():
    source = (ROOT / "src" / "tagging_core.py").read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert 'entry.model_dir / "model.onnx"' not in code


# --- REQ-A5 / A9: EXIF と alpha（意図的変更） --------------------------------------

def _image_with_orientation(path: Path, size=(40, 20), orientation: int = 6) -> Image.Image:
    image = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    exif[0x0112] = orientation
    image.save(path, exif=exif)
    return image


def test_req_a5_tagger_applies_exif_orientation(tmp_path):
    image_path = tmp_path / "oriented.jpg"
    _image_with_orientation(image_path, size=(40, 20), orientation=6)  # 90度回転して表示
    tagger = _FakeTagger()
    tagging_core.process_image_loop(
        tagger, _loop_settings(tmp_path), [image_path], None, None, None, None,
    )
    assert len(tagger.seen) == 1
    assert tagger.seen[0].size == (20, 40)


def test_req_a5_taggers_without_exif_keep_pixels(tmp_path):
    image_path = tmp_path / "plain.png"
    original = Image.new("RGB", (17, 13), (1, 2, 3))
    original.putpixel((0, 0), (250, 10, 20))
    original.save(image_path)
    tagger = _FakeTagger()
    tagging_core.process_image_loop(
        tagger, _loop_settings(tmp_path), [image_path], None, None, None, None,
    )
    assert tagger.seen[0].tobytes() == original.tobytes()


def test_req_a9_default_alpha_background_is_white():
    assert tagging_core.InferenceConfig().alpha_background_rgb == (255, 255, 255)


def test_req_a9_tagger_flattens_alpha_onto_the_default_background():
    source = Image.new("RGBA", (1, 1), (255, 0, 0, 128))
    flattened = tagging_core.flatten_alpha_to_rgb(source, tagging_core.DEFAULT_ALPHA_BACKGROUND_RGB)
    assert flattened.mode == "RGB"
    red, green, blue = flattened.getpixel((0, 0))
    assert red == 255
    assert abs(green - 127) <= 1
    assert abs(blue - 127) <= 1


def test_req_a9_tagger_loop_keeps_alpha_until_inference(tmp_path):
    image_path = tmp_path / "transparent.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 0)).save(image_path)
    tagger = _FakeTagger()
    tagging_core.process_image_loop(
        tagger, _loop_settings(tmp_path), [image_path], None, None, None, None,
    )
    assert tagger.seen[0].mode == "RGBA"


def test_req_a9_florence_flattens_alpha_and_applies_exif(tmp_path):
    captioner = object.__new__(caption_core.Florence2Captioner)
    captioner.config = caption_core.CaptionerConfig(model_dir=tmp_path, image_size=8)

    transparent = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    arr = captioner._preprocess_image(transparent)
    # 白へ平坦化 → normalize_mean/std で (1 - 0.485) / 0.229。
    expected = (255 / 255 - 0.485) / 0.229
    assert arr.shape == (1, 3, 8, 8)
    assert arr[0, 0, 0, 0] == pytest.approx(expected, abs=1e-5)

    oriented_path = tmp_path / "oriented.png"
    oriented = Image.new("RGB", (4, 2), (255, 0, 0))
    for x in range(2, 4):
        for y in range(2):
            oriented.putpixel((x, y), (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6
    oriented.save(oriented_path, exif=exif)
    with Image.open(oriented_path) as opened:
        loaded = opened.copy()
    got = captioner._preprocess_image(loaded)
    # exif_transpose を適用した画像の前処理結果と一致し、未適用のものとは異なる。
    expected_arr = captioner._preprocess_image(ImageOps.exif_transpose(loaded))
    assert np.allclose(got, expected_arr)
    no_exif = loaded.copy()
    no_exif.info.pop("exif", None)
    assert not np.allclose(got, captioner._preprocess_image(no_exif))


# --- REQ-A6: リサイズのクランプ -----------------------------------------------------

def _prepare(images: list[np.ndarray], input_size: int = 448):
    fake_self = types.SimpleNamespace(
        inference_config=tagging_core.InferenceConfig(input_size=input_size),
    )
    return tagging_core.OnnxTagger.prepare_batch_from_rgb_np(fake_self, images)


def test_req_a6_extreme_aspect_ratio_does_not_collapse_to_zero():
    extreme = np.zeros((1, 4480, 3), dtype=np.uint8)  # 高さ1px・幅4480px
    batch, _ = _prepare([extreme])
    assert batch.shape == (1, 3, 448, 448)


def test_req_a6_normal_sizes_keep_the_previous_dimensions():
    # 拡大率が整数になるサイズでは round() と従来の int() が一致する（挙動不変）。
    for width, height in [(896, 448), (448, 224), (672, 336), (300, 300)]:
        ratio = min(448 / width, 448 / height)
        expected = (int(width * ratio), int(height * ratio))
        mask_cfg = types.SimpleNamespace(
            inference_config=tagging_core.InferenceConfig(input_size=448, extra_inputs=("padding_mask",)),
        )
        _, extra = tagging_core.OnnxTagger.prepare_batch_from_rgb_np(
            mask_cfg, [np.zeros((height, width, 3), dtype=np.uint8)],
        )
        filled = ~extra["padding_mask"][0]
        got = (int(filled.any(axis=0).sum()), int(filled.any(axis=1).sum()))
        assert got == expected, (width, height)


# --- REQ-A7: preprocessor_config.json と max_new_tokens -----------------------------

def test_req_a7_preprocessor_config_drives_resampling(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"resample": 3}), encoding="utf-8")
    config = caption_core.build_captioner_config(tmp_path, {})
    assert config.resampling == "bicubic"
    assert config.max_new_tokens == 200  # 従来の captioner 既定のまま


def test_req_a7_missing_preprocessor_config_falls_back_to_bilinear(tmp_path):
    config = caption_core.build_captioner_config(tmp_path, {})
    assert config.resampling == "bilinear"


@pytest.mark.parametrize("value, expected", [
    (2, "bilinear"),
    ("BICUBIC", "bicubic"),
    ("PILImageResampling.LANCZOS", "lanczos"),
    (99, "bilinear"),
    ("nearest-neighbour", "bilinear"),
])
def test_req_a7_resample_values(tmp_path, value, expected):
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"resample": value}), encoding="utf-8")
    assert caption_core.build_captioner_config(tmp_path, {}).resampling == expected


def test_req_a7_max_new_tokens_still_comes_from_model_config(tmp_path):
    config = caption_core.build_captioner_config(tmp_path, {"captioner": {"max_new_tokens": 321}})
    assert config.max_new_tokens == 321


# --- REQ-A8: 全カテゴリ上限0の扱い（意図的変更） ------------------------------------

def _zero_limits() -> dict:
    return {category: 0 for category in tagging_core.TagCategory}


def test_req_a8_all_zero_limits_are_success_not_failure(tmp_path):
    tagger = _FakeTagger(tag_result=tagging_core.TagResult())
    changed, failed = _run_loop(
        tmp_path, tagger, ["a.png"], MAX_TAGS_PER_CATEGORY=_zero_limits(),
    )
    assert failed == []
    assert len(changed) == 1
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == ""
    assert changed[0].previous_content is None


def test_req_a8_empty_result_with_normal_limits_is_still_a_failure(tmp_path):
    tagger = _FakeTagger(tag_result=tagging_core.TagResult())
    changed, failed = _run_loop(tmp_path, tagger, ["a.png"])
    assert changed == []
    assert [path.name for path in failed] == ["a.png"]
    assert not (tmp_path / "a.txt").exists()
