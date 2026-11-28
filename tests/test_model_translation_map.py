"""Shared tag translation map tests（REQ-1.8 / REQ-1.4 の R11-4）。

以前は `load_tag_translation_map` を丸ごと偽物へ差し替え、「偽の戻り値が素通りする
こと」と「呼び出し回数」しか見ていなかった。それでは本物の CSV パースが壊れても
テストが通ってしまう。ここでは

- `MainWindow._load_shared_tag_translation_map()` が渡す引数（3 つの探索先）を
  assert し、
- 差し替えた関数は本物（`tag_utils.load_tag_translation_map`）へ委譲して、
  CSV のパース・位置対応・英語フォールバックまでを実データで検証する。

PixAI v0.9 / v1.0 のどちらも「モデル個別の翻訳設定を持たず、共有マップを使う」
（`model_config.json` に `tag_translation` キーが無い）ことも同時に固定する。
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import constants
import main_window
import tag_utils
from model_registry import get_model_entry

PIXAI_MODEL_IDS = ("pixai-tagger-v0.9", "pixai-tagger-v1.0")


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_english_csv(directory: Path, tag_names: list[str]) -> None:
    """PixAI 形式（3 列目がタグ名）。アンダースコアは空白へ変換される。"""
    _write_csv(
        directory / "selected_tags.csv",
        ["id", "tag_id", "name", "category", "count", "ips"],
        [[str(i), str(i), name, "general", "0", "0"] for i, name in enumerate(tag_names)],
    )


def _write_language_csv(directory: Path, suffix: str, translations: list[str]) -> None:
    """翻訳 CSV は 1 列目が翻訳（ヘッダー行は読み飛ばされる）。"""
    _write_csv(directory / f"selected_tags_{suffix}.csv", ["name"],
               [[text] for text in translations])


def _point_constants_at(monkeypatch, model_dir: Path, primary_root: Path,
                        resource_root: Path) -> None:
    monkeypatch.setattr(constants, "MODEL_PATH", model_dir / "model.onnx")
    monkeypatch.setattr(constants, "MODELS_DIR", primary_root)
    monkeypatch.setattr(constants, "MODELS_RESOURCE_DIR", resource_root)


def test_pixai_models_use_the_shared_translation_map_not_a_per_model_config():
    for model_id in PIXAI_MODEL_IDS:
        entry = get_model_entry(model_id)
        assert entry is not None, model_id
        assert "tag_translation" not in entry.config, (
            f"{model_id} は共有マップ前提（モデル個別の tag_translation を持たない）"
        )


def test_shared_map_wiring_and_real_csv_parsing(monkeypatch, tmp_path):
    """呼び出し引数（3 探索先）と、本物のパーサの結果を同時に固定する。"""
    model_dir = tmp_path / "model"
    primary_root = tmp_path / "models"
    resource_root = tmp_path / "resource_models"

    _write_english_csv(model_dir, ["1girl", "solo", "blue_eyes"])
    _write_language_csv(model_dir, "jp", ["1人の女の子", "ソロ", ""])
    _point_constants_at(monkeypatch, model_dir, primary_root, resource_root)

    calls: list[tuple[Path, ...]] = []

    def recording_loader(*search_dirs):
        # 記録しつつ、本物の実装へ委譲する（内側のロジックもテストされる）。
        calls.append(tuple(search_dirs))
        return tag_utils.load_tag_translation_map(*search_dirs)

    monkeypatch.setattr(main_window, "load_tag_translation_map", recording_loader)

    mapping = main_window.MainWindow._load_shared_tag_translation_map(SimpleNamespace())

    # 探索順は「モデルの実体 → ユーザー側 models/ → 同梱 resource」で固定。
    assert calls == [(
        model_dir,
        primary_root / constants.MODEL_DIR_NAME,
        resource_root / constants.MODEL_DIR_NAME,
    )]

    # 英語タグ（アンダースコア→空白）がキーになり、日本語は 1 列目に入る。
    assert set(mapping) == {"1girl", "solo", "blue eyes"}
    assert mapping["1girl"][0] == "1人の女の子"
    assert mapping["solo"][0] == "ソロ"
    # 空の翻訳はその位置だけ英語へフォールバックする（他の言語も英語）。
    assert mapping["blue eyes"] == ["blue eyes"] * len(tag_utils._TRANSLATION_LANGUAGE_SUFFIXES)


def test_shared_map_falls_back_per_file_across_models_dirs(monkeypatch, tmp_path):
    """英語 CSV と翻訳 CSV が別ディレクトリでも、ファイル単位のフォールバックで効く。

    frozen ビルドでは同梱 CSV が models/ 側にあり、旧レイアウトでは
    MODEL_PATH.parent が別名ディレクトリを指すことがある。片方にしか無いファイルを
    もう片方から拾えないと翻訳が黙って消える。
    """
    model_dir = tmp_path / "legacy_model"
    primary_root = tmp_path / "models"
    resource_root = tmp_path / "resource_models"
    primary_dir = primary_root / constants.MODEL_DIR_NAME

    _write_english_csv(primary_dir, ["1girl", "solo"])
    _write_language_csv(resource_root / constants.MODEL_DIR_NAME, "jp", ["1人の女の子", "ソロ"])
    _point_constants_at(monkeypatch, model_dir, primary_root, resource_root)

    mapping = main_window.MainWindow._load_shared_tag_translation_map(SimpleNamespace())

    assert mapping["1girl"][0] == "1人の女の子"
    assert mapping["solo"][0] == "ソロ"


def test_shared_map_is_empty_when_no_english_csv_exists(monkeypatch, tmp_path):
    """英語 CSV がどこにも無ければ空（呼び出し側は英語表示へフォールバックする）。"""
    _point_constants_at(monkeypatch, tmp_path / "missing",
                        tmp_path / "models", tmp_path / "resource_models")
    assert main_window.MainWindow._load_shared_tag_translation_map(SimpleNamespace()) == {}
