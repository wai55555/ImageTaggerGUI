"""undo_manager の Undo 説明文が lang/*.ini から引かれることの回帰テスト。

以前は undo_manager.py の description() に日本語がハードコードされていて、
多言語対応アプリなのに Undo/Redo ボタンのツールチップだけ常に日本語だった。
このテストは (1) どの言語でもキーが解決し {placeholder} が残らないこと、
(2) ソースに UI 文字列としての日本語が戻ってこないこと、
(3) 日本語の文言が移行前と変わっていないことを固定する。

Offline only - no network, no GUI. Run:  rtk pytest tests/test_undo_descriptions_i18n.py -q
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest

import app_settings
import undo_manager as U
from locale_manager import LocaleManager

LANGS = ["en", "ja", "de", "es", "fr", "ko", "ru", "zh_CN", "zh_TW"]

# 日本語（ひらがな・カタカナ・漢字）。コメント／docstring は日本語のままで良いので、
# 判定は description() の戻り値に対してだけ行う。
_CJK = re.compile(r"[぀-ヿ一-鿿]")


def _actions():
    """description() を持つ全 UndoAction を、分岐ごとに1つずつ。"""
    return [
        ("AddTagsAction/1", U.AddTagsAction(Path("a.txt"), ["cat"])),
        ("AddTagsAction/3", U.AddTagsAction(Path("a.txt"), ["cat", "dog", "fox"])),
        ("AddTagsAction/5", U.AddTagsAction(
            Path("a.txt"), ["cat", "dog", "fox", "owl", "ant"])),
        ("RemoveTagAction", U.RemoveTagAction(Path("a.txt"), "cat", 0)),
        ("EditCaptionAction", U.EditCaptionAction(Path("a.txt"), "before", "after")),
        ("OverwriteFileAction", U.OverwriteFileAction(Path("dir/a.txt"), "p", "n")),
        ("AppendTagsActionV2", U.AppendTagsActionV2(
            Path("dir/a.txt"), "p", "n", ["cat", "dog"])),
        ("CompositeUndoAction", U.CompositeUndoAction([])),
        ("BulkAddTagsAction/1", U.BulkAddTagsAction(
            [Path("a.txt")] * 3, ["cat"], "append")),
        ("BulkAddTagsAction/2", U.BulkAddTagsAction(
            [Path("a.txt")] * 3, ["cat", "dog"], "append")),
        ("BulkRemoveTagsAction", U.BulkRemoveTagsAction("cat", [(Path("a.txt"), 0)] * 3)),
    ]


@pytest.fixture
def wire_language(monkeypatch):
    """`app_settings._get_string` を指定言語へ差し替える（テスト後に自動で戻る）。

    set_get_string_func() はモジュールグローバルを書き換えるので、他テストへ
    漏らさないよう monkeypatch 経由で当てる。undo_manager._desc() は呼び出しごとに
    app_settings から属性を引くため、これで反映される。
    """
    def _wire(lang: str) -> LocaleManager:
        lm = LocaleManager(lang, ROOT / "lang")
        monkeypatch.setattr(app_settings, "_get_string", lm.get_string)
        return lm
    return _wire


@pytest.mark.parametrize("lang", LANGS)
def test_every_undo_description_resolves_in_every_language(lang, wire_language):
    lm = wire_language(lang)
    for name, action in _actions():
        desc = action.description()
        assert desc, f"{lang} / {name}: 空の説明文"
        # キーがそのまま出ている = その言語の ini にキーが無い（en へのフォールバックも
        # 効いていない）。locale_manager は en を先に読むので、本来ここには来ない。
        assert not desc.startswith("Undo_Desc_"), f"{lang} / {name}: 未翻訳 ({desc})"
        # format() に渡し漏れたプレースホルダが残っていない。
        assert "{" not in desc and "}" not in desc, f"{lang} / {name}: {desc}"
        # ツールチップは Undo_Action の {desc} に差し込まれる。そこでも壊れないこと。
        tooltip = lm.get_string("MainWindow", "Undo_Action", desc=desc)
        assert desc in tooltip and "{desc}" not in tooltip


def test_non_japanese_locales_get_no_japanese_text(wire_language):
    """日本語以外の言語で日本語が出ないこと（ハードコード再発の検出）。"""
    for lang in [l for l in LANGS if l not in ("ja", "zh_CN", "zh_TW")]:
        wire_language(lang)
        for name, action in _actions():
            desc = action.description()
            assert not _CJK.search(desc), f"{lang} / {name}: 日本語が残っている ({desc})"


def test_japanese_wording_is_unchanged_by_the_move_to_ini(wire_language):
    """移行前に description() がハードコードしていた日本語と一字一句同じであること。"""
    wire_language("ja")
    expected = {
        "AddTagsAction/1": "「cat」の追加",
        "AddTagsAction/3": "「cat, dog, fox」の追加",
        "AddTagsAction/5": "「cat, dog, fox...」など5個のタグの追加",
        "RemoveTagAction": "「cat」の削除",
        "EditCaptionAction": "キャプションの編集",
        "OverwriteFileAction": "「a.txt」の上書き",
        "AppendTagsActionV2": "「a.txt」へ2件のタグを追記",
        "CompositeUndoAction": "タグ付けによる0ファイルの変更",
        "BulkAddTagsAction/1": "「cat」の一括追加（3ファイル）",
        "BulkAddTagsAction/2": "2個のタグの一括追加（3ファイル）",
        "BulkRemoveTagsAction": "「cat」の一括削除（3ファイル）",
    }
    actual = {name: action.description() for name, action in _actions()}
    assert actual == expected


def test_composite_label_from_the_caller_still_wins(wire_language):
    """main_window が Undo_Batch_Tagging から翻訳して渡す label を上書きしないこと。"""
    lm = wire_language("en")
    label = lm.get_string("MainWindow", "Undo_Batch_Tagging", count=7)
    assert U.CompositeUndoAction([], label=label).description() == label
    assert label != U.CompositeUndoAction([]).description()


def test_description_does_not_raise_before_the_locale_is_wired(monkeypatch):
    """LocaleManager 配線前（起動直後）でも description() が例外を投げないこと。"""
    monkeypatch.setattr(app_settings, "_get_string",
                        app_settings.default_get_string_fallback)
    for name, action in _actions():
        assert action.description(), name


def test_no_ui_string_in_undo_manager_hardcodes_japanese():
    """description() 相当の `return "<日本語>"` がソースへ戻っていないこと。

    コメント／docstring の日本語は対象外（このリポジトリの規約）。
    """
    offenders = []
    for lineno, line in enumerate(
            (ROOT / "src" / "undo_manager.py").read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith(("return ", "return\t")):
            continue
        if _CJK.search(stripped):
            offenders.append(f"undo_manager.py:{lineno}: {stripped}")
    assert not offenders, "UI文字列がハードコードされています:\n" + "\n".join(offenders)
