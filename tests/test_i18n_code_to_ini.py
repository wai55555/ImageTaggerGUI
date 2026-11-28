"""コード → lang/*.ini 方向の i18n 検査（REQ-1.6 / R8-8・R11-7）。

既存の test_lang_parity.py は「ini 同士」の整合しか見ていないため、コードが参照する
キーが ini に無くても（実行時は生キーが表示されるだけで）テストは通っていた。
ここでは src/*.py の `get_string(...)` 呼び出しを AST で集め、**全言語の ini に
そのセクション／キーが存在すること**を検査する。

- 言語一覧は `lang/*.ini` の走査から得る（テスト側にハードコードしない。R11-7）。
- 引数が文字列リテラルでない呼び出し（f-string・変数）は静的に解決できないので
  対象外。ここは「リテラルで書かれた参照」の検査であり、動的キーは既存の
  tests/test_localization_coverage.py（実行時の解決確認）が受け持つ。
- 現時点で全言語に無いキー（R8-4／R8-6 系）は Phase 8C の REQ-8C.1 が登録する。
  Phase 1 では「新たな欠落を増やさない」ことと「台帳が実測と一致している」ことを
  固定する（キーが登録されたら台帳を消すまで失敗する）。
"""
import ast
import configparser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
LANG_DIR = ROOT / "lang"

# コード側で翻訳を引く関数名（モジュール関数・メソッド・self._get_string 等）。
_GET_STRING_CALLS = {"get_string", "_get_string", "_get_string_internal"}

# Phase 8C（REQ-8C.1）が 9 言語すべてへ登録する既知の欠落（2026-11-28 実測）。
# セクションごと存在しないもの:
_KNOWN_MISSING_SECTIONS = {
    "ConfigUtils": "R8-6: debug ログ専用セクションが全 9 言語の ini に無い",
    "Utils": "R8-6 同類: ログ書き込み失敗の文言がどの ini にも無い",
}
# セクションはあるが、一部キーだけ全 9 言語に無いもの:
_KNOWN_MISSING_KEYS = {
    # R8-4: 全言語（英語含む）で欠けており GUI ログに生キーが出る。
    ("MainWindow", "Reloading_Image_List"): "R8-4 同類: 起動ログの文言",
    ("MainWindow", "Saving_UI_Settings"): "R8-4 同類: 終了時保存ログの文言",
    ("MainWindow", "Warning_Tagging_Already_Running"): "R8-4: GUI ログに生キーが出る",
    # R8-6 同類: core / workers のログ文言（debug_log.txt に生キーが残る）。
    ("TaggerCore", "Config_File_NotFound"): "R8-6 同類: core の起動時ログ",
    ("TaggerCore", "Error_Required_Libraries_NotFound"): "R8-6 同類: core の依存不足ログ",
    ("TaggerCore", "Error_Tagger_Init_Failed_GUI"): "R8-6 同類: core の初期化失敗ログ",
    ("TaggerCore", "Fatal_Error_Config_Load_Failed"): "R8-6 同類: core の致命的エラーログ",
    ("TaggerCore", "Fatal_Error_Config_Load_Failed_GUI"): "R8-6 同類: core の致命的エラーログ",
    ("TaggerCore", "Info_Required_Libraries_Missing"): "R8-6 同類: core の依存不足ログ",
    ("TaggerCore", "Info_Tagging_Core_Main_Start"): "R8-6 同類: core の開始ログ",
    ("TaggerCore", "Tagging_Process_Complete"): "R8-6 同類: core の完了ログ",
    ("TaggerCore", "Total_Image_Files_Found"): "R8-6 同類: core の件数ログ",
    ("TaggerCore", "Total_Processing_Time"): "R8-6 同類: core の所要時間ログ",
    ("TaggerCore", "Warning_No_Image_Files_Found_GUI"): "R8-6 同類: core の警告ログ",
    ("TaggerCore", "Warning_No_Image_Files_Found_In_Dir"): "R8-6 同類: core の警告ログ",
    ("Workers", "BulkTagWorker_Error_File_Processing_Failed"): "R8-6 同類: 一括処理の失敗ログ",
    ("Workers", "BulkTagWorker_File_Processing_Failed"): "R8-6 同類: 一括処理の失敗ログ",
    ("Workers", "DownloaderWorker_Skip_Existing"): "R8-6 同類: DL スキップのログ",
    ("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV_Debug"): "R8-6 同類: DL スキップのログ",
}


def _language_files() -> list[Path]:
    return sorted(LANG_DIR.glob("*.ini"))


def _read_ini(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as handle:
        config.read_file(handle, source=str(path))
    return config


def _literal_get_string_pairs() -> tuple[set[tuple[str, str]], list[tuple[str, int]]]:
    """src/*.py の get_string 呼び出しから (section, key) を集める。

    戻り値の 2 要素目は「引数がリテラルでないため検査できない呼び出し」の
    (file, line)。検査対象外であることを可視化するために返す。
    """
    pairs: set[tuple[str, str]] = set()
    dynamic: list[tuple[str, int]] = []
    for path in sorted(SRC_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            if name not in _GET_STRING_CALLS:
                continue
            args = node.args
            literal_pair = (
                len(args) >= 2
                and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str)
                and isinstance(args[1], ast.Constant) and isinstance(args[1].value, str)
            )
            if literal_pair:
                pairs.add((args[0].value, args[1].value))
            else:
                dynamic.append((path.name, node.lineno))
    return pairs, dynamic


@pytest.fixture(scope="module")
def language_configs() -> dict[str, configparser.ConfigParser]:
    files = _language_files()
    assert files, f"lang/*.ini が 1 つも無い: {LANG_DIR}"
    configs = {path.stem: _read_ini(path) for path in files}
    assert "en" in configs, "基準言語 en.ini が無い"
    return configs


def test_src_uses_get_string_with_literal_keys():
    """検査対象が空振りしていないこと（AST 走査の健全性）。"""
    pairs, _dynamic = _literal_get_string_pairs()
    assert len(pairs) > 100, f"リテラルの get_string 呼び出しが少なすぎる: {len(pairs)}"


def test_every_code_key_exists_in_every_language(language_configs):
    pairs, _dynamic = _literal_get_string_pairs()

    unexpected: dict[tuple[str, str], list[str]] = {}
    still_missing: set[tuple[str, str]] = set()
    for section, key in sorted(pairs):
        missing = [lang for lang, cfg in language_configs.items()
                   if not cfg.has_option(section, key)]
        if not missing:
            continue
        if section in _KNOWN_MISSING_SECTIONS or (section, key) in _KNOWN_MISSING_KEYS:
            still_missing.add((section, key))
            continue
        unexpected[(section, key)] = missing

    assert not unexpected, (
        "コードが参照しているのに lang/*.ini に無いキーがある（生キーが表示される）:\n"
        + "\n".join(
            f"  {section} / {key}: {', '.join(langs)}"
            for (section, key), langs in unexpected.items()
        )
    )

    # 台帳が実測と一致していること（登録済みキーを台帳から消し忘れない）。
    known = {
        pair for pair in pairs
        if pair[0] in _KNOWN_MISSING_SECTIONS or pair in _KNOWN_MISSING_KEYS
    }
    assert still_missing == known, (
        "既知の欠落台帳が実測とずれている。Phase 8C でキーを登録したら台帳も更新すること。\n"
        f"  台帳にあって実測で解消済み: {sorted(known - still_missing)}\n"
        f"  実測で欠落しているが台帳に無い: {sorted(still_missing - known)}"
    )


def test_known_missing_sections_are_actually_absent_in_every_language(language_configs):
    """「セクションごと無い」台帳が古くなっていないこと。"""
    stale = [section for section in _KNOWN_MISSING_SECTIONS
             if any(cfg.has_section(section) for cfg in language_configs.values())]
    assert not stale, (
        "既知の欠落セクションが ini に追加されている。台帳を更新すること: "
        + ", ".join(stale)
    )


def test_language_list_is_discovered_not_hardcoded(language_configs):
    """言語一覧はディスク上の ini と一致し、アプリ側の対応言語もそこから導ける。"""
    from locale_manager import SHIPPED_LANGUAGE_CODES

    assert set(language_configs) == set(SHIPPED_LANGUAGE_CODES), (
        "lang/*.ini と locale_manager.SHIPPED_LANGUAGE_CODES がずれている"
    )
