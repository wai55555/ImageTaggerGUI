"""ローカライズ経路の回帰テスト（docs/260922_localization_magic_number_review.md）。

test_lang_parity.py は lang/*.ini 同士の整合しか見ないため、次の3つを検出できない
（同レビューの「テスト上の見落とし」節）。ここでそれを埋める。

1. OS 言語コードの正規化結果が lang 内の実ファイルに存在すること
2. コードが参照する翻訳キーが全言語で解決すること（生キー・未差し込みが出ない）
3. 診断・GPU・モデル一覧・候補選択のエラー理由が表示層で訳されること

Offline only - no network. Run:  rtk pytest tests/test_localization_coverage.py -q
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import vlm_diagnostics as D
import vlm_errors as E
import vlm_router as R
from locale_manager import (
    LocaleManager, SHIPPED_LANGUAGE_CODES, available_language_codes,
    normalize_language_code,
)

LANGS = ["en", "ja", "de", "es", "fr", "ko", "ru", "zh_CN", "zh_TW"]
# 日本語（ひらがな・カタカナ・漢字）。他言語に混ざっていたらハードコード残り。
_JA = re.compile(r"[぀-ヿ]")


@pytest.fixture(scope="module")
def managers():
    return {lang: LocaleManager(lang, ROOT / "lang") for lang in LANGS}


# --- 1. OS 言語コードの正規化 -------------------------------------------------------

def test_available_language_codes_finds_the_shipped_files():
    codes = available_language_codes(ROOT / "lang")
    assert sorted(codes) == sorted(SHIPPED_LANGUAGE_CODES)


def test_available_language_codes_falls_back_when_nothing_is_listable(tmp_path):
    """lang の配置に失敗していても言語選択を壊さない。"""
    assert available_language_codes(tmp_path / "missing") == list(SHIPPED_LANGUAGE_CODES)
    assert available_language_codes(None) == list(SHIPPED_LANGUAGE_CODES)


@pytest.mark.parametrize("raw,expected", [
    # 以前の実装が落としていた言語（Windows の LANGID 表に無かった）
    ("es-ES", "es"), ("es_MX", "es"), ("ru-RU", "ru"),
    # 中国語: zh.ini は存在しないので、必ず zh_CN / zh_TW のどちらかへ寄せる
    ("zh-CN", "zh_CN"), ("zh_CN.UTF-8", "zh_CN"), ("zh-SG", "zh_CN"),
    ("zh-Hans", "zh_CN"), ("zh", "zh_CN"),
    ("zh-TW", "zh_TW"), ("zh-HK", "zh_TW"), ("zh-MO", "zh_TW"), ("zh_Hant", "zh_TW"),
    # 地域付きは言語部分へ、codeset / modifier は無視
    ("ja-JP", "ja"), ("ja_JP.UTF-8", "ja"), ("de-AT", "de"), ("fr_CA@euro", "fr"),
    ("ko-KR", "ko"), ("en-GB", "en"),
    # 翻訳が無い言語・壊れた値は英語へ
    ("pt-BR", "en"), ("xx-YY", "en"), ("", "en"), ("   ", "en"),
])
def test_normalize_language_code(raw, expected):
    codes = available_language_codes(ROOT / "lang")
    assert normalize_language_code(raw, codes) == expected


def test_normalize_language_code_accepts_none():
    assert normalize_language_code(None) == "en"


def test_every_normalized_os_language_has_a_real_ini_file():
    """正規化結果は必ず実在する lang/<code>.ini を指す。

    この値は初回起動時に config.ini へ保存され、以後の起動でも使われるため、
    存在しないコード（かつての "zh"）を返すと利用者はずっと英語表示になる。
    """
    codes = available_language_codes(ROOT / "lang")
    raws = ["en-US", "ja-JP", "de-DE", "es-ES", "fr-FR", "ko-KR", "ru-RU",
            "zh-CN", "zh-TW", "zh-HK", "zh-SG", "zh-MO", "zh", "pt-BR", "ar-SA",
            "", "not a locale"]
    for raw in raws:
        resolved = normalize_language_code(raw, codes)
        assert (ROOT / "lang" / f"{resolved}.ini").is_file(), (raw, resolved)


def test_get_os_language_returns_a_shipped_code():
    from main_window import get_os_language
    assert get_os_language() in SHIPPED_LANGUAGE_CODES


# --- 2 / 3. 表示層でエラー理由が訳されること ----------------------------------------

def test_every_diagnostic_item_and_status_label_is_translated(managers):
    for lang, lm in managers.items():
        for name in D.DIAG_ITEM_LABEL_KEYS:
            label = D.item_label(name, lm.get_string)
            assert label and not label.startswith("Diag_Item_"), (lang, name)
        for status in D.DiagStatus:
            label = D.status_label(status, lm.get_string)
            assert label and not label.startswith("Diag_Status_"), (lang, status)
            if lang not in ("ja",):
                assert not _JA.search(label), (lang, status, label)
    # 日本語では内部IDがそのまま出ない（"PASS" を素通しする実装への退行検出）。
    ja = managers["ja"].get_string
    for status in D.DiagStatus:
        assert D.status_label(status, ja) != status.value, status
    # "DNS / TCP" と "TLS" は日本語でも同じ略語なので、意図的に一致する。
    acronym_only = {"DNS / TCP", "TLS"}
    for name in D.DIAG_ITEM_LABEL_KEYS:
        if name in acronym_only:
            assert D.item_label(name, ja) == name, name
            continue
        assert D.item_label(name, ja) != name, name


def test_every_error_reason_label_is_translated(managers):
    for lang, lm in managers.items():
        for reason in E.VlmErrorReason:
            label = E.reason_label(reason, lm.get_string)
            assert label and not label.startswith("Error_Reason_"), (lang, reason)
            # 訳せた言語では enum の生の値（"rate_limited" 等）が出ない。
            assert "_" not in label or lang == "en", (lang, reason, label)


def test_diag_item_detail_is_translated_and_keeps_the_raw_suffix(managers):
    """翻訳した本文の後ろに、サーバーが返した生の文字列を素で足す。"""
    item = D.DiagItem("HTTP response", D.DiagStatus.FAIL,
                      "401 auth rejected: bad key",
                      detail_key="Diag_D_Auth_Rejected",
                      detail_args={"status": 401},
                      detail_suffix="bad key")
    for lang, lm in managers.items():
        text = D.item_detail(item, lm.get_string)
        assert "401" in text and text.endswith("bad key"), (lang, text)
        assert "{" not in text and "}" not in text, (lang, text)
        assert not text.startswith("Diag_D_"), (lang, text)


def test_diag_item_without_a_key_keeps_the_raw_value(managers):
    """URL やモデルIDのような「値」は翻訳対象ではないのでそのまま出す。"""
    item = D.DiagItem("URL format", D.DiagStatus.PASS, "https://example.test/v1")
    for lm in managers.values():
        assert D.item_detail(item, lm.get_string) == "https://example.test/v1"


def test_diag_arg_key_resolves_the_nested_reason(managers):
    """差し込む値自体が翻訳キーの場合（トランスポート層の失敗理由）も訳す。"""
    item = D.DiagItem(
        "HTTP response", D.DiagStatus.FAIL, "timeout: read timed out",
        detail_key="Diag_D_Transport_Error",
        detail_args={"reason": D.DiagArgKey(
            "Vlm", E.reason_label_key(E.VlmErrorReason.TIMEOUT), "timeout")},
        detail_suffix="read timed out")
    ja = D.item_detail(item, managers["ja"].get_string)
    en = D.item_detail(item, managers["en"].get_string)
    assert ja != en, "理由ラベルが言語で切り替わっていない"
    assert ja.endswith("read timed out") and "タイムアウト" in ja


def test_report_lines_are_fully_translated(managers):
    report = D.DiagReport(connection_id="builtin-gemini")
    report.add("Auth", D.DiagStatus.FAIL, "bearer required but no credential found",
               detail_key="Diag_D_Credential_Required", detail_args={"type": "bearer"})
    report.add("TLS", D.DiagStatus.SKIP, "plain http", detail_key="Diag_D_Plain_Http")
    for lang, lm in managers.items():
        lines = D.format_report_lines(report, lm.get_string)
        assert len(lines) == 2
        joined = "\n".join(lines)
        assert "Diag_" not in joined, (lang, joined)
        assert "{" not in joined and "}" not in joined, (lang, joined)
        if lang != "ja":
            assert not _JA.search(joined), (lang, joined)


def test_attempt_error_text_translates_reason_and_body(managers):
    error = E.VlmAttemptError(
        E.VlmErrorReason.BAD_RESPONSE, 200, "no model ids in response",
        message_key="ModelList_No_Model_Ids")
    for lang, lm in managers.items():
        text = E.attempt_error_text(error, lm.get_string)
        assert "ModelList_" not in text and "bad_response" not in text, (lang, text)
        if lang != "ja":
            assert not _JA.search(text), (lang, text)


def test_attempt_error_text_keeps_a_raw_server_message(managers):
    """message_key を持たない（サーバー本文そのまま）エラーは英語のまま出す。"""
    error = E.VlmAttemptError(E.VlmErrorReason.SERVER_ERROR, 500, "upstream exploded")
    text = E.attempt_error_text(error, managers["ja"].get_string)
    assert text.endswith("upstream exploded")
    assert "サーバーエラー" in text


def test_localized_candidate_failure_translates_reason_and_exclusions(managers):
    excluded = {"builtin-gemini": "no_auth", "builtin-vercel": "not_verified"}
    for lang, lm in managers.items():
        text = R.localized_candidate_failure("no_verified_candidate", excluded,
                                             lm.get_string)
        # 理由コードは併記する（問い合わせ時に言語を跨いで照合できるように）。
        assert "no_verified_candidate" in text
        assert "builtin-gemini" in text and "builtin-vercel" in text
        assert "Reject_" not in text and "Exclude_" not in text, (lang, text)
        if lang != "ja":
            assert not _JA.search(text), (lang, text)
    # 理由コードの後ろに続く「対処案」そのものが訳されている（除外理由だけが
    # 訳されて助言は英語のまま、という退行を検出する）。
    ja = R.localized_candidate_failure("no_verified_candidate", None,
                                       managers["ja"].get_string)
    en = R.localized_candidate_failure("no_verified_candidate", None,
                                       managers["en"].get_string)
    assert ja != en
    hint_ja = ja.split("no_verified_candidate:", 1)[1]
    assert _JA.search(hint_ja), hint_ja
    for lang, lm in managers.items():
        if lang == "ja":
            continue
        hint = R.localized_candidate_failure(
            "no_verified_candidate", None, lm.get_string).split(":", 1)[1]
        assert not _JA.search(hint), (lang, hint)


def test_localized_candidate_failure_handles_unknown_codes(managers):
    text = R.localized_candidate_failure("something_new", {"c": "brand_new_reason"},
                                         managers["ja"].get_string)
    assert "something_new" in text and "brand_new_reason" in text
    assert "Reject_" not in text and "Exclude_" not in text


def test_gpu_worker_translates_keyed_log_messages(managers, monkeypatch):
    """gpu_runtime は英語で通知し、ワーカー側が [Gpu] から訳して表示する。"""
    from workers import GpuRuntimeDownloadWorker

    worker = GpuRuntimeDownloadWorker(managers["ja"].get_string)
    shown: list[tuple[str, str]] = []
    worker.log_message.connect(lambda msg, color: shown.append((msg, color)))
    worker._on_log("downloading cudnn64_9.dll", "info",
                   key="Runtime_Downloading", args={"name": "cudnn64_9.dll"})
    worker._on_log("install aborted: boom", "error",
                   key="Runtime_Install_Aborted", args={"detail": "boom"})
    # キーを持たない通知（生の例外文など）は英語のまま出す。
    worker._on_log("raw passthrough", "warn")
    assert shown[0][0] == "cudnn64_9.dll をダウンロード中"
    assert shown[1] == ("導入を中止しました: boom", "red")
    assert shown[2] == ("raw passthrough", "orange")


def _gpu_log_calls():
    """gpu_runtime.py の `log(...)` 呼び出しを (行番号, key) で列挙する。"""
    import ast
    source = (ROOT / "src" / "gpu_runtime.py").read_text(encoding="utf-8")
    calls = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "log"):
            continue
        key = ""
        for kw in node.keywords:
            if kw.arg == "key" and isinstance(kw.value, ast.Constant):
                key = str(kw.value.value)
        calls.append((node.lineno, key))
    return calls


def test_every_gpu_runtime_log_call_carries_a_translation_key(managers):
    """gpu_runtime の全 log() 呼び出しが翻訳キーを伴うこと。

    キーの無い通知は workers._on_log が英語のまま画面へ流すため、1箇所でも
    落ちると全言語でそこだけ英語になる。呼び出し側を直接検査する。
    """
    calls = _gpu_log_calls()
    assert calls, "gpu_runtime に log() 呼び出しが見つからない"
    missing = [line for line, key in calls if not key]
    assert not missing, f"gpu_runtime.py の log() に key= が無い行: {missing}"
    for lang, lm in managers.items():
        for line, key in calls:
            assert lm.translations.get("Gpu", key, fallback=None), (lang, line, key)


def test_custom_connection_choices_are_translated(managers):
    """実際のダイアログ上で、認証方式と接続先の選択肢が訳されていること。

    ヘルパー（_combo）だけを見ると、ダイアログ側が translate を渡し忘れても
    気付けないので、ダイアログを組み立てて表示ラベルを確認する。
    """
    from PySide6.QtWidgets import QApplication
    import custom_connection_dialog as C

    QApplication.instance() or QApplication([])
    for lang, lm in managers.items():
        dialog = C.CustomConnectionDialog(lm.get_string)
        try:
            for combo, values in ((dialog.auth_type_combo,
                                   ["none", "bearer", "header_key", "query_key"]),
                                  (dialog.locality_combo, None)):
                labels = [combo.itemText(i) for i in range(combo.count())]
                if values is not None:
                    assert [combo.itemData(i)
                            for i in range(combo.count())] == values, lang
                assert all(not t.startswith("Custom_") for t in labels), (lang, labels)
                if lang != "ja":
                    assert not any(_JA.search(t) for t in labels), (lang, labels)
            # 日本語では英語固定だった文言がそのまま出ない。
            if lang == "ja":
                auth_labels = [dialog.auth_type_combo.itemText(i)
                               for i in range(dialog.auth_type_combo.count())]
                assert "None" not in auth_labels, auth_labels
                locality_labels = [dialog.locality_combo.itemText(i)
                                   for i in range(dialog.locality_combo.count())]
                assert "Auto" not in locality_labels, locality_labels
            # プロトコル名は製品名なので訳さない。
            protocols = [dialog.protocol_combo.itemText(i)
                         for i in range(dialog.protocol_combo.count())]
            assert "OpenAI Chat Completions" in protocols, lang
        finally:
            dialog.close()
