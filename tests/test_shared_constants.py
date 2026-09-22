"""重複していた固定値が1箇所定義になっていることの回帰テスト
（docs/260922_localization_magic_number_review.md「マジックナンバー・重複固定値」）。

どれも「片方だけ変えると無言で食い違う」種類の値なので、共有元を参照している
こと自体をソースに対して検査する（値そのものを二重に書き写すテストにすると、
テスト側が3つ目の複製になってしまう）。

Offline only - no network. Run:  rtk pytest tests/test_shared_constants.py -q
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import constants
import tagging_core
import vlm_profiles


def _source(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8")


# --- 進捗通知の間引き ---------------------------------------------------------------

def test_progress_step_matches_the_previous_hand_written_formula():
    """`max(1, (total + 199) // 200)` と同じ結果であること（挙動を変えない共通化）。"""
    for total in range(1, 3000):
        assert constants.progress_step_for(total) == max(1, (total + 199) // 200), total


@pytest.mark.parametrize("total,expected_step", [
    (0, 1), (1, 1), (200, 1),
    # 天井除算でないと 201〜399 枚で step=1 になり間引きが効かない。
    (201, 2), (399, 2), (400, 2), (1000, 5), (20000, 100),
])
def test_progress_step_keeps_the_notification_count_within_the_budget(total, expected_step):
    step = constants.progress_step_for(total)
    assert step == expected_step
    if total:
        calls = -(-total // step)
        assert calls <= constants.PROGRESS_SIGNAL_BUDGET, (total, step, calls)


def _code_lines(name: str) -> str:
    """コメント行と行末コメントを落としたソース。文面の言及を誤検出しないため。"""
    out = []
    for line in _source(name).splitlines():
        stripped = line.split("#", 1)[0]
        if stripped.strip():
            out.append(stripped)
    return "\n".join(out)


def test_no_loop_hand_rolls_the_progress_step_any_more():
    """3つの処理ループが constants.progress_step_for() を使っていること。"""
    for name in ("tagging_core.py", "caption_core.py", "vlm_worker.py"):
        source = _source(name)
        assert "constants.progress_step_for(" in source, name
        # 以前の直書き式が戻っていない（コメント内の言及は対象外）。
        code = _code_lines(name)
        assert "+ 199) // 200" not in code, name
        assert re.search(r"//\s*200\b", code) is None, name


# --- 画像拡張子 ---------------------------------------------------------------------

def test_image_extensions_have_a_single_definition():
    source = _source("tagging_core.py")
    assert "constants.IMAGE_EXTENSIONS" in source
    # ローカルでの再定義が戻っていない。
    assert not re.search(r'IMAGE_EXTENSIONS\s*=\s*\[', source), \
        "tagging_core が拡張子一覧を再定義している"


def test_image_scan_covers_exactly_the_shared_extensions(tmp_path):
    for ext in list(constants.IMAGE_EXTENSIONS) + [".txt", ".bmp"]:
        (tmp_path / f"sample{ext}").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.png").write_bytes(b"x")
    found = {p.suffix for p in tagging_core.get_image_paths_recursive(tmp_path)}
    assert found == set(constants.IMAGE_EXTENSIONS)


# --- キャプション自動保存の遅延 -----------------------------------------------------

def test_caption_autosave_delay_is_shared_by_both_views():
    for name in ("main_window.py", "grid_view_widget.py"):
        source = _source(name)
        assert "constants.CAPTION_AUTOSAVE_DELAY_MS" in source, name
        assert "setInterval(1200)" not in source, name


# --- 最大タグ数の上限 ---------------------------------------------------------------

def test_max_tag_caps_are_shared_between_the_main_window_and_the_dialog():
    main = _source("ui_main_window.py")
    dialog = _source("custom_dialogs.py")
    assert "constants.MAX_TAGS_CAP_GENERAL" in main
    assert "constants.MAX_TAGS_CAP_CHARACTER" in main
    assert "constants.MAX_TAGS_CAP_GENERAL" in dialog
    assert "constants.MAX_TAGS_CAP_CHARACTER" in dialog
    # 直書きに戻っていない。
    assert "'Limits', 0, 150, 1" not in main
    assert "'Limits', 0, 10, 1" not in main
    assert '{"character": 10}' not in dialog


def test_category_dialog_caps_match_the_shared_constants():
    import inspect

    from custom_dialogs import CategoryTagSettingsDialog

    signature = inspect.signature(CategoryTagSettingsDialog.__init__)
    assert (signature.parameters["max_limit"].default
            == constants.MAX_TAGS_CAP_GENERAL)
    assert (CategoryTagSettingsDialog._LIMIT_CAPS["character"]
            == constants.MAX_TAGS_CAP_CHARACTER)


# --- VLM 出力トークンの既定値・範囲 -------------------------------------------------

def test_vlm_output_token_bounds_have_a_single_definition():
    import app_settings
    import vlm_transport

    # 設定の既定値、保存値の補正、UI の入力範囲、助言がすべて同じ境界を使う。
    settings = app_settings.load_settings(app_settings.get_default_config())
    assert settings.vlm.max_output_tokens == vlm_profiles.DEFAULT_MAX_OUTPUT_TOKENS
    assert vlm_transport._SUGGESTED_MAX_OUTPUT_TOKENS == vlm_profiles.DEFAULT_MAX_OUTPUT_TOKENS

    low = vlm_profiles.GenerationProfile.from_mapping({"max_output_tokens": -5})
    high = vlm_profiles.GenerationProfile.from_mapping({"max_output_tokens": 10 ** 9})
    assert low.max_output_tokens == vlm_profiles.MIN_MAX_OUTPUT_TOKENS
    assert high.max_output_tokens == vlm_profiles.MAX_MAX_OUTPUT_TOKENS

    for name in ("app_settings.py", "vlm_settings_dialog.py", "vlm_transport.py"):
        source = _source(name)
        assert "3072" not in source or name == "vlm_transport.py", name
    dialog = _source("vlm_settings_dialog.py")
    assert "setRange(16, 32768)" not in dialog
    assert "MIN_MAX_OUTPUT_TOKENS" in dialog and "MAX_MAX_OUTPUT_TOKENS" in dialog


def test_settings_spinbox_range_matches_the_profile_clamp():
    from PySide6.QtWidgets import QApplication
    import app_settings
    from vlm_settings_dialog import VlmSettingsDialog

    QApplication.instance() or QApplication([])
    settings = app_settings.load_settings(app_settings.get_default_config())
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        assert dialog.max_tokens.minimum() == vlm_profiles.MIN_MAX_OUTPUT_TOKENS
        assert dialog.max_tokens.maximum() == vlm_profiles.MAX_MAX_OUTPUT_TOKENS
    finally:
        dialog.close()


# --- Qt の data role ---------------------------------------------------------------

def test_custom_connection_role_is_a_named_user_role():
    from PySide6.QtCore import Qt
    import vlm_settings_dialog as V

    assert V._CUSTOM_CONNECTION_ID_ROLE >= Qt.ItemDataRole.UserRole
    source = _source("vlm_settings_dialog.py")
    assert "setData(1000" not in source and "data(1000)" not in source
    assert source.count("_CUSTOM_CONNECTION_ID_ROLE") >= 3


def test_custom_connection_id_round_trips_through_the_named_role():
    from PySide6.QtWidgets import QApplication, QListWidgetItem
    import vlm_settings_dialog as V

    QApplication.instance() or QApplication([])
    item = QListWidgetItem("route")
    item.setData(V._CUSTOM_CONNECTION_ID_ROLE, "custom-abc123")
    assert item.data(V._CUSTOM_CONNECTION_ID_ROLE) == "custom-abc123"


# --- タグ出力上限の暗黙値 -----------------------------------------------------------

def test_default_tag_output_cap_is_named():
    assert tagging_core.DEFAULT_TAG_OUTPUT_CAP == 100
    source = _source("tagging_core.py")
    assert "DEFAULT_TAG_OUTPUT_CAP" in source
    assert "if cat_limits else 100" not in source


def test_default_tag_output_cap_applies_only_without_per_category_limits():
    """max_tags を渡さない呼び出し（テスト・スクリプト利用）だけに効くこと。"""
    source = _source("tagging_core.py")
    # 呼び出し側が上限を渡したらその合計が使われる、という分岐が残っている。
    assert "sum(cat_limits.values()) if cat_limits" in source
