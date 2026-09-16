from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt

import app_settings
import constants
from utils import write_debug_log
from model_registry import ModelEntry, config_mapping
from tag_utils import load_tag_translation_map

if TYPE_CHECKING:
    from main_window import MainWindow


class ModelModeController:
    """
    Consolidates every side effect that must fire when the active model changes
    (design.md 6.8): character-tag UI visibility, translation-map reload,
    threshold/limit slider defaults, and tagger<->captioner display switching.

    MainWindow owns exactly one instance of this and delegates to it via
    on_model_changed(); the branching logic itself does not live in MainWindow.
    """

    def __init__(self, main_window: "MainWindow") -> None:
        self.mw = main_window

    def on_model_changed(self, entry: ModelEntry) -> None:
        write_debug_log(f"ModelModeController: switching to model_id={entry.model_id} ({entry.model_type})")
        self._apply_character_ui_visibility(entry)
        self._reload_translation_map(entry)
        self._apply_threshold_defaults(entry)
        self._apply_model_type_ui(entry)
        self.mw._check_model_status_and_update_ui()

    def _apply_character_ui_visibility(self, entry: ModelEntry) -> None:
        """design.md 6.6節: grey out character-category controls for models without one."""
        supports_character = bool(config_mapping(entry.config, "ui").get("supports_character_tag", True))
        for key in ("thresholds_character", "limits_character"):
            slider_pair = self.mw._sliders.get(key)
            if slider_pair:
                slider, value_label = slider_pair
                slider.setEnabled(supports_character)
                value_label.setEnabled(supports_character)

    def _reload_translation_map(self, entry: ModelEntry) -> None:
        """Regardless of which model is currently selected, look up by tag name from there."""
        mw = self.mw
        mw.tag_translation_map = load_tag_translation_map(constants.MODEL_PATH.parent,
                                                       constants.MODELS_DIR / constants.MODEL_DIR_NAME,
                                                       constants.MODELS_RESOURCE_DIR / constants.MODEL_DIR_NAME)
        mw.display_current_tag_page()
        mw._display_image_tag_page()
        mw.grid_view_widget.set_tag_display_language(mw._tag_display_language, mw.tag_translation_map)

    # Fallback per-category defaults when a model_config.json doesn't spell them out.
    _FALLBACK_THRESHOLDS = {"general": 0.40, "character": 0.65, "rating": 0.50,
                            "copyright": 0.50, "artist": 0.50, "meta": 0.50,
                            "model": 0.50, "quality": 0.50, "year": 0.50}
    _FALLBACK_LIMITS = {"general": 55, "character": 1, "rating": 0, "copyright": 0,
                        "artist": 0, "meta": 0, "model": 0, "quality": 0, "year": 0}

    def _model_default_threshold(self, ui_cfg: dict, category: str) -> float:
        explicit = ui_cfg.get("default_thresholds", {}).get(category)
        if explicit is not None:
            return float(explicit)
        if category == "general":
            return float(ui_cfg.get("default_threshold", self._FALLBACK_THRESHOLDS["general"]))
        if category == "character":
            return float(ui_cfg.get("default_character_threshold",
                                    ui_cfg.get("default_threshold", self._FALLBACK_THRESHOLDS["character"])))
        # rating / copyright / artist / meta / model / quality / year: use the model's
        # own default_threshold so "set the model to X everywhere" is a one-value change.
        return float(ui_cfg.get("default_threshold", self._FALLBACK_THRESHOLDS.get(category, 0.50)))

    def _model_default_limit(self, ui_cfg: dict, category: str) -> int:
        explicit = ui_cfg.get("default_limits", {}).get(category)
        if explicit is not None:
            return int(explicit)
        return int(self._FALLBACK_LIMITS.get(category, 0))

    def _apply_threshold_defaults(self, entry: ModelEntry) -> None:
        """Seed each category this model produces with its recommended threshold / max-tag
        default - but only for categories the user has not adjusted by hand ("touched").
        Runs on every model switch and once at startup (design: 2026-08-31 user decision)."""
        ui_cfg = config_mapping(entry.config, "ui")
        categories = ui_cfg.get("categories", ["general", "character"])

        thresholds = self.mw.settings.thresholds
        limits = self.mw.settings.limits
        threshold_touched = app_settings.parse_touched(thresholds.touched)
        limit_touched = app_settings.parse_touched(limits.touched)

        for category in categories:
            if not hasattr(thresholds, category):
                continue
            if category not in threshold_touched:
                setattr(thresholds, category, self._model_default_threshold(ui_cfg, category))
            if category not in limit_touched:
                setattr(limits, category, self._model_default_limit(ui_cfg, category))

        self.mw._sync_settings_sliders()
        self.mw.save_current_config()

    def _apply_model_type_ui(self, entry: ModelEntry) -> None:
        """design.md 8.5節 / spec.md 8.3節: switch the tag-button grid <-> caption text editor,
        and hide tag-only bulk-edit/Undo controls that make no sense for free-text captions.

        テキスト編集欄を出すのは「選択モデルが captioner」または「VLM接続を使う」のとき。
        VLM の出力はプロンプト次第（タグ列 / 自然文）だが、任意テキストを安全に扱えるのは
        テキスト編集欄なので、VLM ON のときはそちらを使う。"""
        mw = self.mw
        model_is_captioner = entry.model_type == "captioner"
        use_vlm = bool(getattr(mw.settings.vlm, "enabled", False))
        text_ui = model_is_captioner or use_vlm

        # VLM 使用中はローカルモデルは働かないので、モデル欄をグレーアウトして
        # 誤操作を防ぐ（チェックを外すと元に戻る）。処理中は _set_main_controls_enabled
        # が別途ロックする（どちらも「VLM ON なら無効」で一致するので競合しない）。
        if hasattr(mw, "model_combo"):
            mw.model_combo.setEnabled(not use_vlm and mw._controls_enabled)
            mw.model_combo.setToolTip(
                mw.locale_manager.get_string("Vlm", "Model_Combo_Locked_Tooltip") if use_vlm
                else mw.locale_manager.get_string("ModelDescriptions", entry.model_id))

        mw.tag_grid_container.setVisible(not text_ui)
        mw.image_tag_prev_page_btn.setVisible(not text_ui)
        mw.image_tag_next_page_btn.setVisible(not text_ui)
        mw.add_single_tag_label.setVisible(not text_ui)
        mw.add_single_tag_line.setVisible(not text_ui)
        mw.add_single_tag_button.setVisible(not text_ui)
        # 一括削除／一括追加＋しきい値・最大タグ数スライダーは ONNX タガー専用。
        # 個別 hide ではなくブロックごと隠すことで、消えた跡の空白と片欠けを防ぎ、
        # 縦スプリッターの再配分（_rebalance_right_splitter）が効くようにする。
        if hasattr(mw, "tagger_config_block"):
            mw.tagger_config_block.setVisible(not text_ui)
        # Undo/Redo stays visible in text mode - text edits are undoable too
        # (2026-08-31 user decision).
        if hasattr(mw, "grid_view_widget"):
            mw.grid_view_widget.set_caption_mode(text_ui)
        # Grid-view (3x3 edit) tag search filter is a separate widget tree
        # (grid_view_widget.py) not yet retrofitted for caption mode - out of scope for
        # this pass (task.md follow-up item).

        mw.caption_text_edit.setVisible(text_ui)
        # task_combo（Florence-2 の詳細度）はローカル captioner のときだけ（VLM では無関係）。
        mw.task_combo.setVisible(model_is_captioner and not use_vlm)
        # 生成テキストの挿入位置（前に追加 / 後に追加 / 上書き）はテキスト UI 共通。
        if hasattr(mw, "caption_placement_widget"):
            mw.caption_placement_widget.setVisible(text_ui)
        if hasattr(mw, "vlm_settings_button"):
            mw.vlm_settings_button.setVisible(use_vlm)
            mw.vlm_single_test_button.setVisible(use_vlm)
        if hasattr(mw, "use_vlm_check") and mw.use_vlm_check.isChecked() != use_vlm:
            mw.use_vlm_check.blockSignals(True)
            mw.use_vlm_check.setChecked(use_vlm)
            mw.use_vlm_check.blockSignals(False)

        # tagger_config_block が消えた縦領域をキャプション編集欄へ回す。モードが実際に
        # 切り替わったときだけ実行し、利用者が手でドラッグしたサイズを毎回踏み潰さない。
        if hasattr(mw, "_rebalance_right_splitter"):
            mw._rebalance_right_splitter(text_ui)

        # tag_panel 末尾のばねを、text_ui のときは殺す（caption_text_edit を縦いっぱいへ）、
        # tagger のときは効かせる（タグ関連ウィジェットを上詰めにし下端に余白を残す）。
        if hasattr(mw, "_tag_panel_layout"):
            mw._tag_panel_layout.setStretch(mw._tag_panel_bottom_spacer_index, 0 if text_ui else 1)
            mw._tag_panel_layout.activate()

        # 既存ファイルの扱いは captioner でも4モードすべて選べる（existing_mode_combo の
        # APPEND 項目は ui_main_window.py で作成時から常に有効で、無効化している箇所は
        # どこにも無い）。「そのファイルに書くか」を決めるだけで、実際の組み合わせ方は
        # caption placement が担当するため（combine_caption）、特別扱いは不要。

        if model_is_captioner:
            # Default to the model's verbose task (MORE_DETAILED_CAPTION) on every switch
            # to a captioner - the shorter caption levels are rarely what's wanted
            # (2026-08-31 user decision). The user can still pick another level per session.
            default_task = config_mapping(entry.config, "captioner").get("default_task", "MORE_DETAILED_CAPTION")
            if mw.settings.caption.task != default_task:
                mw.settings.caption.task = default_task
                mw.save_current_config()
            idx = mw.task_combo.findData(default_task)
            if idx >= 0:
                mw.task_combo.setCurrentIndex(idx)
        if not text_ui:
            # reload_tags_only() early-returns in text mode, so _all_tags is empty for the
            # whole text session. Repopulate it (and the grid tag cache) when the tag panel
            # is showing again, or the bulk-delete panel stays empty.
            mw.reload_tags_only()

        # Reload whichever display (tag buttons or caption text) matches the new mode
        # for the currently-selected image.
        current_item = mw.image_list.currentItem()
        if current_item:
            image_path = Path(mw.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
            mw._load_image_tags(image_path)
