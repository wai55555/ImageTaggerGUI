"""スライダーと AppSettings の同期、touched 管理、設定の保存。REQ-6.3 / Phase 6 で
main_window.py から切り出した。既定値の決定は ModelModeController が持ち、
ここはウィジェットへの反映と保存だけを担当する（挙動は変えない）。"""
from __future__ import annotations

import sys
from typing import Any, Mapping

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QGridLayout, QLabel, QSlider

import app_settings
from app_settings import save_config
from custom_dialogs import CategoryTagSettingsDialog
from model_registry import config_mapping
from utils import write_debug_log


def _ui(name: str):
    """Qt 部品は main_window のモジュール属性として遅延解決する（テストの差し替えを保つ）。"""
    return getattr(sys.modules["main_window"], name)


class SettingsBinding:
    """MainWindow への参照を持ち、切り出した責務だけを担当する（REQ-6.7）。"""

    def __init__(self, mw) -> None:
        self.mw = mw

    def save_current_config(self):
        """Updates the settings object with the current UI state and saves it to file."""
        write_debug_log(self.mw.locale_manager.get_string("MainWindow", "Saving_UI_Settings"))

        # Only update geometry if in main view and not maximized/minimized
        if self.mw.central_widget.currentWidget() == self.mw.main_view_widget and not self.mw.isMaximized() and not self.mw.isMinimized():
            self.mw.settings.window.geometry = f"{self.mw.width()}x{self.mw.height()}+{self.mw.x()}+{self.mw.y()}"

        save_config(self.mw.settings)

    def create_slider_group(self, layout: QGridLayout, section: str, min_val: float, max_val: float, step: float, keys: Mapping[str, int]):
        """Creates a labeled slider and connects it to the settings object."""
        resolution = 1 / step if step < 1 else 1
        
        for key, row_index in keys.items():
            is_float = step < 1
            label_suffix = self.mw.locale_manager.get_string("MainWindow", "Threshold_Suffix") if section == 'Thresholds' else self.mw.locale_manager.get_string("MainWindow", "Max_Count_Suffix")
            # [CategoryDialog] Category_<key> は「詳細」ダイアログ向けに
            # "General（一般）" のような訳語併記になっているため、しきい値ラベルの
            # 頭に付けると "General（一般） しきい値:" と冗長になる。ここは
            # 短いカテゴリ名のまま出す。
            label = QLabel(f"{key.capitalize()} {label_suffix}:")
            
            settings_section = getattr(self.mw.settings, section.lower())
            initial_val = getattr(settings_section, key)
                
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * resolution), int(max_val * resolution))
            slider.setValue(int(initial_val * resolution))
            
            value_label = QLabel(f"{initial_val:.2f}" if is_float else str(int(initial_val)))
            value_label.setFixedWidth(50)
            self.mw._sliders[f"{section.lower()}_{key}"] = (slider, value_label)

            def update_value(value: int, k: str = key, s: Any = settings_section, res: float = resolution, v_label: QLabel = value_label, is_flt: bool = is_float):
                real_val = value / res
                v_label.setText(f"{real_val:.2f}" if is_flt else str(int(real_val)))
                setattr(s, k, real_val if is_flt else int(real_val))
                if not self.mw._suppress_slider_touch:
                    # A hand move marks the category "touched" so a later model switch
                    # won't overwrite it with that model's default (see _apply_threshold_defaults).
                    s.touched = app_settings.add_touched(s.touched, k)
                # REQ-5.5 / R7-6: つまみ 1 目盛りごとに全設定を書かない。保存は
                # ドラッグ完了後にまとめる（プログラム的同期ではそもそも呼ばない）。
                if not self.mw._suppress_slider_save:
                    self.mw._save_config_soon()

            slider.valueChanged.connect(update_value)
            # ドラッグ完了時（キーボード操作は valueChanged 側）に確定保存する。
            slider.sliderReleased.connect(self.mw.save_current_config)

            layout.addWidget(label, row_index, 0)
            layout.addWidget(slider, row_index, 1)
            layout.addWidget(value_label, row_index, 2)

    def _save_config_soon(self) -> None:
        """スライダー操作の保存をまとめる（既定 400ms。REQ-5.5 / R7-6）。"""
        if not hasattr(self.mw, "_config_save_timer"):
            self.mw._config_save_timer = _ui("QTimer")(self.mw)
            self.mw._config_save_timer.setSingleShot(True)
            self.mw._config_save_timer.setInterval(400)
            self.mw._config_save_timer.timeout.connect(self.mw.save_current_config)
        self.mw._config_save_timer.start()

    def _sync_settings_sliders(self):
        """Refresh the four main-window sliders (general/character threshold & max) from
        self.mw.settings without their valueChanged handler flagging them as user-touched."""
        self.mw._suppress_slider_touch = True
        self.mw._suppress_slider_save = True
        try:
            for key, (slider, value_label) in self.mw._sliders.items():
                section_name, cat = key.split('_', 1)
                value = getattr(getattr(self.mw.settings, section_name), cat)
                is_float = section_name == 'thresholds'
                res = 100.0 if is_float else 1.0
                slider.setValue(int(round(value * res)))
                value_label.setText(f"{value:.2f}" if is_float else str(int(value)))
        finally:
            self.mw._suppress_slider_touch = False
            self.mw._suppress_slider_save = False

    @Slot()
    def _open_category_settings_dialog(self):
        """Opens the per-category threshold / max-tag dialog for the selected model."""
        if self.mw._category_settings_dialog is not None and self.mw._category_settings_dialog.isVisible():
            self.mw._category_settings_dialog.raise_()
            self.mw._category_settings_dialog.activateWindow()
            return

        def on_changed():
            # The dialog has already written into self.mw.settings; mirror the main sliders,
            # persist, and refresh any open tag view that depends on the values.
            self.mw._sync_settings_sliders()
            self.mw.save_current_config()

        def on_reset():
            # Drop "touched" for this model's categories, then re-apply its defaults.
            model_cats = set(self.mw._current_model_categories())
            for section in (self.mw.settings.thresholds, self.mw.settings.limits):
                kept = app_settings.parse_touched(section.touched) - model_cats
                section.touched = ",".join(sorted(kept))
            self.mw._model_mode._apply_threshold_defaults(self.mw._current_model_entry())
            self.mw._sync_settings_sliders()
            self.mw.save_current_config()
            if self.mw._category_settings_dialog is not None:
                self.mw._category_settings_dialog.reload_from_settings()

        self.mw._category_settings_dialog = CategoryTagSettingsDialog(
            self.mw,
            get_string=self.mw.locale_manager.get_string,
            categories=self.mw._current_model_categories(),
            thresholds=self.mw.settings.thresholds,
            limits=self.mw.settings.limits,
            on_changed=on_changed,
            on_reset=on_reset,
        )
        self.mw._category_settings_dialog.finished.connect(lambda _: setattr(self.mw, "_category_settings_dialog", None))
        self.mw._category_settings_dialog.show()

    def _current_model_categories(self) -> list[str]:
        """Tag categories the selected model produces (ui.categories in its config),
        filtered to known names and returned in the canonical TAG_CATEGORY_NAMES order
        so the per-category dialog rows are always consistent."""
        cats = config_mapping(self.mw._current_model_entry().config, "ui").get("categories") or ["general", "character"]
        cats_set = set(cats)
        return [c for c in app_settings.TAG_CATEGORY_NAMES if c in cats_set]
