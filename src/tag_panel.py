"""タグ表示・編集パネル（ページング表示、単体タグ追加/削除、翻訳マップの反映、
ホバーハイライト用キャッシュ）。REQ-6.4 / Phase 6 で main_window.py から切り出した。
ロック判定の正本は MainWindow._is_processing() のまま（このクラスは状態を持たない）。"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QApplication, QMessageBox

import constants
import tag_utils
from undo_manager import AddTagsAction, RemoveTagAction
from utils import write_debug_log


def _ui(name: str):
    """Qt 部品は main_window のモジュール属性として遅延解決する（テストの差し替えを保つ）。"""
    return getattr(sys.modules["main_window"], name)


class TagPanel:
    """MainWindow への参照を持ち、切り出した責務だけを担当する（REQ-6.7）。"""

    def __init__(self, mw) -> None:
        self.mw = mw

    def display_current_tag_page(self):
        """Displays the current page of bulk tags."""
        for button in self.mw.tag_buttons:
            button.deleteLater()
        self.mw.tag_buttons.clear()

        total_tags = len(self.mw._all_tags)
        start_index = self.mw._current_page * constants.TAGS_PER_PAGE
        end_index = min(start_index + constants.TAGS_PER_PAGE, total_tags)
        
        if total_tags == 0:
            self.mw.loading_label.setText(self.mw.locale_manager.get_string("MainWindow", "Tag_File_Txt_Not_Found"))
        else:
            self.mw.loading_label.setText(self.mw.locale_manager.get_string("MainWindow", "Displaying_Tags_Count_And_Click_Delete", total_tags=total_tags, start_index=start_index + 1, end_index=end_index))
            
            current_page_tags = self.mw._all_tags[start_index:end_index]
            
            # 翻訳リストのインデックスを取得
            lang_index = self.mw._get_translation_index(self.mw._tag_display_language)

            for i, (tag_name, count) in enumerate(current_page_tags):
                display_text = tag_name
                
                # 英語以外かつ、辞書にタグが存在する場合
                if lang_index != -1:
                    try:
                        # 指定した言語の翻訳を取得（REQ-H5: `_`->' ' 正規化キーへ
                        # フォールバックし、アンダースコア変換オフでも引けるようにする）
                        translations = tag_utils.get_tag_translations(
                            self.mw.tag_translation_map, tag_name)
                        if translations is not None and len(translations) > lang_index:
                            display_text = translations[lang_index]
                    except IndexError:
                        pass # エラー時は英語のまま

                button = _ui("QPushButton")(f"{display_text} ({count})")
                # ... (以下のコードはそのまま) ...
                button.setMinimumWidth(self.mw._tag_button_min_width)
                button.setFixedHeight(self.mw._tag_button_min_height)
                button.setToolTip(tag_name) # Tooltip always shows English tag
                button.setProperty("original_tag", tag_name)
                button.clicked.connect(functools.partial(self.mw.delete_tag_all, tag_name))
                button.installEventFilter(self.mw)
                self.mw.tag_button_grid.addWidget(button, i // 4, i % 4)
                self.mw.tag_buttons.append(button)

        self.mw.prev_page_btn.setEnabled(self.mw._current_page > 0)
        self.mw.next_page_btn.setEnabled(end_index < total_tags)
        # REQ-2.4: 表示更新で錠を開けない。解除は処理の終了経路（_set_main_controls_enabled）
        # だけが行う（R7-4。以前はここで無条件に True にしていた）。
        self.mw._set_bulk_controls_enabled(self.mw._controls_enabled)
        _ui("QTimer").singleShot(0, self.mw.update_all_button_alignments)

    def _display_image_tag_page(self):
        """Displays the current page of tags for the selected image."""
        for button in self.mw.tag_buttons_for_image:
            button.deleteLater()
        self.mw.tag_buttons_for_image.clear()

        cols, _rows = self.mw._tag_display_grid_dims()
        tags_per_page = self.mw._get_image_tags_per_page()

        total_tags = len(self.mw._current_image_tags)
        start = self.mw._current_image_tag_page * tags_per_page
        end = min(start + tags_per_page, total_tags)
        
        lang_index = self.mw._get_translation_index(self.mw._tag_display_language)

        for i, tag_name in enumerate(self.mw._current_image_tags[start:end]):
            display_text = tag_name
            
            # 翻訳処理
            if lang_index != -1:
                try:
                    # REQ-H5: 生タグで外れたら正規化キーでも引く
                    translations = tag_utils.get_tag_translations(
                        self.mw.tag_translation_map, tag_name)
                    if translations is not None and len(translations) > lang_index:
                        display_text = translations[lang_index]
                except IndexError:
                    pass

            button = _ui("QPushButton")(display_text)
            button.setMinimumSize(self.mw._tag_button_min_width, self.mw._tag_button_min_height)
            # 生成時点の錠状態を引き継ぐ（実行中に再描画されてもチップが書けない）。
            button.setEnabled(self.mw._controls_enabled)
            button.setToolTip(tag_name) # Tooltip always shows English tag
            button.setProperty("original_tag", tag_name)
            button.clicked.connect(functools.partial(self.mw._delete_image_tag, tag_name))
            button.installEventFilter(self.mw)
            self.mw.tag_display_grid.addWidget(button, i // cols, i % cols)
            self.mw.tag_buttons_for_image.append(button)

        self.mw.image_tag_prev_page_btn.setEnabled(self.mw._current_image_tag_page > 0)
        self.mw.image_tag_next_page_btn.setEnabled(end < total_tags)
        _ui("QTimer").singleShot(0, self.mw.update_all_button_alignments)

    @Slot(int)
    def toggle_tag_language(self, index: int):
        """Toggles the display language of tags between English and Japanese."""
        self.mw._tag_display_language = self.mw.language_combo.currentText()
        self.mw.display_current_tag_page()
        self.mw._display_image_tag_page()
        self.mw.grid_view_widget.set_tag_display_language(self.mw._tag_display_language, self.mw.tag_translation_map)

    def _get_image_tags_per_page(self) -> int:
        """Returns the number of tags to display per page for the current image."""
        cols, rows = self.mw._tag_display_grid_dims()
        return max(1, cols * rows)

    def _get_total_image_tag_pages(self) -> int:
        """Returns the total number of tag pages for the current image."""
        tags_per_page = self.mw._get_image_tags_per_page()
        return max(1, (len(self.mw._current_image_tags) + tags_per_page - 1) // tags_per_page)

    def _get_translation_index(self, language: str) -> int:
        """言語名からリストのインデックスを返す"""
        lang_map = {
            "日本語": 0,
            "Français": 1,
            "Deutsch": 2,
            "Español": 3,
            "Русский": 4,
            "简体中文": 5,
            "繁體中文": 6,
            "한국어": 7
        }
        return lang_map.get(language, -1)

    def _tag_display_grid_dims(self) -> tuple[int, int]:
        """タグ表示の (cols, rows)。config.ini の不正値（0 以下）でも落ちない（R28-1）。

        読み側の意味は変えない（正の値はそのまま）。0 以下のときだけ既定へ倒す。
        """
        window = self.mw.settings.window
        cols = window.tag_display_cols if isinstance(window.tag_display_cols, int) else 0
        rows = window.tag_display_rows if isinstance(window.tag_display_rows, int) else 0
        if cols <= 0:
            window.tag_display_cols = 5
        if rows <= 0:
            window.tag_display_rows = 6
        return max(1, window.tag_display_cols), max(1, window.tag_display_rows)

    def _update_bulk_tag_buttons(self, all_tags: list[tuple[str, int]]):
        """Updates the bulk tag deletion buttons with new tag data."""
        if self.mw.loading_timer:
            self.mw.loading_timer.stop()
        # REQ-2.4: ローダ完了は「読む側」の完了であって、書く側の錠を解く理由ではない。
        # ここで _is_bulk_deleting を落とすと、一括処理中でもガードが外れる（R12-6）。
        self.mw._all_tags = all_tags

        saved_page = getattr(self.mw, '_saved_bulk_page', None)
        if saved_page is not None:
            total_pages = max(1, (len(all_tags) + constants.TAGS_PER_PAGE - 1) // constants.TAGS_PER_PAGE)
            self.mw._current_page = min(saved_page, total_pages - 1)
        else:
            self.mw._current_page = 0
        self.mw._saved_bulk_page = None

        self.mw.display_current_tag_page()
        self.mw._build_tag_cache()

    @Slot()
    def _add_single_tag(self):
        """Adds a new tag to the currently selected image's tag file."""
        # REQ-2.3 / R25-1: 入力欄とボタンは錠で閉じるが、ハンドラ側でも同じ判定で
        # 止める（片方だけに依存しない）。
        if self.mw._is_processing():
            write_debug_log("_add_single_tag ignored while processing")
            return
        current_item = self.mw.image_list.currentItem()
        if not current_item:
            return

        tags_raw = self.mw.add_single_tag_line.text().strip()
        if not tags_raw:
            return

        # REQ-H2: whitespace normalization is shared with the grid and bulk add so
        # "long  hair" cannot become a different tag depending on the entry point.
        new_tags = tag_utils.parse_tag_input(tags_raw)
        if not new_tags:
            return

        image_path = Path(self.mw.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
        txt_path = image_path.with_suffix('.txt')

        try:
            existing_tags = []
            if txt_path.is_file():
                existing_tags = [t.strip() for t in txt_path.read_text('utf-8').split(',') if t.strip()]
            
            # Filter out tags that already exist (REQ-H1: case-insensitive, like
            # tagging_core.merge_tags, so `Solo` is not added next to `solo`).
            tags_to_add = tag_utils.filter_new_tags(existing_tags, new_tags)
            
            if not tags_to_add:
                self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Tag_Already_Exists"), "orange")
                return
            
            for tag in tags_to_add:
                existing_tags.append(tag)
            
            _ui("atomic_write_text")(txt_path, ', '.join(existing_tags), fsync=False, verify=False)
            
            # Record action for undo
            action = AddTagsAction(file_path=txt_path, added_tags=tags_to_add)
            self.mw.undo_manager.push(action)
            self.mw._update_undo_redo_buttons()
            
            self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Tags_Added_To_File", txt_path_name=txt_path.name), "green")
            self.mw.add_single_tag_line.clear()
            self.mw._load_image_tags(image_path, preserve_page=True)
            rel_path = current_item.data(Qt.ItemDataRole.UserRole + 1)
            self.mw._update_tag_cache_entry(rel_path)
            self.mw.reload_tags_only()
        except Exception as e:
            self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Error_Adding_Tags", txt_path_name=txt_path.name, e=e), "red")

    def _delete_image_tag(self, tag_to_delete: str):
        """Deletes a tag from the currently selected image's tag file."""
        # REQ-2.3 / R19-6: 表示中のタグチップからの削除経路も処理中は通さない。
        if self.mw._is_processing():
            write_debug_log(f"_delete_image_tag ignored while processing: {tag_to_delete}")
            return
        write_debug_log(f"[_delete_image_tag] Attempting to delete tag: {tag_to_delete}")
        current_item = self.mw.image_list.currentItem()
        if not current_item:
            write_debug_log("[_delete_image_tag] No image item selected.")
            return

        image_path = Path(self.mw.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
        txt_path = image_path.with_suffix('.txt')
        
        write_debug_log(f"[_delete_image_tag] Image path: {image_path}, TXT path: {txt_path}")

        if not txt_path.is_file():
            write_debug_log(f"[_delete_image_tag] TXT file not found: {txt_path}")
            return
        
        try:
            tags = [t.strip() for t in txt_path.read_text('utf-8').split(',') if t.strip()]
            write_debug_log(f"[_delete_image_tag] Existing tags before deletion: {tags}")

            if tag_to_delete in tags:
                # Record original index for undo
                original_index = tags.index(tag_to_delete)
                
                tags.remove(tag_to_delete)
                write_debug_log(f"[_delete_image_tag] Tags after removal: {tags}")
                _ui("atomic_write_text")(txt_path, ', '.join(tags), fsync=False, verify=False)
                write_debug_log(f"[_delete_image_tag] Successfully wrote tags to file: {txt_path.name}")
                
                # Record action for undo
                action = RemoveTagAction(file_path=txt_path, removed_tag=tag_to_delete, original_index=original_index)
                self.mw.undo_manager.push(action)
                self.mw._update_undo_redo_buttons()
                
                self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Tag_Deleted_From_File", tag_name=tag_to_delete, file_name=txt_path.name), "green")
                self.mw._load_image_tags(image_path, preserve_page=True)
                rel_path = current_item.data(Qt.ItemDataRole.UserRole + 1)
                self.mw._update_tag_cache_entry(rel_path)
                write_debug_log("[_delete_image_tag] UI updated after tag deletion.")
            else:
                write_debug_log(f"[_delete_image_tag] Tag '{tag_to_delete}' not found in file: {txt_path}")
        except Exception as e:
            write_debug_log(f"[_delete_image_tag] Error during tag deletion: {e}")
            self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Error_Deleting_Tag", tag_name=tag_to_delete, file_name=txt_path.name, e=e), "red")

    def add_tag_all(self, prepend: bool):
        # REQ-2.3: 一括追加も .txt を書く経路。処理中は開始できない（自前の
        # _is_bulk_deleting ではなく一本化した判定を見る）。
        if self.mw._is_processing():
            write_debug_log("add_tag_all ignored while processing")
            return
        tags_to_add = (self.mw.add_tag_line if prepend else self.mw.add_tag_line_append).text().strip()
        if not tags_to_add:
            return
        
        if QMessageBox.question(self.mw, self.mw.locale_manager.get_string("MainWindow", "Bulk_Add_Confirmation"), self.mw.locale_manager.get_string("MainWindow", "Confirm_Bulk_Add_Tag", tags_to_add=tags_to_add)) == QMessageBox.StandardButton.Yes:
            self.mw._start_bulk_tag_worker('add', input_dir=Path(self.mw.settings.paths.input_dir), tags=tags_to_add, prepend=prepend)

    def delete_tag_all(self, tag_to_delete: str):
        """Starts a bulk process to delete a tag from all .txt files."""
        if self.mw._is_processing():
            write_debug_log("delete_tag_all ignored while processing")
            return
        if QMessageBox.question(self.mw, self.mw.locale_manager.get_string("MainWindow", "Bulk_Delete_Confirmation"), self.mw.locale_manager.get_string("MainWindow", "Confirm_Bulk_Delete_Tag", tag_to_delete=tag_to_delete)) == QMessageBox.StandardButton.Yes:
            self.mw._start_bulk_tag_worker('delete', input_dir=Path(self.mw.settings.paths.input_dir), tag=tag_to_delete)

    def _highlight_files_for_tag(self, tag_name: str) -> None:
        """指定タグを持つファイルをTagListWidgetでハイライトする。選択中アイテムはスキップする。"""
        from PySide6.QtGui import QBrush, QColor
        color = QColor("#4a5a2a") if self.mw._is_dark_theme else QColor("#c8f0a0")
        brush = QBrush(color)
        current_item = self.mw.image_list.currentItem()
        for i in range(self.mw.image_list.count()):
            item = self.mw.image_list.item(i)
            if item is None or item is current_item:
                continue
            rel_path = item.data(Qt.ItemDataRole.UserRole + 1)
            tags = self.mw._tag_cache.get(rel_path, set())
            if tag_name in tags:
                item.setBackground(brush)

    def _clear_highlight(self) -> None:
        """TagListWidgetの全アイテムのハイライトを解除する。"""
        from PySide6.QtGui import QBrush
        default_brush = QBrush()
        for i in range(self.mw.image_list.count()):
            item = self.mw.image_list.item(i)
            if item is not None:
                item.setBackground(default_brush)

    def update_button_text_alignment(self, button):
        # write_debug_log(f"DEBUG: update_button_text_alignment - button text: {button.text()}")
        font_metrics = button.fontMetrics()
        text_width = font_metrics.horizontalAdvance(button.text())
        
        button_text_space = button.contentsRect().width()

        if text_width > button_text_space:
            button.setStyleSheet("QPushButton { text-align: left; padding-left: 5px; }")
        else:
            button.setStyleSheet("QPushButton { text-align: center; }")

    def update_all_button_alignments(self):
        write_debug_log("DEBUG: update_all_button_alignments called.")
        for button in self.mw.tag_buttons:
            if button.isVisible():
                self.mw.update_button_text_alignment(button)
        for button in self.mw.tag_buttons_for_image:
            if button.isVisible():
                self.mw.update_button_text_alignment(button)

    def _copy_tag_to_clipboard(self, tag_name: str):
        """Copies the tag with a trailing comma to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(tag_name + ", ")
        self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Tag_Copied_To_Clipboard", tag_name=tag_name), "blue")
        write_debug_log(f"Tag copied to clipboard: {tag_name}")
