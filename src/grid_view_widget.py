from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGridLayout, QSizePolicy, QScrollArea, QFrame, QToolTip,
    QApplication, QTextEdit
)
from PySide6.QtCore import Qt, Signal, Slot, QRect, QObject, QEvent, QTimer
from PySide6.QtGui import QPixmap, QWheelEvent, QResizeEvent

import tag_utils
from utils import write_debug_log
from locale_manager import LocaleManager
from app_settings import AppSettings
from custom_dialogs import ClickableLabel, ImageViewerDialog

TAGS_PER_PAGE_GRID = 14


def filter_images_by_tag(
    image_paths: list[Path],
    tag_cache: dict[str, set[str]],
    search_text: str,
    base_dir: Path,
) -> list[Path]:
    """タグキャッシュを使って、search_text を部分一致で含むタグを持つ画像のみを返す。

    - search_text が空の場合は image_paths をそのまま返す
    - カンマ区切りで複数キーワードを指定した場合は AND 検索
    - 大文字・小文字を区別しない
    - tag_cache のキーは相対パス、image_paths は絶対パスなので base_dir で変換する
    """
    if not search_text:
        return list(image_paths)
    needles = [t.strip().lower() for t in search_text.split(",") if t.strip()]
    if not needles:
        return list(image_paths)
    result: list[Path] = []
    for img_path in image_paths:
        try:
            rel_key = str(img_path.relative_to(base_dir))
        except ValueError:
            continue
        tags = tag_cache.get(rel_key, set())
        tags_lower = {t.lower() for t in tags}
        # AND 検索: 全キーワードがいずれかのタグに部分一致する場合のみ含める
        if all(any(needle in tag for tag in tags_lower) for needle in needles):
            result.append(img_path)
    return result


class ImageEditCellWidget(QWidget):
    # --- ADDED: Signal to request image enlargement with its global index ---
    image_enlarge_requested = Signal(int)
    # Signals for undo/redo
    tags_added = Signal(Path, list)  # (file_path, added_tags)
    tag_removed = Signal(Path, str, int)  # (file_path, removed_tag, original_index)
    caption_edited = Signal(Path, str, str, bool)  # (txt_path, old_text, new_text, file_existed_before)
    caption_save_failed = Signal(str)        # error text - surfaced in the main log
    # Signals for tag hover highlight
    tag_hovered = Signal(str)       # ホバー開始: タグ名を渡す
    tag_hover_cleared = Signal()    # ホバー解除

    def __init__(self, locale_manager: LocaleManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.locale_manager = locale_manager
        self._image_path: Path | None = None
        self.tag_buttons: List[QPushButton] = []
        self._current_tag_page = 0
        self._global_index = -1  # To store the image's index in the main list

        self.tag_translation_map: dict[str, list[str]] = {}
        self._tag_display_language: str = "English"
        self._search_text: str = ""
        self._caption_mode: bool = False
        self._caption_loaded_text: str = ""

        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        self.image_label = ClickableLabel()
        self.image_label.setText(self.locale_manager.get_string("GridView", "No_Image"))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setStyleSheet("border: 1px solid grey;")
        # --- MODIFIED: Connect to a slot that emits the new signal ---
        self.image_label.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.image_label, 1)

        # ... (rest of the __init__ method is unchanged) ...
        tag_area_widget = QWidget()
        tag_layout = QVBoxLayout(tag_area_widget)
        tag_layout.setContentsMargins(0, 0, 0, 0)
        tag_layout.setSpacing(2)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.tag_grid_layout = QGridLayout(scroll_widget)
        self.tag_grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        for i in range(2): self.tag_grid_layout.setColumnStretch(i, 1)
        for i in range(7): self.tag_grid_layout.setRowStretch(i, 1)
        scroll_area.setWidget(scroll_widget)
        tag_layout.addWidget(scroll_area, 1)
        tag_pagination_layout = QHBoxLayout()
        self.prev_tag_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Previous_Page"))
        self.prev_tag_page_btn.clicked.connect(self._prev_tag_page)
        self.next_tag_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Next_Page"))
        self.next_tag_page_btn.clicked.connect(self._next_tag_page)
        tag_pagination_layout.addWidget(self.prev_tag_page_btn)
        tag_pagination_layout.addWidget(self.next_tag_page_btn)
        tag_layout.addLayout(tag_pagination_layout)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        tag_layout.addWidget(separator)
        add_tag_layout = QHBoxLayout()
        self.add_tag_line = QLineEdit()
        self.add_tag_line.setPlaceholderText(self.locale_manager.get_string("GridView", "Add_Tags_Placeholder"))
        self.add_tag_line.returnPressed.connect(self._add_tag)
        add_button = QPushButton(self.locale_manager.get_string("GridView", "Add_Button"))
        add_button.clicked.connect(self._add_tag)
        add_tag_layout.addWidget(self.add_tag_line)
        add_tag_layout.addWidget(add_button)
        tag_layout.addLayout(add_tag_layout)
        self._tag_area_widget = tag_area_widget
        layout.addWidget(tag_area_widget, 1)

        # Captioner-mode counterpart of the tag area: a single free-text editor bound to
        # the image's .txt file, saved on focus-out. Hidden unless set_caption_mode(True).
        self.caption_edit = QTextEdit()
        self.caption_edit.setPlaceholderText(
            self.locale_manager.get_string("GridView", "Caption_Placeholder"))
        self.caption_edit.installEventFilter(self)
        self.caption_edit.hide()
        layout.addWidget(self.caption_edit, 1)
        # Autosave ~1.2s after the last keystroke (mirrors MainWindow's caption editor),
        # so the edit commits and Undo becomes available without waiting for focus-out.
        self._caption_save_timer = QTimer(self)
        self._caption_save_timer.setSingleShot(True)
        self._caption_save_timer.setInterval(1200)
        self._caption_save_timer.timeout.connect(self._save_caption)
        self.caption_edit.textChanged.connect(self._caption_save_timer.start)

    # --- ADDED: Slot to handle the double click and emit the request ---
    @Slot()
    def _on_double_click(self):
        if self._global_index != -1:
            self.image_enlarge_requested.emit(self._global_index)

    # --- MODIFIED: load_data now accepts the global index ---
    def load_data(self, image_path: Path, global_index: int):
        prev_path = self._image_path
        if self._caption_mode:
            # Flush a pending debounced edit for the outgoing image before switching.
            if prev_path is not None and prev_path != image_path:
                self._save_caption()
            # Reload the .txt unless the user has unsaved edits for this same image.
            # Skipping only while "dirty" protects text being typed during a re-render
            # (resize, page refresh) while still picking up external changes - notably
            # Undo/Redo, which rewrites the file and then refreshes the page. Keeping the
            # stale editor there would let the next autosave write the undone text back.
            is_dirty = self.caption_edit.toPlainText() != self._caption_loaded_text
            self._image_path = image_path
            self._global_index = global_index
            self._rescale_image()
            if prev_path != image_path:
                self._current_tag_page = 0
                self._load_caption_text()
            elif not is_dirty:
                self._load_caption_text()
            return
        self._image_path = image_path
        self._global_index = global_index
        # 同じ画像の場合はタグページを保持、異なる画像の場合のみリセット
        if prev_path != image_path:
            self._current_tag_page = 0
        self._update_tag_display()
    
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

    # ... (rest of the class is unchanged) ...
    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self._caption_mode:
            # Only rescale the image - never reload the .txt here, or a caption still in
            # the autosave debounce window would be discarded on every resize.
            self._rescale_image()
        else:
            self._update_tag_display()
    def _rescale_image(self):
        if self._image_path and self.image_label.width() > 0 and self.image_label.height() > 0:
            pixmap = QPixmap(str(self._image_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                              Qt.AspectRatioMode.KeepAspectRatio,
                                              Qt.TransformationMode.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

    def _load_caption_text(self):
        """Captioner mode: load the image's .txt as free text into caption_edit."""
        self._caption_save_timer.stop()
        if not self._image_path:
            self.caption_edit.blockSignals(True)
            self.caption_edit.clear()
            self.caption_edit.blockSignals(False)
            self._caption_loaded_text = ""
            return
        txt_path = tag_utils.get_txt_path(self._image_path)
        text = ""
        try:
            if txt_path.is_file():
                text = txt_path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        self.caption_edit.blockSignals(True)
        self.caption_edit.setPlainText(text)
        self.caption_edit.blockSignals(False)
        self._caption_loaded_text = text

    def _save_caption(self):
        """Captioner mode: write caption_edit back to the .txt (focus-out or autosave); emit for undo."""
        self._caption_save_timer.stop()
        if not self._caption_mode or not self._image_path:
            return
        txt_path = tag_utils.get_txt_path(self._image_path)
        new_text = self.caption_edit.toPlainText()
        existed_before = txt_path.is_file()
        old_text = ""
        try:
            if existed_before:
                old_text = txt_path.read_text(encoding="utf-8")
        except Exception:
            old_text = ""
        if new_text == old_text:
            return
        try:
            txt_path.write_text(new_text, encoding="utf-8")
        except Exception as e:
            write_debug_log(f"grid caption save failed for {txt_path.name}: {type(e).__name__}: {e}")
            self.caption_save_failed.emit(f"{txt_path.name}: {type(e).__name__}: {e}")
            return
        self._caption_loaded_text = new_text
        self.caption_edited.emit(txt_path, old_text, new_text, existed_before)

    def set_caption_mode(self, on: bool):
        # Flush a pending caption edit while the save path is still active.
        if self._caption_mode and not on:
            self._save_caption()
        self._caption_mode = on
        self._tag_area_widget.setVisible(not on)
        self.caption_edit.setVisible(on)
        self._update_tag_display()

    def _update_tag_display(self):
        self._rescale_image()
        if self._caption_mode:
            self._load_caption_text()
            return
        for btn in self.tag_buttons:
            btn.deleteLater()
        self.tag_buttons.clear()
        if not self._image_path:
            self.prev_tag_page_btn.setEnabled(False)
            self.next_tag_page_btn.setEnabled(False)
            return
        txt_path = tag_utils.get_txt_path(self._image_path)
        tags = tag_utils.read_tags(txt_path)
        total_tags = len(tags)
        # ページ範囲チェック（タグ削除等でページが範囲外になった場合の補正）
        total_pages = max(1, (total_tags + TAGS_PER_PAGE_GRID - 1) // TAGS_PER_PAGE_GRID)
        if self._current_tag_page >= total_pages:
            self._current_tag_page = max(0, total_pages - 1)
        start_index = self._current_tag_page * TAGS_PER_PAGE_GRID
        end_index = min(start_index + TAGS_PER_PAGE_GRID, total_tags)
        current_page_tags = tags[start_index:end_index]
        
        # 翻訳リストのインデックスを取得
        lang_index = self._get_translation_index(self._tag_display_language)
        
        for i, tag in enumerate(current_page_tags):
            display_text = tag
            
            # 英語以外かつ、辞書にタグが存在する場合
            if lang_index != -1 and tag in self.tag_translation_map:
                try:
                    translations = self.tag_translation_map[tag]
                    if len(translations) > lang_index:
                        display_text = translations[lang_index]
                except IndexError:
                    pass # エラー時は英語のまま
            
            btn = QPushButton(display_text)
            btn.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            # ハイライト判定: 検索文字列が非空かつタグ名に部分一致する場合（AND検索の各キーワードで判定）
            needles = [t.strip().lower() for t in self._search_text.split(",") if t.strip()]
            style = "font-size: 11pt; text-align: left; padding-left: 3px;"
            if needles and any(n in tag.lower() for n in needles):
                is_dark = self.palette().window().color().lightness() < 128
                hl_color = "#4a5a2a" if is_dark else "#c8f0a0"
                style += f" background-color: {hl_color};"
            btn.setStyleSheet(style)
            btn.setToolTip(tag) # Tooltip always shows English tag
            btn.setProperty("original_tag", tag)
            btn.clicked.connect(lambda checked=False, t=tag: self._remove_tag(t))
            btn.installEventFilter(self)
            row = i % 7; col = i // 7
            self.tag_grid_layout.addWidget(btn, row, col)
            self.tag_buttons.append(btn)
        self.prev_tag_page_btn.setEnabled(self._current_tag_page > 0)
        self.next_tag_page_btn.setEnabled(end_index < total_tags)
    @Slot()
    def _add_tag(self):
        if not self._image_path:
            return
        tags_to_add_str = self.add_tag_line.text().strip()
        if not tags_to_add_str: return
        tags_to_add = [t.strip() for t in tags_to_add_str.split(',') if t.strip()]
        if not tags_to_add: return
        txt_path = tag_utils.get_txt_path(self._image_path)
        existing_tags = tag_utils.read_tags(txt_path)
        actually_new_tags: list[str] = []; any_duplicates = False
        for tag in tags_to_add:
            if tag in existing_tags: any_duplicates = True
            else:
                if tag not in actually_new_tags: actually_new_tags.append(tag)
        self.add_tag_line.clear()
        if any_duplicates:
            tooltip_message = self.locale_manager.get_string("GridView", "Tooltip_Duplication")
            tooltip_pos = self.add_tag_line.mapToGlobal(self.add_tag_line.rect().bottomLeft())
            QToolTip.showText(tooltip_pos, tooltip_message, self.add_tag_line)
        if actually_new_tags:
            if tag_utils.add_tags_to_file(txt_path, actually_new_tags):
                self._update_tag_display()
                # Emit signal for undo/redo
                self.tags_added.emit(txt_path, actually_new_tags)
    @Slot(str)
    def _remove_tag(self, tag_to_remove: str):
        if not self._image_path: return
        txt_path = tag_utils.get_txt_path(self._image_path)
        
        # Get original index before removal
        existing_tags = tag_utils.read_tags(txt_path)
        original_index = existing_tags.index(tag_to_remove) if tag_to_remove in existing_tags else -1
        
        if tag_utils.remove_tag_from_file(txt_path, tag_to_remove):
            tags = tag_utils.read_tags(txt_path)
            total_pages = (len(tags) + TAGS_PER_PAGE_GRID - 1) // TAGS_PER_PAGE_GRID
            total_pages = max(1, total_pages)
            if self._current_tag_page >= total_pages: self._current_tag_page = max(0, total_pages - 1)
            self._update_tag_display()
            
            # Emit signal for undo/redo
            if original_index >= 0:
                self.tag_removed.emit(txt_path, tag_to_remove, original_index)
    @Slot()
    def _prev_tag_page(self):
        if self._current_tag_page > 0:
            self._current_tag_page -= 1
            self._update_tag_display()
    @Slot()
    def _next_tag_page(self):
        if not self._image_path: return
        txt_path = tag_utils.get_txt_path(self._image_path)
        tags = tag_utils.read_tags(txt_path)
        if (self._current_tag_page + 1) * TAGS_PER_PAGE_GRID < len(tags):
            self._current_tag_page += 1
            self._update_tag_display()
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filters events from watched objects to handle right-click and hover on tag buttons."""
        if watched is self.caption_edit and event.type() == QEvent.Type.FocusOut:
            self._save_caption()
            return False

        if event.type() == QEvent.Type.MouseButtonRelease:
            from PySide6.QtGui import QMouseEvent
            assert isinstance(event, QMouseEvent)
            if event.button() == Qt.MouseButton.RightButton:
                # Check if this is a tag button
                if isinstance(watched, QPushButton) and watched.property("original_tag"):
                    tag_name = watched.property("original_tag")
                    self._copy_tag_to_clipboard(tag_name)
                    return True # Event handled

        elif event.type() == QEvent.Type.Enter:
            if isinstance(watched, QPushButton) and watched.property("original_tag"):
                self.tag_hovered.emit(watched.property("original_tag"))
                return False

        elif event.type() == QEvent.Type.Leave:
            if isinstance(watched, QPushButton) and watched.property("original_tag"):
                self.tag_hover_cleared.emit()
                return False
        
        return super().eventFilter(watched, event)
    
    def _copy_tag_to_clipboard(self, tag_name: str):
        """Copies the tag with a trailing comma to clipboard."""
        from PySide6.QtGui import QCursor
        from PySide6.QtCore import QTimer
        clipboard = QApplication.clipboard()
        clipboard.setText(tag_name + ", ")
        # Show a tooltip to indicate success at the current cursor position
        cursor_pos = QCursor.pos()
        tooltip_text = self.locale_manager.get_string("MainWindow", "Copied_Tag_Tooltip", tag_name=tag_name)
        QToolTip.showText(cursor_pos, tooltip_text, self)
        QTimer.singleShot(2000, QToolTip.hideText)
    
    def clear_data(self):
        self._image_path = None
        self._global_index = -1
        self._current_tag_page = 0
        self.image_label.clear()
        self.image_label.setText(self.locale_manager.get_string("GridView", "No_Image"))
        for btn in self.tag_buttons: btn.deleteLater()
        self.tag_buttons.clear()
        self.prev_tag_page_btn.setEnabled(False)
        self.next_tag_page_btn.setEnabled(False)
        self.caption_edit.blockSignals(True)
        self.caption_edit.clear()
        self.caption_edit.blockSignals(False)
        self._caption_loaded_text = ""

    def set_tag_display_language(self, language: str, translation_map: dict[str, list[str]]):
        self._tag_display_language = language
        self.tag_translation_map = translation_map
        self._update_tag_display()

    def set_editing_enabled(self, enabled: bool):
        """Lock the cell's editing widgets while a worker rewrites the same .txt files.
        The image and its tag/caption text stay visible - only edits are blocked."""
        if not enabled:
            # Commit anything still inside the autosave debounce before locking.
            self._save_caption()
        self._tag_area_widget.setEnabled(enabled)
        self.caption_edit.setReadOnly(not enabled)

    def set_search_text(self, text: str):
        """検索文字列を設定し、タグ表示を更新する。"""
        self._search_text = text
        # In caption mode the search bar is hidden and irrelevant; skipping the refresh
        # here avoids reloading the .txt (and losing an unsaved edit) on every page render.
        if not self._caption_mode:
            self._update_tag_display()

class GridViewWidget(QWidget):
    back_to_main_requested = Signal()
    # Signals for undo/redo
    tags_added = Signal(Path, list)  # (file_path, added_tags)
    tag_removed = Signal(Path, str, int)  # (file_path, removed_tag, original_index)
    caption_edited = Signal(Path, str, str, bool)  # (txt_path, old_text, new_text, file_existed_before)
    caption_save_failed = Signal(str)
    # Signals for tag hover highlight
    tag_hovered = Signal(str)
    tag_hover_cleared = Signal()
    
    def __init__(self, settings: AppSettings, locale_manager: LocaleManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.settings = settings
        self.locale_manager = locale_manager
        self._image_paths: List[Path] = []
        self._filtered_image_paths: list[Path] = []
        self._current_page = 0
        self._tag_cache: dict[str, set[str]] = {}
        self._search_text: str = ""
        self._base_dir: Path | None = None
        self._caption_mode: bool = False
        self._editing_enabled: bool = True
        self.cells: List[ImageEditCellWidget] = []
        
        # --- ADDED: State management for the dialog ---
        self._image_viewer_dialog: ImageViewerDialog | None = None
        self._current_dialog_image_index = -1

        self.initUI()
    
    def initUI(self):
        # ... (initUI layout setup is unchanged) ...
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        top_bar_layout = QHBoxLayout()
        
        self.back_button = QPushButton(self.locale_manager.get_string("GridView", "Back_To_Main_View"))
        self.back_button.clicked.connect(self.back_to_main_requested.emit)

        # Search bar (placed in top bar, left of back button)
        self.search_line_edit = QLineEdit()
        self.search_line_edit.setPlaceholderText(self.locale_manager.get_string("GridView", "Search_By_Tag"))
        self.search_line_edit.setClearButtonEnabled(True)
        self.search_line_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search_line_edit.textChanged.connect(self._on_search_changed)

        top_bar_layout.addWidget(self.search_line_edit, 1)
        top_bar_layout.addWidget(self.back_button)
        main_layout.addLayout(top_bar_layout)

        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(2)
        main_layout.addLayout(self.grid_layout, 1)

        for i in range(9):
            cell = ImageEditCellWidget(self.locale_manager)
            # --- ADDED: Connect the cell's request signal to the handler slot ---
            cell.image_enlarge_requested.connect(self.show_enlarged_image_at_index)
            # Connect undo/redo signals
            cell.tags_added.connect(self.tags_added.emit)
            cell.tag_removed.connect(self.tag_removed.emit)
            cell.caption_edited.connect(self.caption_edited.emit)
            cell.caption_save_failed.connect(self.caption_save_failed.emit)
            cell.tag_hovered.connect(self.tag_hovered.emit)
            cell.tag_hover_cleared.connect(self.tag_hover_cleared.emit)
            self.cells.append(cell)
            row, col = divmod(i, 3)
            self.grid_layout.addWidget(cell, row, col)
            
        # ... (initUI pagination setup is unchanged) ...
        pagination_layout = QHBoxLayout()
        self.prev_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Previous_9"))
        self.prev_page_btn.clicked.connect(self.prev_page)
        self.prev_page_btn.setMinimumHeight(40)
        self.prev_page_btn.setStyleSheet("font-size: 14pt;")
        
        # Undo button (placed after "Previous 9")
        self.undo_button = QPushButton(
            self.locale_manager.get_string("MainWindow", "Undo_Button"))
        self.undo_button.setEnabled(False)
        self.undo_button.setMaximumWidth(100)
        self.undo_button.setMinimumHeight(40)
        self.undo_button.setStyleSheet("font-size: 14pt;")
        self.undo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Undo_No_Actions"))
        
        self.page_label = QLabel(self.locale_manager.get_string(
            "GridView", "Page_Label", current=1, total=1))
        
        # Redo button (placed before "Next 9")
        self.redo_button = QPushButton(
            self.locale_manager.get_string("MainWindow", "Redo_Button"))
        self.redo_button.setEnabled(False)
        self.redo_button.setMaximumWidth(100)
        self.redo_button.setMinimumHeight(40)
        self.redo_button.setStyleSheet("font-size: 14pt;")
        self.redo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Redo_No_Actions"))
        
        self.next_page_btn = QPushButton(self.locale_manager.get_string("GridView", "Next_9"))
        self.next_page_btn.clicked.connect(self.next_page)
        self.next_page_btn.setMinimumHeight(40)
        self.next_page_btn.setStyleSheet("font-size: 14pt;")
        
        pagination_layout.addStretch(3)
        pagination_layout.addWidget(self.prev_page_btn)
        pagination_layout.addSpacing(15)
        pagination_layout.addWidget(self.undo_button)
        pagination_layout.addStretch(2)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addStretch(2)
        pagination_layout.addWidget(self.redo_button)
        pagination_layout.addSpacing(15)
        pagination_layout.addWidget(self.next_page_btn)
        pagination_layout.addStretch(3)
        main_layout.addLayout(pagination_layout)

    @Slot(str)
    def _on_search_changed(self, text: str):
        """検索欄のテキスト変更時に呼ばれるスロット。"""
        self._search_text = text
        self._apply_filter()
        self._current_page = 0
        self._display_page()

    def set_caption_mode(self, on: bool):
        """Switch every cell between tag-editing and free-text caption editing.
        The tag search bar is tag-based, so it is hidden in caption mode."""
        self._caption_mode = on
        self.search_line_edit.setVisible(not on)
        if on:
            self._search_text = ""
            self.search_line_edit.clear()
        for cell in self.cells:
            cell.set_caption_mode(on)
        self._apply_filter()
        self._display_page()

    def _apply_filter(self):
        """_search_text と _tag_cache を使って _filtered_image_paths を生成する。"""
        if self._caption_mode or not self._search_text or not self._tag_cache or self._base_dir is None:
            self._filtered_image_paths = list(self._image_paths)
        else:
            self._filtered_image_paths = filter_images_by_tag(
                self._image_paths, self._tag_cache, self._search_text, self._base_dir
            )

    def _display_page(self):
        start_index = self._current_page * 9
        page_paths = self._filtered_image_paths[start_index : start_index + 9]

        for i, cell in enumerate(self.cells):
            if i < len(page_paths):
                # --- MODIFIED: Pass the global index to the cell ---
                global_index = start_index + i
                cell.load_data(page_paths[i], global_index)
                cell.set_search_text(self._search_text)
                cell.show()
            else:
                cell.clear_data()
                cell.hide()
        
        self._update_pagination_controls()

    # --- ADDED: Centralized handler for showing and managing the dialog ---
    @Slot(int)
    def show_enlarged_image_at_index(self, index: int):
        self._current_dialog_image_index = index

        if self._image_viewer_dialog is None:
            self._image_viewer_dialog = ImageViewerDialog(self, zoom_and_pan_enabled=True)
            self._image_viewer_dialog.nextImageRequested.connect(self._navigate_dialog_image_next)
            self._image_viewer_dialog.prevImageRequested.connect(self._navigate_dialog_image_prev)
            self._image_viewer_dialog.finished.connect(self._on_image_viewer_closed)
        
        self._load_image_into_dialog(index)

        if not self._image_viewer_dialog.isVisible():
            self._image_viewer_dialog.show()
            self._image_viewer_dialog.activateWindow()

    # --- ADDED: Helper method to load/update the image in the dialog ---
    def _load_image_into_dialog(self, index: int):
        if not (0 <= index < len(self._image_paths)) or not self._image_viewer_dialog:
            return

        path = self._image_paths[index]
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return

        # Set initial size and position only if it's the first time showing
        if not self._image_viewer_dialog.isVisible():
            # マルチディスプレイ対策: 常に primaryScreen ではなく、本体ウィンドウが乗っている
            # スクリーンを基準にする（main_window.show_enlarged_image と同じ方針）。
            main_window = self.window()
            anchor = main_window.frameGeometry().topLeft()
            target_screen = (QApplication.screenAt(anchor)
                              or main_window.screen()
                              or QApplication.primaryScreen())
            screen_rect = target_screen.availableGeometry()
            scaled_size = pixmap.size().scaled(screen_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self._image_viewer_dialog.resize(scaled_size)
            self._image_viewer_dialog.move(screen_rect.center() - self._image_viewer_dialog.rect().center())
        
        self._image_viewer_dialog.setPixmap(pixmap)

    # --- ADDED: Slots for handling navigation signals from the dialog ---
    @Slot()
    def _navigate_dialog_image_next(self):
        if self._current_dialog_image_index < len(self._image_paths) - 1:
            self._current_dialog_image_index += 1
            self._load_image_into_dialog(self._current_dialog_image_index)

    @Slot()
    def _navigate_dialog_image_prev(self):
        if self._current_dialog_image_index > 0:
            self._current_dialog_image_index -= 1
            self._load_image_into_dialog(self._current_dialog_image_index)

    # --- ADDED: Cleanup slot for when the dialog is closed ---
    @Slot()
    def _on_image_viewer_closed(self):
        self._image_viewer_dialog = None


    # ... (rest of the file is unchanged) ...
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.prev_page()
            event.accept()
        elif delta < 0:
            self.next_page()
            event.accept()
        else:
            super().wheelEvent(event)
    def load_images(self, image_paths: List[Path], tag_cache: dict[str, set[str]] | None = None, base_dir: Path | None = None):
        self._image_paths = image_paths
        if tag_cache is not None:
            self._tag_cache = tag_cache
        if base_dir is not None:
            self._base_dir = base_dir
        self._search_text = ""
        self.search_line_edit.clear()
        self._filtered_image_paths = list(image_paths)
        self._current_page = 0
        self._display_page()
    def prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._display_page()
    def next_page(self):
        if (self._current_page + 1) * 9 < len(self._filtered_image_paths):
            self._current_page += 1
            self._display_page()
    def _update_pagination_controls(self):
        if self._search_text and not self._filtered_image_paths:
            self.page_label.setText(self.locale_manager.get_string(
                "GridView", "Page_Label_Empty"))
            self.prev_page_btn.setEnabled(False)
            self.next_page_btn.setEnabled(False)
            return
        total_pages = (len(self._filtered_image_paths) + 8) // 9
        total_pages = max(1, total_pages)
        self.page_label.setText(self.locale_manager.get_string(
            "GridView", "Page_Label",
            current=self._current_page + 1, total=total_pages))
        self.prev_page_btn.setEnabled(self._current_page > 0)
        self.next_page_btn.setEnabled((self._current_page + 1) * 9 < len(self._filtered_image_paths))

    def set_tag_display_language(self, language: str, translation_map: dict[str, list[str]]):
        for cell in self.cells:
            cell.set_tag_display_language(language, translation_map)
    
    def set_editing_enabled(self, enabled: bool):
        """Called while a tagging/captioning run is active. Deliberately leaves the
        "back to main view" button and the page navigation alive so the user is not
        trapped in the grid - only the controls that write .txt files are locked."""
        self._editing_enabled = enabled
        for cell in self.cells:
            cell.set_editing_enabled(enabled)
        if not enabled:
            # Only force them off here. Turning editing back on must NOT read the buttons'
            # own disabled state (that can never recover) - MainWindow re-drives them from
            # the UndoManager via update_undo_redo_buttons().
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)

    def update_undo_redo_buttons(self, can_undo: bool, can_redo: bool, undo_desc: str, redo_desc: str):
        """Updates the enabled state and tooltips of undo/redo buttons in grid view."""
        self.undo_button.setEnabled(can_undo and self._editing_enabled)
        self.redo_button.setEnabled(can_redo and self._editing_enabled)
        
        if can_undo:
            self.undo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Undo_Action", desc=undo_desc))
        else:
            self.undo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Undo_No_Actions"))
        
        if can_redo:
            self.redo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Redo_Action", desc=redo_desc))
        else:
            self.redo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Redo_No_Actions"))
    
    def refresh_current_page(self):
        """Refreshes the current page display after undo/redo operations."""
        self._apply_filter()
        self._display_page()

    def update_tag_cache(self, tag_cache: dict[str, set[str]]):
        """タグキャッシュを更新し、現在の検索条件でフィルタリングを再適用する。"""
        self._tag_cache = tag_cache
        self._apply_filter()
        self._display_page()
