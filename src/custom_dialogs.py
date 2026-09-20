from enum import IntFlag
from typing import Any, Callable, Sequence

from PySide6.QtCore import (
    Qt, Signal, QPoint, QRect, QEvent, QPointF
)
from PySide6.QtWidgets import (
    QLabel, QDialog, QApplication, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QSlider, QPushButton, QWidget
)
from PySide6.QtGui import (
    QMouseEvent, QPixmap, QKeyEvent, QWheelEvent, QResizeEvent, QPainter
)

import app_settings


class CategoryTagSettingsDialog(QDialog):
    """Per-category threshold / max-tag-count editor, reached from the "詳細" button.

    Shows one row per tag category the selected model produces. Moving a slider writes
    straight into the shared Thresholds / Limits settings objects, flags that category
    as user-"touched" (so a later model switch won't overwrite it), and calls
    `on_changed` so MainWindow can mirror the main sliders and persist.
    """

    _THRESHOLD_RES = 100  # slider steps per 1.0
    # Per-category max-tag slider caps. general / character MUST match the main window's
    # sliders (150 / 10) so the shared value can't display differently in the two places.
    _LIMIT_CAPS = {"character": 10}

    def __init__(
        self,
        parent: QWidget | None,
        *,
        get_string: Callable[..., str],
        categories: Sequence[str],
        thresholds: Any,
        limits: Any,
        on_changed: Callable[[], None],
        on_reset: Callable[[], None],
        max_limit: int = 150,
    ) -> None:
        super().__init__(parent)
        self._get_string = get_string
        self._categories = list(categories)
        self._thresholds = thresholds
        self._limits = limits
        self._on_changed = on_changed
        self._on_reset = on_reset
        self._max_limit = max_limit
        self._loading = False
        self._rows: dict[str, tuple[QSlider, QLabel, QSlider, QLabel]] = {}

        self.setWindowTitle(get_string("CategoryDialog", "Title"))
        self.setModal(False)

        layout = QVBoxLayout(self)
        hint = QLabel(get_string("CategoryDialog", "Hint"))
        hint.setWordWrap(True)
        layout.addWidget(hint)

        grid_box = QGroupBox()
        grid = QGridLayout(grid_box)
        grid.addWidget(QLabel(get_string("CategoryDialog", "Header_Category")), 0, 0)
        grid.addWidget(QLabel(get_string("CategoryDialog", "Header_Threshold")), 0, 1, 1, 2)
        grid.addWidget(QLabel(get_string("CategoryDialog", "Header_MaxTags")), 0, 3, 1, 2)

        for i, category in enumerate(self._categories, start=1):
            grid.addWidget(QLabel(self._category_label(category)), i, 0)

            thr_slider = QSlider(Qt.Orientation.Horizontal)
            thr_slider.setRange(0, self._THRESHOLD_RES)
            thr_value = QLabel()
            thr_value.setFixedWidth(44)
            thr_slider.valueChanged.connect(lambda v, c=category: self._on_threshold_changed(c, v))
            grid.addWidget(thr_slider, i, 1)
            grid.addWidget(thr_value, i, 2)

            lim_slider = QSlider(Qt.Orientation.Horizontal)
            lim_slider.setRange(0, self._LIMIT_CAPS.get(category, self._max_limit))
            lim_value = QLabel()
            lim_value.setFixedWidth(44)
            lim_slider.valueChanged.connect(lambda v, c=category: self._on_limit_changed(c, v))
            grid.addWidget(lim_slider, i, 3)
            grid.addWidget(lim_value, i, 4)

            self._rows[category] = (thr_slider, thr_value, lim_slider, lim_value)

        layout.addWidget(grid_box)

        note = QLabel(get_string("CategoryDialog", "ZeroNote"))
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        reset_btn = QPushButton(get_string("CategoryDialog", "Reset_Button"))
        reset_btn.clicked.connect(self._handle_reset)
        close_btn = QPushButton(get_string("CategoryDialog", "Close_Button"))
        close_btn.clicked.connect(self.close)
        buttons.addWidget(reset_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self.reload_from_settings()

    def _category_label(self, category: str) -> str:
        label = self._get_string("CategoryDialog", f"Category_{category}")
        # LocaleManager returns the key itself when a string is missing.
        return category.capitalize() if label == f"Category_{category}" else label

    def reload_from_settings(self) -> None:
        """Re-sync every slider from the current Thresholds / Limits values.

        Out-of-range stored values (e.g. a Character limit of 50 written by an older
        build whose dialog allowed 0..150) are clamped AND written back, so the dialog,
        the main-window sliders and what inference actually uses cannot disagree.
        """
        self._loading = True
        clamped_any = False
        try:
            for category, (thr_slider, thr_value, lim_slider, lim_value) in self._rows.items():
                raw_thr = float(getattr(self._thresholds, category, 0.0))
                raw_lim = int(getattr(self._limits, category, 0))
                thr = max(0.0, min(1.0, raw_thr))
                cap = self._LIMIT_CAPS.get(category, self._max_limit)
                lim = max(0, min(cap, raw_lim))
                if thr != raw_thr:
                    setattr(self._thresholds, category, thr)
                    clamped_any = True
                if lim != raw_lim:
                    setattr(self._limits, category, lim)
                    clamped_any = True
                thr_slider.setValue(int(round(thr * self._THRESHOLD_RES)))
                thr_value.setText(f"{thr:.2f}")
                lim_slider.setValue(lim)
                lim_value.setText(str(lim))
        finally:
            self._loading = False
        if clamped_any:
            # Mirror to the main sliders and persist the corrected values.
            self._on_changed()

    def _on_threshold_changed(self, category: str, raw: int) -> None:
        value = raw / self._THRESHOLD_RES
        self._rows[category][1].setText(f"{value:.2f}")
        if self._loading:
            return
        setattr(self._thresholds, category, value)
        self._thresholds.touched = app_settings.add_touched(self._thresholds.touched, category)
        self._on_changed()

    def _on_limit_changed(self, category: str, raw: int) -> None:
        self._rows[category][3].setText(str(raw))
        if self._loading:
            return
        setattr(self._limits, category, int(raw))
        self._limits.touched = app_settings.add_touched(self._limits.touched, category)
        self._on_changed()

    def _handle_reset(self) -> None:
        self._on_reset()

class ClickableLabel(QLabel):
    """A QLabel that emits a 'doubleClicked' signal on a double-click."""
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def hasHeightForWidth(self) -> bool:
        pixmap: QPixmap | None = self.pixmap()
        return pixmap is not None and not pixmap.isNull() # type: ignore

    def heightForWidth(self, width: int) -> int:
        if self.pixmap() and not self.pixmap().isNull() and self.pixmap().width() > 0:
            return int(width * (self.pixmap().height() / self.pixmap().width()))
        return super().heightForWidth(width)
    
    def sizeHint(self):
        if self.pixmap() and not self.pixmap().isNull():
            return self.pixmap().size()
        return super().sizeHint()

class ImageViewerDialog(QDialog):
    """
    A frameless dialog that displays an image. Operates in two distinct modes:
    1. Navigation Mode (zoom_and_pan_enabled=False): For MainWindow.
    2. Zoom/Pan Mode (zoom_and_pan_enabled=True): For GridView, with window-drag panning.
    """
    nextImageRequested = Signal()
    prevImageRequested = Signal()

    class ResizeHandle(IntFlag):
        NoHandle = 0x00; Left = 0x01; Right = 0x02; Top = 0x04; Bottom = 0x08
        TopLeft = Top | Left; TopRight = Top | Right
        BottomLeft = Bottom | Left; BottomRight = Bottom | Right

    def __init__(self, parent: QWidget | None = None, tag_panel_rect: QRect = QRect(), zoom_and_pan_enabled: bool = False):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground) # type: ignore
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover) # type: ignore
        self.setMinimumSize(200, 200)

        self._zoom_and_pan_enabled = zoom_and_pan_enabled

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMouseTracking(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label)
        
        # --- State variables ---
        self._resizing = False
        self._resize_edge = self.ResizeHandle.NoHandle
        self._last_mouse_global_pos = QPoint()
        self._tag_panel_rect = tag_panel_rect
        self._original_pixmap: QPixmap | None = None
        self._window_move_offset = QPoint()
        self._scale_factor = 1.0
        self._pan_offset: QPointF = QPointF(0, 0)
    
    def show_image(self, pixmap: QPixmap, dialog_width: int, dialog_height: int):
        self._original_pixmap = pixmap
        self.resize(dialog_width, dialog_height)
        self._update_image_display()

    def setPixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self.reset_view()

    def reset_view(self):
        if not (self._zoom_and_pan_enabled and self._original_pixmap): return
        
        pixmap_size = self._original_pixmap.size()
        label_size = self.image_label.size()
        if pixmap_size.isEmpty() or label_size.isEmpty(): return

        w_ratio = label_size.width() / pixmap_size.width()
        h_ratio = label_size.height() / pixmap_size.height()
        self._scale_factor = min(w_ratio, h_ratio)
        
        scaled_size = pixmap_size * self._scale_factor
        self._pan_offset = QPointF(
            (label_size.width() - scaled_size.width()) / 2,
            (label_size.height() - scaled_size.height()) / 2
        )
        self._update_image_display()

    def _update_image_display(self):
        if not self._original_pixmap or self._original_pixmap.isNull():
            self.image_label.clear()
            return

        if self._zoom_and_pan_enabled:
            # Create transformed pixmap in memory for zoom effect
            target_pixmap = QPixmap(self.size()) # Use dialog size for the canvas
            target_pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(target_pixmap)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.translate(self._pan_offset)
            painter.scale(self._scale_factor, self._scale_factor)
            painter.drawPixmap(0, 0, self._original_pixmap)
            painter.end()
            self.image_label.setPixmap(target_pixmap)
        else:
            # Simple scaled pixmap for navigation mode
            scaled = self._original_pixmap.scaled(self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)

    def wheelEvent(self, event: QWheelEvent):
        modifiers = event.modifiers()
        delta = event.angleDelta().y()
        
        if (self._zoom_and_pan_enabled and modifiers == Qt.KeyboardModifier.ControlModifier) or \
           (not self._zoom_and_pan_enabled):
            if delta > 0: self.prevImageRequested.emit()
            elif delta < 0: self.nextImageRequested.emit()
            event.accept()
        elif self._zoom_and_pan_enabled and self._original_pixmap:
            if delta == 0: return
            zoom_factor = 1.15 if delta > 0 else 1 / 1.15
            mouse_pos = QPointF(event.position())
            
            image_pos_before_zoom = (mouse_pos - self._pan_offset) / self._scale_factor # type: ignore
            self._scale_factor *= zoom_factor
            self._pan_offset = mouse_pos - image_pos_before_zoom * self._scale_factor
            
            self._update_image_display()
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._resize_edge = self._get_resize_handle(event.pos())
            if self._resize_edge != self.ResizeHandle.NoHandle: # type: ignore # type: ignore
                self._resizing = True
                self._last_mouse_global_pos = event.globalPosition().toPoint()
            else:
                # Both modes will now initiate a window move
                self._window_move_offset = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._resizing:
            diff = event.globalPosition().toPoint() - self._last_mouse_global_pos
            self._last_mouse_global_pos = event.globalPosition().toPoint()
            new_rect = self.geometry()
            if self._resize_edge & self.ResizeHandle.Left: new_rect.setLeft(new_rect.left() + diff.x())
            if self._resize_edge & self.ResizeHandle.Right: new_rect.setRight(new_rect.right() + diff.x())
            if self._resize_edge & self.ResizeHandle.Top: new_rect.setTop(new_rect.top() + diff.y())
            if self._resize_edge & self.ResizeHandle.Bottom: new_rect.setBottom(new_rect.bottom() + diff.y())
            
            # Aspect ratio lock only for navigation mode
            if not self._zoom_and_pan_enabled and self._original_pixmap and not self._original_pixmap.isNull():
                ratio = self._original_pixmap.width() / self._original_pixmap.height() if self._original_pixmap.height() > 0 else 1
                if self._resize_edge & (self.ResizeHandle.Left | self.ResizeHandle.Right):
                    new_height = int(new_rect.width() / ratio)
                    if self._resize_edge & self.ResizeHandle.Top: new_rect.setTop(new_rect.bottom() - new_height)
                    else: new_rect.setBottom(new_rect.top() + new_height)
                elif self._resize_edge & (self.ResizeHandle.Top | self.ResizeHandle.Bottom):
                    new_width = int(new_rect.height() * ratio)
                    if self._resize_edge & self.ResizeHandle.Left: new_rect.setLeft(new_rect.right() - new_width)
                    else: new_rect.setRight(new_rect.left() + new_width)
            self.setGeometry(self._snap_to_edges(new_rect))

        # --- MODIFIED: Panning is now window moving for BOTH modes ---
        elif event.buttons() == Qt.MouseButton.LeftButton:
            new_pos = self.pos() + event.pos() - self._window_move_offset
            self.move(self._snap_to_edges(QRect(new_pos, self.size())).topLeft())
        
        else:
            self.setCursor(self._get_cursor_for_position(event.pos()))
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._resizing = False
            self._resize_edge = self.ResizeHandle.NoHandle
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent):
        # In both modes, the view needs to be updated on resize
        if self._zoom_and_pan_enabled:
            self.reset_view()
        else:
            self._update_image_display()
        super().resizeEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_W, Qt.Key_K, Qt.Key_Left, Qt.Key_H, Qt.Key.Key_A): # type: ignore
            self.prevImageRequested.emit()
        elif key in (Qt.Key_Down, Qt.Key_S, Qt.Key_J, Qt.Key_Right, Qt.Key_L, Qt.Key.Key_D): # type: ignore
            self.nextImageRequested.emit()
        elif key == Qt.Key.Key_Escape:
            self.close()
        elif self._zoom_and_pan_enabled and key == Qt.Key.Key_R:
            self.reset_view()
        else:
            super().keyPressEvent(event)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.close()

    def leaveEvent(self, event: QEvent):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

    def _get_cursor_for_position(self, pos: QPoint):
        handle = self._get_resize_handle(pos)
        if handle in (self.ResizeHandle.TopLeft, self.ResizeHandle.BottomRight): return Qt.CursorShape.SizeFDiagCursor
        if handle in (self.ResizeHandle.TopRight, self.ResizeHandle.BottomLeft): return Qt.CursorShape.SizeBDiagCursor
        if handle & (self.ResizeHandle.Left | self.ResizeHandle.Right): return Qt.CursorShape.SizeHorCursor
        if handle & (self.ResizeHandle.Top | self.ResizeHandle.Bottom): return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def _get_resize_handle(self, pos: QPoint) -> IntFlag:
        margin = 15; width = self.width(); height = self.height()
        handle = self.ResizeHandle.NoHandle
        if pos.x() < margin: handle |= self.ResizeHandle.Left
        if pos.x() > width - margin: handle |= self.ResizeHandle.Right
        if pos.y() < margin: handle |= self.ResizeHandle.Top
        if pos.y() > height - margin: handle |= self.ResizeHandle.Bottom
        return handle

    def _snap_to_edges(self, current_rect: QRect) -> QRect:
        snapped_rect = QRect(current_rect)
        # マルチディスプレイ対策: ドラッグ/リサイズ中にこのダイアログが実際に乗っている
        # スクリーンを基準にする（常に primaryScreen だと非メインモニタ側で誤ってスナップする）。
        target_screen = self.screen() or QApplication.primaryScreen()
        screen = target_screen.availableGeometry()
        threshold = 30
        if abs(snapped_rect.left() - screen.left()) < threshold: snapped_rect.setLeft(screen.left())
        if abs(snapped_rect.right() - screen.right()) < threshold: snapped_rect.setRight(screen.right())
        if abs(snapped_rect.top() - screen.top()) < threshold: snapped_rect.setTop(screen.top())
        if abs(snapped_rect.bottom() - screen.bottom()) < threshold: snapped_rect.setBottom(screen.bottom())
        if self._tag_panel_rect and not self._tag_panel_rect.isNull():
            if abs(snapped_rect.right() - self._tag_panel_rect.left()) < threshold:
                snapped_rect.setRight(self._tag_panel_rect.left())
        return snapped_rect
