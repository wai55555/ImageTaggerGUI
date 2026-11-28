"""Phase 8B（グリッド／ビューア）の回帰テスト。対象: F（REQ-8B.1）。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _make_image(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (4, 4)).save(path)
    return path


def test_enlarged_dialog_follows_the_filtered_list(tmp_path, monkeypatch):
    """F: 検索フィルタ中でも拡大表示と前後ナビが同じ一覧を辿る。"""
    import grid_view_widget

    _qt_app()
    shown: list[str] = []

    class _FakePixmap:
        def __init__(self, path=""):
            shown.append(str(path))

        def isNull(self):
            return False

        def size(self):
            from PySide6.QtCore import QSize

            return QSize(4, 4)

        def scaled(self, *a, **k):
            return self

    class _FakeDialog:
        def __init__(self):
            self.pixmaps = []

        def setPixmap(self, pixmap):
            self.pixmaps.append(pixmap)

        def isVisible(self):
            return True

        def show(self):
            pass

        def activateWindow(self):
            pass

        def resize(self, *a):
            pass

        def move(self, *a):
            pass

        def rect(self):
            from PySide6.QtCore import QRect

            return QRect(0, 0, 10, 10)

        def close(self):
            pass

        def deleteLater(self):
            pass

    monkeypatch.setattr(grid_view_widget, "QPixmap", _FakePixmap)

    class _Locale:
        def get_string(self, section, key, **kwargs):
            return key

    widget = grid_view_widget.GridViewWidget.__new__(grid_view_widget.GridViewWidget)
    images = [_make_image(tmp_path / f"{i}.png") for i in range(3)]
    widget._image_paths = list(images)
    # 検索で 0.png が除外された状態（セルは 1,2 を表示している）
    widget._filtered_image_paths = [images[1], images[2]]
    widget._current_dialog_image_index = 0
    widget._image_viewer_dialog = _FakeDialog()

    widget._load_image_into_dialog(0)
    assert shown[-1].endswith("1.png")     # 未フィルタの 0.png ではない

    widget._navigate_dialog_image_next()
    assert shown[-1].endswith("2.png")

    # 末尾で止まる（フィルタ後の件数で判定）
    widget._navigate_dialog_image_next()
    assert shown[-1].endswith("2.png")
