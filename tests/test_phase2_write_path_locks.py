"""Phase 2（`.txt` 書き込みロックの統一）の回帰テスト。

対象: R2-7（中核）、R7-4、R12-6、R12-7、R12-9、R19-6、R21-1、R25-1、R35-1。

方針:
- 「処理中か」の判定は `MainWindow._is_processing()` が唯一の正本であること（REQ-2.2）。
- 全ての書き込み経路が、錠（コントロールの無効化）とハンドラ側ガードの**両方**で
  止まること（REQ-2.3）。
- 表示更新・ローダ完了が錠を解かないこと（REQ-2.4）。
- フォルダ切替は処理中に受理しないこと（REQ-2.5 / U-1）。
- 後始末の待機中に新しい実行が始まっても、外側の後始末が新しいスレッドへ触らないこと
  （REQ-2.6 / R35-1）。
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _qt_app():
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _inside_repo(path: Path) -> bool:
    root = Path(__file__).resolve().parents[1]
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


class _Signal:
    """QObject のシグナルの最小代用（connect だけを記録する）。"""

    def __init__(self) -> None:
        self.slots: list = []

    def connect(self, slot) -> None:
        self.slots.append(slot)


class _FakeThread:
    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.quit_called = False
        self.deleted = False
        self.finished = _Signal()
        self.started = _Signal()

    def isRunning(self) -> bool:
        return self.running

    def isFinished(self) -> bool:
        return not self.running

    def quit(self) -> None:
        self.quit_called = True

    def start(self) -> None:
        self.started_here = True

    def wait(self, ms: int = 0) -> bool:
        return not self.running

    def deleteLater(self) -> None:
        self.deleted = True


class _FakeBulkWorker:
    def __init__(self, get_string=None) -> None:
        self.log_message = _Signal()
        self.finished = _Signal()
        self.bulk_add_completed = _Signal()
        self.bulk_delete_completed = _Signal()

    def moveToThread(self, thread) -> None:
        self.thread = thread

    def deleteLater(self) -> None:
        self.deleted = True


def _make_window(monkeypatch, tmp_path: Path):
    """実 config・実フォルダに触れない MainWindow を作る（Phase 1 の隔離前提）。"""
    app = _qt_app()
    import main_window
    import onnx_providers

    monkeypatch.setattr(onnx_providers, "has_nvidia_gpu", lambda *a, **k: False)
    monkeypatch.setattr(main_window.MainWindow, "_maybe_prompt_gpu_setup", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_maybe_prompt_update_check", lambda self: None)

    import app_settings

    assert not _inside_repo(Path(app_settings.CONFIG_PATH)), app_settings.CONFIG_PATH

    window = main_window.MainWindow()
    app.processEvents()
    # 読み込み系のスレッドを走らせない（このテストの対象はロック判定だけ）。
    window.settings.paths.input_dir = str(tmp_path / "no_such_input_dir")
    window.update_log = lambda *a, **k: None
    return app, window


def _add_image_item(window, image_path: Path, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QListWidgetItem

    rel = image_path.name
    window.settings.paths.input_dir = str(image_path.parent)
    item = QListWidgetItem(rel)
    item.setData(Qt.ItemDataRole.UserRole + 1, rel)
    window.image_list.addItem(item)
    window.image_list.setCurrentItem(item)


# ---------------------------------------------------------------------------
# REQ-2.2: 判定の一本化
# ---------------------------------------------------------------------------


def test_is_processing_reads_the_existing_state_owners(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    try:
        assert window._is_processing() is False

        window._is_downloading = True
        assert window._is_processing() is True
        window._is_downloading = False

        window._is_bulk_deleting = True
        assert window._is_processing() is True
        window._is_bulk_deleting = False

        window._tagger_thread = _FakeThread(running=True)
        assert window._is_processing() is True
        window._tagger_thread = None

        window._bulk_tag_thread = _FakeThread(running=True)
        assert window._is_processing() is True
        window._bulk_tag_thread = None

        # 停止済みのスレッドは「処理中」ではない。
        window._tagger_thread = _FakeThread(running=False)
        assert window._is_processing() is False
        window._tagger_thread = None
    finally:
        window._tagger_thread = None
        window._bulk_tag_thread = None
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-2.3 / R2-7: 一括処理は本体の錠もかける（片方だけに依存しない）
# ---------------------------------------------------------------------------


def test_bulk_start_locks_every_write_path_and_finish_unlocks(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    import main_window

    fake_thread = _FakeThread(running=True)
    monkeypatch.setattr(main_window, "QThread", lambda: fake_thread)
    monkeypatch.setattr(main_window, "BulkTagWorker", _FakeBulkWorker)
    try:
        window._start_bulk_tag_worker("delete", input_dir=tmp_path, tag="x")

        assert window._is_processing() is True
        # 錠の実体: 一括用だけでなく、本体の書き込み経路が全て閉じる。
        assert window.add_tag_line.isEnabled() is False          # 一括追加（R2-7）
        assert window.add_single_tag_line.isEnabled() is False      # 単体追加（R25-1）
        assert window.add_single_tag_button.isEnabled() is False
        assert window.use_gpu_check.isEnabled() is False            # GPU トグル（R21-1）
        assert window.grid_view_button.isEnabled() is False         # グリッド切替（R2-7）
        assert window.undo_button.isEnabled() is False              # Undo（R2-7）
        assert window.caption_text_edit.isEnabled() is False
        assert window.vlm_single_test_button.isEnabled() is False   # 1枚テスト（R12-9）

        # 終了経路だけが解錠する。
        fake_thread.running = False
        window._on_bulk_tag_finished()
        assert window._is_processing() is False
        assert window.add_single_tag_button.isEnabled() is True
        assert window.use_gpu_check.isEnabled() is True
        assert window.grid_view_button.isEnabled() is True
    finally:
        window._bulk_tag_thread = None
        window._bulk_tag_worker = None
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-2.4 / R7-4 / R12-6: 表示更新・ローダ完了は錠を解かない
# ---------------------------------------------------------------------------


def test_tag_page_refresh_does_not_unlock_controls(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    try:
        window._set_main_controls_enabled(False)
        window._is_bulk_deleting = True

        window.display_current_tag_page()
        window._display_image_tag_page()

        assert window._is_processing() is True
        assert window.add_tag_line.isEnabled() is False
        assert window.add_tag_button.isEnabled() is False
    finally:
        window._is_bulk_deleting = False
        window.close()
        app.processEvents()


def test_loader_completion_does_not_clear_the_bulk_guard(monkeypatch, tmp_path):
    """R12-6: TagLoader の遅延完了が一括ガードを解除しない。"""
    app, window = _make_window(monkeypatch, tmp_path)
    try:
        window._set_main_controls_enabled(False)
        window._is_bulk_deleting = True

        window._update_bulk_tag_buttons([])

        assert window._is_bulk_deleting is True
        assert window._is_processing() is True
        assert window.add_tag_line.isEnabled() is False
    finally:
        window._is_bulk_deleting = False
        window.close()
        app.processEvents()


def test_tag_chips_are_created_locked_while_processing(monkeypatch, tmp_path):
    """R19-6: 表示中画像のタグチップが実行中でも押せる状態で作られない。"""
    app, window = _make_window(monkeypatch, tmp_path)
    try:
        window._set_main_controls_enabled(False)
        window._current_image_tags = ["solo", "1girl"]
        window._display_image_tag_page()
        app.processEvents()

        assert window.tag_buttons_for_image
        assert all(not b.isEnabled() for b in window.tag_buttons_for_image)
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-2.5 / R12-7 / U-1: フォルダ切替は処理中に受理しない
# ---------------------------------------------------------------------------


def test_folder_switch_is_rejected_while_processing(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    logs: list[tuple[str, str]] = []
    window.update_log = lambda msg, color=None: logs.append((msg, color))
    target = tmp_path / "other"
    target.mkdir()
    try:
        window.input_line.setText(str(tmp_path))
        window._batch_failed_paths = [tmp_path / "keep.png"]
        window._is_bulk_deleting = True

        window._handle_folder_drop(str(target))

        assert window.input_line.text() == str(tmp_path)      # 切り替わっていない
        assert window._batch_failed_paths == [tmp_path / "keep.png"]  # 破棄していない
        assert logs and logs[-1][1] == "orange"

        # 処理が終われば通常どおり受理される。
        window._is_bulk_deleting = False
        window._handle_folder_drop(str(target))
        assert window.input_line.text() == str(target)
    finally:
        window._is_bulk_deleting = False
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-2.3 / R25-1 / R19-6 / R12-9 / Undo: ハンドラ側のガード
# ---------------------------------------------------------------------------


def test_single_tag_add_and_chip_delete_are_blocked_while_processing(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"not really a png")
        txt = image.with_suffix(".txt")
        txt.write_text("old", encoding="utf-8")
        _add_image_item(window, image, tmp_path)

        window._is_bulk_deleting = True

        window.add_single_tag_line.setText("new")
        window._add_single_tag()
        window._delete_image_tag("old")

        assert txt.read_text(encoding="utf-8") == "old"
        assert window.undo_manager.can_undo() is False
    finally:
        window._is_bulk_deleting = False
        window.close()
        app.processEvents()


def test_bulk_add_delete_and_undo_are_blocked_while_processing(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(window, "_start_bulk_tag_worker",
                        lambda *a, **k: calls.append("bulk"))
    monkeypatch.setattr(window.undo_manager, "undo", lambda: calls.append("undo") or True)
    try:
        window._is_bulk_deleting = True

        window.add_tag_line.setText("tag")
        window.add_tag_all(prepend=False)
        window.delete_tag_all("tag")
        window._perform_undo()

        assert calls == []
    finally:
        window._is_bulk_deleting = False
        window.close()
        app.processEvents()


def test_vlm_single_test_is_blocked_while_a_bulk_run_is_active(monkeypatch, tmp_path):
    """R12-9: 一括処理中に VLM 単体テストを開始できない。"""
    app, window = _make_window(monkeypatch, tmp_path)
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"x")
        _add_image_item(window, image, tmp_path)
        window._is_bulk_deleting = True

        window._run_vlm_single_test()

        assert window._tagger_thread is None
    finally:
        window._is_bulk_deleting = False
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-2.6 / R35-1: 後始末の再入
# ---------------------------------------------------------------------------


def test_cleanup_leaves_a_run_that_started_during_the_wait_alone(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    import main_window

    old_thread = _FakeThread(running=True)
    new_thread = _FakeThread(running=True)
    window._tagger_thread = old_thread
    window._tagger_worker = None

    state = {"swapped": False}

    def fake_process_events(*_args, **_kwargs):
        if not state["swapped"]:
            state["swapped"] = True
            # 待機中に新しい実行が始まり、窓口が差し替わる。
            window._tagger_thread = new_thread
            window._tagger_worker = None
            old_thread.running = False  # 旧スレッドは自然終了した
        return True

    monkeypatch.setattr(main_window.QApplication, "processEvents", fake_process_events)
    before = len(main_window._detached_threads)
    try:
        window._cleanup_tagger_thread()

        assert state["swapped"] is True
        # 新しい実行は外側の後始末に触られない。
        assert window._tagger_thread is new_thread
        assert new_thread.deleted is False
        assert len(main_window._detached_threads) == before
        assert old_thread not in main_window._detached_threads
    finally:
        window._tagger_thread = None
        window.close()
        app.processEvents()


def test_cleanup_detaches_a_stuck_thread_without_terminating(monkeypatch, tmp_path):
    app, window = _make_window(monkeypatch, tmp_path)
    import main_window

    stuck = _FakeThread(running=True)
    # terminate() は存在しない（呼べば AttributeError になる）。NFR-12 の固定。
    assert not hasattr(stuck, "terminate")
    window._tagger_thread = stuck
    window._tagger_worker = None

    ticks = iter([0.0, 0.0, 6.0, 6.0, 6.0])
    monkeypatch.setattr(main_window.time, "monotonic", lambda: next(ticks, 6.0))
    monkeypatch.setattr(main_window.QApplication, "processEvents", lambda *a, **k: True)
    try:
        window._cleanup_tagger_thread()

        assert stuck in main_window._detached_threads
        assert window._tagger_thread is None
    finally:
        if stuck in main_window._detached_threads:
            main_window._detached_threads.remove(stuck)
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-2.3: グリッドセル側の書き込み経路（タグ追加／タグ削除／キャプション保存）
# ---------------------------------------------------------------------------


def _make_cell(tmp_path: Path):
    import grid_view_widget
    from locale_manager import LocaleManager

    lang_dir = Path(grid_view_widget.__file__).resolve().parent.parent / "lang"
    cell = grid_view_widget.ImageEditCellWidget(LocaleManager("en", lang_dir))
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    txt = image.with_suffix(".txt")
    txt.write_text("old", encoding="utf-8")
    cell._image_path = image
    # 実際の使い方と同じ手順でキャプション欄をロードしておく（未ロードのままだと
    # 「空文字で上書き」という別の不具合を踏む。それは Phase 4 の担当）。
    cell.set_caption_mode(True)
    return cell, txt


def test_grid_cell_ignores_writes_while_locked(monkeypatch, tmp_path):
    app = _qt_app()
    cell, txt = _make_cell(tmp_path)
    try:
        cell.set_editing_enabled(False)
        cell.add_tag_line.setText("new tag")
        cell._add_tag()
        cell._remove_tag("old")
        cell.caption_edit.setPlainText("changed caption")
        cell._save_caption()
        app.processEvents()

        assert txt.read_text(encoding="utf-8") == "old"
    finally:
        cell.deleteLater()
        app.processEvents()


def test_grid_cell_still_flushes_an_edit_before_locking(tmp_path):
    app = _qt_app()
    cell, txt = _make_cell(tmp_path)
    try:
        cell.caption_edit.setPlainText("edited before the run")
        cell.set_editing_enabled(False)
        app.processEvents()

        assert txt.read_text(encoding="utf-8") == "edited before the run"
    finally:
        cell.deleteLater()
        app.processEvents()
