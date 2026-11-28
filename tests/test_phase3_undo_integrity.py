"""Phase 3（Undo / 履歴の整合性）の回帰テスト。

対象: R1-4、R2-9、R2-10、R2-12、R4-6、R5-5、R9-8、R11-11、R13-6、R20-2、R20-4、R22-4。
意味論は docs/261129_phase3_undo_semantics.md（REQ-3.1）が正本。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import undo_manager as U


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# REQ-3.3 / R22-4 / R1-4: 一括追加は「実際に追加されたタグ」だけを戻す
# ---------------------------------------------------------------------------


def test_bulk_add_undo_keeps_preexisting_tags_and_user_order(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    _write(a, "solo")                     # 既存タグあり
    _write(b, "")                         # 新規作成されたファイル

    action = U.BulkAddTagsAction(
        file_tags=[(a, ["blue eyes", "new tag"]), (b, ["blue eyes", "new tag"])],
        position="append",
    )

    assert action.undo() is U.UndoResult.FULL
    assert _read(a) == "solo"             # 元からあったタグは消さない
    assert not b.exists()                  # このアクションが作ったファイルは残さない

    assert action.redo() is U.UndoResult.FULL
    assert _read(a) == "solo, blue eyes, new tag"   # 利用者の指定順を保つ
    assert _read(b) == "blue eyes, new tag"         # 消えたファイルも作り直す

    assert action.undo() is U.UndoResult.FULL
    assert _read(a) == "solo"
    assert not b.exists()


def test_bulk_add_prepend_is_symmetric(tmp_path):
    a = tmp_path / "a.txt"
    _write(a, "solo")
    action = U.BulkAddTagsAction(file_tags=[(a, ["x", "y"])], position="prepend")

    action.undo()
    assert _read(a) == "solo"
    action.redo()
    assert _read(a) == "x, y, solo"
    action.undo()
    assert _read(a) == "solo"


# ---------------------------------------------------------------------------
# REQ-3.3 / R5-5: 一括削除は全出現位置を戻す
# ---------------------------------------------------------------------------


def test_bulk_delete_restores_every_occurrence(tmp_path):
    a = tmp_path / "a.txt"
    _write(a, "cat, solo, cat, cat")

    # cat は 0, 2, 3 番目にあった
    action = U.BulkRemoveTagsAction("cat", [(a, [0, 2, 3])])
    assert action.undo() is U.UndoResult.FULL  # まだ消していないので何もしない（冪等）
    assert _read(a) == "cat, solo, cat, cat"

    assert action.redo() is U.UndoResult.FULL
    assert _read(a) == "solo"

    assert action.undo() is U.UndoResult.FULL
    assert _read(a) == "cat, solo, cat, cat"   # 件数も位置も元どおり


# ---------------------------------------------------------------------------
# REQ-3.1 / NFR-10: 部分成功を全成功と偽らない
# ---------------------------------------------------------------------------


def test_partial_failure_moves_history_and_reports_targets(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    _write(a, "x")
    _write(b, "x")

    manager = U.UndoManager()
    manager.push(U.BulkAddTagsAction(file_tags=[(a, ["y"]), (b, ["y"])], position="append"))
    _write(a, "x, y")
    _write(b, "x, y")
    b.unlink()                                # 片方だけ消える（一時的失敗）

    assert manager.undo() is True
    assert manager.last_result is U.UndoResult.PARTIAL
    assert manager.last_failed_paths == [b]
    assert manager.undo_stack == []           # 履歴は進む
    assert manager.redo_stack                 # redo できる
    assert _read(a) == "x"                    # 戻せた分は戻っている


def test_transient_failure_keeps_the_action_for_retry(tmp_path):
    a = tmp_path / "a.txt"
    _write(a, "x")

    manager = U.UndoManager()
    action = U.BulkAddTagsAction(file_tags=[(a, ["y"])], position="append")
    manager.push(action)
    a.unlink()                                # 親フォルダは残る = 一時的失敗

    assert manager.undo() is False
    assert manager.undo_stack == [action]     # 再試行できるよう先頭へ戻す


# ---------------------------------------------------------------------------
# REQ-3.1 / R11-11: 恒久失敗は履歴から外してデッドロックしない
# ---------------------------------------------------------------------------


def test_permanent_failure_is_dropped_so_the_next_undo_works(tmp_path):
    missing_dir = tmp_path / "gone"
    missing = missing_dir / "b.txt"
    missing_dir.mkdir()
    _write(missing, "x")
    other = tmp_path / "a.txt"
    _write(other, "x")

    manager = U.UndoManager()
    manager.push(U.BulkAddTagsAction(file_tags=[(other, ["y"])], position="append"))
    _write(other, "x, y")
    manager.push(U.BulkAddTagsAction(file_tags=[(missing, ["y"])], position="append"))
    _write(missing, "x, y")

    # 対象 .txt と親フォルダの両方が消える = 再試行しても変わらない
    missing.unlink()
    missing_dir.rmdir()

    assert manager.undo() is False
    assert manager.undo_stack and manager.undo_stack[0].file_tags[0][0] == other
    # デッドロックしない: 次の Undo は前のアクションを戻せる
    assert manager.undo() is True
    assert _read(other) == "x"


def test_composite_reports_three_values(tmp_path):
    a = tmp_path / "a.txt"
    missing = tmp_path / "missing.txt"
    _write(a, "x")

    ok = U.BulkAddTagsAction(file_tags=[(a, ["y"])], position="append")
    _write(a, "x, y")
    bad = U.BulkAddTagsAction(file_tags=[(missing, ["y"])], position="append")

    composite = U.CompositeUndoAction([bad, ok])
    result = composite.undo()
    assert result is U.UndoResult.PARTIAL
    assert composite.failed_paths == [missing]
    assert _read(a) == "x"


# ---------------------------------------------------------------------------
# REQ-3.4 / R13-6: 「不在の復元」は自分の書き込みと一致するときだけ
# ---------------------------------------------------------------------------


def test_edit_caption_undo_does_not_unlink_a_file_changed_by_others(tmp_path):
    path = tmp_path / "a.txt"
    _write(path, "someone else wrote this")

    action = U.EditCaptionAction(
        file_path=path, old_text="", new_text="my caption", file_existed_before=False)
    assert action.undo() is False          # 消さない
    assert path.exists()
    assert _read(path) == "someone else wrote this"

    # 自分の書き込みのままなら消す（従来どおり）
    _write(path, "my caption")
    assert action.undo() is True
    assert not path.exists()


def test_snapshot_undo_does_not_unlink_a_file_changed_by_others(tmp_path):
    path = tmp_path / "a.txt"
    _write(path, "detour")

    action = U.OverwriteFileAction(
        file_path=path, previous_content=None, new_content="mine")
    assert action.undo() is False
    assert path.exists()

    _write(path, "mine")
    assert action.undo() is True
    assert not path.exists()


# ---------------------------------------------------------------------------
# REQ-3.7 / R2-9: 新規作成ファイルの Undo は空ファイルを残さない
# ---------------------------------------------------------------------------


def test_add_tags_undo_removes_a_file_it_created(tmp_path):
    path = tmp_path / "a.txt"
    action = U.AddTagsAction(file_path=path, added_tags=["new"])
    assert action.redo() is True                 # 作成
    assert _read(path) == "new"
    assert action.undo() is True
    assert not path.exists()


def test_add_tags_undo_keeps_a_preexisting_file_with_other_tags(tmp_path):
    path = tmp_path / "a.txt"
    _write(path, "solo, new")
    action = U.AddTagsAction(file_path=path, added_tags=["new"])
    assert action.undo() is True
    assert _read(path) == "solo"


# ---------------------------------------------------------------------------
# REQ-3.6 / R9-8: 登録時点の絶対パスに固定される（CWD を変えても正しいファイル）
# ---------------------------------------------------------------------------


def test_actions_pin_abspath_at_construction(tmp_path, monkeypatch):
    folder = tmp_path / "work"
    folder.mkdir()
    path = folder / "a.txt"
    _write(path, "x, y")

    monkeypatch.chdir(folder)
    action = U.BulkAddTagsAction(file_tags=[(Path("a.txt"), ["y"])], position="append")
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)                    # CWD を変える

    assert action.undo() is U.UndoResult.FULL
    assert _read(path) == "x"                   # 元のファイルが戻る
    assert not (other / "a.txt").exists()


# ---------------------------------------------------------------------------
# REQ-3.5 / R20-2・R2-10・R20-4: MainWindow 側の履歴の扱い
# ---------------------------------------------------------------------------


def _qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _make_window(monkeypatch, tmp_path: Path):
    app = _qt_app()
    import main_window
    import onnx_providers

    monkeypatch.setattr(onnx_providers, "has_nvidia_gpu", lambda *a, **k: False)
    monkeypatch.setattr(main_window.MainWindow, "_maybe_prompt_gpu_setup", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_maybe_prompt_update_check", lambda self: None)

    window = main_window.MainWindow()
    app.processEvents()
    window.settings.paths.input_dir = str(tmp_path / "no_such_input_dir")
    window.update_log = lambda *a, **k: None
    return app, window, main_window


def test_folder_switch_clears_the_undo_history(monkeypatch, tmp_path):
    """R20-2: 前フォルダの .txt を Undo で書き換えない。"""
    app, window, _ = _make_window(monkeypatch, tmp_path)
    old = tmp_path / "old"
    old.mkdir()
    new = tmp_path / "new"
    new.mkdir()
    try:
        txt = old / "a.txt"
        _write(txt, "solo, added")
        window.undo_manager.push(U.AddTagsAction(file_path=txt, added_tags=["added"]))
        window._update_undo_redo_buttons()
        assert window.undo_manager.can_undo() is True

        window._handle_folder_drop(str(new))

        assert window.undo_manager.can_undo() is False
        assert window.undo_manager.can_redo() is False
        assert window.undo_button.isEnabled() is False
        assert _read(txt) == "solo, added"   # 前フォルダのファイルは触られない
    finally:
        window.close()
        app.processEvents()


def test_undo_refreshes_the_grid_tag_cache(monkeypatch, tmp_path):
    """R2-10: Undo/Redo 後にグリッドのキャッシュが古いままにならない。"""
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        window._tag_cache = {"a.png": {"solo"}}
        window.grid_view_widget.update_tag_cache(window._tag_cache)
        window._refresh_ui_after_undo_redo()
        app.processEvents()
        assert window.grid_view_widget._tag_cache is window._tag_cache
    finally:
        window.close()
        app.processEvents()


def test_bulk_action_over_the_cap_is_not_recorded(monkeypatch, tmp_path):
    """R20-4: 上限を超えた一括操作は Undo を作らない（既存履歴は保持）。"""
    app, window, _ = _make_window(monkeypatch, tmp_path)
    import constants

    try:
        existing = U.AddTagsAction(file_path=tmp_path / "keep.txt", added_tags=["t"])
        window.undo_manager.push(existing)

        entries = [(tmp_path / f"{i}.txt", ["t"])
                   for i in range(constants.UNDO_BATCH_SNAPSHOT_LIMIT + 1)]
        window._on_bulk_add_completed(entries, ["t"], "append")

        assert window.undo_manager.undo_stack == [existing]
    finally:
        window.close()
        app.processEvents()
