"""Phase 4（データ喪失・不整合の防止）の回帰テスト。

対象: R1-3、R2-4、R2-8、R4-2、R4-4、R4-5、R7-8、R7-9、R12-5、R14-1、R18-2、
R23-2、R30-3、R31-1。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    window.settings.paths.input_dir = str(tmp_path)
    window.update_log = lambda *a, **k: None
    return app, window, main_window


def _make_unreadable(monkeypatch, target: Path) -> None:
    """対象パスだけ読み取り失敗させる（権限では再現できない環境のため）。"""
    real = Path.read_text

    def boom(self, *args, **kwargs):
        if self == target:
            raise OSError("simulated unreadable file")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boom)


# ---------------------------------------------------------------------------
# REQ-4.1 / R1-3・R2-4: 読めない既存内容を「空」として上書きしない
# ---------------------------------------------------------------------------


def test_main_caption_save_skips_when_existing_txt_is_unreadable(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"x")
        txt = image.with_suffix(".txt")
        _write(txt, "original caption")
        monkeypatch.setattr(window, "_is_text_output_mode", lambda: True)
        window._current_caption_image_path = image
        window.caption_text_edit.setPlainText("edited caption")
        _make_unreadable(monkeypatch, txt)

        window._save_current_caption()

        assert txt.read_bytes().decode("utf-8") == "original caption"
        assert window.undo_manager.can_undo() is False   # Undo 記録も作らない
    finally:
        window.close()
        app.processEvents()


def test_grid_caption_save_skips_when_existing_txt_is_unreadable(monkeypatch, tmp_path):
    import grid_view_widget
    from locale_manager import LocaleManager

    app = _qt_app()
    lang = Path(grid_view_widget.__file__).resolve().parent.parent / "lang"
    cell = grid_view_widget.ImageEditCellWidget(LocaleManager("en", lang))
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"x")
        txt = image.with_suffix(".txt")
        _write(txt, "original")
        cell._image_path = image
        cell.set_caption_mode(True)
        cell.caption_edit.setPlainText("changed")
        failures: list[str] = []
        emitted: list = []
        cell.caption_save_failed.connect(failures.append)
        cell.caption_edited.connect(lambda *a: emitted.append(a))
        _make_unreadable(monkeypatch, txt)

        cell._save_caption()

        assert txt.read_bytes().decode("utf-8") == "original"
        assert failures and emitted == []
    finally:
        cell.deleteLater()
        app.processEvents()


def test_grid_add_tag_skips_when_existing_txt_is_unreadable(monkeypatch, tmp_path):
    import grid_view_widget
    from locale_manager import LocaleManager

    _qt_app()
    monkeypatch.setattr(grid_view_widget.QToolTip, "showText", lambda *a, **k: None)
    lang = Path(grid_view_widget.__file__).resolve().parent.parent / "lang"
    cell = grid_view_widget.ImageEditCellWidget(LocaleManager("en", lang))
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"x")
        txt = image.with_suffix(".txt")
        _write(txt, "existing")
        cell._image_path = image
        cell.add_tag_line.setText("new")
        # グリッドのタグ追加は tag_utils 経由で読む（長パス変換を通す共通部品）。
        import tag_utils

        real_read = tag_utils.read_text

        def boom_read(path, *args, **kwargs):
            if Path(path) == txt:
                raise OSError("simulated unreadable file")
            return real_read(path, *args, **kwargs)

        monkeypatch.setattr(tag_utils, "read_text", boom_read)

        cell._add_tag()

        assert txt.read_bytes().decode("utf-8") == "existing"
    finally:
        cell.deleteLater()


# ---------------------------------------------------------------------------
# REQ-4.2 / R4-2: 壊れた VLM JSON を黙って空にせず、保存前に原本を保全する
# ---------------------------------------------------------------------------


def test_broken_connections_json_is_reported_and_backed_up(monkeypatch, tmp_path):
    import vlm_config

    path = tmp_path / "vlm_connections.json"
    path.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(vlm_config, "VLM_CONNECTIONS_PATH", path)

    assert vlm_config.load_custom_connections() == []
    problems = vlm_config.load_problems()
    assert "vlm_connections.json" in problems

    assert vlm_config.save_custom_connections([{"connection_id": "c1"}]) is True
    assert vlm_config.load_custom_connections() == [{"connection_id": "c1"}]
    assert vlm_config.load_problems() == {}

    backups = list(tmp_path.glob("vlm_connections.json.broken-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{ this is not json"


def test_broken_profiles_json_is_reported(monkeypatch, tmp_path):
    import vlm_config

    path = tmp_path / "vlm_profiles.json"
    path.write_text(json.dumps({"version": 1, "profiles": "oops"}), encoding="utf-8")
    monkeypatch.setattr(vlm_config, "VLM_PROFILES_PATH", path)

    assert vlm_config.load_user_profiles() == []
    assert "vlm_profiles.json" in vlm_config.load_problems()


# ---------------------------------------------------------------------------
# REQ-4.3: flush の非対称をなくす
# ---------------------------------------------------------------------------


def test_same_path_reload_keeps_unsaved_caption(monkeypatch, tmp_path):
    """R12-5: リサイズ（同一パスの再ロード）で未保存編集を失わない。"""
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"x")
        _write(image.with_suffix(".txt"), "from disk")
        monkeypatch.setattr(window, "_is_text_output_mode", lambda: True)
        window._load_caption_for_image(image)
        window.caption_text_edit.setPlainText("typing...")

        window._load_caption_for_image(image)      # 同じパス（リサイズ相当）

        assert window.caption_text_edit.toPlainText() == "typing..."
    finally:
        window.close()
        app.processEvents()


def test_close_event_flushes_the_pending_caption(monkeypatch, tmp_path):
    """R4-4: 終了時にデバウンス中の編集を取りこぼさない。"""
    app, window, _ = _make_window(monkeypatch, tmp_path)
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    txt = image.with_suffix(".txt")
    _write(txt, "old")
    monkeypatch.setattr(window, "_is_text_output_mode", lambda: True)
    window._current_caption_image_path = image
    window._caption_loaded_text = "old"
    window.caption_text_edit.setPlainText("final words")

    window.close()
    app.processEvents()

    assert txt.read_text(encoding="utf-8") == "final words"


def test_switching_to_grid_flushes_the_main_caption(monkeypatch, tmp_path):
    """R18-2: グリッドへ切り替える前にメイン画面の編集を確定する。"""
    app, window, _ = _make_window(monkeypatch, tmp_path)
    calls: list[bool] = []
    monkeypatch.setattr(window, "_save_current_caption", lambda: calls.append(True))
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    try:
        # 画像 0 件の早期 return でも flush の順序を壊さないことだけを見る
        window._show_grid_view()
        assert calls == [True]
    finally:
        window.close()
        app.processEvents()


def test_grid_cell_flushes_before_clearing_reused_cell(monkeypatch, tmp_path):
    """R2-8: セル再利用で未保存のキャプションを捨てない。"""
    import grid_view_widget
    from locale_manager import LocaleManager

    app = _qt_app()
    lang = Path(grid_view_widget.__file__).resolve().parent.parent / "lang"
    cell = grid_view_widget.ImageEditCellWidget(LocaleManager("en", lang))
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"x")
        txt = image.with_suffix(".txt")
        _write(txt, "old")
        cell._image_path = image
        cell.set_caption_mode(True)
        cell.caption_edit.setPlainText("edited")
        cell.clear_data()
        app.processEvents()
        assert txt.read_text(encoding="utf-8") == "edited"
    finally:
        cell.deleteLater()
        app.processEvents()


def test_grid_caption_reload_keeps_unsaved_edit(monkeypatch, tmp_path):
    """R23-2 / R14-1: 言語切替・モード再入の再読込で編集中のテキストを失わない。"""
    import grid_view_widget
    from locale_manager import LocaleManager

    _qt_app()
    lang = Path(grid_view_widget.__file__).resolve().parent.parent / "lang"
    cell = grid_view_widget.ImageEditCellWidget(LocaleManager("en", lang))
    try:
        image = tmp_path / "a.png"
        image.write_bytes(b"x")
        _write(image.with_suffix(".txt"), "disk text")
        cell._image_path = image
        cell.set_caption_mode(True)
        cell.caption_edit.setPlainText("typing in the cell")

        cell._load_caption_text()        # 言語切替・再入の共通入口
        assert cell.caption_edit.toPlainText() == "typing in the cell"
    finally:
        cell.deleteLater()


# ---------------------------------------------------------------------------
# REQ-4.4 / R7-9: プログラム的な表示で自動保存タイマーを武装させない
# ---------------------------------------------------------------------------


def test_vlm_single_test_result_does_not_arm_the_autosave(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        window._caption_save_timer.stop()
        window._on_vlm_single_test_result("generated caption", "conn", "model")
        app.processEvents()
        assert window.caption_text_edit.toPlainText() == "generated caption"
        assert window._caption_save_timer.isActive() is False
        assert window._caption_loaded_text == "generated caption"
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-4.5 / R7-8: フォルダ切替の早期 return で前フォルダの状態を残さない
# ---------------------------------------------------------------------------


def test_empty_folder_switch_discards_previous_state(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        monkeypatch.setattr(window, "_is_text_output_mode", lambda: True)
        previous = tmp_path / "previous.png"
        previous.write_bytes(b"x")
        _write(previous.with_suffix(".txt"), "text")
        window._load_caption_for_image(previous)
        window._current_image_tags = ["solo"]
        window._all_tags = [("solo", 1)]
        window._tag_cache = {"previous.png": {"solo"}}

        window.settings.paths.input_dir = str(tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        window.reload_image_list()

        assert window._current_caption_image_path is None
        assert window.caption_text_edit.toPlainText() == ""
        assert window._current_image_tags == []
        assert window._all_tags == []
        assert window._tag_cache == {}
        assert window.grid_view_widget._tag_cache == {}
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-4.6 / R30-3: disk 側で落ちた verified をメモリへ反映する
# ---------------------------------------------------------------------------


def test_model_status_changed_reloads_verified_models_from_disk(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        import app_settings

        settings = app_settings.load_settings(app_settings.get_default_config())
        settings.model.verified_models = {settings.model.model_id: False}
        assert app_settings.save_config(settings) is True
        window.settings.model.verified_models = {settings.model.model_id: True}

        window._on_model_status_changed()

        assert window.settings.model.verified_models.get(settings.model.model_id) is False
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-4.7 / R31-1: 正本の CSV が現れたら、非空のフォールバックマップでも読み直す
# ---------------------------------------------------------------------------


def test_tag_map_is_reloaded_when_the_primary_csv_appears(monkeypatch, tmp_path):
    app, window, main_window = _make_window(monkeypatch, tmp_path)
    try:
        model_dir = tmp_path / "models"
        monkeypatch.setattr(main_window.constants, "MODEL_PATH", model_dir / "model.onnx")

        loaded: list[int] = []

        def fake_load(*_args, **_kwargs):
            loaded.append(1)
            return {"cat": ["neko"]} if (model_dir / "selected_tags.csv").is_file() else {}

        monkeypatch.setattr(main_window, "load_tag_translation_map", fake_load)
        window.tag_translation_map = {}          # 初回は正本が無い
        window._tag_map_stamp = None
        assert window._refresh_shared_tag_translation_map() is False

        # 非空のフォールバックマップが入っている状態で正本が現れる
        window.tag_translation_map = {"fallback": ["x"]}
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "selected_tags.csv").write_text("id,tag_id,name\n", encoding="utf-8")

        assert window._refresh_shared_tag_translation_map() is True
        assert window.tag_translation_map == {"cat": ["neko"]}
        assert len(loaded) == 2
    finally:
        window.close()
        app.processEvents()
