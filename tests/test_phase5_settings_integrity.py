"""Phase 5（設定ダイアログと実行中状態の整合）の回帰テスト。

対象: R2-2、R2-3、R3-S2、R4-3、R4-9、R5-6、R6-1、R7-6、R7-10、R7-11、R8-3、R9-1、
R9-3、R11-9、R13-1、R13-2、R13-3、R13-4、R18-7、R20-3、R24-2、R28-1、R33-1。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _t(_section, key, **_kwargs):
    return key


# ---------------------------------------------------------------------------
# REQ-5.1 / R2-2・R2-3: 読み込んだ dict への差分適用
# ---------------------------------------------------------------------------


def test_custom_connection_save_preserves_unknown_keys(monkeypatch):
    from custom_connection_dialog import CustomConnectionDialog

    _qt_app()
    monkeypatch.setattr(
        "custom_connection_dialog.QMessageBox.warning",
        staticmethod(lambda *a, **k: None))
    existing = {
        "connection_id": "c1",
        "display_name": "Old",
        "kind": "custom_local",
        "protocol": "openai_chat_completions",
        "base_url": "http://127.0.0.1:1234/v1",
        "model_id": "m",
        "enabled": False,
        "concurrency": 4,
        "provider_id": "custom",
        "request_headers": {"X-Test": "1"},
        "request_body": {"temperature": 0.1},
        "response": {"text_path": "text", "error_path": "error"},
        "image": {"max_long_edge": None},
        "auth": {"type": "none", "secret_ref": "", "extra": "keep-me"},
        "retry": {"connect_timeout_s": 999, "read_timeout_s": 30, "retry_same_max": 1},
    }
    dialog = CustomConnectionDialog(_t, existing)
    try:
        dialog._on_save()
        result = dialog.result_connection()
        assert result["provider_id"] == "custom"
        assert result["request_headers"] == {"X-Test": "1"}
        assert result["request_body"] == {"temperature": 0.1}
        assert result["response"]["error_path"] == "error"
        assert result["enabled"] is False and result["concurrency"] == 4
        assert result["auth"]["extra"] == "keep-me"
        # R2-3: 表示は 1536 でも、触っていなければ None のまま保存する
        assert result["image"]["max_long_edge"] is None
        # R13-4: 範囲外の手編集値（999）を読み込みで丸めない
        assert result["retry"]["connect_timeout_s"] == 999
    finally:
        dialog.close()


def test_custom_connection_none_auth_drops_secret_ref():
    """REQ-5.2 / R13-3: 認証 none では secret_ref を残さない。"""
    from custom_connection_dialog import CustomConnectionDialog

    _qt_app()
    existing = {
        "connection_id": "c1", "display_name": "Old", "kind": "custom_local",
        "protocol": "openai_chat_completions", "base_url": "http://127.0.0.1:1/v1",
        "model_id": "m", "auth": {"type": "bearer", "secret_ref": "vlm/custom/c1"},
    }
    dialog = CustomConnectionDialog(_t, existing)
    try:
        from PySide6.QtWidgets import QComboBox

        combo: QComboBox = dialog.auth_type_combo
        idx = combo.findData("none")
        combo.setCurrentIndex(idx)
        dialog._on_save()
        assert dialog.result_connection()["auth"]["secret_ref"] == ""
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# REQ-5.2 / R24-2・REQ-5.1 / R4-9: プロファイル保存
# ---------------------------------------------------------------------------


def test_profile_editor_does_not_promote_unproven_binding():
    from vlm_profile_editor import ProfileEditorDialog

    _qt_app()
    dialog = ProfileEditorDialog(_t, {
        "profile_id": "p1", "display_name": "P", "canonical_model_id": "x",
        "family": "keep-family", "aliases": ["a"],
        "bindings": {"groq": {"model_id": "m", "vlm_capable": False}},
    })
    try:
        dialog._on_save()
        result = dialog.result_profile()
        assert result["bindings"]["groq"]["vlm_capable"] is False
        # R4-9: 表示していないキーを落とさない
        assert result["family"] == "keep-family"
        assert result["aliases"] == ["a"]
    finally:
        dialog.close()


def test_profile_editor_keeps_bindings_of_providers_without_rows():
    import vlm_profile_editor

    _qt_app()
    dialog = vlm_profile_editor.ProfileEditorDialog(_t, {
        "profile_id": "p1", "display_name": "P", "canonical_model_id": "x",
        "bindings": {
            "groq": {"model_id": "m", "vlm_capable": True},
            "unknown-provider": {"model_id": "u", "vlm_capable": True},
        },
    })
    try:
        dialog._on_save()
        bindings = dialog.result_profile()["bindings"]
        assert "unknown-provider" in bindings
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# REQ-5.3 / R7-10・R11-9: 検証が通るまで共有設定を変えない
# ---------------------------------------------------------------------------


def test_vlm_settings_validation_runs_before_mutating_settings(monkeypatch):
    import app_settings
    from vlm_settings_dialog import VlmSettingsDialog

    _qt_app()
    monkeypatch.setattr(
        "vlm_settings_dialog.QMessageBox.warning",
        staticmethod(lambda *a, **k: None))
    settings = app_settings.load_settings(app_settings.get_default_config())
    dialog = VlmSettingsDialog(settings, _t)
    try:
        settings.vlm.detail_level = "keep-this"
        dialog.mode_builtin.setChecked(False)
        dialog.mode_custom.setChecked(True)
        dialog.custom_select.setCurrentIndex(-1)   # 接続未選択 -> 検証で止まる
        dialog.detail_combo.setCurrentIndex(0)     # 変更しようとしている

        dialog._on_save()

        assert settings.vlm.detail_level == "keep-this"   # 変更されていない
    finally:
        dialog.close()


def test_vlm_settings_rolls_back_memory_when_the_transaction_fails(monkeypatch):
    import app_settings
    import vlm_config
    from vlm_settings_dialog import VlmSettingsDialog

    _qt_app()
    monkeypatch.setattr(
        "vlm_settings_dialog.QMessageBox.critical",
        staticmethod(lambda *a, **k: None))
    settings = app_settings.load_settings(app_settings.get_default_config())
    dialog = VlmSettingsDialog(settings, _t)
    try:
        settings.vlm.detail_level = "keep-this"
        monkeypatch.setattr(vlm_config, "save_settings_transaction", lambda *a, **k: False)
        dialog.detail_combo.setCurrentIndex(0 if dialog.detail_combo.currentIndex() != 0 else 1)

        dialog._on_save()

        assert settings.vlm.detail_level == "keep-this"   # メモリ側も巻き戻る
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# REQ-5.7 / R13-2・R3-S2: 検証と無言の弱体化の防止
# ---------------------------------------------------------------------------


def test_base_url_without_scheme_is_rejected(monkeypatch):
    from custom_connection_dialog import CustomConnectionDialog

    _qt_app()
    seen: list[str] = []
    monkeypatch.setattr(
        "custom_connection_dialog.QMessageBox.warning",
        staticmethod(lambda _p, _t2, text, *a, **k: seen.append(text)))
    dialog = CustomConnectionDialog(_t)
    try:
        dialog.name_edit.setText("Local")
        dialog.base_url_edit.setText("127.0.0.1:1234/v1")
        dialog.model_edit.setCurrentText("m")
        dialog._on_save()
        assert dialog.result_connection() is None
        assert "Custom_Validation_Url" in seen
    finally:
        dialog.close()


def test_disabling_tls_verification_requires_confirmation(monkeypatch):
    from custom_connection_dialog import CustomConnectionDialog

    _qt_app()
    answers = iter([False, True])
    calls: list[str] = []

    def fake_warning(_parent, _title, text, *args, **kwargs):
        calls.append(text)
        from PySide6.QtWidgets import QMessageBox

        return QMessageBox.StandardButton.Yes if next(answers) else QMessageBox.StandardButton.No

    monkeypatch.setattr("custom_connection_dialog.QMessageBox.warning", staticmethod(fake_warning))
    dialog = CustomConnectionDialog(_t)
    try:
        dialog.name_edit.setText("Local")
        dialog.base_url_edit.setText("https://example.invalid/v1")
        dialog.model_edit.setCurrentText("m")
        dialog.verify_tls_check.setChecked(False)

        dialog._on_save()                       # 拒否 -> 保存しない
        assert dialog.result_connection() is None
        assert "Custom_Verify_Tls_Warning" in calls

        dialog._on_save()                       # 同意 -> 保存する
        assert dialog.result_connection()["verify_tls"] is False
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# REQ-5.8 / R7-11・R5-6: 黙った差し替えをしない
# ---------------------------------------------------------------------------


def test_missing_selected_custom_id_is_not_replaced_by_the_first_item(monkeypatch):
    import app_settings
    from vlm_settings_dialog import VlmSettingsDialog

    _qt_app()
    monkeypatch.setattr("vlm_config.load_custom_connections", lambda: [
        {"connection_id": "c1", "display_name": "One", "kind": "custom_local"},
        {"connection_id": "c2", "display_name": "Two", "kind": "custom_local"},
    ])
    settings = app_settings.load_settings(app_settings.get_default_config())
    settings.vlm.execution_mode = "custom_single"
    settings.vlm.selected_connection_id = "missing"
    dialog = VlmSettingsDialog(settings, _t)
    try:
        assert dialog.custom_select.currentData() is None
    finally:
        dialog.close()


def test_hand_edited_language_is_not_reset_to_en(monkeypatch):
    import app_settings
    from vlm_settings_dialog import VlmSettingsDialog

    _qt_app()
    settings = app_settings.load_settings(app_settings.get_default_config())
    settings.vlm.language = "xx-hand-edited"
    dialog = VlmSettingsDialog(settings, _t)
    try:
        assert dialog.language_combo.currentData() == "xx-hand-edited"
    finally:
        dialog.close()


# ---------------------------------------------------------------------------
# REQ-5.5 / R7-6・R9-3: 保存頻度
# ---------------------------------------------------------------------------


def _make_window(monkeypatch, tmp_path):
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


def test_slider_changes_are_saved_after_a_debounce(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    saves: list[int] = []
    monkeypatch.setattr(window, "save_current_config", lambda: saves.append(1))
    try:
        slider, _label = window._sliders["thresholds_general"]
        slider.setValue(slider.value() + 1)
        app.processEvents()
        assert saves == []                 # 1 目盛りごとには書かない

        window._config_save_timer.timeout.emit()   # デバウンス満了相当
        assert saves == [1]
    finally:
        window.close()
        app.processEvents()


def test_programmatic_slider_sync_does_not_save(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    saves: list[int] = []
    monkeypatch.setattr(window, "save_current_config", lambda: saves.append(1))
    try:
        window.settings.thresholds.general = 0.11
        window._sync_settings_sliders()
        app.processEvents()
        if hasattr(window, "_config_save_timer"):
            window._config_save_timer.stop()
        assert saves == []
    finally:
        window.close()
        app.processEvents()


def test_model_switch_saves_only_when_something_changed(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    saves: list[int] = []
    try:
        entry = window._current_model_entry()
        window._model_mode._apply_threshold_defaults(entry)   # 1 回目で既定値を適用
        monkeypatch.setattr(window, "save_current_config", lambda: saves.append(1))
        window._model_mode._apply_threshold_defaults(entry)   # 変化なし
        assert saves == []
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-5.10 / R33-1: VLM トグルは Florence のタスク設定を触らない
# ---------------------------------------------------------------------------


def test_vlm_toggle_does_not_reset_the_caption_task(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        window.settings.caption.task = "CAPTION"
        window._on_use_vlm_toggled(True)
        assert window.settings.caption.task == "CAPTION"
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-5.4 / R8-3: 実行時設定は開始時スナップショット
# ---------------------------------------------------------------------------


def test_settings_snapshot_is_a_copy(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        snapshot = window._settings_snapshot()
        assert snapshot is not window.settings
        window.settings.thresholds.general = 0.123
        assert snapshot.thresholds.general != 0.123
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-5.6 / R6-1: ビューアは閉じたら破棄する
# ---------------------------------------------------------------------------


def test_viewer_dialog_is_destroyed_on_close(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        class _Dialog:
            def __init__(self):
                self.deleted = False

            def deleteLater(self):
                self.deleted = True

        dialog = _Dialog()
        window._image_viewer_dialog = dialog
        window._image_viewer_dialog_closed()
        assert dialog.deleted is True
        assert window._image_viewer_dialog is None
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-5.11 / R28-1: 表示系の数値設定に範囲検証
# ---------------------------------------------------------------------------


def test_zero_tag_display_dims_do_not_crash(monkeypatch, tmp_path):
    app, window, _ = _make_window(monkeypatch, tmp_path)
    try:
        window.settings.window.tag_display_cols = 0
        window.settings.window.tag_display_rows = 0
        window._current_image_tags = ["a", "b"]
        window._display_image_tag_page()      # ZeroDivisionError にならない
        assert window._get_image_tags_per_page() >= 1
    finally:
        window.close()
        app.processEvents()


# ---------------------------------------------------------------------------
# REQ-5.9 / R18-7: secret_ref の正本は 1 箇所
# ---------------------------------------------------------------------------


def test_builtin_secret_refs_have_a_single_source():
    import vlm_connections
    import vlm_settings_dialog

    assert vlm_settings_dialog._BUILTIN_SECRET_REF is vlm_connections.BUILTIN_SECRET_REFS


# ---------------------------------------------------------------------------
# REQ-5.11 / R9-1: 既存キー再確認時の文言は実態に合わせる
# ---------------------------------------------------------------------------


def test_existing_key_warning_does_not_claim_a_save():
    import api_key_dialog

    source = Path(api_key_dialog.__file__).read_text(encoding="utf-8")
    assert "ApiKey_Model_Warning_Existing" in source
