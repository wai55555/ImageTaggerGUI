"""Phase 1 のテスト基盤そのものの検査（REQ-1.2 / REQ-1.4）と、
これまでどのテストからも import されていなかった 3 モジュールの最小テスト。

- REQ-1.2: MainWindow を組み立てるテストは、実 config.ini と実 input_dir に
  触れないことをテスト自体で保証する。隔離の実体は tests/conftest.py。
- REQ-1.4（R11-5）: custom_widgets / model_mode_controller /
  vlm_prompt_preview_dialog は import されるだけでテストが無かった。ここで
  主要経路を固定する。tag_utils と grid_view_widget は
  tests/test_phase2_atomic_writes.py が担当しているので重複して作らない。
  F（拡大表示の配線）の回帰テストは Phase 8B の担当であり、本ファイルは
  grid_view_widget を import できる置き場があることの確認に留める。
"""
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QMimeData, Qt, QUrl
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[1]


def _qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _inside_repo(path) -> bool:
    try:
        Path(path).resolve().relative_to(REPO_ROOT.resolve())
    except (ValueError, OSError):
        return False
    return True


# --- REQ-1.2: MainWindow の隔離 -------------------------------------------------

def test_real_state_guard_covers_the_user_files(real_state):
    """実データ指紋（completion 判定の「前後で変化しない」確認）が機能していること。

    conftest の autouse fixture がテストごとに同じ指紋を比べている。ここでは
    監視対象に実 config.ini / .env / 実フォルダが入っていることを確かめる。
    """
    snapshot = real_state()
    assert {"config.ini", ".env", "vlm_connections.json", "vlm_profiles.json",
            "inputs", "models"} <= set(snapshot)
    # このテスト自身が何も書き換えていない。
    assert real_state() == snapshot


def test_main_window_is_built_against_isolated_config_and_input_dir(monkeypatch, tmp_path):
    """MainWindow 構築が実 config.ini / 実 inputs/ を読まない（R11-2）。"""
    import app_settings
    import constants
    import main_window
    import onnx_providers
    import tagging_core
    import utils
    import vlm_config

    monkeypatch.setattr(onnx_providers, "has_nvidia_gpu", lambda *a, **k: False)
    # 起動時プロンプト／更新チェックはこのテストの対象外（実ダイアログ・ネットワークを避ける）。
    monkeypatch.setattr(main_window.MainWindow, "_maybe_prompt_gpu_setup", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_maybe_prompt_update_check", lambda self: None)

    write_targets = {
        "app_settings.CONFIG_PATH": app_settings.CONFIG_PATH,
        "constants.CONFIG_PATH": constants.CONFIG_PATH,
        "utils.CONFIG_PATH": utils.CONFIG_PATH,
        "tagging_core.CONFIG_PATH": tagging_core.CONFIG_PATH,
        "vlm_config.VLM_CONNECTIONS_PATH": vlm_config.VLM_CONNECTIONS_PATH,
        "vlm_config.VLM_PROFILES_PATH": vlm_config.VLM_PROFILES_PATH,
    }
    for name, path in write_targets.items():
        assert not _inside_repo(path), f"{name} が実ファイルを指している: {path}"

    app = _qt_app()
    window = main_window.MainWindow()
    try:
        for _ in range(5):
            app.processEvents()
        input_dir = Path(window.settings.paths.input_dir)
        assert not _inside_repo(input_dir), (
            f"input_dir がリポジトリ内（実フォルダ）を指している: {input_dir}"
        )
        assert input_dir != REPO_ROOT / "inputs"
        # closeEvent の保存も隔離先へ書かれる（実 config.ini は変化しない）。
        assert not _inside_repo(app_settings.CONFIG_PATH)
    finally:
        window.close()
        app.processEvents()


def test_grid_view_widget_is_importable_from_the_shared_test_base():
    """Phase 8B の F（拡大表示）回帰テストの置き場があること（REQ-1.4）。"""
    import grid_view_widget

    assert hasattr(grid_view_widget, "GridViewWidget")
    assert hasattr(grid_view_widget, "ImageEditCellWidget")


# --- REQ-1.4: custom_widgets ----------------------------------------------------

class _DropEventStub:
    """`dropEvent()` に渡す最小のイベント（QMimeData は本物）。

    PySide6 + Windows では QDropEvent を自前で生成して直接 `dropEvent()` へ渡すと
    プロセスが fail-fast（0xC0000409）するため、Qt のイベント実体は使わず、テスト
    対象である `PathLineEdit.dropEvent` の分岐だけを本物の URL で駆動する。
    """

    def __init__(self, path: Path):
        self._mime = QMimeData()
        self._mime.setUrls([QUrl.fromLocalFile(str(path))])
        self.accepted = False

    def mimeData(self) -> QMimeData:
        return self._mime

    def acceptProposedAction(self) -> None:
        self.accepted = True

    def ignore(self) -> None:
        self.accepted = False


def _post_key(widget, key) -> None:
    QApplication.postEvent(
        widget, QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))
    _qt_app().processEvents()


def test_path_line_edit_accepts_a_local_folder_drop(tmp_path):
    from custom_widgets import PathLineEdit

    folder = tmp_path / "pictures"
    folder.mkdir()
    edit = PathLineEdit(get_string=lambda section, key, **kw: key)
    dropped: list[str] = []
    edit.folder_dropped.connect(dropped.append)

    event = _DropEventStub(folder)
    edit.dropEvent(event)

    assert Path(edit.text()) == folder
    assert [Path(item) for item in dropped] == [folder]


def test_path_line_edit_ignores_a_non_folder_drop(tmp_path):
    from custom_widgets import PathLineEdit

    file_path = tmp_path / "a.txt"
    file_path.write_text("x", encoding="utf-8")
    edit = PathLineEdit()
    dropped: list[str] = []
    edit.folder_dropped.connect(dropped.append)

    event = _DropEventStub(file_path)
    edit.dropEvent(event)

    assert edit.text() == ""
    assert event.accepted is False
    assert dropped == []


def test_tag_list_widget_moves_selection_with_keyboard():
    from custom_widgets import TagListWidget

    widget = TagListWidget()
    widget.addItems(["a", "b", "c"])
    clicked: list[object] = []
    widget.itemClicked.connect(clicked.append)

    widget.setCurrentRow(0)
    _post_key(widget, Qt.Key.Key_Down)
    assert widget.currentRow() == 1
    assert clicked and clicked[-1].text() == "b"

    _post_key(widget, Qt.Key.Key_Up)
    assert widget.currentRow() == 0


def test_tag_list_widget_stops_at_the_edges():
    from custom_widgets import TagListWidget

    widget = TagListWidget()
    widget.addItems(["only"])
    widget.setCurrentRow(0)

    _post_key(widget, Qt.Key.Key_Down)
    assert widget.currentRow() == 0

    _post_key(widget, Qt.Key.Key_Up)
    assert widget.currentRow() == 0


# --- REQ-1.4: model_mode_controller ---------------------------------------------

def _controller_stub():
    import app_settings

    settings = app_settings.load_settings(app_settings.get_default_config())
    stub = SimpleNamespace(settings=settings, _sliders={},
                           sync_calls=[], save_calls=[])
    stub._sync_settings_sliders = lambda: stub.sync_calls.append(True)
    stub.save_current_config = lambda: stub.save_calls.append(True) or True
    return stub


def _entry(config: dict):
    from model_registry import ModelEntry

    return ModelEntry(model_id="test-model", model_name="Test", model_type="tagger",
                      model_dir=Path("/tmp/test-model"), config=config)


def test_model_default_threshold_and_limit_resolution():
    from model_mode_controller import ModelModeController

    controller = ModelModeController(_controller_stub())

    ui_cfg = {"categories": ["general", "character"],
              "default_threshold": 0.55,
              "default_thresholds": {"general": 0.7},
              "default_limits": {"character": 3}}
    assert controller._model_default_threshold(ui_cfg, "general") == 0.7
    assert controller._model_default_threshold(ui_cfg, "character") == 0.55
    assert controller._model_default_threshold(ui_cfg, "rating") == 0.55
    assert controller._model_default_limit(ui_cfg, "character") == 3
    assert controller._model_default_limit(ui_cfg, "general") == 55

    # 明示が無ければフォールバック既定（モデル非依存）。
    assert controller._model_default_threshold({}, "general") == 0.40
    assert controller._model_default_threshold({}, "character") == 0.65
    assert controller._model_default_limit({}, "general") == 55


def test_apply_threshold_defaults_respects_touched_and_saves(tmp_path):
    import app_settings
    from model_mode_controller import ModelModeController

    stub = _controller_stub()
    controller = ModelModeController(stub)
    entry = _entry({"ui": {"categories": ["general", "character"],
                           "default_threshold": 0.55,
                           "default_thresholds": {"general": 0.7},
                           "default_limits": {"character": 2}}})

    controller._apply_threshold_defaults(entry)

    assert stub.settings.thresholds.general == 0.7
    assert stub.settings.thresholds.character == 0.55
    assert stub.settings.limits.character == 2
    assert stub.sync_calls == [True]
    assert stub.save_calls == [True]

    # 手動で触ったカテゴリは上書きしない（touched の意味論）。
    stub.settings.thresholds.touched = "general"
    stub.settings.limits.touched = "character"
    stub.settings.thresholds.general = 0.11
    stub.settings.limits.character = 9
    controller._apply_threshold_defaults(entry)

    assert stub.settings.thresholds.general == 0.11
    assert stub.settings.limits.character == 9
    assert app_settings.parse_touched("general") == {"general"}


def test_apply_character_ui_visibility_disables_unsupported_controls():
    from PySide6.QtWidgets import QLabel, QSlider

    from model_mode_controller import ModelModeController

    stub = _controller_stub()
    slider = QSlider()
    label = QLabel()
    stub._sliders = {"thresholds_character": (slider, label)}
    controller = ModelModeController(stub)

    controller._apply_character_ui_visibility(
        _entry({"ui": {"supports_character_tag": False}}))
    assert slider.isEnabled() is False
    assert label.isEnabled() is False

    controller._apply_character_ui_visibility(
        _entry({"ui": {"supports_character_tag": True}}))
    assert slider.isEnabled() is True
    assert label.isEnabled() is True


# --- REQ-1.4: vlm_prompt_preview_dialog -----------------------------------------

def _preview(custom: bool = False):
    from vlm_prompt_preview import PromptPreview, PromptPreviewSetting

    return PromptPreview(
        profile_id="test-profile",
        system_prompt="SYSTEM",
        user_prompt="USER",
        settings=(
            PromptPreviewSetting(name="prompt_mode", value="standard",
                                 destination="prompt"),
            PromptPreviewSetting(name="image_max_long_edge", value=None,
                                 destination="image_preprocess"),
        ),
        routes=(),
        placements=(),
        custom_system_prompt_active=custom,
    )


def _label_texts(dialog) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [label.text() for label in dialog.findChildren(QLabel)]


def test_vlm_prompt_preview_dialog_renders_the_prompt_and_settings():
    from locale_manager import LocaleManager
    from vlm_prompt_preview_dialog import VlmPromptPreviewDialog

    _qt_app()
    locale = LocaleManager("en", REPO_ROOT / "lang")
    dialog = VlmPromptPreviewDialog(_preview(), locale.get_string)
    try:
        assert dialog.windowTitle() == locale.get_string("Vlm", "PromptPreview_Title")
        assert dialog.system_edit.toPlainText() == "SYSTEM"
        assert dialog.user_edit.toPlainText() == "USER"
        texts = _label_texts(dialog)
        # None の設定値は「プロバイダー既定」と表示される（実装の None 意味論）。
        provider_default = locale.get_string("Vlm", "PromptPreview_ProviderDefault")
        assert any(text.startswith(provider_default) for text in texts), texts
        # 既定プロンプトのときはカスタム上書きの注記を出さない。
        assert locale.get_string("Vlm", "PromptPreview_CustomOverride") not in texts
    finally:
        dialog.close()


def test_vlm_prompt_preview_dialog_marks_a_custom_system_prompt():
    from locale_manager import LocaleManager
    from vlm_prompt_preview_dialog import VlmPromptPreviewDialog

    _qt_app()
    locale = LocaleManager("en", REPO_ROOT / "lang")
    dialog = VlmPromptPreviewDialog(_preview(custom=True), locale.get_string)
    try:
        texts = _label_texts(dialog)
        assert locale.get_string("Vlm", "PromptPreview_CustomOverride") in texts
    finally:
        dialog.close()
