import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from PySide6.QtCore import (
    Q_ARG,
    Q_RETURN_ARG,
    QEvent,
    QMetaObject,
    QObject,
    QPoint,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QKeyEvent,
    QPalette,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QWidget,
)

import app_settings  # Added import
import constants
import tag_utils
import vlm_config
from app_settings import load_config, load_settings, save_config  # Updated import
from custom_dialogs import ClickableLabel, ImageViewerDialog
from custom_widgets import PathLineEdit, TagListWidget
from grid_view_widget import GridViewWidget
from locale_manager import (
    LocaleManager,
    available_language_codes,
    normalize_language_code,
)
from model_mode_controller import ModelModeController
from model_registry import ModelEntry, config_mapping, discover_models, get_model_entry
from settings_binding import SettingsBinding
from startup_prompts import StartupPrompts
from tag_panel import TagPanel
from tag_utils import load_tag_translation_map
from tagging_core import ExistingFileMode, OverwriteDecision
from text_io import atomic_write_text
from ui_main_window import Ui_MainWindow
from undo_manager import (
    AddTagsAction,
    AppendTagsActionV2,
    BulkAddTagsAction,
    BulkRemoveTagsAction,
    CompositeUndoAction,
    EditCaptionAction,
    OverwriteFileAction,
    RemoveTagAction,
    UndoManager,
    UndoResult,
)
from utils import write_debug_log
from vlm_worker import VlmCaptionWorker
from worker_host import WorkerHost
from workers import (
    BulkTagWorker,
    CaptionerThreadWorker,
    DownloaderWorker,
    GpuRuntimeDownloadWorker,
    TaggerThreadWorker,
    TagLoader,
    UpdateCheckWorker,
)

# GetUserDefaultUILanguage() の primary language ID → 言語コード。
# LCIDToLocaleName() が使えなかったときだけ使う保険なので、同梱している9言語ぶん。
_WIN_PRIMARY_LANGID_TO_CODE = {
    0x09: "en", 0x11: "ja", 0x07: "de", 0x0c: "fr",
    0x12: "ko", 0x0a: "es", 0x19: "ru", 0x04: "zh",
}
# 中国語だけは primary language ID では簡体字／繁体字が決まらないので
# sublanguage ID（上位ビット）で分ける。1=TW, 2=CN, 3=HK, 4=SG, 5=MO。
_WIN_CHINESE_SUBLANG_TO_CODE = {
    0x01: "zh_TW", 0x02: "zh_CN", 0x03: "zh_TW", 0x04: "zh_CN", 0x05: "zh_TW",
}


def _os_language_raw() -> str:
    """OS の UI 言語を、正規化前の生のタグ（`ja-JP` / `zh_TW` 等）で返す。"""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # GetUserDefaultUILanguage returns a LANGID, e.g. 0x0411 for ja-JP.
            lang_id = kernel32.GetUserDefaultUILanguage()
            # LCIDToLocaleName は "ja-JP" / "zh-TW" のような BCP-47 名を返す。
            # 自前の LANGID 表では地域が落ちるため、中国語の簡体字／繁体字を
            # 区別できない（zh.ini は存在しない）。まずこちらを使う。
            buffer = ctypes.create_unicode_buffer(85)  # LOCALE_NAME_MAX_LENGTH
            if kernel32.LCIDToLocaleName(lang_id, buffer, len(buffer), 0) and buffer.value:
                return buffer.value
            primary = lang_id & 0x3FF
            if primary == 0x04:
                return _WIN_CHINESE_SUBLANG_TO_CODE.get(lang_id >> 10, "zh")
            return _WIN_PRIMARY_LANGID_TO_CODE.get(primary, "")
        except Exception as e:
            write_debug_log(f"Failed to get OS language via ctypes: {e}")

    # Non-Windows (macOS, Linux), or Windows if ctypes failed.
    # locale.getdefaultlocale() is more reliable in bundled apps than QLocale
    # as it doesn't depend on the QApplication instance's state.
    # 地域まで含めた値をそのまま返す（以前はここで '_' の前だけを取っており、
    # zh_CN / zh_TW が存在しない "zh" に潰れていた）。
    try:
        import locale
        lang_code, _ = locale.getdefaultlocale()
        return lang_code or ""
    except (ImportError, ValueError, IndexError) as e:
        write_debug_log(f"Failed to get OS language via locale: {e}")
        return ""
# 退避スレッドの置き場は worker_host へ移した（REQ-6.1）。既存の参照名は
# モジュール属性として再公開する（下の __getattr__）。
# 更新チェックのリリースページを開くのに使う（実装は startup_prompts）。
# 既存テストが main_window.webbrowser.open を差し替えるため、ここでも束縛する。
import webbrowser  # noqa: F401
import worker_host


def __getattr__(name):
    """Phase 6: 退避リスト関連は worker_host が正本（既存の参照を壊さない）。"""
    if name in {"_detached_threads", "detached_thread_count",
                "wait_for_detached_threads", "_prune_detached_threads"}:
        return getattr(worker_host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_os_language() -> str:
    """OS の UI 言語を、実在する翻訳ファイルの言語コードへ正規化して返す。

    戻り値は必ず `lang/<code>.ini` が存在するコード（見つからなければ "en"）。
    この値は初回起動時に config.ini へ保存され、以後の起動でも使われるため、
    存在しないコードを返すと利用者はずっと英語表示のままになる。
    """
    raw = _os_language_raw()
    codes = available_language_codes(constants.LANG_DIR, constants.LANG_RESOURCE_DIR)
    normalized = normalize_language_code(raw, codes)
    write_debug_log(f"OS language '{raw}' resolved to '{normalized}'")
    return normalized

class StoppableWorker(Protocol): # type: ignore
    """A protocol for worker objects that have a thread-safe stop() method."""
    def stop(self) -> None:
        ...

# --- Main Window ---

class MainWindow(QMainWindow):
    """Main application window for image tagging."""
    # --- UI Elements ---
    central_widget: QStackedWidget
    main_view_widget: QWidget
    grid_view_widget: GridViewWidget
    splitter: QSplitter
    image_list: TagListWidget
    right_vertical_splitter: QSplitter
    input_line: PathLineEdit
    grid_view_button: QPushButton
    image_label: ClickableLabel
    tag_display_grid: QGridLayout
    tag_grid_container: QWidget
    caption_text_edit: QTextEdit
    task_combo: QComboBox
    image_tag_prev_page_btn: QPushButton
    image_tag_next_page_btn: QPushButton
    add_single_tag_line: QLineEdit
    add_single_tag_button: QPushButton
    run_button: QPushButton
    loading_label: QLabel
    tag_button_grid: QGridLayout
    language_combo: QComboBox
    model_combo: QComboBox
    existing_mode_combo: QComboBox
    target_mode_combo: QComboBox
    caption_placement_widget: QWidget
    caption_placement_group: QButtonGroup
    caption_placement_buttons: dict[str, QPushButton]
    prev_page_btn: QPushButton
    next_page_btn: QPushButton
    add_tag_line: QLineEdit
    add_tag_button: QPushButton
    add_tag_line_append: QLineEdit
    add_tag_button_append: QPushButton
    log_output: QTextEdit
    undo_button: QPushButton
    redo_button: QPushButton
    bulk_delete_group: QWidget
    bulk_add_group: QWidget
    tagger_config_block: QWidget
    shared_run_block: QWidget
    category_settings_button: QPushButton
    # Phase 2（R21-1）で錠の対象に加えた GPU トグル（生成は ui_main_window）。
    use_gpu_check: QCheckBox

    # --- Signals ---
    request_overwrite_check = Signal(str, str)
    _request_worker_stop = Signal() # Signal to request workers to stop
    
    def __init__(self):
        super().__init__()
        
        self._initialize_settings_and_locale()
        self._initialize_state()
        
        self.ui = Ui_MainWindow()
        self.ui.setup_ui(self)
        
        self.language_combo.currentIndexChanged.connect(self.toggle_tag_language)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        self.task_combo.currentIndexChanged.connect(self._on_task_combo_changed)

        self._apply_image_list_selection_style()
        self._install_event_filters()

        write_debug_log(self.locale_manager.get_string("MainWindow", "MainWindow_Init_Complete"))

        QTimer.singleShot(0, self.initial_load)

    def _initialize_settings_and_locale(self):
        """Loads configuration and initializes the localization manager."""
        config = load_config()
        self.settings = load_settings(config)

        if not self.settings.language_code:
            os_lang = get_os_language()
            self.settings.language_code = os_lang
            save_config(self.settings)

        # 保存済みの「ライブ一覧で確認したVLMモデルID」をプロセス内カタログへ戻す。
        # is_vlm_model_id() の判定が再起動を跨いで一致し、明示的に選んだ経路が
        # 黙って別経路へ差し替わらないようにする（260922 レビュー指摘）。
        vlm_config.restore_discovered_vlm_ids(self.settings.vlm)

        self.locale_manager = LocaleManager(self.settings.language_code, constants.LANG_DIR, constants.LANG_RESOURCE_DIR)
        app_settings.set_get_string_func(self.locale_manager.get_string) # Add this line
        write_debug_log(self.locale_manager.get_string("MainWindow", "Application_Startup"))

        self._is_dark_theme = QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128
        self._log_color_map = self._get_log_color_map()
        
    def _initialize_state(self):
        """Initializes all state variables for the main window."""
        self._is_shutting_down = False # Flag to prevent race conditions on close
        # Thread and worker management
        self._tagger_thread: QThread | None = None
        self._tagger_worker: TaggerThreadWorker | CaptionerThreadWorker | VlmCaptionWorker | None = None
        self._vlm_settings_dialog = None
        self._download_thread: QThread | None = None
        self._downloader_worker: DownloaderWorker | None = None
        self._gpu_dl_thread: QThread | None = None
        self._gpu_dl_worker: GpuRuntimeDownloadWorker | None = None
        self._gpu_dl_progress: QProgressDialog | None = None
        self._update_check_thread: QThread | None = None
        self._update_check_worker: UpdateCheckWorker | None = None
        self._bulk_tag_thread: QThread | None = None
        self._bulk_tag_worker: BulkTagWorker | None = None
        self.tag_thread: QThread | None = None
        self.tag_worker: TagLoader | None = None

        # UI State variables
        self._all_tags: list[tuple[str, int]] = []
        self._current_page: int = 0  # For bulk tag view
        self._current_image_tags: list[str] = []
        self._current_image_tag_page: int = 0
        self._original_image_pixmap: QPixmap | None = None
        self.tag_buttons: list[QPushButton] = []
        self.tag_buttons_for_image: list[QPushButton] = []

        # Processing State variables
        self._is_downloading = False
        self._is_bulk_deleting = False 
        # ASK セッション中に「常に〜」が選ばれたときの上書き先。タギング開始ごとに
        # None へリセットされ、config.ini の保存値は変更しない（spec.md 1.3節）。
        self._session_mode_override: ExistingFileMode | None = None
        # ワーカーからの進捗通知を GUI へ反映した最後の時刻（issue #10: 毎画像反映すると固まる）。
        self._progress_last_shown_at: float = 0.0
        # 直近の一括バッチで失敗した入力画像パス。次回「失敗のみ」実行の対象。
        # セッション内のみ保持し、フォルダ変更で破棄、再起動は跨がない
        # （260903_vlm-gap-fix.md todo 5）。1枚テストは batch_failed を出さないので上書きされない。
        self._batch_failed_paths: list[Path] = []

        # Timers and Dialogs
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(100)
        # Autosave for the caption editor: commit ~1.2s after the last keystroke so the
        # edit is persisted and the Undo button lights up without waiting for focus-out.
        self._caption_save_timer = QTimer(self)
        self._caption_save_timer.setSingleShot(True)
        self._caption_save_timer.setInterval(constants.CAPTION_AUTOSAVE_DELAY_MS)
        self._caption_save_timer.timeout.connect(self._save_current_caption)
        self.loading_timer: QTimer | None = None
        self.loading_state = 0
        self._sliders: dict[str, tuple[QSlider, QLabel]] = {}
        # Set True while pushing model-default / dialog values into the main sliders so
        # their valueChanged handler doesn't mis-flag those as user edits ("touched").
        self._suppress_slider_touch: bool = False
        # True の間はスライダーの値変更で設定を保存しない（プログラム的同期。
        # REQ-5.5 / R7-6）。
        self._suppress_slider_save: bool = False
        # False while a download/tagging run holds the UI lock; consulted by
        # _update_undo_redo_buttons so a mid-run action push cannot re-enable Undo.
        self._controls_enabled: bool = True
        # 直近に適用した text_ui（captioner / VLM）状態。None は未適用。
        # _rebalance_right_splitter がモード切替時だけ縦割当を組み替えるための差分基準。
        self._last_text_ui: bool | None = None
        self._category_settings_dialog: QDialog | None = None
        self._image_viewer_dialog: ImageViewerDialog | None = None
        
        
        # Undo/Redo Manager
        self.undo_manager = UndoManager(max_history=50)

        # Consolidates all model-switch side effects (design.md 6.8節).
        self._model_mode = ModelModeController(self)
        # 起動時プロンプト（GPU 導入提案・更新チェック）の判定と状態（REQ-6.2）。
        self._startup = StartupPrompts(self)
        # タグ表示・編集パネル（REQ-6.4）。ロック判定の正本は MainWindow のまま。
        self._tag_panel = TagPanel(self)
        # スライダー同期・touched・設定保存（REQ-6.3）。
        self._settings_binding = SettingsBinding(self)
        # スレッド・ワーカーの生成/停止/後始末（REQ-6.1）。
        self._workers = WorkerHost(self)
        
        self.tag_translation_map: dict[str, list[str]] = {}
        # REQ-4.7 / R31-1: 翻訳 CSV の「正本」が現れたかを判定するためのスタンプ。
        self._tag_map_stamp: tuple[bool, float] | None = None
        # キャプション編集欄が最後にロード／保存した内容（REQ-4.3 の dirty 判定）。
        self._caption_loaded_text: str = ""
        self._tag_display_language: str = "English"
        self._current_caption_image_path: Path | None = None

        # Tag cache for hover highlight feature
        self._tag_cache: dict[str, set[str]] = {}

        # Constants
        self._tag_button_min_width = 100
        self._tag_button_min_height = 25

    def _report_vlm_config_problems(self) -> None:
        """REQ-4.2: 壊れた VLM JSON を「黙って空」にしない（起動時の GUI ログ）。"""
        problems = vlm_config.load_problems()
        if not problems:
            return
        names = ", ".join(sorted(problems))
        write_debug_log(f"vlm config problems: {problems}")
        self.update_log(self.locale_manager.get_string(
            "Vlm", "Config_Broken_Log", names=names), "red")

    def initial_load(self):
        """Performs the initial loading of images and tags after the main window is shown."""
        self._report_vlm_config_problems()
        self.tag_translation_map = self._load_shared_tag_translation_map()
        self.reload_image_list()
        self.reload_tags_only()
        self._update_undo_redo_buttons()
        # Apply the persisted model's full UI state exactly as if it had just been picked
        # from the combo: tagger<->captioner layout, character-tag control visibility,
        # this model's recommended threshold defaults, and the run/download button state.
        # Without this, launching with a captioner (Florence-2) persisted shows the tagger
        # UI (no caption box / task selector), and the threshold sliders keep whatever
        # global value was last saved instead of the selected model's own defaults.
        self._model_mode.on_model_changed(self._current_model_entry())
        self._maybe_prompt_gpu_setup()
        # Called after the GPU prompt returns (its QMessageBox.exec() is synchronous),
        # so the update check's worker thread isn't even started until the user has
        # answered it. The update dialog itself is shown later from the worker's
        # finished slot, which additionally waits out any modal still open (e.g. the
        # GPU download progress) - see _show_update_prompt.
        self._maybe_prompt_update_check()

    def _load_shared_tag_translation_map(self) -> dict[str, list[str]]:
        """Load PixAI's shared translations; tags without a match stay in English."""
        return load_tag_translation_map(constants.MODEL_PATH.parent,
                                        constants.MODELS_DIR / constants.MODEL_DIR_NAME,
                                        constants.MODELS_RESOURCE_DIR / constants.MODEL_DIR_NAME)

    def _shared_tag_csv_stamp(self) -> tuple[bool, float]:
        """正本（最優先ディレクトリ）の selected_tags.csv の有無と mtime。"""
        path = constants.MODEL_PATH.parent / "selected_tags.csv"
        try:
            return (True, path.stat().st_mtime) if path.is_file() else (False, 0.0)
        except OSError:
            return (False, 0.0)

    def _refresh_shared_tag_translation_map(self, *, force: bool = False) -> bool:
        """REQ-4.7 / R31-1: 翻訳の正本が現れた／更新されたときだけ読み直す。

        以前は「マップが空のときだけ」再読込していたため、フォールバック先の
        非空マップが入っていると、ワーカーが最優先ディレクトリへ取得した
        selected_tags.csv がそのセッション中ずっと反映されなかった。
        読み直しの判断は正本ファイルのスタンプ（有無＋mtime）で行う（NFR-4）。
        """
        stamp = self._shared_tag_csv_stamp()
        if not force and self._tag_map_stamp == stamp:
            return False
        self._tag_map_stamp = stamp
        new_map = self._load_shared_tag_translation_map()
        if not new_map:
            return False
        self.tag_translation_map = new_map
        self.display_current_tag_page()
        self._display_image_tag_page()
        self.grid_view_widget.set_tag_display_language(
            self._tag_display_language, self.tag_translation_map)
        return True

    def _maybe_prompt_gpu_setup(self):
        return self._startup._maybe_prompt_gpu_setup()

    def _maybe_prompt_update_check(self):
        return self._startup._maybe_prompt_update_check()

    def _on_update_check_finished(self, info):
        return self._startup._on_update_check_finished(info)

    def _show_update_prompt(self, info):
        return self._startup._show_update_prompt(info)

    def _start_gpu_runtime_download(self):
        return self._startup._start_gpu_runtime_download()

    def _cancel_gpu_runtime_download(self):
        return self._startup._cancel_gpu_runtime_download()

    def _on_gpu_dl_progress_destroyed(self):
        return self._startup._on_gpu_dl_progress_destroyed()

    def _on_gpu_runtime_progress(self, percent: int, done_mb: float, total_mb: float):
        return self._startup._on_gpu_runtime_progress(percent, done_mb, total_mb)

    def _on_gpu_runtime_finished(self, ok: bool):
        return self._startup._on_gpu_runtime_finished(ok)

    def _apply_image_list_selection_style(self):
        """ダークモード時にファイルリストの選択色を見やすい色に上書きする。"""
        if self._is_dark_theme:
            self.image_list.setStyleSheet(constants.STYLE_LIST_ITEM_SELECTED_DARK)

    def _install_event_filters(self):
        """Installs event filters on widgets to intercept specific key presses."""
        # Intercept Ctrl+Up/Down on QLineEdit widgets to prevent event propagation
        self.add_single_tag_line.installEventFilter(self)
        self.add_tag_line.installEventFilter(self)
        self.add_tag_line_append.installEventFilter(self)
        self.caption_text_edit.installEventFilter(self)
        self.caption_text_edit.textChanged.connect(self._caption_save_timer.start)

    def _reset_folder_state(self) -> None:
        """前フォルダの編集状態・対象・キャッシュを破棄する（REQ-4.5 / R7-8）。

        早期 return の経路（フォルダ作成失敗・画像 0 件）でも呼ぶ。先に未保存の
        キャプションを確定してから捨てる（REQ-4.3。書く先は「前の」画像なので、
        切替後の対象には影響しない）。
        """
        self._save_current_caption()
        self._caption_save_timer.stop()
        self._current_caption_image_path = None
        self._caption_loaded_text = ""
        self.caption_text_edit.blockSignals(True)
        self.caption_text_edit.clear()
        self.caption_text_edit.blockSignals(False)
        self._all_tags = []
        self._current_image_tags = []
        self._current_image_tag_page = 0
        self._tag_cache = {}
        self.grid_view_widget.update_tag_cache(self._tag_cache)
        self._display_image_tag_page()
        self.display_current_tag_page()

    def reload_image_list(self, auto_select_path: str | None = None):
        """Reloads the list of images from the input directory."""
        self.image_label.setPixmap(QPixmap()) # Explicitly clear pixmap
        input_dir_path = Path(self.settings.paths.input_dir)
        self.image_list.clear()
        self.image_label.setText(self.locale_manager.get_string("MainWindow", "Loading_Image_List"))

        if not self._ensure_input_dir_exists(input_dir_path):
            # REQ-4.5 / R7-8: 切替に失敗しても前フォルダの状態を残さない。
            self._reset_folder_state()
            return

        write_debug_log(self.locale_manager.get_string("MainWindow", "Reloading_Image_List", input_dir_path=input_dir_path))
        image_paths = self._get_image_paths(input_dir_path)

        if not image_paths:
            self.image_label.setText(self.locale_manager.get_string("MainWindow", "Image_Files_Not_Found"))
            self.update_log(self.locale_manager.get_string("MainWindow", "Warning_No_Image_Files_Found", input_dir_path_name=input_dir_path.name), "orange")
            # REQ-4.5 / R7-8:「画像なし」と編集可能状態を共存させない。
            self._reset_folder_state()
            return
            
        selected_item = self._populate_image_list(image_paths, auto_select_path)
            
        if not selected_item and self.image_list.count() > 0:
            selected_item = self.image_list.item(0)
        
        if selected_item:
            self.image_list.setCurrentItem(selected_item)
            # Schedule the image loading to ensure the widget is sized correctly.
            # Re-fetch the current item at fire time instead of capturing selected_item
            # in the closure: if reload_image_list() runs again before this timer fires
            # (e.g. a fast-failing tagging run completes within 100ms), image_list.clear()
            # deletes the underlying C++ object and the captured reference goes dangling.
            QTimer.singleShot(100, lambda: self._load_and_fit_image(self.image_list.currentItem()))
        else:
            self._clear_image_display()

        self._build_tag_cache()
        self.update_log(self.locale_manager.get_string("MainWindow", "List_Updated_Total_Images", count=len(image_paths)), "blue")

    def _ensure_input_dir_exists(self, path: Path) -> bool:
        """Checks if the input directory exists, creating it if necessary. Returns True on success."""
        if not path.is_dir():
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.update_log(self.locale_manager.get_string("MainWindow", "Info_Input_Folder_Created", input_dir_path_name=path.name), "blue")
                return True
            except Exception as e:
                self.update_log(self.locale_manager.get_string("MainWindow", "Error_Input_Folder_Creation_Failed", input_dir_path_name=path.name, e=e), "red")
                self.image_label.setText(self.locale_manager.get_string("MainWindow", "Input_Folder_Not_Found"))
                return False
        return True

    def _get_image_paths(self, base_path: Path) -> list[Path]:
        """Recursively finds all image files in the given directory."""
        paths: list[Path] = []
        for ext in constants.IMAGE_EXTENSIONS:
             paths.extend(base_path.rglob(f"*{ext}")) 
        return sorted(paths)

    def _populate_image_list(self, paths: list[Path], auto_select: str | None) -> QListWidgetItem | None:
        """Adds image paths to the list widget and determines which item to select."""
        selected_item = None
        input_dir = Path(self.settings.paths.input_dir)
        for path in paths:
            relative_path = str(path.relative_to(input_dir))
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole + 1, relative_path)
            self.image_list.addItem(item)
            
            if auto_select and (path.name == auto_select or relative_path == auto_select):
                 selected_item = item
        
        if not selected_item and self.image_list.count() > 0:
            selected_item = self.image_list.item(0)
        
        return selected_item

    def reload_tags_only(self, preserve_page: bool = False):
        """Reloads the aggregated tag list for bulk editing asynchronously."""
        if self._is_text_output_mode():
            # Bulk tag aggregation comma-splits every .txt file; running it on free-text
            # captions would shred sentences into meaningless fragments (design.md 1.4節).
            # The bulk add/delete UI is hidden in caption mode anyway (spec.md 8.3節).
            return

        if preserve_page:
            self._saved_bulk_page = self._current_page
        else:
            self._saved_bulk_page = None

        if self.tag_thread and self.tag_thread.isRunning():
            old_thread, old_worker = self.tag_thread, self.tag_worker
            # Stop the existing thread gracefully
            if old_worker:
                old_worker.stop()
                # 古いスレッドのシグナルを切断して二重発火・混線を防ぐ
                # （REQ-6.6(b) / R8-1: tags_loaded だけでなく finished も切る）。
                for signal, slot in (
                        (old_worker.tags_loaded, self._update_bulk_tag_buttons),
                        (old_worker.finished, self._on_tag_loader_finished)):
                    try:
                        signal.disconnect(slot)
                    except (RuntimeError, TypeError):
                        pass
            old_thread.quit()
            if not self._wait_for_tagger_thread_stop(old_thread, timeout=1.0):
                # U-6: terminate() は使わない（読み取り専用のローダなので自然終了を待てる）。
                # 参照はモジュール変数が持ち、終了後に破棄する。
                write_debug_log("tag loader did not stop in time; detaching without terminate()")
                self._detach_running_thread(old_thread, old_worker)
            else:
                old_thread.deleteLater()
                if old_worker:
                    old_worker.deleteLater()
            self.tag_thread = None
            self.tag_worker = None
        
        input_dir_path = Path(self.settings.paths.input_dir)
        if not input_dir_path.is_dir():
            return

        self.loading_label.setText(self.locale_manager.get_string("Constants", "Loading_Tag_List"))
        self._set_bulk_controls_enabled(False)
        
        self.tag_thread = QThread()
        self.tag_worker = TagLoader(input_dir_path, self.locale_manager.get_string)
        self.tag_worker.moveToThread(self.tag_thread)
        
        self.tag_thread.started.connect(self.tag_worker.run)
        self.tag_worker.tags_loaded.connect(self._update_bulk_tag_buttons)
        self.tag_worker.finished.connect(self._on_tag_loader_finished)
        
        if self.loading_timer is None:
            self.loading_timer = QTimer(self)
            self.loading_timer.timeout.connect(self._animate_loading_label)

        self.tag_thread.start()
        self.loading_timer.start(100)
    
    def _load_and_fit_image(self, item: QListWidgetItem | None):
        """Loads the image and its tags for the selected list item."""
        if item is None:
            self._clear_image_display()
            return

        image_path = Path(self.settings.paths.input_dir) / item.data(Qt.ItemDataRole.UserRole + 1)
        
        if not image_path.is_file():
            self._clear_image_display()
            self.image_label.setText(self.locale_manager.get_string("MainWindow", "File_Not_Found", image_relative_path=image_path.name))
            return
        
        try:
            image = QImage(str(image_path))
            if image.isNull():
                raise ValueError("Failed to load QImage")
            
            pixmap = QPixmap.fromImage(image)
            self._original_image_pixmap = pixmap
            
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
            self._load_image_tags(image_path)
            
            if self._image_viewer_dialog and self._image_viewer_dialog.isVisible():
                self.show_enlarged_image()
        except Exception as e:
            write_debug_log(self.locale_manager.get_string("MainWindow", "Image_Load_Error_Debug", image_path=image_path, e=e))
            self._clear_image_display()
            self.image_label.setText(self.locale_manager.get_string("MainWindow", "Image_Display_Error", image_relative_path=image_path.name, e=e))

    def _load_image_tags(self, image_path: Path, preserve_page: bool = False):
        """Loads tags (tag mode) or the raw .txt into the text editor (captioner / VLM mode)."""
        if self._is_text_output_mode():
            self._load_caption_for_image(image_path)
            return

        txt_path = image_path.with_suffix('.txt')
        tag_content = ""
        if txt_path.is_file():
            try:
                tag_content = txt_path.read_text(encoding='utf-8').strip()
            except Exception as e:
                self.update_log(self.locale_manager.get_string("MainWindow", "Error_Tag_File_Load_Failed", e=e, txt_path_name=txt_path.name), "red")
        
        self._current_image_tags = [tag.strip() for tag in tag_content.split(',') if tag.strip()]
        
        if preserve_page:
            # タグ数変化に応じてページ番号をクランプする
            total_pages = self._get_total_image_tag_pages()
            self._current_image_tag_page = min(self._current_image_tag_page, total_pages - 1)
        else:
            self._current_image_tag_page = 0
        
        self._display_image_tag_page()

    def _load_caption_for_image(self, image_path: Path) -> None:
        """Captioner mode: load the .txt file as free text, no comma-splitting (design.md 8.5節)."""
        same_path = self._current_caption_image_path == image_path
        # Commit any edit still pending in the debounce window for the PREVIOUS image
        # before its path/content is replaced, otherwise the edit is silently lost.
        if self._current_caption_image_path is not None and not same_path:
            self._save_current_caption()
        txt_path = image_path.with_suffix('.txt')
        content = ""
        if txt_path.is_file():
            try:
                content = txt_path.read_text(encoding='utf-8')
            except Exception as e:
                self.update_log(self.locale_manager.get_string("MainWindow", "Error_Tag_File_Load_Failed", e=e, txt_path_name=txt_path.name), "red")
        # REQ-4.3 / R12-5: 同じパスの再ロード（リサイズ等）で未保存の編集を
        # ファイル内容で上書きしない。dirty でなければ外部変更を拾う。
        if same_path and self.caption_text_edit.toPlainText() != self._caption_loaded_text:
            return
        self._current_caption_image_path = image_path
        self._caption_save_timer.stop()
        self.caption_text_edit.blockSignals(True)
        self.caption_text_edit.setPlainText(content)
        self.caption_text_edit.blockSignals(False)
        self._caption_loaded_text = content

    def _save_current_caption(self) -> None:
        """Saves caption_text_edit's content back to the currently loaded image's .txt file.
        Called on focus-out (eventFilter) rather than on every keystroke. Records an
        EditCaptionAction so the change is undoable, and only when something changed."""
        self._caption_save_timer.stop()
        if not self._is_text_output_mode():
            return
        image_path = self._current_caption_image_path
        if image_path is None:
            return
        txt_path = image_path.with_suffix('.txt')
        new_text = self.caption_text_edit.toPlainText()
        existed_before = txt_path.is_file()
        old_text = ""
        if existed_before:
            try:
                old_text = txt_path.read_text(encoding='utf-8')
            except Exception as e:
                # REQ-4.1 / NFR-11: 読めない既存内容を「空だった」として上書きしない。
                # Undo 記録も作らない（old_text="" で積むと Undo が元を破壊する）。
                write_debug_log(
                    f"caption save skipped: existing .txt is unreadable "
                    f"({txt_path.name}: {type(e).__name__}: {e})")
                self.update_log(self.locale_manager.get_string(
                    "MainWindow", "Error_Tag_File_Load_Failed", e=e,
                    txt_path_name=txt_path.name), "red")
                return
        if new_text == old_text:
            self._caption_loaded_text = new_text
            return
        try:
            # REQ-2.3: 一時ファイル + os.replace で原子的に（REQ-2.7 の (iii) 段）。
            atomic_write_text(txt_path, new_text, fsync=False, verify=False)
            write_debug_log(self.locale_manager.get_string("MainWindow", "Caption_Saved", txt_path_name=txt_path.name))
        except Exception as e:
            self.update_log(self.locale_manager.get_string("MainWindow", "Error_Caption_Save_Failed", txt_path_name=txt_path.name, e=e), "red")
            return
        self._caption_loaded_text = new_text
        self.undo_manager.push(EditCaptionAction(
            file_path=txt_path, old_text=old_text, new_text=new_text, file_existed_before=existed_before))
        self._update_undo_redo_buttons()

    @Slot(int)
    def _on_existing_mode_changed(self, index: int):
        """既存ファイルの扱いを設定へ即時反映し、既存の保存フローで永続化する
        （spec.md 6.1節）。"""
        mode = self.existing_mode_combo.itemData(index)
        if not mode or mode == self.settings.behavior.existing_file_mode:
            return
        self.settings.behavior.existing_file_mode = mode
        self.save_current_config()
        write_debug_log(f"existing_file_mode -> {mode}")

    @Slot(int)
    def _on_target_mode_changed(self, index: int):
        """一括処理の対象選択（すべて / 未生成のみ / 失敗のみ / 選択画像のみ）を
        設定へ即時反映し、既存の保存フローで永続化する（260903_vlm-gap-fix.md todo 5）。
        実行中の判定は開始時スナップショットなので、走行中は combo をロックする。"""
        mode = self.target_mode_combo.itemData(index)
        if not mode or mode == self.settings.behavior.target_mode:
            return
        self.settings.behavior.target_mode = mode
        self.save_current_config()
        write_debug_log(f"target_mode -> {mode}")

    @Slot()
    def _on_caption_placement_changed(self, button=None):
        """生成キャプションの挿入位置（前に追加 / 後に追加 / 上書き）を保存する。
        QButtonGroup が排他なので、押されたボタンだけが checked のまま残る。"""
        if button is None:
            button = self.caption_placement_group.checkedButton()
        if button is None:
            return
        placement = button.property("placement")
        if not placement or placement == self.settings.caption.placement:
            return
        self.settings.caption.placement = placement
        self.save_current_config()
        write_debug_log(f"caption placement -> {placement}")

    def _is_text_output_mode(self) -> bool:
        """出力が自由テキストになるモード = ローカル captioner か「VLM接続を使う」。
        タグボタン一覧やタグ集計をこのモードで動かすと自然文をカンマで刻んでしまう。"""
        try:
            if self._current_model_entry().model_type == "captioner":
                return True
        except Exception:
            pass
        return bool(self.settings.vlm.enabled)

    @Slot(bool)
    def _on_use_vlm_toggled(self, checked: bool):
        """「VLM接続を使う」トグル。選択中モデルの種別（tagger / captioner）とは独立で、
        ON のときローカルモデルの代わりにネットワーク VLM で生成する。出力形式は VLM
        プロンプト次第（既定は詳細キャプション。タグ列も可）。"""
        checked = bool(checked)
        if checked == self.settings.vlm.enabled:
            return
        # Save while the OLD output mode is still active.  Disabling VLM first can make
        # _save_current_caption() decide that this is a tagger view and silently discard
        # an edit still waiting in the debounce timer.
        self._save_current_caption()
        self.settings.vlm.enabled = checked
        self.save_current_config()
        write_debug_log(f"use_vlm -> {checked}")
        self.vlm_settings_button.setVisible(checked)
        self.vlm_single_test_button.setVisible(checked)
        # タグ欄 <-> テキスト欄など UI 表示を更新。モデル自体は変わっていないので、
        # カテゴリ既定値や captioner の既定タスクを再シード＆永続化しない（R33-1）。
        self._model_mode.on_model_changed(self._current_model_entry(), seed_defaults=False)

    def _on_use_gpu_toggled(self, checked: bool):
        """タガーの実行プロバイダーをGPU(CUDA)/CPUで切り替える。「VLM接続を使う」行の
        右端にあるチェックボックスで、GPUが実際に使える（ビルド対応＋コンポーネント
        導入済み）場合のみ表示される（ui_main_window._create_input_group）。

        ONで`cuda`、OFFで`cpu`を`config.ini [Behavior] onnx_device`に書く。GPU DLLは
        起動時に既にプロセスへpreload済みなので、次回のタグ付け実行から即座に反映
        される（アプリ再起動が要るのはGPUコンポーネントの初回導入直後だけ）。
        `auto`のまま一度もこのチェックボックスに触れていないユーザーの設定は、
        toggled シグナルが発火しない限り上書きしない（ui_main_window 側で
        setChecked() を toggled.connect() より先に呼んでいるのはこのため）。"""
        checked = bool(checked)
        new_device = "cuda" if checked else "cpu"
        if new_device == self.settings.behavior.onnx_device:
            return
        self.settings.behavior.onnx_device = new_device
        self.save_current_config()
        write_debug_log(f"use_gpu -> {checked} (onnx_device={new_device})")

    @Slot()
    def _open_vlm_settings(self):
        """VLM 設定ダイアログを開く（260901_VLM_design.md 6.3節）。"""
        from vlm_settings_dialog import VlmSettingsDialog
        if self._vlm_settings_dialog is not None and self._vlm_settings_dialog.isVisible():
            self._vlm_settings_dialog.raise_()
            return
        self._vlm_settings_dialog = VlmSettingsDialog(self.settings, self.locale_manager.get_string, self)
        self._vlm_settings_dialog.finished.connect(lambda _=0: setattr(self, "_vlm_settings_dialog", None))
        self._vlm_settings_dialog.show()

    @Slot()
    def _run_vlm_single_test(self):
        """選択画像1枚で実際に VLM 生成し、結果を編集欄へ出しつつ .txt へも保存する
        （既存ファイルの扱い・挿入位置は一括処理と同じ設定に従う。2026-09 ユーザー決定）。"""
        if self._is_processing():
            self.update_log(self.locale_manager.get_string("MainWindow", "Busy_Processing_Active"), "orange")
            return
        current_item = self.image_list.currentItem()
        if not current_item:
            self.update_log(self.locale_manager.get_string("Vlm", "Error_No_Selected_Image"), "red")
            return
        rel = current_item.data(Qt.ItemDataRole.UserRole + 1)
        selected_path = Path(self.settings.paths.input_dir) / rel

        # ASK/overwrite decisions belong to a batch session and must not leak into
        # the independent single-image test.
        self._session_mode_override = None
        self._cleanup_tagger_thread()
        self._update_ui_for_processing(True, 'tagging')
        self._tagger_thread = QThread()
        self._tagger_worker = VlmCaptionWorker(
            self._settings_snapshot(), self._make_decision_requester(), self.locale_manager.get_string,
            selected_file_path=selected_path, single_test=True)
        self._tagger_worker.moveToThread(self._tagger_thread)
        self._tagger_worker.log_message.connect(self.update_log)
        self._tagger_worker.single_test_result.connect(self._on_vlm_single_test_result)
        self._tagger_worker.binding_verified.connect(self._on_vlm_binding_verified)
        self._tagger_worker.batch_completed.connect(self._on_batch_completed)
        self._tagger_worker.finished.connect(self._on_tagger_finished)
        # worker.finished はメインスレッドの後処理だけでなく、実行スレッド自身も
        # 終了させる。後処理側の cleanup だけに任せると、正常終了でも「残存スレッド
        # の終了を待っています...」が通常ログへ残る。
        self._tagger_worker.finished.connect(self._tagger_thread.quit)
        self._tagger_thread.started.connect(self._tagger_worker.run_captioning)
        self._tagger_thread.start()

    @Slot(str, str, str)
    def _on_vlm_single_test_result(self, caption: str, connection_display: str, model_id: str):
        """1枚テストの結果をキャプション編集欄へ表示する（結果は一括処理と同じ経路で.txtへ保存される）。"""
        self.update_log(self.locale_manager.get_string(
            "Vlm", "Test_Result_Header", conn=connection_display, model=model_id), "green")
        if hasattr(self, "caption_text_edit"):
            # REQ-4.4 / R7-9: プログラム的な表示で自動保存タイマーを武装させない。
            # 保存はワーカーが同じ内容で既に行っている。
            self.caption_text_edit.blockSignals(True)
            self.caption_text_edit.setPlainText(caption)
            self.caption_text_edit.blockSignals(False)
            self._caption_loaded_text = caption

    @Slot(str, str)
    def _on_vlm_binding_verified(self, provider_id: str, profile_id: str):
        """実出力を確認できた内蔵 binding を `[Vlm] verified_bindings` に永続化する
        （次回以降 VERIFIED 扱い。UI スレッドで config を書く）。"""
        if vlm_config.mark_binding_verified(self.settings.vlm, provider_id, profile_id=profile_id):
            self.save_current_config()
            self.update_log(self.locale_manager.get_string(
                "Vlm", "Binding_Verified", provider=provider_id, profile=profile_id), "green")

    @Slot(int)
    def _on_task_combo_changed(self, index: int):
        """Persists the selected Florence-2 caption detail level (spec.md 8.2節)."""
        task_key = self.task_combo.itemData(index)
        if not task_key or task_key == self.settings.caption.task:
            return
        self.settings.caption.task = task_key
        self.save_current_config()

    def _clear_image_display(self):
        """Clears the image viewer and tag display."""
        self.image_label.clear()
        self.image_label.setText(self.locale_manager.get_string("MainWindow", "Image_Not_Selected"))
        self._original_image_pixmap = None
        self._current_image_tags = []
        self._current_image_tag_page = 0
        self._display_image_tag_page()
        # Drop the caption editor state too - otherwise a pending edit would later be
        # written back to a file from the old folder / no-longer-selected image.
        self._save_current_caption()
        self._caption_save_timer.stop()
        self._current_caption_image_path = None
        self._caption_loaded_text = ""
        self.caption_text_edit.blockSignals(True)
        self.caption_text_edit.clear()
        self.caption_text_edit.blockSignals(False)
        if self._image_viewer_dialog:
            self._image_viewer_dialog.close()

    def _update_bulk_tag_buttons(self, all_tags: list[tuple[str, int]]):
        return self._tag_panel._update_bulk_tag_buttons(all_tags)

    def _tag_display_grid_dims(self) -> tuple[int, int]:
        return self._tag_panel._tag_display_grid_dims()

    def _get_image_tags_per_page(self) -> int:
        return self._tag_panel._get_image_tags_per_page()

    def _get_total_image_tag_pages(self) -> int:
        return self._tag_panel._get_total_image_tag_pages()

    # ヘルパーメソッドを追加（クラス内に追加してください）
    def _get_translation_index(self, language: str) -> int:
        return self._tag_panel._get_translation_index(language)

    def display_current_tag_page(self):
        return self._tag_panel.display_current_tag_page()
        
    def _display_image_tag_page(self):
        return self._tag_panel._display_image_tag_page()

    @Slot(int)
    def toggle_tag_language(self, index: int):
        return self._tag_panel.toggle_tag_language(index)

    # .txt を書きうる実行が 1 つでも進行中か（REQ-2.2 の正本）。各ハンドラは
    # 自前の部分的なガードではなくこの判定を見る。ダウンロードは .txt を書かないが、
    # ロック中に編集経路だけが通る非対称を作らないため含める。メモリ内の状態だけを
    # 参照し、経路ごとにファイル I/O を足さない（REQ-2.7）。状態の所有者は既存の
    # フラグ群のままで、判定だけをここへ一本化する（NFR-9）。
    def _is_processing(self) -> bool:
        """True while a run that can write .txt files is active."""
        return bool(
            self._is_downloading
            or self._is_bulk_deleting
            or (self._tagger_thread is not None and self._tagger_thread.isRunning())
            or (self._bulk_tag_thread is not None and self._bulk_tag_thread.isRunning())
        )

    def _update_ui_for_processing(self, is_running: bool, process_type: str):
        """Updates UI elements based on whether a process is starting or stopping."""
        if is_running:
            style = constants.STYLE_BTN_RED
            if process_type == 'tagging':
                text = self.locale_manager.get_string("Constants", "Stop_Tagging_Process")
            else: # downloading
                text = f"{self.locale_manager.get_string('Constants', 'Stop_Button_Text')} (0%)"
            
            self._set_main_controls_enabled(False)
        else:
            self._set_main_controls_enabled(True)
            self._check_model_status_and_update_ui()
            return

        self.run_button.setStyleSheet(style)
        self.run_button.setText(text)
        self.run_button.setEnabled(True)

    def _set_main_controls_enabled(self, enabled: bool):
        """Enables or disables main UI controls during processing."""
        if not enabled:
            # Commit a pending caption edit before locking the editor, so it isn't lost
            # (and can't race the worker that is about to rewrite the same .txt).
            self._save_current_caption()
        self._controls_enabled = enabled
        self.input_line.setEnabled(enabled)
        self.grid_view_button.setEnabled(enabled)
        self.image_list.setEnabled(enabled)
        # Locked during a download/tagging run: switching models mid-run would repoint
        # settings/UI away from the model the active worker is using. Also stays disabled
        # while "Use VLM connection" is on - the local model is not in play then.
        self.model_combo.setEnabled(enabled and not self.settings.vlm.enabled)
        self.task_combo.setEnabled(enabled)
        # The active worker already captured EXISTING_FILE_MODE / CAPTION_PLACEMENT at
        # start; changing them mid-run has no effect on this run but is immediately
        # persisted to config.ini, silently changing the default for the next run
        # (PR#16 review). Lock them the same way model_combo/task_combo are locked.
        self.existing_mode_combo.setEnabled(enabled)
        # 対象選択も開始時スナップショットなので走行中はロックする。
        self.target_mode_combo.setEnabled(enabled)
        self.caption_placement_widget.setEnabled(enabled)
        if hasattr(self, "use_vlm_check"):
            self.use_vlm_check.setEnabled(enabled)
            self.vlm_settings_button.setEnabled(enabled)
            # 走行中は1枚テストも押せないようにする（設定ボタンと同じ理由）。
            self.vlm_single_test_button.setEnabled(enabled)
        # The worker rewrites the same .txt files, so every path that can also write them
        # has to be locked: the main caption box, the grid-view cells, Undo/Redo,
        # the single-tag add row, the displayed tag chips and the GPU toggle.
        self.caption_text_edit.setEnabled(enabled)
        self.grid_view_widget.set_editing_enabled(enabled)
        # R25-1: 単体タグ追加の入力欄とボタン
        self.add_single_tag_line.setEnabled(enabled)
        self.add_single_tag_button.setEnabled(enabled)
        # R19-6: 表示中画像のタグチップ（新規生成分は _display_image_tag_page が
        # 現在の錠状態で作る。ここでは既に存在する分を追従させる）。
        for button in self.tag_buttons_for_image:
            button.setEnabled(enabled)
        # R21-1: GPU トグルは実行中のワーカーの provider を変えないが、設定を
        # 書き換えるので同じ錠に入れる。
        if hasattr(self, "use_gpu_check"):
            self.use_gpu_check.setEnabled(enabled)
        self.undo_button.setEnabled(enabled and self.undo_manager.can_undo())
        self.redo_button.setEnabled(enabled and self.undo_manager.can_redo())
        if enabled:
            # Rebuild the grid's Undo/Redo state from the history, not from the buttons.
            self.grid_view_widget.update_undo_redo_buttons(
                self.undo_manager.can_undo(), self.undo_manager.can_redo(),
                self.undo_manager.get_undo_description(), self.undo_manager.get_redo_description())
        self._set_bulk_controls_enabled(enabled)

    def _set_bulk_controls_enabled(self, enabled: bool):
        """Enables or disables bulk action controls."""
        for button in self.tag_buttons:
            button.setEnabled(enabled)
        self.add_tag_button.setEnabled(enabled)
        self.add_tag_button_append.setEnabled(enabled)
        self.add_tag_line.setEnabled(enabled)
        self.add_tag_line_append.setEnabled(enabled)

    def _rebalance_right_splitter(self, text_ui: bool) -> None:
        """モデル種別／VLM トグルで text_ui（captioner / VLM）が変わったときだけ、右パネルの
        縦スプリッターの割当を組み替える。毎回呼んでも実際に切り替わったときしか動かないので、
        利用者が手でドラッグしたサイズを踏み潰さない。サイズ定数は constants に集約
        （ui_main_window の初期 setSizes と同じ出所）。"""
        if text_ui == self._last_text_ui:
            return
        self._last_text_ui = text_ui
        sp = getattr(self, "right_vertical_splitter", None)
        if sp is None:
            return
        if text_ui:
            # section[1] は shared_run_block の高さに固定し、伸縮は viewer(0) と log(2) が受ける。
            sp.setStretchFactor(1, 0)
            sp.setSizes(constants.RIGHT_SPLIT_TEXT)
        else:
            sp.setStretchFactor(1, 2)
            sp.setSizes(constants.RIGHT_SPLIT_TAGGER)

    def closeEvent(self, event: QCloseEvent):
        """Handles the window closing event to save settings and stop threads gracefully."""
        self._is_shutting_down = True
        write_debug_log("DEBUG: closeEvent triggered. _is_shutting_down = True")
        # REQ-4.3 / R4-4: 終了時も自動保存のデバウンス中身を取りこぼさない。
        self._save_current_caption()
        self.grid_view_widget.flush_pending_edits()
        self.save_current_config()

        threads_to_stop: list[tuple[QThread | None, StoppableWorker | None]] = [ # type: ignore
            (self._download_thread, self._downloader_worker),
            (self._gpu_dl_thread, self._gpu_dl_worker),
            (self._update_check_thread, self._update_check_worker),
            (self._tagger_thread, self._tagger_worker),
            (self._bulk_tag_thread, self._bulk_tag_worker),
            (self.tag_thread, self.tag_worker)
        ]

        # First, request all running threads to stop by calling their thread-safe stop() method
        for thread, worker in threads_to_stop:
            if thread and thread.isRunning():
                write_debug_log(f"DEBUG: closeEvent: Requesting worker {type(worker).__name__} in thread {thread} to stop.")
                if worker and hasattr(worker, 'stop'):
                    # This is a cross-thread method call, but it is safe
                    # because the worker's stop() method is thread-safe.
                    write_debug_log(f"DEBUG: closeEvent: Calling stop() on worker {type(worker).__name__}.")
                    worker.stop()
                    write_debug_log(f"DEBUG: closeEvent: Returned from stop() on worker {type(worker).__name__}.")
                write_debug_log(f"DEBUG: closeEvent: Calling quit() on thread {thread}.")
                thread.quit()

        # Now, wait for them to finish
        for thread, worker in threads_to_stop:
            if thread and thread.isRunning():
                write_debug_log(f"DEBUG: closeEvent: Waiting for thread {thread} to finish...")
                # REQ-6.6(a)（U-6）: terminate() は使わない。processEvents を挟む待機で
                # GUI スレッドを生かしたままにすると、待機中のワーカーが張る
                # BlockingQueuedConnection（上書き確認ブリッジ）も処理できるため、
                # 「確認したのにデッドロックする」窓も塞がる（R8-2）。
                if self._wait_for_tagger_thread_stop(thread, timeout=5.0):
                    write_debug_log(f"DEBUG: closeEvent: Thread {thread} finished gracefully.")
                else:
                    write_debug_log(
                        f"DEBUG: closeEvent: Thread {thread} did not finish in time; "
                        "detaching without terminate()")
                    self._detach_running_thread(thread, worker)

        # 退避済みスレッド（_cleanup_tagger_thread が5秒待っても終わらず手放した分）も
        # ここで待つ。closeEvent が列挙していなかったため、稼働中のまま破棄されて
        # プロセスがクラッシュしうる状態だった（260922 レビュー指摘）。
        # 待ちきれなくても terminate はしない: 参照はモジュール変数が持ち続けるので、
        # 破棄されてクラッシュすることはない。
        if not worker_host.wait_for_detached_threads(5000):
            write_debug_log(
                "DEBUG: closeEvent: detached thread(s) outlived the wait; "
                "references are kept for the remaining process lifetime")

        write_debug_log("DEBUG: closeEvent: Proceeding with application close.")
        super().closeEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filters events from watched objects (e.g., QLineEdit, QPushButton)."""
        if event.type() == QEvent.Type.KeyPress:
            assert isinstance(event, QKeyEvent)
            # Check if the event is from one of our QLineEdit widgets
            if watched in (self.add_single_tag_line, self.add_tag_line, self.add_tag_line_append):
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                        if not event.isAutoRepeat():
                            delta = -1 if event.key() == Qt.Key.Key_Up else 1
                            self.navigate_image_list(delta)
                        return True # Event handled, stop further processing
        
        # Handle right-click on tag buttons
        elif event.type() == QEvent.Type.MouseButtonRelease:
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
                tag_name = watched.property("original_tag")
                self._highlight_files_for_tag(tag_name)
                return False

        elif event.type() == QEvent.Type.Leave:
            if isinstance(watched, QPushButton) and watched.property("original_tag"):
                self._clear_highlight()
                return False

        elif event.type() == QEvent.Type.FocusOut:
            if watched is self.caption_text_edit:
                self._save_current_caption()
                return False

        # Pass the event on to the parent class if not handled
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent):
        """Handles global key presses for the main window (when not on a specific widget)."""
        super().keyPressEvent(event)




    def wheelEvent(self, event: QWheelEvent):
        write_debug_log(f"DEBUG: wheelEvent - angleDelta: {event.angleDelta().y()}")
        """Handles mouse wheel events for image navigation."""
        if self.image_list.geometry().contains(event.position().toPoint()):
             super().wheelEvent(event)
             return
        
        # If cursor is elsewhere, perform global navigation.
        if event.angleDelta().y() > 0:
            self.navigate_image_list(-1)
        elif event.angleDelta().y() < 0:
            self.navigate_image_list(1)
        event.accept()
    
    @Slot()
    def select_image_item(self, item: QListWidgetItem):
        write_debug_log(f"DEBUG: select_image_item - item: {item.text()}")
        """Slot for when an item in the image list is clicked."""
        self._clear_highlight()
        self._load_and_fit_image(item)

    @Slot(str, str)
    def _handle_folder_drop(self, folder_path: str, file_to_select: str | None = None):
        write_debug_log(f"DEBUG: _handle_folder_drop - folder_path: {folder_path}, file_to_select: {file_to_select}")
        """Handles the logic for when a folder is dropped or selected."""
        # U-1（REQ-2.5）: フォルダ切替は書き込み先を差し替える開始経路なので、処理中は
        # 受理しない。3 経路（D&D／Browse／パス入力）はここへ集約されているので、
        # 判定はこの 1 箇所で足りる（保留パスは持たない = 状態の所有者を増やさない）。
        if self._is_processing():
            write_debug_log(f"_handle_folder_drop rejected while processing: {folder_path}")
            self.update_log(self.locale_manager.get_string("MainWindow", "Busy_Processing_Active"), "orange")
            return
        # REQ-3.5 / R20-2: フォルダをまたいだ Undo を防ぐ。履歴は切替時に消す
        # （_batch_failed_paths の破棄と同じ方針。どの切替経路でもここを通る）。
        self.undo_manager.clear()
        self._update_undo_redo_buttons()
        # フォルダが変わったら「失敗のみ」用のセッション失敗集合は無効。破棄する
        # （前フォルダの失敗パスを別フォルダの実行対象にしないため。todo 5）。
        self._batch_failed_paths = []
        self.input_line.setText(folder_path)
        self.reload_image_list(file_to_select)
        self.reload_tags_only()
    
    @Slot()
    def browse_folder(self):
        write_debug_log("DEBUG: browse_folder called.")
        """Opens a dialog to select an input folder."""
        dir_path = QFileDialog.getExistingDirectory(self, self.locale_manager.get_string("MainWindow", "Select_Input_Folder"), self.input_line.text())
        if dir_path:
            self._handle_folder_drop(dir_path)

    @Slot()
    def _on_input_path_changed(self):
        """Handles folder path changes from the input line editingFinished signal."""
        folder_path = self.input_line.text().strip()
        if Path(folder_path).is_dir():
            # We don't have a file to select, so pass None.
            self._handle_folder_drop(folder_path, None)
        else:
            # Using a hardcoded string for now. Should be added to locale.
            self.update_log(self.locale_manager.get_string(
                "MainWindow", "Error_Invalid_Folder_Path", folder_path=folder_path), "red")

    def dragEnterEvent(self, event: QDragEnterEvent):
        write_debug_log(f"DEBUG: dragEnterEvent - hasUrls: {event.mimeData().hasUrls()}")
        """Handles drag enter events to accept valid file types."""
        if event.mimeData().hasUrls():
            path = Path(event.mimeData().urls()[0].toLocalFile())
            if path.is_dir() or path.suffix.lower() in constants.IMAGE_EXTENSIONS + ['.txt']:
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        write_debug_log(f"DEBUG: dropEvent - hasUrls: {event.mimeData().hasUrls()}")
        """Handles drop events to load folders or files."""
        path = Path(event.mimeData().urls()[0].toLocalFile())
        folder_path, file_to_select = None, None

        if path.is_dir():
            folder_path = str(path)
        elif path.suffix.lower() in constants.IMAGE_EXTENSIONS:
            folder_path, file_to_select = str(path.parent), path.name
        elif path.suffix.lower() == '.txt':
            folder_path = str(path.parent)
            for ext in constants.IMAGE_EXTENSIONS:
                img_path = path.with_suffix(ext)
                if img_path.is_file():
                    file_to_select = img_path.name
                    break
        
        if folder_path:
            self._handle_folder_drop(folder_path, file_to_select)
            event.acceptProposedAction()

    def resizeEvent(self, event: QResizeEvent):
        write_debug_log(f"DEBUG: resizeEvent - size: {event.size().width()}x{event.size().height()}")
        """Starts a timer to handle resizing after it has finished."""
        self._resize_timer.start()
        super().resizeEvent(event)

    @Slot()
    def _handle_resize_debounced(self):
        write_debug_log("DEBUG: _handle_resize_debounced called.")
        """Reloads the currently displayed image to fit the new window size."""
        current_item = self.image_list.currentItem()
        if current_item and self.image_label.pixmap():
            self._load_and_fit_image(current_item)
        self.update_all_button_alignments()

    def update_button_text_alignment(self, button: QPushButton):
        return self._tag_panel.update_button_text_alignment(button)

    def update_all_button_alignments(self):
        return self._tag_panel.update_all_button_alignments()


    def navigate_image_list(self, delta: int):
        write_debug_log(f"DEBUG: _navigate_image_list called with delta: {delta}")
        """Navigates the image list up or down by a given delta."""
        current = self.image_list.currentRow()
        new_row = current + delta
        if 0 <= new_row < self.image_list.count():
            item = self.image_list.item(new_row)
            self.image_list.setCurrentItem(item)
            self._clear_highlight()
            self._load_and_fit_image(item)

    def _change_tag_page(self, delta: int):
        """Changes the displayed page for bulk tags."""
        new_page = self._current_page + delta
        total_pages = (len(self._all_tags) + constants.TAGS_PER_PAGE - 1) // constants.TAGS_PER_PAGE
        if 0 <= new_page < total_pages:
            self._current_page = new_page
            self.display_current_tag_page()

    def _change_image_tag_page(self, delta: int):
        """Changes the displayed page for single image tags."""
        new_page = self._current_image_tag_page + delta
        total_pages = self._get_total_image_tag_pages()
        if 0 <= new_page < total_pages:
            self._current_image_tag_page = new_page
            self._display_image_tag_page()

    @Slot()
    def _add_single_tag(self):
        return self._tag_panel._add_single_tag()

    def _delete_image_tag(self, tag_to_delete: str):
        return self._tag_panel._delete_image_tag(tag_to_delete)
    
    def _copy_tag_to_clipboard(self, tag_name: str):
        return self._tag_panel._copy_tag_to_clipboard(tag_name)

    def add_tag_all(self, prepend: bool):
        return self._tag_panel.add_tag_all(prepend)
    
    def delete_tag_all(self, tag_to_delete: str):
        return self._tag_panel.delete_tag_all(tag_to_delete)

    # --- Thread and Process Management ---

    def toggle_download_or_start_tagging(self):
        """Main action button logic: starts or stops download/tagging."""
        self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Process_Generic"), "black")
        QApplication.processEvents() # Ensure the log message is displayed immediately
        if self._is_downloading:
            self._stop_download_thread()
        elif self._tagger_thread and self._tagger_thread.isRunning():
            self._stop_tagging_thread()
        else:
            # 「VLM接続を使う」ならローカルモデル不要 -> モデル未DLでも起動できる。
            if self.settings.vlm.enabled or self._is_model_available():
                self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Tagging_Process"), "black")
                self._start_tagging_thread()
            else:
                # _start_download_thread() logs "Starting_Model_Download" itself.
                self._start_download_thread()

    def _settings_snapshot(self):
        return self._workers._settings_snapshot()

    def _start_tagging_thread(self):
        return self._workers._start_tagging_thread()

    @Slot(int, int)
    def _on_tagging_progress(self, done: int, total: int):
        """ワーカーからの進捗通知。毎画像来るので ~0.5s に1回だけ GUI へ反映する
        （issue #10: 1画像=1ログだと Qt イベントキューが飽和して固まる）。"""
        if self._is_shutting_down:
            return
        now = time.monotonic()
        if done < total and now - self._progress_last_shown_at < 0.5:
            return
        self._progress_last_shown_at = now
        # 停止処理中はボタン文言を「停止中...」に固定したいので触らない。
        if not self.run_button.isEnabled():
            return
        base = self.locale_manager.get_string("Constants", "Stop_Tagging_Process")
        self.run_button.setText(f"{base} ({done} / {total})")

    @Slot(list)
    def _on_batch_failed(self, failed_paths: list):
        """直近の一括バッチで失敗（または停止で未処理）だった入力画像パスを
        セッション記憶する。次回「失敗のみ」モードの対象になる。成功時は空リストで
        置換される。1枚テストは batch_failed を出さないので上書きされない
        （260903_vlm-gap-fix.md todo 5）。"""
        self._batch_failed_paths = [Path(p) for p in failed_paths]
        write_debug_log(f"batch_failed: {len(self._batch_failed_paths)} 件を記憶（次回の失敗のみ実行対象）")

    @Slot(list)
    def _on_batch_completed(self, changed_files: list):
        """タギング/キャプション1回分の変更を、Undo履歴上の1エントリとして積む
        （spec.md 4.2節）。50件の履歴が大量画像処理で即座に消費されるのを防ぐ。"""
        if not changed_files:
            return
        if len(changed_files) > constants.UNDO_BATCH_SNAPSHOT_LIMIT:
            # 全ファイルの旧内容＋新内容を1つの CompositeUndoAction に抱えると、
            # 数万ファイル規模ではメモリを圧迫する（issue #10）。今回のバッチだけ
            # スナップショットを諦める。既存の Undo 履歴（別バッチ・バルク編集等）は
            # 無関係なので消さない（PR#16 レビュー指摘: 以前は clear() で全部消していた）。
            self.update_log(self.locale_manager.get_string(
                "MainWindow", "Undo_Batch_Too_Large", count=len(changed_files)), "orange")
            write_debug_log(
                f"batch_completed: {len(changed_files)} 件 > {constants.UNDO_BATCH_SNAPSHOT_LIMIT} のため今回のバッチは Undo スナップショットを作成しません（既存履歴は保持）")
            return
        actions = []
        for change in changed_files:
            if change.was_append:
                actions.append(AppendTagsActionV2(
                    file_path=change.path, previous_content=change.previous_content,
                    new_content=change.new_content, added_tags=list(change.added_tags)))
            else:
                actions.append(OverwriteFileAction(
                    file_path=change.path, previous_content=change.previous_content,
                    new_content=change.new_content))
        label = self.locale_manager.get_string(
            "MainWindow", "Undo_Batch_Tagging", count=len(actions))
        self.undo_manager.push(CompositeUndoAction(actions=actions, label=label))
        self._update_undo_redo_buttons()
        write_debug_log(f"batch_completed: {len(actions)} 件のファイル変更を1つのUndoエントリにまとめました")

    def _wait_for_tagger_thread_stop(self, thread: QThread, timeout: float = 5.0) -> bool:
        # インスタンス状態を使わない（self=None でも呼べる。既存テスト互換）。
        return worker_host.wait_for_thread_stop(thread, timeout)

    def _cleanup_tagger_thread(self):
        return self._workers._cleanup_tagger_thread()

    def _detach_running_thread(self, thread: QThread, worker) -> None:
        # インスタンス状態を使わない（self=None でも呼べる。既存テスト互換）。
        worker_host.detach_running_thread(thread, worker)

    def _stop_tagging_thread(self):
        return self._workers._stop_tagging_thread()

    def _start_download_thread(self):
        return self._workers._start_download_thread()

    def _stop_download_thread(self):
        return self._workers._stop_download_thread()
    
    def _start_bulk_tag_worker(self, mode: str, **kwargs: Any):
        return self._workers._start_bulk_tag_worker(mode, **kwargs)

    # --- Thread Finished Slots ---

    @Slot()
    def _on_tagger_finished(self):
        """Cleans up after the tagging thread has finished."""
        if self._is_shutting_down:
            return # Do not perform post-processing if the app is closing

        current_item = self.image_list.currentItem()
        path_to_reselect = None
        if current_item:
            # Store the relative path of the current file to re-select it later
            path_to_reselect = current_item.data(Qt.ItemDataRole.UserRole + 1)

        # The worker may have just fetched PixAI's selected_tags.csv on our behalf.
        # REQ-4.7: 既に非空でも、正本が現れた／更新されたなら読み直す。
        self._refresh_shared_tag_translation_map()

        self.reload_image_list(auto_select_path=path_to_reselect)
        self.reload_tags_only()

        self._update_ui_for_processing(False, 'tagging')

        # worker.finished は _on_tagger_finished（ここ）と thread.quit の両方に
        # connect されている（この順）。ここで即座に _cleanup_tagger_thread() を
        # 呼ぶと、まだキューに残っている thread.quit がこのイベントループの
        # ターンでは処理されておらず、cleanup 側の isRunning() チェックが
        # 「まだ動いている」と見えて無駄な wait/強制終了に倒れうる。
        # QTimer.singleShot(0, ...) で次のイベントループターンへ回すことで、
        # 先に thread.quit が処理される順序を保証する。
        QTimer.singleShot(0, self._cleanup_tagger_thread)

    @Slot(bool)
    def _on_download_finished(self, success: bool):
        """Cleans up after the download thread has finished."""
        if self._is_shutting_down:
            return

        write_debug_log(f"Download finished signal received with success: {success}")
        self._is_downloading = False
        self._set_main_controls_enabled(True) # Re-enable main controls

        if success:
            self.update_log(self.locale_manager.get_string("MainWindow", "Model_Download_Complete"), "green")
            # Update only the model verification status without overwriting user-modified settings
            config = load_config()
            self.settings.model.verified_models = load_settings(config).model.verified_models
            self._check_model_status_and_update_ui() # On success, check status to show "TAG" button
            
            # Reload tag translation map as files are now available
            self.tag_translation_map = self._load_shared_tag_translation_map()
            
            # Update UI with new translation map
            self.display_current_tag_page()
            self._display_image_tag_page()
            self.grid_view_widget.set_tag_display_language(self._tag_display_language, self.tag_translation_map)
        else:
            self.update_log(self.locale_manager.get_string("MainWindow", "Model_Download_Failed"), "red")
            self._check_model_status_and_update_ui(force_download=True) # On failure/stop, force "Download" button
            
        if self._download_thread:
            self._download_thread.quit()
            self._download_thread.wait()
            if self._downloader_worker:
                self._downloader_worker.deleteLater()
            self._download_thread.deleteLater()
            self._download_thread = self._downloader_worker = None

    @Slot()
    def _on_tag_loader_finished(self):
        """Cleans up after the tag loader thread has finished."""
        if self._is_shutting_down:
            return

        if self.tag_thread:
            self.tag_thread.quit()
            self.tag_thread.wait()
            if self.tag_worker:
                self.tag_worker.deleteLater()
            self.tag_thread.deleteLater()
            self.tag_thread = self.tag_worker = None

    @Slot()
    def _on_bulk_tag_finished(self):
        """Cleans up after the bulk tag worker thread has finished."""
        if self._is_shutting_down:
            return

        # REQ-2.4: 解除は処理の終了経路だけが行う。先にスレッドを畳んで一括フラグを
        # 落とし、その上で錠を開ける（逆順にすると isRunning() が真のまま再描画が走り、
        # 錠が戻らない）。
        if self._bulk_tag_thread:
            self._bulk_tag_thread.quit()
            self._bulk_tag_thread.wait()
            if self._bulk_tag_worker:
                self._bulk_tag_worker.deleteLater()
            self._bulk_tag_thread.deleteLater()
            self._bulk_tag_thread = self._bulk_tag_worker = None
        self._is_bulk_deleting = False
        self._set_main_controls_enabled(True)
        # 実行ボタンは錠の対象外（タグ付け/DL 中は「停止」として使う）なので、
        # 一括終了時にモデル状態から戻す。
        self._check_model_status_and_update_ui()
        self.reload_tags_only(preserve_page=True)
    
    @Slot(list, list, str)
    def _on_bulk_add_completed(self, file_tags: list[tuple[Path, list[str]]], added_tags: list[str], position: str):
        """Records bulk add operation for undo.

        REQ-3.3（R22-4）: 渡されるのはファイルごとの「実際に追加されたタグ」。
        全体の added_tags はログ用で、undo は file_tags を使う。
        """
        if file_tags:
            if len(file_tags) > constants.UNDO_BATCH_SNAPSHOT_LIMIT:
                # REQ-3.8（R20-4）: バッチ側と同じ上限。超えたら今回の一括だけ
                # Undo を作らない（既存履歴は保持する）。
                self.update_log(self.locale_manager.get_string(
                    "MainWindow", "Undo_Batch_Too_Large", count=len(file_tags)), "orange")
                write_debug_log(
                    f"bulk_add: {len(file_tags)} 件 > {constants.UNDO_BATCH_SNAPSHOT_LIMIT} のため Undo を作成しません")
                return
            action = BulkAddTagsAction(file_tags=file_tags, position=position)
            self.undo_manager.push(action)
            self._update_undo_redo_buttons()
            input_dir = Path(self.settings.paths.input_dir)
            for fp, _tags in file_tags:
                rel_path = str(fp.relative_to(input_dir))
                self._update_tag_cache_entry(rel_path)
            write_debug_log(f"Bulk add action recorded: {len(file_tags)} files, {len(added_tags)} tags")
    
    @Slot(str, list)
    def _on_bulk_delete_completed(self, removed_tag: str, file_tag_positions: list[tuple[Path, list[int]]]):
        """Records bulk delete operation for undo.

        REQ-3.3（R5-5）: 各ファイルの全出現位置を受け取り、undo で全部戻す。
        """
        if file_tag_positions:
            if len(file_tag_positions) > constants.UNDO_BATCH_SNAPSHOT_LIMIT:
                self.update_log(self.locale_manager.get_string(
                    "MainWindow", "Undo_Batch_Too_Large", count=len(file_tag_positions)), "orange")
                write_debug_log(
                    f"bulk_delete: {len(file_tag_positions)} 件 > {constants.UNDO_BATCH_SNAPSHOT_LIMIT} のため Undo を作成しません")
                return
            action = BulkRemoveTagsAction(removed_tag=removed_tag, file_positions=file_tag_positions)
            self.undo_manager.push(action)
            self._update_undo_redo_buttons()
            input_dir = Path(self.settings.paths.input_dir)
            for fp, _positions in file_tag_positions:
                rel_path = str(fp.relative_to(input_dir))
                self._update_tag_cache_entry(rel_path)
            write_debug_log(f"Bulk delete action recorded: {len(file_tag_positions)} files")
    
    @Slot(Path, list)
    def _on_gridview_tags_added(self, file_path: Path, added_tags: list[str]):
        """Records tag addition from GridView for undo."""
        action = AddTagsAction(file_path=file_path, added_tags=added_tags)
        self.undo_manager.push(action)
        self._update_undo_redo_buttons()
        write_debug_log(f"GridView add action recorded: {len(added_tags)} tags to {file_path.name}")
        self._build_tag_cache()
    
    @Slot(Path, str, int)
    def _on_gridview_tag_removed(self, file_path: Path, removed_tag: str, original_index: int):
        """Records tag removal from GridView for undo."""
        action = RemoveTagAction(file_path=file_path, removed_tag=removed_tag, original_index=original_index)
        self.undo_manager.push(action)
        self._update_undo_redo_buttons()
        write_debug_log(f"GridView remove action recorded: '{removed_tag}' from {file_path.name}")
        self._build_tag_cache()

    @Slot(Path, str, str, bool)
    def _on_gridview_caption_edited(self, file_path: Path, old_text: str, new_text: str, file_existed_before: bool):
        """Records a caption edit made in GridView (captioner mode) for undo. The cell has
        already written the file; this only pushes the action."""
        self.undo_manager.push(EditCaptionAction(
            file_path=file_path, old_text=old_text, new_text=new_text, file_existed_before=file_existed_before))
        self._update_undo_redo_buttons()
        write_debug_log(f"GridView caption edit recorded: {file_path.name}")

    @Slot(str)
    def _on_gridview_caption_save_failed(self, detail: str):
        self.update_log(self.locale_manager.get_string("MainWindow", "Error_Caption_Save_Failed", txt_path_name=detail, e=""), "red")

   

    def save_current_config(self):
        return self._settings_binding.save_current_config()
        # The log message can be distracting for this frequent operation, so it's commented out.
        # self.update_log(self.locale_manager.get_string("MainWindow", "Settings_Saved"), "green")

    def _current_model_entry(self) -> ModelEntry:
        """Returns the ModelEntry for the currently selected model_id (spec.md 8.4節).
        Falls back to the PixAI pseudo-entry (always first) for an unknown/stale model_id."""
        entry = get_model_entry(self.settings.model.model_id)
        if entry is not None:
            return entry
        # Stale / hidden-manual-download model_id: normalize the setting to the fallback
        # so the downloader and every other consumer agree on which model is active.
        fallback = discover_models()[0]
        if self.settings.model.model_id != fallback.model_id:
            write_debug_log(f"Unknown model_id '{self.settings.model.model_id}'; falling back to '{fallback.model_id}'.")
            self.settings.model.model_id = fallback.model_id
        return fallback

    def _current_model_dir(self) -> Path:
        return self._current_model_entry().model_dir

    def _is_model_available(self) -> bool:
        """Checks if the currently selected model is verified and its files exist (spec.md 8.4節)."""
        entry = self._current_model_entry()
        if entry.model_id == "pixai-tagger-v0.9":
            # Unchanged from before multi-model support (NFR-3): trust file presence
            # directly rather than gating on verified_models, since installs that
            # downloaded PixAI before multi-model support (or whose config.ini was
            # ever regenerated) never get a chance to set that flag again - the
            # download flow only runs when the user clicks the button, which they
            # won't do for a model that already works.
            return constants.MODEL_PATH.is_file() and constants.TAGS_CSV_PATH.is_file()
        required_files = config_mapping(entry.config, "network", "files").keys()
        if not required_files:
            return False
        if entry.config.get("manual_download", False):
            # No in-app download to hash-verify against (gated repo): the model_registry
            # scan already hid this entry until every file was placed by hand, so file
            # presence alone is the availability signal - same trust-the-files policy as
            # the PixAI special-case above.
            return all((entry.model_dir / f).is_file() for f in required_files)
        if not self.settings.model.verified_models.get(entry.model_id, False):
            return False
        return all((entry.model_dir / f).is_file() for f in required_files)

    @Slot(int)
    def _on_model_combo_changed(self, index: int):
        """Handles the user picking a different model from model_combo (spec.md 8.2節)."""
        model_id = self.model_combo.itemData(index)
        if not model_id or model_id == self.settings.model.model_id:
            return
        self._save_current_caption()
        self.settings.model.model_id = model_id
        self.save_current_config()
        self.model_combo.setToolTip(self.locale_manager.get_string("ModelDescriptions", model_id))
        # The per-category dialog's rows are bound to the previous model's category list.
        if self._category_settings_dialog is not None:
            self._category_settings_dialog.close()
        self._model_mode.on_model_changed(self._current_model_entry())

    def _get_log_color_map(self) -> dict[str, str]:
        """Returns the appropriate color map based on the detected theme."""
        theme_colors = {
            True: {"red": constants.COLOR_LOG_ERROR_DARK, "green": constants.COLOR_LOG_SUCCESS_DARK, "blue": constants.COLOR_LOG_INFO_DARK, "orange": constants.COLOR_LOG_WARN_DARK, "black": constants.COLOR_LOG_DEFAULT_DARK},
            False: {"red": constants.COLOR_LOG_ERROR_LIGHT, "green": constants.COLOR_LOG_SUCCESS_LIGHT, "blue": constants.COLOR_LOG_INFO_LIGHT, "orange": constants.COLOR_LOG_WARN_LIGHT, "black": constants.COLOR_LOG_DEFAULT_LIGHT}
        }
        return theme_colors[self._is_dark_theme]

    def create_slider_group(self, layout: QGridLayout, section: str, min_val: float, max_val: float, step: float, keys: Mapping[str, int]):
        return self._settings_binding.create_slider_group(layout, section, min_val, max_val, step, keys)

    def _current_model_categories(self) -> list[str]:
        return self._settings_binding._current_model_categories()

    def _save_config_soon(self) -> None:
        return self._settings_binding._save_config_soon()

    def _sync_settings_sliders(self):
        return self._settings_binding._sync_settings_sliders()

    @Slot()
    def _open_category_settings_dialog(self):
        return self._settings_binding._open_category_settings_dialog()

    def update_log(self, message: str, color: str = "black"):
        """Appends a colored, timestamped message to the log output."""
        html_color = self._log_color_map.get(color, self._log_color_map["black"])
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        html_message = f'<span style="color:{html_color};">{timestamp}{message}</span>'
        
        self.log_output.append(html_message)

        if self.log_output.document().blockCount() > constants.MAX_LOG_LINES:
            cursor = self.log_output.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(cursor.MoveOperation.NextBlock, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _build_tag_cache(self) -> None:
        """入力ディレクトリ内の全画像ファイルのタグキャッシュを構築する。"""
        self._tag_cache = {}
        input_dir = Path(self.settings.paths.input_dir)
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item is None:
                continue
            rel_path = item.data(Qt.ItemDataRole.UserRole + 1)
            txt_path = (input_dir / rel_path).with_suffix('.txt')
            tags = tag_utils.read_tags(txt_path)
            self._tag_cache[rel_path] = set(tags)
        # REQ-3.7 / R2-10: ここは新しい dict を作るので、グリッドへも渡し直さないと
        # 検索フィルタが古いキャッシュを参照し続ける（Undo/Redo・ローダ完了の共通経路）。
        self.grid_view_widget.update_tag_cache(self._tag_cache)

    def _update_tag_cache_entry(self, rel_path: str) -> None:
        """指定ファイルのキャッシュエントリを更新する。"""
        input_dir = Path(self.settings.paths.input_dir)
        txt_path = (input_dir / rel_path).with_suffix('.txt')
        tags = tag_utils.read_tags(txt_path)
        self._tag_cache[rel_path] = set(tags)

    def _highlight_files_for_tag(self, tag_name: str) -> None:
        return self._tag_panel._highlight_files_for_tag(tag_name)

    def _clear_highlight(self) -> None:
        return self._tag_panel._clear_highlight()

    def _update_undo_redo_buttons(self):
        """Updates the enabled state and tooltips of undo/redo buttons."""
        can_undo = self.undo_manager.can_undo()
        can_redo = self.undo_manager.can_redo()
        
        self.undo_button.setEnabled(can_undo and self._controls_enabled)
        self.redo_button.setEnabled(can_redo and self._controls_enabled)
        
        if can_undo:
            desc = self.undo_manager.get_undo_description()
            self.undo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Undo_Action", desc=desc))
        else:
            self.undo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Undo_No_Actions"))
        
        if can_redo:
            desc = self.undo_manager.get_redo_description()
            self.redo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Redo_Action", desc=desc))
        else:
            self.redo_button.setToolTip(self.locale_manager.get_string("MainWindow", "Redo_No_Actions"))
        
        # Update grid view buttons as well
        self.grid_view_widget.update_undo_redo_buttons(can_undo, can_redo, 
                                                        self.undo_manager.get_undo_description(),
                                                        self.undo_manager.get_redo_description())
    
    @Slot()
    def _perform_undo(self):
        """Performs an undo operation."""
        if self._is_processing():
            write_debug_log("_perform_undo ignored while processing")
            return
        if self.undo_manager.undo():
            self.update_log(self.locale_manager.get_string("MainWindow", "Undo_Success"), "green")
            self._report_partial_undo()
            self._refresh_ui_after_undo_redo()
        else:
            self.update_log(self.locale_manager.get_string("MainWindow", "Undo_Failed"), "red")
        self._update_undo_redo_buttons()

    def _report_partial_undo(self) -> None:
        """部分成功を全成功として見せない（REQ-3.1 / NFR-10）。"""
        if self.undo_manager.last_result is not UndoResult.PARTIAL:
            return
        failed = self.undo_manager.last_failed_paths
        self.update_log(self.locale_manager.get_string(
            "MainWindow", "Undo_Partial_Failure", count=len(failed)), "orange")
        for path in failed[:10]:
            write_debug_log(f"undo partial failure: {path}")
    
    @Slot()
    def _perform_redo(self):
        """Performs a redo operation."""
        if self._is_processing():
            write_debug_log("_perform_redo ignored while processing")
            return
        if self.undo_manager.redo():
            self.update_log(self.locale_manager.get_string("MainWindow", "Redo_Success"), "green")
            self._report_partial_undo()
            self._refresh_ui_after_undo_redo()
        else:
            self.update_log(self.locale_manager.get_string("MainWindow", "Redo_Failed"), "red")
        self._update_undo_redo_buttons()
    
    def _refresh_ui_after_undo_redo(self):
        """Refreshes UI elements after undo/redo operations."""
        # Refresh current image tags (preserve page to avoid jumping back to page 0)
        current_item = self.image_list.currentItem()
        if current_item:
            image_path = Path(self.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
            self._load_image_tags(image_path, preserve_page=True)
        
        # Rebuild tag cache to reflect undo/redo changes
        # （グリッドへの反映は _build_tag_cache が行う。R2-10）
        self._build_tag_cache()
        
        # Refresh bulk tag list
        self.reload_tags_only(preserve_page=True)
        
        # Refresh grid view
        self.grid_view_widget.refresh_current_page()

    @Slot()
    def _on_model_status_changed(self):
        """REQ-4.6（U-3）: ワーカーがディスク側で変えた verified をメモリへ反映する。

        再読込はこのシグナル経由に限定する（通常の UI 更新では disk を読まない。
        NFR-4）。読み直さないと、次の無関係な設定保存が古いメモリ値でディスクを
        巻き戻す（R30-3）。
        """
        if self._is_shutting_down:
            return
        try:
            self.settings.model.verified_models = load_settings(load_config()).model.verified_models
        except Exception as e:
            write_debug_log(f"model_status_changed: could not reload verified_models: {e}")
        self._check_model_status_and_update_ui()

    def _check_model_status_and_update_ui(self, auto_start_download: bool = False, force_download: bool = False):
        """Checks for model files and updates the run button's state and appearance."""
        if not force_download and (self.settings.vlm.enabled or self._is_model_available()):
            self.run_button.setText(self.locale_manager.get_string("Constants", "Tag_Button_Text"))
            self.run_button.setStyleSheet(constants.STYLE_BTN_GREEN)
            self.run_button.setEnabled(True)
        else:
            self.run_button.setText(self.locale_manager.get_string("Constants", "Download_Start_No_Model"))
            self.run_button.setStyleSheet(constants.STYLE_BTN_ORANGE)
            self.run_button.setEnabled(True)
            if auto_start_download:
                self.update_log(self.locale_manager.get_string("MainWindow", "Info_Model_NotFound_Start_Download"), "orange")

    @Slot(int, float, float)
    def _update_download_progress(self, percentage: int, downloaded_mb: float, total_mb: float):
        """Updates the run button text with the download percentage."""
        if self._is_downloading:
            stop_text = self.locale_manager.get_string("Constants", "Stop_Button_Text")
            button_text = f"{stop_text} ({percentage}%) {downloaded_mb:.1f}/{total_mb:.1f} MB"
            self.run_button.setText(button_text)
    
    def _animate_loading_label(self):
        """Animates the loading label with dots."""
        self.loading_state = (self.loading_state + 1) % 4
        dots = '.' * self.loading_state
        self.loading_label.setText(f"{self.locale_manager.get_string('Constants', 'Loading_Tag_List')}{dots}")


    @Slot()
    def show_enlarged_image(self):
        """Shows the currently selected image in the area to the left of the tag panel,
        maximised to fit, so the tags stay visible and clickable while viewing it.
        The display is chosen from the main window's own screen (multi-monitor safe)."""
        if not self._original_image_pixmap or self._original_image_pixmap.isNull():
            return

        # マルチディスプレイ対策: 本体ウィンドウの左上が乗っているスクリーンを基準にする。
        # 左上が画面外（前回終了時のオフスクリーン位置など）なら順にフォールバック。
        anchor = self.frameGeometry().topLeft()
        target_screen = (QApplication.screenAt(anchor)
                         or self.screen()
                         or QApplication.primaryScreen())
        screen_geom = target_screen.availableGeometry()

        tag_panel_widget = self.tag_display_grid.parentWidget().parentWidget() # Get the splitter child widget
        if not tag_panel_widget:
            return

        # 画像を出す領域＝「タグパネルの左端」から「基準スクリーンの左端」までの帯。
        # ここに収まる最大サイズにフィットさせ、タグ欄には決してかぶせない（編集用途）。
        tag_panel_left_global = tag_panel_widget.mapToGlobal(QPoint(0, 0)).x()
        available_width = tag_panel_left_global - screen_geom.x()
        available_width = max(200, min(available_width, screen_geom.width()))
        available_height = screen_geom.height()

        if self._original_image_pixmap.height() == 0:
            return
        img_ratio = self._original_image_pixmap.width() / self._original_image_pixmap.height()

        dialog_width = available_width
        dialog_height = int(dialog_width / img_ratio) if img_ratio > 0 else available_height

        if dialog_height > available_height:
            dialog_height = available_height
            dialog_width = int(dialog_height * img_ratio)

        dialog_width = max(200, min(dialog_width, available_width))
        dialog_height = max(200, min(dialog_height, available_height))

        # 右辺をタグパネルの左端に合わせて配置（タグ欄に非かぶり）。基準スクリーンの
        # 左端より外へは出さない。縦は基準スクリーン内で中央寄せ。
        dialog_x = max(screen_geom.x(), tag_panel_left_global - dialog_width)
        dialog_y = screen_geom.y() + (screen_geom.height() - dialog_height) // 2

        if self._image_viewer_dialog is None:
            # Launch in navigation mode (default)
            self._image_viewer_dialog = ImageViewerDialog(self) # Removed tag_panel_global_rect
            self._image_viewer_dialog.finished.connect(self._image_viewer_dialog_closed)
            self._image_viewer_dialog.nextImageRequested.connect(lambda: self.navigate_image_list(1))
            self._image_viewer_dialog.prevImageRequested.connect(lambda: self.navigate_image_list(-1))

        self._image_viewer_dialog.show_image(self._original_image_pixmap, dialog_width, dialog_height)
        self._image_viewer_dialog.setGeometry(dialog_x, dialog_y, dialog_width, dialog_height)
        self._image_viewer_dialog.show()
        self._image_viewer_dialog.activateWindow()

    @Slot()
    def _image_viewer_dialog_closed(self):
        """Slot to clean up the reference to the image viewer dialog when it closes.

        REQ-5.6 / R6-1: 参照を落とすだけでなく破棄する（閉じるたびに親の子として
        残り、フル解像度 pixmap を抱えたまま蓄積していた）。順序を制御するため
        WA_DeleteOnClose ではなく finished 起点で deleteLater する。
        """
        dialog = self._image_viewer_dialog
        self._image_viewer_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    @Slot()
    def _show_grid_view(self):
        """Switches the central widget to the GridViewWidget."""
        image_paths = self._get_image_paths(Path(self.settings.paths.input_dir))
        if not image_paths:
            QMessageBox.information(self, self.locale_manager.get_string("MainWindow", "No_Images_Found_Title"), self.locale_manager.get_string("MainWindow", "No_Images_Found_Message"))
            return

        # REQ-4.3 / R18-2: グリッドへ切り替える前に未保存のキャプションを確定する。
        self._save_current_caption()
        self.central_widget.setCurrentWidget(self.grid_view_widget)
        self.setWindowTitle(f"{constants.MSG_WINDOW_TITLE} - "
                            + self.locale_manager.get_string("GridView", "Window_Title"))
        self.grid_view_widget.set_caption_mode(self._is_text_output_mode())
        self.grid_view_widget.load_images(image_paths, self._tag_cache, Path(self.settings.paths.input_dir))
        self.showMaximized()
        self.update_log(self.locale_manager.get_string("MainWindow", "Switched_To_Grid_View"), "blue")

    @Slot()
    def _show_main_view(self):
        """Switches the central widget back to the main view."""
        # REQ-4.3: グリッド側の未保存編集も確定してから戻る。
        self.grid_view_widget.flush_pending_edits()
        self.central_widget.setCurrentWidget(self.main_view_widget)
        self.setWindowTitle(constants.MSG_WINDOW_TITLE)
        self.showNormal() # Or restore previous geometry
        self.update_log(self.locale_manager.get_string("MainWindow", "Switched_Back_To_Main_View"), "blue")
        self.reload_tags_only()
        self._load_and_fit_image(self.image_list.currentItem())

    # --- 既存ファイル処理: 追加同期機構（design.md 4.1節） ---------------------
    #
    # 旧実装（QEventLoop + 共有変数 _overwrite_response）は、ワーカースレッドと GUI
    # スレッドの間で手作りの同期を行っていたためレースが起き、大量画像の「常に上書き」で
    # 処理が凍結することがあった（Issue #12）。ここでは Qt の BlockingQueuedConnection に
    # 委ね、共有可変状態を持たない形に置き換えている。

    _SESSION_OVERRIDE_DECISION = {
        ExistingFileMode.OVERWRITE: OverwriteDecision.OVERWRITE,
        ExistingFileMode.SKIP: OverwriteDecision.SKIP,
        ExistingFileMode.APPEND: OverwriteDecision.APPEND,
    }

    @Slot(str, result=str)
    def ask_existing_file_decision(self, file_path_str: str) -> str:
        """GUI スレッドで実行され、ワーカーへ決定を文字列で返す。

        Qt のメタオブジェクト経由で Enum をそのまま渡すのは環境差があるため、
        OverwriteDecision の名前（"OVERWRITE" / "SKIP" / "APPEND"）で受け渡しする。
        """
        if self._is_shutting_down:
            return OverwriteDecision.SKIP.name
        if self._session_mode_override is not None:
            return self._SESSION_OVERRIDE_DECISION[self._session_mode_override].name
        return self._ask_overwrite_confirmation(Path(file_path_str)).name

    def _make_decision_requester(self) -> Callable[[Path], OverwriteDecision]:
        """ワーカースレッドから GUI スレッドの ask_existing_file_decision を
        同期呼び出しするブリッジを作る（design.md 4.1節）。

        ダイアログが本当に必要なのは「ASK モードで、まだ『常に〜』が押されていない」
        ときだけ。それ以外は GUI スレッドへの BlockingQueuedConnection を張らずに
        ワーカー側で即決する:
        - 終了処理中は SKIP（closeEvent が thread.wait 中だと invokeMethod がデッドロック
          しうるため、往復自体を回避する）
        - セッション内オーバーライド（「常に上書き/スキップ/追記」）が決まっていれば
          その決定を返す（10K 枚で 10K 回の同期クロススレッド呼び出しを省く）
        """
        def resolve(file_path: Path) -> OverwriteDecision:
            if self._is_shutting_down:
                return OverwriteDecision.SKIP
            override = self._session_mode_override  # GIL-atomic な単一参照の読み取り
            if override is not None:
                return self._SESSION_OVERRIDE_DECISION[override]

            result = QMetaObject.invokeMethod(
                self, "ask_existing_file_decision",
                Qt.ConnectionType.BlockingQueuedConnection,
                Q_RETURN_ARG(str), Q_ARG(str, str(file_path)),
            )
            try:
                return OverwriteDecision[str(result)]
            except KeyError:
                # 戻り値取得に失敗した環境では、既存ファイルに触れない側へ倒す
                write_debug_log(f"ask_existing_file_decision: 想定外の戻り値 {result!r} -> SKIP として扱います")
                return OverwriteDecision.SKIP
        return resolve

    def _ask_overwrite_confirmation(self, file_path: Path) -> OverwriteDecision:
        """実際にダイアログを表示する。GUI スレッドで実行されること（spec.md 6.2節）。

        上段 [上書き] [スキップ] [追記] / 下段 [常に上書き] [常にスキップ] [常に追記]。
        「常に〜」を選ぶとセッション内オーバーライドを設定し、以降は確認を省略する。
        追記系のボタンはテキスト出力モード（captioner / VLM）では表示しない
        （追記の可否は「キャプション挿入位置」トグルが担うため、ここでは出さない）。
        """
        is_captioner = self._is_text_output_mode()

        msg = QMessageBox(self)
        msg.setWindowTitle(self.locale_manager.get_string("MainWindow", "Overwrite_Confirmation_Title"))
        msg.setText(self.locale_manager.get_string("MainWindow", "Overwrite_Confirmation_Message", path_name=file_path.name))
        msg.setIcon(QMessageBox.Icon.Question)

        btn_yes = msg.addButton(self.locale_manager.get_string("MainWindow", "Overwrite"), QMessageBox.ButtonRole.YesRole)
        btn_skip = msg.addButton(self.locale_manager.get_string("MainWindow", "Skip"), QMessageBox.ButtonRole.NoRole)
        btn_append = None if is_captioner else msg.addButton(
            self.locale_manager.get_string("MainWindow", "Append_Button"), QMessageBox.ButtonRole.AcceptRole)
        btn_yes_all = msg.addButton(self.locale_manager.get_string("MainWindow", "Always_Overwrite"), QMessageBox.ButtonRole.YesRole)
        btn_no_all = msg.addButton(self.locale_manager.get_string("MainWindow", "Always_Skip"), QMessageBox.ButtonRole.NoRole)
        btn_append_all = None if is_captioner else msg.addButton(
            self.locale_manager.get_string("MainWindow", "Always_Append_Button"), QMessageBox.ButtonRole.AcceptRole)

        msg.exec()
        clicked = msg.clickedButton()
        # REQ-5.6 / R20-3: ファイルごとに作るので、使い終わったら破棄する
        # （親の子として残り続け、大量ファイルで蓄積していた）。
        msg.deleteLater()

        if clicked == btn_yes_all:
            self._session_mode_override = ExistingFileMode.OVERWRITE
            return OverwriteDecision.OVERWRITE
        if clicked == btn_no_all:
            self._session_mode_override = ExistingFileMode.SKIP
            return OverwriteDecision.SKIP
        if btn_append_all is not None and clicked == btn_append_all:
            self._session_mode_override = ExistingFileMode.APPEND
            return OverwriteDecision.APPEND
        if clicked == btn_yes:
            return OverwriteDecision.OVERWRITE
        if btn_append is not None and clicked == btn_append:
            return OverwriteDecision.APPEND
        # btn_skip、ダイアログを閉じた場合（clicked is None）はいずれもスキップ扱い
        return OverwriteDecision.SKIP

    @Slot(str)
    def _update_input_dir(self, text: str):
        """Updates the input directory path in the settings."""
        self.settings.paths.input_dir = text
