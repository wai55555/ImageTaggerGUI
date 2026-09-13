from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol
import functools
import sys
import time

from PySide6.QtCore import (
    Qt, QThread, QObject, Signal, Slot, QTimer, QPoint, QEvent,
    QMetaObject, Q_ARG, Q_RETURN_ARG
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget,
    QGridLayout, QLabel, QLineEdit, QPushButton,
    QSlider, QTextEdit, QFileDialog, QMessageBox, QDialog,
    QStackedWidget, QApplication, QSplitter, QListWidgetItem, QComboBox, QButtonGroup,
    QProgressDialog
)
from PySide6.QtGui import (
    QPixmap, QImage, QKeyEvent, QResizeEvent, QDragEnterEvent,
    QDropEvent, QCloseEvent, QWheelEvent,
    QPalette
)

from utils import write_debug_log
import constants
import app_settings # Added import
from app_settings import load_config, load_settings, save_config # Updated import
from custom_widgets import PathLineEdit, TagListWidget
import tag_utils
from tag_utils import load_tag_translation_map
from tagging_core import ExistingFileMode, OverwriteDecision
from custom_dialogs import ClickableLabel, ImageViewerDialog, CategoryTagSettingsDialog
from grid_view_widget import GridViewWidget
from workers import DownloaderWorker, TaggerThreadWorker, CaptionerThreadWorker, TagLoader, BulkTagWorker, GpuRuntimeDownloadWorker
from vlm_worker import VlmCaptionWorker
from locale_manager import LocaleManager
from ui_main_window import Ui_MainWindow
from undo_manager import (
    UndoManager, AddTagsAction, RemoveTagAction, BulkAddTagsAction, BulkRemoveTagsAction,
    EditCaptionAction, OverwriteFileAction, AppendTagsActionV2, CompositeUndoAction,
)
from model_registry import ModelEntry, config_mapping, discover_models, get_model_entry
from model_mode_controller import ModelModeController

def get_os_language() -> str:
    """
    Gets the OS's UI language in a robust, cross-platform way.
    Uses ctypes for Windows, QLocale for macOS/Linux, and falls back to locale.
    """
    try:
        if sys.platform == "win32":
            # Windows: Use ctypes to call Windows API for the most reliable result.
            import ctypes
            windll = ctypes.windll.kernel32
            # GetUserDefaultUILanguage returns a LANGID, e.g., 0x0411 for ja-JP
            lang_id = windll.GetUserDefaultUILanguage()
            # Primary language ID is in the lower 10 bits
            primary_lang_id = lang_id & 0x3FF
            # A map for common primary language IDs to ISO 639-1 codes
            lang_map = {0x09: "en", 0x11: "ja", 0x07: "de", 0x0c: "fr", 0x12: "ko", 0x04: "zh"}
            return lang_map.get(primary_lang_id, "en")
    except Exception as e:
        write_debug_log(f"Failed to get OS language via ctypes: {e}")

    # Fallback for non-Windows (macOS, Linux) or if ctypes fails.
    # locale.getdefaultlocale() is more reliable in bundled apps than QLocale
    # as it doesn't depend on the QApplication instance's state.
    try:
        import locale
        lang_code, _ = locale.getdefaultlocale()
        return lang_code.split('_')[0] if lang_code else "en"
    except (ImportError, ValueError, IndexError) as e:
        write_debug_log(f"Failed to get OS language via locale: {e}")
        return "en"

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
        self._caption_save_timer.setInterval(1200)
        self._caption_save_timer.timeout.connect(self._save_current_caption)
        self.loading_timer: QTimer | None = None
        self.loading_state = 0
        self._sliders: dict[str, tuple[QSlider, QLabel]] = {}
        # Set True while pushing model-default / dialog values into the main sliders so
        # their valueChanged handler doesn't mis-flag those as user edits ("touched").
        self._suppress_slider_touch: bool = False
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
        self._last_navigation_event_time: datetime | None = None # 追加
        
        self.tag_translation_map: dict[str, list[str]] = {}
        self._tag_display_language: str = "English"
        self._current_caption_image_path: Path | None = None

        # Tag cache for hover highlight feature
        self._tag_cache: dict[str, set[str]] = {}

        # Constants
        self._tag_button_min_width = 100
        self._tag_button_min_height = 25

    def initial_load(self):
        """Performs the initial loading of images and tags after the main window is shown."""
        self.tag_translation_map = load_tag_translation_map(constants.MODEL_PATH.parent,
                                                       constants.MODELS_DIR / constants.MODEL_DIR_NAME,
                                                       constants.MODELS_RESOURCE_DIR / constants.MODEL_DIR_NAME)
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

    def _maybe_prompt_gpu_setup(self):
        """NVIDIA GPU が使えるビルドで GPU コンポーネント未整備なら、ダウンロードを尋ねる。

        3 択: ダウンロードする / 今はしない（次回また尋ねる）/ 使わない（次回から聞かない）。
        「使わない」を選んだ後に気が変わった人は config.ini [Behavior] gpu_setup_prompt = ask
        で復活できる。ただし gpu_runtime/ が中途半端に壊れている場合は、dismissed でも
        修復のために 1 度だけ尋ねる。

        `CUDAExecutionProvider` は onnxruntime-gpu にコンパイルされているという静的な
        事実でしかなく、実機に NVIDIA GPU があるかは見ていない。なので
        `onnx_providers.has_nvidia_gpu()`（nvidia-smi の有無）で実機確認を挟み、
        AMD/Intel/GPU 無し環境に無意味な約1.8GBの提案をしないようにする
        （cubic review, PR #21）。
        """
        beh = self.settings.behavior
        if beh.onnx_device == "cpu":
            return
        if self._gpu_dl_thread and self._gpu_dl_thread.isRunning():
            return
        try:
            import onnxruntime
            import onnx_providers
            import gpu_runtime
            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                return  # this build has no CUDA support -> nothing to offer
            if onnx_providers.gpu_runtime_ready():
                return  # components already installed
            if not onnx_providers.has_nvidia_gpu():
                return  # no NVIDIA GPU detected on this machine -> nothing to offer
            # dismissed suppresses the prompt, except when a partly-installed gpu_runtime/
            # is sitting there broken (crashed download etc.) - offer to repair it once.
            partial = onnx_providers.gpu_runtime_dir().exists()
            if beh.gpu_setup_prompt != "ask" and not partial:
                return
            spec = gpu_runtime.load_component_spec()
        except Exception:
            return
        if spec is None:
            return  # no (valid) gpu_components.json bundled -> dev build or unpinned

        gb = gpu_runtime.spec_total_bytes(spec) / (1024 ** 3)
        size_txt = (f"{gb:.1f} GB" if gb >= 0.05
                    else self.locale_manager.get_string("Gpu", "Prompt_Size_Unknown"))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(self.locale_manager.get_string("Gpu", "Prompt_Title"))
        box.setText(self.locale_manager.get_string("Gpu", "Prompt_Body", size=size_txt))
        dl_btn = box.addButton(self.locale_manager.get_string("Gpu", "Prompt_Download"),
                               QMessageBox.ButtonRole.AcceptRole)
        box.addButton(self.locale_manager.get_string("Gpu", "Prompt_Later"),
                      QMessageBox.ButtonRole.RejectRole)
        never_btn = box.addButton(self.locale_manager.get_string("Gpu", "Prompt_Never"),
                                  QMessageBox.ButtonRole.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is dl_btn:
            self._start_gpu_runtime_download()
        elif clicked is never_btn:
            beh.gpu_setup_prompt = "dismissed"
            self.save_current_config()
            if partial:
                # This was the repair prompt (a broken gpu_runtime/ made `partial`
                # true and bypassed the dismissed check above) - without clearing
                # it, `partial` stays true forever and this same prompt would keep
                # reappearing on every launch even though the user just said Never
                # (CodeRabbit review, PR #21). `gpu_runtime` was already imported
                # above (same function scope).
                try:
                    gpu_runtime.GpuRuntimeInstaller().uninstall()
                except Exception as exc:  # noqa: BLE001
                    write_debug_log(f"_maybe_prompt_gpu_setup: could not clear broken gpu_runtime/ ({exc!r})")

    def _start_gpu_runtime_download(self):
        if self._gpu_dl_thread and self._gpu_dl_thread.isRunning():
            return
        self.update_log(self.locale_manager.get_string("Gpu", "Download_Start"), "black")

        progress = QProgressDialog(
            self.locale_manager.get_string("Gpu", "Download_Progress_Label"),
            self.locale_manager.get_string("Gpu", "Download_Cancel"), 0, 100, self)
        progress.setWindowTitle(self.locale_manager.get_string("Gpu", "Prompt_Title"))
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        progress.canceled.connect(self._cancel_gpu_runtime_download)
        # WA_DeleteOnClose means the underlying C++ object can be destroyed by the
        # user clicking the dialog's own close button (not just our Cancel/close()
        # calls) at any time. `destroyed` fires synchronously right when that
        # happens, so this is the one place that reliably keeps our reference in
        # sync - without it, a later access (from the async finished handler) can
        # hit an already-deleted PySide6 wrapper and raise RuntimeError
        # (cubic review, PR #21).
        progress.destroyed.connect(self._on_gpu_dl_progress_destroyed)
        self._gpu_dl_progress = progress

        self._gpu_dl_thread = QThread()
        self._gpu_dl_worker = GpuRuntimeDownloadWorker(self.locale_manager.get_string)
        self._gpu_dl_worker.moveToThread(self._gpu_dl_thread)
        self._gpu_dl_worker.log_message.connect(self.update_log)
        self._gpu_dl_worker.progress_update.connect(self._on_gpu_runtime_progress)
        self._gpu_dl_worker.download_finished.connect(self._on_gpu_runtime_finished)
        self._gpu_dl_thread.started.connect(self._gpu_dl_worker.run_download)
        self._gpu_dl_thread.start()
        progress.show()

    def _cancel_gpu_runtime_download(self):
        if self._gpu_dl_worker:
            self.update_log(self.locale_manager.get_string("Gpu", "Download_Cancelling"), "orange")
            self._gpu_dl_worker.stop()

    def _on_gpu_dl_progress_destroyed(self):
        self._gpu_dl_progress = None

    def _on_gpu_runtime_progress(self, percent: int, done_mb: float, total_mb: float):
        if not self._gpu_dl_progress:
            return
        if total_mb > 0:
            self._gpu_dl_progress.setRange(0, 100)
            self._gpu_dl_progress.setValue(percent)
            self._gpu_dl_progress.setLabelText(
                self.locale_manager.get_string("Gpu", "Download_Progress_Detail",
                                               done=f"{done_mb:.0f}", total=f"{total_mb:.0f}"))
        else:
            self._gpu_dl_progress.setRange(0, 0)  # indeterminate

    def _on_gpu_runtime_finished(self, ok: bool):
        if self._is_shutting_down:
            return  # closeEvent stops/joins the thread itself
        # Capture before the worker is torn down below: install() returns False for
        # both cancellation and genuine failure, collapsing them into one bool.
        # Without this, cancelling a download pops the scary "Download failed"
        # dialog even though the user asked for exactly this (CodeRabbit, PR #21).
        cancelled = bool(self._gpu_dl_worker and self._gpu_dl_worker.is_stopped())
        if self._gpu_dl_progress:
            # close() counts as a cancel for QProgressDialog and would re-fire
            # canceled -> _cancel_gpu_runtime_download; drop the connection first.
            # Both calls are guarded: _on_gpu_dl_progress_destroyed (connected to
            # `destroyed`) is the normal way this reference gets cleared, but a
            # user-initiated close (WA_DeleteOnClose) can race ahead of us.
            try:
                self._gpu_dl_progress.canceled.disconnect(self._cancel_gpu_runtime_download)
                self._gpu_dl_progress.close()  # WA_DeleteOnClose frees it
            except (RuntimeError, TypeError):
                pass  # already destroyed (e.g. user closed the dialog directly)
            self._gpu_dl_progress = None
        if self._gpu_dl_thread:
            self._gpu_dl_thread.quit()
            self._gpu_dl_thread.wait()
            if self._gpu_dl_worker:
                self._gpu_dl_worker.deleteLater()
            self._gpu_dl_thread.deleteLater()
            self._gpu_dl_thread = self._gpu_dl_worker = None
        if cancelled:
            return  # already logged via Download_Cancelling / Worker_Stopped
        title = self.locale_manager.get_string("Gpu", "Prompt_Title")
        if ok:
            QMessageBox.information(self, title,
                                   self.locale_manager.get_string("Gpu", "Download_Done_Restart"))
        else:
            QMessageBox.warning(self, title,
                                self.locale_manager.get_string("Gpu", "Download_Failed"))

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

    def reload_image_list(self, auto_select_path: str | None = None):
        """Reloads the list of images from the input directory."""
        self.image_label.setPixmap(QPixmap()) # Explicitly clear pixmap
        input_dir_path = Path(self.settings.paths.input_dir)
        self.image_list.clear()
        self.image_label.setText(self.locale_manager.get_string("MainWindow", "Loading_Image_List"))

        if not self._ensure_input_dir_exists(input_dir_path):
            return

        write_debug_log(self.locale_manager.get_string("MainWindow", "Reloading_Image_List", input_dir_path=input_dir_path))
        image_paths = self._get_image_paths(input_dir_path)

        if not image_paths:
            self.image_label.setText(self.locale_manager.get_string("MainWindow", "Image_Files_Not_Found"))
            self.update_log(self.locale_manager.get_string("MainWindow", "Warning_No_Image_Files_Found", input_dir_path_name=input_dir_path.name), "orange")
            return
            
        selected_item = self._populate_image_list(image_paths, auto_select_path)
            
        if not selected_item and self.image_list.count() > 0:
            selected_item = self.image_list.item(0)
        
        if selected_item:
            self.image_list.setCurrentItem(selected_item)
            # Schedule the image loading to ensure the widget is sized correctly
            QTimer.singleShot(100, lambda: self._load_and_fit_image(selected_item))
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
            # Stop the existing thread gracefully
            if self.tag_worker:
                self.tag_worker.stop()
                # 古いスレッドのシグナルを切断して二重発火を防ぐ
                try:
                    self.tag_worker.tags_loaded.disconnect(self._update_bulk_tag_buttons)
                except RuntimeError:
                    pass
            self.tag_thread.quit()
            self.tag_thread.wait(1000) # Wait up to 1 second for the thread to finish
            if self.tag_thread.isRunning():
                self.tag_thread.terminate() # Force terminate if it doesn't stop
            
            # Clean up old thread and worker
            if self.tag_worker:
                self.tag_worker.deleteLater()
            self.tag_thread.deleteLater()
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
        # Commit any edit still pending in the debounce window for the PREVIOUS image
        # before its path/content is replaced, otherwise the edit is silently lost.
        if self._current_caption_image_path is not None and self._current_caption_image_path != image_path:
            self._save_current_caption()
        self._current_caption_image_path = image_path
        txt_path = image_path.with_suffix('.txt')
        content = ""
        if txt_path.is_file():
            try:
                content = txt_path.read_text(encoding='utf-8')
            except Exception as e:
                self.update_log(self.locale_manager.get_string("MainWindow", "Error_Tag_File_Load_Failed", e=e, txt_path_name=txt_path.name), "red")
        self._caption_save_timer.stop()
        self.caption_text_edit.blockSignals(True)
        self.caption_text_edit.setPlainText(content)
        self.caption_text_edit.blockSignals(False)

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
            except Exception:
                old_text = ""
        if new_text == old_text:
            return
        try:
            txt_path.write_text(new_text, encoding='utf-8')
            write_debug_log(self.locale_manager.get_string("MainWindow", "Caption_Saved", txt_path_name=txt_path.name))
        except Exception as e:
            self.update_log(self.locale_manager.get_string("MainWindow", "Error_Caption_Save_Failed", txt_path_name=txt_path.name, e=e), "red")
            return
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
        except Exception:  # noqa: BLE001
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
        # タグ欄 <-> テキスト欄など UI 表示を更新。
        self._model_mode.on_model_changed(self._current_model_entry())

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
        if self._tagger_thread and self._tagger_thread.isRunning():
            self.update_log(self.locale_manager.get_string("MainWindow", "Warning_Tagging_Already_Running"), "orange")
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
            self.settings, self._make_decision_requester(), self.locale_manager.get_string,
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
            self.caption_text_edit.setPlainText(caption)

    @Slot(str, str)
    def _on_vlm_binding_verified(self, provider_id: str, profile_id: str):
        """実出力を確認できた内蔵 binding を `[Vlm] verified_bindings` に永続化する
        （次回以降 VERIFIED 扱い。UI スレッドで config を書く）。"""
        import vlm_config
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
        self.caption_text_edit.blockSignals(True)
        self.caption_text_edit.clear()
        self.caption_text_edit.blockSignals(False)
        if self._image_viewer_dialog:
            self._image_viewer_dialog.close()

    def _update_bulk_tag_buttons(self, all_tags: list[tuple[str, int]]):
        """Updates the bulk tag deletion buttons with new tag data."""
        if self.loading_timer:
            self.loading_timer.stop()
        self._is_bulk_deleting = False
        self._all_tags = all_tags

        saved_page = getattr(self, '_saved_bulk_page', None)
        if saved_page is not None:
            total_pages = max(1, (len(all_tags) + constants.TAGS_PER_PAGE - 1) // constants.TAGS_PER_PAGE)
            self._current_page = min(saved_page, total_pages - 1)
        else:
            self._current_page = 0
        self._saved_bulk_page = None

        self.display_current_tag_page()
        self._build_tag_cache()

    def _get_image_tags_per_page(self) -> int:
        """Returns the number of tags to display per page for the current image."""
        return max(1, self.settings.window.tag_display_cols * self.settings.window.tag_display_rows)

    def _get_total_image_tag_pages(self) -> int:
        """Returns the total number of tag pages for the current image."""
        tags_per_page = self._get_image_tags_per_page()
        return max(1, (len(self._current_image_tags) + tags_per_page - 1) // tags_per_page)

    # ヘルパーメソッドを追加（クラス内に追加してください）
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

    def display_current_tag_page(self):
        """Displays the current page of bulk tags."""
        for button in self.tag_buttons:
            button.deleteLater()
        self.tag_buttons.clear()

        total_tags = len(self._all_tags)
        start_index = self._current_page * constants.TAGS_PER_PAGE
        end_index = min(start_index + constants.TAGS_PER_PAGE, total_tags)
        
        if total_tags == 0:
            self.loading_label.setText(self.locale_manager.get_string("MainWindow", "Tag_File_Txt_Not_Found"))
        else:
            self.loading_label.setText(self.locale_manager.get_string("MainWindow", "Displaying_Tags_Count_And_Click_Delete", total_tags=total_tags, start_index=start_index + 1, end_index=end_index))
            
            current_page_tags = self._all_tags[start_index:end_index]
            
            # 翻訳リストのインデックスを取得
            lang_index = self._get_translation_index(self._tag_display_language)

            for i, (tag_name, count) in enumerate(current_page_tags):
                display_text = tag_name
                
                # 英語以外かつ、辞書にタグが存在する場合
                if lang_index != -1 and tag_name in self.tag_translation_map:
                    try:
                        # 指定した言語の翻訳を取得
                        translations = self.tag_translation_map[tag_name]
                        if len(translations) > lang_index:
                            display_text = translations[lang_index]
                    except IndexError:
                        pass # エラー時は英語のまま

                button = QPushButton(f"{display_text} ({count})")
                # ... (以下のコードはそのまま) ...
                button.setMinimumWidth(self._tag_button_min_width)
                button.setFixedHeight(self._tag_button_min_height)
                button.setToolTip(tag_name) # Tooltip always shows English tag
                button.setProperty("original_tag", tag_name)
                button.clicked.connect(functools.partial(self.delete_tag_all, tag_name))
                button.installEventFilter(self)
                self.tag_button_grid.addWidget(button, i // 4, i % 4)
                self.tag_buttons.append(button)

        self.prev_page_btn.setEnabled(self._current_page > 0)
        self.next_page_btn.setEnabled(end_index < total_tags)
        self._set_bulk_controls_enabled(True)
        QTimer.singleShot(0, self.update_all_button_alignments)
        
    def _display_image_tag_page(self):
        """Displays the current page of tags for the selected image."""
        for button in self.tag_buttons_for_image:
            button.deleteLater()
        self.tag_buttons_for_image.clear()

        cols = self.settings.window.tag_display_cols
        tags_per_page = self._get_image_tags_per_page()

        total_tags = len(self._current_image_tags)
        start = self._current_image_tag_page * tags_per_page
        end = min(start + tags_per_page, total_tags)
        
        lang_index = self._get_translation_index(self._tag_display_language)

        for i, tag_name in enumerate(self._current_image_tags[start:end]):
            display_text = tag_name
            
            # 翻訳処理
            if lang_index != -1 and tag_name in self.tag_translation_map:
                try:
                    translations = self.tag_translation_map[tag_name]
                    if len(translations) > lang_index:
                        display_text = translations[lang_index]
                except IndexError:
                    pass

            button = QPushButton(display_text)
            button.setMinimumSize(self._tag_button_min_width, self._tag_button_min_height)
            button.setToolTip(tag_name) # Tooltip always shows English tag
            button.setProperty("original_tag", tag_name)
            button.clicked.connect(functools.partial(self._delete_image_tag, tag_name))
            button.installEventFilter(self)
            self.tag_display_grid.addWidget(button, i // cols, i % cols)
            self.tag_buttons_for_image.append(button)

        self.image_tag_prev_page_btn.setEnabled(self._current_image_tag_page > 0)
        self.image_tag_next_page_btn.setEnabled(end < total_tags)
        QTimer.singleShot(0, self.update_all_button_alignments)

    @Slot(int)
    def toggle_tag_language(self, index: int):
        """Toggles the display language of tags between English and Japanese."""
        self._tag_display_language = self.language_combo.currentText()
        self.display_current_tag_page()
        self._display_image_tag_page()
        self.grid_view_widget.set_tag_display_language(self._tag_display_language, self.tag_translation_map)

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
        # has to be locked: the main caption box, the grid-view cells, and Undo/Redo.
        self.caption_text_edit.setEnabled(enabled)
        self.grid_view_widget.set_editing_enabled(enabled)
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
        self.save_current_config()

        threads_to_stop: list[tuple[QThread | None, StoppableWorker | None]] = [ # type: ignore
            (self._download_thread, self._downloader_worker),
            (self._gpu_dl_thread, self._gpu_dl_worker),
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
                # Wait up to 5 seconds. This blocks the GUI, which is acceptable on close.
                if not thread.wait(5000):
                    write_debug_log(f"ERROR: closeEvent: Thread {thread} did not finish gracefully, terminating.")
                    thread.terminate() # Last resort
                else:
                    write_debug_log(f"DEBUG: closeEvent: Thread {thread} finished gracefully.")

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
            self.update_log(f"無効なフォルダパスです: {folder_path}", "red")

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
        for button in self.tag_buttons:
            if button.isVisible():
                self.update_button_text_alignment(button)
        for button in self.tag_buttons_for_image:
            if button.isVisible():
                self.update_button_text_alignment(button)


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
        """Adds a new tag to the currently selected image's tag file."""
        current_item = self.image_list.currentItem()
        if not current_item:
            return

        tags_raw = self.add_single_tag_line.text().strip()
        if not tags_raw:
            return

        # (Normalization logic for full-width chars, spaces, commas...)
        tags_processed = ' '.join(tags_raw.replace('ã€€', ' ').split())
        while ',,' in tags_processed:
            tags_processed = tags_processed.replace(',,', ',')
        new_tags = [t.strip() for t in tags_processed.split(',') if t.strip()]
        if not new_tags:
            return

        image_path = Path(self.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
        txt_path = image_path.with_suffix('.txt')

        try:
            existing_tags = []
            if txt_path.is_file():
                existing_tags = [t.strip() for t in txt_path.read_text('utf-8').split(',') if t.strip()]
            
            # Filter out tags that already exist
            tags_to_add = [tag for tag in new_tags if tag not in existing_tags]
            
            if not tags_to_add:
                self.update_log(self.locale_manager.get_string("MainWindow", "Tag_Already_Exists"), "orange")
                return
            
            for tag in tags_to_add:
                existing_tags.append(tag)
            
            txt_path.write_text(', '.join(existing_tags), 'utf-8')
            
            # Record action for undo
            action = AddTagsAction(file_path=txt_path, added_tags=tags_to_add)
            self.undo_manager.push(action)
            self._update_undo_redo_buttons()
            
            self.update_log(self.locale_manager.get_string("MainWindow", "Tags_Added_To_File", txt_path_name=txt_path.name), "green")
            self.add_single_tag_line.clear()
            self._load_image_tags(image_path, preserve_page=True)
            rel_path = current_item.data(Qt.ItemDataRole.UserRole + 1)
            self._update_tag_cache_entry(rel_path)
            self.reload_tags_only()
        except Exception as e:
            self.update_log(self.locale_manager.get_string("MainWindow", "Error_Adding_Tags", txt_path_name=txt_path.name, e=e), "red")

    def _delete_image_tag(self, tag_to_delete: str):
        """Deletes a tag from the currently selected image's tag file."""
        write_debug_log(f"[_delete_image_tag] Attempting to delete tag: {tag_to_delete}")
        current_item = self.image_list.currentItem()
        if not current_item:
            write_debug_log("[_delete_image_tag] No image item selected.")
            return

        image_path = Path(self.settings.paths.input_dir) / current_item.data(Qt.ItemDataRole.UserRole + 1)
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
                txt_path.write_text(', '.join(tags), 'utf-8')
                write_debug_log(f"[_delete_image_tag] Successfully wrote tags to file: {txt_path.name}")
                
                # Record action for undo
                action = RemoveTagAction(file_path=txt_path, removed_tag=tag_to_delete, original_index=original_index)
                self.undo_manager.push(action)
                self._update_undo_redo_buttons()
                
                self.update_log(self.locale_manager.get_string("MainWindow", "Tag_Deleted_From_File", tag_name=tag_to_delete, file_name=txt_path.name), "green")
                self._load_image_tags(image_path, preserve_page=True)
                rel_path = current_item.data(Qt.ItemDataRole.UserRole + 1)
                self._update_tag_cache_entry(rel_path)
                write_debug_log("[_delete_image_tag] UI updated after tag deletion.")
            else:
                write_debug_log(f"[_delete_image_tag] Tag '{tag_to_delete}' not found in file: {txt_path}")
        except Exception as e:
            write_debug_log(f"[_delete_image_tag] Error during tag deletion: {e}")
            self.update_log(self.locale_manager.get_string("MainWindow", "Error_Deleting_Tag", tag_name=tag_to_delete, file_name=txt_path.name, e=e), "red")
    
    def _copy_tag_to_clipboard(self, tag_name: str):
        """Copies the tag with a trailing comma to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(tag_name + ", ")
        self.update_log(self.locale_manager.get_string("MainWindow", "Tag_Copied_To_Clipboard", tag_name=tag_name), "blue")
        write_debug_log(f"Tag copied to clipboard: {tag_name}")

    def add_tag_all(self, prepend: bool):
        if self._is_bulk_deleting:
            return
        tags_to_add = (self.add_tag_line if prepend else self.add_tag_line_append).text().strip()
        if not tags_to_add:
            return
        
        if QMessageBox.question(self, self.locale_manager.get_string("MainWindow", "Bulk_Add_Confirmation"), self.locale_manager.get_string("MainWindow", "Confirm_Bulk_Add_Tag", tags_to_add=tags_to_add)) == QMessageBox.StandardButton.Yes:
            self._start_bulk_tag_worker('add', input_dir=Path(self.settings.paths.input_dir), tags=tags_to_add, prepend=prepend)
    
    def delete_tag_all(self, tag_to_delete: str):
        """Starts a bulk process to delete a tag from all .txt files."""
        if self._is_bulk_deleting:
            return
        if QMessageBox.question(self, self.locale_manager.get_string("MainWindow", "Bulk_Delete_Confirmation"), self.locale_manager.get_string("MainWindow", "Confirm_Bulk_Delete_Tag", tag_to_delete=tag_to_delete)) == QMessageBox.StandardButton.Yes:
            self._start_bulk_tag_worker('delete', input_dir=Path(self.settings.paths.input_dir), tag=tag_to_delete)

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

    def _start_tagging_thread(self):
        """Initializes and starts the TaggerThreadWorker."""
        if self._tagger_thread and self._tagger_thread.isRunning():
            self.update_log(self.locale_manager.get_string("MainWindow", "Warning_Tagging_Already_Running"), "orange")
            return

        self._cleanup_tagger_thread()

        # 「選択画像のみ」で画像が未選択なら空実行。専用の1行ログを出して即戻る
        # （既存の「画像がありません」を流用しない。260903_vlm-gap-fix.md todo 5）。
        if str(self.settings.behavior.target_mode).upper() == "SELECTED" and self.image_list.currentItem() is None:
            self.update_log(self.locale_manager.get_string("MainWindow", "Target_No_Selected"), "orange")
            return

        self._session_mode_override = None
        self._progress_last_shown_at = 0.0

        self._update_ui_for_processing(True, 'tagging')

        # Get selected file to prioritize it
        selected_path: Path | None = None
        current_item = self.image_list.currentItem()
        if current_item:
            relative_path = current_item.data(Qt.ItemDataRole.UserRole + 1)
            selected_path = Path(self.settings.paths.input_dir) / relative_path

        # Which worker class to instantiate is the one piece of model_type branching that
        # lives outside ModelModeController: it's a thread/worker-startup decision, not a
        # UI display-state one (design.md 6.8節 "スコープ外の明記").
        # 「VLM接続を使う」が ON ならネットワーク VLM（ローカルモデル不要、tagger でも
        # captioner でも VLM が担当）。OFF なら従来どおりモデル種別で分岐。
        use_vlm = bool(self.settings.vlm.enabled)
        is_captioner = (not use_vlm) and self._current_model_entry().model_type == "captioner"

        # 「失敗のみ」モード用のセッション失敗集合を渡す（他モードでは無視される）。
        failed_paths = list(self._batch_failed_paths)

        self._tagger_thread = QThread()
        if use_vlm:
            self._tagger_worker = VlmCaptionWorker(
                self.settings, self._make_decision_requester(), self.locale_manager.get_string,
                selected_file_path=selected_path, failed_paths=failed_paths)
            self._tagger_worker.single_test_result.connect(self._on_vlm_single_test_result)
            self._tagger_worker.binding_verified.connect(self._on_vlm_binding_verified)
            self._tagger_thread.started.connect(self._tagger_worker.run_captioning)
        elif is_captioner:
            self._tagger_worker = CaptionerThreadWorker(self.settings, self._make_decision_requester(), self.locale_manager.get_string, selected_file_path=selected_path, failed_paths=failed_paths)
            self._tagger_thread.started.connect(self._tagger_worker.run_captioning)
        else:
            self._tagger_worker = TaggerThreadWorker(self.settings, self._make_decision_requester(), self.locale_manager.get_string, selected_file_path=selected_path, failed_paths=failed_paths)
            self._tagger_thread.started.connect(self._tagger_worker.run_tagging)
        self._tagger_worker.moveToThread(self._tagger_thread)

        self._tagger_worker.log_message.connect(self.update_log)
        self._tagger_worker.batch_completed.connect(self._on_batch_completed)
        self._tagger_worker.batch_failed.connect(self._on_batch_failed)
        self._tagger_worker.progress_update.connect(self._on_tagging_progress)
        self._tagger_worker.model_status_changed.connect(self._check_model_status_and_update_ui)
        self._tagger_worker.finished.connect(self._on_tagger_finished)
        # 正常終了時は worker の finished シグナルから QThread も終了させる。
        self._tagger_worker.finished.connect(self._tagger_thread.quit)

        self._tagger_thread.start()

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

    def _cleanup_tagger_thread(self):
        """Safely cleans up the existing tagger thread and worker."""
        if self._tagger_thread:
            if self._tagger_thread.isRunning():
                # 通常は worker.finished -> thread.quit で既に終了処理へ入っている。
                # ここに到達するのは停止や異常経路だけなので、ユーザー向けの通常ログ
                # へ「残存スレッド」を出さず、詳細はデバッグログへ残す。
                write_debug_log("tagger thread still running during cleanup; waiting")
                self._tagger_thread.quit()
                if not self._tagger_thread.wait(5000):
                    write_debug_log("tagger thread did not finish within cleanup timeout")
                    self.update_log(self.locale_manager.get_string(
                        "MainWindow", "Thread_Shutdown_Failed"), "red")
                    self._tagger_thread.terminate()
                    self._tagger_thread.wait(1000)
            self._tagger_thread.deleteLater()
            self._tagger_thread = None
        if self._tagger_worker:
            self._tagger_worker.deleteLater()
            self._tagger_worker = None

    def _stop_tagging_thread(self):
        """Requests the tagging thread to stop."""
        if self._tagger_worker:
            self.update_log(self.locale_manager.get_string("MainWindow", "Stopping_Tagging_Process"), "orange")
            self.run_button.setText(self.locale_manager.get_string("Constants", "Stopping_Process"))
            self.run_button.setEnabled(False)
            self._tagger_worker.stop()
            QApplication.processEvents() # Ensure UI remains responsive during stop

    def _start_download_thread(self):
        """Initializes and starts the DownloaderWorker."""
        if self._download_thread and self._download_thread.isRunning():
            return

        self._is_downloading = True
        self._update_ui_for_processing(True, 'downloading')
        self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Model_Download"), "black")

        self._download_thread = QThread()
        self._downloader_worker = DownloaderWorker(self.locale_manager.get_string, model_id=self.settings.model.model_id)
        self._downloader_worker.moveToThread(self._download_thread)

        self._downloader_worker.log_message.connect(self.update_log)
        self._downloader_worker.progress_update.connect(self._update_download_progress)
        self._downloader_worker.download_finished.connect(self._on_download_finished)
        self._download_thread.started.connect(self._downloader_worker.run_download)
        
        self._download_thread.start()

    def _stop_download_thread(self):
        """Requests the download thread to stop."""
        if self._downloader_worker:
            self.update_log(self.locale_manager.get_string("MainWindow", "Stopping_Model_Download"), "orange")
            self.run_button.setText(self.locale_manager.get_string("Constants", "Stopping_Process"))
            self.run_button.setEnabled(False)
            self._downloader_worker.stop()
    
    def _start_bulk_tag_worker(self, mode: str, **kwargs: Any):
        """Initializes and starts the BulkTagWorker for add/delete operations."""
        self._is_bulk_deleting = True
        self._set_bulk_controls_enabled(False)
        self.update_log(self.locale_manager.get_string("MainWindow", "Starting_Bulk_Tag_Operation", mode=mode), "blue")

        self._bulk_tag_thread = QThread()
        worker = BulkTagWorker(self.locale_manager.get_string)
        self._bulk_tag_worker = worker
        worker.moveToThread(self._bulk_tag_thread)

        worker.log_message.connect(self.update_log)
        worker.finished.connect(self._on_bulk_tag_finished)
        worker.bulk_add_completed.connect(self._on_bulk_add_completed)
        worker.bulk_delete_completed.connect(self._on_bulk_delete_completed)
        
        if mode == 'add':
            self._bulk_tag_thread.started.connect(lambda: worker.run_bulk_add(kwargs['input_dir'], kwargs['tags'], kwargs['prepend']))
        else: # delete
            self._bulk_tag_thread.started.connect(lambda: worker.run_bulk_delete(kwargs['input_dir'], kwargs['tag']))

        self._bulk_tag_thread.start()

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

        if not self.tag_translation_map:
            # The worker may have just fetched PixAI's selected_tags.csv on our behalf.
            self.tag_translation_map = load_tag_translation_map(constants.MODEL_PATH.parent,
                                                       constants.MODELS_DIR / constants.MODEL_DIR_NAME,
                                                       constants.MODELS_RESOURCE_DIR / constants.MODEL_DIR_NAME)
            self.grid_view_widget.set_tag_display_language(self._tag_display_language, self.tag_translation_map)

        self.reload_image_list(auto_select_path=path_to_reselect)
        self.reload_tags_only()

        self._update_ui_for_processing(False, 'tagging')
        
        self._cleanup_tagger_thread()

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
            self.tag_translation_map = load_tag_translation_map(constants.MODEL_PATH.parent,
                                                       constants.MODELS_DIR / constants.MODEL_DIR_NAME,
                                                       constants.MODELS_RESOURCE_DIR / constants.MODEL_DIR_NAME)
            
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

        self.reload_tags_only(preserve_page=True) # This will re-enable controls on finish
        if self._bulk_tag_thread:
            self._bulk_tag_thread.quit()
            self._bulk_tag_thread.wait()
            if self._bulk_tag_worker:
                self._bulk_tag_worker.deleteLater()
            self._bulk_tag_thread.deleteLater()
            self._bulk_tag_thread = self._bulk_tag_worker = None
    
    @Slot(list, list, str)
    def _on_bulk_add_completed(self, file_paths: list[Path], added_tags: list[str], position: str):
        """Records bulk add operation for undo."""
        if file_paths:
            action = BulkAddTagsAction(file_paths=file_paths, added_tags=added_tags, position=position)
            self.undo_manager.push(action)
            self._update_undo_redo_buttons()
            input_dir = Path(self.settings.paths.input_dir)
            for fp in file_paths:
                rel_path = str(fp.relative_to(input_dir))
                self._update_tag_cache_entry(rel_path)
            write_debug_log(f"Bulk add action recorded: {len(file_paths)} files, {len(added_tags)} tags")
    
    @Slot(str, list)
    def _on_bulk_delete_completed(self, removed_tag: str, file_tag_positions: list[tuple[Path, int]]):
        """Records bulk delete operation for undo."""
        if file_tag_positions:
            action = BulkRemoveTagsAction(removed_tag=removed_tag, file_tag_positions=file_tag_positions)
            self.undo_manager.push(action)
            self._update_undo_redo_buttons()
            input_dir = Path(self.settings.paths.input_dir)
            for fp, _ in file_tag_positions:
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
        self.grid_view_widget.update_tag_cache(self._tag_cache)
    
    @Slot(Path, str, int)
    def _on_gridview_tag_removed(self, file_path: Path, removed_tag: str, original_index: int):
        """Records tag removal from GridView for undo."""
        action = RemoveTagAction(file_path=file_path, removed_tag=removed_tag, original_index=original_index)
        self.undo_manager.push(action)
        self._update_undo_redo_buttons()
        write_debug_log(f"GridView remove action recorded: '{removed_tag}' from {file_path.name}")
        self._build_tag_cache()
        self.grid_view_widget.update_tag_cache(self._tag_cache)

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
        """Updates the settings object with the current UI state and saves it to file."""
        write_debug_log(self.locale_manager.get_string("MainWindow", "Saving_UI_Settings"))

        # Only update geometry if in main view and not maximized/minimized
        if self.central_widget.currentWidget() == self.main_view_widget and not self.isMaximized() and not self.isMinimized():
            self.settings.window.geometry = f"{self.width()}x{self.height()}+{self.x()}+{self.y()}"

        save_config(self.settings)
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
        """Creates a labeled slider and connects it to the settings object."""
        resolution = 1 / step if step < 1 else 1
        
        for key, row_index in keys.items():
            is_float = step < 1
            label_suffix = self.locale_manager.get_string("MainWindow", "Threshold_Suffix") if section == 'Thresholds' else self.locale_manager.get_string("MainWindow", "Max_Count_Suffix")
            label = QLabel(f"{key.capitalize()} {label_suffix}:")
            
            settings_section = getattr(self.settings, section.lower())
            initial_val = getattr(settings_section, key)
                
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * resolution), int(max_val * resolution))
            slider.setValue(int(initial_val * resolution))
            
            value_label = QLabel(f"{initial_val:.2f}" if is_float else str(int(initial_val)))
            value_label.setFixedWidth(50)
            self._sliders[f"{section.lower()}_{key}"] = (slider, value_label)

            def update_value(value: int, k: str = key, s: Any = settings_section, res: float = resolution, v_label: QLabel = value_label, is_flt: bool = is_float):
                real_val = value / res
                v_label.setText(f"{real_val:.2f}" if is_flt else str(int(real_val)))
                setattr(s, k, real_val if is_flt else int(real_val))
                if not self._suppress_slider_touch:
                    # A hand move marks the category "touched" so a later model switch
                    # won't overwrite it with that model's default (see _apply_threshold_defaults).
                    s.touched = app_settings.add_touched(s.touched, k)
                self.save_current_config() # Save settings whenever a slider value changes

            slider.valueChanged.connect(update_value)

            layout.addWidget(label, row_index, 0)
            layout.addWidget(slider, row_index, 1)
            layout.addWidget(value_label, row_index, 2)

    def _current_model_categories(self) -> list[str]:
        """Tag categories the selected model produces (ui.categories in its config),
        filtered to known names and returned in the canonical TAG_CATEGORY_NAMES order
        so the per-category dialog rows are always consistent."""
        cats = config_mapping(self._current_model_entry().config, "ui").get("categories") or ["general", "character"]
        cats_set = set(cats)
        return [c for c in app_settings.TAG_CATEGORY_NAMES if c in cats_set]

    def _sync_settings_sliders(self):
        """Refresh the four main-window sliders (general/character threshold & max) from
        self.settings without their valueChanged handler flagging them as user-touched."""
        self._suppress_slider_touch = True
        try:
            for key, (slider, value_label) in self._sliders.items():
                section_name, cat = key.split('_', 1)
                value = getattr(getattr(self.settings, section_name), cat)
                is_float = section_name == 'thresholds'
                res = 100.0 if is_float else 1.0
                slider.setValue(int(round(value * res)))
                value_label.setText(f"{value:.2f}" if is_float else str(int(value)))
        finally:
            self._suppress_slider_touch = False

    @Slot()
    def _open_category_settings_dialog(self):
        """Opens the per-category threshold / max-tag dialog for the selected model."""
        if self._category_settings_dialog is not None and self._category_settings_dialog.isVisible():
            self._category_settings_dialog.raise_()
            self._category_settings_dialog.activateWindow()
            return

        def on_changed():
            # The dialog has already written into self.settings; mirror the main sliders,
            # persist, and refresh any open tag view that depends on the values.
            self._sync_settings_sliders()
            self.save_current_config()

        def on_reset():
            # Drop "touched" for this model's categories, then re-apply its defaults.
            model_cats = set(self._current_model_categories())
            for section in (self.settings.thresholds, self.settings.limits):
                kept = app_settings.parse_touched(section.touched) - model_cats
                section.touched = ",".join(sorted(kept))
            self._model_mode._apply_threshold_defaults(self._current_model_entry())
            self._sync_settings_sliders()
            self.save_current_config()
            if self._category_settings_dialog is not None:
                self._category_settings_dialog.reload_from_settings()

        self._category_settings_dialog = CategoryTagSettingsDialog(
            self,
            get_string=self.locale_manager.get_string,
            categories=self._current_model_categories(),
            thresholds=self.settings.thresholds,
            limits=self.settings.limits,
            on_changed=on_changed,
            on_reset=on_reset,
        )
        self._category_settings_dialog.finished.connect(lambda _: setattr(self, "_category_settings_dialog", None))
        self._category_settings_dialog.show()

    def _reconnect_sliders(self):
        """Reconnects all sliders to the current self.settings object."""
        write_debug_log("Reconnecting sliders to the new settings object.")
        for key, (slider, value_label) in self._sliders.items():
            try:
                section_name, key_name = key.split('_', 1)
                settings_section = getattr(self.settings, section_name)
                initial_val = getattr(settings_section, key_name)
                is_float = isinstance(initial_val, float)
                resolution = 100.0 if is_float else 1.0

                # Disconnect old connections to prevent multiple updates
                slider.valueChanged.disconnect()

                # Update UI to reflect new settings
                slider.setValue(int(initial_val * resolution))
                value_label.setText(f"{initial_val:.2f}" if is_float else str(int(initial_val)))

                # Create and connect the new update function
                def new_update_value(value: int, s: Any = settings_section, k: str = key_name, res: float = resolution,
                                     v_label: QLabel = value_label, is_flt: bool = is_float):
                    real_val = value / res
                    v_label.setText(f"{real_val:.2f}" if is_flt else str(int(real_val)))
                    setattr(s, k, real_val)

                slider.valueChanged.connect(new_update_value)
            except Exception as e:
                write_debug_log(f"Error reconnecting slider for '{key}': {e}")

    @Slot(str, str)
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

    def _update_tag_cache_entry(self, rel_path: str) -> None:
        """指定ファイルのキャッシュエントリを更新する。"""
        input_dir = Path(self.settings.paths.input_dir)
        txt_path = (input_dir / rel_path).with_suffix('.txt')
        tags = tag_utils.read_tags(txt_path)
        self._tag_cache[rel_path] = set(tags)

    def _highlight_files_for_tag(self, tag_name: str) -> None:
        """指定タグを持つファイルをTagListWidgetでハイライトする。選択中アイテムはスキップする。"""
        from PySide6.QtGui import QColor, QBrush
        color = QColor("#4a5a2a") if self._is_dark_theme else QColor("#c8f0a0")
        brush = QBrush(color)
        current_item = self.image_list.currentItem()
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item is None or item is current_item:
                continue
            rel_path = item.data(Qt.ItemDataRole.UserRole + 1)
            tags = self._tag_cache.get(rel_path, set())
            if tag_name in tags:
                item.setBackground(brush)

    def _clear_highlight(self) -> None:
        """TagListWidgetの全アイテムのハイライトを解除する。"""
        from PySide6.QtGui import QBrush
        default_brush = QBrush()
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item is not None:
                item.setBackground(default_brush)

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
        if self.undo_manager.undo():
            self.update_log(self.locale_manager.get_string("MainWindow", "Undo_Success"), "green")
            self._refresh_ui_after_undo_redo()
        else:
            self.update_log(self.locale_manager.get_string("MainWindow", "Undo_Failed"), "red")
        self._update_undo_redo_buttons()
    
    @Slot()
    def _perform_redo(self):
        """Performs a redo operation."""
        if self.undo_manager.redo():
            self.update_log(self.locale_manager.get_string("MainWindow", "Redo_Success"), "green")
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
        self._build_tag_cache()
        
        # Refresh bulk tag list
        self.reload_tags_only(preserve_page=True)
        
        # Refresh grid view
        self.grid_view_widget.refresh_current_page()

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
        """Slot to clean up the reference to the image viewer dialog when it closes."""
        self._image_viewer_dialog = None

    @Slot()
    def _show_grid_view(self):
        """Switches the central widget to the GridViewWidget."""
        image_paths = self._get_image_paths(Path(self.settings.paths.input_dir))
        if not image_paths:
            QMessageBox.information(self, self.locale_manager.get_string("MainWindow", "No_Images_Found_Title"), self.locale_manager.get_string("MainWindow", "No_Images_Found_Message"))
            return

        self.central_widget.setCurrentWidget(self.grid_view_widget)
        self.setWindowTitle(f"{constants.MSG_WINDOW_TITLE} - Grid View")
        self.grid_view_widget.set_caption_mode(self._is_text_output_mode())
        self.grid_view_widget.load_images(image_paths, self._tag_cache, Path(self.settings.paths.input_dir))
        self.showMaximized()
        self.update_log(self.locale_manager.get_string("MainWindow", "Switched_To_Grid_View"), "blue")

    @Slot()
    def _show_main_view(self):
        """Switches the central widget back to the main view."""
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
