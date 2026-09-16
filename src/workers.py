from pathlib import Path
import json
import os
import requests
import threading
from typing import Callable, Sequence
from collections import Counter

from PySide6.QtCore import QObject, Signal, Slot

from utils import write_debug_log, calculate_sha256, GetString, default_get_string_fallback
from constants import (
    DOWNLOAD_URLS, MODEL_PATH, TAGS_CSV_PATH, MODEL_POINTER_PATH
)
from app_settings import AppSettings, update_model_verification_status
from get_pointer_huggingface import get_model_info_from_pointer
from tagging_core import (
    setup_tagger_from_settings, process_image_loop, get_image_paths_recursive,
    FileChange, OverwriteDecision, parse_target_mode, filter_target_images,
)
import model_registry
import caption_core

def _looks_like_tags_csv(data: bytes) -> bool:
    """Cheap sanity check for the downloaded selected_tags.csv: a header naming the tag
    column plus at least one data row. Also rejects an HTML error page served with 200."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return "name" in lines[0].lower()


def ensure_pixai_tags_csv(get_string: GetString, log: Callable[[str, str], None] | None = None,
                          stop_checker: Callable[[], bool] | None = None) -> bool:
    """Make sure PixAI's selected_tags.csv is present, downloading just that file if not.

    tag_utils.load_tag_translation_map() uses it as the English key list that every
    selected_tags_<lang>.csv is matched against, and the app deliberately reads
    translations from PixAI's directory no matter which model is selected. It is
    normally fetched as part of the PixAI model download, so a user who only ever
    downloads a different model would get no tag translations at all even though the
    eight translation CSVs ship with the app. It is ~0.6MB, so fetch it on demand.

    Returns True when the file is available afterwards.
    """
    if TAGS_CSV_PATH.is_file():
        return True
    if stop_checker and stop_checker():
        return False
    url = DOWNLOAD_URLS.get(TAGS_CSV_PATH)
    if not url:
        return False

    tmp = TAGS_CSV_PATH.with_name(TAGS_CSV_PATH.name + ".part")
    try:
        write_debug_log("Fetching PixAI selected_tags.csv (tag-translation key list).")
        if log:
            log(get_string("Workers", "Fetching_Translation_Base_Csv"), "blue")
        # Short connect/read timeouts and a chunked read so a stop request is honoured
        # within a chunk instead of blocking the tagging thread for the whole timeout.
        chunks: list[bytes] = []
        with requests.get(url, timeout=(5, 10), stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=65536):
                if stop_checker and stop_checker():
                    write_debug_log("selected_tags.csv fetch cancelled by stop request.")
                    return False
                chunks.append(chunk)
        data = b"".join(chunks)

        if not _looks_like_tags_csv(data):
            write_debug_log("Fetched selected_tags.csv failed its sanity check; discarding.")
            return False

        # Write via a temp file and rename: a half-written CSV would pass the is_file()
        # check above on every later run and silently produce partial translations.
        TAGS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, TAGS_CSV_PATH)
        write_debug_log(f"selected_tags.csv saved to {TAGS_CSV_PATH}")
        return True
    except Exception as e:
        # Translations simply fall back to English tag names - not worth failing over.
        write_debug_log(f"Could not fetch selected_tags.csv: {type(e).__name__}: {e}")
        return False
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


class DownloaderWorker(QObject):
    """Downloads model files and verifies their integrity."""
    log_message = Signal(str, str)
    progress_update = Signal(int, float, float) # percentage, downloaded_mb, total_mb
    download_finished = Signal(bool) # success/failure

    def __init__(self, get_string: GetString | None = None, model_id: str = "pixai-tagger-v0.9"):
        super().__init__()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self.model_id = model_id
        self._stop_event = threading.Event()
        self._file_sizes: dict[Path, int] = {} # To store expected file sizes
        self._hash_verified_files: set[Path] = set()  # Which file(s) get SHA256-checked for the current model

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def _download_single_file(self, file_path: Path, url: str, expected_sha256: str | None = None) -> bool:
        """
        Downloads a single file with progress updates and optional SHA256 verification.
        Returns True on success, False otherwise.
        """
        if self.is_stopped():
            return False

        file_name = file_path.name
        # 期待される最終サイズ。モデルポインターから取得した値があればそれを使用。
        # なければ0で初期化し、content-lengthから取得を試みる。
        expected_final_size = self._file_sizes.get(file_path, 0)
        
        if file_path.exists():
            if file_path == TAGS_CSV_PATH:
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV", file_name=file_path.name), "blue")
                write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV_Debug", file_path_name=file_path.name)), self.get_string)
                return True

            local_size = file_path.stat().st_size
            if expected_final_size > 0: # 期待サイズが分かっている場合のみチェック
                if local_size > expected_final_size:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_LocalFileTooLarge", file_name=file_path.name), "red")
                    return False
                elif local_size == expected_final_size:
                    if file_path in self._hash_verified_files and expected_sha256:
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
                        local_sha256 = calculate_sha256(file_path)
                        if local_sha256.lower() == expected_sha256.lower():
                            self._mark_model_as_verified()
                            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_HashMatch", file_name=file_path.name), "green")
                            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_HashMatch_Log", file_path_name=file_path.name)), self.get_string)
                            return True
                        else:
                            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
                            file_path.unlink(missing_ok=True)
                            return False
                    else:
                        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Skip_Existing", file_path_name=file_path.name)), self.get_string)
                        return True

        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Downloading_Model", file_name=file_name, total_size=f"{expected_final_size/1024/1024:.2f}"), "blue")
        
        downloaded_size = file_path.stat().st_size if file_path.exists() else 0
        mode = 'ab' if downloaded_size > 0 else 'wb'

        try:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_URL_Connect_Start", url=url)), self.get_string)
            headers = {'Range': f'bytes={downloaded_size}-'}
            response = requests.get(url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()
            content_length = int(response.headers.get('content-length', 0))
            
            # 進捗表示用の合計サイズを決定
            current_total_size_for_progress = expected_final_size if expected_final_size > 0 else content_length

            if content_length > 0:
                if response.status_code == 206:
                    # レジュームダウンロードの場合、進捗表示用の合計サイズを更新
                    current_total_size_for_progress = downloaded_size + content_length
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Resume_Download", content_length=f"{content_length/1024/1024:.2f}", total_size=f"{current_total_size_for_progress/1024/1024:.2f}"), "black")
                else:
                    if downloaded_size > 0:
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Warning_200_OK_Restart"), "orange")
                        downloaded_size = 0
                        mode = 'wb'
                    # 新規ダウンロードの場合、進捗表示用の合計サイズを更新
                    current_total_size_for_progress = content_length
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Info_FileSize_Header", total_size=f"{current_total_size_for_progress/1024/1024:.2f}"), "black")
            
            last_percent = int(downloaded_size * 100 / current_total_size_for_progress) if current_total_size_for_progress > 0 else 0
            self.progress_update.emit(last_percent, downloaded_size / 1024 / 1024, current_total_size_for_progress / 1024 / 1024)
            
            with open(file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.is_stopped(): break
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if current_total_size_for_progress > 0: 
                        percent = min(100, int(downloaded_size * 100 / current_total_size_for_progress))
                        self.progress_update.emit(percent, downloaded_size / 1024 / 1024, current_total_size_for_progress / 1024 / 1024)
                        last_percent = percent
            
            if self.is_stopped():
                write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Download_Aborted", file_name=file_name)), self.get_string)
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Aborted_User", file_name=file_name), "red")
                return False
            
            final_size = file_path.stat().st_size
            
            # 最終的なファイルサイズチェックは expected_final_size と比較
            # expected_final_size が0より大きい場合のみチェックを行う
            if expected_final_size > 0 and final_size != expected_final_size:
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Size_Mismatch", downloaded_size=f"{final_size/1024/1024:.2f}", total_size=f"{expected_final_size/1024/1024:.2f}"), "red")
                return False
            # expected_final_size が0の場合（期待サイズ不明の場合）は、content_length が0より大きい場合は content_length と比較する
            # 期待サイズ不明の場合は current_total_size_for_progress と比較する。
            # 206 Partial Content では content_length は「残りバイト数」でしかないため、
            # そのまま比較すると正常に再開・完了したファイルを不一致として弾いてしまう。
            elif expected_final_size == 0 and current_total_size_for_progress > 0 and final_size != current_total_size_for_progress:
                 self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Size_Mismatch", downloaded_size=f"{final_size/1024/1024:.2f}", total_size=f"{current_total_size_for_progress/1024/1024:.2f}"), "red")
                 return False

            if file_path in self._hash_verified_files and expected_sha256:
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
                local_sha256 = calculate_sha256(file_path)
                if local_sha256.lower() != expected_sha256.lower():
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
                    file_path.unlink(missing_ok=True)
                    return False
                else:
                    self._mark_model_as_verified()
            
            if not self.is_stopped():
                self.progress_update.emit(100, current_total_size_for_progress / 1024 / 1024, current_total_size_for_progress / 1024 / 1024)
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Complete", file_name=file_name), "green")
            
            return True
                
        except requests.exceptions.RequestException as e:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Network_Error", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Network_Failed"), "red")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
            return False
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Unexpected_Error", e=e)), self.get_string)
            error_msg = self.get_string("Workers", "DownloaderWorker_Error_Unexpected_File_Access")
            self.log_message.emit(error_msg, "red")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
            return False

    def _mark_model_as_verified(self):
        try:
            update_model_verification_status(self.model_id, True, self.get_string)
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_ModelVerified_Success"), "green")
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_ModelVerified_Success_Debug")), self.get_string)

        except Exception as e:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_ModelVerified_Fail"), "red")
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_ModelVerified_Fail_Debug", e=e)), self.get_string)

    @Slot()
    def run_download(self):
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Start")), self.get_string)

        if self.model_id == "pixai-tagger-v0.9":
            self._run_download_pixai_legacy()
            return

        entry = model_registry.get_model_entry(self.model_id)
        if entry is None:
            write_debug_log(f"DownloaderWorker: unknown model_id '{self.model_id}', cannot download.")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_NoPointerURL"), "red")
            self.download_finished.emit(False)
            return

        self._download_from_manifest(entry.model_dir, model_registry.config_mapping(entry.config, "network"))

    def _run_download_pixai_legacy(self):
        """Unchanged PixAI download path (design.md 5章, NFR-3: byte-for-byte the same as before multi-model support)."""
        all_success = True
        self._hash_verified_files = {MODEL_PATH}

        model_pointer_url = DOWNLOAD_URLS.get(MODEL_POINTER_PATH)
        if not model_pointer_url:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_NoPointerURL"), "red")
            self.download_finished.emit(False)
            return

        expected_sha256, expected_size = get_model_info_from_pointer(model_pointer_url, self.get_string)
        if not expected_sha256 or not expected_size:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_FailedToGetModelInfo"), "red")
            self.download_finished.emit(False)
            return

        self._file_sizes[MODEL_PATH] = expected_size

        for file_path, url in DOWNLOAD_URLS.items():
            if file_path == MODEL_POINTER_PATH:
                continue

            if self.is_stopped():
                all_success = False
                break

            success = self._download_single_file(file_path, url, expected_sha256 if file_path == MODEL_PATH else None)
            if not success:
                all_success = False
                break

        self.download_finished.emit(all_success)
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Download_Thread_Exit")), self.get_string)

    @staticmethod
    def _sidecar_looks_complete(file_path: Path) -> bool:
        """Cheap integrity check for a non-hash-verified sidecar already on disk."""
        try:
            if file_path.stat().st_size == 0:
                return False
            if file_path.suffix.lower() == ".json":
                with file_path.open("r", encoding="utf-8") as f:
                    json.load(f)
            return True
        except Exception:
            return False

    def _download_from_manifest(self, model_dir: Path, network_config: dict):
        """
        Generic downloader for any model described by a model_config.json "network" block
        (spec.md 6.1節). Rather than requiring a separate pointer_url per file in the config,
        the pointer URL is derived from each file's own resolve/main URL (every HF repo we've
        checked follows this convention: .../resolve/main/<f> <-> .../raw/main/<f>), which also
        lets a model with several hash_verified_file entries (e.g. Florence-2's 4 onnx files)
        get one pointer lookup each, instead of assuming a single shared pointer_url.
        """
        files: dict[str, str] = network_config.get("files", {})
        hash_verified_raw = network_config.get("hash_verified_file")
        if isinstance(hash_verified_raw, str):
            hash_verified_names: list[str] = [hash_verified_raw]
        else:
            hash_verified_names = list(hash_verified_raw) if hash_verified_raw else []
        self._hash_verified_files = {model_dir / name for name in hash_verified_names}

        expected_sha_by_path: dict[Path, str] = {}
        for name in hash_verified_names:
            url = files.get(name)
            if not url:
                continue
            pointer_url = url.replace("/resolve/main/", "/raw/main/")
            expected_sha256, expected_size = get_model_info_from_pointer(pointer_url, self.get_string)
            if expected_sha256 and expected_size:
                file_path = model_dir / name
                expected_sha_by_path[file_path] = expected_sha256
                self._file_sizes[file_path] = expected_size
            else:
                write_debug_log(f"DownloaderWorker: could not resolve pointer info for {name} ({pointer_url}); will download without hash verification.")

        # Fetch the shared tag-translation key list first: it is tiny, and doing it here
        # means translations work even if the (much larger) model download is aborted.
        if not self.is_stopped():
            ensure_pixai_tags_csv(self.get_string, self.log_message.emit, self.is_stopped)

        all_success = True
        for file_name, url in files.items():
            if self.is_stopped():
                all_success = False
                break
            file_path = model_dir / file_name

            if file_path not in self._hash_verified_files and file_path.is_file():
                # Small metadata/tag JSON/CSV sidecars aren't hash-verified, and HF often
                # serves them gzip-encoded with no Content-Length, so there's no reliable
                # size to compare against. Trust an existing copy only if it passes a cheap
                # sanity check (parseable JSON / non-empty) - a copy left truncated by an
                # interrupted earlier run must be re-fetched, not trusted forever.
                if self._sidecar_looks_complete(file_path):
                    write_debug_log(f"DownloaderWorker: {file_name} already exists and looks complete, skipping (not hash-verified).")
                    continue
                write_debug_log(f"DownloaderWorker: {file_name} exists but looks truncated/corrupt; re-downloading.")
                file_path.unlink(missing_ok=True)

            success = self._download_single_file(file_path, url, expected_sha_by_path.get(file_path))
            if not success:
                all_success = False
                break

        # Mark verified once the whole manifest is on disk. Without this, a model whose
        # pointer hashes never resolve (or that declares no hash_verified_file) downloads
        # fully but stays "unverified" - _is_model_available then keeps offering Download.
        if all_success:
            self._mark_model_as_verified()

        self.download_finished.emit(all_success)
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Download_Thread_Exit")), self.get_string)

class GpuRuntimeDownloadWorker(QObject):
    """任意ダウンロードの GPU コンポーネント（CUDA provider DLL ＋ NVIDIA ランタイム）を
    取得する薄いワーカー。実処理は gpu_runtime.GpuRuntimeInstaller。"""

    log_message = Signal(str, str)                 # message, color
    progress_update = Signal(int, float, float)    # percent, done_mb, total_mb
    download_finished = Signal(bool)               # success

    _LEVEL_COLOR = {"info": "blue", "warn": "orange", "error": "red"}

    def __init__(self, get_string: GetString | None = None):
        super().__init__()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = max(0, min(100, int(done * 100 / total)))
            self.progress_update.emit(pct, done / 1024 / 1024, total / 1024 / 1024)
        else:
            self.progress_update.emit(0, done / 1024 / 1024, 0.0)

    def _on_log(self, message: str, level: str = "info") -> None:
        write_debug_log(f"GpuRuntimeDownloadWorker: {message}")
        self.log_message.emit(message, self._LEVEL_COLOR.get(level, "black"))

    @Slot()
    def run_download(self):
        # download_finished は必ず 1 回発火させる（_run 内で何が起きても）。でないと
        # MainWindow 側の進捗ダイアログとスレッドが後始末されずに残る。
        ok = False
        try:
            ok = self._run()
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            write_debug_log(f"GpuRuntimeDownloadWorker: fatal {exc!r}")
        self.download_finished.emit(ok)

    def _run(self) -> bool:
        import gpu_runtime
        import onnx_providers

        spec = gpu_runtime.load_component_spec()
        if spec is None:
            self._on_log(self.get_string("Gpu", "Worker_NoSpec"), "error")
            return False
        installer = gpu_runtime.GpuRuntimeInstaller()
        # This worker only runs when gpu_runtime_ready() is False (the prompt gates on
        # that). If a stale/partial gpu_runtime/ is sitting there - a cancelled run, or a
        # previous release whose manifest listed differently named DLLs - clear it first
        # so install() starts clean (install() overwrites by name but won't delete
        # files only the old manifest knew about).
        try:
            if onnx_providers.gpu_runtime_dir().exists():
                installer.uninstall()
        except Exception as exc:  # noqa: BLE001
            write_debug_log(f"GpuRuntimeDownloadWorker: uninstall-before-reinstall skipped ({exc!r})")
        try:
            ok = installer.install(spec, progress_cb=self._on_progress,
                                   log_cb=self._on_log, stop_cb=self.is_stopped)
        except Exception as exc:  # noqa: BLE001 - defensive; installer already guards
            write_debug_log(f"GpuRuntimeDownloadWorker: unexpected {exc!r}")
            ok = False
        if ok:
            self._on_log(self.get_string("Gpu", "Worker_Done"), "info")
        elif self.is_stopped():
            self._on_log(self.get_string("Gpu", "Worker_Stopped"), "warn")
        else:
            self._on_log(self.get_string("Gpu", "Worker_Failed"), "error")
        return ok


class TaggerThreadWorker(QObject):
    """Tagging Worker"""
    log_message = Signal(str, str)
    model_status_changed = Signal()
    finished = Signal()
    running_state_changed = Signal(bool)
    reload_image_list_signal = Signal()
    # バッチ1回で書き換えたファイル群。MainWindow 側で CompositeUndoAction にまとめる。
    batch_completed = Signal(list)
    # バッチ1回で失敗した画像パス群。FAILED モードの次回実行がそのまま failed_paths
    # として渡せるよう、入力画像パス（.txt ではない）で集める。成功時は空。
    batch_failed = Signal(list)
    # 進捗（done, total）。ルーチンの「処理中／成功」ログを GUI へ垂れ流す代わりに
    # これを毎画像発行し、GUI 側で throttle する（issue #10: Qt キュー飽和対策）。
    progress_update = Signal(int, int)
    def __init__(self, settings: AppSettings, decision_requester: Callable[[Path], OverwriteDecision] | None, get_string: GetString | None = None, selected_file_path: Path | None = None, failed_paths: Sequence[Path] = ()):
        super().__init__()
        self._settings: AppSettings = settings
        # ASK モードのときだけ呼ばれる GUI 応答ブリッジ（design.md 3.1節）。
        # ASK 以外ではワーカーからこれを呼ぶことはない。
        self._decision_requester = decision_requester
        self._selected_file_path = selected_file_path
        self._failed_paths = tuple(failed_paths)
        self._stop_event = threading.Event()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def _mark_model_as_unverified(self):
        try:
            update_model_verification_status(self._settings.model.model_id, False, self.get_string)
            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_ModelUnverified"), "orange")
            self.model_status_changed.emit()
        except Exception as e:
            write_debug_log(str(f"Debug: Failed to save model unverified status: {e}"), self.get_string)

    @Slot()
    def run_tagging(self):
        write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Start")), self.get_string)
        self.running_state_changed.emit(True)
        write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Tagging_Process_Start")), self.get_string)
        failed: list[Path] = []
        try:
            tagger, settings_dict = setup_tagger_from_settings(self._settings, self.get_string)
            if not tagger or not settings_dict:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Error_Tagger_Init_Failed"), "red")
                self._mark_model_as_unverified()
                # running_state_changed / reload / finished are all emitted once by the
                # finally block below - do not emit them here too (double reload / double
                # finished otherwise).
                return

            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Loading_Model"), "black")

            # Surface which execution provider the tagger session actually got, so the
            # user can see GPU acceleration is (or isn't) active without opening the log
            # file. onnx_providers already handled any fallback; this is just a report.
            try:
                on_gpu = tagger.session.get_providers()[0] == "CUDAExecutionProvider"
            except Exception:
                on_gpu = False
            self.log_message.emit(
                self.get_string("Workers", "TaggerThreadWorker_Provider_Gpu" if on_gpu
                                else "TaggerThreadWorker_Provider_Cpu"),
                "green" if on_gpu else "black")

            # Whichever model is tagging, tag translations are looked up against PixAI's
            # selected_tags.csv - grab it here if a previous run never pulled it in.
            ensure_pixai_tags_csv(self.get_string, self.log_message.emit, self.is_stopped)

            input_dir = Path(settings_dict['INPUT_DIR'])
            image_paths = get_image_paths_recursive(input_dir)

            # 対象フィルタは列挙直後に適用する。SELECTED モードでは従来の
            # selected-to-front 並べ替えを置き換える（filter が [selected] を返す）。
            target_mode = parse_target_mode(getattr(self._settings.behavior, "target_mode", "ALL"), self.get_string)
            image_paths = filter_target_images(image_paths, target_mode, failed_paths=self._failed_paths, selected=self._selected_file_path)

            if not image_paths:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Warning_No_Image_Files", input_dir=input_dir), "orange")
                return  # finally block emits running_state_changed / reload / finished

            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Total_Image_Files", count=len(image_paths)), "blue")

            def log_to_gui(message: str, color: str):
                write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Core_Log", message=message)), self.get_string)
                self.log_message.emit(message, color)

            changed_files = process_image_loop(
                tagger=tagger,
                image_paths=image_paths,
                settings=settings_dict,
                decision_resolver=self._decision_requester,
                log_gui=log_to_gui,
                stop_checker=self.is_stopped,
                get_string=self.get_string,
                progress_cb=self.progress_update.emit,
                failed_paths=failed,
            )
            self.batch_completed.emit(changed_files)
            self.batch_failed.emit(failed)
            
        except Exception as e:
            import traceback
            error_message = self.get_string("Workers", "TaggerThreadWorker_Fatal_Exception", type_e_name=type(e).__name__, e=e, traceback_exc=traceback.format_exc())
            self.log_message.emit(error_message, "red")
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Runtime_Exception", e=e, traceback_exc=traceback.format_exc())), self.get_string)
            self.batch_failed.emit(failed)
        
        finally:
            self.running_state_changed.emit(False)
            self.reload_image_list_signal.emit()
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Thread_Exit")), self.get_string)

class CaptionerThreadWorker(QObject):
    """
    Florence-2 (captioner model_type) counterpart of TaggerThreadWorker (spec.md 6.2節).
    Same signal shape and lifecycle so MainWindow's wiring is symmetric; internally it calls
    caption_core.setup_captioner_from_settings / process_caption_loop instead of tagging_core's.
    """
    log_message = Signal(str, str)
    model_status_changed = Signal()
    finished = Signal()
    running_state_changed = Signal(bool)
    reload_image_list_signal = Signal()
    # バッチ1回で書き換えたファイル群。MainWindow 側で CompositeUndoAction にまとめる。
    batch_completed = Signal(list)
    batch_failed = Signal(list)
    # 進捗（done, total）。GUI 側で throttle する（issue #10: Qt キュー飽和対策）。
    progress_update = Signal(int, int)

    def __init__(self, settings: AppSettings, decision_requester: Callable[[Path], OverwriteDecision] | None, get_string: GetString | None = None, selected_file_path: Path | None = None, failed_paths: Sequence[Path] = ()):
        super().__init__()
        self._settings: AppSettings = settings
        # ASK モードのときだけ呼ばれる GUI 応答ブリッジ（design.md 3.1節）。
        # ASK 以外ではワーカーからこれを呼ぶことはない。
        self._decision_requester = decision_requester
        self._selected_file_path = selected_file_path
        self._failed_paths = tuple(failed_paths)
        self._stop_event = threading.Event()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _mark_model_as_unverified(self):
        try:
            update_model_verification_status(self._settings.model.model_id, False, self.get_string)
            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_ModelUnverified"), "orange")
            self.model_status_changed.emit()
        except Exception as e:
            write_debug_log(str(f"Debug: Failed to save model unverified status: {e}"), self.get_string)

    @Slot()
    def run_captioning(self):
        write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Start")), self.get_string)
        self.running_state_changed.emit(True)
        failed: list[Path] = []
        try:
            captioner, settings_dict = caption_core.setup_captioner_from_settings(self._settings, self.get_string)
            if not captioner or not settings_dict:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Error_Tagger_Init_Failed"), "red")
                if caption_core.ort is None or caption_core.Tokenizer is None or caption_core.Image is None:
                    # Most common cause for a captioner: the optional deps aren't installed
                    # in this environment (tokenizers was added to requirements.txt later).
                    # The downloaded model files are intact - don't clear verified, or the
                    # user is forced to re-download the whole model after `pip install`.
                    self.log_message.emit(self.get_string("CaptionCore", "Required_Libraries_NotFound"), "red")
                else:
                    self._mark_model_as_unverified()
                return  # finally block emits running_state_changed / reload / finished

            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Loading_Model"), "black")

            # No ensure_pixai_tags_csv() here on purpose: captions are free text and never
            # go through the tag translation map, so fetching it would be unrelated network
            # work - and because a failed fetch writes nothing, it would be retried (and pay
            # the connect/read timeout) on every captioning run. The tagger worker and the
            # downloader still fetch it, covering every path that actually needs it.

            input_dir = Path(settings_dict['INPUT_DIR'])
            image_paths = get_image_paths_recursive(input_dir)

            target_mode = parse_target_mode(getattr(self._settings.behavior, "target_mode", "ALL"), self.get_string)
            image_paths = filter_target_images(image_paths, target_mode, failed_paths=self._failed_paths, selected=self._selected_file_path)

            if not image_paths:
                self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Warning_No_Image_Files", input_dir=input_dir), "orange")
                return  # finally block emits running_state_changed / reload / finished

            self.log_message.emit(self.get_string("Workers", "TaggerThreadWorker_Total_Image_Files", count=len(image_paths)), "blue")

            def log_to_gui(message: str, color: str):
                self.log_message.emit(message, color)

            changed_files = caption_core.process_caption_loop(
                captioner=captioner,
                settings=settings_dict,
                image_paths=image_paths,
                decision_resolver=self._decision_requester,
                log_gui=log_to_gui,
                stop_checker=self.is_stopped,
                get_string=self.get_string,
                progress_cb=self.progress_update.emit,
                failed_paths=failed,
            )
            self.batch_completed.emit(changed_files)
            self.batch_failed.emit(failed)

        except Exception as e:
            import traceback
            error_message = self.get_string("Workers", "TaggerThreadWorker_Fatal_Exception", type_e_name=type(e).__name__, e=e, traceback_exc=traceback.format_exc())
            self.log_message.emit(error_message, "red")
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Runtime_Exception", e=e, traceback_exc=traceback.format_exc())), self.get_string)
            self.batch_failed.emit(failed)

        finally:
            self.running_state_changed.emit(False)
            self.reload_image_list_signal.emit()
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "TaggerThreadWorker_Thread_Exit")), self.get_string)

class TagLoader(QObject):
    """Worker to asynchronously load tag files from an image folder"""
    log_message = Signal(str, str)
    tags_loaded = Signal(list)
    finished = Signal()
    def __init__(self, folder: Path, get_string: GetString | None = None):
        super().__init__()
        self.folder = folder
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def run(self):
        write_debug_log(str(self.get_string("Workers", "TagLoader_Start", folder=self.folder)), self.get_string)
        counter: Counter[str] = Counter()
        files = list(self.folder.rglob("*.txt"))
        try:
            for txt in files:
                if self.is_stopped():
                    break
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        tags = [t.strip() for t in f.read().split(",") if t.strip()]
                        counter.update(tags)
                except Exception as e:
                    write_debug_log(str(self.get_string("Workers", "TagLoader_TXT_Load_Failed", txt_name=txt.name, e=e)), self.get_string)
            
            if not self.is_stopped():
                all_tags: list[tuple[str, int]] = counter.most_common() 
                self.tags_loaded.emit(all_tags)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "TagLoader_Fatal_Error", e=e)), self.get_string)
            if not self.is_stopped():
                self.tags_loaded.emit([])
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "TagLoader_Thread_Exit")), self.get_string)

class BulkTagWorker(QObject):
    """Worker to execute bulk tag editing (add/delete)"""
    log_message = Signal(str, str)
    finished = Signal()
    bulk_add_completed = Signal(list, list, str)  # (file_paths, added_tags, position)
    bulk_delete_completed = Signal(str, list)  # (removed_tag, file_tag_positions)

    def __init__(self, get_string: GetString | None = None):
        super().__init__()
        self.get_string: GetString = get_string if get_string else default_get_string_fallback
        self._stop_event = threading.Event()

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        is_set = self._stop_event.is_set()
        if is_set:
            write_debug_log(f"DEBUG: {type(self).__name__}.is_stopped() returning True.")
        return is_set

    def _process_tag_file(self, txt_file_path: Path, tag_operation_callback: Callable[[list[str]], list[str]]) -> bool:
        try:
            with open(txt_file_path, "r", encoding="utf-8") as f:
                existing_tags = [t.strip() for t in f.read().split(',') if t.strip()]
            
            modified_tags = tag_operation_callback(existing_tags)
            
            if set(existing_tags) != set(modified_tags):
                new_content = ", ".join(modified_tags)
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return True
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_File_Processing_Failed", txt_name=txt_file_path.name, e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_File_Processing_Failed", txt_name=txt_file_path.name), "red")
        return False

    @Slot(Path, str)
    def run_bulk_delete(self, input_dir: Path, tag_to_delete: str):
        write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Start", tag_to_delete=tag_to_delete)), self.get_string)
        count = 0
        file_tag_positions: list[tuple[Path, int]] = []
        
        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                
                # Record original position before deletion
                try:
                    with open(txt, "r", encoding="utf-8") as f:
                        existing_tags = [t.strip() for t in f.read().split(',') if t.strip()]
                    
                    if tag_to_delete in existing_tags:
                        original_index = existing_tags.index(tag_to_delete)
                        
                        def delete_callback(tags: list[str]) -> list[str]:
                            return [t for t in tags if t != tag_to_delete]

                        if self._process_tag_file(txt, delete_callback):
                            file_tag_positions.append((txt, original_index))
                            count += 1
                except Exception as e:
                    write_debug_log(f"Error processing {txt}: {e}", self.get_string)
            
            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Complete", count=count, tag_to_delete=tag_to_delete), "green")
                # Emit signal with undo information
                self.bulk_delete_completed.emit(tag_to_delete, file_tag_positions)
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Count", count=count)), self.get_string)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Delete", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Delete", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Delete_Thread_Exit")), self.get_string)

    @Slot(Path, str, bool)
    def run_bulk_add(self, input_dir: Path, tags_to_add: str, prepend: bool):
        write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Start", tags_to_add=tags_to_add)), self.get_string)
        count = 0
        modified_files: list[Path] = []
        
        new_tags_to_add = sorted(list(set([t.strip() for t in tags_to_add.split(',') if t.strip()])))
        if not new_tags_to_add:
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Warning_No_Valid_Tags_To_Add"), "orange")
            self.finished.emit()
            return

        try:
            for txt in input_dir.rglob("*.txt"):
                if self.is_stopped():
                    break
                
                def add_callback(existing_tags: list[str]) -> list[str]:
                    if prepend:
                        return [tag for tag in new_tags_to_add if tag not in existing_tags] + existing_tags
                    else:
                        return existing_tags + [tag for tag in new_tags_to_add if tag not in existing_tags]

                if self._process_tag_file(txt, add_callback):
                    modified_files.append(txt)
                    count += 1

            if not self.is_stopped():
                self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Complete", count=count, tags_to_add=tags_to_add), "green")
                # Emit signal with undo information
                position = "prepend" if prepend else "append"
                self.bulk_add_completed.emit(modified_files, new_tags_to_add, position)
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Count", count=count)), self.get_string)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Unexpected_Error_Bulk_Add", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "BulkTagWorker_Error_Unexpected_Bulk_Add", e=e), "red")
        finally:
            self.finished.emit()
            write_debug_log(str(self.get_string("Workers", "BulkTagWorker_Bulk_Add_Thread_Exit")), self.get_string)
