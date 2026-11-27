from pathlib import Path
import csv
import json
import os
import requests
import threading
from dataclasses import dataclass
from typing import Callable, Sequence
from collections import Counter

from PySide6.QtCore import QObject, Signal, Slot

from utils import write_debug_log, calculate_sha256, GetString, default_get_string_fallback
from text_io import atomic_write_text
from constants import (
    DOWNLOAD_URLS, MODEL_PATH, TAGS_CSV_PATH, MODEL_POINTER_PATH
)
from app_settings import AppSettings, update_model_verification_status
from get_pointer_huggingface import get_model_info_from_pointer
from tagging_core import (
    setup_tagger_from_settings, process_image_loop, get_image_paths_recursive,
    FileChange, OverwriteDecision, parse_target_mode, filter_target_images,
    TAGS_CSV_COLUMN_COUNTS,
)
import tagging_core
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


@dataclass(frozen=True)
class FileDownloadResult:
    """1ファイル分の DL 結果。

    `bool()` は成功／失敗（既存の `if not success:` をそのまま使える）。
    `suspected_incomplete` は REQ-E6 の「不完全の疑い」で、成功／失敗とは別軸。
    期待サイズも Content-Length も無いまま正常終端したファイルは、完全性を
    証明できないため呼び出し側は verified を付けてはいけない。
    """
    ok: bool
    suspected_incomplete: bool = False

    def __bool__(self) -> bool:
        return self.ok


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
        # REQ-E3/E6: CSV サイドカーの形式確認に使う、このモデルの tags_csv.format。
        self._tags_csv_format: str | None = None

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self):
        return self._stop_event.is_set()

    def _download_single_file(
        self,
        file_path: Path,
        url: str,
        expected_sha256: str | None = None,
        *,
        mark_verified: bool = True,
        allow_restart: bool = True,
    ) -> "FileDownloadResult":
        """Downloads a single file with progress updates and optional SHA256 verification.

        `mark_verified=False` is passed by the manifest path, which marks the model
        verified only after every file succeeded (REQ-E2). The legacy PixAI path keeps
        the default: its model.onnx success is the only place that marks it.

        `suspected_incomplete=True` means the transfer ended normally but neither an
        expected size nor a Content-Length was available, so completeness cannot be
        proven (REQ-E6). The caller must then not mark the model verified.

        `allow_restart` bounds the single 416 re-fetch so a server that keeps answering
        416 cannot recurse forever (REQ-E4).
        """
        if self.is_stopped():
            return FileDownloadResult(False)

        file_name = file_path.name
        # 期待される最終サイズ。モデルポインターから取得した値があればそれを使用。
        # なければ0で初期化し、content-lengthから取得を試みる。
        expected_final_size = self._file_sizes.get(file_path, 0)

        if file_path.exists():
            if file_path == TAGS_CSV_PATH:
                # REQ-E6: 実在だけで「取得済み」と判断しない。形式確認を通ったものだけ
                # 再利用し、途中で切れた CSV は取り直す（従来は無条件にスキップしていた）。
                if self._sidecar_looks_complete(file_path, self._tags_csv_format):
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV", file_name=file_path.name), "blue")
                    write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Skip_Existing_Tags_CSV_Debug", file_path_name=file_path.name)), self.get_string)
                    return FileDownloadResult(True)
                write_debug_log(f"DownloaderWorker: existing {file_name} does not look like a complete tags CSV; re-downloading.")
                file_path.unlink(missing_ok=True)
            else:
                local_size = file_path.stat().st_size
                if expected_final_size > 0: # 期待サイズが分かっている場合のみチェック
                    if local_size > expected_final_size:
                        # REQ-E7: 削除しないと何度再実行しても同じ分岐で即失敗し、アプリ内から
                        # 永遠に回復できない。ハッシュ不一致分岐と同じく unlink して自己回復させる
                        # （ログ文言も「削除される」と約束している）。
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_LocalFileTooLarge", file_name=file_path.name), "red")
                        file_path.unlink(missing_ok=True)
                        return FileDownloadResult(False)
                    elif local_size == expected_final_size:
                        return FileDownloadResult(
                            self._verify_complete_local_file(
                                file_path, expected_sha256, mark_verified=mark_verified)
                        )

        file_path.parent.mkdir(parents=True, exist_ok=True)
        if expected_final_size > 0:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Downloading_Model", file_name=file_name, total_size=f"{expected_final_size/1024/1024:.2f}"), "blue")
        else:
            # REQ-E9: 期待サイズが無いファイルに "(0.00 MB)" と表示しない。
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Downloading_Model_UnknownSize", file_name=file_name), "blue")

        downloaded_size = file_path.stat().st_size if file_path.exists() else 0
        mode = 'ab' if downloaded_size > 0 else 'wb'

        try:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_URL_Connect_Start", url=url)), self.get_string)
            headers = {'Range': f'bytes={downloaded_size}-'}
            # REQ-E8: レスポンスは with で確実に閉じる（例外時も解放される）。
            with requests.get(url, stream=True, timeout=10, headers=headers) as response:
                if response.status_code == 416:
                    # REQ-E4: 416 は「Range の開始位置が既にファイル末尾以降」。
                    # 期待サイズ既知でローカルがちょうどそのサイズか、SHA256 が分かっていれば
                    # 「完全」を証明できる（後者は gpu_runtime と同じ考え方）。
                    can_prove_complete = (
                        (expected_final_size > 0 and downloaded_size == expected_final_size)
                        or bool(expected_sha256)
                    )
                    if can_prove_complete:
                        write_debug_log(f"DownloaderWorker: server returned 416 for {file_name} but the local file can be proven complete; verifying it instead of re-downloading.")
                        return FileDownloadResult(
                            self._verify_complete_local_file(
                                file_path, expected_sha256, mark_verified=mark_verified)
                        )
                    # 完全とみなせない場合は部分ファイルを捨てて最初から取り直す。
                    write_debug_log(f"DownloaderWorker: server returned 416 for {file_name}; discarding the partial file and restarting from scratch.")
                    file_path.unlink(missing_ok=True)
                    if allow_restart:
                        return self._download_single_file(
                            file_path, url, expected_sha256,
                            mark_verified=mark_verified, allow_restart=False)
                    return FileDownloadResult(False)

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
                            # REQ-5.16: 8KB チャンクごとに emit すると 1.2GB のモデルで
                            # 約 15 万回のクロススレッドシグナルになる。パーセントが
                            # 変化したときだけ発行する（最大 101 回＋完了時の 100）。
                            # 停止チェック（is_stopped）の粒度は変えない。
                            if percent != last_percent:
                                self.progress_update.emit(percent, downloaded_size / 1024 / 1024, current_total_size_for_progress / 1024 / 1024)
                                last_percent = percent

                if self.is_stopped():
                    write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Download_Aborted", file_name=file_name)), self.get_string)
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Aborted_User", file_name=file_name), "red")
                    return FileDownloadResult(False)

                final_size = file_path.stat().st_size

                # 最終的なファイルサイズチェックは expected_final_size と比較
                # expected_final_size が0より大きい場合のみチェックを行う
                if expected_final_size > 0 and final_size != expected_final_size:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Size_Mismatch", downloaded_size=f"{final_size/1024/1024:.2f}", total_size=f"{expected_final_size/1024/1024:.2f}"), "red")
                    return FileDownloadResult(False)
                # expected_final_size が0の場合（期待サイズ不明の場合）は、content_length が0より大きい場合は content_length と比較する
                # 期待サイズ不明の場合は current_total_size_for_progress と比較する。
                # 206 Partial Content では content_length は「残りバイト数」でしかないため、
                # そのまま比較すると正常に再開・完了したファイルを不一致として弾いてしまう。
                elif expected_final_size == 0 and current_total_size_for_progress > 0 and final_size != current_total_size_for_progress:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Size_Mismatch", downloaded_size=f"{final_size/1024/1024:.2f}", total_size=f"{current_total_size_for_progress/1024/1024:.2f}"), "red")
                    return FileDownloadResult(False)

                has_hash_target = file_path in self._hash_verified_files and bool(expected_sha256)
                suspected_incomplete = (
                    not has_hash_target
                    and expected_final_size == 0
                    and content_length == 0
                )
                if suspected_incomplete:
                    # REQ-E6(a): 警告を残し、(b) 呼び出し側で verified を付けない。
                    write_debug_log(f"DownloaderWorker: {file_name} finished without an expected size or Content-Length; its completeness cannot be verified.")

                if file_path in self._hash_verified_files and expected_sha256:
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
                    local_sha256 = calculate_sha256(file_path)
                    if local_sha256.lower() != expected_sha256.lower():
                        self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
                        file_path.unlink(missing_ok=True)
                        return FileDownloadResult(False)
                    elif mark_verified:
                        self._mark_model_as_verified()

                if not self.is_stopped():
                    self.progress_update.emit(100, current_total_size_for_progress / 1024 / 1024, current_total_size_for_progress / 1024 / 1024)
                    self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Download_Complete", file_name=file_name), "green")

                return FileDownloadResult(True, suspected_incomplete)

        except requests.exceptions.RequestException as e:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Network_Error", e=e)), self.get_string)
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Network_Failed"), "red")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
            return FileDownloadResult(False)
        except Exception as e:
            write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Unexpected_Error", e=e)), self.get_string)
            error_msg = self.get_string("Workers", "DownloaderWorker_Error_Unexpected_File_Access")
            self.log_message.emit(error_msg, "red")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_Details", type_e_name=type(e).__name__, e=e), "red")
            return FileDownloadResult(False)

    def _verify_complete_local_file(self, file_path: Path, expected_sha256: str | None, *,
                                    mark_verified: bool) -> bool:
        """サイズ既知で「既に完全」と分かっているローカルファイルを検証する。

        既存ファイルのスキップ分岐と 416 応答の両方から使う（REQ-E4）。
        `mark_verified=False` はマニフェスト経路（REQ-E2）。
        """
        if file_path in self._hash_verified_files and expected_sha256:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_VerifyingHash", file_name=file_path.name), "blue")
            local_sha256 = calculate_sha256(file_path)
            if local_sha256.lower() == expected_sha256.lower():
                if mark_verified:
                    self._mark_model_as_verified()
                self.log_message.emit(self.get_string("Workers", "DownloaderWorker_HashMatch", file_name=file_path.name), "green")
                write_debug_log(str(self.get_string("Workers", "DownloaderWorker_HashMatch_Log", file_path_name=file_path.name)), self.get_string)
                return True
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_HashMismatch", file_name=file_path.name), "red")
            file_path.unlink(missing_ok=True)
            return False
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Skip_Existing", file_path_name=file_path.name)), self.get_string)
        return True

    def _mark_model_as_unverified(self):
        """REQ-E6(b): 不完全の疑いがあるファイルを含むモデルの verified を取り消す。"""
        try:
            update_model_verification_status(self.model_id, False, self.get_string)
            write_debug_log(f"DownloaderWorker: cleared the verified flag for '{self.model_id}' because a downloaded file could not be proven complete.")
        except Exception as e:
            write_debug_log(f"DownloaderWorker: failed to clear the verified flag for '{self.model_id}': {e}")

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
        """Download entry point. Guarantees exactly one `download_finished` emission.

        REQ-E5: an unexpected exception used to leave the UI stuck in "downloading" and
        orphan the thread, because the signal was emitted inside each path's helper.
        """
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Start")), self.get_string)
        ok = False
        try:
            ok = self._download()
        except Exception as exc:  # noqa: BLE001 - last-resort guard for the worker
            write_debug_log(f"DownloaderWorker: fatal {exc!r}")
        self.download_finished.emit(ok)
        write_debug_log(str(self.get_string("Workers", "DownloaderWorker_Download_Thread_Exit")), self.get_string)

    def _download(self) -> bool:
        entry = model_registry.get_model_entry(self.model_id)
        if entry is not None:
            # REQ-E3/E6: CSV サイドカーの形式確認に、そのモデルの tags_csv.format を使う。
            self._tags_csv_format = model_registry.config_mapping(entry.config, "tags_csv").get("format")

        if self.model_id == "pixai-tagger-v0.9":
            return self._run_download_pixai_legacy()

        if entry is None:
            write_debug_log(f"DownloaderWorker: unknown model_id '{self.model_id}', cannot download.")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_NoPointerURL"), "red")
            return False

        return self._download_from_manifest(entry.model_dir, model_registry.config_mapping(entry.config, "network"))

    def _run_download_pixai_legacy(self) -> bool:
        """Unchanged PixAI download path (design.md 5章, NFR-3) - except REQ-E2/E6."""
        all_success = True
        self._hash_verified_files = {MODEL_PATH}

        model_pointer_url = DOWNLOAD_URLS.get(MODEL_POINTER_PATH)
        if not model_pointer_url:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_NoPointerURL"), "red")
            return False

        expected_sha256, expected_size = get_model_info_from_pointer(model_pointer_url, self.get_string)
        if not expected_sha256 or not expected_size:
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_FailedToGetModelInfo"), "red")
            return False

        self._file_sizes[MODEL_PATH] = expected_size

        suspected_incomplete = False
        for file_path, url in DOWNLOAD_URLS.items():
            if file_path == MODEL_POINTER_PATH:
                continue

            if self.is_stopped():
                all_success = False
                break

            result = self._download_single_file(file_path, url, expected_sha256 if file_path == MODEL_PATH else None)
            if not result:
                all_success = False
                break
            if result.suspected_incomplete:
                suspected_incomplete = True

        if all_success and suspected_incomplete:
            # REQ-E6(b): レガシー経路はモデル本体の印しが後続ファイルより先に付くため、
            # 不完全が判明した時点で取り消す。
            self._mark_model_as_unverified()

        return all_success

    def _sidecar_looks_complete(self, file_path: Path, csv_format: str | None = None) -> bool:
        """Cheap integrity check for a non-hash-verified sidecar already on disk.

        JSON must parse. CSV must be structurally sound (REQ-E3): every row has the same
        number of columns and the header matches the model's declared csv_format. When
        `csv_format` is unknown, only the column-count consistency is checked.
        """
        try:
            if file_path.stat().st_size == 0:
                return False
            suffix = file_path.suffix.lower()
            if suffix == ".json":
                with file_path.open("r", encoding="utf-8") as f:
                    json.load(f)
                return True
            if suffix == ".csv":
                return self._csv_sidecar_looks_complete(file_path, csv_format)
            return True
        except Exception:
            return False

    @staticmethod
    def _csv_sidecar_looks_complete(file_path: Path, csv_format: str | None) -> bool:
        """CSV サイドカーの形式確認（REQ-E3）。

        行の境界で綺麗に切れた切り詰めは、パースも列構造も成立するため検出できない
        （計画 §13 REQ-E3 に明記された限界。行の途中で切れたものと列構造の崩れを検出する）。
        """
        expected_columns = TAGS_CSV_COLUMN_COUNTS.get(csv_format or "")
        try:
            with file_path.open("r", encoding="utf-8-sig", newline="") as fp:
                rows = [row for row in csv.reader(fp) if any(cell.strip() for cell in row)]
        except (OSError, UnicodeDecodeError, csv.Error):
            return False
        if len(rows) < 2:  # ヘッダ＋最低1データ行
            return False
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            return False
        return expected_columns is None or len(rows[0]) == expected_columns

    def _download_from_manifest(self, model_dir: Path, network_config: dict) -> bool:
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
        pointer_failures: list[str] = []
        for name in hash_verified_names:
            url = files.get(name)
            if not url:
                pointer_failures.append(name)
                continue
            pointer_url = url.replace("/resolve/main/", "/raw/main/")
            expected_sha256, expected_size = get_model_info_from_pointer(pointer_url, self.get_string)
            if expected_sha256 and expected_size:
                file_path = model_dir / name
                expected_sha_by_path[file_path] = expected_sha256
                self._file_sizes[file_path] = expected_size
            else:
                pointer_failures.append(name)

        if pointer_failures:
            # REQ-E1: 期待ハッシュ／サイズが無い状態で DL を進めない（レガシー経路に揃える）。
            # 進めてしまうと、ハッシュ検証なしのファイルに verified が付きうる。
            write_debug_log(f"DownloaderWorker: could not resolve pointer info for {pointer_failures}; aborting the download.")
            self.log_message.emit(self.get_string("Workers", "DownloaderWorker_Error_FailedToGetModelInfo"), "red")
            return False

        # Fetch the shared tag-translation key list first: it is tiny, and doing it here
        # means translations work even if the (much larger) model download is aborted.
        if not self.is_stopped():
            ensure_pixai_tags_csv(self.get_string, self.log_message.emit, self.is_stopped)

        all_success = True
        suspected_incomplete = False
        for file_name, url in files.items():
            if self.is_stopped():
                all_success = False
                break
            file_path = model_dir / file_name

            if file_path not in self._hash_verified_files and file_path.is_file():
                # Small metadata/tag JSON/CSV sidecars aren't hash-verified, and HF often
                # serves them gzip-encoded with no Content-Length, so there's no reliable
                # size to compare against. Trust an existing copy only if it passes a cheap
                # sanity check (parseable JSON / structurally sound CSV) - a copy left
                # truncated by an interrupted earlier run must be re-fetched, not trusted
                # forever.
                if self._sidecar_looks_complete(file_path, self._tags_csv_format):
                    write_debug_log(f"DownloaderWorker: {file_name} already exists and looks complete, skipping (not hash-verified).")
                    continue
                write_debug_log(f"DownloaderWorker: {file_name} exists but looks truncated/corrupt; re-downloading.")
                file_path.unlink(missing_ok=True)

            # REQ-E2: マニフェスト経路では verified をここで付けず、全ファイル成功後に付ける。
            result = self._download_single_file(
                file_path, url, expected_sha_by_path.get(file_path), mark_verified=False)
            if not result:
                all_success = False
                break
            if result.suspected_incomplete:
                suspected_incomplete = True

        # Mark verified only once the whole manifest is on disk and no file's completeness
        # is in doubt (REQ-E2/E6). Without this, a model whose pointer hashes never resolve
        # (or that declares no hash_verified_file) downloads fully but stays "unverified" -
        # _is_model_available then keeps offering Download.
        if all_success and not suspected_incomplete:
            self._mark_model_as_verified()
        elif all_success:
            write_debug_log("DownloaderWorker: not marking the model verified because at least one file could not be proven complete.")

        return all_success

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
        # REQ-5.19: 進捗シグナルの間引き用（直前に発行したパーセント）。
        self._last_progress_percent = -1

    def stop(self):
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _on_progress(self, done: int, total: int) -> None:
        # REQ-5.19: gpu_runtime は 256KB チャンクごとに呼ぶ（合計約 2GB で約 8,000 回）。
        # パーセントが変化したときだけ emit する。合計不明（total == 0）はパーセントが
        # 定義できないため現行どおり毎回発行する。
        if total > 0:
            pct = max(0, min(100, int(done * 100 / total)))
            if pct != self._last_progress_percent:
                self._last_progress_percent = pct
                self.progress_update.emit(pct, done / 1024 / 1024, total / 1024 / 1024)
        else:
            self.progress_update.emit(0, done / 1024 / 1024, 0.0)

    def _on_log(self, message: str, level: str = "info", *,
                key: str = "", args: dict | None = None) -> None:
        """gpu_runtime からの進捗ログを、利用者の言語でログ欄へ出す。

        `message` は英語のまま（デバッグログはこちらを残す）。`key` を伴う通知は
        [Gpu] セクションから訳して表示する。gpu_runtime 側は LocaleManager を
        持たない純粋なモジュールなので、翻訳はここで行う。
        """
        write_debug_log(f"GpuRuntimeDownloadWorker: {message}")
        shown = message
        if key:
            translated = self.get_string("Gpu", key, **(args or {}))
            if translated and translated != key:
                shown = translated
        self.log_message.emit(shown, self._LEVEL_COLOR.get(level, "black"))

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
        # Only clear out a previous *completed* install (manifest.json present means
        # an older spec/ORT-version fully installed here before; its files may not
        # overlap by name with the current spec's, so a plain install() wouldn't clean
        # them up on its own - install() overwrites by name but won't delete files
        # only the old manifest knew about).
        #
        # Do NOT uninstall() just because gpu_runtime/ exists at all: install() now
        # resumes an interrupted download instead of restarting the ~2GB total from
        # zero (see gpu_runtime.GpuRuntimeInstaller.install()/`_download`), and the
        # state that makes that possible - gpu_runtime/ present with only a
        # `.staging/` subdirectory and no manifest.json yet, left by a prior
        # cancelled/failed attempt - is exactly what the old unconditional uninstall()
        # here used to wipe on every single retry, silently defeating resume entirely.
        try:
            if (onnx_providers.gpu_runtime_dir() / gpu_runtime._MANIFEST_NAME).is_file():
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


class UpdateCheckWorker(QObject):
    """起動時の新バージョン確認（update_checker.check_for_update、HTTPS GET 1回）を
    GUI スレッドから外すための薄いワーカー。

    同期で呼ぶと、パケットを黙って落とすファイアウォールや遅い DNS の環境で
    ウィンドウが表示直後に数秒固まる（requests の timeout は接続・読み取りそれぞれに
    かかる）。このアプリの他のネットワーク処理（モデル DL・GPU DL・VLM）と同じく
    QThread で回す。結果は `check_finished(UpdateInfo | None)` で 1 回だけ通知。
    """

    check_finished = Signal(object)   # update_checker.UpdateInfo | None

    def __init__(self, current_version: str):
        super().__init__()
        self._current_version = current_version
        self._stop_event = threading.Event()

    def stop(self):
        # requests.get は途中で中断できないので、これは closeEvent のプロトコル
        # （StoppableWorker）を満たすための印だけ。実際の打ち切りは closeEvent 側の
        # wait(5000) → terminate() に委ねる（他ワーカーと同じ扱い）。
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    @Slot()
    def run_check(self):
        # check_finished は必ず 1 回発火させる（MainWindow 側のスレッド後始末が
        # これを待つ）。check_for_update 自体は例外を外に出さない設計だが、念のため。
        info = None
        try:
            import update_checker
            info = update_checker.check_for_update(self._current_version)
        except Exception as exc:  # noqa: BLE001 - last-resort guard
            write_debug_log(f"UpdateCheckWorker: unexpected {exc!r}")
        if self.is_stopped():
            info = None  # shutting down: never surface a dialog from here
        self.check_finished.emit(info)


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
                if tagging_core.ort is None or tagging_core.np is None or tagging_core.Image is None:
                    # REQ-E11: ライブラリ不足はモデルファイルの問題ではない。Captioner 側と同じく
                    # 環境要因を区別し、数百 MB の再 DL を強いる unverified 化をしない。
                    write_debug_log("TaggerThreadWorker: required libraries are missing; keeping the model's verified flag.")
                    self.log_message.emit(self.get_string("TaggerCore", "Onnxruntime_Not_Installed"), "red")
                else:
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
                # REQ-2.3: 一時ファイル + os.replace で原子的に書く（(iii) 段）。
                atomic_write_text(txt_file_path, new_content, fsync=False, verify=False)
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
