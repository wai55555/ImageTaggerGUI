"""ワーカースレッドの生成・開始・停止・終了待ちと、退避リストの管理。
REQ-6.1 / Phase 6 で main_window.py から切り出した。
`_detached_threads` はモジュール変数のまま（稼働中 QThread の最後の参照を
ウィンドウ破棄で失わないための既存設計を変えない）。"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt

from utils import write_debug_log


def _ui(name: str):
    """Qt 部品は main_window のモジュール属性として遅延解決する（テストの差し替えを保つ）。"""
    return getattr(sys.modules["main_window"], name)




# 停止要求に応じなかったスレッドを、実際に終了するまで保持する置き場。通常は空。
#
# MainWindow のインスタンス属性ではなく**モジュール変数**に置くのが要点。
# ウィンドウが破棄されるとインスタンス属性のリストも一緒に消え、稼働中の QThread に
# 対する最後の Python 参照が失われる。PySide6 は親を持たない QThread を Python 側の
# 所有物として扱うため、そこで C++ オブジェクトが破棄され、Qt の
# 「Deleting a running QThread」経路でプロセスが fail-fast する
# （最小再現で終了コード 0xC0000409 を確認、260922 レビュー指摘）。
# `finished → deleteLater` の配線だけでは、finished より前に参照が消えるこの経路を
# 防げない。
_detached_threads: list = []  # QThread（main_window の import を遅延解決するため型は付けない）


def detached_thread_count() -> int:
    """まだ終了していない退避済みスレッドの数（テスト・終了処理用）。"""
    _prune_detached_threads()
    return len(_detached_threads)


def _prune_detached_threads() -> None:
    """終了済み／既に破棄済みのぶんをリストから外す。"""
    global _detached_threads
    alive: list[QThread] = []
    for thread in _detached_threads:
        try:
            if not thread.isFinished():
                alive.append(thread)
        except RuntimeError:  # 既に C++ 側が破棄済み
            pass
    _detached_threads = alive


def wait_for_detached_threads(timeout_ms: int = 5000) -> bool:
    """退避済みスレッドの終了を待つ。全て終了できたら True。

    終了処理から呼ぶ。待ちきれなくても強制破棄はしない（稼働中の QThread を
    破棄する方がクラッシュとして重い）。プロセス終了まで参照を持ち続けるため、
    ここで False を返してもリストからは外さない。
    """
    _prune_detached_threads()
    if not _detached_threads:
        return True
    deadline = time.monotonic() + max(0.0, timeout_ms / 1000.0)
    for thread in list(_detached_threads):
        try:
            if thread.isFinished():
                continue
            remaining = max(0.0, deadline - time.monotonic())
            thread.quit()
            thread.wait(int(remaining * 1000))
        except RuntimeError:
            continue
    _prune_detached_threads()
    if _detached_threads:
        write_debug_log(
            f"{len(_detached_threads)} detached thread(s) still running at shutdown; "
            "keeping the references alive instead of destroying them")
    return not _detached_threads


def detach_running_thread(thread, worker) -> None:
    """終了しなかったスレッドを、実際に終了した後で破棄するよう繋いで手放す。

    ここで直接 deleteLater() してはいけない。deleteLater() の DeferredDelete は
    「その QObject が所属するスレッド」へ post されるが、_ui("QThread") *オブジェクト* の
    所属は管理対象のワーカースレッドではなく生成元（メインスレッド）なので、
    次にメインのイベントループが回った時点で——ワーカーの終了を待たずに——破棄
    される。Qt のドキュメントどおり、稼働中の _ui("QThread") を破棄すると
    "Deleting a running _ui("QThread") will probably result in a program crash"。
    以前ここには「deleteLater は稼働中でも安全（破棄は終了後まで遅延される）」
    というコメントがあったが、これは誤りだった（260922 PR#27 レビュー指摘）。

    代わりに Qt 公式の後始末イディオム（finished → deleteLater）を使う。
    QThreadPrivate::finish() は running=false / finished=true を立ててから
    finished を emit するので、メインのイベントループが DeferredDelete を
    処理する時点では確実に停止済みになる。

    参照はモジュール変数 `_detached_threads` が持つ。MainWindow の属性に置くと、
    ウィンドウ破棄時にリストごと最後の参照が消えて、まさに避けたかった
    「稼働中の _ui("QThread") の破棄」が起きる（closeEvent は退避済みスレッドを
    列挙していなかった）。終了処理は wait_for_detached_threads() で待つ。

    スレッドが永久に終わらない場合はプロセス終了まで参照が残る（リークだが、
    クラッシュより望ましい）。
    """
    # 既に終了・破棄済みのぶんを掃除する。detach 自体が稀なのでここで十分。
    _prune_detached_threads()
    _detached_threads.append(thread)
    if worker is not None:
        thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)


def wait_for_thread_stop(thread, timeout: float = 5.0) -> bool:
    """tagging スレッドの停止待ちだけを切り出したもの（REQ-2.6）。

    `processEvents()` を挟む以上、待機中に新しい実行が始まりうる。呼び出し側は
    待機対象をローカルに固定し、`self.mw._tagger_thread` が差し替わっていないかを
    戻り値と併せて確認する。素朴な wait(5000) はメインスレッドのイベントループを
    丸ごと止めてしまい、同じ finished シグナルから2本目に繋がっている thread.quit
    や、ワーカースレッド側の後始末が必要とするメインスレッドとのやり取りが処理される
    機会を奪う。ここで小刻みに processEvents() を挟みながら待つことで、その機会を
    与えつつ、合計の待ち時間の上限（既定 5 秒）は変えない。
    """
    deadline = time.monotonic() + timeout
    while thread.isRunning() and time.monotonic() < deadline:
        _ui("QApplication").processEvents()
        thread.wait(50)
    return not thread.isRunning()


def QThread(*args, **kwargs):
    """QThread を main_window のモジュール属性から遅延解決して生成する。

    既存テストが `main_window.QThread` を差し替えて生成を観察しているため、
    切り出し後も同じ差し替えが効くようにする（挙動は変えない）。
    """
    return _ui("QThread")(*args, **kwargs)


class WorkerHost:
    """MainWindow への参照を持ち、切り出した責務だけを担当する（REQ-6.7）。"""

    def __init__(self, mw) -> None:
        self.mw = mw

    def _settings_snapshot(self):
        """REQ-5.4 / R8-3: 実行中の UI 変更が走行中バッチへ混入しないよう、
        開始時点の設定の複製をワーカーへ渡す（次回開始から新しい値が効く）。"""
        return copy.deepcopy(self.mw.settings)

    def _start_tagging_thread(self):
        """Initializes and starts the _ui("TaggerThreadWorker")."""
        if self.mw._is_processing():
            # REQ-2.3: 一括処理中などに開始経路が生きていても、ここで止まる。
            write_debug_log("_start_tagging_thread ignored while processing")
            return
        if self.mw._tagger_thread and self.mw._tagger_thread.isRunning():
            self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Warning_Tagging_Already_Running"), "orange")
            return

        self.mw._cleanup_tagger_thread()

        # 「選択画像のみ」で画像が未選択なら空実行。専用の1行ログを出して即戻る
        # （既存の「画像がありません」を流用しない。260903_vlm-gap-fix.md todo 5）。
        if str(self.mw.settings.behavior.target_mode).upper() == "SELECTED" and self.mw.image_list.currentItem() is None:
            self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Target_No_Selected"), "orange")
            return

        self.mw._session_mode_override = None
        self.mw._progress_last_shown_at = 0.0

        self.mw._update_ui_for_processing(True, 'tagging')

        # Get selected file to prioritize it
        selected_path: Path | None = None
        current_item = self.mw.image_list.currentItem()
        if current_item:
            relative_path = current_item.data(Qt.ItemDataRole.UserRole + 1)
            selected_path = Path(self.mw.settings.paths.input_dir) / relative_path

        # Which worker class to instantiate is the one piece of model_type branching that
        # lives outside ModelModeController: it's a thread/worker-startup decision, not a
        # UI display-state one (design.md 6.8節 "スコープ外の明記").
        # 「VLM接続を使う」が ON ならネットワーク VLM（ローカルモデル不要、tagger でも
        # captioner でも VLM が担当）。OFF なら従来どおりモデル種別で分岐。
        use_vlm = bool(self.mw.settings.vlm.enabled)
        is_captioner = (not use_vlm) and self.mw._current_model_entry().model_type == "captioner"

        # 「失敗のみ」モード用のセッション失敗集合を渡す（他モードでは無視される）。
        failed_paths = list(self.mw._batch_failed_paths)

        self.mw._tagger_thread = QThread()
        snapshot = self.mw._settings_snapshot()
        if use_vlm:
            self.mw._tagger_worker = _ui("VlmCaptionWorker")(
                snapshot, self.mw._make_decision_requester(), self.mw.locale_manager.get_string,
                selected_file_path=selected_path, failed_paths=failed_paths)
            self.mw._tagger_worker.single_test_result.connect(self.mw._on_vlm_single_test_result)
            self.mw._tagger_worker.binding_verified.connect(self.mw._on_vlm_binding_verified)
            self.mw._tagger_thread.started.connect(self.mw._tagger_worker.run_captioning)
        elif is_captioner:
            self.mw._tagger_worker = _ui("CaptionerThreadWorker")(snapshot, self.mw._make_decision_requester(), self.mw.locale_manager.get_string, selected_file_path=selected_path, failed_paths=failed_paths)
            self.mw._tagger_thread.started.connect(self.mw._tagger_worker.run_captioning)
        else:
            self.mw._tagger_worker = _ui("TaggerThreadWorker")(snapshot, self.mw._make_decision_requester(), self.mw.locale_manager.get_string, selected_file_path=selected_path, failed_paths=failed_paths)
            self.mw._tagger_thread.started.connect(self.mw._tagger_worker.run_tagging)
        self.mw._tagger_worker.moveToThread(self.mw._tagger_thread)

        self.mw._tagger_worker.log_message.connect(self.mw.update_log)
        self.mw._tagger_worker.batch_completed.connect(self.mw._on_batch_completed)
        self.mw._tagger_worker.batch_failed.connect(self.mw._on_batch_failed)
        self.mw._tagger_worker.progress_update.connect(self.mw._on_tagging_progress)
        self.mw._tagger_worker.model_status_changed.connect(self.mw._on_model_status_changed)
        self.mw._tagger_worker.finished.connect(self.mw._on_tagger_finished)
        # 正常終了時は worker の finished シグナルから QThread も終了させる。
        self.mw._tagger_worker.finished.connect(self.mw._tagger_thread.quit)

        self.mw._tagger_thread.start()

    def _stop_tagging_thread(self):
        """Requests the tagging thread to stop."""
        if self.mw._tagger_worker:
            self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Stopping_Tagging_Process"), "orange")
            self.mw.run_button.setText(self.mw.locale_manager.get_string("Constants", "Stopping_Process"))
            self.mw.run_button.setEnabled(False)
            self.mw._tagger_worker.stop()
            _ui("QApplication").processEvents() # Ensure UI remains responsive during stop

    def _cleanup_tagger_thread(self):
        """Safely cleans up the existing tagger thread and worker."""
        # 待機中に self.mw._tagger_thread が差し替わりうるので、対象はローカルに固定する
        # （REQ-2.6 / R35-1）。
        thread = self.mw._tagger_thread
        worker = self.mw._tagger_worker
        if thread is None:
            if worker is not None:
                worker.deleteLater()
                self.mw._tagger_worker = None
            return

        if thread.isRunning():
            # worker.finished は _on_tagger_finished（これがこの cleanup を呼ぶ）と
            # thread.quit の両方に connect されており（この順）、この cleanup 自体が
            # その1本目の帰結として、2本目の thread.quit がまだ処理される前に走る
            # ことがある——「異常経路だけ」という想定は正確ではなく、正常完了でも
            # 普通に起こりうる経路。
            write_debug_log("tagger thread still running during cleanup; waiting")
            thread.quit()
            stopped = self.mw._wait_for_tagger_thread_stop(thread)
            if self.mw._tagger_thread is not thread:
                # 待機中に新しい処理が始まり、窓口が新しいスレッドへ差し替わった。
                # 外側の後始末が新しいスレッド・状態に触れてはいけない（R35-1）。
                write_debug_log("cleanup: a new run started during the wait; "
                                "leaving the new thread and worker attached")
                return
            if not stopped:
                write_debug_log("tagger thread did not finish within cleanup timeout; "
                                 "detaching without terminate()")
                self.mw.update_log(self.mw.locale_manager.get_string(
                    "MainWindow", "Thread_Shutdown_Slow"), "orange")
                # QThread.terminate()（Windows では TerminateThread 相当）は危険:
                # HTTPS ソケット処理などネイティブコードの最中に強制終了すると、
                # ヒープ/ローダーロックを保持したまま死に、以後プロセス全体が
                # ハングしうることを実機（オフスクリーン一括テスト）で確認した
                # （2026-09 VLM デバッグ）。無理に殺さず、この参照だけを手放す。
                # スレッド自身は自分のイベントループが処理される限り、いずれ
                # 自然に終了する。
                self.mw._detach_running_thread(thread, worker)
                self.mw._tagger_thread = None
                self.mw._tagger_worker = None
                return
        thread.deleteLater()
        if self.mw._tagger_thread is thread:
            self.mw._tagger_thread = None
        if worker is not None:
            worker.deleteLater()
        if self.mw._tagger_worker is worker:
            self.mw._tagger_worker = None



    def _start_download_thread(self):
        """Initializes and starts the _ui("DownloaderWorker")."""
        if self.mw._download_thread and self.mw._download_thread.isRunning():
            return

        self.mw._is_downloading = True
        self.mw._update_ui_for_processing(True, 'downloading')
        self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Starting_Model_Download"), "black")

        self.mw._download_thread = QThread()
        self.mw._downloader_worker = _ui("DownloaderWorker")(self.mw.locale_manager.get_string, model_id=self.mw.settings.model.model_id)
        self.mw._downloader_worker.moveToThread(self.mw._download_thread)

        self.mw._downloader_worker.log_message.connect(self.mw.update_log)
        self.mw._downloader_worker.progress_update.connect(self.mw._update_download_progress)
        self.mw._downloader_worker.download_finished.connect(self.mw._on_download_finished)
        self.mw._download_thread.started.connect(self.mw._downloader_worker.run_download)
        
        self.mw._download_thread.start()

    def _stop_download_thread(self):
        """Requests the download thread to stop."""
        if self.mw._downloader_worker:
            self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Stopping_Model_Download"), "orange")
            self.mw.run_button.setText(self.mw.locale_manager.get_string("Constants", "Stopping_Process"))
            self.mw.run_button.setEnabled(False)
            self.mw._downloader_worker.stop()

    def _start_bulk_tag_worker(self, mode: str, **kwargs: Any):
        """Initializes and starts the _ui("BulkTagWorker") for add/delete operations."""
        # 錠を先にかける（_set_main_controls_enabled が未保存のキャプションを flush する
        # ので、_is_bulk_deleting を立てるのはその後）。
        # REQ-2.3 / R2-7: 一括処理中も本体の錠をかける。以前は一括用コントロールだけを
        # 無効化していたため、実行ボタン・単体タグ編集・グリッド切替・Undo が生きていた。
        self.mw._set_main_controls_enabled(False)
        # 実行ボタンはタグ付け/DL 中は「停止」として生きているが、一括処理に停止手段は
        # 無い。ここだけは明示的に閉じる（処理中にタグ付けを開始できないようにする）。
        self.mw.run_button.setEnabled(False)
        self.mw._is_bulk_deleting = True
        self.mw.update_log(self.mw.locale_manager.get_string("MainWindow", "Starting_Bulk_Tag_Operation", mode=mode), "blue")

        self.mw._bulk_tag_thread = QThread()
        worker = _ui("BulkTagWorker")(self.mw.locale_manager.get_string)
        self.mw._bulk_tag_worker = worker
        worker.moveToThread(self.mw._bulk_tag_thread)

        worker.log_message.connect(self.mw.update_log)
        worker.finished.connect(self.mw._on_bulk_tag_finished)
        worker.bulk_add_completed.connect(self.mw._on_bulk_add_completed)
        worker.bulk_delete_completed.connect(self.mw._on_bulk_delete_completed)
        
        if mode == 'add':
            self.mw._bulk_tag_thread.started.connect(lambda: worker.run_bulk_add(kwargs['input_dir'], kwargs['tags'], kwargs['prepend']))
        else: # delete
            self.mw._bulk_tag_thread.started.connect(lambda: worker.run_bulk_delete(kwargs['input_dir'], kwargs['tag']))

        self.mw._bulk_tag_thread.start()