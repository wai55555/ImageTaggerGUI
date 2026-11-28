"""起動時のプロンプト調整（GPU 導入提案と更新チェック）。REQ-6.2 / Phase 6 で
main_window.py から切り出した。判定と状態管理だけを持ち、ダイアログの表示は
MainWindow 側の薄い委譲（@Slot）を経由する（挙動は変えない）。"""
from __future__ import annotations



from PySide6.QtCore import Qt



from datetime import datetime
from datetime import timedelta
from datetime import timezone
from utils import write_debug_log
from workers import GpuRuntimeDownloadWorker
from workers import UpdateCheckWorker
import constants
import webbrowser


def _ui(name: str):
    """Qt の部品は main_window のモジュール属性として解決する。

    切り出し前は `main_window.QMessageBox` 等をテストが差し替えていた（実モーダルで
    ハングしないための定石）。切り出し後も同じ差し替えが効くよう、import 束縛では
    なくモジュール属性を遅延解決する（挙動は変えない）。
    """
    import sys

    return getattr(sys.modules["main_window"], name)


class StartupPrompts:
    """MainWindow への参照を持ち、切り出した責務だけを担当する（REQ-6.7）。"""

    def __init__(self, mw) -> None:
        self.mw = mw

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
        beh = self.mw.settings.behavior
        if beh.onnx_device == "cpu":
            return
        if self.mw._gpu_dl_thread and self.mw._gpu_dl_thread.isRunning():
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
                    else self.mw.locale_manager.get_string("Gpu", "Prompt_Size_Unknown"))
        box = _ui("QMessageBox")(self.mw)
        box.setIcon(_ui("QMessageBox").Icon.Question)
        box.setWindowTitle(self.mw.locale_manager.get_string("Gpu", "Prompt_Title"))
        box.setText(self.mw.locale_manager.get_string("Gpu", "Prompt_Body", size=size_txt))
        dl_btn = box.addButton(self.mw.locale_manager.get_string("Gpu", "Prompt_Download"),
                               _ui("QMessageBox").ButtonRole.AcceptRole)
        box.addButton(self.mw.locale_manager.get_string("Gpu", "Prompt_Later"),
                      _ui("QMessageBox").ButtonRole.RejectRole)
        never_btn = box.addButton(self.mw.locale_manager.get_string("Gpu", "Prompt_Never"),
                                  _ui("QMessageBox").ButtonRole.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is dl_btn:
            self.mw._start_gpu_runtime_download()
        elif clicked is never_btn:
            beh.gpu_setup_prompt = "dismissed"
            self.mw.save_current_config()
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
                if onnx_providers.gpu_runtime_dir().exists():
                    # uninstall() swallows per-file OSError (locked/permission-denied
                    # files) so gpu_runtime/ can survive it; the repair prompt will
                    # then keep reappearing despite Never (cubic review, PR #21).
                    # There's no unconditional way to force-remove a locked file, so
                    # this is diagnostic only - the existing `onnx_device = cpu`
                    # config escape hatch still stops the prompt outright.
                    write_debug_log("_maybe_prompt_gpu_setup: gpu_runtime/ still present after "
                                    "uninstall() - the repair prompt may reappear next launch")

    def _maybe_prompt_update_check(self):
        """起動時に新バージョンの有無を確認し、あれば通知する（通知型のみ。ダウンロード・
        自己更新は一切しない - docs/260917_updater_av_safety_design.md）。

        3択: ダウンロードページを開く / 今はしない（次回また訊く）/ このバージョンを
        スキップ（次に新しい版が出るまで確認しない）。

        起動のたびに通信しないよう、前回チェックから24時間経っていなければ何もしない。
        **チェックを試みた時点で（結果の成否に関わらず）update_check_last を更新する**
        ので、ネットワーク不通でも次の24時間は再試行しない（design doc 6章、失敗を
        無限リトライしない）。

        ここではガード（dismissed / 24h / 実行中）と update_check_last の保存だけを
        して、通信は UpdateCheckWorker（_ui("QThread")）に投げる。結果は
        `_on_update_check_finished` → `_show_update_prompt` で GUI スレッドに戻って
        表示する。`initial_load()` で `_maybe_prompt_gpu_setup()` の後に呼ばれる。
        """
        beh = self.mw.settings.behavior
        if beh.update_check == "dismissed":
            return
        if self.mw._update_check_thread is not None:
            # Still running, or finished but its check_finished slot hasn't run yet
            # (which is what clears this) - either way a second thread must not be
            # started on top of it.
            return
        if beh.update_check_last:
            try:
                last = datetime.fromisoformat(beh.update_check_last)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                # A timestamp in the future (clock set back, hand-edited value) would
                # otherwise suppress the check until the clock catches up - treat it
                # as "unknown" instead so it just gets overwritten below.
                if last <= now and now - last < timedelta(hours=24):
                    return
            except ValueError:
                pass  # 壊れた値は「未確認」扱いで続行（このまま下で上書きされる）

        # 成否に関わらずここで更新する（design doc 6章の意図通り）。
        beh.update_check_last = datetime.now(timezone.utc).isoformat()
        self.mw.save_current_config()

        # The HTTPS GET runs off the GUI thread: done synchronously here it would
        # freeze the just-shown window for up to the connect+read timeouts on a
        # network that silently drops packets (every other network call in this
        # app is on a _ui("QThread") for the same reason). The dialog is shown from
        # _on_update_check_finished, back on the GUI thread.
        self.mw._update_check_thread = _ui("QThread")()
        self.mw._update_check_worker = UpdateCheckWorker(constants.APP_VERSION)
        self.mw._update_check_worker.moveToThread(self.mw._update_check_thread)
        self.mw._update_check_worker.check_finished.connect(self.mw._on_update_check_finished)
        self.mw._update_check_thread.started.connect(self.mw._update_check_worker.run_check)
        self.mw._update_check_thread.start()

    def _on_update_check_finished(self, info):
        """UpdateCheckWorker の結果を受けて、必要ならダイアログを出す（GUI スレッド）。

        `info` は update_checker.UpdateInfo か None（新版なし／取得失敗、区別しない）。
        """
        if self.mw._update_check_thread:
            self.mw._update_check_thread.quit()
            self.mw._update_check_thread.wait()
            if self.mw._update_check_worker:
                self.mw._update_check_worker.deleteLater()
            self.mw._update_check_thread.deleteLater()
            self.mw._update_check_thread = self.mw._update_check_worker = None
        if self.mw._is_shutting_down or info is None:
            return
        self.mw._show_update_prompt(info)

    def _show_update_prompt(self, info):
        """新版 `info` を 3 択ダイアログで提示する。別のモーダル（GPU 導入プロンプトや
        その進捗ダイアログ等）が開いている間は重ねずに待ち、閉じてから出す。"""
        if self.mw._is_shutting_down:
            return
        if _ui("QApplication").activeModalWidget() is not None:
            # Never stack a second modal on top of one that's already open (the GPU
            # prompt runs a nested exec() loop, and a slot firing inside it would
            # otherwise open this dialog right on top of it). Poll until it's gone.
            _ui("QTimer").singleShot(1000, lambda: self.mw._show_update_prompt(info))
            return
        beh = self.mw.settings.behavior
        if beh.update_skip_version:
            import update_checker
            skip = update_checker.parse_version(beh.update_skip_version)
            latest = update_checker.parse_version(info.version)
            if skip is not None and latest is not None and latest <= skip:
                return  # このバージョン（以下）はユーザーが既にスキップ済み

        # Tags in this repo are "vX.Y.Z"; show the running version the same way so
        # the two numbers in the dialog read consistently.
        current_txt = constants.APP_VERSION
        if info.version[:1] in ("v", "V") and current_txt[:1] not in ("v", "V"):
            current_txt = "v" + current_txt
        box = _ui("QMessageBox")(self.mw)
        box.setIcon(_ui("QMessageBox").Icon.Information)
        box.setWindowTitle(self.mw.locale_manager.get_string("Updater", "Prompt_Title"))
        box.setText(self.mw.locale_manager.get_string("Updater", "Prompt_Body",
                                                    version=info.version, current=current_txt))
        open_btn = box.addButton(self.mw.locale_manager.get_string("Updater", "Prompt_Open"),
                                 _ui("QMessageBox").ButtonRole.AcceptRole)
        box.addButton(self.mw.locale_manager.get_string("Updater", "Prompt_Later"),
                      _ui("QMessageBox").ButtonRole.RejectRole)
        skip_btn = box.addButton(self.mw.locale_manager.get_string("Updater", "Prompt_Skip"),
                                 _ui("QMessageBox").ButtonRole.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is open_btn:
            webbrowser.open(info.html_url)
        elif clicked is skip_btn:
            beh.update_skip_version = info.version
            self.mw.save_current_config()

    def _start_gpu_runtime_download(self):
        if self.mw._gpu_dl_thread and self.mw._gpu_dl_thread.isRunning():
            return
        self.mw.update_log(self.mw.locale_manager.get_string("Gpu", "Download_Start"), "black")

        progress = _ui("QProgressDialog")(
            self.mw.locale_manager.get_string("Gpu", "Download_Progress_Label"),
            self.mw.locale_manager.get_string("Gpu", "Download_Cancel"), 0, 100, self.mw)
        progress.setWindowTitle(self.mw.locale_manager.get_string("Gpu", "Prompt_Title"))
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(0)
        progress.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        progress.canceled.connect(self.mw._cancel_gpu_runtime_download)
        # WA_DeleteOnClose means the underlying C++ object can be destroyed by the
        # user clicking the dialog's own close button (not just our Cancel/close()
        # calls) at any time. `destroyed` fires synchronously right when that
        # happens, so this is the one place that reliably keeps our reference in
        # sync - without it, a later access (from the async finished handler) can
        # hit an already-deleted PySide6 wrapper and raise RuntimeError
        # (cubic review, PR #21).
        progress.destroyed.connect(self.mw._on_gpu_dl_progress_destroyed)
        self.mw._gpu_dl_progress = progress

        self.mw._gpu_dl_thread = _ui("QThread")()
        self.mw._gpu_dl_worker = GpuRuntimeDownloadWorker(self.mw.locale_manager.get_string)
        self.mw._gpu_dl_worker.moveToThread(self.mw._gpu_dl_thread)
        self.mw._gpu_dl_worker.log_message.connect(self.mw.update_log)
        self.mw._gpu_dl_worker.progress_update.connect(self.mw._on_gpu_runtime_progress)
        self.mw._gpu_dl_worker.download_finished.connect(self.mw._on_gpu_runtime_finished)
        self.mw._gpu_dl_thread.started.connect(self.mw._gpu_dl_worker.run_download)
        self.mw._gpu_dl_thread.start()
        progress.show()

    def _cancel_gpu_runtime_download(self):
        if self.mw._gpu_dl_worker:
            self.mw.update_log(self.mw.locale_manager.get_string("Gpu", "Download_Cancelling"), "orange")
            self.mw._gpu_dl_worker.stop()

    def _on_gpu_dl_progress_destroyed(self):
        self.mw._gpu_dl_progress = None

    def _on_gpu_runtime_progress(self, percent: int, done_mb: float, total_mb: float):
        if not self.mw._gpu_dl_progress:
            return
        if total_mb > 0:
            self.mw._gpu_dl_progress.setRange(0, 100)
            self.mw._gpu_dl_progress.setValue(percent)
            self.mw._gpu_dl_progress.setLabelText(
                self.mw.locale_manager.get_string("Gpu", "Download_Progress_Detail",
                                               done=f"{done_mb:.0f}", total=f"{total_mb:.0f}"))
        else:
            self.mw._gpu_dl_progress.setRange(0, 0)  # indeterminate

    def _on_gpu_runtime_finished(self, ok: bool):
        if self.mw._is_shutting_down:
            return  # closeEvent stops/joins the thread itself
        # Capture before the worker is torn down below: install() returns False for
        # both cancellation and genuine failure, collapsing them into one bool.
        # Without this, cancelling a download pops the scary "Download failed"
        # dialog even though the user asked for exactly this (CodeRabbit, PR #21).
        # `not ok` guards a narrow race (cubic review, PR #21): install() can
        # finish successfully right as the user clicks Cancel, so is_stopped() may
        # be true even though ok is also true - that must still show the success
        # dialog, not be silently swallowed as "cancelled".
        cancelled = bool(not ok and self.mw._gpu_dl_worker and self.mw._gpu_dl_worker.is_stopped())
        if self.mw._gpu_dl_progress:
            # close() counts as a cancel for _ui("QProgressDialog") and would re-fire
            # canceled -> _cancel_gpu_runtime_download; drop the connection first.
            # Both calls are guarded: _on_gpu_dl_progress_destroyed (connected to
            # `destroyed`) is the normal way this reference gets cleared, but a
            # user-initiated close (WA_DeleteOnClose) can race ahead of us.
            try:
                self.mw._gpu_dl_progress.canceled.disconnect(self.mw._cancel_gpu_runtime_download)
                self.mw._gpu_dl_progress.close()  # WA_DeleteOnClose frees it
            except (RuntimeError, TypeError):
                pass  # already destroyed (e.g. user closed the dialog directly)
            self.mw._gpu_dl_progress = None
        if self.mw._gpu_dl_thread:
            self.mw._gpu_dl_thread.quit()
            self.mw._gpu_dl_thread.wait()
            if self.mw._gpu_dl_worker:
                self.mw._gpu_dl_worker.deleteLater()
            self.mw._gpu_dl_thread.deleteLater()
            self.mw._gpu_dl_thread = self.mw._gpu_dl_worker = None
        if cancelled:
            return  # already logged via Download_Cancelling / Worker_Stopped
        title = self.mw.locale_manager.get_string("Gpu", "Prompt_Title")
        if ok:
            _ui("QMessageBox").information(self.mw, title,
                                   self.mw.locale_manager.get_string("Gpu", "Download_Done_Restart"))
        else:
            _ui("QMessageBox").warning(self.mw, title,
                                self.mw.locale_manager.get_string("Gpu", "Download_Failed"))
