"""VLM 一括キャプション生成と1枚テストのバックグラウンド Worker
（260901_VLM_spec.md 10・13章 / design.md 4.9・10章）。

CaptionerThreadWorker とシグナル形状・ライフサイクルを合わせ、MainWindow 側の配線を
対称にする。停止要求はネットワーク要求の前後で確認する。停止途中の生成結果は保存しない。
"""
from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

import constants
from utils import GetString, default_get_string_fallback, write_debug_log
from tagging_core import (
    ExistingFileMode, FileChange, OverwriteDecision,
    get_image_paths_recursive, parse_existing_file_mode,
    parse_target_mode, filter_target_images,
)
import vlm_config
import vlm_persistence
import vlm_secrets
from vlm_errors import VlmAttemptError, VlmErrorReason
from vlm_image import ImagePreprocessConfig, prepare_image
from vlm_profiles import build_system_prompt, build_user_prompt
from vlm_router import (
    ExecutionMode, explain_candidate_failure, localized_candidate_failure,
    select_candidates,
)
from vlm_transport import VlmExecutor


def _error_reason_for_log(error) -> str:
    """ログへ出すエラー名に、設定確認に必要な詳細を付ける。"""
    if error is None:
        return "unknown"
    reason = getattr(getattr(error, "reason", None), "value", "unknown")
    detail = " ".join(str(getattr(error, "message", "") or "").split())
    if not detail:
        return reason
    return f"{reason}: {detail[:600]}"


class VlmDiagnosticsWorker(QObject):
    """接続診断をバックグラウンドで実行する（NFR-002: UI スレッドで通信しない）。"""

    report_ready = Signal(object)   # DiagReport
    finished = Signal()

    def __init__(self, conn, api_key: str | None, *, lightweight: bool = False):
        super().__init__()
        self._conn = conn
        self._api_key = api_key
        self._lightweight = lightweight

    @Slot()
    def run(self) -> None:
        try:
            from vlm_diagnostics import diagnose
            report = diagnose(self._conn, self._api_key, do_live_request=True,
                              lightweight=self._lightweight)
            self.report_ready.emit(report)
        except Exception as e:  # noqa: BLE001
            from vlm_diagnostics import DiagReport, DiagStatus
            detail = f"{type(e).__name__}: {e}"
            write_debug_log(f"vlm diagnostics worker error: {detail}")
            report = DiagReport(connection_id=getattr(self._conn, "connection_id", ""))
            report.add("Internal diagnostic error", DiagStatus.FAIL, detail)
            self.report_ready.emit(report)
        finally:
            self.finished.emit()


class VlmModelListWorker(QObject):
    """プロバイダーの利用可能モデル一覧をバックグラウンドで取得する。"""

    result_ready = Signal(str, object)   # (connection_id, list[ModelCatalogEntry] | VlmAttemptError)
    finished = Signal()

    def __init__(self, conn, api_key: str | None):
        super().__init__()
        self._conn = conn
        self._api_key = api_key

    @Slot()
    def run(self) -> None:
        try:
            from vlm_model_list import fetch_model_catalog
            res = fetch_model_catalog(self._conn, self._api_key)
            self.result_ready.emit(self._conn.connection_id, res)
        except Exception as e:  # noqa: BLE001
            write_debug_log(f"vlm model-list worker error: {e}")
            # None を渡すと呼び出し側が "None" をそのまま表示してしまう。
            # 失敗理由を持つ VlmAttemptError にして actionable な文言を残す。
            err = VlmAttemptError(VlmErrorReason.UNKNOWN, None, f"{type(e).__name__}: {e}")
            self.result_ready.emit(getattr(self._conn, "connection_id", ""), err)
        finally:
            self.finished.emit()


class VlmCaptionWorker(QObject):
    """ネットワーク VLM でキャプションを付ける Worker。"""

    log_message = Signal(str, str)
    model_status_changed = Signal()
    finished = Signal()
    running_state_changed = Signal(bool)
    reload_image_list_signal = Signal()
    batch_completed = Signal(list)          # list[FileChange] -> MainWindow が Undo にまとめる
    batch_failed = Signal(list)             # list[Path] 入力画像パス。FAILED 再実行の failed_paths に対応
    progress_update = Signal(int, int)      # (done, total)
    single_test_result = Signal(str, str, str)  # (caption, connection_display, model_id) 保存はしない
    binding_verified = Signal(str, str)     # (provider_id, profile_id) 実出力を確認できた -> MainWindow が永続化

    def __init__(self, settings, decision_requester=None, get_string: GetString | None = None,
                 selected_file_path: Path | None = None, single_test: bool = False,
                 failed_paths: Sequence[Path] = ()):
        super().__init__()
        self._settings = settings
        self._decision_requester = decision_requester
        self._selected_file_path = selected_file_path
        self._single_test = single_test
        self._failed_paths = tuple(failed_paths)
        self._stop_event = threading.Event()
        self.get_string: GetString = get_string or default_get_string_fallback

    # --- 停止 ---
    def stop(self) -> None:
        write_debug_log(f"DEBUG: {type(self).__name__}.stop() called.")
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    # --- エントリポイント（CaptionerThreadWorker と同名） ---
    @Slot()
    def run_captioning(self) -> None:
        if self._single_test:
            self._run_single_test()
        else:
            self._run_batch()

    # --- 共通セットアップ ---
    def _build_runtime(self):
        vlm = self._settings.vlm
        gen_profile = vlm_config.build_generation_profile(vlm)
        policy = vlm_config.build_router_policy(vlm)
        model_profile = vlm_config.resolve_model_profile(vlm)
        if model_profile is None:
            self.log_message.emit(self.get_string("Vlm", "Error_Model_Profile_Unknown",
                                                  profile=vlm.model_profile_id), "red")
            return None
        # 内蔵接続の model_id は選択プロファイルの binding（＋ override）から埋める。
        connections = vlm_config.build_connection_map(vlm, model_profile)

        # builtin_fallback の順序＝有効集合は config の connection_order に従う。
        # connection_order から外された provider は候補から完全に除外する（足し戻さない）。
        if policy.execution_mode is ExecutionMode.BUILTIN_FALLBACK:
            order = vlm_config.ordered_builtin_provider_ids(vlm, model_profile)
            # 設定画面でチェックできた「binding は無いが検証済み override がある」
            # 経路も候補に含める。素の binding だけで絞ると、利用者がモデル一覧から
            # 実在IDを選んでチェックした経路が一度も試されない（260922 PR#27）。
            effective = vlm_config.profile_with_override_bindings(vlm, model_profile)
            ordered_bindings = {pid: effective.bindings[pid]
                                for pid in order if pid in effective.bindings}
            model_profile = dataclasses.replace(model_profile, bindings=ordered_bindings)

        has_auth = {}
        supports_image = {}
        for cid, conn in connections.items():
            has_auth[cid] = conn.auth.type == "none" or bool(vlm_secrets.get_secret(conn.auth.secret_ref))
            supports_image[cid] = True

        candidates = select_candidates(model_profile, connections, policy,
                                       has_auth=has_auth, supports_image=supports_image)
        executor = VlmExecutor(connections, vlm_secrets.get_secret, stop_checker=self.is_stopped,
                               on_attempt_start=self._on_attempt_start)
        image_cfg = ImagePreprocessConfig(
            max_long_edge=gen_profile.image_max_long_edge,
            fmt=gen_profile.image_format, jpeg_quality=gen_profile.image_jpeg_quality)
        if policy.execution_mode is ExecutionMode.CUSTOM_SINGLE:
            custom_conn = connections.get(policy.selected_connection_id or "")
            if custom_conn is not None and custom_conn.image_max_long_edge:
                image_cfg = dataclasses.replace(
                    image_cfg, max_long_edge=custom_conn.image_max_long_edge)
        return {
            "connections": connections, "gen_profile": gen_profile, "policy": policy,
            "candidates": candidates, "executor": executor,
            "system_prompt": build_system_prompt(gen_profile),
            "user_prompt": build_user_prompt(gen_profile),
            "image_cfg": image_cfg,
        }

    def _on_attempt_start(self, conn, attempt: int) -> None:
        """VlmExecutor が接続を試行する直前に呼ぶ通知。UI へ「今何をしているか」を出す。

        write_debug_log() は [Debug] debug_log がオフだと何も残さないので、これは
        （Debug 設定に関係なく常に見える）log_message 経由にする。1リクエストが
        最大 read_timeout_s（既定60秒）かかる上、同一接続で再試行もするため、これが
        無いと成功/失敗が返るまで画面が完全に無反応に見える。
        """
        if attempt > 1:
            self.log_message.emit(self.get_string(
                "Vlm", "Attempt_Retry", connection=conn.display_name, attempt=attempt), "blue")
        else:
            self.log_message.emit(self.get_string(
                "Vlm", "Attempt_Start", connection=conn.display_name), "blue")

    def _spec_base_for(self, image_path: Path, rt) -> dict | None:
        try:
            prepared = prepare_image(image_path, rt["image_cfg"])
        except Exception as e:  # noqa: BLE001
            write_debug_log(f"vlm: image prepare failed for {image_path.name}: {e}")
            return None
        return {
            "image": prepared, "profile": rt["gen_profile"],
            "system_prompt": rt["system_prompt"], "user_prompt": rt["user_prompt"],
        }

    # --- 1枚テスト（結果を編集欄へ表示し、そのまま .txt へも保存する） ---
    # 旧仕様は自動保存しなかったが、テスト後に画像リストと .txt を再読み込みするため
    # 編集欄には一瞬しか出ず実質使えなかった。既存ファイルの扱い・挿入位置は一括処理と
    # 同じ設定に従い、Undo 1件として積む（2026-09 ユーザー決定）。
    def _run_single_test(self) -> None:
        self.running_state_changed.emit(True)
        changed: list[FileChange] = []
        try:
            rt = self._build_runtime()
            if rt is None:
                return
            if not rt["candidates"].has_candidates:
                # 画面へは訳した理由を出し、英語の原文はデバッグログへ残す。
                write_debug_log("vlm: " + explain_candidate_failure(
                    rt["candidates"].rejected_reason, rt["candidates"].excluded))
                self.log_message.emit(self.get_string("Vlm", "Error_No_Candidate",
                                                      reason=localized_candidate_failure(
                                                          rt["candidates"].rejected_reason,
                                                          rt["candidates"].excluded,
                                                          self.get_string)), "red")
                return
            image_path = self._selected_file_path
            if image_path is None or not Path(image_path).is_file():
                self.log_message.emit(self.get_string("Vlm", "Error_No_Selected_Image"), "red")
                return
            image_path = Path(image_path)
            spec_base = self._spec_base_for(image_path, rt)
            if spec_base is None:
                self.log_message.emit(self.get_string("Vlm", "Error_Image_Prepare_Failed",
                                                      name=image_path.name), "red")
                return
            result = rt["executor"].caption_one(spec_base, rt["candidates"].connection_ids)
            self._emit_attempt_summary(image_path.name, result)
            # _emit_attempt_summary already reports executor-observed cancellation.
            if result.stopped:
                return
            if self.is_stopped():
                self.log_message.emit(self.get_string("Vlm", "Stopped_By_User"), "orange")
                return
            if not result.ok:
                return

            conn = rt["connections"].get(result.connection_id)
            self.single_test_result.emit(result.text or "",
                                         conn.display_name if conn else result.connection_id,
                                         result.model_id)
            self._note_verified(conn)

            output_path = image_path.with_suffix(".txt")
            mode = parse_existing_file_mode(self._settings.behavior.existing_file_mode, self.get_string)
            will_write, decision = self._resolve_existing(output_path, mode)
            # ASK ダイアログ表示中に停止された場合、その結果を書き出さない。
            if self.is_stopped():
                self.log_message.emit(self.get_string("Vlm", "Stopped_By_User"), "orange")
                return
            if not will_write:
                self.log_message.emit(self.get_string("Vlm", "Single_Test_Skipped_Existing",
                                                      name=output_path.name), "orange")
                return
            eff_placement = ("OVERWRITE" if decision is OverwriteDecision.OVERWRITE
                             else self._settings.caption.placement)
            if decision is OverwriteDecision.APPEND and eff_placement == "OVERWRITE":
                eff_placement = "APPEND"
            try:
                outcome = vlm_persistence.save_caption(output_path, result.text or "", eff_placement)
            except Exception as e:  # noqa: BLE001 - 読めない既存等
                write_debug_log(f"vlm single test save failed for {output_path.name}: {type(e).__name__}: {e}")
                self.log_message.emit(self.get_string("Vlm", "Save_Failed", name=output_path.name), "red")
                return
            if outcome.written:
                changed.append(FileChange(
                    path=output_path, previous_content=outcome.previous_content,
                    new_content=outcome.new_content, was_append=False, added_tags=()))
        except Exception as e:  # noqa: BLE001
            import traceback
            self.log_message.emit(self.get_string("Vlm", "Error_Fatal", e=str(e)), "red")
            write_debug_log(f"vlm single test fatal: {traceback.format_exc()}")
        finally:
            self.batch_completed.emit(changed)
            self.running_state_changed.emit(False)
            self.reload_image_list_signal.emit()
            self.finished.emit()

    # --- 一括処理 ---
    def _run_batch(self) -> None:
        self.running_state_changed.emit(True)
        changed: list[FileChange] = []
        failed: list[Path] = []
        try:
            rt = self._build_runtime()
            if rt is None:
                return
            candidates = rt["candidates"]
            if not candidates.has_candidates:
                write_debug_log("vlm: " + explain_candidate_failure(
                    candidates.rejected_reason, candidates.excluded))
                self.log_message.emit(self.get_string("Vlm", "Error_No_Candidate",
                                                      reason=localized_candidate_failure(
                                                          candidates.rejected_reason,
                                                          candidates.excluded,
                                                          self.get_string)), "red")
                for cid, why in candidates.excluded.items():
                    write_debug_log(f"vlm: candidate excluded {cid}: {why}")
                return

            input_dir = Path(self._settings.paths.input_dir)
            image_paths = get_image_paths_recursive(input_dir)
            target_mode = parse_target_mode(getattr(self._settings.behavior, "target_mode", "ALL"), self.get_string)
            image_paths = filter_target_images(image_paths, target_mode, failed_paths=self._failed_paths, selected=self._selected_file_path)
            if not image_paths:
                self.log_message.emit(self.get_string("Vlm", "Warn_No_Images", dir=str(input_dir)), "orange")
                return

            total = len(image_paths)
            mode = parse_existing_file_mode(self._settings.behavior.existing_file_mode, self.get_string)
            placement = self._settings.caption.placement
            step = constants.progress_step_for(total)
            n_written = n_skipped = n_errors = n_unchanged = 0
            last_conn = ""

            self.log_message.emit(self.get_string("Vlm", "Batch_Start", count=total), "blue")
            executor: VlmExecutor = rt["executor"]

            def remaining_failures(start: int) -> list[Path]:
                """Return unprocessed inputs that would still require VLM work."""
                pending = image_paths[start:]
                if mode is ExistingFileMode.SKIP:
                    return [p for p in pending if not p.with_suffix(".txt").is_file()]
                return list(pending)

            exhausted_logged = False
            for i, image_path in enumerate(image_paths):
                if self.is_stopped():
                    self.log_message.emit(self.get_string("Vlm", "Stopped_By_User"), "orange")
                    failed.extend(remaining_failures(i))
                    break
                if (i + 1) % step == 0 or i == total - 1:
                    self.progress_update.emit(i + 1, total)

                output_path = image_path.with_suffix(".txt")
                will_write, decision = self._resolve_existing(output_path, mode)
                if not will_write:
                    n_skipped += 1
                    continue

                # 既存出力をSKIPする画像は、接続が尽きていても失敗では
                # ない。それを解決した後、実際に生成が必要な画像でだけ打ち切る。
                if not executor.live_candidates(candidates.connection_ids):
                    # break で残り全部を一括で見捨てない: ここでの「候補が尽きた」は
                    # 恒久的な EXCLUDE だけでなく、レート制限のような時間で解ける
                    # cooldown が原因のこともある。この画像だけ失敗にして次へ進めば、
                    # 後続画像の処理に要する実時間の分だけ cooldown が明けている
                    # 可能性があり、バッチ全体を無駄に早期終了させずに済む
                    # （2026-09 VLM デバッグ: 実機11枚バッチで、レート制限中の接続と
                    # 誤って除外された接続が重なり、break のせいで4枚目以降7枚が
                    # 一度も試行されずに打ち切られたのを確認）。ただしメッセージは
                    # 大量画像で毎回スパムしないよう、バッチにつき1回だけ出す
                    # （PR時からの既存仕様: image_failed の乱発を避ける）。
                    n_errors += 1
                    failed.append(image_path)
                    if not exhausted_logged:
                        self.log_message.emit(self.get_string("Vlm", "All_Connections_Exhausted"), "red")
                        exhausted_logged = True
                    continue
                eff_placement = "OVERWRITE" if decision is OverwriteDecision.OVERWRITE else placement
                # 「常に追記」を選んでいるのに placement が既定の OVERWRITE のままだと
                # 既存キャプションを丸ごと捨ててしまう（PR#16 の caption_core 修正と同方針）。
                if decision is OverwriteDecision.APPEND and eff_placement == "OVERWRITE":
                    eff_placement = "APPEND"

                spec_base = self._spec_base_for(image_path, rt)
                if spec_base is None:
                    n_errors += 1
                    failed.append(image_path)
                    self.log_message.emit(self.get_string("Vlm", "Image_Error",
                                                          name=image_path.name), "red")
                    continue

                result = executor.caption_one(spec_base, candidates.connection_ids)
                if self.is_stopped() or result.stopped:
                    self.log_message.emit(self.get_string("Vlm", "Stopped_By_User"), "orange")
                    failed.extend(remaining_failures(i))
                    break
                if result.stop_job:
                    reason = _error_reason_for_log(result.error) if result.error else "prompt_format_error"
                    self.log_message.emit(self.get_string("Vlm", "Job_Stopped", reason=reason), "red")
                    failed.extend(remaining_failures(i))
                    break
                if not result.ok:
                    n_errors += 1
                    failed.append(image_path)
                    reason = _error_reason_for_log(result.error)
                    self.log_message.emit(self.get_string("Vlm", "Image_Failed",
                                                          name=image_path.name, reason=reason), "red")
                    continue
                self._note_verified(rt["connections"].get(result.connection_id))
                if result.connection_id and result.connection_id != last_conn:
                    if last_conn:
                        conn = rt["connections"].get(result.connection_id)
                        self.log_message.emit(self.get_string(
                            "Vlm", "Switched_Connection",
                            to=conn.display_name if conn else result.connection_id), "orange")
                    last_conn = result.connection_id

                try:
                    outcome = vlm_persistence.save_caption(output_path, result.text or "", eff_placement)
                except Exception as e:  # noqa: BLE001 - 読めない既存等
                    n_errors += 1
                    failed.append(image_path)
                    write_debug_log(f"vlm: save failed for {output_path.name}: {type(e).__name__}: {e}")
                    self.log_message.emit(self.get_string("Vlm", "Save_Failed", name=output_path.name), "red")
                    continue

                if not outcome.written:
                    n_unchanged += 1
                    continue
                n_written += 1
                # VLM の保存は常に .txt 全文の書き換え。Undo は全文スナップショットで
                # 統一（was_append=False → OverwriteFileAction。「0件のタグを追記」の
                # ような不自然なラベルを避ける）。
                changed.append(FileChange(
                    path=output_path, previous_content=outcome.previous_content,
                    new_content=outcome.new_content, was_append=False, added_tags=()))

            self.log_message.emit(self.get_string(
                "Vlm", "Batch_Summary", total=total, written=n_written, skipped=n_skipped,
                unchanged=n_unchanged, errors=n_errors), "blue")
        except Exception as e:  # noqa: BLE001
            import traceback
            self.log_message.emit(self.get_string("Vlm", "Error_Fatal", e=str(e)), "red")
            write_debug_log(f"vlm batch fatal: {traceback.format_exc()}")
        finally:
            # 例外で途中終了しても、それまでに実書き込みしたファイルは Undo 対象にする。
            # 失敗画像パスも同様に部分結果で出す（成功時は空）。単体テスト側では出さない。
            self.batch_completed.emit(changed)
            self.batch_failed.emit(failed)
            self.running_state_changed.emit(False)
            self.reload_image_list_signal.emit()
            self.finished.emit()

    # --- 既存ファイル判定（captioner と同じ考え方） ---
    def _resolve_existing(self, output_path: Path, mode: ExistingFileMode) -> tuple[bool, OverwriteDecision | None]:
        if not output_path.is_file():
            return True, None
        if mode is ExistingFileMode.SKIP:
            return False, OverwriteDecision.SKIP
        if mode is ExistingFileMode.OVERWRITE:
            return True, OverwriteDecision.OVERWRITE
        if mode is ExistingFileMode.APPEND:
            return True, OverwriteDecision.APPEND
        # ASK
        if self.is_stopped():
            return False, OverwriteDecision.SKIP
        if self._decision_requester is None:
            return False, OverwriteDecision.SKIP
        decision = self._decision_requester(output_path)
        if decision is OverwriteDecision.SKIP:
            return False, decision
        return True, decision

    def _note_verified(self, conn) -> None:
        """実出力を確認できた内蔵 binding を1度だけ MainWindow へ知らせる（永続化はあちら）。"""
        if conn is None or getattr(conn, "is_custom", True) or not getattr(conn, "provider_id", ""):
            return
        seen = getattr(self, "_verified_emitted", None)
        if seen is None:
            seen = self._verified_emitted = set()
        if conn.provider_id in seen:
            return
        seen.add(conn.provider_id)
        self.binding_verified.emit(conn.provider_id, self._settings.vlm.model_profile_id)

    def _emit_attempt_summary(self, name: str, result) -> None:
        for a in result.attempts:
            write_debug_log(f"vlm test {name}: {a.connection_id} -> {a.error_reason} ({a.error_class})")
        if result.ok:
            self.log_message.emit(self.get_string("Vlm", "Test_Success",
                                                  conn=result.connection_id, model=result.model_id), "green")
        elif result.stopped:
            self.log_message.emit(self.get_string("Vlm", "Stopped_By_User"), "orange")
        else:
            reason = _error_reason_for_log(result.error)
            self.log_message.emit(self.get_string("Vlm", "Test_Failed", reason=reason), "red")
