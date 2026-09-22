"""APIキー登録ダイアログ（260901_VLM_spec.md 15章 / design.md 6.3節）。

「取得ページを開く → 貼り付け → 自動チェック → 保存して閉じる」を1画面で完結させる。
未ログイン/未サインアップのユーザーも迷わないよう、手順を文章で示す。
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from PySide6.QtCore import Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

import vlm_secrets
from vlm_diagnostics import DiagStatus, is_billing_or_credit_block, item_detail
from vlm_worker import VlmDiagnosticsWorker

GetString = Callable[..., str]


class ApiKeyDialog(QDialog):
    def __init__(self, get_string: GetString, *, display_name: str, secret_ref: str,
                 conn, key_url: str, login_url: str, instructions: str,
                 cloudflare_account_id: str = "",
                 on_cloudflare_verified: Callable[[str], None] | None = None,
                 anthropic_workspace_id: str = "",
                 on_anthropic_workspace_saved: Callable[[str], None] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._t = get_string
        self._secret_ref = secret_ref
        self._conn = conn
        self._key_url = key_url
        self._login_url = login_url
        self._is_cloudflare = getattr(conn, "provider_id", "") == "cloudflare"
        self._is_anthropic = getattr(conn, "provider_id", "") == "anthropic"
        self._cloudflare_account_id = cloudflare_account_id
        self._on_cloudflare_verified = on_cloudflare_verified
        self._anthropic_workspace_id = anthropic_workspace_id
        self._on_anthropic_workspace_saved = on_anthropic_workspace_saved
        self._saved = False
        self._cancel_requested = False
        self._pending_done_result: int | None = None
        self._check_thread: QThread | None = None
        self._check_worker: VlmDiagnosticsWorker | None = None
        self.setWindowTitle(get_string("Vlm", "ApiKey_Title", service=display_name))
        self.setMinimumWidth(460)
        self._build(display_name, instructions)

    def _build(self, display_name: str, instructions: str) -> None:
        root = QVBoxLayout(self)

        head = QLabel(self._t("Vlm", "ApiKey_Header", service=display_name))
        head.setWordWrap(True)
        root.addWidget(head)

        # locale の値に含まれる "\n" リテラルを実改行へ（configparser はエスケープを展開しない）。
        steps = QLabel(instructions.replace("\\n", "\n"))
        steps.setWordWrap(True)
        steps.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(steps)

        link_row = QHBoxLayout()
        open_key = QPushButton(self._t("Vlm", "ApiKey_Open_Key_Page"))
        open_key.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._key_url)))
        link_row.addWidget(open_key)
        if self._login_url and self._login_url != self._key_url:
            open_login = QPushButton(self._t("Vlm", "ApiKey_Open_Login_Page"))
            open_login.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self._login_url)))
            link_row.addWidget(open_login)
        link_row.addStretch(1)
        root.addLayout(link_row)

        url_hint = QLabel(f"{self._key_url}")
        url_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        url_hint.setStyleSheet("color: gray;")
        root.addWidget(url_hint)

        # 既に登録済みなら、それを（値は伏せたまま）明示する。空欄だと未登録に見える。
        where = vlm_secrets.secret_status(self._secret_ref)
        registered = where != "missing"
        where_label = {
            "keyring": self._t("Vlm", "ApiKey_Where_Keyring"),
            "env": self._t("Vlm", "ApiKey_Where_Env"),
            "session": self._t("Vlm", "ApiKey_Where_Session"),
        }.get(where, where)
        if self._is_cloudflare:
            registered_key = "ApiKey_Current_Registered_Cloudflare"
        elif self._is_anthropic:
            registered_key = "ApiKey_Current_Registered_Anthropic"
        else:
            registered_key = "ApiKey_Current_Registered"
        self.current_label = QLabel(
            self._t("Vlm", registered_key, where=where_label) if registered
            else self._t("Vlm", "ApiKey_Current_None"))
        self.current_label.setStyleSheet("color: gray;")
        root.addWidget(self.current_label)

        self.account_id_edit: QLineEdit | None = None
        if self._is_cloudflare:
            account_row = QHBoxLayout()
            account_row.addWidget(QLabel(self._t("Vlm", "Settings_Cf_Account")))
            self.account_id_edit = QLineEdit(self._cloudflare_account_id)
            self.account_id_edit.setPlaceholderText(self._t("Vlm", "ApiKey_Cf_Account_Placeholder"))
            account_row.addWidget(self.account_id_edit, 1)
            root.addLayout(account_row)

        self.workspace_id_edit: QLineEdit | None = None
        if self._is_anthropic:
            workspace_row = QHBoxLayout()
            workspace_row.addWidget(QLabel(self._t("Vlm", "ApiKey_Anthropic_Workspace")))
            self.workspace_id_edit = QLineEdit(self._anthropic_workspace_id)
            self.workspace_id_edit.setPlaceholderText(
                self._t("Vlm", "ApiKey_Anthropic_Workspace_Placeholder"))
            workspace_row.addWidget(self.workspace_id_edit, 1)
            root.addLayout(workspace_row)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText(self._t(
            "Vlm", "ApiKey_Paste_Placeholder_Update" if registered else "ApiKey_Paste_Placeholder"))
        self.key_edit.returnPressed.connect(self._check_and_save)
        root.addWidget(self.key_edit)

        self.save_btn = QPushButton(self._t("Vlm", "ApiKey_Verify_And_Save"))
        self.save_btn.clicked.connect(self._check_and_save)
        root.addWidget(self.save_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        root.addWidget(self.status)

        self.copy_status_btn = QPushButton(self._t("Vlm", "ApiKey_Copy_Details"))
        self.copy_status_btn.clicked.connect(self._copy_status)
        root.addWidget(self.copy_status_btn)

        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        box.rejected.connect(self.reject)
        root.addWidget(box)

    # --- verify -------------------------------------------------------------
    def _check_and_save(self) -> None:
        if self._check_thread is not None:
            return
        entered_key = self.key_edit.text().strip()
        key = entered_key
        if not key and (self._is_cloudflare or self._is_anthropic):
            key = vlm_secrets.get_secret(self._secret_ref) or ""
        if not key:
            self.status.setText(self._t("Vlm", "ApiKey_Empty"))
            return
        check_conn = self._conn
        if self._is_cloudflare:
            account_id = self.account_id_edit.text().strip() if self.account_id_edit else ""
            if not account_id:
                self.status.setText(self._t("Vlm", "ApiKey_Cf_Account_Empty"))
                return
            if re.fullmatch(r"[0-9a-fA-F]{32}", account_id) is None:
                self.status.setText(self._t("Vlm", "ApiKey_Cf_Account_Invalid"))
                return
            self._pending_account_id = account_id
            check_conn = replace(
                self._conn,
                base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            )
        if self._is_anthropic:
            workspace_id = self.workspace_id_edit.text().strip() if self.workspace_id_edit else ""
            if workspace_id and re.fullmatch(r"wrkspc_[A-Za-z0-9]+", workspace_id) is None:
                self.status.setText(self._t("Vlm", "ApiKey_Anthropic_Workspace_Invalid"))
                return
            self._pending_workspace_id = workspace_id
            headers = dict(getattr(self._conn, "request_headers", {}) or {})
            # 入力欄が空でも、ANTHROPIC_WORKSPACE_ID 由来で接続に載っている
            # workspace ヘッダーは残す（消すと複数 workspace のキーで確認が失敗する）。
            if workspace_id:
                headers["anthropic-workspace-id"] = workspace_id
            check_conn = replace(self._conn, request_headers=headers)
        self._set_busy(True)
        self.status.setText(self._t("Vlm", "ApiKey_Checking"))
        self._pending_key = key
        self._pending_key_is_new = bool(entered_key)
        self._check_thread = QThread(self)
        # キー登録時は画像生成POSTを避け、モデル一覧GETだけで認証・到達性を確認する。
        # 料金／推論レート制限に影響する実生成の確認は、設定画面の「接続診断」で行う。
        self._check_worker = VlmDiagnosticsWorker(check_conn, key, lightweight=True)
        self._check_worker.moveToThread(self._check_thread)
        self._check_thread.started.connect(self._check_worker.run)
        self._check_worker.report_ready.connect(self._on_report)
        self._check_worker.finished.connect(self._check_thread.quit)
        self._check_thread.finished.connect(self._on_check_thread_done)
        self._check_thread.start()

    def _diag_detail(self, item) -> str:
        """診断項目の詳細を、利用者へ見せる言語で返す。

        DiagItem.detail は英語のまま保つ（is_billing_or_credit_block() 等の
        文字列判定とデバッグログがそれに依存している）。画面へ出す直前に訳す。
        """
        if item is None:
            return ""
        return item_detail(item, self._t)

    def _on_report(self, report) -> None:
        """キーが使えるかの判定。サーバーが 401/403 で弾いた＝キー不正。到達不能／
        DNS・TLS 失敗＝未成立で保存しない。それ以外はサーバーが応答している＝キーは
        認証を通っているので保存する（モデル ID 違い等は警告どまりで別途直す）。
        登録フローの実リクエストはモデル一覧GETだけで、画像生成は行わない。"""
        self._verify_failed = ""
        self._model_warning = ""
        self._service_warning = ""
        items = {i.name: i for i in report.items}
        auth = items.get("Auth")
        http = items.get("HTTP response")
        extraction = items.get("Caption extraction")
        # HTTPへ到達する前のDNS/TLS失敗や、診断ワーカー内部の例外も汎用文言へ
        # 潰さず表示する。Cloudflareは成功条件が厳しいため、このフォールバックがないと
        # 「キーが受け付けられませんでした」だけになり原因を判別できない。
        first_failure = next(
            (self._diag_detail(i) for i in report.items
             if i.status is DiagStatus.FAIL and getattr(i, "detail", "")), "")

        if self._is_anthropic and http is not None and http.status is not DiagStatus.PASS:
            low = (http.detail or "").lower()
            if "anthropic-workspace-id" in low or "workspace" in low:
                self._verify_failed = self._diag_detail(http) or first_failure \
                    or self._t("Vlm", "ApiKey_Failed_Generic")
                return

        if self._is_cloudflare:
            # Account ID付きの軽量モデル一覧確認、またはフル診断の応答抽出まで通った
            # ときに保存する。キー単体の token/verify 成功では閉じない。
            if (getattr(report, "lightweight", False)
                    and getattr(report, "can_mark_binding_verified", False)):
                return
            if (http is not None and http.status is DiagStatus.PASS
                    and extraction is not None and extraction.status is DiagStatus.PASS):
                return
            self._verify_failed = (
                (self._diag_detail(http)
                 if http is not None and http.status is not DiagStatus.PASS else "")
                or self._diag_detail(extraction)
                or (self._diag_detail(auth)
                    if auth is not None and auth.status is DiagStatus.FAIL else "")
                or first_failure
                or self._t("Vlm", "ApiKey_Failed_Generic")
            )
            return

        if http is not None and http.status is DiagStatus.PASS:
            return   # 200: キーもモデルも通った
        if http is not None and is_billing_or_credit_block(http.detail):
            # Vercelのカード未登録やOpenAIの残高不足など。キー自体は受理されているので
            # 保存し、モデルID不正とは別の請求・利用枠警告を表示する。
            self._service_warning = self._diag_detail(http)
            return
        if (auth is not None and auth.status is DiagStatus.FAIL) or report.http_status in (401, 403):
            self._verify_failed = (self._diag_detail(http)
                                   or self._diag_detail(auth)
                                   or self._t("Vlm", "ApiKey_Failed_Generic"))
            return
        if report.http_status is not None:
            # サーバーが 401/403 以外で応答した＝キーは認証を通っている。
            # 200 でない理由（モデル ID 違い等）は警告どまりで、キーは保存する。
            self._model_warning = (self._diag_detail(http)
                                   or f"HTTP {report.http_status}")
            return
        # 実応答まで到達しなかった（DNS/TLS 失敗・到達不能）→ 保存しない
        for name in ("TLS", "DNS / TCP", "Request build"):
            it = items.get(name)
            if it is not None and it.status is DiagStatus.FAIL:
                self._verify_failed = (self._diag_detail(it)
                                       or self._t("Vlm", "ApiKey_Failed_Generic"))
                return
        self._verify_failed = self._t("Vlm", "ApiKey_Not_Reached")

    def _on_check_thread_done(self) -> None:
        if self._check_worker is not None:
            self._check_worker.deleteLater()
            self._check_worker = None
        if self._check_thread is not None:
            self._check_thread.deleteLater()
            self._check_thread = None
        if self._cancel_requested:
            self._set_busy(False)
            result = self._pending_done_result
            self._pending_done_result = None
            if result is not None:
                QDialog.done(self, result)
            return
        failed = getattr(self, "_verify_failed", self._t("Vlm", "ApiKey_Failed_Generic"))
        if failed:
            self._set_busy(False)
            self.status.setText(self._t("Vlm", "ApiKey_Failed", detail=failed))
            return
        # 成功: 保存して確認メッセージ → 閉じる。persisted は「keyring へ永続保存できたか」。
        persisted = False
        if getattr(self, "_pending_key_is_new", True):
            persisted = vlm_secrets.set_secret(self._secret_ref, self._pending_key,
                                               persist=vlm_secrets.keyring_available())
            where = self._t("Vlm", "ApiKey_Stored_Keyring" if persisted else "ApiKey_Stored_Session")
            success_msg = self._t("Vlm", "ApiKey_Saved_Confirm", where=where)
        else:
            success_msg = self._t("Vlm", "ApiKey_Connected_Existing")
        if self._is_cloudflare and self._on_cloudflare_verified is not None:
            self._on_cloudflare_verified(self._pending_account_id)
        if self._is_anthropic and self._on_anthropic_workspace_saved is not None:
            self._on_anthropic_workspace_saved(getattr(self, "_pending_workspace_id", ""))
        self._saved = True
        service_warn = getattr(self, "_service_warning", "")
        model_warn = getattr(self, "_model_warning", "")
        if service_warn:
            msg = self._t("Vlm", "ApiKey_Service_Warning",
                          success=success_msg, detail=service_warn)
            self._set_busy(False)
            self.status.setText(msg)
            # キー保存後の課金・利用枠警告は確認結果として完了している。長文の
            # 本文を選択／コピーできる警告ダイアログを出してから自動的に閉じる。
            self._show_service_warning(msg)
            self.accept()
            return
        if model_warn:
            where = self._t("Vlm", "ApiKey_Stored_Keyring" if persisted else "ApiKey_Stored_Session")
            msg = self._t("Vlm", "ApiKey_Saved_Model_Warning", where=where, detail=model_warn)
        else:
            msg = success_msg
        QMessageBox.information(self, self.windowTitle(), msg)
        self.accept()

    def _copy_status(self) -> None:
        QGuiApplication.clipboard().setText(self.status.text())

    def _show_service_warning(self, msg: str) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(self.windowTitle())
        box.setText(msg)
        box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        copy_btn = box.addButton(self._t("Vlm", "ApiKey_Copy_Details"),
                                 QMessageBox.ButtonRole.ActionRole)
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(msg))
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()

    def _set_busy(self, busy: bool) -> None:
        self.key_edit.setEnabled(not busy)
        if self.account_id_edit is not None:
            self.account_id_edit.setEnabled(not busy)
        if self.workspace_id_edit is not None:
            self.workspace_id_edit.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)

    def saved(self) -> bool:
        return self._saved

    # --- lifecycle: 検証中の close は GUI を止めず、終了後に完了させる ---
    def _defer_done_until_check_finishes(self, result: int) -> bool:
        th = getattr(self, "_check_thread", None)
        if th is None:
            return False
        # The QThread can already report not-running while its queued finished handler
        # is still waiting in the GUI event loop. Mark cancellation based on object
        # existence so that handler cannot persist a key after Cancel was requested.
        self._cancel_requested = True
        if self._pending_done_result is None:
            self._pending_done_result = result
        try:
            self._check_worker.report_ready.disconnect()
        except (RuntimeError, TypeError, AttributeError):
            pass
        if th.isRunning():
            th.quit()
        self.setEnabled(False)
        return True

    def done(self, r: int) -> None:
        if r != QDialog.DialogCode.Accepted and self._defer_done_until_check_finishes(r):
            return
        QDialog.done(self, r)

    def closeEvent(self, event) -> None:
        if self._defer_done_until_check_finishes(QDialog.DialogCode.Rejected):
            event.ignore()
            return
        super().closeEvent(event)
