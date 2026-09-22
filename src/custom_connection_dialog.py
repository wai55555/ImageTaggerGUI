"""カスタム接続編集ダイアログ（260901_VLM_design.md 6.4節 / implement_plan 8章）。

外部・ローカルを含む任意の VLM 接続を1件編集する。秘密値はこのダイアログでは
vlm_secrets 経由でのみ扱い、返す dict には含めない。
"""
from __future__ import annotations

from typing import Callable
from urllib.parse import urlparse

from PySide6.QtCore import QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

import vlm_secrets
from vlm_connections import ConnectionKind, ConnectionLocality, VlmConnection, resolve_custom_kind
from vlm_config import new_connection_id
from vlm_model_list import ModelCatalogEntry, catalog_entry_from_id, filter_vlm_catalog
from vlm_worker import VlmModelListWorker

GetString = Callable[..., str]

_PROTOCOLS = [
    ("openai_chat_completions", "OpenAI Chat Completions"),
    ("openai_responses", "OpenAI Responses API"),
    ("anthropic_messages", "Anthropic Messages API"),
    ("gemini_generate_content", "Google Gemini generateContent"),
]
# (保存値, 表示ラベルの翻訳キー)。保存値は内部IDのまま固定し、表示だけ訳す。
_AUTH_TYPES = [
    ("none", "Custom_Auth_None"),
    ("bearer", "Custom_Auth_Bearer"),
    ("header_key", "Custom_Auth_Header_Key"),
    ("query_key", "Custom_Auth_Query_Key"),
]
_LOCALITY = [
    (ConnectionLocality.AUTO, "Custom_Locality_Auto"),
    (ConnectionLocality.LOCAL, "Custom_Locality_Local"),
    (ConnectionLocality.EXTERNAL, "Custom_Locality_External"),
]


class CustomConnectionDialog(QDialog):
    def __init__(self, get_string: GetString, existing: dict | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._t = get_string
        self._existing = dict(existing or {})
        self._result: dict | None = None
        self._model_thread: QThread | None = None
        self._model_worker: VlmModelListWorker | None = None
        self._pending_done: int | None = None
        # Consent is scoped to the exact external HTTP URL. Editing the URL requires
        # a fresh confirmation before either model discovery or Save can use it.
        self._confirmed_external_http_url = ""
        self.setWindowTitle(get_string("Vlm", "Custom_Dialog_Title"))
        self.setMinimumWidth(460)
        self._build()
        self._load(self._existing)

    # --- UI ---
    def _build(self) -> None:
        root = QVBoxLayout(self)

        basic = QGroupBox(self._t("Vlm", "Custom_Section_Basic"))
        bf = QFormLayout(basic)
        self.name_edit = QLineEdit()
        self.locality_combo = _combo(_LOCALITY, self._t)
        self.protocol_combo = _combo(_PROTOCOLS)
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("http://127.0.0.1:1234/v1")
        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self.model_edit = QComboBox()
        self.model_edit.setEditable(True)
        self.model_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_edit.setMinimumWidth(260)
        self.model_edit.lineEdit().setPlaceholderText(self._t("Vlm", "Custom_Field_ModelId"))
        self.model_fetch_btn = QPushButton(self._t("Vlm", "Settings_Route_FetchModels"))
        self.model_fetch_btn.setToolTip(self._t("Vlm", "Settings_Route_FetchModels_Tooltip"))
        self.model_fetch_btn.clicked.connect(self._fetch_model_list)
        model_layout.addWidget(self.model_edit, 1)
        model_layout.addWidget(self.model_fetch_btn)
        self.model_status = QLabel()
        self.model_status.setWordWrap(True)
        bf.addRow(self._t("Vlm", "Custom_Field_Name"), self.name_edit)
        bf.addRow(self._t("Vlm", "Custom_Field_Locality"), self.locality_combo)
        bf.addRow(self._t("Vlm", "Custom_Field_Protocol"), self.protocol_combo)
        bf.addRow(self._t("Vlm", "Custom_Field_BaseUrl"), self.base_url_edit)
        bf.addRow(self._t("Vlm", "Custom_Field_ModelId"), model_row)
        bf.addRow("", self.model_status)
        root.addWidget(basic)

        auth = QGroupBox(self._t("Vlm", "Custom_Section_Auth"))
        af = QFormLayout(auth)
        self.auth_type_combo = _combo(_AUTH_TYPES, self._t)
        self.auth_header_edit = QLineEdit("Authorization")
        self.auth_query_edit = QLineEdit("key")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText(self._t("Vlm", "Custom_ApiKey_Placeholder"))
        self.persist_key_check = QCheckBox(self._t("Vlm", "Custom_ApiKey_Persist"))
        self.persist_key_check.setChecked(vlm_secrets.keyring_available())
        self.verify_tls_check = QCheckBox(self._t("Vlm", "Custom_Field_VerifyTls"))
        self.verify_tls_check.setChecked(True)
        af.addRow(self._t("Vlm", "Custom_Field_AuthType"), self.auth_type_combo)
        af.addRow(self._t("Vlm", "Custom_Field_AuthHeader"), self.auth_header_edit)
        af.addRow(self._t("Vlm", "Custom_Field_AuthQuery"), self.auth_query_edit)
        af.addRow(self._t("Vlm", "Custom_Field_ApiKey"), self.api_key_edit)
        af.addRow("", self.persist_key_check)
        af.addRow("", self.verify_tls_check)
        root.addWidget(auth)

        adv = QGroupBox(self._t("Vlm", "Custom_Section_Advanced"))
        gf = QFormLayout(adv)
        self.connect_timeout = _spin(1, 120, 10)
        self.read_timeout = _spin(5, 900, 120)
        self.retry_same = _spin(0, 5, 1)
        self.text_path_edit = QLineEdit()
        self.text_path_edit.setPlaceholderText("choices[0].message.content")
        self.max_edge = _spin(256, 8192, 1536)
        gf.addRow(self._t("Vlm", "Custom_Field_ConnectTimeout"), self.connect_timeout)
        gf.addRow(self._t("Vlm", "Custom_Field_ReadTimeout"), self.read_timeout)
        gf.addRow(self._t("Vlm", "Custom_Field_RetrySame"), self.retry_same)
        gf.addRow(self._t("Vlm", "Custom_Field_TextPath"), self.text_path_edit)
        gf.addRow(self._t("Vlm", "Custom_Field_MaxEdge"), self.max_edge)
        root.addWidget(adv)

        root.addWidget(QLabel(self._t("Vlm", "Custom_Header_Note")))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        self._save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        root.addWidget(buttons)

        self.auth_type_combo.currentIndexChanged.connect(self._sync_auth_rows)
        self.base_url_edit.editingFinished.connect(self._fetch_model_list)
        self.api_key_edit.editingFinished.connect(self._fetch_model_list)
        self.protocol_combo.currentIndexChanged.connect(self._fetch_model_list)
        self._sync_auth_rows()

    def _sync_auth_rows(self) -> None:
        atype = self.auth_type_combo.currentData()
        # bearer は常に Authorization ヘッダーを使い、header_name は実行時に無視される
        # （apply_connection_auth が header_name を見るのは header_key のときだけ）。
        self.auth_header_edit.setEnabled(atype == "header_key")
        self.auth_query_edit.setEnabled(atype == "query_key")
        self.api_key_edit.setEnabled(atype != "none")
        self.persist_key_check.setEnabled(atype != "none" and vlm_secrets.keyring_available())

    # --- data ---
    def _load(self, data: dict) -> None:
        self.name_edit.setText(str(data.get("display_name", "")))
        _select(self.protocol_combo, data.get("protocol", "openai_chat_completions"))
        self.base_url_edit.setText(str(data.get("base_url", "")))
        self.model_edit.setCurrentText(str(data.get("model_id", "")))
        auth = data.get("auth", {}) if isinstance(data.get("auth"), dict) else {}
        _select(self.auth_type_combo, auth.get("type", "none"))
        self.auth_header_edit.setText(str(auth.get("header_name", "Authorization")))
        self.auth_query_edit.setText(str(auth.get("query_param", "key")))
        retry = data.get("retry", {}) if isinstance(data.get("retry"), dict) else {}
        self.connect_timeout.setValue(_int(retry.get("connect_timeout_s"), 10))
        self.read_timeout.setValue(_int(retry.get("read_timeout_s"), 120))
        self.retry_same.setValue(_int(retry.get("retry_same_max"), 1))
        resp = data.get("response", {}) if isinstance(data.get("response"), dict) else {}
        self.text_path_edit.setText(str(resp.get("text_path", "") or data.get("text_path", "")))
        self.verify_tls_check.setChecked(bool(data.get("verify_tls", True)))
        img = data.get("image", {}) if isinstance(data.get("image"), dict) else {}
        self.max_edge.setValue(_int(img.get("max_long_edge"), 1536))
        kind_raw = str(data.get("kind", "")).lower()
        if kind_raw == "custom_local":
            _select_enum(self.locality_combo, ConnectionLocality.LOCAL)
        elif kind_raw == "custom_external":
            _select_enum(self.locality_combo, ConnectionLocality.EXTERNAL)
        else:
            _select_enum(self.locality_combo, ConnectionLocality.AUTO)
        self._sync_auth_rows()

    # --- model list ---------------------------------------------------------
    def _model_list_connection(self) -> VlmConnection | None:
        base_url = self.base_url_edit.text().strip()
        if not base_url:
            return None
        cid = str(self._existing.get("connection_id") or "custom-model-list")
        locality = self.locality_combo.currentData()
        kind = resolve_custom_kind(locality, base_url)
        atype = str(self.auth_type_combo.currentData() or "none")
        return VlmConnection.from_mapping({
            "connection_id": cid,
            "display_name": self.name_edit.text().strip() or cid,
            "kind": kind.value,
            "protocol": self.protocol_combo.currentData(),
            "base_url": base_url,
            "model_id": self.model_edit.currentText().strip(),
            "verify_tls": self.verify_tls_check.isChecked(),
            "auth": {
                "type": atype,
                "header_name": self.auth_header_edit.text().strip() or "Authorization",
                "query_param": self.auth_query_edit.text().strip() or "key",
            },
        })

    def _model_list_key(self) -> str | None:
        if self.auth_type_combo.currentData() == "none":
            return None
        key = self.api_key_edit.text().strip()
        if key:
            return key
        auth = self._existing.get("auth", {})
        ref = auth.get("secret_ref", "") if isinstance(auth, dict) else ""
        return vlm_secrets.get_secret(ref) or None

    def _confirm_external_http(self, base_url: str, kind: ConnectionKind) -> bool:
        """Obtain consent before any cleartext request can leave this dialog."""
        try:
            scheme = urlparse(base_url).scheme.lower()
        except ValueError:
            scheme = ""
        if kind is not ConnectionKind.CUSTOM_EXTERNAL or scheme != "http":
            return True
        if self._confirmed_external_http_url == base_url:
            return True
        answer = QMessageBox.warning(
            self,
            self._t("Vlm", "Custom_Insecure_Auth_Title"),
            self._t("Vlm", "Custom_Insecure_Auth_Warning"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._confirmed_external_http_url = base_url
        return True

    def _fetch_model_list(self) -> None:
        if self._model_thread is not None:
            return
        conn = self._model_list_connection()
        if conn is None:
            return
        # editingFinished can trigger discovery before Save. Ask before resolving an
        # entered/stored key or constructing the worker, so declining sends nothing.
        if not self._confirm_external_http(conn.base_url, conn.kind):
            return
        self.model_status.setText(self._t("Vlm", "Settings_Route_FetchModels_Busy"))
        self.model_fetch_btn.setEnabled(False)
        self._save_button.setEnabled(False)
        self._model_thread = QThread(self)
        self._model_worker = VlmModelListWorker(conn, self._model_list_key())
        self._model_worker.moveToThread(self._model_thread)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.result_ready.connect(self._on_model_list)
        self._model_worker.finished.connect(self._model_thread.quit)
        self._model_thread.finished.connect(self._model_fetch_done)
        self._model_thread.start()

    @Slot(str, object)
    def _on_model_list(self, _connection_id: str, result) -> None:
        if not isinstance(result, list):
            detail = getattr(result, "message", "") or str(result)
            self.model_status.setText(self._t(
                "Vlm", "Settings_Route_FetchModels_Fail", detail=detail))
            return
        entries = [entry if isinstance(entry, ModelCatalogEntry)
                   else catalog_entry_from_id("", str(entry)) for entry in result]
        vlm_ids = [entry.model_id for entry in filter_vlm_catalog(
            entries, exclude_disabled_markers=False)]
        current = self.model_edit.currentText().strip()
        self.model_edit.blockSignals(True)
        self.model_edit.clear()
        self.model_edit.addItems(vlm_ids)
        if current in vlm_ids:
            self.model_edit.setCurrentText(current)
        elif vlm_ids:
            self.model_edit.setCurrentIndex(0)
        else:
            self.model_edit.setCurrentText(current)
        self.model_edit.blockSignals(False)
        self.model_status.setText(self._t(
            "Vlm", "Settings_Route_FetchModels_Ok", n=len(vlm_ids)))

    def _model_fetch_done(self) -> None:
        if self._model_worker is not None:
            self._model_worker.deleteLater()
            self._model_worker = None
        if self._model_thread is not None:
            self._model_thread.deleteLater()
            self._model_thread = None
        self.model_fetch_btn.setEnabled(True)
        self._save_button.setEnabled(True)
        if self._pending_done is not None:
            result = self._pending_done
            self._pending_done = None
            QDialog.done(self, result)

    def done(self, result: int) -> None:
        thread = self._model_thread
        if thread is not None and thread.isRunning():
            self._pending_done = result
            try:
                self._model_worker.result_ready.disconnect()
            except (RuntimeError, TypeError, AttributeError):
                pass
            thread.quit()
            self.setEnabled(False)
            return
        super().done(result)

    def _on_save(self) -> None:
        name = self.name_edit.text().strip()
        base_url = self.base_url_edit.text().strip()
        model_id = self.model_edit.currentText().strip()
        if not name or not base_url or not model_id:
            QMessageBox.warning(self, self._t("Vlm", "Custom_Dialog_Title"),
                                self._t("Vlm", "Custom_Validation_Required"))
            return

        cid = self._existing.get("connection_id") or new_connection_id()
        locality = self.locality_combo.currentData()
        kind = resolve_custom_kind(locality, base_url)
        atype = self.auth_type_combo.currentData()
        secret_ref = self._existing.get("auth", {}).get("secret_ref") if isinstance(self._existing.get("auth"), dict) else ""
        secret_ref = secret_ref or (f"vlm/custom/{cid}" if atype != "none" else "")

        # Reuse model-discovery consent. This is deliberately a confirmation rather
        # than a blanket runtime ban; explicitly trusted external HTTP remains usable.
        if not self._confirm_external_http(base_url, kind):
            return

        if atype != "none":
            key = self.api_key_edit.text().strip()
            if key:
                persist_key = self.persist_key_check.isChecked()
                # セッション限定に切り替えるときは、以前 keyring へ保存した値を先に
                # 消す。残すと set_secret(persist=False) はセッション上書きを足すだけで、
                # 再起動後に get_secret が古い keyring 値を返してしまう。
                if not persist_key and not vlm_secrets.delete_secret(secret_ref):
                    # keyring から消せなかった。セッション上書きは効くが、再起動後は
                    # 古い keyring 値が復活しうる旨を明示する（保存自体は続行）。
                    QMessageBox.warning(
                        self, self._t("Vlm", "Custom_Dialog_Title"),
                        self._t("Vlm", "Custom_Key_Persist_Remove_Failed"))
                vlm_secrets.set_secret(secret_ref, key, persist=persist_key)
        else:
            # 認証なしに変更したら、以前保存した鍵は残さない。
            old_ref = self._existing.get("auth", {}).get("secret_ref") if isinstance(self._existing.get("auth"), dict) else ""
            if old_ref:
                vlm_secrets.delete_secret(old_ref)

        self._result = {
            "connection_id": cid,
            "display_name": name,
            "kind": kind.value,
            "protocol": self.protocol_combo.currentData(),
            "base_url": base_url,
            "model_id": model_id,
            "enabled": bool(self._existing.get("enabled", True)),
            "verify_tls": self.verify_tls_check.isChecked(),
            # UI からは編集させないが、保存済みの値は往復で失わないよう持ち越す
            # （実行時に効果は無いが、編集のたびに 1 へ落ちるのを防ぐ）。
            "concurrency": self._existing.get("concurrency", 1),
            "auth": {
                "type": atype,
                "secret_ref": secret_ref,
                "header_name": self.auth_header_edit.text().strip() or "Authorization",
                "query_param": self.auth_query_edit.text().strip() or "key",
            },
            "retry": {
                "connect_timeout_s": self.connect_timeout.value(),
                "read_timeout_s": self.read_timeout.value(),
                "retry_same_max": self.retry_same.value(),
            },
            "response": {"text_path": self.text_path_edit.text().strip()},
            "image": {"max_long_edge": self.max_edge.value()},
        }
        self.accept()

    def result_connection(self) -> dict | None:
        return self._result

    def closeEvent(self, event) -> None:
        thread = self._model_thread
        if thread is not None and thread.isRunning():
            self.done(QDialog.DialogCode.Rejected)
            event.ignore()
            return
        super().closeEvent(event)


def _combo(pairs, translate=None) -> QComboBox:
    """(保存値, ラベル) の並びからコンボを作る。

    `translate` を渡すと、ラベルを [Vlm] の翻訳キーとして解決する（引けなければ
    キーをそのまま表示しないよう、素のラベルへ落とす）。プロトコル名のような
    製品名は訳さないので translate を渡さない。
    """
    c = QComboBox()
    for value, label in pairs:
        text = label
        if translate is not None:
            resolved = translate("Vlm", label)
            if resolved and resolved != label:
                text = resolved
        c.addItem(text, value)
    return c


def _select(combo: QComboBox, value) -> None:
    idx = combo.findData(value)
    combo.setCurrentIndex(idx if idx >= 0 else 0)


def _select_enum(combo: QComboBox, value) -> None:
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return
    combo.setCurrentIndex(0)


def _int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _spin(lo: int, hi: int, val: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s
