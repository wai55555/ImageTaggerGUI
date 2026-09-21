"""VLM 設定ダイアログ（260901_VLM_design.md 6.3節 / implement_plan 7章）。

通常画面に複雑な HTTP 設定は出さず、ここで管理する:
  - キャプションプロファイル（表示のみ）
  - 実行モード（内蔵の厳格フォールバック / カスタム接続単独）
  - フォールバック経路（内蔵3接続の有効・APIキー・診断）
  - 接続経路とプロバイダー側の料金注意
  - 詳細な出力設定（プロンプトモード・詳細度・文数・キャラクター名・Markdown・最大トークン）
  - カスタム接続の追加・編集・削除
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QPushButton, QRadioButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

import vlm_config
import vlm_models
import vlm_profiles
import vlm_secrets
import app_settings
from custom_connection_dialog import CustomConnectionDialog
from vlm_connections import ConnectionKind, VlmConnection
from vlm_diagnostics import DiagStatus
from vlm_model_list import (
    ModelCatalogEntry, catalog_entry_from_id, filter_vlm_catalog,
)
from vlm_prompt_preview import PromptPreviewRoute, build_prompt_preview
from vlm_prompt_preview_dialog import VlmPromptPreviewDialog
from vlm_worker import VlmDiagnosticsWorker, VlmModelListWorker

GetString = Callable[..., str]

# 出力設定コンボの選択肢。表示ラベルは locale（[Vlm] Opt_*）から引くので、ここは
# 保存値（config.ini に書く文字列）だけを持つ。
# 出力言語は今は "en" 固定（プロンプトが英語前提）。将来対応言語が増えたらここへ
# 足せば UI に出る。当面はコンボを無効化してグレー表示する。
_LANGUAGE_KEYS = ["en"]
_DETAIL_KEYS = ["standard", "detailed", "maximum_detail"]
_SENTENCE_KEYS = ["automatic_long_detailed", "1", "2", "3", "4", "5"]
_CHARNAME_KEYS = ["do_not_identify", "explicit_only", "allow_guessing"]
_MARKDOWN_KEYS = ["disabled", "allowed"]
_PROMPT_MODE_KEYS = ["standard", "dataset_long", "short_tags"]

# フォールバック経路グリッドの列。up/down は1セルに横並びで入れる。
(_ROUTE_COL_UPDOWN, _ROUTE_COL_ENABLED, _ROUTE_COL_NAME, _ROUTE_COL_MODEL,
 _ROUTE_COL_LIST, _ROUTE_COL_REGISTER, _ROUTE_COL_STATUS, _ROUTE_COL_DIAG) = range(8)

_BUILTIN_SECRET_REF = {
    "builtin-gemini": "vlm/gemini/api_key",
    "builtin-openrouter": "vlm/openrouter/api_key",
    "builtin-cloudflare": "vlm/cloudflare/api_token",
    "builtin-groq": "vlm/groq/api_key",
    "builtin-nvidia": "vlm/nvidia/api_key",
    # "builtin-mistral": "vlm/mistral/api_key",  # Pixtral内蔵経路は一時停止
    "builtin-huggingface": "vlm/huggingface/api_token",
    "builtin-vercel": "vlm/vercel/api_key",
    "builtin-openai": "vlm/openai/api_key",
    "builtin-anthropic": "vlm/anthropic/api_key",
    # "builtin-ovhcloud": "vlm/ovhcloud/api_key",
}

# 各サービスの「APIキー取得ページ」。key_url が直接キー作成ページ、login_url は
# 未ログイン時に案内する入口。instructions_key は locale の説明文キー。
_PROVIDER_KEY_INFO = {
    "gemini": {
        "key_url": "https://aistudio.google.com/app/api-keys",
        "login_url": "https://aistudio.google.com/",
        "instructions_key": "ApiKey_Steps_Gemini",
    },
    "openrouter": {
        "key_url": "https://openrouter.ai/workspaces/default/keys",
        "login_url": "https://openrouter.ai/",
        "instructions_key": "ApiKey_Steps_OpenRouter",
    },
    "cloudflare": {
        "key_url": "https://dash.cloudflare.com/?to=/:account/ai/workers-ai",
        "login_url": "https://dash.cloudflare.com/sign-up",
        "instructions_key": "ApiKey_Steps_Cloudflare",
    },
    "groq": {
        "key_url": "https://console.groq.com/keys",
        "login_url": "https://console.groq.com/login",
        "instructions_key": "ApiKey_Steps_Groq",
    },
    "nvidia": {
        "key_url": "https://build.nvidia.com/settings/api-keys",
        "login_url": "https://build.nvidia.com/login",
        "instructions_key": "ApiKey_Steps_Nvidia",
    },
    # "mistral": {  # Pixtral内蔵経路は一時停止
    #     "key_url": "https://console.mistral.ai/api-keys",
    #     "login_url": "https://console.mistral.ai/",
    #     "instructions_key": "ApiKey_Steps_Mistral",
    # },
    "huggingface": {
        "key_url": "https://huggingface.co/settings/tokens",
        "login_url": "https://huggingface.co/login",
        "instructions_key": "ApiKey_Steps_HuggingFace",
    },
    "vercel": {
        # Team slugを含むキー画面URLはアカウントごとに異なるため、共通のDashboard入口。
        "key_url": "https://vercel.com/dashboard",
        "login_url": "https://vercel.com/login",
        "instructions_key": "ApiKey_Steps_Vercel",
    },
    "openai": {
        "key_url": "https://platform.openai.com/api-keys",
        "login_url": "https://platform.openai.com/login",
        "instructions_key": "ApiKey_Steps_OpenAI",
    },
    "anthropic": {
        "key_url": "https://console.anthropic.com/settings/keys",
        "login_url": "https://console.anthropic.com/",
        "instructions_key": "ApiKey_Steps_Anthropic",
    },
    "xai": {
        "key_url": "https://console.x.ai/welcome?path=%2Fapi-keys",
        "login_url": "https://console.x.ai/",
        "instructions_key": "ApiKey_Steps_Xai",
    },
    # OVHcloud は日本居住者環境で実機検証できるまで無効。
    # "ovhcloud": {
    #     "key_url": "https://www.ovh.com/manager/#/public-cloud/",
    #     "login_url": "https://www.ovh.com/auth/",
    #     "instructions_key": "ApiKey_Steps_OVHcloud",
    # },
}


class VlmSettingsDialog(QDialog):
    def __init__(self, settings, get_string: GetString, parent: QWidget | None = None):
        super().__init__(parent)
        self._settings = settings
        self._vlm = settings.vlm
        self._vlm_before_dialog = dataclasses.replace(self._vlm)
        self._dialog_saved = False
        self._immediate_settings_saved = False
        self._t = get_string
        self._custom_connections: list[dict] = vlm_config.load_custom_connections()
        self._diag_thread: QThread | None = None
        self._diag_worker: VlmDiagnosticsWorker | None = None
        self._diag_pending_profile_id: str | None = None
        self._pending_done: int | None = None
        self.setWindowTitle(get_string("Vlm", "Settings_Title"))
        self.setMinimumWidth(520)
        self._build()
        self._load()

    def _opts(self, prefix: str, keys: list[str]) -> list[tuple[str, str]]:
        """保存値 key と locale から引いた表示ラベルの組にする（[Vlm] <prefix>_<key>）。"""
        return [(k, self._t("Vlm", f"{prefix}_{k}")) for k in keys]

    def _build(self) -> None:
        root = QVBoxLayout(self)

        prof = QGroupBox(self._t("Vlm", "Settings_Profile"))
        pf = QFormLayout(prof)
        prow = QHBoxLayout()
        self.profile_combo = QComboBox()
        prow.addWidget(self.profile_combo, 1)
        self.profile_new_btn = QPushButton(self._t("Vlm", "Profile_New"))
        self.profile_dup_btn = QPushButton(self._t("Vlm", "Profile_Duplicate"))
        self.profile_edit_btn = QPushButton(self._t("Vlm", "Profile_Edit"))
        self.profile_del_btn = QPushButton(self._t("Vlm", "Profile_Delete"))
        self.profile_new_btn.clicked.connect(self._new_profile)
        self.profile_dup_btn.clicked.connect(self._dup_profile)
        self.profile_edit_btn.clicked.connect(self._edit_profile)
        self.profile_del_btn.clicked.connect(self._del_profile)
        for b in (self.profile_new_btn, self.profile_dup_btn, self.profile_edit_btn, self.profile_del_btn):
            prow.addWidget(b)
        pf.addRow(self._t("Vlm", "Settings_Caption_Profile"), prow)
        self.profile_canon_label = QLabel()
        self.profile_canon_label.setStyleSheet("color: gray;")
        self.profile_canon_label.setWordWrap(True)
        pf.addRow("", self.profile_canon_label)
        root.addWidget(prof)
        self._reload_profiles()

        mode = QGroupBox(self._t("Vlm", "Settings_Exec_Mode"))
        mv = QVBoxLayout(mode)
        self.mode_builtin = QRadioButton(self._t("Vlm", "Settings_Mode_Builtin"))
        self.mode_custom = QRadioButton(self._t("Vlm", "Settings_Mode_Custom"))
        row = QHBoxLayout()
        row.addWidget(self.mode_custom)
        self.custom_select = QComboBox()
        row.addWidget(self.custom_select, 1)
        mv.addWidget(self.mode_builtin)
        mv.addLayout(row)
        root.addWidget(mode)

        routes = QGroupBox(self._t("Vlm", "Settings_Routes"))
        rv = QVBoxLayout(routes)
        self._route_rows: dict[str, dict] = {}
        # 経路の優先順位はこのリストの順。▲▼ ボタンで並べ替える。
        self._route_order: list[str] = []
        # 経路行は QGridLayout で組む。行ごとに別レイアウトにすると、プロバイダー名や
        # モデルID・状態ラベルの文字幅の違いで列がガタガタにずれるため（グリッドなら
        # 各列が全行の最大幅にそろう）。モデルID列だけ伸縮させて余白を吸わせる。
        self._routes_grid = QGridLayout()
        self._routes_grid.setContentsMargins(0, 0, 0, 0)
        self._routes_grid.setHorizontalSpacing(6)
        self._routes_grid.setVerticalSpacing(4)
        self._routes_grid.setColumnStretch(_ROUTE_COL_MODEL, 1)
        rv.addLayout(self._routes_grid)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        self.strict_check = QCheckBox(self._t("Vlm", "Settings_Strict_Identity"))
        self.strict_check.setToolTip(self._t("Vlm", "Settings_Strict_Identity_Tooltip"))
        rv.addWidget(self.strict_check)
        self.route_cost_note = QLabel(self._t("Vlm", "Settings_Route_Cost_Note"))
        self.route_cost_note.setStyleSheet("color: gray;")
        self.route_cost_note.setToolTip(self._t("Vlm", "Settings_Route_Cost_Note"))
        rv.addWidget(self.route_cost_note)
        root.addWidget(routes)

        det = QGroupBox(self._t("Vlm", "Settings_Detail"))
        dfrm = QFormLayout(det)
        self.prompt_mode_combo = _combo(self._opts("Opt_PromptMode", _PROMPT_MODE_KEYS))
        self.prompt_mode_combo.setToolTip(self._t("Vlm", "Settings_PromptMode_Tooltip"))
        self.detail_combo = _combo(self._opts("Opt_Detail", _DETAIL_KEYS))
        self.sentence_combo = _combo(self._opts("Opt_Sentence", _SENTENCE_KEYS))
        self.charname_combo = _combo(self._opts("Opt_CharName", _CHARNAME_KEYS))
        self.markdown_combo = _combo(self._opts("Opt_Markdown", _MARKDOWN_KEYS))
        self.language_combo = _combo(self._opts("Opt_Language", _LANGUAGE_KEYS))
        # 当面は English 固定。選択式にしておくが操作不可（グレー）にする。
        self.language_combo.setEnabled(len(_LANGUAGE_KEYS) > 1)
        self.language_combo.setToolTip(self._t("Vlm", "Settings_Language_Fixed_Tooltip"))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(16, 32768)
        dfrm.addRow(self._t("Vlm", "Settings_PromptMode"), self.prompt_mode_combo)
        dfrm.addRow(self._t("Vlm", "Settings_Language"), self.language_combo)
        dfrm.addRow(self._t("Vlm", "Settings_DetailLevel"), self.detail_combo)
        dfrm.addRow(self._t("Vlm", "Settings_SentenceMode"), self.sentence_combo)
        dfrm.addRow(self._t("Vlm", "Settings_CharName"), self.charname_combo)
        dfrm.addRow(self._t("Vlm", "Settings_Markdown"), self.markdown_combo)
        dfrm.addRow(self._t("Vlm", "Settings_MaxTokens"), self.max_tokens)
        self.prompt_preview_btn = QPushButton(self._t("Vlm", "PromptPreview_Button"))
        self.prompt_preview_btn.setToolTip(self._t("Vlm", "PromptPreview_Button_Tooltip"))
        self.prompt_preview_btn.clicked.connect(self._open_prompt_preview)
        dfrm.addRow("", self.prompt_preview_btn)
        root.addWidget(det)
        self.prompt_mode_combo.currentIndexChanged.connect(self._on_prompt_mode_changed)

        cust = QGroupBox(self._t("Vlm", "Settings_Custom"))
        cvv = QVBoxLayout(cust)
        self.custom_list = QListWidget()
        cvv.addWidget(self.custom_list)
        cbtns = QHBoxLayout()
        self.add_custom_btn = QPushButton(self._t("Vlm", "Settings_Custom_Add"))
        self.edit_custom_btn = QPushButton(self._t("Vlm", "Settings_Custom_Edit"))
        self.del_custom_btn = QPushButton(self._t("Vlm", "Settings_Custom_Delete"))
        self.add_custom_btn.clicked.connect(self._add_custom)
        self.edit_custom_btn.clicked.connect(self._edit_custom)
        self.del_custom_btn.clicked.connect(self._delete_custom)
        for b in (self.add_custom_btn, self.edit_custom_btn, self.del_custom_btn):
            cbtns.addWidget(b)
        cvv.addLayout(cbtns)
        root.addWidget(cust)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._on_save)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)
        root.addWidget(buttons)

        self.mode_custom.toggled.connect(lambda on: self.custom_select.setEnabled(on))

    def _make_route_row(self, conn) -> dict:
        # up/down は1つのグリッドセルに収めるため小さなコンテナにまとめる。
        updown = QWidget()
        ud = QHBoxLayout(updown)
        ud.setContentsMargins(0, 0, 0, 0)
        ud.setSpacing(2)
        up = QPushButton("▲")
        down = QPushButton("▼")
        up.setFixedWidth(28)
        down.setFixedWidth(28)
        ud.addWidget(up)
        ud.addWidget(down)
        up.clicked.connect(lambda _=False, cid=conn.connection_id: self._move_route(cid, -1))
        down.clicked.connect(lambda _=False, cid=conn.connection_id: self._move_route(cid, +1))
        enabled = QCheckBox()
        name = QLabel(conn.display_name)
        # プロバイダー名は全行の左端をそろえる。Cloudflareの表示名は短い固定名にする。
        name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        model_combo = QComboBox()
        model_combo.setEditable(True)
        model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        model_combo.setMinimumWidth(190)
        model_combo.lineEdit().setPlaceholderText(self._t("Vlm", "Settings_Route_ModelId"))
        model_combo.setToolTip(self._t("Vlm", "Settings_Route_ModelId_Tooltip"))
        if conn.model_id:
            model_combo.addItem(conn.model_id)
            model_combo.setCurrentText(conn.model_id)
        else:
            model_combo.setCurrentText("")
        model_combo.lineEdit().editingFinished.connect(
            lambda cid=conn.connection_id: self._on_model_id_edited(cid))
        model_combo.activated.connect(
            lambda _i, cid=conn.connection_id: self._on_model_id_edited(cid))
        list_btn = QPushButton(self._t("Vlm", "Settings_Route_FetchModels"))
        list_btn.setToolTip(self._t("Vlm", "Settings_Route_FetchModels_Tooltip"))
        list_btn.clicked.connect(lambda _=False, cid=conn.connection_id: self._fetch_models(cid))
        register_btn = QPushButton(self._t("Vlm", "Settings_Register_ApiKey"))
        register_btn.clicked.connect(lambda _=False, cid=conn.connection_id: self._open_api_key_dialog(cid))
        status = QLabel()
        # 状態文は言語・保存場所・確認済み表示の組み合わせで長さが変わるため、
        # 列幅を文字列に追従させない。全文はツールチップへ残し、表示は必要なら省略する。
        status.setFixedWidth(180)
        status.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        diag_btn = QPushButton(self._t("Vlm", "Settings_Diagnose"))
        diag_btn.clicked.connect(lambda _=False, cid=conn.connection_id: self._diagnose_one(cid))
        return {
            "updown": updown, "up": up, "down": down, "name": name,
            "enabled": enabled, "model_edit": model_combo, "list_btn": list_btn,
            "register": register_btn,
            "status": status, "diag_btn": diag_btn,
            "secret_ref": _BUILTIN_SECRET_REF.get(conn.connection_id, conn.auth.secret_ref),
            "conn": conn,
        }

    _ROUTE_CELLS = (
        ("updown", _ROUTE_COL_UPDOWN), ("enabled", _ROUTE_COL_ENABLED),
        ("name", _ROUTE_COL_NAME), ("model_edit", _ROUTE_COL_MODEL),
        ("list_btn", _ROUTE_COL_LIST), ("register", _ROUTE_COL_REGISTER),
        ("status", _ROUTE_COL_STATUS), ("diag_btn", _ROUTE_COL_DIAG),
    )

    def _rebuild_routes(self) -> None:
        """内蔵経路の行を作り直す。全プロバイダーを出し、選択プロファイルに binding が
        あるものはその model_id を、無いものは空欄（＝ここに実 ID を入れて接続を試す）。"""
        for r in self._route_rows.values():
            for key in ("updown", "enabled", "name", "model_edit", "list_btn",
                        "register", "status", "diag_btn"):
                w = r[key]
                w.setParent(None)
                w.deleteLater()
        self._route_rows.clear()
        self._route_order.clear()
        profile = vlm_config.resolve_model_profile(self._vlm)
        if hasattr(self, "profile_canon_label"):
            self.profile_canon_label.setText(
                self._t("Vlm", "Settings_Profile_Canonical",
                        id=profile.canonical_model_id) if profile is not None else "")
        conns = [c for c in vlm_config.build_connection_map(self._vlm, profile).values()
                 if c.kind is ConnectionKind.BUILTIN]
        order = vlm_config.ordered_builtin_provider_ids(self._vlm, profile)
        conns.sort(key=lambda c: order.index(c.provider_id) if c.provider_id in order else 99)
        for conn in conns:
            row = self._make_route_row(conn)
            # このプロファイルに binding が無い経路でも、APIキー登録後に公式のVLM一覧を
            # 探索・診断できるようにする。bindingがない経路は同一モデルの自動フォール
            # バックには入れないため、実行対象チェックだけ無効化する。
            has_binding = profile is None or profile.binding_for(conn.provider_id) is not None
            if not has_binding:
                row["name"].setStyleSheet("color: gray;")
                row["name"].setToolTip(self._t("Vlm", "Settings_Route_No_Binding"))
                row["enabled"].setEnabled(False)
            self._route_rows[conn.connection_id] = row
            self._route_order.append(conn.connection_id)
        self._relayout_routes()
        self._apply_route_states()

    def _apply_route_states(self) -> None:
        profile = vlm_config.resolve_model_profile(self._vlm)
        order = set(vlm_config.ordered_builtin_provider_ids(self._vlm, profile))
        for cid, r in self._route_rows.items():
            provider = r["conn"].provider_id
            r["enabled"].setChecked(provider in order)
            self._refresh_route_status(cid)

    def _on_profile_changed(self) -> None:
        pid = self.profile_combo.currentData()
        is_user = vlm_config.is_user_profile(pid or "")
        self.profile_edit_btn.setEnabled(is_user)
        self.profile_del_btn.setEnabled(is_user)
        if not pid or pid == self._vlm.model_profile_id:
            return
        self._vlm.model_profile_id = pid
        self._rebuild_routes()

    def _reload_profiles(self, select_id: str | None = None) -> None:
        want = select_id or self.profile_combo.currentData() or self._vlm.model_profile_id
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for p in vlm_config.all_profiles():
            self.profile_combo.addItem(p.display_name, p.profile_id)
        i = self.profile_combo.findData(want)
        self.profile_combo.setCurrentIndex(i if i >= 0 else 0)
        self.profile_combo.blockSignals(False)
        pid = self.profile_combo.currentData() or ""
        is_user = vlm_config.is_user_profile(pid)
        self.profile_edit_btn.setEnabled(is_user)
        self.profile_del_btn.setEnabled(is_user)

    def _profile_dict_for(self, profile_id: str) -> dict:
        p = next((x for x in vlm_config.all_profiles() if x.profile_id == profile_id), None)
        if p is None:
            return {}
        return {
            "profile_id": p.profile_id, "display_name": p.display_name,
            "canonical_model_id": p.canonical_model_id,
            "bindings": {prov: {"model_id": b.model_id,
                                 "vlm_capable": b.vlm_capable}
                         for prov, b in p.bindings.items()},
        }

    def _open_profile_editor(self, src: dict | None, *, read_only: bool = False) -> None:
        from vlm_profile_editor import ProfileEditorDialog
        dlg = ProfileEditorDialog(self._t, src, read_only=read_only, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_profile() is None:
            return
        result = dlg.result_profile()
        users = [d for d in vlm_config.load_user_profiles() if d.get("profile_id") != result["profile_id"]]
        users.append(result)
        if not vlm_config.save_user_profiles(users):
            # 書き込みに失敗したのに未保存のプロファイルを選択・永続化しない。
            QMessageBox.critical(self, self._t("Vlm", "Settings_Title"),
                                 self._t("Vlm", "Settings_Save_Failed"))
            return
        self._vlm.model_profile_id = result["profile_id"]
        self._reload_profiles(result["profile_id"])
        self._rebuild_routes()

    def _new_profile(self) -> None:
        self._open_profile_editor(None)

    def _dup_profile(self) -> None:
        src = self._profile_dict_for(self.profile_combo.currentData() or "")
        if src:
            src.pop("profile_id", None)   # 新規 ID を振らせる
            src["display_name"] = f'{src.get("display_name", "")} (copy)'
        self._open_profile_editor(src)

    def _edit_profile(self) -> None:
        pid = self.profile_combo.currentData() or ""
        if vlm_config.is_user_profile(pid):
            self._open_profile_editor(self._profile_dict_for(pid))

    def _del_profile(self) -> None:
        pid = self.profile_combo.currentData() or ""
        if not vlm_config.is_user_profile(pid):
            return
        if QMessageBox.question(self, self._t("Vlm", "Settings_Profile"),
                                self._t("Vlm", "Profile_Delete_Confirm")) != QMessageBox.StandardButton.Yes:
            return
        previous_users = vlm_config.load_user_profiles()
        users = [d for d in previous_users if d.get("profile_id") != pid]
        if not vlm_config.save_user_profiles(users):
            QMessageBox.critical(self, self._t("Vlm", "Settings_Title"),
                                 self._t("Vlm", "Settings_Save_Failed"))
            return
        self._reload_profiles(vlm_config.all_profiles()[0].profile_id if vlm_config.all_profiles() else None)
        self._vlm.model_profile_id = self.profile_combo.currentData() or self._vlm.model_profile_id
        if self._vlm_before_dialog.model_profile_id == pid:
            # Profile deletion is immediate. Persist its fallback and move the
            # cancellation snapshot too, so Cancel cannot resurrect a deleted ID.
            if not self._persist_immediate_settings():
                vlm_config.save_user_profiles(previous_users)
                self._vlm.model_profile_id = pid
                self._reload_profiles(pid)
                QMessageBox.critical(self, self._t("Vlm", "Settings_Title"),
                                     self._t("Vlm", "Settings_Save_Failed"))
                return
            self._vlm_before_dialog.model_profile_id = self._vlm.model_profile_id
        self._rebuild_routes()

    def _on_model_id_edited(self, cid: str) -> None:
        r = self._route_rows.get(cid)
        if r is None:
            return
        text = r["model_edit"].currentText().strip()
        profile = vlm_config.resolve_model_profile(self._vlm)
        if text and not vlm_models.is_vlm_model_id(profile, r["conn"].provider_id, text):
            # 手入力でも非VLM／別プロファイルのモデルを採用しない。build_connection_map
            # が使う現在の安全なIDへ表示を戻し、既存の有効なoverrideは保持する。
            combo = r["model_edit"]
            combo.blockSignals(True)
            combo.setCurrentText(r["conn"].model_id)
            combo.blockSignals(False)
            self._set_route_status(r, self._t(
                "Vlm", "Settings_Route_ModelId_NotVlm",
                profile=(profile.display_name if profile else self._vlm.model_profile_id)))
            return
        vlm_config.set_model_id_override(self._vlm, r["conn"].provider_id, text,
                                         profile_id=self._vlm.model_profile_id)
        if text:
            r["conn"].model_id = text   # 診断・キー登録がこの場で新IDを使えるように

    # --- モデル一覧の取得 -------------------------------------------------------
    def _fetch_models(self, cid: str) -> None:
        if getattr(self, "_ml_thread", None) is not None:
            return
        r = self._route_rows.get(cid)
        if r is None:
            return
        conn_map = vlm_config.build_connection_map(
            self._vlm, vlm_config.resolve_model_profile(self._vlm))
        conn = conn_map.get(cid)
        if conn is None:
            return
        api_key = vlm_secrets.get_secret(conn.auth.secret_ref) if conn.auth.type != "none" else None
        self._set_route_status(r, self._t("Vlm", "Settings_Route_FetchModels_Busy"))
        self._ml_pending_cid = cid
        self._ml_thread = QThread(self)
        self._ml_worker = VlmModelListWorker(conn, api_key)
        self._ml_worker.moveToThread(self._ml_thread)
        self._ml_thread.started.connect(self._ml_worker.run)
        self._ml_worker.result_ready.connect(self._on_model_list)
        self._ml_worker.finished.connect(self._ml_thread.quit)
        self._ml_thread.finished.connect(self._ml_cleanup)
        for rr in self._route_rows.values():
            rr["list_btn"].setEnabled(False)
        self._ml_thread.start()

    @Slot(str, object)
    def _on_model_list(self, connection_id: str, result) -> None:
        cid = connection_id or getattr(self, "_ml_pending_cid", "")
        r = self._route_rows.get(cid)
        if r is None:
            return
        if not isinstance(result, list):
            detail = getattr(result, "message", "") or str(result)
            self._set_route_status(r, self._t("Vlm", "Settings_Route_FetchModels_Fail", detail=detail))
            return

        provider_id = r["conn"].provider_id
        entries = [entry if isinstance(entry, ModelCatalogEntry)
                   else catalog_entry_from_id(provider_id, str(entry))
                   for entry in result]
        vlm_entries = filter_vlm_catalog(entries)
        vlm_ids = [entry.model_id for entry in vlm_entries]
        new_ids = vlm_models.new_vlm_model_ids(provider_id, vlm_ids)
        vlm_models.register_discovered_vlm_ids(provider_id, vlm_ids)
        r["model_ids"] = vlm_ids
        combo = r["model_edit"]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(vlm_ids)
        combo.blockSignals(False)

        profile = vlm_config.resolve_model_profile(self._vlm)
        # フォールバックは「同一モデルを多プロバイダーで回す」設計。一覧全体は
        # VLMと確認できたものを表示し、その中から選択プロファイルに一番合うものを
        # 自動で当てる。別モデルを使う場合はプロファイル編集で明示的に割り当てる。
        best, score = vlm_models.match_model_id(profile, r["conn"].provider_id, vlm_ids) \
            if profile is not None else (None, 0.0)
        okmsg = self._t("Vlm", "Settings_Route_FetchModels_Ok", n=len(vlm_ids))
        if best is not None:
            combo.blockSignals(True)
            combo.setCurrentText(best)
            combo.blockSignals(False)
            self._on_model_id_edited(cid)
            key = "Settings_Route_FetchModels_Exact" if score >= 0.999 \
                else "Settings_Route_FetchModels_Matched"
            status = f"{okmsg} — " + self._t("Vlm", key, id=best)
        else:
            combo.blockSignals(True)
            combo.setCurrentText("")
            combo.blockSignals(False)
            status = f"{okmsg} — " + self._t(
                "Vlm", "Settings_Route_FetchModels_NoMatch",
                profile=(profile.display_name if profile else self._vlm.model_profile_id))
        if new_ids:
            # 出荷カタログ(_ALL_PROFILES / _KNOWN_VISION_MODEL_IDS)に未登録の
            # VLM対応モデルが見つかった場合、自動でプロファイルへ追加はせず、通知だけ行う
            # (過去にGroqの偽バインディングを誤って登録した反省から、未検証IDの自動採用はしない)。
            status += " — " + self._t(
                "Vlm", "Settings_Route_NewModelsDetected",
                n=len(new_ids), ids=", ".join(new_ids[:5]))
        self._set_route_status(r, status)

    def _ml_cleanup(self) -> None:
        if getattr(self, "_ml_worker", None) is not None:
            self._ml_worker.deleteLater()
            self._ml_worker = None
        if getattr(self, "_ml_thread", None) is not None:
            self._ml_thread.deleteLater()
            self._ml_thread = None
        for rr in self._route_rows.values():
            rr["list_btn"].setEnabled(True)
        self._finish_pending_done_if_ready()

    def _relayout_routes(self) -> None:
        # グリッドから全セルを外す（ウィジェットは消さない）。順序を _route_order の
        # とおりに並べ直す。
        while self._routes_grid.count():
            self._routes_grid.takeAt(0)
        for pos, cid in enumerate(self._route_order):
            r = self._route_rows[cid]
            for key, col in self._ROUTE_CELLS:
                self._routes_grid.addWidget(r[key], pos, col)
            r["up"].setEnabled(pos > 0)
            r["down"].setEnabled(pos < len(self._route_order) - 1)

    def _move_route(self, cid: str, delta: int) -> None:
        i = self._route_order.index(cid)
        j = i + delta
        if 0 <= j < len(self._route_order):
            self._route_order[i], self._route_order[j] = self._route_order[j], self._route_order[i]
            self._relayout_routes()

    # --- load / save ---
    def _load(self) -> None:
        idx = self.profile_combo.findData(self._vlm.model_profile_id)
        self.profile_combo.blockSignals(True)
        self.profile_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.profile_combo.blockSignals(False)
        if idx < 0 and self.profile_combo.count():
            self._vlm.model_profile_id = self.profile_combo.currentData()
        self._rebuild_routes()

        self.mode_builtin.setChecked(self._vlm.execution_mode != "custom_single")
        self.mode_custom.setChecked(self._vlm.execution_mode == "custom_single")
        self._refresh_custom_list()
        self.custom_select.setEnabled(self.mode_custom.isChecked())

        self.strict_check.setChecked(bool(getattr(self._vlm, "strict_identity", False)))

        _select(self.prompt_mode_combo, getattr(self._vlm, "prompt_mode", "standard"))
        _select(self.detail_combo, self._vlm.detail_level)
        _select(self.sentence_combo, self._vlm.sentence_mode)
        _select(self.charname_combo, self._vlm.character_name_mode)
        _select(self.markdown_combo, self._vlm.markdown)
        _select(self.language_combo, self._vlm.language or "en")
        self.max_tokens.setValue(int(self._vlm.max_output_tokens))
        self._on_prompt_mode_changed()
        # 経路行の有効状態は _rebuild_routes -> _apply_route_states で反映済み。

    def _current_generation_profile(self) -> vlm_profiles.GenerationProfile:
        """Build a preview profile from current widgets, including unsaved values."""
        return vlm_profiles.GenerationProfile.from_mapping({
            "profile_id": self._vlm.generation_profile_id,
            "language": self.language_combo.currentData() or "en",
            "detail_level": self.detail_combo.currentData(),
            "sentence_mode": self.sentence_combo.currentData(),
            "character_name_mode": self.charname_combo.currentData(),
            "markdown": self.markdown_combo.currentData(),
            "prompt_mode": self.prompt_mode_combo.currentData(),
            "max_output_tokens": self.max_tokens.value(),
            "custom_system_prompt": getattr(self._vlm, "custom_system_prompt", ""),
            "temperature": getattr(self._vlm, "temperature", None),
            "top_p": getattr(self._vlm, "top_p", None),
            "image_max_long_edge": getattr(self._vlm, "image_max_long_edge", 1536),
            "image_format": getattr(self._vlm, "image_format", "auto"),
            "image_jpeg_quality": getattr(self._vlm, "image_jpeg_quality", 90),
        })

    def _prompt_preview_routes(self) -> list[PromptPreviewRoute]:
        """Collect only visible route metadata; never read or expose credentials."""
        routes: list[PromptPreviewRoute] = []
        if self.mode_custom.isChecked():
            connection_id = self.custom_select.currentData()
            raw = next((c for c in self._custom_connections
                        if c.get("connection_id") == connection_id), None)
            try:
                conn = VlmConnection.from_mapping(raw) if raw is not None else None
            except (KeyError, ValueError, TypeError):
                conn = None
            if conn is not None:
                routes.append(PromptPreviewRoute(
                    conn.display_name, conn.model_id, conn.protocol))
            return routes

        for cid in self._route_order:
            row = self._route_rows[cid]
            if not row["enabled"].isChecked():
                continue
            routes.append(PromptPreviewRoute(
                row["name"].text(),
                row["model_edit"].currentText().strip() or row["conn"].model_id,
                row["conn"].protocol,
            ))
        return routes

    def _open_prompt_preview(self) -> None:
        preview = build_prompt_preview(
            self._current_generation_profile(), routes=self._prompt_preview_routes())
        VlmPromptPreviewDialog(preview, self._t, self).exec()

    def _on_prompt_mode_changed(self) -> None:
        """Only standard mode consumes the fine-grained prompt clauses."""
        standard = self.prompt_mode_combo.currentData() == "standard"
        for widget in (self.detail_combo, self.sentence_combo,
                       self.charname_combo, self.markdown_combo):
            widget.setEnabled(standard)

    def _refresh_route_status(self, cid: str) -> None:
        r = self._route_rows[cid]
        st = vlm_secrets.secret_status(r["secret_ref"]) if r["secret_ref"] else "missing"
        text = self._t("Vlm", f"Settings_Key_Status_{st}")
        token = f"{self._vlm.model_profile_id}:{r['conn'].provider_id}"
        if token in self._vlm.verified_set():
            text += "  " + self._t("Vlm", "Settings_Route_Verified")
        profile = vlm_config.resolve_model_profile(self._vlm)
        override = self._vlm.model_id_override_map().get(token)
        if override and not vlm_models.is_vlm_model_id(profile, r["conn"].provider_id, override):
            text += "  " + self._t("Vlm", "Settings_Route_ModelId_NotVlm",
                                     profile=(profile.display_name if profile else self._vlm.model_profile_id))
        self._set_route_status(r, text)

    @staticmethod
    def _set_route_status(row: dict, text: str) -> None:
        """固定幅の状態欄へ表示し、全文はツールチップで確認できるようにする。"""
        label = row["status"]
        full_text = str(text or "")
        label.setToolTip(full_text)
        label.setText(label.fontMetrics().elidedText(
            full_text, Qt.TextElideMode.ElideRight, label.width()))

    def _refresh_custom_list(self) -> None:
        self.custom_list.clear()
        self.custom_select.clear()
        for c in self._custom_connections:
            label = f'{c.get("display_name", c["connection_id"])}  [{c.get("kind", "?")}]'
            item = QListWidgetItem(label)
            item.setData(1000, c["connection_id"])
            self.custom_list.addItem(item)
            self.custom_select.addItem(label, c["connection_id"])
        if self._vlm.selected_connection_id:
            idx = self.custom_select.findData(self._vlm.selected_connection_id)
            if idx >= 0:
                self.custom_select.setCurrentIndex(idx)

    def _add_custom(self) -> None:
        dlg = CustomConnectionDialog(self._t, None, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_connection():
            self._custom_connections.append(dlg.result_connection())
            self._refresh_custom_list()

    def _edit_custom(self) -> None:
        cid = self._selected_custom_id()
        if cid is None:
            return
        current = next((c for c in self._custom_connections if c["connection_id"] == cid), None)
        if current is None:
            return
        dlg = CustomConnectionDialog(self._t, current, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_connection():
            updated = dlg.result_connection()
            self._custom_connections = [updated if c["connection_id"] == cid else c
                                        for c in self._custom_connections]
            self._refresh_custom_list()

    def _delete_custom(self) -> None:
        cid = self._selected_custom_id()
        if cid is None:
            return
        if QMessageBox.question(self, self._t("Vlm", "Settings_Custom"),
                                self._t("Vlm", "Settings_Custom_Delete_Confirm")) != QMessageBox.StandardButton.Yes:
            return
        self._custom_connections = [c for c in self._custom_connections if c["connection_id"] != cid]
        if self._vlm.selected_connection_id == cid:
            self._vlm.selected_connection_id = ""
        self._refresh_custom_list()

    def _selected_custom_id(self) -> str | None:
        item = self.custom_list.currentItem()
        return item.data(1000) if item else None

    def _diagnose_one(self, cid: str) -> None:
        # 通信は UI スレッドで行わない（NFR-002）。ボタンを無効化してワーカーへ。
        if getattr(self, "_diag_thread", None) is not None:
            return
        conn_map = vlm_config.build_connection_map(
            self._vlm, vlm_config.resolve_model_profile(self._vlm))
        conn = conn_map.get(cid)
        if conn is None:
            return
        api_key = vlm_secrets.get_secret(conn.auth.secret_ref) if conn.auth.type != "none" else None

        # report_ready はワーカースレッドから飛ぶ。lambda など QObject でないスロットへ
        # つなぐと Qt が受け手のスレッド親和性を判定できず Direct 接続になり、_show_diag_report
        # （QWidget を作り exec() する）がワーカースレッドで走ってクラッシュする。
        # 必ず QObject のバウンドメソッドへつなぎ、対象 conn は self に持たせる。
        self._diag_pending_conn = conn
        self._diag_pending_profile_id = self._vlm.model_profile_id
        self._diag_thread = QThread(self)
        self._diag_worker = VlmDiagnosticsWorker(conn, api_key)
        self._diag_worker.moveToThread(self._diag_thread)
        self._diag_thread.started.connect(self._diag_worker.run)
        self._diag_worker.report_ready.connect(self._on_diag_report)
        self._diag_worker.finished.connect(self._diag_thread.quit)
        self._diag_thread.finished.connect(self._diag_cleanup)
        self._set_diag_buttons_enabled(False)
        self._diag_thread.start()

    @Slot(object)
    def _on_diag_report(self, report) -> None:
        conn = getattr(self, "_diag_pending_conn", None)
        if conn is None:
            return
        # 通常は本文抽出まで通ったら VERIFIED。429 は認証済みで到達したことだけを
        # 確認済みとして記録し、診断レポート自体は WARN のまま本文成功と区別する。
        if (getattr(report, "can_mark_binding_verified", False)
                and not getattr(conn, "is_custom", True)
                and getattr(conn, "provider_id", "")):
            if vlm_config.mark_binding_verified(
                    self._vlm, conn.provider_id, profile_id=self._diag_pending_profile_id):
                self._persist_immediate_settings()
                cid = next((c for c, r in self._route_rows.items()
                            if r["conn"].provider_id == conn.provider_id), None)
                if cid:
                    self._refresh_route_status(cid)
        self._show_diag_report(conn, report)

    def _open_api_key_dialog(self, cid: str) -> None:
        from api_key_dialog import ApiKeyDialog
        conn_map = vlm_config.build_connection_map(
            self._vlm, vlm_config.resolve_model_profile(self._vlm))
        conn = conn_map.get(cid)
        r = self._route_rows.get(cid)
        if conn is None or r is None:
            return
        info = _PROVIDER_KEY_INFO.get(conn.provider_id, {})
        dlg = ApiKeyDialog(
            self._t,
            display_name=conn.display_name,
            secret_ref=r["secret_ref"] or conn.auth.secret_ref,
            conn=conn,
            key_url=info.get("key_url", ""),
            login_url=info.get("login_url", ""),
            instructions=self._t("Vlm", info.get("instructions_key", "ApiKey_Steps_Generic")),
            cloudflare_account_id=(self._vlm.cloudflare_account_id
                                   if conn.provider_id == "cloudflare" else ""),
            on_cloudflare_verified=(self._on_cloudflare_verified
                                    if conn.provider_id == "cloudflare" else None),
            anthropic_workspace_id=(self._vlm.anthropic_workspace_id
                                    if conn.provider_id == "anthropic" else ""),
            on_anthropic_workspace_saved=(self._on_anthropic_workspace_saved
                                           if conn.provider_id == "anthropic" else None),
            parent=self,
        )
        dlg.exec()
        self._refresh_route_status(cid)

    def _on_cloudflare_verified(self, account_id: str) -> None:
        """接続確認に使えた Account ID を保存する。

        ここを通る確認はモデル一覧GETだけで、プロファイルが実際に使うモデルへは
        到達していない（Cloudflareにそのモデルが存在しなくても、アカウント自体は
        認証を通る）。そのためここでは binding の「検証済み」を立てない。検証済みへ
        昇格させるのは、実際にそのモデルへリクエストを送って確認する「接続診断」
        （フル診断、_on_diag_report）か、実際のキャプション生成成功
        （main_window._on_vlm_binding_verified）のときだけにする。
        """
        self._vlm.cloudflare_account_id = account_id
        self._persist_immediate_settings()

    def _on_anthropic_workspace_saved(self, workspace_id: str) -> None:
        """検証に使えた任意のWorkspace IDを保存する。空は単一Workspaceキーを表す。"""
        self._vlm.anthropic_workspace_id = workspace_id
        self._persist_immediate_settings()

    def _set_diag_buttons_enabled(self, enabled: bool) -> None:
        for r in self._route_rows.values():
            r["diag_btn"].setEnabled(enabled)

    def _has_running_background_work(self) -> bool:
        return any(
            thread is not None and thread.isRunning()
            for thread in (getattr(self, "_diag_thread", None),
                           getattr(self, "_ml_thread", None))
        )

    def _defer_done_until_background_finishes(self, result: int) -> bool:
        """Disconnect UI results and close only after every worker thread exits."""
        if not self._has_running_background_work():
            return False
        if self._pending_done is None:
            self._pending_done = result
        diag_thread = getattr(self, "_diag_thread", None)
        if diag_thread is not None and diag_thread.isRunning():
            try:
                self._diag_worker.report_ready.disconnect()
            except (RuntimeError, TypeError, AttributeError):
                pass
            diag_thread.quit()
        ml_thread = getattr(self, "_ml_thread", None)
        if ml_thread is not None and ml_thread.isRunning():
            try:
                self._ml_worker.result_ready.disconnect()
            except (RuntimeError, TypeError, AttributeError):
                pass
            ml_thread.quit()
        self.setEnabled(False)
        return True

    def _finish_pending_done_if_ready(self) -> None:
        if self._pending_done is None or self._has_running_background_work():
            return
        result = self._pending_done
        self._pending_done = None
        if result != QDialog.DialogCode.Accepted and not self._dialog_saved:
            self._restore_unsaved_vlm()
        QDialog.done(self, result)

    def done(self, r: int) -> None:
        # accept() / reject() 双方の通り道。Close ボタンも X も window X もここを通る。
        if self._defer_done_until_background_finishes(r):
            return
        if r != QDialog.DialogCode.Accepted and not self._dialog_saved:
            self._restore_unsaved_vlm()
        QDialog.done(self, r)

    def closeEvent(self, event) -> None:
        if self._defer_done_until_background_finishes(QDialog.DialogCode.Rejected):
            event.ignore()
            return
        if not self._dialog_saved:
            self._restore_unsaved_vlm()
        super().closeEvent(event)

    def _diag_cleanup(self) -> None:
        if getattr(self, "_diag_worker", None) is not None:
            self._diag_worker.deleteLater()
            self._diag_worker = None
        if getattr(self, "_diag_thread", None) is not None:
            self._diag_thread.deleteLater()
            self._diag_thread = None
        self._diag_pending_conn = None
        self._diag_pending_profile_id = None
        self._set_diag_buttons_enabled(True)
        self._finish_pending_done_if_ready()

    def _show_diag_report(self, conn, report) -> None:
        lines = [f"[{i.status.value}] {i.name}: {i.detail}" for i in report.items]
        if getattr(report, "can_mark_binding_verified", False):
            http_item = report.item("HTTP response")
            extraction_item = report.item("Caption extraction")
            content_verified = (
                not getattr(report, "lightweight", False)
                and http_item is not None and http_item.status is DiagStatus.PASS
                and extraction_item is not None
                and extraction_item.status is DiagStatus.PASS
            )
            summary_key = ("Settings_Diagnose_Content_Verified"
                           if content_verified
                           else "Settings_Diagnose_Reachability_Verified")
            lines.append("[PASS] " + self._t("Vlm", summary_key))
        icon = {DiagStatus.PASS: QMessageBox.Icon.Information,
                DiagStatus.WARN: QMessageBox.Icon.Warning,
                DiagStatus.FAIL: QMessageBox.Icon.Critical}.get(report.overall, QMessageBox.Icon.Information)
        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(self._t("Vlm", "Settings_Diagnose"))
        box.setText(f"{conn.display_name}: {report.overall.value}")
        box.setDetailedText("\n".join(lines))
        box.exec()

    def _on_save(self) -> None:
        v = self._vlm
        custom_mode = self.mode_custom.isChecked()
        selected_custom_id = self.custom_select.currentData() if custom_mode else None
        if custom_mode and not any(
                c.get("connection_id") == selected_custom_id for c in self._custom_connections):
            QMessageBox.warning(self, self._t("Vlm", "Settings_Exec_Mode"),
                                self._t("Vlm", "Settings_Custom_Select_Required"))
            return
        v.model_profile_id = self.profile_combo.currentData() or v.model_profile_id
        v.execution_mode = "custom_single" if custom_mode else "builtin_fallback"
        v.selected_connection_id = (
            (selected_custom_id or "") if custom_mode else v.selected_connection_id)
        v.strict_identity = self.strict_check.isChecked()
        v.prompt_mode = self.prompt_mode_combo.currentData() or "standard"
        v.detail_level = self.detail_combo.currentData()
        v.sentence_mode = self.sentence_combo.currentData()
        v.character_name_mode = self.charname_combo.currentData()
        v.markdown = self.markdown_combo.currentData()
        v.language = self.language_combo.currentData() or "en"
        v.max_output_tokens = int(self.max_tokens.value())
        # 経路の順序＝有効集合。▲▼ で決めた self._route_order のうち、有効チェックが
        # 入っている provider だけを、その順で connection_order に書く。
        enabled_providers = [self._route_rows[cid]["conn"].provider_id
                             for cid in self._route_order
                             if self._route_rows[cid]["enabled"].isChecked()]
        if not custom_mode and not enabled_providers:
            QMessageBox.warning(self, self._t("Vlm", "Settings_Routes"),
                                self._t("Vlm", "Settings_Route_Select_Required"))
            return
        if enabled_providers:
            v.connection_order = ",".join(enabled_providers)
        # API キーは「APIキー登録」ボタン経由で即時保存されるので、ここでは扱わない。

        saved = vlm_config.save_settings_transaction(
            self._custom_connections,
            config_path=app_settings.CONFIG_PATH,
            save_config_callback=lambda: app_settings.save_config(self._settings),
        )
        if not saved:
            QMessageBox.critical(self, self._t("Vlm", "Settings_Title"),
                                 self._t("Vlm", "Settings_Save_Failed"))
            return
        self._dialog_saved = True
        self.accept()

    def _restore_unsaved_vlm(self) -> None:
        """Close/Cancelでダイアログだけの変更を元へ戻す。

        APIキー登録・接続確認は即時保存の操作なので、その結果を保持する。
        """
        immediate = {"verified_bindings", "cloudflare_account_id", "anthropic_workspace_id"}
        ordered_profiles = vlm_config.all_profiles()
        valid_profile_ids = {p.profile_id for p in ordered_profiles}
        fallback_profile_id = self._vlm_before_dialog.model_profile_id
        if fallback_profile_id not in valid_profile_ids:
            fallback_profile_id = next(
                (p.profile_id for p in ordered_profiles), self._vlm.model_profile_id)
        for field in dataclasses.fields(self._vlm):
            if field.name not in immediate:
                value = getattr(self._vlm_before_dialog, field.name)
                if field.name == "model_profile_id":
                    value = fallback_profile_id
                setattr(self._vlm, field.name, value)
        # A diagnostic/API-key callback may have saved the whole in-memory
        # dataclass while regular dialog edits were still pending. Re-save the
        # restored snapshot plus the intentionally retained immediate fields.
        if self._immediate_settings_saved and app_settings.save_config(self._settings):
            self._immediate_settings_saved = False

    def _persist_immediate_settings(self) -> bool:
        saved = app_settings.save_config(self._settings)
        if saved:
            self._immediate_settings_saved = True
        return saved


def _combo(pairs) -> QComboBox:
    c = QComboBox()
    for value, label in pairs:
        c.addItem(label, value)
    return c


def _select(combo: QComboBox, value) -> None:
    idx = combo.findData(str(value))
    combo.setCurrentIndex(idx if idx >= 0 else 0)
