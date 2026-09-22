"""VLM Phase 3+ tests: worker existing-file handling, exhaustion break, dialogs.

Offline only. Run:  rtk pytest tests/test_vlm_phase3.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# isolate config.ini: test_settings_dialog_roundtrip calls VlmSettingsDialog._on_save,
# which persists via save_config(). Without this it overwrites the real config.ini.
import app_settings as _A
_A.CONFIG_PATH = Path(tempfile.mkdtemp()) / "config.ini"
import constants as _C
_C.CONFIG_PATH = _A.CONFIG_PATH
# isolate vlm_connections.json / vlm_profiles.json too (the dialog reads/writes them).
import vlm_config as _VC
_vcdir = Path(tempfile.mkdtemp())
_VC.VLM_CONNECTIONS_PATH = _vcdir / "vlm_connections.json"
_VC.VLM_PROFILES_PATH = _vcdir / "vlm_profiles.json"

from PySide6.QtWidgets import QApplication
from PIL import Image

# QApplication (not QCoreApplication) so the settings-dialog test can build widgets
# in the same process as the worker tests.
_APP = QApplication.instance() or QApplication([])

import vlm_transport as T
from vlm_transport import RawHttpResponse
import vlm_config
import vlm_secrets
import vlm_models as M


def _ok_body(text):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2}}


def _setup(tmpdir, existing_mode, placement, monkeypatch, existing_txt=None):
    import app_settings as A
    import dataclasses
    s = A.load_settings(A.get_default_config())
    s.paths.input_dir = str(tmpdir)
    s.vlm.enabled = True
    # Several tests using this fixture rely on gemini failing to parse the
    # OpenAI-shaped mock body below (gemini's real protocol expects a different
    # JSON shape) and falling over to openrouter, which does parse it - so this
    # fixture needs multiple candidates regardless of the shipped
    # DEFAULT_VLM_CONNECTION_ORDER (now just "gemini").
    s.vlm.connection_order = "gemini,openrouter,cloudflare"
    s.behavior.existing_file_mode = existing_mode
    s.caption.placement = placement
    for i in range(2):
        Image.new("RGB", (8, 8)).save(tmpdir / f"i{i}.png")
    if existing_txt is not None:
        (tmpdir / "i0.txt").write_text(existing_txt, encoding="utf-8")

    verified = {pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED,
                                         provider_constraint=None)
                for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()}
    monkeypatch.setattr(
        vlm_config, "resolve_model_profile",
        lambda v: dataclasses.replace(M.GEMMA_4_26B_A4B_IT, bindings=verified))
    monkeypatch.setattr(vlm_secrets, "get_secret", lambda ref: "FAKEKEY")
    return s


def _run(worker):
    logs = []
    worker.log_message.connect(lambda m, c: logs.append((m, c)))
    worker.run_captioning()
    return logs


def test_append_mode_default_placement_coerced(monkeypatch):
    """existing_file_mode=APPEND but placement stays default OVERWRITE -> must append, not overwrite."""
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "APPEND", "OVERWRITE", monkeypatch, existing_txt="1girl, solo")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("a natural language description"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        _run(VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else "")))
    finally:
        T.execute_http = old
    out = (d / "i0.txt").read_text(encoding="utf-8")
    assert out == "1girl, solo\na natural language description", repr(out)
    print("  APPEND mode + default OVERWRITE placement -> coerced to append: OK")


def test_single_test_saves_to_txt(monkeypatch):
    """The single-image test now writes the .txt (same save path as the batch) and
    reports one FileChange for undo."""
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "OVERWRITE", "OVERWRITE", monkeypatch, existing_txt="old caption")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("fresh vlm caption"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        w = VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else ""),
                             selected_file_path=d / "i0.png", single_test=True)
        changes = []
        w.batch_completed.connect(lambda lst: changes.extend(lst))
        shown = []
        w.single_test_result.connect(lambda cap, c, m: shown.append(cap))
        verified = []
        w.binding_verified.connect(lambda prov, prof: verified.append((prov, prof)))
        w.run_captioning()
    finally:
        T.execute_http = old
    assert (d / "i0.txt").read_text(encoding="utf-8") == "fresh vlm caption"
    assert shown == ["fresh vlm caption"]
    assert len(changes) == 1 and changes[0].previous_content == "old caption"
    # gemini's builtin protocol is gemini_generate_content; the mock returns OpenAI-shaped
    # JSON, so gemini fails to parse and the run succeeds via openrouter.
    assert verified == [("openrouter", "gemma-4-31b-it")], verified
    # the other image is untouched - single test only touches the selected one
    assert not (d / "i1.txt").exists()
    print("  single test writes the selected .txt + one undo FileChange: OK")


def test_single_test_skip_mode_does_not_write(monkeypatch):
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "SKIP", "OVERWRITE", monkeypatch, existing_txt="keep me")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("nope"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        w = VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else ""),
                             selected_file_path=d / "i0.png", single_test=True)
        changes = []
        w.batch_completed.connect(lambda lst: changes.extend(lst))
        w.run_captioning()
    finally:
        T.execute_http = old
    assert (d / "i0.txt").read_text(encoding="utf-8") == "keep me"
    assert changes == []
    print("  single test respects SKIP mode (no write, no undo entry): OK")


def test_skip_mode_leaves_existing(monkeypatch):
    app = _APP
    d = Path(tempfile.mkdtemp())
    s = _setup(d, "SKIP", "OVERWRITE", monkeypatch, existing_txt="keep me")
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(200, {}, _ok_body("new"), "")
    try:
        from vlm_worker import VlmCaptionWorker
        logs = _run(VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else "")))
    finally:
        T.execute_http = old
    assert (d / "i0.txt").read_text(encoding="utf-8") == "keep me"
    assert (d / "i1.txt").read_text(encoding="utf-8") == "new"
    print("  SKIP mode leaves existing .txt untouched: OK")


def test_all_connections_excluded_logs_exhaustion_once(monkeypatch):
    """Every connection returns 401 -> excluded -> the batch keeps going (each
    remaining image gets a fresh live_candidates() check, in case a temporary
    rate-limit cooldown elsewhere clears later - 2026-09 VLM debugging), but
    the exhaustion message is logged only once, not once per remaining image."""
    app = _APP
    d = Path(tempfile.mkdtemp())
    for i in range(2, 30):
        Image.new("RGB", (8, 8)).save(d / f"i{i}.png")
    s = _setup(d, "OVERWRITE", "OVERWRITE", monkeypatch)
    old = T.execute_http
    T.execute_http = lambda req, **kw: RawHttpResponse(401, {}, {"error": {"message": "bad"}}, "unauthorized")
    try:
        from vlm_worker import VlmCaptionWorker
        worker = VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else ""))
        failed_paths = []
        worker.batch_failed.connect(failed_paths.extend)
        logs = _run(worker)
    finally:
        T.execute_http = old
    exhausted = [m for m, c in logs if "All_Connections_Exhausted" in m]
    image_failed = [m for m, c in logs if "Image_Failed" in m]
    assert len(exhausted) == 1, f"exhaustion message must be logged exactly once, got {len(exhausted)}"
    # every image must still end up recorded as failed (not silently dropped) even though
    # the batch keeps looping past the first exhaustion instead of stopping outright.
    # _setup() seeds i0/i1 itself, plus this test's own i2..i29 = 30 images total.
    assert len(failed_paths) == 30, f"all 30 images must be marked failed, got {len(failed_paths)}"
    # first image triggers 3x 401 -> all excluded; subsequent images short-circuit (no per-image error spam)
    assert len(image_failed) <= 1, f"too many per-image error lines: {len(image_failed)}"
    print(f"  all-excluded -> single exhaustion message (image_failed lines: {len(image_failed)}): OK")


def test_settings_dialog_roundtrip(monkeypatch):
    app = _APP
    import app_settings as A
    s = A.load_settings(A.get_default_config())
    T2 = lambda sec, key, **kw: key
    from vlm_settings_dialog import VlmSettingsDialog
    from PySide6.QtCore import Qt
    dlg = VlmSettingsDialog(s, T2)
    assert dlg.profile_combo.currentData() == "gemma-4-31b-it"
    assert [dlg._route_rows[cid]["conn"].provider_id for cid in dlg._route_order[:5]] == [
        "gemini", "openrouter", "cloudflare", "groq", "nvidia"]
    assert dlg._route_rows["builtin-cloudflare"]["name"].text() == "Cloudflare"
    for row in dlg._route_rows.values():
        assert row["status"].width() == 180
        assert row["name"].alignment() & Qt.AlignmentFlag.AlignLeft
    dlg._custom_connections.append({
        "connection_id": "test-custom", "display_name": "Test custom",
        "kind": "custom_local", "protocol": "openai_chat_completions",
        "base_url": "http://127.0.0.1:1234/v1", "model_id": "local-vlm",
        "auth": {"type": "none"},
    })
    dlg._refresh_custom_list()
    dlg.mode_custom.setChecked(True)
    dlg.max_tokens.setValue(1500)
    assert dlg.language_combo.currentData() == "en" and not dlg.language_combo.isEnabled()
    assert not hasattr(dlg, "cf_account_edit")
    assert dlg._route_rows["builtin-huggingface"]["enabled"].isChecked() is False
    for cid in ("builtin-vercel", "builtin-openai", "builtin-anthropic"):
        assert dlg._route_rows[cid]["enabled"].isChecked() is False
    assert "builtin-ovhcloud" not in dlg._route_rows
    dlg._on_cloudflare_verified("fedcba9876543210fedcba9876543210")
    assert dlg.strict_check.isChecked() is False       # default off
    dlg.strict_check.setChecked(True)
    dlg._on_save()
    assert s.vlm.execution_mode == "custom_single"
    assert s.vlm.max_output_tokens == 1500
    assert not hasattr(s.vlm, "free_only")
    assert s.vlm.cloudflare_account_id == "fedcba9876543210fedcba9876543210"
    # Account-id confirmation is a models-list GET only, never a request to the
    # profile's actual bound model - it must not mark that binding verified.
    assert "gemma-4-31b-it:cloudflare" not in s.vlm.verified_set()
    assert s.vlm.language == "en"
    assert s.vlm.strict_identity is True
    dlg._on_anthropic_workspace_saved("wrkspc_test123")
    assert s.vlm.anthropic_workspace_id == "wrkspc_test123"
    assert A.load_settings(A.load_config()).vlm.anthropic_workspace_id == "wrkspc_test123"

    import vlm_config
    assert vlm_config.build_router_policy(s.vlm).allow_declared_identity is False
    s.vlm.strict_identity = False
    assert vlm_config.build_router_policy(s.vlm).allow_declared_identity is True

    # Pin profile-aware resolution for the catalog assertions without leaking it to
    # later tests.
    monkeypatch.setattr(
        vlm_config, "resolve_model_profile",
        lambda v: next(
            (p for p in vlm_config.all_profiles()
             if p.profile_id == v.model_profile_id), None))
    gpt_i = dlg.profile_combo.findData("openai-gpt-5.6-luna")
    dlg.profile_combo.setCurrentIndex(gpt_i)
    assert dlg._route_rows["builtin-openai"]["model_edit"].currentText() == "gpt-5.6-luna"
    assert dlg._route_rows["builtin-openai"]["conn"].protocol == "openai_responses"
    assert dlg._route_rows["builtin-openai"]["enabled"].isChecked() is True
    for pid, model_id in (
        ("openai-gpt-5.6-sol", "gpt-5.6-sol"),
        ("openai-gpt-5.6-terra", "gpt-5.6-terra"),
    ):
        idx = dlg.profile_combo.findData(pid)
        assert idx >= 0
        dlg.profile_combo.setCurrentIndex(idx)
        assert dlg._route_rows["builtin-openai"]["model_edit"].currentText() == model_id
    claude_i = dlg.profile_combo.findData("claude-haiku-4-5")
    dlg.profile_combo.setCurrentIndex(claude_i)
    assert dlg._route_rows["builtin-anthropic"]["model_edit"].currentText() == \
        "claude-haiku-4-5-20251001"
    assert dlg._route_rows["builtin-anthropic"]["conn"].protocol == "anthropic_messages"
    assert dlg._route_rows["builtin-anthropic"]["enabled"].isChecked() is True
    for pid, model_id in (
        ("claude-fable-5-1", "claude-fable-5-1"),
        ("claude-fable-5", "claude-fable-5"),
        ("claude-opus-5", "claude-opus-5"),
        ("claude-opus-4-8", "claude-opus-4-8"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("claude-opus-4-6", "claude-opus-4-6"),
        ("claude-opus-4-5", "claude-opus-4-5-20251101"),
        ("claude-sonnet-5", "claude-sonnet-5"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
    ):
        idx = dlg.profile_combo.findData(pid)
        assert idx >= 0
        dlg.profile_combo.setCurrentIndex(idx)
        assert dlg._route_rows["builtin-anthropic"]["model_edit"].currentText() == model_id
    print("  settings dialog round-trip (incl. strict_identity <-> allow_declared_identity): OK")


def test_settings_dialog_rejects_non_vlm_model():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    s.vlm.model_profile_id = "qwen3.8-27b"
    old_resolver = vlm_config.resolve_model_profile
    vlm_config.resolve_model_profile = lambda v: next(
        (p for p in vlm_config.all_profiles() if p.profile_id == v.model_profile_id), None)
    dlg = None
    try:
        dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
        row = dlg._route_rows["builtin-groq"]
        assert row["enabled"].isEnabled()

        row["model_edit"].setCurrentText("groq/compound-mini")
        dlg._on_model_id_edited("builtin-groq")
        # Reverts to the route's actual bound model_id (fixed 2026-09-22 to
        # include the required "qwen/" prefix - confirmed live against Groq's
        # real catalog).
        assert row["model_edit"].currentText() == "qwen/qwen3.8-27b"
        assert "qwen3.8-27b:groq" not in s.vlm.model_id_override_map()

        dlg._on_model_list("builtin-groq", [
            "groq/compound-mini", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile",
        ])
        assert row["model_ids"] == ["qwen/qwen3.8-27b"]
    finally:
        if dlg is not None:
            dlg.close()
        vlm_config.resolve_model_profile = old_resolver
    print("  settings dialog rejects non-VLM model IDs and filters model list: OK")


def test_custom_connection_persists_routes_and_executes(monkeypatch):
    """The custom dialog result must survive JSON reload and reach the executor."""
    import app_settings as A
    from custom_connection_dialog import CustomConnectionDialog
    from vlm_connections import ConnectionKind
    from vlm_router import select_candidates
    from vlm_transport import VlmExecutor
    from vlm_image import PreparedImage
    from vlm_profiles import GenerationProfile
    import vlm_secrets

    stored_secret = {}
    monkeypatch.setattr(
        vlm_secrets, "set_secret",
        lambda ref, value, **kwargs: (stored_secret.__setitem__(ref, value) or True))
    dialog = CustomConnectionDialog(lambda sec, key, **kw: key)
    dialog.name_edit.setText("Local OpenAI-compatible VLM")
    dialog.locality_combo.setCurrentIndex(dialog.locality_combo.findData("local"))
    dialog.protocol_combo.setCurrentIndex(
        dialog.protocol_combo.findData("openai_chat_completions"))
    dialog.base_url_edit.setText("http://127.0.0.1:1234/v1")
    dialog.model_edit.setCurrentText("local-gemma-vision")
    dialog.auth_type_combo.setCurrentIndex(dialog.auth_type_combo.findData("bearer"))
    dialog.api_key_edit.setText("LOCALKEY")
    dialog._on_save()
    raw = dialog.result_connection()
    assert raw is not None
    assert raw["kind"] == "custom_local"
    assert raw["auth"]["type"] == "bearer"
    assert "api_key" not in raw
    assert stored_secret[raw["auth"]["secret_ref"]] == "LOCALKEY"
    assert _VC.save_custom_connections([raw]) is True

    loaded = _VC.load_custom_connections()
    assert loaded == [raw]
    settings = A.load_settings(A.get_default_config())
    settings.vlm.execution_mode = "custom_single"
    settings.vlm.selected_connection_id = raw["connection_id"]
    connections = _VC.build_connection_map(settings.vlm, M.GEMMA_4_31B_IT)
    conn = connections[raw["connection_id"]]
    assert conn.kind is ConnectionKind.CUSTOM_LOCAL
    assert conn.model_id == "local-gemma-vision"

    policy = _VC.build_router_policy(settings.vlm)
    candidates = select_candidates(
        M.GEMMA_4_31B_IT, connections, policy,
        has_auth={raw["connection_id"]: True})
    assert candidates.connection_ids == [raw["connection_id"]]

    seen = {}
    old_execute = T.execute_http

    def _execute(req, **kwargs):
        seen["request"] = req
        return RawHttpResponse(200, {}, _ok_body("local caption"), "")

    T.execute_http = _execute
    try:
        result = VlmExecutor(
            connections, lambda ref: stored_secret.get(ref)).caption_one({
            "image": PreparedImage(b"\xff\xd8\xff", "image/jpeg"),
            "profile": GenerationProfile(),
            "system_prompt": "system",
            "user_prompt": "user",
        }, candidates.connection_ids)
    finally:
        T.execute_http = old_execute
    assert result.ok and result.text == "local caption"
    assert seen["request"].method == "POST"
    assert seen["request"].url == "http://127.0.0.1:1234/v1/chat/completions"
    assert seen["request"].headers["Authorization"] == "Bearer LOCALKEY"
    assert seen["request"].json_body["model"] == "local-gemma-vision"
    assert isinstance(seen["request"].json_body["messages"][1]["content"], list)
    print("  custom connection: dialog -> JSON reload -> local route -> executor: OK")


def test_external_http_requires_confirmation_before_fetch_or_save(monkeypatch):
    from custom_connection_dialog import CustomConnectionDialog
    from PySide6.QtWidgets import QMessageBox
    from vlm_connections import ConnectionKind
    import custom_connection_dialog as CCD
    import vlm_secrets

    answers = [
        QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
        QMessageBox.StandardButton.No,
    ]
    monkeypatch.setattr(CCD.QMessageBox, "warning", lambda *a, **k: answers.pop(0))
    stored = []
    monkeypatch.setattr(
        vlm_secrets, "set_secret",
        lambda ref, value, **kwargs: (stored.append((ref, value)) or True))

    dialog = CustomConnectionDialog(lambda sec, key, **kw: key)
    dialog.name_edit.setText("Explicit external proxy")
    dialog.locality_combo.setCurrentIndex(dialog.locality_combo.findData("external"))
    dialog.base_url_edit.setText("http://example.com/v1")
    dialog.model_edit.setCurrentText("vision-model")
    dialog.auth_type_combo.setCurrentIndex(dialog.auth_type_combo.findData("bearer"))
    dialog.api_key_edit.setText("SECRET")

    # editingFinished invokes model discovery before Save. Declining must happen
    # before even resolving the entered/stored key or creating a worker thread.
    key_lookups = []
    monkeypatch.setattr(
        dialog, "_model_list_key",
        lambda: (key_lookups.append(True) or "SECRET"))
    dialog._fetch_model_list()
    assert dialog._model_thread is None
    assert key_lookups == []
    assert stored == []

    # Save asks through the same gate. Once accepted, the exact URL is remembered.
    dialog._on_save()
    assert dialog.result_connection() is not None
    assert stored and stored[0][1] == "SECRET"
    assert dialog._confirm_external_http(
        "http://example.com/v1", ConnectionKind.CUSTOM_EXTERNAL) is True

    # Images are sensitive even when the endpoint requires no API key.
    no_auth = CustomConnectionDialog(lambda sec, key, **kw: key)
    no_auth.name_edit.setText("Unauthenticated external proxy")
    no_auth.locality_combo.setCurrentIndex(no_auth.locality_combo.findData("external"))
    no_auth.base_url_edit.setText("http://images.example.com/v1")
    no_auth.model_edit.setCurrentText("vision-model")
    assert no_auth.auth_type_combo.currentData() == "none"
    no_auth._on_save()
    assert no_auth.result_connection() is None
    assert answers == []


def test_settings_close_is_deferred_without_waiting_for_diagnostics():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    class Signal:
        def __init__(self):
            self.disconnected = False

        def disconnect(self):
            self.disconnected = True

    class Worker:
        def __init__(self):
            self.report_ready = Signal()

        def deleteLater(self):
            pass

    class Thread:
        def __init__(self):
            self.running = True
            self.quit_called = False

        def isRunning(self):
            return self.running

        def quit(self):
            self.quit_called = True

        def deleteLater(self):
            pass

    dialog = VlmSettingsDialog(
        A.load_settings(A.get_default_config()), lambda sec, key, **kw: key)
    worker = Worker()
    thread = Thread()
    dialog._diag_worker = worker
    dialog._diag_thread = thread

    dialog.reject()
    assert dialog._pending_done == dialog.DialogCode.Rejected
    assert thread.quit_called and worker.report_ready.disconnected

    thread.running = False
    dialog._diag_cleanup()
    assert dialog._pending_done is None
    assert dialog._diag_thread is None


def test_custom_connection_model_list_keeps_only_vlm_models():
    from custom_connection_dialog import CustomConnectionDialog
    from vlm_model_list import ModelCatalogEntry

    dialog = CustomConnectionDialog(lambda sec, key, **kw: key)
    try:
        dialog.model_edit.setCurrentText("old-model")
        dialog._on_model_list("custom-model-list", [
            ModelCatalogEntry("local-text", False, True, "test"),
            ModelCatalogEntry("local-vlm", True, True, "test"),
        ])
        assert [dialog.model_edit.itemText(i) for i in range(dialog.model_edit.count())] == ["local-vlm"]
        assert dialog.model_edit.currentText() == "local-vlm"
    finally:
        dialog.close()
    print("  custom connection model list filters non-VLM entries: OK")


def test_lightweight_confirmation_does_not_mark_binding_verified():
    """A lightweight API-key check only does a models-list GET - it never sends a
    request to the profile's actual bound model_id, so it cannot prove that model
    works. Registering a key must not silently promote the currently-selected
    profile's binding to VERIFIED on that basis alone (this is exactly the shape
    of bug that hid a broken gemma-4-31b-it -> cloudflare binding: the account's
    key was valid, so the unrelated model binding got marked verified even
    though the model itself did not exist for this account). Only a full
    connection diagnostic that reaches the actual model, or a real successful
    generation, may promote a binding to verified."""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    settings = A.load_settings(A.get_default_config())
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        assert not hasattr(dialog, "_on_api_key_binding_confirmed")
        dialog._on_cloudflare_verified("acct123")
        assert "gemma-4-31b-it:cloudflare" not in settings.vlm.verified_set()
        assert settings.vlm.cloudflare_account_id == "acct123"
    finally:
        dialog.close()
    print("  lightweight/account-id confirmation alone does not mark a binding verified: OK")


def test_settings_dialog_keeps_unbound_route_discoverable():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    s.vlm.model_profile_id = "gemma-4-26b-a4b-it"
    old_resolver = vlm_config.resolve_model_profile
    vlm_config.resolve_model_profile = lambda v: next(
        (p for p in vlm_config.all_profiles() if p.profile_id == v.model_profile_id), None)
    dlg = None
    try:
        dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
        row = dlg._route_rows["builtin-groq"]
        assert not row["enabled"].isEnabled()
        assert row["model_edit"].isEnabled()
        assert row["list_btn"].isEnabled()
        assert row["diag_btn"].isEnabled()

        # A user who manually types/selects a real VLM model id for an unbound
        # route must be able to check it immediately (not stay stuck grayed out
        # until the dialog is reopened) - this is what a user reported after using
        # "Fetch models" on OpenRouter and picking a listed id that stayed
        # unchecked.
        row["model_edit"].setCurrentText("qwen/qwen3.8-27b")
        dlg._on_model_id_edited("builtin-groq")
        assert row["enabled"].isEnabled()
        assert row["has_override"] is True
        # clearing the override back out must re-disable it.
        row["model_edit"].setCurrentText("")
        dlg._on_model_id_edited("builtin-groq")
        assert not row["enabled"].isEnabled()
        for cid, other_provider in (
            ("builtin-nvidia", "nvidia"),
            ("builtin-openai", "openai"),
            ("builtin-anthropic", "anthropic"),
        ):
            other = dlg._route_rows[cid]
            assert other["conn"].provider_id == other_provider
            assert not other["enabled"].isEnabled()
            assert other["model_edit"].isEnabled()
            assert other["list_btn"].isEnabled()
            assert other["register"].isEnabled()
            assert other["diag_btn"].isEnabled()

        dlg._on_model_list("builtin-groq", [
            "groq/compound-mini", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile",
        ])
        assert row["model_ids"] == ["qwen/qwen3.8-27b"]
        # qwen/qwen3.8-27b is already a shipped binding for this provider -> no
        # "new model detected" notice should appear.
        assert "Settings_Route_NewModelsDetected" not in row["status"].toolTip()
        from vlm_model_list import ModelCatalogEntry
        dlg._on_model_list("builtin-groq", [
            ModelCatalogEntry("provider/new-vision", True, True, "live metadata"),
            ModelCatalogEntry("provider/text-only", False, True, "live metadata"),
        ])
        assert row["model_ids"] == ["provider/new-vision"]
        # "provider/new-vision" is not in the shipped catalog for groq -> flagged,
        # but not auto-selected/added (the combo selection assertions below still
        # require an explicit setCurrentText further down).
        assert "Settings_Route_NewModelsDetected" in row["status"].toolTip()
        row["model_edit"].setCurrentText("provider/new-vision")
        dlg._on_model_id_edited("builtin-groq")
        assert vlm_config.build_connection_map(
            s.vlm, vlm_config.resolve_model_profile(s.vlm)
        )["builtin-groq"].model_id == "provider/new-vision"
        dlg._on_model_list("builtin-openai", [
            "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "text-only-model",
        ])
        assert dlg._route_rows["builtin-openai"]["model_ids"] == [
            "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"
        ]
        dlg._on_model_list("builtin-anthropic", [
            "claude-fable-5-1", "claude-fable-5", "claude-opus-5",
            "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
            "claude-opus-4-5-20251101", "claude-sonnet-5", "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
            "text-only-model",
        ])
        assert dlg._route_rows["builtin-anthropic"]["model_ids"] == [
            "claude-fable-5-1", "claude-fable-5", "claude-opus-5",
            "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
            "claude-opus-4-5-20251101", "claude-sonnet-5", "claude-sonnet-4-6",
            "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
        ]
        row["model_edit"].setCurrentText("qwen/qwen3.8-27b")
        dlg._on_model_id_edited("builtin-groq")
        assert vlm_config.build_connection_map(
            s.vlm, vlm_config.resolve_model_profile(s.vlm)
        )["builtin-groq"].model_id == "qwen/qwen3.8-27b"
    finally:
        if dlg is not None:
            dlg.close()
        vlm_config.resolve_model_profile = old_resolver
    print("  unbound route remains available for VLM discovery and diagnosis: OK")


def test_routes_recommended_tab_shows_profile_native_provider():
    """260922_vlm_fallback_ui_candidate_c_plan.md: the "recommended" view must not
    hard-code Gemini - it shows whatever ordered_builtin_provider_ids() resolves
    to for the *current* profile, filtered to routes that actually have a
    binding. For the shipped default profile (gemma-4-31b-it) that is Gemini."""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
    try:
        assert dlg._routes_view_mode == "recommended"
        assert dlg._visible_route_cids() == ["builtin-gemini"]
        for cid, r in dlg._route_rows.items():
            assert r["name"].isHidden() is (cid != "builtin-gemini"), cid
        # up/down reordering is meaningless with ~1 visible row; hidden in this mode.
        assert dlg._route_rows["builtin-gemini"]["updown"].isHidden()
        assert dlg._routes_empty_label.isHidden()
    finally:
        dlg.close()
    print("  recommended tab shows only the default profile's native provider (Gemini): OK")


def test_routes_recommended_tab_follows_profile_switch():
    """Switching the Caption profile combo to a Claude profile must move the
    recommended tab's visible routes to Anthropic (+ its Vercel alias route),
    not leave it stuck on Gemini - this reuses the existing
    ordered_builtin_provider_ids()/_on_profile_changed() wiring, no new
    per-profile logic. Since connection_order (default: just "gemini") has zero
    overlap with claude-opus-5's own bindings, the existing fallback in
    ordered_builtin_provider_ids() switches to *all* of that profile's bound
    providers (anthropic + vercel), not only the first one - this is pre-existing,
    intentional behavior (a newly selected profile becomes fully usable right
    away), not something this UI change introduces."""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
    try:
        idx = dlg.profile_combo.findData("claude-opus-5")
        assert idx >= 0
        dlg.profile_combo.setCurrentIndex(idx)
        assert dlg._visible_route_cids() == ["builtin-anthropic", "builtin-vercel"]
        assert dlg._route_rows["builtin-anthropic"]["name"].isHidden() is False
        assert dlg._route_rows["builtin-gemini"]["name"].isHidden() is True
    finally:
        dlg.close()
    print("  recommended tab follows profile switch (Gemini -> Anthropic): OK")


def test_routes_recommended_tab_empty_for_bindingless_profile():
    """A profile with no builtin bindings at all (custom-connection-only) must
    show the "no built-in connection" message instead of an empty grid or a
    misleading leftover checked-but-grayed row."""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    bindingless = M.VlmModelProfile(
        profile_id="user-custom-only", display_name="Custom only",
        canonical_model_id="custom/model")
    old_resolver = vlm_config.resolve_model_profile
    vlm_config.resolve_model_profile = lambda v: bindingless
    dlg = None
    try:
        dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
        assert dlg._visible_route_cids() == []
        assert dlg._routes_empty_label.isHidden() is False
        assert dlg._routes_empty_label.text() == "Settings_Routes_Recommended_Empty"
    finally:
        if dlg is not None:
            dlg.close()
        vlm_config.resolve_model_profile = old_resolver
    print("  recommended tab shows the no-binding message for a bindingless profile: OK")


def test_routes_all_tab_shows_every_route():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
    try:
        dlg.routes_mode_all.setChecked(True)
        assert dlg._routes_view_mode == "all"
        assert set(dlg._visible_route_cids()) == set(dlg._route_order)
        for cid, r in dlg._route_rows.items():
            assert r["name"].isHidden() is False, cid
        assert dlg._routes_empty_label.isHidden()
    finally:
        dlg.close()
    print("  'show all' tab shows every builtin route: OK")


def test_routes_all_tab_caps_scroll_height_to_about_four_rows():
    """"すべて表示"(最大10行)がダイアログ全体を長くしすぎないよう、経路欄
    だけを約4行分の高さに固定してスクロールさせる(260922のフィードバック:
    「すべて表示の時に縦の長さが長すぎる」)。ダイアログ本体の実際の高さが
    モード切替で伸びるとしても、それは経路欄が(1〜2行想定の「おすすめ」から)
    最大4行分に広がった差分だけで、10行分の高さまでは絶対に伸びない。"""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
    try:
        dlg.show()
        QApplication.instance().processEvents()
        dialog_height_before = dlg.height()
        recommended_scroll_height = dlg._routes_scroll.height()

        dlg.routes_mode_all.setChecked(True)
        QApplication.instance().processEvents()
        all_scroll_height = dlg._routes_scroll.height()
        # 全10行がそのまま並んだ高さより明確に小さい(スクロールが必要になる)。
        full_grid_height = dlg._routes_grid.sizeHint().height()
        assert all_scroll_height < full_grid_height
        # おおむね4行分程度(1行あたりの高さ*4 + 余白)に収まっている。
        per_row = full_grid_height / len(dlg._route_order)
        assert all_scroll_height <= per_row * 4 + 16
        # ダイアログ本体が伸びるとしても、経路欄の伸び幅(おすすめ→すべて表示)を
        # 超えては伸びない(=10行分の高さまで際限なく伸びるバグの再発防止)。
        scroll_height_delta = all_scroll_height - recommended_scroll_height
        assert dlg.height() <= dialog_height_before + scroll_height_delta + 16
        assert all_scroll_height >= recommended_scroll_height
    finally:
        dlg.close()
    print("  'show all' tab caps the routes area to ~4 rows with its own scrollbar"
          " instead of growing the whole dialog: OK")


def test_routes_recommended_tab_updates_when_checkbox_toggled():
    """Unchecking the sole visible route in the recommended tab must make it
    disappear from that view immediately (switch to the empty message), not
    require reopening the dialog or switching tabs."""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
    try:
        assert dlg._visible_route_cids() == ["builtin-gemini"]
        dlg._route_rows["builtin-gemini"]["enabled"].setChecked(False)
        assert dlg._visible_route_cids() == []
        assert dlg._routes_empty_label.isHidden() is False
    finally:
        dlg.close()
    print("  unchecking the only recommended route switches to the empty message live: OK")


def test_routes_recommended_tab_shows_checked_override_only_route():
    """A route with no binding for the current profile but an explicit,
    user-chosen model id override (set via "Fetch models" + picking an id, see
    test_settings_dialog_keeps_unbound_route_discoverable) must appear in the
    recommended tab once checked, not stay invisible forever just because it
    has no binding. Reported by a user: checking OpenRouter after choosing a
    model id via "fetch models" never made it show up under "recommended"."""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    s = A.load_settings(A.get_default_config())
    s.vlm.model_profile_id = "gemma-4-26b-a4b-it"
    old_resolver = vlm_config.resolve_model_profile
    vlm_config.resolve_model_profile = lambda v: next(
        (p for p in vlm_config.all_profiles() if p.profile_id == v.model_profile_id), None)
    dlg = None
    try:
        dlg = VlmSettingsDialog(s, lambda sec, key, **kw: key)
        row = dlg._route_rows["builtin-groq"]
        assert not row["has_binding"]
        row["model_edit"].setCurrentText("qwen/qwen3.8-27b")
        dlg._on_model_id_edited("builtin-groq")
        assert row["enabled"].isEnabled()
        assert "builtin-groq" not in dlg._visible_route_cids()  # not checked yet
        row["enabled"].setChecked(True)
        assert "builtin-groq" in dlg._visible_route_cids()
        assert row["name"].isHidden() is False
    finally:
        if dlg is not None:
            dlg.close()
        vlm_config.resolve_model_profile = old_resolver
    print("  recommended tab shows a checked override-only route: OK")


def test_settings_transaction_rolls_back_both_files(tmp_path, monkeypatch):
    connections_path = tmp_path / "vlm_connections.json"
    config_path = tmp_path / "config.ini"
    connections_path.write_text("old-connections", encoding="utf-8")
    config_path.write_text("old-config", encoding="utf-8")
    monkeypatch.setattr(vlm_config, "VLM_CONNECTIONS_PATH", connections_path)

    def fail_after_partial_config_write():
        config_path.write_text("partial-new-config", encoding="utf-8")
        return False

    assert not vlm_config.save_settings_transaction(
        [{"connection_id": "new"}], config_path=config_path,
        save_config_callback=fail_after_partial_config_write)
    assert connections_path.read_text(encoding="utf-8") == "old-connections"
    assert config_path.read_text(encoding="utf-8") == "old-config"


def test_cancel_repersists_regular_fields_after_immediate_confirmation(tmp_path, monkeypatch):
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(A, "CONFIG_PATH", config_path)
    settings = A.load_settings(A.get_default_config())
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        assert settings.vlm.strict_identity is False
        settings.vlm.strict_identity = True  # representative unsaved regular edit
        # _on_cloudflare_verified persists immediately (account id), same as the
        # old binding-confirmation path used to - a convenient trigger for
        # "an immediate write already happened" independent of the regular edit above.
        dialog._on_cloudflare_verified("acct123")
        assert A.load_settings(A.load_config()).vlm.strict_identity is True
        assert A.load_settings(A.load_config()).vlm.cloudflare_account_id == "acct123"

        dialog._restore_unsaved_vlm()
        reloaded = A.load_settings(A.load_config()).vlm
        assert settings.vlm.strict_identity is False
        assert reloaded.strict_identity is False
        assert reloaded.cloudflare_account_id == "acct123"
    finally:
        dialog.close()


def test_deleted_selected_profile_is_not_restored_on_cancel(tmp_path, monkeypatch):
    import app_settings as A
    from PySide6.QtWidgets import QMessageBox
    from vlm_settings_dialog import VlmSettingsDialog

    monkeypatch.setattr(A, "CONFIG_PATH", tmp_path / "config.ini")
    monkeypatch.setattr(vlm_config, "VLM_PROFILES_PATH", tmp_path / "vlm_profiles.json")
    assert vlm_config.save_user_profiles([{
        "profile_id": "user-delete-me", "display_name": "Delete me",
        "canonical_model_id": "vendor/custom-vlm",
        "bindings": {"groq": {"model_id": "vendor/custom-vlm", "vlm_capable": True}},
    }])
    settings = A.load_settings(A.get_default_config())
    settings.vlm.model_profile_id = "user-delete-me"
    assert A.save_config(settings)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        dialog._del_profile()
        assert not vlm_config.is_user_profile("user-delete-me")
        assert settings.vlm.model_profile_id != "user-delete-me"
        dialog._restore_unsaved_vlm()
        assert settings.vlm.model_profile_id != "user-delete-me"
        assert A.load_settings(A.load_config()).vlm.model_profile_id != "user-delete-me"
    finally:
        dialog.close()


def test_profile_editor_preserves_legacy_catalog_validated_binding():
    from vlm_profile_editor import ProfileEditorDialog

    dialog = ProfileEditorDialog(lambda sec, key, **kw: key, {
        "profile_id": "user-legacy", "display_name": "Legacy",
        "canonical_model_id": "vendor/custom-vlm",
        "bindings": {"groq": {"model_id": "vendor/custom-vlm"}},
    })
    dialog._on_save()
    result = dialog.result_profile()
    assert result is not None
    assert result["bindings"]["groq"] == {
        "model_id": "vendor/custom-vlm", "vlm_capable": True}


def test_cancel_uses_first_ordered_profile_for_missing_snapshot():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    settings = A.load_settings(A.get_default_config())
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        expected = vlm_config.all_profiles()[0].profile_id
        dialog._vlm_before_dialog.model_profile_id = "missing-profile"
        settings.vlm.model_profile_id = "also-missing"
        dialog._restore_unsaved_vlm()
        assert settings.vlm.model_profile_id == expected
    finally:
        dialog.close()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
