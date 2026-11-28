"""Phase F（§12 / REQ-F1〜F9）の検証テスト。

VLM の「どのモデルに送るか」「検証済みと印すか」の判定を固定する。
ネットワーク・実 keyring・実 .env・実 config.ini は使わない（§15.2）。

総括ファイルに具体例のあるケース（REQ-F1、F2）は、テスト名から追えるように
指摘 ID・プロファイル ID・候補 ID をテスト名へ入れている（§15.3）。
"""
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from PIL import Image

import constants
import app_settings as A
import utils
import vlm_config
import vlm_diagnostics as D
import vlm_model_list as ML
import vlm_models as M
import vlm_router as R
import vlm_secrets
from vlm_connections import default_builtin_connections
from vlm_protocols import get_protocol
from vlm_transport import RawHttpResponse


def _isolate_config(tmp_path: Path, monkeypatch) -> None:
    """実 config.ini / VLM JSON を読まないよう、全パスを一時ディレクトリへ向ける。"""
    cfg = tmp_path / "config.ini"
    monkeypatch.setattr(A, "CONFIG_PATH", cfg)
    monkeypatch.setattr(constants, "CONFIG_PATH", cfg)
    monkeypatch.setattr(utils, "CONFIG_PATH", cfg)
    monkeypatch.setattr(vlm_config, "VLM_CONNECTIONS_PATH", tmp_path / "vlm_connections.json")
    monkeypatch.setattr(vlm_config, "VLM_PROFILES_PATH", tmp_path / "vlm_profiles.json")


def _settings(tmp_path: Path, monkeypatch):
    _isolate_config(tmp_path, monkeypatch)
    return A.load_settings(A.get_default_config())


# --- REQ-F1（R10-6）match_model_id のサイズガード ---------------------------------


def test_req_f1_r10_6_gemma_4_31b_profile_does_not_match_gemma_4_26b_a4b_it():
    """総括 R10-6 の具体例: GEMMA_4_31B_IT に対し "google/gemma-4-26b-a4b-it" は一致しない。

    世代トークン "4" だけを共有する別サイズの兄弟が、ペナルティなしで
    「31B プロファイルの一致」として返っていた（呼び出し側は自動適用する）。
    """
    profile = M.GEMMA_4_31B_IT
    for provider in ("openrouter", "gemini", "vercel"):
        best, score = M.match_model_id(profile, provider, ["google/gemma-4-26b-a4b-it"])
        assert best is None, (provider, best, score)
        assert score < 0.34, (provider, score)
    # 同じ 31B 同士は従来どおり一致する。
    best, score = M.match_model_id(profile, "gemini", ["gemma-4-31b-it"])
    assert best == "gemma-4-31b-it" and score == 1.0


def test_req_f1_r10_6_opus_4_5_profile_does_not_match_opus_4_8():
    """総括 R10-6 の具体例: Opus 4.5 の参照 {4,5} と opus-4-8 の {4,8} は一致しない。"""
    profile = M.CLAUDE_OPUS_4_5
    best, score = M.match_model_id(profile, "anthropic", ["claude-opus-4-8"])
    assert best is None, (best, score)
    assert score < 0.34, score

    # 日付様トークン（8桁）はサイズ判定から除外する。canonical の 20251101 を持つ
    # プロファイルに、日付なし表記の同一モデル候補は従来どおり一致する。
    best, _ = M.match_model_id(profile, "anthropic", ["anthropic/claude-opus-4.5"])
    assert best == "anthropic/claude-opus-4.5"
    same = M.default_registry().get("claude-opus-4-5")
    best, score = M.match_model_id(same, "anthropic", ["claude-opus-4-5"])
    assert best == "claude-opus-4-5"


def test_req_f1_size_guard_passes_equal_sets_and_models_without_size_tokens():
    """数字トークン集合が一致する候補と、双方とも数字を持たない候補は通す。"""
    numbers = M.VlmModelProfile(
        profile_id="test-numbers", display_name="n", canonical_model_id="vendor/vision-27b",
        bindings={"p": M.ModelBinding("p", "vendor/vision-27b")})
    assert M.looks_same_family(numbers, "vendor/vision-27b-instruct") is True
    assert M.looks_same_family(numbers, "vendor/vision-12b-instruct") is False

    no_numbers = M.VlmModelProfile(
        profile_id="test-nonumbers", display_name="n", canonical_model_id="vendor/vision",
        bindings={"p": M.ModelBinding("p", "vendor/vision")})
    # 双方とも数字トークンを持たない → 集合が共に空で一致 → ガード通過。
    assert M._size_tokens_agree(M._sig_tokens("vendor/vision"),
                               M._sig_tokens("vendor/vision")) is True
    assert M.looks_same_family(no_numbers, "vendor/vision") is True
    # 一方だけが数字を持つ場合はサイズ違いとして弾く。
    assert M._size_tokens_agree(M._sig_tokens("vendor/vision"),
                               M._sig_tokens("vendor/vision-2")) is False
    assert M.looks_same_family(no_numbers, "vendor/vision-2") is False


# --- REQ-F2（R10-7）兄弟モデルの除外の範囲 ---------------------------------------


def test_req_f2_r10_7_qwen3_6_27b_is_not_usable_for_qwen3_8_27b_on_groq():
    """総括 R10-7 の具体例: "qwen/qwen3.6-27b" が QWEN3_8_27B の groq 向けに通らない。

    QWEN3_6_27B は groq binding を持たないため、binding を持つ兄弟だけを見る
    旧実装ではカタログ認識から漏れ、looks_same_family() の弱い数値ガードで
    「使用可能」として通っていた。
    """
    profile = M.QWEN3_8_27B
    assert M.looks_same_family(profile, "qwen/qwen3.6-27b") is False
    assert M.is_vlm_model_id(profile, "groq", "qwen/qwen3.6-27b") is False
    assert M.match_model_id(profile, "groq", ["qwen/qwen3.6-27b"])[0] is None
    # 正しい binding は従来どおり通る。
    assert M.is_vlm_model_id(profile, "groq", "qwen/qwen3.8-27b") is True


def test_req_f2_sibling_recognition_uses_the_whole_profile_catalog():
    """兄弟の認識は binding の有無に依らず、プロファイルカタログ全体から行う。"""
    # QWEN3_6_27B は groq binding を持たないが、既知モデルとして認識される。
    assert M.QWEN3_6_27B.binding_for("groq") is None
    known = M._known_profile_for_model("groq", "qwen/qwen3.6-27b")
    assert known is not None and known.profile_id == M.QWEN3_6_27B.profile_id
    # 既知モデルでない ID は従来どおり「未知」のまま。
    assert M._known_profile_for_model("groq", "qwen/qwen9.9-99b") is None


def test_req_f2_looks_same_family_rejects_other_generation_and_size():
    """looks_same_family 自身の数値ガードも「集合の一致」へ揃える。"""
    gemma26 = M.GEMMA_4_26B_A4B_IT
    assert M.looks_same_family(gemma26, "google/gemma-4-26b-a4b-it:free") is True
    assert M.looks_same_family(gemma26, "gemma-3-27b-it") is False      # 別サイズ・別世代
    assert M.looks_same_family(M.GEMMA_4_31B_IT, "@cf/google/gemma-4-26b-a4b-it") is False
    assert M.looks_same_family(gemma26, "gpt-4o") is False


# --- REQ-F3（R19-4）match_model_id の同点処理の決定性 -----------------------------


def test_req_f3_match_result_does_not_depend_on_candidate_order():
    """候補の順序を入れ替えても一致結果が変わらない（返却順に依存しない）。"""
    profile = M.QWEN3_8_27B
    candidates = ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "qwen/qwen3.8-27b-instruct"]
    first = M.match_model_id(profile, "openrouter", candidates)
    assert first[0] == "qwen/qwen3.8-27b"
    assert M.match_model_id(profile, "openrouter", list(reversed(candidates))) == first

    # 大小文字だけ違う重複は、辞書順で先の表記を代表にして決定的に潰す（後勝ちに
    # ならない）。どちらの入力順でも同じ代表が返る。
    gemma = M.GEMMA_4_26B_A4B_IT
    dups = ["GOOGLE/Gemma-4-26b-A4B-it", "google/gemma-4-26b-a4b-it"]
    results = {M.match_model_id(gemma, "openrouter", order) for order in
               (dups, list(reversed(dups)))}
    assert len(results) == 1, results
    assert results.pop()[0] == "GOOGLE/Gemma-4-26b-A4B-it"


# --- REQ-F4（R18-3）binding 検証の記録先プロファイル -------------------------------


def _fake_vlm_runtime(profile_id: str, on_caption=None):
    """`_build_runtime` の戻り値を差し替えるための最小ダブル。"""
    from vlm_image import ImagePreprocessConfig

    connections = {"builtin-gemini": SimpleNamespace(
        connection_id="builtin-gemini", provider_id="gemini", is_custom=False,
        display_name="Gemini API")}

    def _caption_one(spec_base, ids):
        if on_caption is not None:
            on_caption()
        return SimpleNamespace(ok=True, text="hello world",
                               connection_id="builtin-gemini", model_id="m1",
                               stopped=False, stop_job=False, attempts=[], error=None)

    executor = SimpleNamespace(live_candidates=lambda ids: ids, caption_one=_caption_one)
    candidates = SimpleNamespace(connection_ids=["builtin-gemini"], has_candidates=True,
                                 excluded={})
    return {
        "connections": connections, "gen_profile": None, "policy": None,
        "candidates": candidates, "executor": executor,
        "model_profile_id": profile_id,
        "system_prompt": "", "user_prompt": "",
        "image_cfg": ImagePreprocessConfig(max_long_edge=512, fmt="JPEG", jpeg_quality=80),
    }


def test_req_f4_binding_verified_is_recorded_for_the_start_snapshot_profile(
        tmp_path, monkeypatch):
    """バッチ途中でプロファイルが切り替わっても、記録先は開始時のスナップショット。"""
    from vlm_worker import VlmCaptionWorker

    d = tmp_path / "imgs"
    d.mkdir()
    Image.new("RGB", (8, 8)).save(d / "a.png")

    s = _settings(tmp_path, monkeypatch)
    s.paths.input_dir = str(d)
    s.behavior.existing_file_mode = "OVERWRITE"
    s.caption.placement = "OVERWRITE"
    s.vlm.model_profile_id = "gemma-4-31b-it"

    def _switch_profile():
        # モードレスの設定ダイアログでバッチ途中に保存された状況を模す。
        s.vlm.model_profile_id = "claude-opus-4-5"

    rt = _fake_vlm_runtime("gemma-4-31b-it", on_caption=_switch_profile)
    monkeypatch.setattr(VlmCaptionWorker, "_build_runtime", lambda self: rt)

    w = VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else ""))
    verified = []
    w.binding_verified.connect(lambda provider, profile: verified.append((provider, profile)))
    w.run_captioning()

    assert verified == [("gemini", "gemma-4-31b-it")], verified


# --- REQ-F5（R22-1）can_mark_binding_verified の到達可能性例外 --------------------


def _extraction_item(body, *, protocol="gemini_generate_content"):
    status, detail, *_ = D._classify_extraction(
        RawHttpResponse(200, {}, body, "{}"), get_protocol(protocol))
    return status, detail


def test_req_f5_truncated_diagnostic_marks_binding_verified():
    """診断トークン上限で切れた 200 は、実際の詳細文で到達確認として扱う。

    旧実装は "endpoint reachable" を探していたが、このケースで生成される詳細は
    "endpoint is reachable, but ..." であり、例外がデッドだった。
    """
    status, detail = _extraction_item({"candidates": [{"finishReason": "MAX_TOKENS"}]})
    assert status is D.DiagStatus.WARN
    assert "endpoint is reachable" in detail.lower(), detail

    rep = D.DiagReport("builtin-gemini")
    rep.add("HTTP response", D.DiagStatus.PASS, "200 OK")
    rep.add("Caption extraction", status, detail)
    rep.http_status = 200
    assert rep.can_mark_binding_verified is True

    # content policy 等の一般 WARN は到達確認に含めない（従来どおり False）。
    status_cp, detail_cp = _extraction_item(
        {"promptFeedback": {"blockReason": "SAFETY"},
         "candidates": [{"finishReason": "SAFETY"}]})
    assert status_cp is D.DiagStatus.WARN
    blocked = D.DiagReport("builtin-gemini")
    blocked.add("HTTP response", D.DiagStatus.PASS, "200 OK")
    blocked.add("Caption extraction", status_cp, detail_cp)
    blocked.http_status = 200
    assert blocked.can_mark_binding_verified is False


# --- REQ-F6（R10-9）死んだ能力判定の整理（挙動不変） ------------------------------


def test_req_f6_router_has_no_dead_image_capability_exclusion():
    """router 側の死んだ no_image_support 除外は削除し、能力判定はモデル ID 層のみ。"""
    params = inspect.signature(R.select_candidates).parameters
    assert "supports_image" not in params
    assert "no_image_support" not in R.EXCLUSION_REASON_KEYS
    # 既存のルーティング（能力判定を除いた候補選定）は従来どおり動く。
    profile = M.GEMMA_4_26B_A4B_IT
    conns = {c.connection_id: c for c in default_builtin_connections()}
    candidates = R.select_candidates(profile, conns, R.RouterPolicy(),
                                     has_auth={cid: True for cid in conns})
    assert "builtin-gemini" in candidates.connection_ids
    assert "no_image_support" not in candidates.excluded.values()


# --- REQ-F7（R10-10）router の候補順序 -------------------------------------------


def test_req_f7_candidate_order_follows_connections_not_bindings():
    """router 単体でも connections の並びを尊重する（呼び出し側の事前ソート不要）。"""
    import dataclasses

    profile = dataclasses.replace(M.GEMMA_4_26B_A4B_IT, bindings={
        pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED,
                                 provider_constraint=None)
        for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()})
    # bindings の宣言順は gemini → openrouter。connections 側で逆に並べれば逆順になる。
    assert list(profile.bindings)[:2] == ["gemini", "openrouter"]
    conns = {c.connection_id: c for c in default_builtin_connections()}
    reordered = R.order_connections_by_provider(conns, ["openrouter", "gemini"])
    auth = {cid: True for cid in conns}
    candidates = R.select_candidates(profile, reordered, R.RouterPolicy(), has_auth=auth)
    assert candidates.connection_ids[:2] == ["builtin-openrouter", "builtin-gemini"]
    # 並べ替えを渡さなければ、従来どおり bindings の宣言順（テンプレート順）のまま。
    plain = R.select_candidates(profile, conns, R.RouterPolicy(), has_auth=auth)
    assert plain.connection_ids[:2] == ["builtin-gemini", "builtin-openrouter"]


def test_req_f7_worker_orders_candidates_by_connection_order(tmp_path, monkeypatch):
    """実行経路の候補順は connection_order（UI の経路順）で決まる。"""
    import dataclasses

    from vlm_worker import VlmCaptionWorker

    s = _settings(tmp_path, monkeypatch)
    s.vlm.enabled = True
    s.vlm.connection_order = "openrouter,gemini"
    verified = {pid: dataclasses.replace(b, identity_status=M.ModelIdentityStatus.VERIFIED,
                                        provider_constraint=None)
                for pid, b in M.GEMMA_4_26B_A4B_IT.bindings.items()}
    monkeypatch.setattr(vlm_config, "resolve_model_profile",
                        lambda v: dataclasses.replace(M.GEMMA_4_26B_A4B_IT, bindings=verified))
    monkeypatch.setattr(vlm_secrets, "get_secret", lambda ref: "FAKEKEY")

    w = VlmCaptionWorker(s, get_string=lambda *a, **k: (a[-1] if a else ""))
    rt = w._build_runtime()
    assert rt is not None
    assert rt["candidates"].connection_ids == ["builtin-openrouter", "builtin-gemini"]

    s.vlm.connection_order = "gemini,openrouter"
    rt2 = w._build_runtime()
    assert rt2["candidates"].connection_ids == ["builtin-gemini", "builtin-openrouter"]


# --- REQ-F8（R18-5）xai の環境変数別名 --------------------------------------------


def _clear_xai_env(monkeypatch):
    """XAI_API_KEY を空にし、終了時に元の値へ戻す（後続テストへ漏らさない）。"""
    original = os.environ.pop("XAI_API_KEY", None)
    monkeypatch.delenv("PIXAI_VLM_XAI_API_KEY", raising=False)
    saved_session = dict(vlm_secrets._session_store)
    vlm_secrets._session_store.clear()

    def _restore():
        os.environ.pop("XAI_API_KEY", None)
        if original is not None:
            os.environ["XAI_API_KEY"] = original
        vlm_secrets._session_store.clear()
        vlm_secrets._session_store.update(saved_session)

    return _restore


def test_req_f8_xai_api_key_resolves_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm_secrets, "keyring", None)   # 実 keyring を触らない
    restore = _clear_xai_env(monkeypatch)
    try:
        monkeypatch.setenv("XAI_API_KEY", "xai-env-key")
        assert "XAI_API_KEY" in vlm_secrets._ENV_ALIASES["vlm/xai/api_key"]
        assert vlm_secrets.get_secret("vlm/xai/api_key") == "xai-env-key"
    finally:
        restore()


def test_req_f8_xai_api_key_resolves_from_dotenv(tmp_path, monkeypatch):
    """`.env` 経由でも XAI_API_KEY が効く（一時ディレクトリのみを読む）。"""
    monkeypatch.setattr(vlm_secrets, "keyring", None)
    restore = _clear_xai_env(monkeypatch)
    base = tmp_path / "base"
    base.mkdir()
    (base / ".env").write_text("XAI_API_KEY=from-dotenv\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    # BASE_DIR/.env の実ファイルを読まないよう、定数側を一時ディレクトリへ差し替える。
    monkeypatch.setattr(constants, "BASE_DIR", base)
    old_cwd = os.getcwd()
    try:
        os.chdir(work)
        vlm_secrets._load_dotenv()
        assert os.environ.get("XAI_API_KEY") == "from-dotenv"
        assert vlm_secrets.get_secret("vlm/xai/api_key") == "from-dotenv"
    finally:
        os.chdir(old_cwd)
        restore()


def test_req_f8_provider_enumerations_agree():
    """プロバイダ集合の列挙箇所（4箇所＋KNOWN）が一致する（再発防止）。"""
    import vlm_profile_editor
    import vlm_settings_dialog

    builtin = {c.provider_id for c in default_builtin_connections()}
    assert set(vlm_config.KNOWN_BUILTIN_PROVIDERS) == builtin
    assert set(vlm_settings_dialog._PROVIDER_KEY_INFO) == builtin
    assert set(vlm_profile_editor._PROVIDERS) == builtin
    assert set(vlm_profile_editor._SECRET_REF) == builtin

    # 内蔵接続の secret_ref は必ず環境変数別名を持つ（mistral は停止中の経路なので除外）。
    secret_refs = {c.auth.secret_ref for c in default_builtin_connections() if c.auth.secret_ref}
    assert secret_refs <= set(vlm_secrets._ENV_ALIASES)
    assert set(vlm_secrets._ENV_ALIASES) - secret_refs == {"vlm/mistral/api_key"}
    # プロファイルエディタの secret_ref 対応表も同じ値を使う。
    assert {ref for ref in vlm_profile_editor._SECRET_REF.values()} == secret_refs


# --- REQ-F9（R22-3）デッドな互換 API の削除 ---------------------------------------


def test_req_f9_dead_compat_model_list_api_is_removed():
    """fetch_model_ids／_extract_ids は削除され、src 内に参照が残らない。"""
    assert not hasattr(ML, "fetch_model_ids")
    assert not hasattr(ML, "_extract_ids")
    src = Path(__file__).resolve().parent.parent / "src"
    for path in src.glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert "fetch_model_ids" not in text, path
        assert "_extract_ids" not in text, path
