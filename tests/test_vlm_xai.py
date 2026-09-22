"""xAI Grok 内蔵 VLM 経路のオフラインテスト。

- 内蔵接続テンプレート（Responses API / bearer / api.x.ai）
- 出荷プロファイル（grok-4-6）と xai binding
- モデル能力判定（grok-4 系は VLM、grok-3 / grok-2 素体・code/build は非 VLM）
- /v1/language-models の {"models":[{id,input_modalities}]} 解析
- build_connection_map で Grok プロファイル選択時に xai 経路が有効化される
- Responses プロトコルのリクエスト形（endpoint / instructions / input_image 文字列）

実行: rtk pytest tests/test_vlm_xai.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import vlm_config
import vlm_connections
import vlm_model_list
import vlm_models
from app_settings import Vlm
from vlm_models import classify_model_capability
from vlm_profiles import GenerationProfile
from vlm_protocols import VlmCallSpec, get_protocol


class _FakeImage:
    data_url = "data:image/jpeg;base64,QUFBQQ=="
    base64 = "QUFBQQ=="
    mime_type = "image/jpeg"


def test_builtin_xai_connection_template() -> None:
    conns = {c.connection_id: c for c in vlm_connections.default_builtin_connections()}
    x = conns["builtin-xai"]
    assert x.provider_id == "xai"
    assert x.protocol == "openai_responses"
    assert x.base_url == "https://api.x.ai/v1"
    assert x.auth.type == "bearer"
    assert x.auth.secret_ref == "vlm/xai/api_key"
    assert x.model_id == ""  # 選択プロファイルの binding から補完する
    assert "xai" in vlm_config.KNOWN_BUILTIN_PROVIDERS


def test_shipped_grok_profiles() -> None:
    profiles = {p.profile_id: p for p in vlm_models.default_registry().all_profiles()}
    # grok-4-3 was removed: an older model priced the same as grok-4-6
    # (2026-09-22 user decision).
    assert "grok-4-6" in profiles
    assert "grok-4-3" not in profiles
    g = profiles["grok-4-6"]
    assert g.canonical_model_id == "grok-4.6"
    assert g.binding_for("xai").model_id == "grok-4.6"
    assert g.quantization_is_strict()  # provider_managed
    # Grok は今のところ xai 直販のみを内蔵経路にする
    assert set(g.bindings) == {"xai"}


def test_grok_capability_classification() -> None:
    for mid in ("grok-4.6", "grok-4.5", "grok-4.3", "grok-4-fast",
                "grok-4.20-multi-agent-0309", "grok-2-vision-1212", "grok-vision-beta"):
        ok, why = classify_model_capability("xai", mid)
        assert ok is True, (mid, why)
    for mid in ("grok-3", "grok-3-mini", "grok-3-fast", "grok-2", "grok-2-1212",
                "grok-beta", "grok-code-fast-1", "grok-build-0.1"):
        ok, why = classify_model_capability("xai", mid)
        assert ok is False, (mid, why)


def test_language_models_catalog_parsing() -> None:
    body = {
        "models": [
            {"id": "grok-4.6", "input_modalities": ["text", "image"],
             "output_modalities": ["text"], "aliases": ["latest"]},
            {"id": "grok-4.3", "input_modalities": ["text", "image"],
             "output_modalities": ["text"]},
            {"id": "grok-3", "input_modalities": ["text"], "output_modalities": ["text"]},
            {"id": "grok-2-image-1212", "input_modalities": ["text"],
             "output_modalities": ["image"]},
        ]
    }
    catalog = vlm_model_list._extract_catalog(body, "xai")
    by_id = {e.model_id: e for e in catalog}
    assert by_id["grok-4.6"].is_vlm is True
    assert by_id["grok-4.3"].is_vlm is True
    assert by_id["grok-3"].is_vlm is False
    assert by_id["grok-2-image-1212"].is_vlm is False  # 画像出力モデルは弾く
    vlm_ids = [e.model_id for e in vlm_model_list.filter_vlm_catalog(catalog)]
    assert vlm_ids == ["grok-4.6", "grok-4.3"]
    assert vlm_model_list._extract_ids(body) == [
        "grok-4.6", "grok-4.3", "grok-3", "grok-2-image-1212"]


def test_prompt_image_token_price_signals_vlm() -> None:
    # xAI の /v1/models は input_modalities を持たないが画像入力の課金単価を返す。
    ok, why = classify_model_capability(
        "xai", "grok-4.6",
        {"prompt_text_token_price": 20000, "prompt_image_token_price": 20000})
    assert ok is True, why
    # 画像単価があっても出力がテキストでなければ弾く。
    off, _ = classify_model_capability(
        "xai", "some-image-gen",
        {"prompt_image_token_price": 10, "output_modalities": ["image"]})
    assert off is False
    # 明示的な capabilities.vision=false は画像単価より優先される。
    neg, why = classify_model_capability(
        "xai", "grok-textish",
        {"prompt_image_token_price": 20000, "capabilities": {"vision": False}})
    assert neg is False, why
    # 単価ゼロ／未指定は判定材料にしない（静的判定へ流れる）。
    none_price, _ = classify_model_capability(
        "xai", "mystery-model", {"prompt_image_token_price": 0})
    assert none_price is None


def test_language_models_url_selection() -> None:
    conn = {c.connection_id: c for c in vlm_connections.default_builtin_connections()}["builtin-xai"]
    captured: dict[str, str] = {}

    def fake_execute_http(req, **_kw):
        captured["url"] = req.url
        raise RuntimeError("stop after url capture")

    orig = vlm_model_list.execute_http
    vlm_model_list.execute_http = fake_execute_http
    try:
        try:
            vlm_model_list._fetch_model_body(conn, "xai-KEY", connect_timeout=1, read_timeout=1)
        except RuntimeError:
            pass
    finally:
        vlm_model_list.execute_http = orig
    assert captured["url"] == "https://api.x.ai/v1/language-models"


def test_grok_profile_enables_only_xai_route() -> None:
    vs = Vlm()
    vs.model_profile_id = "grok-4-6"
    mp = vlm_config.resolve_model_profile(vs)
    cm = vlm_config.build_connection_map(vs, mp)
    assert cm["builtin-xai"].enabled is True
    assert cm["builtin-xai"].model_id == "grok-4.6"
    # Grok プロファイルは他プロバイダーの binding を持たないので他経路は無効化される
    assert cm["builtin-gemini"].enabled is False
    assert cm["builtin-openai"].enabled is False


def test_grok_responses_request_shape() -> None:
    conn = {c.connection_id: c for c in vlm_connections.default_builtin_connections()}["builtin-xai"]
    proto = get_protocol(conn.protocol)
    spec = VlmCallSpec(
        model_id="grok-4.6", system_prompt="SYS",
        user_prompt="Caption this image.", image=_FakeImage(), profile=GenerationProfile())
    req = proto.build_request(conn.base_url, "xai-KEY", spec)
    assert req.url == "https://api.x.ai/v1/responses"
    assert req.headers["Authorization"] == "Bearer xai-KEY"
    body = req.json_body
    assert body["model"] == "grok-4.6"
    assert body["instructions"] == "SYS"
    parts = body["input"][0]["content"]
    assert [p["type"] for p in parts] == ["input_text", "input_image"]
    # xAI Responses API は image_url を「文字列の data URL」で受ける
    assert isinstance(parts[1]["image_url"], str)
    assert parts[1]["image_url"].startswith("data:image/")
    assert body["max_output_tokens"] == GenerationProfile().max_output_tokens


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"ALL {len(tests)} xAI Grok tests PASSED")
