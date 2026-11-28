"""`model_registry.discover_models()` 自体のテスト（REQ-1.5 / R3-F6）。

同梱 model_config.json 群を「読めること」は 77 件側 Phase A の
tests/test_phase_a_inference_input.py が既に固定している。ここで埋めるのは、
その手前にあった空白——**ディレクトリ走査から ModelEntry 組み立てまでの
discover_models() 本体**である。スキーマ不正・重複・manual_download ゲート・
ユーザー側/同梱側の優先順位を、テンポラリの models ディレクトリで検証する。
"""
import json
from pathlib import Path

import pytest

import constants
import model_registry


def _write_manifest(root: Path, dir_name: str, payload) -> Path:
    directory = root / dir_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model_config.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _manifest(model_id: str, model_type: str = "tagger", **extra) -> dict:
    return {
        "model_id": model_id,
        "model_name": f"{model_id} display",
        "model_type": model_type,
        "network": {"files": {"model.onnx": "https://example.invalid/model.onnx"}},
        "inference": {"input_size": 448, "layout": "nchw", "channel_order": "rgb"},
        **extra,
    }


@pytest.fixture()
def models_dirs(monkeypatch, tmp_path):
    """ユーザー側 models/ と同梱 resource models/ をテンポラリへ向ける。"""
    user_root = tmp_path / "models"
    resource_root = tmp_path / "resource_models"
    user_root.mkdir()
    resource_root.mkdir()
    monkeypatch.setattr(constants, "MODELS_DIR", user_root)
    monkeypatch.setattr(constants, "MODELS_RESOURCE_DIR", resource_root)
    monkeypatch.setattr(constants, "MODEL_PATH", user_root / constants.MODEL_DIR_NAME / "model.onnx")
    return user_root, resource_root


def test_pixai_pseudo_entry_always_comes_first(models_dirs):
    user_root, _ = models_dirs
    _write_manifest(user_root, "zzz_unlisted", _manifest("zzz-model"))

    entries = model_registry.discover_models()

    assert entries[0].model_id == "pixai-tagger-v0.9"
    assert entries[0].config == model_registry._PIXAI_CONFIG
    assert entries[0].model_dir == constants.MODEL_PATH.parent
    assert [e.model_id for e in entries] == ["pixai-tagger-v0.9", "zzz-model"]


def test_display_order_then_alphabetical_for_unlisted(models_dirs):
    user_root, _ = models_dirs
    _write_manifest(user_root, "a_unlisted", _manifest("aaa-model"))
    _write_manifest(user_root, "b_unlisted", _manifest("bbb-model"))
    _write_manifest(user_root, "camie", _manifest("camie_tagger_v2"))
    _write_manifest(user_root, "pixai_v1", _manifest("pixai-tagger-v1.0"))

    ids = [e.model_id for e in model_registry.discover_models()]

    # キュレーション順（pixai-v1.0 → camie）→ 未掲載は model_id 昇順。
    assert ids == ["pixai-tagger-v0.9", "pixai-tagger-v1.0", "camie_tagger_v2",
                   "aaa-model", "bbb-model"]


def test_model_dir_is_the_user_writable_models_dir(models_dirs):
    user_root, resource_root = models_dirs
    _write_manifest(resource_root, "florence", _manifest("florence2_base_ft", "captioner"))

    entry = model_registry.get_model_entry("florence2_base_ft")

    assert entry is not None
    # 設定は同梱側から読んでも、DL 済みファイルの置き場はユーザー側。
    assert entry.model_dir == user_root / "florence"


def test_user_side_wins_over_resource_side_for_the_same_directory(models_dirs):
    user_root, resource_root = models_dirs
    _write_manifest(user_root, "same_dir", _manifest("user-version"))
    _write_manifest(resource_root, "same_dir", _manifest("resource-version"))

    ids = [e.model_id for e in model_registry.discover_models()]

    assert "user-version" in ids
    assert "resource-version" not in ids


def test_invalid_manifests_are_skipped(models_dirs):
    user_root, _ = models_dirs
    _write_manifest(user_root, "broken_json", "{not json")
    _write_manifest(user_root, "not_object", ["a", "list"])
    _write_manifest(user_root, "bad_model_id_type", {"model_id": 123})
    _write_manifest(user_root, "empty_model_id", {"model_id": "   "})
    _write_manifest(user_root, "bad_model_type", _manifest("bad-type", "embedder"))
    _write_manifest(user_root, "valid", _manifest("valid-model"))

    ids = [e.model_id for e in model_registry.discover_models()]

    assert ids == ["pixai-tagger-v0.9", "valid-model"]


def test_duplicate_model_id_keeps_the_first_alphabetical_directory(models_dirs):
    user_root, _ = models_dirs
    _write_manifest(user_root, "a_first", _manifest("dup-model", model_name="first"))
    _write_manifest(user_root, "b_second", _manifest("dup-model", model_name="second"))

    entries = [e for e in model_registry.discover_models() if e.model_id == "dup-model"]

    assert len(entries) == 1
    assert entries[0].model_name == "first"


def test_manual_download_model_is_hidden_until_all_files_exist(models_dirs):
    user_root, _ = models_dirs
    files = {"onnx/encoder.onnx": "https://example.invalid/e.onnx",
             "model.onnx": "https://example.invalid/m.onnx"}
    payload = _manifest("manual-model", manual_download=True,
                        network={"files": files})
    directory = _write_manifest(user_root, "manual", payload)

    assert "manual-model" not in [e.model_id for e in model_registry.discover_models()]

    # 必須ファイルが揃えば通常モデルと同じく現れる。
    (directory / "onnx").mkdir()
    (directory / "onnx" / "encoder.onnx").write_bytes(b"x")
    (directory / "model.onnx").write_bytes(b"y")

    entry = model_registry.get_model_entry("manual-model")
    assert entry is not None
    assert entry.model_dir == directory


def test_manifest_without_network_files_stays_hidden_when_manual(models_dirs):
    """manual_download で network.files が無いものは「揃っている」と見なさない。"""
    user_root, _ = models_dirs
    _write_manifest(user_root, "manual_empty", _manifest("manual-empty", manual_download=True))

    assert "manual-empty" not in [e.model_id for e in model_registry.discover_models()]


def test_missing_models_dirs_do_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "MODELS_DIR", tmp_path / "missing_models")
    monkeypatch.setattr(constants, "MODELS_RESOURCE_DIR", tmp_path / "missing_resource")

    assert [e.model_id for e in model_registry.discover_models()] == ["pixai-tagger-v0.9"]


def test_primary_onnx_file_name_falls_back_for_multi_onnx_manifests(models_dirs):
    user_root, _ = models_dirs
    payload = _manifest("multi-onnx")
    payload["network"]["files"] = {
        "a.onnx": "https://example.invalid/a.onnx",
        "b.onnx": "https://example.invalid/b.onnx",
    }
    _write_manifest(user_root, "multi", payload)

    entry = model_registry.get_model_entry("multi-onnx")

    assert entry is not None
    assert model_registry.primary_onnx_file_name(entry) == "model.onnx"
