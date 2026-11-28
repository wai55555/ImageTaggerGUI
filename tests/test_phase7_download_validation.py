"""Phase 7（モデル DL 経路の入力検証）の回帰テスト。

対象: REQ-7.1〜7.7（R ID なし。§1.2 の訂正版方針）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import workers

# ---------------------------------------------------------------------------
# REQ-7.1: 安全な相対パスだけを許可する
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "model.onnx",
    "onnx/vision_encoder_quantized.onnx",
    "onnx/decoder_model_merged.onnx",
    "sub/dir/file.bin",
])
def test_safe_relative_paths_are_accepted(name):
    assert workers._is_safe_relative_path(name) is True


@pytest.mark.parametrize("name", [
    "",
    "   ",
    "/etc/passwd",
    "C:/Windows/system32/x.dll",
    "c:x.dll",
    "..",
    "../escape.bin",
    "onnx/../../escape.bin",
    "onnx//file.bin",
    "onnx/",
    "dir\\..\\escape.bin",
    None,
    123,
])
def test_unsafe_paths_are_rejected(name):
    assert workers._is_safe_relative_path(name) is False


# ---------------------------------------------------------------------------
# REQ-7.3: https かつ Hugging Face のホストだけ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://huggingface.co/microsoft/Florence-2-base/resolve/main/model.onnx",
    "https://cdn-lfs.huggingface.co/repos/x/y",
    "https://cdn-lfs-us-1.hf.co/repos/x/y",
])
def test_allowed_model_urls(url):
    assert workers._is_allowed_model_url(url) is True


@pytest.mark.parametrize("url", [
    "http://huggingface.co/x/y/resolve/main/m.onnx",
    "https://example.com/model.onnx",
    "https://huggingface.co.evil.example/model.onnx",
    "file:///C:/models/model.onnx",
    "not a url",
    "",
    None,
])
def test_rejected_model_urls(url):
    assert workers._is_allowed_model_url(url) is False


# ---------------------------------------------------------------------------
# REQ-7.2: resolve した書き込み先が model_dir 配下に収まること（U-9）
# ---------------------------------------------------------------------------


def test_destination_must_stay_inside_the_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert workers._is_inside_directory(model_dir / "onnx" / "a.onnx", model_dir)
    assert workers._is_inside_directory(model_dir / "a.onnx", model_dir)

    outside = tmp_path / "outside"
    outside.mkdir()
    assert not workers._is_inside_directory(outside / "a.onnx", model_dir)


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlink 非対応")
def test_symlink_escape_is_rejected_by_resolution(tmp_path):
    """U-9: リンク自体は拒否せず、resolve 先の配下判定で弾く。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (model_dir / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink を作れない環境")
    assert not workers._is_inside_directory(model_dir / "link" / "a.onnx", model_dir)


# ---------------------------------------------------------------------------
# REQ-7.6: 同梱モデルの全キー・全 URL が検証を通る（正常系を壊さない）
# ---------------------------------------------------------------------------


def test_every_bundled_model_config_passes_validation():
    root = Path(__file__).resolve().parents[1] / "models"
    configs = sorted(root.glob("*/model_config.json"))
    assert configs, "同梱 model_config.json が見つからない"
    checked_keys = 0
    for config_path in configs:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        network = data.get("network")
        if not isinstance(network, dict):
            continue
        files = network.get("files") or {}
        for name, url in files.items():
            assert workers._is_safe_relative_path(name), (config_path, name)
            assert workers._is_allowed_model_url(url), (config_path, url)
            checked_keys += 1
        verified = network.get("hash_verified_file")
        verified_names = [verified] if isinstance(verified, str) else list(verified or [])
        for name in verified_names:
            assert workers._is_safe_relative_path(name), (config_path, name)
    assert checked_keys > 0


# ---------------------------------------------------------------------------
# REQ-7.4: 置換が効かない URL ではポインターを取りに行かない
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.pointer_calls: list[str] = []
        self.messages: list[tuple[str, str]] = []

    def get_string(self, section, key, **kwargs):
        return key

    def log(self, message, color=None):
        self.messages.append((message, color))


def _worker(monkeypatch):
    rec = _Recorder()
    worker = workers.DownloaderWorker(rec.get_string)
    worker.log_message.connect(rec.log)
    return worker, rec


def test_pointer_lookup_is_skipped_when_the_url_is_not_resolve_main(monkeypatch, tmp_path):
    worker, rec = _worker(monkeypatch)
    calls: list[str] = []

    def fake_pointer(url, get_string):
        calls.append(url)
        return "sha", 10

    monkeypatch.setattr(workers, "get_model_info_from_pointer", fake_pointer)
    network = {"files": {"model.onnx": "https://huggingface.co/o/r/raw/main/model.onnx"},
               "hash_verified_file": ["model.onnx"]}

    assert worker._download_from_manifest(tmp_path, network) is False
    assert calls == []                     # ポインターを取りに行かない
    assert any("DownloaderWorker_Error_FailedToGetModelInfo" in m for m, _ in rec.messages)


def test_manifest_with_unsafe_file_name_aborts_before_any_download(monkeypatch, tmp_path):
    worker, rec = _worker(monkeypatch)
    downloaded: list[Path] = []

    def fake_download(self, file_path, url, expected_sha256=None, mark_verified=False):
        downloaded.append(file_path)
        raise AssertionError("must not download after validation failure")

    monkeypatch.setattr(workers.DownloaderWorker, "_download_single_file", fake_download)
    network = {"files": {"../escape.onnx": "https://huggingface.co/o/r/resolve/main/x.onnx"}}

    assert worker._download_from_manifest(tmp_path, network) is False
    assert downloaded == []
    assert any("DownloaderWorker_Error_UnsafeManifest" in m for m, _ in rec.messages)


def test_manifest_with_unexpected_host_aborts(monkeypatch, tmp_path):
    worker, _rec = _worker(monkeypatch)
    network = {"files": {"model.onnx": "https://evil.example/model.onnx"}}
    assert worker._download_from_manifest(tmp_path, network) is False
