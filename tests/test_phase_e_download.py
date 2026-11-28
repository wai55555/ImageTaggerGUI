"""Phase E（docs/261127_plan-review-findings-fix.md §11）の回帰テスト。

対象: R3-F3 / R28-2 / R3-F7 / R4-7 / R10-1 / R29-1 / R32-4 / R1-9 / R15-3 / R16-3
（REQ-E1〜E9、E11）。REQ-E10（複数ファイル DL の全体進捗）は既存 Phase 4 の
PERF-1 完了後が前提のため、本ファイルの対象外。

ネットワークへは出ない。`requests.get` と `get_model_info_from_pointer` を
モックし、書き込み先はすべて tmp_path。実 models/・実 config.ini は触らない。

Offline only - no network, no GUI. Run:  pytest tests/test_phase_e_download.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

import app_settings
import model_registry
import workers


def _get_string(section, key, **kwargs):
    return key


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class _FakeResponse:
    """requests.Response の最小代替（context manager / iter_content / headers）。"""

    def __init__(self, body: bytes = b"", *, status: int = 200,
                 headers: dict | None = None, error: Exception | None = None):
        self._body = body
        self.status_code = status
        self.headers = headers or {}
        self._error = error
        self.closed = False

    def raise_for_status(self):
        if self._error is not None:
            raise self._error
        if self.status_code >= 400:
            raise workers.requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start:start + chunk_size]

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _FakeGet:
    """URL -> 応答（または応答リスト / 例外）を順に返す requests.get 代替。"""

    def __init__(self, responses):
        self._responses = responses
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        item = self._responses[url]
        if isinstance(item, list):
            item = item.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Worker:
    """DownloaderWorker と、そのシグナルを集めるリストの組。"""

    def __init__(self, model_id="test-model"):
        self.worker = workers.DownloaderWorker(_get_string, model_id=model_id)
        self.logs: list[str] = []
        self.progress: list[tuple[int, float, float]] = []
        self.finished: list[bool] = []
        self.worker.log_message.connect(lambda msg, color: self.logs.append(msg))
        self.worker.progress_update.connect(
            lambda pct, done, total: self.progress.append((pct, done, total)))
        self.worker.download_finished.connect(self.finished.append)


def _entry(model_dir: Path, config: dict, model_id: str = "test-model") -> model_registry.ModelEntry:
    return model_registry.ModelEntry(
        model_id=model_id, model_name=model_id, model_type="tagger",
        model_dir=model_dir, config=config,
    )


@pytest.fixture
def no_side_fetches(monkeypatch, tmp_path):
    """実 models/ を触らないための隔離: 翻訳CSV取得と TAGS_CSV_PATH を差し替える。"""
    monkeypatch.setattr(workers, "ensure_pixai_tags_csv", lambda *a, **k: True)
    monkeypatch.setattr(workers, "TAGS_CSV_PATH", tmp_path / "pixai" / "selected_tags.csv")
    monkeypatch.setattr(workers, "MODEL_PATH", tmp_path / "pixai" / "model.onnx")
    monkeypatch.setattr(workers, "MODEL_POINTER_PATH", tmp_path / "pixai" / "model_pointer.txt")


def _record_verification(monkeypatch) -> list[tuple[str, bool]]:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        workers, "update_model_verification_status",
        lambda model_id, verified, get_string: calls.append((model_id, verified)),
    )
    return calls


# --- REQ-E1: ポインタ解決失敗で DL を中断する ---------------------------------------

def test_req_e1_pointer_failure_aborts_before_downloading(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    monkeypatch.setattr(workers.model_registry, "get_model_entry",
                        lambda model_id: _entry(tmp_path, {
                            "network": {
                                "files": {"model.onnx": "https://huggingface.co/test/model/resolve/main/model.onnx"},
                                "hash_verified_file": "model.onnx",
                            },
                        }))
    monkeypatch.setattr(workers, "get_model_info_from_pointer", lambda url, gs=None: (None, None))
    requested: list[str] = []
    monkeypatch.setattr(workers.requests, "get",
                        lambda url, **kw: requested.append(url) or pytest.fail("must not download"))
    verified = _record_verification(monkeypatch)

    work.worker.run_download()

    assert requested == []
    assert work.finished == [False]
    assert verified == []
    assert not (tmp_path / "model.onnx").exists()


# --- REQ-E2: verified を付けるタイミング --------------------------------------------

def _manifest_two_files(tmp_path: Path) -> tuple[dict, dict[str, bytes]]:
    bodies = {"a.onnx": b"AAAA", "b.onnx": b"BBBB"}
    config = {
        "network": {
            "files": {name: f"https://huggingface.co/test/model/resolve/main/{name}" for name in bodies},
            "hash_verified_file": ["a.onnx", "b.onnx"],
        },
    }
    return config, bodies


def test_req_e2_partial_failure_does_not_mark_verified(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    config, bodies = _manifest_two_files(tmp_path)
    monkeypatch.setattr(workers.model_registry, "get_model_entry", lambda model_id: _entry(tmp_path, config))
    monkeypatch.setattr(
        workers, "get_model_info_from_pointer",
        lambda url, gs=None: (_sha(bodies[url.rsplit("/", 1)[-1]]), len(bodies[url.rsplit("/", 1)[-1]])),
    )
    ok = _FakeResponse(bodies["a.onnx"], headers={"content-length": str(len(bodies["a.onnx"]))})
    fail = _FakeResponse(b"", status=404)
    monkeypatch.setattr(workers.requests, "get", _FakeGet({
        "https://huggingface.co/test/model/resolve/main/a.onnx": ok,
        "https://huggingface.co/test/model/resolve/main/b.onnx": fail,
    }))
    verified = _record_verification(monkeypatch)

    work.worker.run_download()

    assert work.finished == [False]
    assert verified == []          # 1ファイル目のハッシュ一致では印さない（REQ-E2）


def test_req_e2_full_success_marks_verified_once(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    config, bodies = _manifest_two_files(tmp_path)
    monkeypatch.setattr(workers.model_registry, "get_model_entry", lambda model_id: _entry(tmp_path, config))
    monkeypatch.setattr(
        workers, "get_model_info_from_pointer",
        lambda url, gs=None: (_sha(bodies[url.rsplit("/", 1)[-1]]), len(bodies[url.rsplit("/", 1)[-1]])),
    )
    monkeypatch.setattr(workers.requests, "get", _FakeGet({
        "https://huggingface.co/test/model/resolve/main/a.onnx": _FakeResponse(bodies["a.onnx"], headers={"content-length": "4"}),
        "https://huggingface.co/test/model/resolve/main/b.onnx": _FakeResponse(bodies["b.onnx"], headers={"content-length": "4"}),
    }))
    verified = _record_verification(monkeypatch)

    work.worker.run_download()

    assert work.finished == [True]
    assert verified == [("test-model", True)]


# --- REQ-E3: CSV サイドカーの完全性判定 --------------------------------------------

def _csv(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_req_e3_structurally_sound_csv_is_complete(tmp_path):
    path = _csv(tmp_path / "tags.csv", [
        "id,tag_id,name,category,count,ips",
        "0,1,alpha,general,10,",
        "1,2,beta,general,20,",
    ])
    assert workers.DownloaderWorker._csv_sidecar_looks_complete(path, "pixai6col") is True


def test_req_e3_truncated_row_is_not_complete(tmp_path):
    path = _csv(tmp_path / "tags.csv", [
        "id,tag_id,name,category,count,ips",
        "0,1,alpha,general,10,",
        "1,2,be",                       # 行の途中で切れている
    ])
    assert workers.DownloaderWorker._csv_sidecar_looks_complete(path, "pixai6col") is False
    assert workers.DownloaderWorker._csv_sidecar_looks_complete(path, None) is False


def test_req_e3_header_must_match_the_declared_format(tmp_path):
    path = _csv(tmp_path / "tags.csv", [
        "tag_id,name,category,count",
        "0,alpha,general,10",
        "1,beta,general,20",
    ])
    assert workers.DownloaderWorker._csv_sidecar_looks_complete(path, "wd4col") is True
    assert workers.DownloaderWorker._csv_sidecar_looks_complete(path, "pixai6col") is False


def test_req_e3_unknown_format_only_checks_row_widths(tmp_path):
    path = _csv(tmp_path / "tags.csv", [
        "a,b,c",
        "1,2,3",
    ])
    assert workers.DownloaderWorker._csv_sidecar_looks_complete(path, None) is True


def test_req_e3_sidecar_check_handles_json_and_empty_files(tmp_path):
    good = tmp_path / "meta.json"
    good.write_text(json.dumps({"ok": True}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    w = workers.DownloaderWorker(_get_string)
    assert w._sidecar_looks_complete(good) is True
    assert w._sidecar_looks_complete(bad) is False
    assert w._sidecar_looks_complete(empty, "pixai6col") is False


# --- REQ-E4: Range 再開の 416 -------------------------------------------------------

def test_req_e4_416_with_a_complete_local_file_succeeds(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    body = b"0123456789"
    target = tmp_path / "model.onnx"
    target.write_bytes(body)
    # 期待サイズは未登録だが SHA256 が分かっている → 416 でもハッシュで完全性を証明できる
    # （gpu_runtime の 416 処理と同じ考え方）。
    work.worker._hash_verified_files = {target}
    response = _FakeResponse(status=416)
    fake_get = _FakeGet({"https://huggingface.co/test/model/resolve/main/model.onnx": response})
    monkeypatch.setattr(workers.requests, "get", fake_get)

    result = work.worker._download_single_file(target, "https://huggingface.co/test/model/resolve/main/model.onnx", _sha(body))

    assert bool(result) is True
    assert len(fake_get.calls) == 1
    assert response.closed is True
    assert target.read_bytes() == body


def test_req_e4_416_with_a_partial_file_restarts_once(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    body = b"0123456789"
    target = tmp_path / "model.onnx"
    target.write_bytes(body[:4])
    first = _FakeResponse(status=416)
    second = _FakeResponse(body, headers={"content-length": str(len(body))})
    fake_get = _FakeGet({"https://huggingface.co/test/model/resolve/main/model.onnx": [first, second]})
    monkeypatch.setattr(workers.requests, "get", fake_get)

    result = work.worker._download_single_file(target, "https://huggingface.co/test/model/resolve/main/model.onnx")

    assert bool(result) is True
    assert len(fake_get.calls) == 2
    assert first.closed is True
    assert target.read_bytes() == body


def test_req_e4_a_second_416_gives_up_instead_of_looping(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    target = tmp_path / "model.onnx"
    target.write_bytes(b"partial")
    fake_get = _FakeGet({"https://huggingface.co/test/model/resolve/main/model.onnx": [_FakeResponse(status=416), _FakeResponse(status=416)]})
    monkeypatch.setattr(workers.requests, "get", fake_get)

    result = work.worker._download_single_file(target, "https://huggingface.co/test/model/resolve/main/model.onnx")

    assert bool(result) is False
    assert len(fake_get.calls) == 2


# --- REQ-E5: download_finished の必発火 --------------------------------------------

def test_req_e5_unknown_model_still_finishes_once(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker(model_id="does-not-exist")
    monkeypatch.setattr(workers.model_registry, "get_model_entry", lambda model_id: None)

    work.worker.run_download()

    assert work.finished == [False]


def test_req_e5_unexpected_exception_finishes_once(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    # network.files が dict でない → 想定外例外（旧実装では download_finished が出なかった）
    monkeypatch.setattr(workers.model_registry, "get_model_entry",
                        lambda model_id: _entry(tmp_path, {"network": {"files": ["model.onnx"]}}))

    work.worker.run_download()

    assert work.finished == [False]


# --- REQ-E6: 期待サイズ不明時の最終サイズ検査 --------------------------------------

def test_req_e6_no_size_and_no_content_length_is_suspected(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    body = b"sidecar-body"
    target = tmp_path / "selected_tags.csv"
    monkeypatch.setattr(workers.requests, "get", _FakeGet({
        "https://huggingface.co/test/model/resolve/main/tags.csv": _FakeResponse(body),   # content-length なし
    }))

    result = work.worker._download_single_file(target, "https://huggingface.co/test/model/resolve/main/tags.csv")

    assert bool(result) is True
    assert result.suspected_incomplete is True
    assert target.read_bytes() == body


def test_req_e6_manifest_does_not_mark_verified_when_a_sidecar_is_suspected(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    model_body = b"MODEL"
    sidecar = b'{"meta": true}'
    config = {
        "network": {
            "files": {
                "model.onnx": "https://huggingface.co/test/model/resolve/main/model.onnx",
                "meta.json": "https://huggingface.co/test/model/resolve/main/meta.json",
            },
            "hash_verified_file": "model.onnx",
        },
        "tags_csv": {"format": "pixai6col"},
    }
    monkeypatch.setattr(workers.model_registry, "get_model_entry", lambda model_id: _entry(tmp_path, config))
    monkeypatch.setattr(workers, "get_model_info_from_pointer",
                        lambda url, gs=None: (_sha(model_body), len(model_body)))
    monkeypatch.setattr(workers.requests, "get", _FakeGet({
        "https://huggingface.co/test/model/resolve/main/model.onnx": _FakeResponse(model_body, headers={"content-length": "5"}),
        "https://huggingface.co/test/model/resolve/main/meta.json": _FakeResponse(sidecar),  # content-length なし
    }))
    verified = _record_verification(monkeypatch)

    work.worker.run_download()

    assert work.finished == [True]
    assert verified == []          # REQ-E6(b): verified を付けない
    assert (tmp_path / "meta.json").read_bytes() == sidecar


def test_req_e6_legacy_clears_verified_when_the_tags_csv_is_suspected(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker(model_id="pixai-tagger-v0.9")
    model_body = b"MODEL"
    csv_body = b"id,tag_id,name,category,count,ips\n0,1,alpha,general,1,\n"
    monkeypatch.setattr(workers, "DOWNLOAD_URLS", {
        workers.MODEL_PATH: "https://huggingface.co/test/model/resolve/main/model.onnx",
        workers.TAGS_CSV_PATH: "https://huggingface.co/test/model/resolve/main/tags.csv",
        workers.MODEL_POINTER_PATH: "https://huggingface.co/test/model/resolve/main/pointer.txt",
    })
    monkeypatch.setattr(workers, "get_model_info_from_pointer",
                        lambda url, gs=None: (_sha(model_body), len(model_body)))
    monkeypatch.setattr(workers.requests, "get", _FakeGet({
        "https://huggingface.co/test/model/resolve/main/model.onnx": _FakeResponse(model_body, headers={"content-length": "5"}),
        "https://huggingface.co/test/model/resolve/main/tags.csv": _FakeResponse(csv_body),   # content-length なし
    }))
    verified = _record_verification(monkeypatch)

    work.worker.run_download()

    assert work.finished == [True]        # DL 自体は継続してよい
    assert verified == [("pixai-tagger-v0.9", True), ("pixai-tagger-v0.9", False)]


# --- REQ-E7: サイズ超過ローカルファイルの行き止まり --------------------------------

def test_req_e7_oversized_local_file_is_removed_and_recovers(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    body = b"0123456789"
    target = tmp_path / "model.onnx"
    target.write_bytes(body + b"EXTRA")
    work.worker._file_sizes[target] = len(body)
    fake_get = _FakeGet({
        "https://huggingface.co/test/model/resolve/main/model.onnx": _FakeResponse(body, headers={"content-length": str(len(body))}),
    })
    monkeypatch.setattr(workers.requests, "get", fake_get)

    first = work.worker._download_single_file(target, "https://huggingface.co/test/model/resolve/main/model.onnx")
    assert bool(first) is False
    assert not target.exists()              # 文言どおり削除する
    assert fake_get.calls == []

    second = work.worker._download_single_file(target, "https://huggingface.co/test/model/resolve/main/model.onnx")
    assert bool(second) is True
    assert target.read_bytes() == body


# --- REQ-E8: レスポンスのクローズ ---------------------------------------------------

def test_req_e8_response_is_closed_on_success(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    response = _FakeResponse(b"body", headers={"content-length": "4"})
    monkeypatch.setattr(workers.requests, "get", _FakeGet({"https://huggingface.co/test/model/resolve/main/f": response}))

    result = work.worker._download_single_file(tmp_path / "f.bin", "https://huggingface.co/test/model/resolve/main/f")

    assert bool(result) is True
    assert response.closed is True


def test_req_e8_response_is_closed_on_failure(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    response = _FakeResponse(error=workers.requests.exceptions.ConnectionError("boom"))
    monkeypatch.setattr(workers.requests, "get", _FakeGet({"https://huggingface.co/test/model/resolve/main/f": response}))

    result = work.worker._download_single_file(tmp_path / "f.bin", "https://huggingface.co/test/model/resolve/main/f")

    assert bool(result) is False
    assert response.closed is True


# --- REQ-E9: DL 開始ログのサイズ表示 ------------------------------------------------

def test_req_e9_unknown_expected_size_does_not_log_zero_mb(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    body = b"abc"
    monkeypatch.setattr(workers.requests, "get", _FakeGet({
        "https://huggingface.co/test/model/resolve/main/f": _FakeResponse(body, headers={"content-length": "3"}),
    }))

    work.worker._download_single_file(tmp_path / "f.bin", "https://huggingface.co/test/model/resolve/main/f")

    assert not any("0.00" in message for message in work.logs)
    assert "DownloaderWorker_Downloading_Model_UnknownSize" in work.logs
    assert "DownloaderWorker_Downloading_Model" not in work.logs


def test_req_e9_known_expected_size_still_logs_the_size(monkeypatch, tmp_path, no_side_fetches):
    work = _Worker()
    body = b"abc"
    target = tmp_path / "model.onnx"
    work.worker._file_sizes[target] = len(body)
    monkeypatch.setattr(workers.requests, "get", _FakeGet({
        "https://huggingface.co/test/model/resolve/main/model.onnx": _FakeResponse(body, headers={"content-length": "3"}),
    }))

    work.worker._download_single_file(target, "https://huggingface.co/test/model/resolve/main/model.onnx")

    assert "DownloaderWorker_Downloading_Model" in work.logs
    assert "DownloaderWorker_Downloading_Model_UnknownSize" not in work.logs


# --- REQ-E11: タガー初期化失敗時の unverified 化の非対称 ---------------------------

def _tagger_settings():
    return app_settings.load_settings(app_settings.get_default_config())


def test_req_e11_missing_libraries_keep_verified(monkeypatch, tmp_path):
    tagger = workers.TaggerThreadWorker(_tagger_settings(), None, get_string=_get_string)
    monkeypatch.setattr(workers, "setup_tagger_from_settings", lambda settings, gs: (None, {}))
    monkeypatch.setattr(workers.tagging_core, "ort", None)
    verified = _record_verification(monkeypatch)

    tagger.run_tagging()

    assert verified == []


def test_req_e11_model_file_problem_still_clears_verified(monkeypatch, tmp_path):
    if workers.tagging_core.ort is None:
        pytest.skip("この環境には onnxruntime が無い（ライブラリ不足側の分岐になる）")
    tagger = workers.TaggerThreadWorker(_tagger_settings(), None, get_string=_get_string)
    monkeypatch.setattr(workers, "setup_tagger_from_settings", lambda settings, gs: (None, {}))
    verified = _record_verification(monkeypatch)

    tagger.run_tagging()

    assert verified == [(tagger._settings.model.model_id, False)]
