"""Phase C（§9、REQ-C1〜C7）の検証テスト。

対象: バッチループの完了・進捗・冪等性。
- REQ-C1（R4-1）致命的例外時の完了シグナル（意図的変更）
- REQ-C2（R2-1）VLM の早期 return が失敗一覧を消さない（意図的変更）
- REQ-C3（R15-2）停止時の要約の件数会計
- REQ-C4（R15-5）進捗の発行位置（意図的変更）
- REQ-C5（R17-4）同 stem 画像の .txt 共有（意図的変更）
- REQ-C6（R23-1）Florence の APPEND／PREPEND の冪等性
- REQ-C7（R24-1）タガー OVERWRITE の同一内容の書き直し

Offline only - no network, no GUI（Qt は offscreen）。実データ・実 config.ini は触らない。
Run: rtk pytest tests/test_phase_c_batch_loops.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from PIL import Image

import app_settings as A
import caption_core
import vlm_worker
import workers
from tagging_core import (
    ExistingFileMode,
    FileChange,
    TagCategory,
    TagPrediction,
    TagResult,
)


# ---------------------------------------------------------------------------
# 共通ハーネス
# ---------------------------------------------------------------------------


def _write_image(path: Path, colour=(8, 8)) -> Path:
    Image.new("RGB", colour).save(path)
    return path


def _get_string(section, key, **kwargs):
    """本番の lang を引かず、キーをそのまま返す（既存テストの慣習）。"""
    return key


class _SummaryRecorder:
    """Batch_Summary の差し込み値だけを記録する get_string。"""

    def __init__(self):
        self.summaries: list[dict] = []

    def __call__(self, section, key, **kwargs):
        if key == "Batch_Summary":
            self.summaries.append(dict(kwargs))
        return key


def _tagger_settings_dict(input_dir: Path, mode=ExistingFileMode.OVERWRITE):
    return {
        "INPUT_DIR": input_dir,
        "EXISTING_FILE_MODE": mode,
        "TAG_THRESHOLDS": {},
        "MAX_TAGS_PER_CATEGORY": {},
        "ENABLE_SOLO_LIMIT": False,
        "CONVERT_UNDERSCORE": False,
    }


def _caption_settings_dict(input_dir: Path, mode=ExistingFileMode.OVERWRITE,
                           placement="OVERWRITE"):
    return {
        "INPUT_DIR": input_dir,
        "EXISTING_FILE_MODE": mode,
        "CAPTION_PLACEMENT": placement,
        "TASK": "t",
    }


class _TaggerStub:
    """`tag` を返すだけのタガー。`tags_for_call` で呼び出しごとに差し替えられる。"""

    tag_meta_lookup: dict = {}

    def __init__(self, tag="1girl", break_on_call=None):
        self._tag = tag
        self._break_on_call = break_on_call
        self.calls = 0
        self.events: list[str] = []

    def infer_batch(self, images, thresholds=None, max_tags=None):
        self.calls += 1
        self.events.append("infer")
        if self._break_on_call is not None and self.calls == self._break_on_call:
            # filter_tags_by_solo_rule / format_tags（プロセスの try 外）で
            # AttributeError になる壊れた結果。ループ全体の中断を再現する。
            return [TagResult(tags=[object()])]
        return [TagResult(tags=[TagPrediction(self._tag, 0.9, TagCategory.GENERAL)])]


class _CaptionerStub:
    config = SimpleNamespace(default_task="t", tasks={"t": "describe"})

    def __init__(self, caption="a caption"):
        self._caption = caption
        self.events: list[str] = []

    def generate(self, image, task_prompt, stop_checker):
        self.events.append("infer")
        return self._caption, False


def _run_tagging(input_dir: Path, images, *, mode=ExistingFileMode.OVERWRITE,
                 tagger=None, stop=None, progress=None, failed=None,
                 changes_out=None, get_string=None):
    import tagging_core
    return tagging_core.process_image_loop(
        tagger=tagger or _TaggerStub(),
        settings=_tagger_settings_dict(input_dir, mode),
        image_paths=list(images),
        decision_resolver=None,
        log_gui=lambda message, colour: None,
        stop_checker=stop or (lambda: False),
        get_string=get_string,
        progress_cb=progress,
        failed_paths=failed,
        changes_out=changes_out,
    )


def _run_caption(input_dir: Path, images, *, mode=ExistingFileMode.OVERWRITE,
                 placement="OVERWRITE", captioner=None, stop=None, progress=None,
                 failed=None, changes_out=None, get_string=None):
    return caption_core.process_caption_loop(
        captioner=captioner or _CaptionerStub(),
        settings=_caption_settings_dict(input_dir, mode, placement),
        image_paths=list(images),
        decision_resolver=None,
        log_gui=lambda message, colour: None,
        stop_checker=stop or (lambda: False),
        get_string=get_string,
        progress_cb=progress,
        failed_paths=failed,
        changes_out=changes_out,
    )


def _app_settings(input_dir: Path, existing_mode="OVERWRITE"):
    s = A.load_settings(A.get_default_config())
    s.paths.input_dir = str(input_dir)
    s.behavior.existing_file_mode = existing_mode
    s.behavior.target_mode = "ALL"
    s.caption.placement = "OVERWRITE"
    return s


def _fake_vlm_runtime(text="hello world"):
    from vlm_image import ImagePreprocessConfig

    def _caption_one(spec_base, ids):
        return SimpleNamespace(ok=True, text=text, connection_id="c1", model_id="m1",
                               stopped=False, stop_job=False, attempts=[], error=None)

    executor = SimpleNamespace(live_candidates=lambda ids: ids, caption_one=_caption_one)
    candidates = SimpleNamespace(connection_ids=["c1"], has_candidates=True, excluded={})
    return {
        "connections": {}, "gen_profile": None, "policy": None,
        "candidates": candidates, "executor": executor,
        # Phase F の REQ-F4 が実行開始時のプロファイル id をスナップショットとして
        # runtime に載せる（バッチ途中の切り替えで検証先がずれないように）。
        "model_profile_id": "p1",
        "system_prompt": "", "user_prompt": "",
        "image_cfg": ImagePreprocessConfig(max_long_edge=512, fmt="JPEG", jpeg_quality=80),
    }


class _VlmRecorder:
    def __init__(self):
        self.logs: list[tuple[str, str]] = []
        self.changes: list = []
        self.failed_emits: list[list] = []
        self.summaries: list[dict] = []

    def get_string(self, section, key, **kwargs):
        if key == "Batch_Summary":
            self.summaries.append(dict(kwargs))
        return key


def _run_vlm(monkeypatch, settings, runtime=None, failed_paths=(), single_test=False):
    rec = _VlmRecorder()

    def _runtime(self):
        return runtime

    monkeypatch.setattr(vlm_worker.VlmCaptionWorker, "_build_runtime", _runtime)
    worker = vlm_worker.VlmCaptionWorker(
        settings, None, get_string=rec.get_string, failed_paths=failed_paths,
        single_test=single_test)
    worker.log_message.connect(lambda m, c: rec.logs.append((m, c)))
    worker.batch_completed.connect(lambda lst: rec.changes.extend(lst))
    worker.batch_failed.connect(lambda lst: rec.failed_emits.append(list(lst)))
    worker.run_captioning()
    return rec


# ---------------------------------------------------------------------------
# REQ-C1（R4-1）致命的例外時の完了シグナル
# ---------------------------------------------------------------------------


def test_c1_tagger_fatal_error_still_emits_finished_file_changes(tmp_path, monkeypatch):
    """3枚目でループが致命的に中断しても、1〜2枚目の FileChanges は届く。"""
    images = [_write_image(tmp_path / f"{i}.png") for i in range(3)]
    tagger = _TaggerStub(break_on_call=3)
    monkeypatch.setattr(workers, "setup_tagger_from_settings",
                        lambda *a, **k: (tagger, _tagger_settings_dict(tmp_path)))
    monkeypatch.setattr(workers, "ensure_pixai_tags_csv", lambda *a, **k: True)

    worker = workers.TaggerThreadWorker(
        _app_settings(tmp_path), None, get_string=_get_string)
    logs, changes, completions = [], [], []
    worker.log_message.connect(lambda m, c: logs.append((m, c)))
    worker.batch_completed.connect(lambda lst: (completions.append(list(lst)), changes.extend(lst)))
    worker.run_tagging()

    assert len(completions) == 1, completions          # ちょうど1回
    assert [c.path.name for c in changes] == ["0.txt", "1.txt"]
    # 未処理分（3枚目）は完了として扱われない（FileChange が無い）
    assert all(c.path.name != "2.txt" for c in changes)
    assert any("Fatal_Exception" in m for m, _ in logs)
    # 処理済みファイルは実際に書かれている
    assert (tmp_path / "0.txt").read_text(encoding="utf-8") == "1girl"


def test_c1_captioner_fatal_error_still_emits_finished_file_changes(tmp_path, monkeypatch):
    """caption 側も同じ形（worker の except 経路で蓄積済み FileChanges を emit する）。"""
    images = [_write_image(tmp_path / f"{i}.png") for i in range(3)]
    monkeypatch.setattr(caption_core, "setup_captioner_from_settings",
                        lambda *a, **k: (_CaptionerStub(), _caption_settings_dict(tmp_path)))

    written = [FileChange(path=images[0].with_suffix(".txt"), previous_content=None,
                          new_content="x", was_append=False)]

    def _explode(*args, **kwargs):
        kwargs["changes_out"].extend(written)
        raise RuntimeError("boom after two writes")

    monkeypatch.setattr(caption_core, "process_caption_loop", _explode)

    worker = workers.CaptionerThreadWorker(
        _app_settings(tmp_path), None, get_string=_get_string)
    changes, completions = [], []
    worker.batch_completed.connect(lambda lst: (completions.append(list(lst)), changes.extend(lst)))
    worker.run_captioning()

    assert len(completions) == 1
    assert changes == written


def test_c1_bulk_tag_worker_emits_completion_on_stop(tmp_path):
    """BulkTagWorker は停止時も完了シグナルを出す（Undo 情報を失わない）。"""
    (tmp_path / "a.txt").write_text("solo, cat", encoding="utf-8")
    worker = workers.BulkTagWorker(get_string=_get_string)
    added, deleted = [], []
    worker.bulk_add_completed.connect(lambda *args: added.append(args))
    worker.bulk_delete_completed.connect(lambda *args: deleted.append(args))
    worker.stop()

    worker.run_bulk_add(tmp_path, "newtag", False)
    worker.run_bulk_delete(tmp_path, "cat")

    assert len(added) == 1 and len(deleted) == 1
    # 停止しているのでファイルは書かれない（完了シグナルだけが出る）
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "solo, cat"


# ---------------------------------------------------------------------------
# REQ-C2（R2-1）VLM の早期 return が失敗一覧を消さない
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["runtime_none", "no_candidates", "no_images"])
def test_c2_vlm_early_return_never_emits_batch_failed(tmp_path, monkeypatch, case):
    settings = _app_settings(tmp_path)
    runtime = _fake_vlm_runtime()
    if case == "runtime_none":
        runtime = None
    elif case == "no_candidates":
        runtime["candidates"] = SimpleNamespace(
            connection_ids=[], has_candidates=False,
            excluded={}, rejected_reason=None)
    # case == "no_images": ディレクトリは空のまま

    rec = _run_vlm(monkeypatch, settings, runtime=runtime)

    assert rec.failed_emits == [], rec.failed_emits
    if case == "no_candidates":
        assert any("No_Candidate" in m for m, _ in rec.logs), rec.logs
    if case == "no_images":
        assert any("Warn_No_Images" in m for m, _ in rec.logs), rec.logs


def test_c2_vlm_started_batch_still_reports_failures(tmp_path, monkeypatch):
    """ループに入った後は従来どおり batch_failed を emit する（成功時は空）。"""
    _write_image(tmp_path / "a.png")
    rec = _run_vlm(monkeypatch, _app_settings(tmp_path), runtime=_fake_vlm_runtime())

    assert rec.failed_emits == [[]]
    assert [c.path.name for c in rec.changes] == ["a.txt"]


# ---------------------------------------------------------------------------
# REQ-C3（R15-2）停止時の要約の件数会計
# ---------------------------------------------------------------------------


def test_c3_tagging_summary_counts_unprocessed_as_errors(tmp_path):
    images = [_write_image(tmp_path / f"{i}.png") for i in range(5)]
    done = {"n": 0}

    def stop():
        return done["n"] >= 2

    tagger = _TaggerStub()
    summary = _SummaryRecorder()

    def progress(done_count, total):
        done["n"] += 1

    failed: list[Path] = []
    _run_tagging(tmp_path, images, tagger=tagger, stop=stop, progress=progress,
                 failed=failed, get_string=summary)

    assert len(summary.summaries) == 1
    row = summary.summaries[0]
    assert row["total"] == 5
    assert row["written"] == 2
    assert row["errors"] == 3          # 未処理分が会計に入る
    assert row["written"] + row["skipped"] + row["unchanged"] + row["errors"] == row["total"]
    assert len(failed) == 3            # 再実行対象も従来どおり


def test_c3_tagging_skip_mode_stop_counts_existing_as_skipped(tmp_path):
    images = [_write_image(tmp_path / f"{i}.png") for i in range(4)]
    for img in images[2:]:             # 2枚は処理済み（.txt あり）
        img.with_suffix(".txt").write_text("done", encoding="utf-8")
    summary = _SummaryRecorder()
    failed: list[Path] = []

    _run_tagging(tmp_path, images, mode=ExistingFileMode.SKIP, stop=lambda: True,
                 failed=failed, get_string=summary)

    row = summary.summaries[0]
    assert (row["written"], row["skipped"], row["unchanged"], row["errors"]) == (0, 2, 0, 2)
    assert row["written"] + row["skipped"] + row["unchanged"] + row["errors"] == row["total"]
    assert failed == images[:2]        # SKIP モードの既存分は失敗に数えない


def test_c3_normal_completion_summary_is_unchanged(tmp_path):
    images = [_write_image(tmp_path / f"{i}.png") for i in range(3)]
    images[2].with_suffix(".txt").write_text("1girl", encoding="utf-8")
    summary = _SummaryRecorder()

    _run_tagging(tmp_path, images, mode=ExistingFileMode.SKIP, get_string=summary)

    row = summary.summaries[0]
    assert (row["written"], row["skipped"], row["unchanged"], row["errors"]) == (2, 1, 0, 0)
    assert row["total"] == 3


def test_c3_vlm_summary_counts_unprocessed_as_errors(tmp_path, monkeypatch):
    for i in range(3):
        _write_image(tmp_path / f"{i}.png")
    settings = _app_settings(tmp_path)
    rec = _VlmRecorder()

    monkeypatch.setattr(vlm_worker.VlmCaptionWorker, "_build_runtime",
                        lambda self: _fake_vlm_runtime())
    worker = vlm_worker.VlmCaptionWorker(settings, None, get_string=rec.get_string)
    worker.log_message.connect(lambda m, c: rec.logs.append((m, c)))
    worker.batch_failed.connect(lambda lst: rec.failed_emits.append(list(lst)))
    worker.stop()
    worker.run_captioning()

    row = rec.summaries[0]
    assert (row["written"], row["skipped"], row["unchanged"], row["errors"]) == (0, 0, 0, 3)
    assert row["written"] + row["skipped"] + row["unchanged"] + row["errors"] == row["total"]
    assert len(rec.failed_emits[-1]) == 3


# ---------------------------------------------------------------------------
# REQ-C4（R15-5）進捗の発行位置
# ---------------------------------------------------------------------------


def test_c4_tagging_progress_is_emitted_after_each_item(tmp_path):
    images = [_write_image(tmp_path / f"{i}.png") for i in range(3)]
    tagger = _TaggerStub()
    events: list[tuple[str, object]] = []

    def progress(done, total):
        # 発行時点で、その件数ぶんの .txt が既に書かれていること（= 処理後）
        written = len(list(tmp_path.glob("*.txt")))
        events.append((f"progress:{done}/{total}", written))

    _run_tagging(tmp_path, images, tagger=tagger, progress=progress)

    assert [e[0] for e in events] == ["progress:1/3", "progress:2/3", "progress:3/3"]
    assert [e[1] for e in events] == [1, 2, 3]   # 1件も先行しない


def test_c4_caption_progress_is_emitted_after_each_item(tmp_path):
    images = [_write_image(tmp_path / f"{i}.png") for i in range(3)]
    events: list[tuple[str, int]] = []

    def progress(done, total):
        events.append((f"{done}/{total}", len(list(tmp_path.glob("*.txt")))))

    _run_caption(tmp_path, images, progress=progress)

    assert [e[0] for e in events] == ["1/3", "2/3", "3/3"]
    assert [e[1] for e in events] == [1, 2, 3]


# ---------------------------------------------------------------------------
# REQ-C5（R17-4）同 stem 画像の .txt 共有
# ---------------------------------------------------------------------------


def test_c5_tagging_skips_duplicate_stem_with_reason(tmp_path):
    png = _write_image(tmp_path / "a.png", (8, 8))
    jpg = _write_image(tmp_path / "a.jpg", (16, 16))
    failed: list[Path] = []
    logs: list[tuple[str, str]] = []

    import tagging_core
    changes = tagging_core.process_image_loop(
        tagger=_TaggerStub(), settings=_tagger_settings_dict(tmp_path),
        image_paths=[png, jpg], decision_resolver=None,
        log_gui=lambda m, c: logs.append((m, c)), stop_checker=lambda: False,
        get_string=None, failed_paths=failed,
    )

    # 先に処理した a.png だけが書き、a.jpg は理由付きでスキップ
    assert [c.path.name for c in changes] == ["a.txt"]
    assert failed == []
    assert any("Tag_Skipped_Duplicate_Output" in m for m, _ in logs), logs
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "1girl"


def test_c5_unique_stems_are_all_processed(tmp_path):
    images = [_write_image(tmp_path / f"{name}.png") for name in ("a", "b")]
    changes = _run_tagging(tmp_path, images)
    assert sorted(c.path.name for c in changes) == ["a.txt", "b.txt"]


def test_c5_caption_skips_duplicate_stem(tmp_path):
    png = _write_image(tmp_path / "a.png")
    jpg = _write_image(tmp_path / "a.jpg")
    failed: list[Path] = []
    changes = _run_caption(tmp_path, [png, jpg], failed=failed)
    assert [c.path.name for c in changes] == ["a.txt"]
    assert failed == []


def test_c5_vlm_skips_duplicate_stem(tmp_path, monkeypatch):
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "a.jpg")
    rec = _run_vlm(monkeypatch, _app_settings(tmp_path), runtime=_fake_vlm_runtime())

    assert [c.path.name for c in rec.changes] == ["a.txt"]
    assert any("Batch_Skipped_Duplicate_Output" in m for m, _ in rec.logs), rec.logs
    assert rec.summaries[0]["skipped"] == 1
    assert rec.summaries[0]["written"] + rec.summaries[0]["skipped"] \
        + rec.summaries[0]["unchanged"] + rec.summaries[0]["errors"] == 2


# ---------------------------------------------------------------------------
# REQ-C6（R23-1）Florence の APPEND／PREPEND の冪等性
# ---------------------------------------------------------------------------


def test_c6_reappending_the_same_caption_does_not_grow_the_file(tmp_path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")

    first = _run_caption(tmp_path, [img], placement="APPEND")
    assert txt.read_text(encoding="utf-8") == "old caption, a caption"
    assert len(first) == 1

    before = txt.stat().st_mtime_ns
    second = _run_caption(tmp_path, [img], placement="APPEND")
    assert txt.read_text(encoding="utf-8") == "old caption, a caption"   # 増えない
    assert second == []                                                 # 書き直さない
    assert txt.stat().st_mtime_ns == before


def test_c6_prepend_is_idempotent_and_different_caption_still_joins(tmp_path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("a caption", encoding="utf-8")

    assert _run_caption(tmp_path, [img], placement="PREPEND") == []
    assert txt.read_text(encoding="utf-8") == "a caption"

    changes = _run_caption(tmp_path, [img], placement="APPEND",
                           captioner=_CaptionerStub(caption="another caption"))
    assert len(changes) == 1
    assert txt.read_text(encoding="utf-8") == "a caption, another caption"


def test_c6_caption_with_commas_is_not_split_into_false_duplicates():
    existing = "a red car, parked in a lot"
    assert caption_core.caption_already_present(existing, "parked") is False
    assert caption_core.caption_already_present(existing, existing) is True
    assert caption_core.combine_caption(existing, existing, "APPEND") == existing


# ---------------------------------------------------------------------------
# REQ-C7（R24-1）タガー OVERWRITE の同一内容の書き直し
# ---------------------------------------------------------------------------


def test_c7_overwrite_with_same_content_is_not_rewritten(tmp_path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("1girl", encoding="utf-8")
    before = txt.stat().st_mtime_ns
    summary = _SummaryRecorder()
    failed: list[Path] = []

    changes = _run_tagging(tmp_path, [img], mode=ExistingFileMode.OVERWRITE,
                           failed=failed, get_string=summary)

    assert changes == []                       # FileChange を emit しない = Undo 履歴を食わない
    assert txt.stat().st_mtime_ns == before    # 書き直さない
    assert failed == []
    assert summary.summaries[0]["unchanged"] == 1
    assert summary.summaries[0]["written"] == 0


def test_c7_overwrite_with_different_content_still_writes(tmp_path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old, tags", encoding="utf-8")

    changes = _run_tagging(tmp_path, [img], mode=ExistingFileMode.OVERWRITE)

    assert len(changes) == 1
    assert changes[0].previous_content == "old, tags"
    assert txt.read_text(encoding="utf-8") == "1girl"
