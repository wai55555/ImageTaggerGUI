"""Phase 3（バッチ処理ループの共通化）の特性テスト。

共通化の**前後で無修正で通る**ことを目的に、3 ループの外から観測できる挙動だけを固定する:
(a) 書かれるファイルと内容、(b) failed の集合と並び・重複、(c) サマリ件数、(d) GUI ログ、
(e) 進捗コールバックの回数と最終到達、(f) FileChange の全フィールド。

VLM（`VlmCaptionWorker._run_batch`）は接続枯渇・stop_job・finally での部分結果 emit など
固有の制御フローを持つため、REQ-3.2 の範囲（既存ファイル事前判定・進捗・停止・失敗収集）
だけを対象に、ここでは「ループ骨格へ寄せない」現行挙動を固定する。
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import caption_core
import tagging_core
from tagging_core import (
    ExistingFileMode,
    FileChange,
    OverwriteDecision,
    TagCategory,
    TagPrediction,
    TagResult,
)

# ---------------------------------------------------------------------------
# ハーネス
# ---------------------------------------------------------------------------


def _write_image(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (4, 4)).save(path)
    return path


class _FakeTagger:
    """process_image_loop が使う最小のタガー（生成だけを差し替える）。"""

    tag_meta_lookup: dict = {}

    def __init__(self, tags=("1girl",), raise_error: bool = False):
        self._tags = list(tags)
        self._raise = raise_error

    def infer_batch(self, images, thresholds=None, max_tags=None):
        if self._raise:
            raise RuntimeError("simulated inference failure")
        return [TagResult(tags=[TagPrediction(name=t, score=0.9, category=TagCategory.GENERAL)
                                for t in self._tags])]


class _FakeCaptioner:
    config = SimpleNamespace(default_task="t", tasks={"t": "<t>"})

    def __init__(self, caption: str = "a caption", cancelled: bool = False,
                 raise_error: bool = False):
        self._caption = caption
        self._cancelled = cancelled
        self._raise = raise_error

    def generate(self, image, task_prompt, stop_checker):
        if self._raise:
            raise RuntimeError("simulated generation failure")
        return self._caption, self._cancelled


def _tagging_settings(input_dir: Path, mode: ExistingFileMode):
    return {
        "INPUT_DIR": input_dir,
        "EXISTING_FILE_MODE": mode,
        "TAG_THRESHOLDS": {},
        "MAX_TAGS_PER_CATEGORY": {},
        "ENABLE_SOLO_LIMIT": False,
        "CONVERT_UNDERSCORE": True,
    }


def _caption_settings(input_dir: Path, mode: ExistingFileMode, placement: str = "OVERWRITE"):
    return {
        "INPUT_DIR": input_dir,
        "EXISTING_FILE_MODE": mode,
        "CAPTION_PLACEMENT": placement,
        "TASK": "t",
    }


class _Recorder:
    """進捗・GUI ログ・失敗パスをまとめて記録する。"""

    def __init__(self):
        self.progress: list[tuple[int, int]] = []
        self.gui: list[tuple[str, str]] = []
        self.failed: list[Path] = []

    def __call__(self, done, total):  # progress_cb
        self.progress.append((done, total))


def _run_tagging(tmp_path: Path, images, *, mode=ExistingFileMode.OVERWRITE,
                 resolver=None, stop=None, tagger=None, rec=None):
    rec = rec or _Recorder()
    changed = tagging_core.process_image_loop(
        tagger=tagger or _FakeTagger(),
        settings=_tagging_settings(tmp_path, mode),
        image_paths=list(images),
        decision_resolver=resolver,
        log_gui=lambda m, c: rec.gui.append((m, c)),
        stop_checker=stop or (lambda: False),
        get_string=None,
        progress_cb=rec,
        failed_paths=rec.failed,
    )
    return changed, rec


def _run_caption(tmp_path: Path, images, *, mode=ExistingFileMode.OVERWRITE,
                 placement="OVERWRITE", resolver=None, stop=None, captioner=None, rec=None):
    rec = rec or _Recorder()
    changed = caption_core.process_caption_loop(
        captioner=captioner or _FakeCaptioner(),
        settings=_caption_settings(tmp_path, mode, placement),
        image_paths=list(images),
        decision_resolver=resolver,
        log_gui=lambda m, c: rec.gui.append((m, c)),
        stop_checker=stop or (lambda: False),
        get_string=None,
        progress_cb=rec,
        failed_paths=rec.failed,
    )
    return changed, rec


# ---------------------------------------------------------------------------
# tagging: モード × 既存ファイルの有無
# ---------------------------------------------------------------------------


def test_tagging_overwrite_writes_new_file(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    changed, rec = _run_tagging(tmp_path, [img])
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == "1girl"
    assert len(changed) == 1
    change = changed[0]
    assert change.path == img.with_suffix(".txt")
    assert change.previous_content is None
    assert change.new_content == "1girl"
    assert change.was_append is False
    assert change.added_tags == ()
    assert rec.failed == []
    assert rec.progress == [(1, 1)]


def test_tagging_overwrite_replaces_existing(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old, tags", encoding="utf-8")
    changed, _ = _run_tagging(tmp_path, [img])
    assert txt.read_text(encoding="utf-8") == "1girl"
    assert changed[0].previous_content == "old, tags"
    assert changed[0].was_append is False


def test_tagging_skip_leaves_existing_file(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("keep me", encoding="utf-8")
    changed, rec = _run_tagging(tmp_path, [img], mode=ExistingFileMode.SKIP)
    assert changed == []
    assert txt.read_text(encoding="utf-8") == "keep me"
    assert rec.failed == []


def test_tagging_append_merges_and_reports_added_tags(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old", encoding="utf-8")
    changed, _ = _run_tagging(tmp_path, [img], mode=ExistingFileMode.APPEND)
    assert txt.read_text(encoding="utf-8") == "old, 1girl"
    assert changed[0].was_append is True
    assert changed[0].added_tags == ("1girl",)


def test_tagging_append_without_new_tags_is_unchanged(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("1girl", encoding="utf-8")
    before = txt.stat().st_mtime_ns
    changed, rec = _run_tagging(tmp_path, [img], mode=ExistingFileMode.APPEND)
    assert changed == []
    assert rec.failed == []
    assert txt.stat().st_mtime_ns == before


def test_tagging_ask_resolver_skip(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("keep me", encoding="utf-8")
    asked: list[Path] = []

    def resolver(path):
        asked.append(path)
        return OverwriteDecision.SKIP

    changed, _ = _run_tagging(tmp_path, [img], mode=ExistingFileMode.ASK, resolver=resolver)
    assert changed == []
    assert asked == [txt]
    assert txt.read_text(encoding="utf-8") == "keep me"


def test_tagging_ask_resolver_append(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old", encoding="utf-8")
    changed, _ = _run_tagging(tmp_path, [img], mode=ExistingFileMode.ASK,
                              resolver=lambda p: OverwriteDecision.APPEND)
    assert txt.read_text(encoding="utf-8") == "old, 1girl"
    assert changed[0].was_append is True


def test_tagging_ask_without_resolver_touches_nothing(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("keep me", encoding="utf-8")
    changed, rec = _run_tagging(tmp_path, [img], mode=ExistingFileMode.ASK, resolver=None)
    assert changed == [] and rec.failed == []
    assert txt.read_text(encoding="utf-8") == "keep me"


def test_tagging_unreadable_existing_file_is_skipped(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_bytes(b"\xff\xfe\x00bad")
    changed, rec = _run_tagging(tmp_path, [img])
    assert changed == []
    assert rec.failed == [img]
    assert txt.read_bytes() == b"\xff\xfe\x00bad"


# ---------------------------------------------------------------------------
# tagging: 停止・失敗・空生成
# ---------------------------------------------------------------------------


def test_tagging_stop_at_loop_start_marks_remaining_failed(tmp_path: Path):
    images = [_write_image(tmp_path / f"{i}.png") for i in range(3)]
    changed, rec = _run_tagging(tmp_path, images, stop=lambda: True)
    assert changed == []
    assert rec.failed == images          # 並びも元の順序
    assert rec.progress == [(1, 3)]      # 進捗は停止判定より先に 1 回出る


def test_tagging_stop_before_ask_resolution_does_not_call_resolver(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    img.with_suffix(".txt").write_text("keep me", encoding="utf-8")
    called: list[Path] = []

    def resolver(path):
        called.append(path)
        return OverwriteDecision.SKIP

    changed, rec = _run_tagging(tmp_path, [img], mode=ExistingFileMode.ASK,
                                resolver=resolver, stop=lambda: True)
    assert called == []                  # resolver（GUI 往復）を呼ばない
    assert changed == []
    assert rec.failed == [img]           # SKIP モードではないので失敗に数える


def test_tagging_ask_stop_after_resolver_processes_that_item(tmp_path: Path):
    """resolver 復帰後の停止再チェックは現行どおり存在しない（停止は次の周回で効く）。"""
    images = [_write_image(tmp_path / "a.png"), _write_image(tmp_path / "b.png")]
    for p in images:
        p.with_suffix(".txt").write_text("old", encoding="utf-8")
    calls = {"n": 0}

    def resolver(path):
        calls["n"] += 1
        return OverwriteDecision.OVERWRITE

    def stop():
        return calls["n"] >= 1           # 1 件解決したら停止要求が立つ

    changed, rec = _run_tagging(tmp_path, images, mode=ExistingFileMode.ASK,
                                resolver=resolver, stop=stop)
    assert images[0].with_suffix(".txt").read_text(encoding="utf-8") == "1girl"
    assert len(changed) == 1
    assert rec.failed == [images[1]]
    assert calls["n"] == 1               # 復帰後に再チェックしないので resolve は 1 回だけ


def test_tagging_skip_mode_stop_does_not_count_existing_as_failed(tmp_path: Path):
    images = [_write_image(tmp_path / "a.png"), _write_image(tmp_path / "b.png")]
    images[0].with_suffix(".txt").write_text("done", encoding="utf-8")
    changed, rec = _run_tagging(tmp_path, images, mode=ExistingFileMode.SKIP, stop=lambda: True)
    assert changed == []
    assert rec.failed == [images[1]]     # 既存 .txt がある a は失敗に数えない


def test_tagging_inference_failure_marks_failed(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    changed, rec = _run_tagging(tmp_path, [img], tagger=_FakeTagger(raise_error=True))
    assert changed == []
    assert rec.failed == [img]
    assert not img.with_suffix(".txt").exists()


def test_tagging_zero_tags_is_failure_unless_all_limits_zero(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    changed, rec = _run_tagging(tmp_path, [img], tagger=_FakeTagger(tags=()))
    assert changed == [] and rec.failed == [img]

    rec2 = _Recorder()
    changed2 = tagging_core.process_image_loop(
        tagger=_FakeTagger(tags=()),
        settings={**_tagging_settings(tmp_path, ExistingFileMode.OVERWRITE),
                  "MAX_TAGS_PER_CATEGORY": {TagCategory.GENERAL: 0}},
        image_paths=[img], decision_resolver=None, log_gui=lambda m, c: None,
        stop_checker=lambda: False, get_string=None, progress_cb=rec2,
        failed_paths=rec2.failed,
    )
    # 上限 0 の空は失敗ではなく「タグ無しで成功」として空ファイルを書く（現行挙動）
    assert rec2.failed == []
    assert len(changed2) == 1 and changed2[0].new_content == ""
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == ""


def test_tagging_failed_paths_dedupe(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    rec = _Recorder()
    rec.failed.append(img)                       # 既に失敗として記録済み
    changed, _ = _run_tagging(tmp_path, [img], tagger=_FakeTagger(raise_error=True), rec=rec)
    assert changed == []
    assert rec.failed == [img]                   # 重複しない


def test_tagging_out_of_input_dir_raises_relative_to_error(tmp_path: Path):
    """現行挙動の固定: 入力ディレクトリ外のパスは relative_to の例外になる。"""
    outside = _write_image(tmp_path.parent / "outside.png")
    with pytest.raises(ValueError):
        _run_tagging(tmp_path, [outside])


def test_tagging_fatal_error_propagates_without_changes(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")

    class Exploding(_FakeTagger):
        def infer_batch(self, images, thresholds=None, max_tags=None):
            raise KeyboardInterrupt  # ループの except Exception では捕まらない

    with pytest.raises(KeyboardInterrupt):
        _run_tagging(tmp_path, [img], tagger=Exploding())


def test_tagging_summary_line_counts(tmp_path: Path):
    images = [_write_image(tmp_path / f"{i}.png") for i in range(3)]
    images[0].with_suffix(".txt").write_text("done", encoding="utf-8")
    changed, rec = _run_tagging(tmp_path, images, mode=ExistingFileMode.SKIP)
    assert len(changed) == 2
    summary = [msg for msg, color in rec.gui if color == "blue"]
    # get_string 未指定のテストではキーがそのまま返る（件数は本番の lang 側で埋まる）
    assert summary == ["Batch_Summary"]


# ---------------------------------------------------------------------------
# caption: モード × placement（決定は SKIP 判定にしか効かない現行挙動）
# ---------------------------------------------------------------------------


def test_caption_overwrite_writes_new_file(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    changed, rec = _run_caption(tmp_path, [img])
    assert img.with_suffix(".txt").read_text(encoding="utf-8") == "a caption"
    assert changed[0].was_append is False
    assert changed[0].previous_content is None
    assert rec.progress == [(1, 1)]


def test_caption_appending_to_existing_file(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")
    changed, _ = _run_caption(tmp_path, [img], placement="APPEND")
    assert txt.read_text(encoding="utf-8") == "old caption, a caption"
    assert changed[0].was_append is True


def test_caption_ask_decision_does_not_override_placement(tmp_path: Path):
    """現行挙動: ASK で APPEND を選んでも placement=OVERWRITE なら既存を捨てる。"""
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")
    changed, _ = _run_caption(tmp_path, [img], mode=ExistingFileMode.ASK,
                              placement="OVERWRITE",
                              resolver=lambda p: OverwriteDecision.APPEND)
    assert txt.read_text(encoding="utf-8") == "a caption"
    assert changed[0].was_append is False


def test_caption_append_mode_with_overwrite_placement_is_treated_as_append(tmp_path: Path):
    """mode=APPEND と placement=OVERWRITE の矛盾は APPEND に補正される（現行仕様）。"""
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")
    changed, _ = _run_caption(tmp_path, [img], mode=ExistingFileMode.APPEND, placement="OVERWRITE")
    assert txt.read_text(encoding="utf-8") == "old caption, a caption"
    assert changed[0].was_append is True


def test_caption_skip_mode_keeps_existing(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("keep", encoding="utf-8")
    changed, rec = _run_caption(tmp_path, [img], mode=ExistingFileMode.SKIP)
    assert changed == [] and rec.failed == []
    assert txt.read_text(encoding="utf-8") == "keep"


def test_caption_empty_generation_is_failure(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("keep", encoding="utf-8")
    changed, rec = _run_caption(tmp_path, [img], captioner=_FakeCaptioner(caption="   "))
    assert changed == [] and rec.failed == [img]
    assert txt.read_text(encoding="utf-8") == "keep"


def test_caption_cancelled_stops_loop_without_partial_write(tmp_path: Path):
    images = [_write_image(tmp_path / "a.png"), _write_image(tmp_path / "b.png")]
    changed, rec = _run_caption(tmp_path, images, captioner=_FakeCaptioner(cancelled=True))
    assert changed == []
    assert rec.failed == images            # 残件を failed へ寄せて抜ける
    assert not images[0].with_suffix(".txt").exists()


def test_caption_unchanged_content_is_not_counted_as_write(tmp_path: Path):
    img = _write_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("a caption", encoding="utf-8")
    before = txt.stat().st_mtime_ns
    changed, rec = _run_caption(tmp_path, [img], placement="OVERWRITE")
    assert changed == []
    assert txt.stat().st_mtime_ns == before


def test_caption_out_of_input_dir_raises_relative_to_error(tmp_path: Path):
    outside = _write_image(tmp_path.parent / "outside2.png")
    with pytest.raises(ValueError):
        _run_caption(tmp_path, [outside])
