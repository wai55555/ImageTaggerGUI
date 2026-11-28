"""Batch-worker target-filter + batch_failed tests (wave2-t4).

Offline only - no network, no GUI. Run:  rtk pytest tests/test_worker_targets.py -q

Covers TaggerThreadWorker / CaptionerThreadWorker (workers.py) and
VlmCaptionWorker batch path (vlm_worker.py):
- FAILED-rerun processes only the failed images, batch_failed carries them
- SELECTED with None selected -> empty run + exactly one warning log line
- ASK decision_requester is called only for included files
- progress totals are the post-filter count
- VLM single-test path never emits batch_failed
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from PIL import Image

import workers
import caption_core
from tagging_core import ExistingFileMode, OverwriteDecision, TagCategory, TagPrediction, TagResult


def _get_string(section, key, **kwargs):
    return key


def _make_images(d: Path, names=("a", "b", "c"), ext=".jpg"):
    paths = []
    for n in names:
        p = d / f"{n}{ext}"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)
    return sorted(paths)


def _tagger_settings(d: Path, existing_mode="OVERWRITE"):
    import app_settings as A
    s = A.load_settings(A.get_default_config())
    s.paths.input_dir = str(d)
    s.behavior.existing_file_mode = existing_mode
    return s


def _tagger_settings_dict(d: Path, existing_mode=ExistingFileMode.OVERWRITE):
    return {
        "INPUT_DIR": Path(d),
        "EXISTING_FILE_MODE": existing_mode,
        "TAG_THRESHOLDS": {},
        "MAX_TAGS_PER_CATEGORY": {},
        "ENABLE_SOLO_LIMIT": False,
        "CONVERT_UNDERSCORE": False,
    }


class _FailSecondTagger:
    """Stub tagger: 2nd infer_batch call raises (targets the middle image)."""

    def __init__(self):
        self.tag_meta_lookup = {}
        self.calls = 0

    def infer_batch(self, images, thresholds=None, max_tags=None):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")
        return [TagResult(tags=[TagPrediction("cat", 0.9, TagCategory.GENERAL)])]


class _OkTagger:
    def __init__(self):
        self.tag_meta_lookup = {}

    def infer_batch(self, images, thresholds=None, max_tags=None):
        return [TagResult(tags=[TagPrediction("cat", 0.9, TagCategory.GENERAL)])]


def _run_tagger(s, monkeypatch, tagger, decision=None,
                mode=ExistingFileMode.OVERWRITE):
    monkeypatch.setattr(workers, "setup_tagger_from_settings",
                        lambda *a, **k: (tagger, _tagger_settings_dict(
                            Path(s.paths.input_dir), mode)))
    monkeypatch.setattr(workers, "ensure_pixai_tags_csv", lambda *a, **k: True)
    w = workers.TaggerThreadWorker(s, decision, get_string=_get_string)
    return w


def _collect(w):
    logs, changes, failed, totals = [], [], [], []
    w.log_message.connect(lambda m, c: logs.append((m, c)))
    w.batch_completed.connect(lambda lst: changes.extend(lst))
    w.batch_failed.connect(lambda lst: failed.extend(lst))
    w.progress_update.connect(lambda done, total: totals.append((done, total)))
    return logs, changes, failed, totals


def test_tagger_failed_rerun_processes_only_failures(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    s = _tagger_settings(d)
    w = _run_tagger(s, monkeypatch, _FailSecondTagger())
    logs, changes, failed, totals = _collect(w)
    w.run_tagging()
    assert [p.name for p in failed] == ["b.jpg"], failed
    assert len(changes) == 2, [c.path.name for c in changes]
    print("  tagger run1 batch_failed carries only b.jpg: OK")

    s2 = _tagger_settings(d)
    s2.behavior.target_mode = "FAILED"
    w2 = workers.TaggerThreadWorker(s2, None, get_string=_get_string,
                                    failed_paths=[d / "b.jpg"])
    monkeypatch.setattr(workers, "setup_tagger_from_settings",
                        lambda *a, **k: (_OkTagger(), _tagger_settings_dict(d)))
    monkeypatch.setattr(workers, "ensure_pixai_tags_csv", lambda *a, **k: True)
    logs2, changes2, failed2, totals2 = _collect(w2)
    w2.run_tagging()
    assert [c.path.name for c in changes2] == ["b.txt"], [c.path.name for c in changes2]
    assert failed2 == [], failed2
    assert totals2 and all(t == 1 for _, t in totals2), totals2
    assert totals2[-1][0] == 1, totals2
    print("  tagger FAILED rerun processes only b.jpg, progress total 1: OK")


def test_tagger_selected_none_empty_run(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    s = _tagger_settings(d)
    s.behavior.target_mode = "SELECTED"
    w = _run_tagger(s, monkeypatch, _OkTagger())
    logs, changes, failed, totals = _collect(w)
    w.run_tagging()
    warn = [(m, c) for m, c in logs if "Warning_No_Image_Files" in m]
    assert len(warn) == 1 and warn[0][1] == "orange", logs
    assert changes == [] and totals == [], (changes, totals)
    print("  tagger SELECTED+None empty run + one warning line: OK")


def test_tagger_ask_called_only_for_included_files(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    (d / "a.txt").write_text("old a", encoding="utf-8")
    (d / "b.txt").write_text("old b", encoding="utf-8")
    s = _tagger_settings(d)
    s.behavior.target_mode = "FAILED"
    asked = []
    w = workers.TaggerThreadWorker(
        s, lambda p: asked.append(p) or OverwriteDecision.OVERWRITE,
        get_string=_get_string, failed_paths=[d / "b.jpg"])
    monkeypatch.setattr(workers, "setup_tagger_from_settings",
                        lambda *a, **k: (_OkTagger(), _tagger_settings_dict(
                            d, ExistingFileMode.ASK)))
    monkeypatch.setattr(workers, "ensure_pixai_tags_csv", lambda *a, **k: True)
    w.run_tagging()
    assert asked == [d / "b.txt"], asked
    print("  tagger ASK reached only the FAILED-included file: OK")


def test_tagger_stop_does_not_report_skip_outputs_as_failed(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    (d / "a.txt").write_text("already tagged", encoding="utf-8")
    s = _tagger_settings(d, existing_mode="SKIP")
    w = _run_tagger(s, monkeypatch, _OkTagger(), mode=ExistingFileMode.SKIP)
    logs, changes, failed, totals = _collect(w)
    w.stop()
    w.run_tagging()
    assert [p.name for p in failed] == ["b.jpg", "c.jpg"], failed
    assert changes == []


def _captioner_settings_dict(d: Path, existing_mode=ExistingFileMode.OVERWRITE):
    return {
        "INPUT_DIR": Path(d),
        "TASK": "t",
        "EXISTING_FILE_MODE": existing_mode,
        "CAPTION_PLACEMENT": "OVERWRITE",
    }


def _fake_captioner(fail_second=False):
    state = {"calls": 0}

    def _generate(image, prompt, stop_checker):
        state["calls"] += 1
        if fail_second and state["calls"] == 2:
            raise RuntimeError("boom")
        return "a plain caption", False

    cap = SimpleNamespace(
        config=SimpleNamespace(default_task="t", tasks={"t": "describe"}),
        generate=_generate,
    )
    return cap


def _run_captioner(s, monkeypatch, cap, mode=ExistingFileMode.OVERWRITE, decision=None,
                   failed_paths=(), selected=None):
    d = Path(s.paths.input_dir)
    monkeypatch.setattr(caption_core, "setup_captioner_from_settings",
                        lambda *a, **k: (cap, _captioner_settings_dict(d, mode)))
    w = workers.CaptionerThreadWorker(s, decision, get_string=_get_string,
                                      selected_file_path=selected,
                                      failed_paths=failed_paths)
    return w


def test_captioner_failed_rerun_processes_only_failures(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    s = _tagger_settings(d)
    w = _run_captioner(s, monkeypatch, _fake_captioner(fail_second=True))
    logs, changes, failed, totals = _collect(w)
    w.run_captioning()
    assert [p.name for p in failed] == ["b.jpg"], failed
    assert len(changes) == 2
    print("  captioner run1 batch_failed carries only b.jpg: OK")

    s2 = _tagger_settings(d)
    s2.behavior.target_mode = "FAILED"
    w2 = _run_captioner(s2, monkeypatch, _fake_captioner(),
                        decision=None, failed_paths=[d / "b.jpg"])
    logs2, changes2, failed2, totals2 = _collect(w2)
    w2.run_captioning()
    assert [c.path.name for c in changes2] == ["b.txt"]
    assert failed2 == [], failed2
    assert totals2 and all(t == 1 for _, t in totals2), totals2
    print("  captioner FAILED rerun processes only b.jpg, progress total 1: OK")


def test_captioner_failure_with_existing_output_is_reported(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    for name in ("a", "b", "c"):
        (d / f"{name}.txt").write_text("old caption", encoding="utf-8")
    s = _tagger_settings(d)
    w = _run_captioner(s, monkeypatch, _fake_captioner(fail_second=True))
    logs, changes, failed, totals = _collect(w)
    w.run_captioning()
    assert [p.name for p in failed] == ["b.jpg"], failed
    assert [c.path.name for c in changes] == ["a.txt", "c.txt"]


def test_captioner_selected_none_empty_run(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    s = _tagger_settings(d)
    s.behavior.target_mode = "SELECTED"
    w = _run_captioner(s, monkeypatch, _fake_captioner())
    logs, changes, failed, totals = _collect(w)
    w.run_captioning()
    warn = [(m, c) for m, c in logs if "Warning_No_Image_Files" in m]
    assert len(warn) == 1 and warn[0][1] == "orange", logs
    assert changes == [] and totals == []
    print("  captioner SELECTED+None empty run + one warning line: OK")


def test_captioner_ask_called_only_for_included_files(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d)
    (d / "a.txt").write_text("old a", encoding="utf-8")
    (d / "b.txt").write_text("old b", encoding="utf-8")
    s = _tagger_settings(d)
    s.behavior.target_mode = "FAILED"
    asked = []
    w = _run_captioner(s, monkeypatch, _fake_captioner(), mode=ExistingFileMode.ASK,
                       decision=lambda p: asked.append(p) or OverwriteDecision.OVERWRITE,
                       failed_paths=[d / "b.jpg"])
    w.run_captioning()
    assert asked == [d / "b.txt"], asked
    print("  captioner ASK reached only the FAILED-included file: OK")


def _vlm_settings(d: Path, existing_mode="OVERWRITE"):
    import app_settings as A
    s = A.load_settings(A.get_default_config())
    s.paths.input_dir = str(d)
    s.behavior.existing_file_mode = existing_mode
    s.caption.placement = "OVERWRITE"
    return s


def _fake_vlm_runtime(fail_second=False):
    from vlm_image import ImagePreprocessConfig
    state = {"calls": 0}

    def _caption_one(spec_base, ids):
        state["calls"] += 1
        if fail_second and state["calls"] == 2:
            return SimpleNamespace(ok=False, text="", connection_id="c1",
                                   stopped=False, stop_job=False, attempts=[],
                                   error=SimpleNamespace(
                                       reason=SimpleNamespace(value="timeout")))
        return SimpleNamespace(ok=True, text="hello world", connection_id="c1",
                               model_id="m1",
                               stopped=False, stop_job=False, attempts=[], error=None)

    executor = SimpleNamespace(
        live_candidates=lambda ids: ids,
        caption_one=_caption_one,
    )
    candidates = SimpleNamespace(connection_ids=["c1"], has_candidates=True, excluded={})
    return {
        "connections": {}, "gen_profile": None, "policy": None,
        "candidates": candidates, "executor": executor,
        # REQ-F4: `_build_runtime` は検証記録用に開始時点のプロファイル id を返す。
        "model_profile_id": "gemma-4-26b-a4b-it",
        "system_prompt": "", "user_prompt": "",
        "image_cfg": ImagePreprocessConfig(max_long_edge=512, fmt="JPEG", jpeg_quality=80),
    }


def _run_vlm(s, monkeypatch, rt, decision=None, failed_paths=(), selected=None,
             single_test=False):
    from vlm_worker import VlmCaptionWorker
    monkeypatch.setattr(VlmCaptionWorker, "_build_runtime", lambda self: rt)
    w = VlmCaptionWorker(s, decision, get_string=_get_string,
                         selected_file_path=selected, single_test=single_test,
                         failed_paths=failed_paths)
    return w


def test_vlm_failed_rerun_processes_only_failures(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d, ext=".png")
    s = _vlm_settings(d)
    w = _run_vlm(s, monkeypatch, _fake_vlm_runtime(fail_second=True))
    logs, changes, failed, totals = _collect(w)
    w.run_captioning()
    assert [p.name for p in failed] == ["b.png"], failed
    assert len(changes) == 2
    print("  vlm run1 batch_failed carries only b.png: OK")

    s2 = _vlm_settings(d)
    s2.behavior.target_mode = "FAILED"
    w2 = _run_vlm(s2, monkeypatch, _fake_vlm_runtime(),
                  failed_paths=[d / "b.png"])
    logs2, changes2, failed2, totals2 = _collect(w2)
    w2.run_captioning()
    assert [c.path.name for c in changes2] == ["b.txt"]
    assert failed2 == [], failed2
    assert totals2 and all(t == 1 for _, t in totals2), totals2
    assert totals2[-1] == (1, 1), totals2
    print("  vlm FAILED rerun processes only b.png, progress total 1: OK")


def test_vlm_selected_none_empty_run(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d, ext=".png")
    s = _vlm_settings(d)
    s.behavior.target_mode = "SELECTED"
    w = _run_vlm(s, monkeypatch, _fake_vlm_runtime())
    logs, changes, failed, totals = _collect(w)
    w.run_captioning()
    assert len(logs) == 1 and "Warn_No_Images" in logs[0][0]
    assert changes == [] and failed == [] and totals == []
    print("  vlm SELECTED+None empty run + one warning line: OK")


def test_vlm_ask_called_only_for_included_files(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d, ext=".png")
    (d / "a.txt").write_text("old a", encoding="utf-8")
    (d / "b.txt").write_text("old b", encoding="utf-8")
    s = _vlm_settings(d, existing_mode="ASK")
    s.behavior.target_mode = "FAILED"
    asked = []
    w = _run_vlm(s, monkeypatch, _fake_vlm_runtime(),
                 decision=lambda p: asked.append(p) or OverwriteDecision.OVERWRITE,
                 failed_paths=[d / "b.png"])
    w.run_captioning()
    assert asked == [d / "b.txt"], asked
    print("  vlm ASK reached only the FAILED-included file: OK")


def test_vlm_single_test_never_emits_batch_failed(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d, ext=".png")
    s = _vlm_settings(d)
    w = _run_vlm(s, monkeypatch, _fake_vlm_runtime(),
                 selected=d / "a.png", single_test=True)
    failed = []
    w.batch_failed.connect(lambda lst: failed.append(lst))
    changes = []
    w.batch_completed.connect(lambda lst: changes.extend(lst))
    w.run_captioning()
    assert failed == [], failed
    assert len(changes) == 1
    print("  vlm single test emits batch_completed but never batch_failed: OK")


def test_vlm_single_test_reports_cancellation_once(tmp_path, monkeypatch):
    d = tmp_path
    images = _make_images(d, names=("a",), ext=".png")
    s = _vlm_settings(d)
    rt = _fake_vlm_runtime()
    rt["executor"].caption_one = lambda spec, ids: SimpleNamespace(
        ok=False, text="", connection_id="", model_id="", stopped=True,
        stop_job=False, attempts=[], error=None)
    w = _run_vlm(s, monkeypatch, rt, selected=images[0], single_test=True)
    logs, _, _, _ = _collect(w)
    w.run_captioning()
    assert [message for message, _ in logs if message == "Stopped_By_User"] == [
        "Stopped_By_User"]


def test_vlm_exhaustion_does_not_fail_skip_outputs(tmp_path, monkeypatch):
    d = tmp_path
    _make_images(d, ext=".png")
    (d / "a.txt").write_text("existing", encoding="utf-8")
    (d / "c.txt").write_text("existing", encoding="utf-8")
    s = _vlm_settings(d, existing_mode="SKIP")
    rt = _fake_vlm_runtime()
    rt["executor"].live_candidates = lambda ids: []
    w = _run_vlm(s, monkeypatch, rt)
    logs, _, failed, _ = _collect(w)
    w.run_captioning()
    assert [p.name for p in failed] == ["b.png"], failed
    assert sum("All_Connections_Exhausted" in message for message, _ in logs) == 1


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
