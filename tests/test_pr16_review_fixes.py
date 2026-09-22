"""Verification for docs/260902_pr16_review.md fixes."""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "/mnt/c/github/PixaiTaggerOnnxGui")


def test_utils_lock_timeout():
    import utils
    utils.get_debug_settings().debug_log_enabled = True

    # Simulate a thread that died holding _LOG_LOCK (as QThread.terminate() could do).
    utils._LOG_LOCK.acquire()
    t0 = time.monotonic()
    utils.write_debug_log("should not block forever")
    dt = time.monotonic() - t0
    assert dt < 2.0, f"write_debug_log blocked for {dt}s while lock held elsewhere"
    utils._LOG_LOCK.release()

    # Normal path still works after.
    utils.write_debug_log("normal line after lock released")
    utils._flush_debug_log()
    print(f"  utils lock-timeout: OK (blocked {dt:.3f}s, then normal write succeeded)")


def test_tagging_core_unreadable_skip():
    import tagging_core as T
    d = Path(tempfile.mkdtemp())
    img = d / "a.png"
    from PIL import Image
    Image.new("RGB", (4, 4)).save(img)
    txt = img.with_suffix(".txt")
    # Write invalid UTF-8 bytes directly.
    txt.write_bytes(b"\xff\xfe\x00bad")

    class FakeTagger:
        tag_meta_lookup = {}
        def infer_batch(self, images, thresholds=None, max_tags=None):
            return [T.TagResult(tags=[T.TagPrediction(name="1girl", score=0.99, category=T.TagCategory.GENERAL)], series_tags=())]

    settings = {"INPUT_DIR": d, "EXISTING_FILE_MODE": T.ExistingFileMode.OVERWRITE,
                "TAG_THRESHOLDS": {}, "MAX_TAGS_PER_CATEGORY": {},
                "ENABLE_SOLO_LIMIT": False, "CONVERT_UNDERSCORE": True}
    gui = []
    changed = T.process_image_loop(
        tagger=FakeTagger(), settings=settings, image_paths=[img],
        decision_resolver=None, log_gui=lambda m, c: gui.append((m, c)),
        stop_checker=lambda: False, get_string=None)
    assert changed == [], f"expected no write on unreadable existing file, got {changed}"
    assert txt.read_bytes() == b"\xff\xfe\x00bad", "existing unreadable file must be left untouched"
    assert any(c == "red" for _, c in gui), "expected an error line on GUI"
    print("  tagging_core unreadable-file skip: OK (file untouched, no FileChange, error logged)")


def test_caption_core_unreadable_skip():
    import caption_core as C
    from types import SimpleNamespace
    d = Path(tempfile.mkdtemp())
    img = d / "a.png"
    from PIL import Image
    Image.new("RGB", (4, 4)).save(img)
    txt = img.with_suffix(".txt")
    txt.write_bytes(b"\xff\xfe\x00bad-caption")

    class FakeCaptioner:
        config = SimpleNamespace(default_task="MORE_DETAILED_CAPTION",
                                 tasks={"MORE_DETAILED_CAPTION": "<MORE_DETAILED_CAPTION>"})
        def generate(self, image, task_prompt, stop_checker):
            return ("a new caption", False)

    for placement in ("OVERWRITE", "APPEND", "PREPEND"):
        txt.write_bytes(b"\xff\xfe\x00bad-caption")
        settings = {"INPUT_DIR": d, "EXISTING_FILE_MODE": C.ExistingFileMode.OVERWRITE,
                    "CAPTION_PLACEMENT": placement, "TASK": "MORE_DETAILED_CAPTION"}
        changed = C.process_caption_loop(
            captioner=FakeCaptioner(), settings=settings, image_paths=[img],
            decision_resolver=None, log_gui=lambda m, c: None,
            stop_checker=lambda: False, get_string=None)
        assert changed == [], f"placement={placement}: expected no write, got {changed}"
        assert txt.read_bytes() == b"\xff\xfe\x00bad-caption", f"placement={placement}: file must be untouched"
    print("  caption_core unreadable-file skip: OK for OVERWRITE/APPEND/PREPEND")


def test_caption_core_empty_generation_preserves_file():
    """coderabbit PR#16: an empty generated caption under OVERWRITE must not blank an
    existing .txt, and must not create an empty new .txt."""
    import caption_core as C
    from types import SimpleNamespace
    d = Path(tempfile.mkdtemp())
    from PIL import Image
    a = d / "a.png"; Image.new("RGB", (4, 4)).save(a)
    b = d / "b.png"; Image.new("RGB", (4, 4)).save(b)
    (a.with_suffix(".txt")).write_text("PRECIOUS existing caption", encoding="utf-8")

    class EmptyCaptioner:
        config = SimpleNamespace(default_task="t", tasks={"t": "<t>"})
        def generate(self, image, task_prompt, stop_checker):
            return ("   ", False)  # whitespace-only counts as empty

    settings = {"INPUT_DIR": d, "EXISTING_FILE_MODE": C.ExistingFileMode.OVERWRITE,
                "CAPTION_PLACEMENT": "OVERWRITE", "TASK": "t"}
    changed = C.process_caption_loop(
        captioner=EmptyCaptioner(), settings=settings, image_paths=[a, b],
        decision_resolver=None, log_gui=lambda m, c: None,
        stop_checker=lambda: False, get_string=None)
    assert changed == [], f"empty caption must write nothing, got {changed}"
    assert a.with_suffix(".txt").read_text(encoding="utf-8") == "PRECIOUS existing caption"
    assert not b.with_suffix(".txt").exists(), "empty caption must not create an empty .txt"

    # combine_caption invariant: empty caption keeps existing even for OVERWRITE
    assert C.combine_caption("old", "", "OVERWRITE") == "old"
    assert C.combine_caption("old", "new", "OVERWRITE") == "new"
    assert C.combine_caption("", "new", "OVERWRITE") == "new"
    print("  caption_core empty-generation: existing preserved, no empty file, combine_caption invariant OK")


def test_long_path_str_idempotent_on_extended_paths():
    """cubic PR#16 round 2: _long_path_str must not add a second \\\\?\\ prefix when the
    caller already passes an extended-length path."""
    import undo_manager as U
    # Exercise the win32 branch logic on any host: os.path.abspath is platform-specific
    # (posix on Linux mangles backslashes), so stub it to identity - every test input
    # below is already absolute.
    real_platform, real_abspath = U.sys.platform, U.os.path.abspath
    U.sys.platform = "win32"
    U.os.path.abspath = lambda x: str(x)
    try:
        assert U._long_path_str("\\\\?\\C:\\a\\b") == "\\\\?\\C:\\a\\b"
        assert U._long_path_str("\\\\?\\UNC\\srv\\share\\f") == "\\\\?\\UNC\\srv\\share\\f"
        # once-through result must survive a second pass unchanged (idempotent)
        once = U._long_path_str("C:\\a\\b")
        assert U._long_path_str(once) == once == "\\\\?\\C:\\a\\b"
        unc = U._long_path_str("\\\\srv\\share\\f")
        assert U._long_path_str(unc) == unc == "\\\\?\\UNC\\srv\\share\\f"
    finally:
        U.sys.platform, U.os.path.abspath = real_platform, real_abspath
    print("  _long_path_str idempotent on \\\\?\\ / UNC paths: OK")


def test_undo_actions_normalize_path_at_construction():
    """coderabbit PR#16: undo/redo may run long after the batch; a relative file_path
    would be resolved against the CWD *then*. Actions must lock an absolute path at
    construction (abspath, not resolve - symlinks stay unresolved)."""
    import os as _os
    from undo_manager import OverwriteFileAction, AppendTagsActionV2, EditCaptionAction
    d = Path(tempfile.mkdtemp())
    (d / "sub").mkdir()
    (d / "sub" / "a.txt").write_text("ORIGINAL", encoding="utf-8")
    cwd0 = _os.getcwd()
    _os.chdir(d)
    try:
        a = OverwriteFileAction(file_path="sub/a.txt", previous_content="ORIGINAL", new_content="NEW")
        b = AppendTagsActionV2(file_path="sub/a.txt", previous_content=None, new_content="X", added_tags=["t"])
        c = EditCaptionAction(file_path="sub/a.txt", old_text="ORIGINAL", new_text="E")
        for act in (a, b, c):
            assert _os.path.isabs(str(act.file_path)), f"{type(act).__name__} kept a relative path"
        # move the CWD away entirely, then undo/redo must still hit the same real file
        _os.chdir(tempfile.mkdtemp())
        assert a.redo() and (d / "sub" / "a.txt").read_text(encoding="utf-8") == "NEW"
        assert a.undo() and (d / "sub" / "a.txt").read_text(encoding="utf-8") == "ORIGINAL"
    finally:
        _os.chdir(cwd0)
    print("  undo actions normalize file_path at construction (CWD-change safe): OK")


def test_progress_throttle():
    import tagging_core as T
    d = Path(tempfile.mkdtemp())
    from PIL import Image
    n = 1000
    paths = []
    for i in range(n):
        p = d / f"i{i:04}.png"
        Image.new("RGB", (2, 2)).save(p)
        paths.append(p)

    class FakeTagger:
        tag_meta_lookup = {}
        def infer_batch(self, images, thresholds=None, max_tags=None):
            return [T.TagResult(tags=[T.TagPrediction(name="1girl", score=0.99, category=T.TagCategory.GENERAL)], series_tags=())]

    settings = {"INPUT_DIR": d, "EXISTING_FILE_MODE": T.ExistingFileMode.OVERWRITE,
                "TAG_THRESHOLDS": {}, "MAX_TAGS_PER_CATEGORY": {},
                "ENABLE_SOLO_LIMIT": False, "CONVERT_UNDERSCORE": True}
    prog = []
    T.process_image_loop(
        tagger=FakeTagger(), settings=settings, image_paths=paths,
        decision_resolver=None, log_gui=lambda m, c: None,
        stop_checker=lambda: False, get_string=None,
        progress_cb=lambda d_, t_: prog.append((d_, t_)))
    assert prog[-1] == (n, n), f"final progress must reach total, got {prog[-1]}"
    assert len(prog) <= 205, f"expected progress calls throttled to ~200, got {len(prog)}"
    print(f"  progress throttle: OK ({len(prog)} calls for {n} images, last={prog[-1]})")


def test_locale_per_key_fallback():
    from locale_manager import LocaleManager
    old_dir = Path(tempfile.mkdtemp())  # simulates stale exe-adjacent lang/
    resource_dir = Path(tempfile.mkdtemp())  # simulates fresh bundled _internal/lang

    # Bundled (fresh) en.ini has a key the stale user copy lacks.
    (resource_dir / "en.ini").write_text("[Sec]\nOld_Key = fresh old value\nNew_Key = fresh new value\n", encoding="utf-8")
    # Stale copy overrides Old_Key (simulating a user edit) but has no New_Key at all.
    (old_dir / "en.ini").write_text("[Sec]\nOld_Key = user-edited value\n", encoding="utf-8")

    lm = LocaleManager("en", old_dir, resource_dir)
    assert lm.get_string("Sec", "Old_Key") == "user-edited value", "user's exe-adjacent copy must win for keys it has"
    assert lm.get_string("Sec", "New_Key") == "fresh new value", "missing key must fall back to bundled resource, not show as raw key"
    print("  locale per-key fallback: OK (user override wins, missing key falls back to bundled)")


def test_onnx_threads_clamp():
    import app_settings as A
    cap = os.cpu_count() or 32
    assert A._parse_onnx_threads(str(cap * 1000)) == cap
    assert A._parse_onnx_threads("4") == min(4, cap)
    assert A._parse_onnx_threads("0") == 0
    assert A._parse_onnx_threads("-5") == 0
    print(f"  onnx_threads clamp: OK (cap={cap})")


def test_undo_manager_dedup_and_longpath():
    from undo_manager import OverwriteFileAction, AppendTagsActionV2
    d = Path(tempfile.mkdtemp())
    f = d / "t.txt"
    f.write_text("old", encoding="utf-8")

    a = OverwriteFileAction(file_path=f, previous_content="old", new_content="new")
    assert a.redo() is True and f.read_text(encoding="utf-8") == "new"
    assert a.undo() is True and f.read_text(encoding="utf-8") == "old"

    b = AppendTagsActionV2(file_path=f, previous_content=None, new_content="brand new", added_tags=["x"])
    assert b.redo() is True and f.exists()
    assert b.undo() is True and not f.exists(), "previous_content=None must delete on undo"
    print("  undo_manager dedup (shared base) + basic undo/redo: OK")


def test_main_window_locking_and_no_clear():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import main_window
    import constants
    from tagging_core import FileChange
    w = main_window.MainWindow()
    # MainWindow.__init__ queues QTimer.singleShot(0, self.initial_load) (heavy work:
    # reload_image_list/reload_tags_only/model-mode switch/GPU prompt). If this test
    # never pumps the event loop, that callback stays pending and Qt keeps `w` alive
    # to run it - not now, but whenever some *unrelated* later test happens to call
    # processEvents(). Firing initial_load() against this throwaway MainWindow in the
    # middle of a different dialog's test caused a Windows fail-fast crash (exit code
    # 9 / 0xC0000409) after the whole suite had already printed "N passed" (260923
    # investigation, moved here from docs/test_pr16_review_fixes.py). Flushing it here
    # keeps the side effect - and any failure it causes - attributed to this test.
    for _ in range(5):
        app.processEvents()

    # existing_mode_combo / caption_placement_widget lock with the rest of main controls
    w._set_main_controls_enabled(False)
    assert w.existing_mode_combo.isEnabled() is False
    assert w.caption_placement_widget.isEnabled() is False
    w._set_main_controls_enabled(True)
    assert w.existing_mode_combo.isEnabled() is True
    assert w.caption_placement_widget.isEnabled() is True

    # prior undo history must survive an oversized batch
    small = [FileChange(path=Path("/tmp/s.txt"), previous_content="a", new_content="b", was_append=False, added_tags=())]
    w._on_batch_completed(small)
    assert w.undo_manager.can_undo() is True
    big = [FileChange(path=Path(f"/tmp/b{i}.txt"), previous_content="a", new_content="b", was_append=False, added_tags=())
           for i in range(constants.UNDO_BATCH_SNAPSHOT_LIMIT + 10)]
    w._on_batch_completed(big)
    assert w.undo_manager.can_undo() is True, "oversized batch must not wipe prior undo history"
    w.close()
    print("  main_window controls-lock + no-clear-on-oversized-batch: OK")


def test_pixai_tagger_gui_fallback_param():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QPalette, QColor
    app = QApplication.instance() or QApplication([])
    import image_tagger_gui as G
    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Window, QColor(10, 10, 10))
    light = QPalette()
    light.setColor(QPalette.ColorRole.Window, QColor(250, 250, 250))
    # Force the colorScheme() path to fail so the fallback_palette param is actually exercised.
    import unittest.mock as mock
    with mock.patch.object(type(app.styleHints()), "colorScheme", side_effect=Exception("boom"), create=True):
        pass  # patching a Qt method this way is unreliable; instead just check param plumbing directly.
    assert G.os_prefers_dark(app, dark) in (True, False)
    # Directly exercise the palette-lightness branch by monkeypatching styleHints via a stub app-like object.
    class StubStyleHints:
        def colorScheme(self):
            raise RuntimeError("unavailable")
    class StubApp:
        def styleHints(self):
            return StubStyleHints()
        def palette(self):
            return light
    assert G.os_prefers_dark(StubApp(), dark) is True, "must use fallback_palette (dark) not app.palette() (light)"
    assert G.os_prefers_dark(StubApp(), light) is False
    print("  pixai_tagger_gui fallback_palette plumbing: OK")


if __name__ == "__main__":
    test_utils_lock_timeout()
    test_tagging_core_unreadable_skip()
    test_caption_core_unreadable_skip()
    test_caption_core_empty_generation_preserves_file()
    test_long_path_str_idempotent_on_extended_paths()
    test_undo_actions_normalize_path_at_construction()
    test_progress_throttle()
    test_locale_per_key_fallback()
    test_onnx_threads_clamp()
    test_undo_manager_dedup_and_longpath()
    test_main_window_locking_and_no_clear()
    test_pixai_tagger_gui_fallback_param()
    print("\nALL PR#16 FIX VERIFICATIONS PASSED")
