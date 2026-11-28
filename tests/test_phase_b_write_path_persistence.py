"""Phase B（書き込み・パス・永続化の堅牢性）のテスト。

対象（docs/261127_plan-review-findings-fix.md §8）:
- REQ-B1（R30-1）: 既存 Phase 2 の対応。検証は既存 Phase 2 で実施済み（本ファイルでは
  再実装しない。UNC 実パスはこの環境で構築できないため §15.2 に従い「未検証」と記録）。
- REQ-B2（R30-2）: 読み取り経路の長パス変換
- REQ-B3（R11-10）: Undo/Redo の改行規約保持
- REQ-B4（R17-6）: .txt サイドカーの BOM
- REQ-B5（R28-3）: 同梱ファイルのシードの失敗隔離
- REQ-B6（R1-7）: config.ini の read-modify-write 競合
"""
from __future__ import annotations

import configparser
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import app_settings
import caption_core
import constants
import tagging_core
import text_io
import undo_manager
import utils
import vlm_config
import vlm_persistence
import workers
from tagging_core import (
    ExistingFileMode,
    TagCategory,
    TagPrediction,
    TagResult,
)


# ---------------------------------------------------------------------------
# 共通ハーネス
# ---------------------------------------------------------------------------


def _write_image(path: Path) -> Path:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)
    return path


def _deep_dir(tmp_path: Path) -> Path:
    """260 文字を超えるディレクトリを作る（環境が許す範囲で）。"""
    path = tmp_path
    for depth in range(4):
        path = path / (chr(ord("a") + depth) * 70)
    path.mkdir(parents=True, exist_ok=False)
    assert len(str(path)) > 260
    return path


class _FakeTagger:
    tag_meta_lookup: dict = {}

    def __init__(self, tags=("1girl",)):
        self._tags = list(tags)

    def infer_batch(self, images, thresholds=None, max_tags=None):
        return [TagResult(tags=[TagPrediction(name=t, score=0.9, category=TagCategory.GENERAL)
                                for t in self._tags])]


class _FakeCaptioner:
    config = SimpleNamespace(default_task="t", tasks={"t": "<t>"})

    def __init__(self, caption: str = "a caption"):
        self._caption = caption

    def generate(self, image, task_prompt, stop_checker):
        return self._caption, False


def _tagging_settings(input_dir: Path, mode: ExistingFileMode = ExistingFileMode.APPEND):
    return {
        "INPUT_DIR": input_dir,
        "EXISTING_FILE_MODE": mode,
        "TAG_THRESHOLDS": {},
        "MAX_TAGS_PER_CATEGORY": {},
        "ENABLE_SOLO_LIMIT": False,
        "CONVERT_UNDERSCORE": True,
    }


def _caption_settings(input_dir: Path, placement: str = "APPEND"):
    return {
        "INPUT_DIR": input_dir,
        "EXISTING_FILE_MODE": ExistingFileMode.APPEND,
        "CAPTION_PLACEMENT": placement,
        "TASK": "t",
    }


def _run_tagging(images, *, tagger=None, mode=ExistingFileMode.APPEND):
    failed: list[Path] = []
    changed = tagging_core.process_image_loop(
        tagger=tagger or _FakeTagger(),
        settings=_tagging_settings(Path(images[0]).parent, mode),
        image_paths=list(images),
        decision_resolver=None,
        log_gui=lambda m, c: None,
        stop_checker=lambda: False,
        get_string=None,
        progress_cb=lambda done, total: None,
        failed_paths=failed,
    )
    return changed, failed


def _run_caption(images, *, captioner=None, placement="APPEND"):
    failed: list[Path] = []
    changed = caption_core.process_caption_loop(
        captioner=captioner or _FakeCaptioner(),
        settings=_caption_settings(Path(images[0]).parent, placement),
        image_paths=list(images),
        decision_resolver=None,
        log_gui=lambda m, c: None,
        stop_checker=lambda: False,
        get_string=None,
        progress_cb=lambda done, total: None,
        failed_paths=failed,
    )
    return changed, failed


# ---------------------------------------------------------------------------
# REQ-B2: 読み取り経路の長パス変換
# ---------------------------------------------------------------------------


def test_tagging_loop_reads_and_writes_under_long_path(monkeypatch, tmp_path: Path):
    """260 文字超パスで読み取り→推論→書き込みが通しで成功する（REQ-B2）。"""
    deep = _deep_dir(tmp_path)
    image = _write_image(deep / "img.png")
    txt = image.with_suffix(".txt")
    txt.write_text("solo", encoding="utf-8")

    opened: list[object] = []
    real_long = tagging_core.long_path_str
    monkeypatch.setattr(tagging_core, "long_path_str",
                        lambda p: opened.append(p) or real_long(p))
    reads: list[Path] = []
    real_read = tagging_core.read_text
    monkeypatch.setattr(tagging_core, "read_text",
                        lambda p, **kw: reads.append(Path(p)) or real_read(p, **kw))

    changed, failed = _run_tagging([image], mode=ExistingFileMode.OVERWRITE)

    assert not failed
    assert len(changed) == 1
    recorded = [Path(p) for p in opened]
    assert image in recorded, "Image.open は長パス変換を通す"
    assert txt in recorded, "既存 .txt の存在判定も長パス変換を通す"
    assert txt in reads, "既存 .txt の読み取りも長パス変換を通す"
    assert txt.read_text(encoding="utf-8") == "1girl"


def test_caption_loop_reads_and_writes_under_long_path(monkeypatch, tmp_path: Path):
    deep = _deep_dir(tmp_path)
    image = _write_image(deep / "img.png")
    txt = image.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")

    opened: list[object] = []
    real_long = caption_core.long_path_str
    monkeypatch.setattr(caption_core, "long_path_str",
                        lambda p: opened.append(p) or real_long(p))
    reads: list[Path] = []
    real_read = caption_core.read_text
    monkeypatch.setattr(caption_core, "read_text",
                        lambda p, **kw: reads.append(Path(p)) or real_read(p, **kw))

    changed, failed = _run_caption([image])

    assert not failed
    assert len(changed) == 1
    recorded = [Path(p) for p in opened]
    assert image in recorded, "画像の open は長パス変換を通す"
    assert txt in recorded, "既存 .txt の存在判定も長パス変換を通す"
    assert txt in reads, "既存 .txt の読み取りも長パス変換を通す"
    assert txt.read_text(encoding="utf-8") == "old caption, a caption"


def test_bulk_tag_worker_reads_through_shared_reader(monkeypatch, tmp_path: Path):
    txt = tmp_path / "a.txt"
    txt.write_text("solo", encoding="utf-8")
    reads: list[Path] = []
    real_read = workers.read_text
    monkeypatch.setattr(workers, "read_text",
                        lambda p, **kw: reads.append(Path(p)) or real_read(p, **kw))

    worker = workers.BulkTagWorker()
    ok = worker._process_tag_file(txt, lambda tags: tags + ["new"])

    assert ok is True
    assert txt in reads
    assert txt.read_text(encoding="utf-8") == "solo, new"


# ---------------------------------------------------------------------------
# REQ-B3: Undo/Redo の改行規約
# ---------------------------------------------------------------------------


def test_vlm_save_undo_redo_preserves_lf_bytes(tmp_path: Path):
    txt = tmp_path / "cap.txt"
    txt.write_bytes(b"old line\nsecond line\n")

    outcome = vlm_persistence.save_caption(txt, "new caption", "APPEND")
    assert outcome.written
    saved_bytes = txt.read_bytes()
    assert b"\r\n" not in saved_bytes

    action = undo_manager.OverwriteFileAction(
        file_path=txt, previous_content=outcome.previous_content,
        new_content=outcome.new_content)
    assert action.newline == "\n"

    assert action.undo() is True
    assert txt.read_bytes() == b"old line\nsecond line\n"
    assert action.redo() is True
    assert txt.read_bytes() == saved_bytes


def test_snapshot_action_preserves_crlf_bytes(tmp_path: Path):
    txt = tmp_path / "cap.txt"
    txt.write_bytes(b"old line\r\nsecond line\r\n")

    outcome = vlm_persistence.save_caption(txt, "new caption", "APPEND")
    assert outcome.written
    # save_caption は既存の CRLF 規約を保持する（§4-1）
    saved_bytes = txt.read_bytes()
    assert b"\r\n" in saved_bytes

    action = undo_manager.OverwriteFileAction(
        file_path=txt, previous_content=outcome.previous_content,
        new_content=outcome.new_content)
    assert action.newline == "\r\n"

    assert action.undo() is True
    assert txt.read_bytes() == b"old line\r\nsecond line\r\n"
    assert action.redo() is True
    assert txt.read_bytes() == saved_bytes


def test_snapshot_action_without_file_falls_back_to_platform_newline(tmp_path: Path):
    txt = tmp_path / "missing.txt"
    action = undo_manager.OverwriteFileAction(
        file_path=txt, previous_content="old", new_content="new")
    assert action.newline is None
    assert action.redo() is True
    assert txt.read_text(encoding="utf-8") == "new"


def test_tagging_undo_redo_round_trip_bytes(tmp_path: Path):
    image = _write_image(tmp_path / "img.png")
    txt = image.with_suffix(".txt")
    txt.write_text("solo", encoding="utf-8")

    changed, failed = _run_tagging([image])
    assert not failed and len(changed) == 1
    change = changed[0]
    written = txt.read_bytes()

    action = undo_manager.AppendTagsActionV2(
        file_path=change.path, previous_content=change.previous_content,
        new_content=change.new_content, added_tags=list(change.added_tags))
    assert action.undo() is True
    assert txt.read_bytes() == b"solo"
    assert action.redo() is True
    assert txt.read_bytes() == written


# ---------------------------------------------------------------------------
# REQ-B4: .txt サイドカーの BOM
# ---------------------------------------------------------------------------


def test_read_tags_strips_bom(tmp_path: Path):
    import tag_utils

    txt = tmp_path / "a.txt"
    txt.write_bytes("\ufeffsolo, 1girl".encode("utf-8"))
    assert tag_utils.read_tags(txt) == ["solo", "1girl"]

    plain = tmp_path / "b.txt"
    plain.write_text("solo, 1girl", encoding="utf-8")
    assert tag_utils.read_tags(plain) == ["solo", "1girl"]


def test_tagging_append_does_not_propagate_bom(tmp_path: Path):
    image = _write_image(tmp_path / "img.png")
    txt = image.with_suffix(".txt")
    txt.write_bytes("\ufeffsolo".encode("utf-8"))

    changed, failed = _run_tagging([image])

    assert not failed and len(changed) == 1
    assert "\ufeff" not in txt.read_text(encoding="utf-8")
    assert txt.read_text(encoding="utf-8") == "solo, 1girl"
    assert changed[0].previous_content == "solo"


def test_caption_loop_strips_bom(tmp_path: Path):
    image = _write_image(tmp_path / "img.png")
    txt = image.with_suffix(".txt")
    txt.write_bytes("\ufeffold caption".encode("utf-8"))

    changed, failed = _run_caption([image])

    assert not failed and len(changed) == 1
    assert "\ufeff" not in txt.read_text(encoding="utf-8")
    assert changed[0].previous_content == "old caption"


def test_vlm_save_strips_bom_and_detects_duplicate(tmp_path: Path):
    txt = tmp_path / "cap.txt"
    txt.write_bytes("\ufeffcap".encode("utf-8"))

    outcome = vlm_persistence.save_caption(txt, "cap", "APPEND")
    assert outcome.written is False
    assert outcome.skipped_reason == "duplicate"

    outcome2 = vlm_persistence.save_caption(txt, "new caption", "APPEND")
    assert outcome2.written is True
    written = txt.read_text(encoding="utf-8")
    assert "\ufeff" not in written
    assert written == "cap\nnew caption"


def test_bulk_tag_worker_strips_bom(tmp_path: Path):
    txt = tmp_path / "a.txt"
    txt.write_bytes("\ufeffsolo, old".encode("utf-8"))
    worker = workers.BulkTagWorker()

    assert worker._process_tag_file(txt, lambda tags: tags + ["new"]) is True
    assert txt.read_text(encoding="utf-8") == "solo, old, new"


# ---------------------------------------------------------------------------
# REQ-B5: 同梱ファイルシードの失敗隔離
# ---------------------------------------------------------------------------


def test_seed_bundled_dir_isolates_copy_failure(monkeypatch, tmp_path: Path):
    resource = tmp_path / "resource"
    (resource / "nested").mkdir(parents=True)
    (resource / "a.txt").write_text("a", encoding="utf-8")
    (resource / "nested" / "b.txt").write_text("b", encoding="utf-8")
    (resource / "nested" / "c.txt").write_text("c", encoding="utf-8")
    (resource / "z.txt").write_text("z", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()

    real_copy = constants.shutil.copy2

    def flaky_copy(src, dst, *args, **kwargs):
        if Path(src).name == "b.txt":
            raise OSError("simulated copy failure")
        return real_copy(src, dst, *args, **kwargs)

    logs: list[str] = []
    monkeypatch.setattr(constants.shutil, "copy2", flaky_copy)
    monkeypatch.setattr(utils, "write_debug_log",
                        lambda message, *a, **kw: logs.append(str(message)))

    constants._seed_bundled_dir(resource, dest)

    assert (dest / "a.txt").read_text(encoding="utf-8") == "a"
    assert (dest / "z.txt").read_text(encoding="utf-8") == "z"
    assert (dest / "nested" / "c.txt").read_text(encoding="utf-8") == "c"
    assert not (dest / "nested" / "b.txt").exists()
    assert any("b.txt" in message and "simulated copy failure" in message for message in logs)
    # .part の残骸を残さない（原子コピー方式は変えない）
    assert not list(dest.rglob("*.part"))


def test_seed_lang_dir_continues_after_models_failure(monkeypatch, tmp_path: Path):
    models_resource = tmp_path / "models_resource"
    models_resource.mkdir()
    (models_resource / "m.txt").write_text("m", encoding="utf-8")
    models_dest = tmp_path / "models"
    models_dest.mkdir()
    lang_resource = tmp_path / "lang_resource"
    lang_resource.mkdir()
    (lang_resource / "ja.ini").write_text("x", encoding="utf-8")
    lang_dest = tmp_path / "lang"
    lang_dest.mkdir()

    real_copy = constants.shutil.copy2

    def flaky_copy(src, dst, *args, **kwargs):
        if Path(src).name == "m.txt":
            raise OSError("read-only install directory")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(constants.shutil, "copy2", flaky_copy)
    monkeypatch.setattr(utils, "write_debug_log", lambda *a, **kw: None)

    # モジュール初期化時と同じ順で呼ぶ。失敗しても2回目（lang）が走ること。
    constants._seed_bundled_dir(models_resource, models_dest)
    constants._seed_bundled_dir(lang_resource, lang_dest)

    assert not (models_dest / "m.txt").exists()
    assert (lang_dest / "ja.ini").read_text(encoding="utf-8") == "x"


# ---------------------------------------------------------------------------
# REQ-B6: config.ini の read-modify-write 競合
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", path)
    return path


def _write_config(path: Path, *, general: str = "0.10", verified: str = "other:False") -> None:
    path.write_text(
        "[Thresholds]\n"
        f"general = {general}\n"
        "\n"
        "[Model]\n"
        "model_id = pixai-tagger-v0.9\n"
        f"verified_models = {verified}\n"
        "\n"
        "[Vlm]\n"
        "verified_bindings = \n",
        encoding="utf-8",
    )


def test_gui_save_does_not_roll_back_worker_verification(config_path: Path):
    """stale-write の再現（REQ-B6(b)）。GUI が古い settings を持っていても巻き戻さない。"""
    _write_config(config_path)
    gui_settings = app_settings.load_settings(app_settings.load_config())

    app_settings.update_model_verification_status(
        "pixai-tagger-v0.9", True, lambda section, key, **kwargs: key)

    # GUI はワーカーの更新を知らないまま、無関係な設定を変えて全体を保存する。
    gui_settings.thresholds.general = 0.42
    assert app_settings.save_config(gui_settings) is True

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.thresholds.general == 0.42
    assert reloaded.model.verified_models == {
        "other": False, "pixai-tagger-v0.9": True}


def test_worker_verification_update_is_not_clobbered(config_path: Path):
    """(b) のディスク優先で、ワーカー自身の検証更新が潰れないこと。"""
    _write_config(config_path, verified="other:False")
    app_settings.update_model_verification_status(
        "pixai-tagger-v0.9", True, lambda section, key, **kwargs: key)

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.model.verified_models["pixai-tagger-v0.9"] is True


def test_gui_binding_verification_is_not_clobbered(config_path: Path):
    """GUI が意図的に追加した verified_bindings はディスク優先で失われない。"""
    _write_config(config_path)
    settings = app_settings.load_settings(app_settings.load_config())
    token = f"{settings.vlm.model_profile_id}:gemini"
    assert vlm_config.mark_binding_verified(settings.vlm, "gemini") is True

    assert app_settings.save_config(settings) is True
    reloaded = app_settings.load_settings(app_settings.load_config())
    assert token in reloaded.vlm.verified_set()


def test_gui_binding_removal_is_not_resurrected(config_path: Path):
    """GUI が意図的に落とした binding を、ディスク優先が復活させないこと。"""
    _write_config(config_path)
    settings = app_settings.load_settings(app_settings.load_config())
    token = f"{settings.vlm.model_profile_id}:gemini"
    vlm_config.mark_binding_verified(settings.vlm, "gemini")
    assert app_settings.save_config(settings) is True

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert token in reloaded.vlm.verified_set()
    assert vlm_config.clear_binding_verified(reloaded.vlm, "gemini") is True
    assert app_settings.save_config(reloaded) is True

    final = app_settings.load_settings(app_settings.load_config())
    assert token not in final.vlm.verified_set()


def test_save_is_serialized_with_worker_read_modify_write(config_path: Path, monkeypatch):
    """(a) lost update: ワーカーの read-modify-write 中に GUI 保存が割り込まない。"""
    _write_config(config_path)
    gui_settings = app_settings.load_settings(app_settings.load_config())
    gui_settings.thresholds.general = 0.25

    real_load = app_settings.load_config
    worker_read = threading.Event()
    release_worker = threading.Event()

    def slow_load():
        config = real_load()
        worker_read.set()
        assert release_worker.wait(5), "test did not release the worker"
        return config

    monkeypatch.setattr(app_settings, "load_config", slow_load)

    def worker():
        app_settings.update_model_verification_status(
            "pixai-tagger-v0.9", True, lambda section, key, **kwargs: key)

    worker_thread = threading.Thread(target=worker)
    worker_thread.start()
    assert worker_read.wait(5)

    save_done = threading.Event()

    def gui_save():
        app_settings.save_config(gui_settings)
        save_done.set()

    gui_thread = threading.Thread(target=gui_save)
    gui_thread.start()
    # ワーカーがロックを保持している間、GUI の保存は待たされる。
    assert not save_done.wait(0.2), "save_config must wait for the worker's critical section"

    release_worker.set()
    worker_thread.join(5)
    gui_thread.join(5)
    assert not worker_thread.is_alive()
    assert not gui_thread.is_alive()

    reloaded = app_settings.load_settings(app_settings.load_config())
    assert reloaded.thresholds.general == 0.25
    assert reloaded.model.verified_models["pixai-tagger-v0.9"] is True


def test_single_save_result_is_unchanged(config_path: Path):
    """単独実行の保存結果は従来と一致（B6 の変更で観測可能な差分を出さない）。"""
    _write_config(config_path, verified="pixai-tagger-v0.9:True")
    settings = app_settings.load_settings(app_settings.load_config())
    assert app_settings.save_config(settings) is True

    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read(config_path, encoding="utf-8")
    assert parsed.get("Model", "verified_models") == "pixai-tagger-v0.9:True"
    assert parsed.getfloat("Thresholds", "general") == 0.10
    assert parsed.get("Vlm", "verified_bindings") == ""


def test_read_text_helper_is_identity_for_short_paths(tmp_path: Path):
    """REQ-B2 の共通部品は短いパスで従来どおり読める。"""
    path = tmp_path / "a.txt"
    path.write_text("x, y", encoding="utf-8")
    assert text_io.read_text(path) == "x, y"
    assert text_io.read_text(path, encoding="utf-8-sig") == "x, y"


def test_edit_caption_action_keeps_platform_newline(monkeypatch, tmp_path: Path):
    """REQ-B3 の変更は _FileSnapshotAction に限る（EditCaptionAction は現行どおり）。"""
    recorded: list[dict] = []
    real_write = undo_manager.atomic_write_text

    def spy(path, text, **kwargs):
        recorded.append(kwargs)
        return real_write(path, text, **kwargs)

    monkeypatch.setattr(undo_manager, "atomic_write_text", spy)
    action = undo_manager.EditCaptionAction(
        file_path=tmp_path / "x.txt", old_text="old", new_text="new")
    assert action.redo() is True
    assert recorded and "newline" not in recorded[0]
    assert recorded[0].get("use_long_path") is False
