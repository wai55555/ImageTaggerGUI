"""Phase 2（書き込みの原子性統一）のテスト。

対象: `src/text_io.py` の共通部品と、それへ移行した各書き込み経路。

固定すること（仕様書 §2.3）:
- 書き込み途中の失敗では対象ファイルが変更前のまま残り、一時ファイルも残らない
- 一時ファイルは .txt ではない・衝突しない名前で、対象と同じディレクトリに作られる
- 改行は経路ごとの現行規約のまま（非 VLM は `newline=None`、VLM だけ検出値）
- config.ini はフル装備（fsync + 検証）、成果物 .txt は (iii) 段（replace のみ）
- symlink / hardlink はリンクを辿って実体へ書く（U-7 の決定）
- `tag_utils` は失敗を呼び出し側へ伝播し、debug log に残す（REQ-2.10）
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import text_io
import undo_manager
import utils
import vlm_persistence
from text_io import atomic_write_text, is_link_target, long_path_str


# ---------------------------------------------------------------------------
# text_io: 基本動作
# ---------------------------------------------------------------------------


def test_atomic_write_round_trip(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "a, b, c")
    assert target.read_text(encoding="utf-8") == "a, b, c"


def test_atomic_write_leaves_no_temporary_files(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "x")
    atomic_write_text(target, "y")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]


def test_atomic_write_overwrites_previous_content(tmp_path: Path):
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_tmp_file_name_is_not_txt_and_unique(tmp_path: Path):
    target = tmp_path / "out.txt"
    seen: list[str] = []
    real_replace = os.replace

    def spy_replace(src, dst, *args, **kwargs):
        seen.append(str(src))
        return real_replace(src, dst, *args, **kwargs)

    original = text_io.os.replace
    text_io.os.replace = spy_replace
    try:
        atomic_write_text(target, "one")
        atomic_write_text(target, "two")
    finally:
        text_io.os.replace = original

    assert len(seen) == 2
    assert seen[0] != seen[1]
    for name in seen:
        assert Path(name).name.endswith(text_io.DEFAULT_TMP_SUFFIX)
        assert not Path(name).name.endswith(".txt")
        # 対象と同じディレクトリ（同一ボリュームでの os.replace を成立させる）
        assert Path(name).parent == Path(long_path_str(tmp_path))


def test_replace_failure_keeps_old_content_and_cleans_tmp(tmp_path: Path):
    target = tmp_path / "out.txt"
    target.write_text("PRECIOUS", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated replace failure")

    original = text_io.os.replace
    text_io.os.replace = boom
    try:
        with pytest.raises(OSError):
            atomic_write_text(target, "new content")
    finally:
        text_io.os.replace = original

    assert target.read_text(encoding="utf-8") == "PRECIOUS"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]


def test_write_failure_keeps_old_content(tmp_path: Path, monkeypatch):
    target = tmp_path / "out.txt"
    target.write_text("PRECIOUS", encoding="utf-8")

    import builtins

    real_open = builtins.open

    def failing_open(file, mode="r", *args, **kwargs):
        if "w" in mode:
            raise OSError("simulated write failure")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(OSError):
        atomic_write_text(target, "new content")
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == "PRECIOUS"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]


def test_verify_detects_mismatch_and_keeps_target(tmp_path: Path):
    """検証は「書いた内容の読み直し」で行う。CRLF を含む content を newline=\"\\n\" で
    書くと検証が一致せず OSError になる（＝検証が実際に効いていることの確認）。"""
    target = tmp_path / "out.txt"
    target.write_text("PRECIOUS", encoding="utf-8")
    with pytest.raises(OSError):
        atomic_write_text(target, "a\r\nb", newline="\n")
    assert target.read_text(encoding="utf-8") == "PRECIOUS"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]


def test_verify_can_be_disabled(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "a\r\nb", newline="\n", verify=False)
    with open(target, encoding="utf-8", newline="") as handle:
        assert handle.read() == "a\r\nb"


def test_makedirs_is_opt_in(tmp_path: Path):
    target = tmp_path / "sub" / "out.txt"
    with pytest.raises(FileNotFoundError):
        atomic_write_text(target, "x")
    assert not target.parent.exists()

    atomic_write_text(target, "x", makedirs=True)
    assert target.read_text(encoding="utf-8") == "x"


def test_newline_none_follows_platform_open_default(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "a\nb")
    expected = b"a\r\nb" if sys.platform == "win32" else b"a\nb"
    assert target.read_bytes() == expected


def test_explicit_newline_is_not_translated(tmp_path: Path):
    target = tmp_path / "out.txt"
    atomic_write_text(target, "a\nb", newline="\n")
    assert target.read_bytes() == b"a\nb"


def test_symlink_is_followed_and_kept_as_link(tmp_path: Path):
    real = tmp_path / "real.txt"
    real.write_text("old", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this platform/user")

    assert is_link_target(link)
    atomic_write_text(link, "new")
    # リンクはリンクのまま、実体が更新される（os.replace 方式だとリンクが切れる）
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "new"


def test_hardlink_is_written_in_place(tmp_path: Path):
    first = tmp_path / "a.txt"
    first.write_text("old", encoding="utf-8")
    second = tmp_path / "b.txt"
    try:
        os.link(first, second)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("hardlink creation not permitted on this platform")

    assert is_link_target(first)
    atomic_write_text(first, "new")
    assert first.read_text(encoding="utf-8") == "new"
    assert second.read_text(encoding="utf-8") == "new"  # 同じ実体（inode）
    assert os.lstat(first).st_nlink == 2


def test_plain_file_is_not_treated_as_link(tmp_path: Path):
    target = tmp_path / "out.txt"
    target.write_text("x", encoding="utf-8")
    assert not is_link_target(target)
    assert not is_link_target(tmp_path / "missing.txt")


def test_long_path_str_keeps_pr16_semantics():
    """REQ-2.8/2.9: 旧 undo_manager._long_path_str と同じ変換（win32 分岐を再現）。"""
    real_platform, real_abspath = text_io.sys.platform, text_io.os.path.abspath
    text_io.sys.platform = "win32"
    text_io.os.path.abspath = lambda x: str(x)
    try:
        assert long_path_str("\\\\?\\C:\\a\\b") == "\\\\?\\C:\\a\\b"
        assert long_path_str("\\\\?\\UNC\\srv\\share\\f") == "\\\\?\\UNC\\srv\\share\\f"
        once = long_path_str("C:\\a\\b")
        assert long_path_str(once) == once == "\\\\?\\C:\\a\\b"
        unc = long_path_str("\\\\srv\\share\\f")
        assert long_path_str(unc) == unc == "\\\\?\\UNC\\srv\\share\\f"
    finally:
        text_io.sys.platform, text_io.os.path.abspath = real_platform, real_abspath


def test_long_path_str_is_identity_off_windows(tmp_path: Path):
    if sys.platform == "win32":
        pytest.skip("non-Windows behaviour")
    assert long_path_str(tmp_path / "a.txt") == os.path.abspath(tmp_path / "a.txt")


def test_undo_manager_long_path_is_thin_delegation():
    """REQ-2.9: 名前は既存テストが参照しているので残し、実装は共通部品へ委譲する。"""
    assert undo_manager._long_path_str(Path("a/b.txt")) == long_path_str(Path("a/b.txt"))


# ---------------------------------------------------------------------------
# 改行引数の固定（REQ-2.5 / §2.3）
# ---------------------------------------------------------------------------


_ATOMIC_WRITE_MODULES = [
    "tagging_core",
    "caption_core",
    "workers",
    "undo_manager",
    "main_window",
    "grid_view_widget",
    "tag_utils",
    "app_settings",
]


@pytest.mark.parametrize("module_name", _ATOMIC_WRITE_MODULES)
def test_non_vlm_sites_do_not_override_newline(module_name):
    """非 VLM 経路は `newline` を渡さない（＝open() 既定＝現行の直接書きと同じ翻訳）。

    例外は `undo_manager._FileSnapshotAction._write` だけ（REQ-B3）。Undo/Redo の
    全文スナップショットは、アクション生成時に検出した改行規約で書き戻さないと
    Windows で LF のファイルが CRLF へ化ける。他の undo/redo 書き戻し（8 箇所）は
    従来どおり `newline` を渡さない。
    """
    source_path = Path(text_io.__file__).with_name(f"{module_name}.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    allowed_newline_calls: set[int] = set()
    if module_name == "undo_manager":
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "_FileSnapshotAction":
                for member in node.body:
                    if isinstance(member, ast.FunctionDef) and member.name == "_write":
                        for inner in ast.walk(member):
                            if (isinstance(inner, ast.Call)
                                    and getattr(inner.func, "id", None) == "atomic_write_text"):
                                allowed_newline_calls.add(id(inner))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "atomic_write_text"
    ]
    assert calls, f"{module_name}: atomic_write_text 呼び出しが見つからない"
    for call in calls:
        if not [kw for kw in call.keywords if kw.arg == "newline"]:
            continue
        assert id(call) in allowed_newline_calls, (
            f"{module_name}:{call.lineno}: 非 VLM 経路で newline を上書きしてはならない"
        )


def test_vlm_site_passes_detected_newline(monkeypatch, tmp_path: Path):
    """VLM 経路だけが「既存ファイルから検出した改行規約」を明示する（従来挙動）。"""
    recorded: list[str | None] = []

    def recorder(path, content, **kwargs):
        recorded.append(kwargs.get("newline"))

    monkeypatch.setattr(vlm_persistence, "atomic_write_text", recorder)

    fresh = tmp_path / "fresh.txt"
    vlm_persistence.save_caption(fresh, "caption", "OVERWRITE")

    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes("existing\r\ncaption".encode("utf-8"))
    vlm_persistence.save_caption(crlf, "more", "APPEND")

    assert recorded == ["\n", "\r\n"]


def test_vlm_calls_shared_writer_with_full_durability(tmp_path: Path):
    """vlm_persistence は従来どおり (i) fsync + 検証 + makedirs で呼ぶ。"""
    calls: list[dict] = []
    import vlm_persistence as V

    def recorder(path, content, **kwargs):
        calls.append(kwargs)
        atomic_write_text(path, content, **kwargs)

    original = V.atomic_write_text
    V.atomic_write_text = recorder
    try:
        V.save_caption(tmp_path / "new.txt", "caption", "OVERWRITE")
    finally:
        V.atomic_write_text = original

    assert calls and calls[0]["fsync"] is True
    assert calls[0]["verify"] is True
    assert calls[0]["makedirs"] is True


def test_vlm_save_creates_missing_parent_directory(tmp_path: Path):
    target = tmp_path / "a" / "b" / "out.txt"
    vlm_persistence.save_caption(target, "caption", "OVERWRITE")
    assert target.read_text(encoding="utf-8") == "caption"


# ---------------------------------------------------------------------------
# 経路ごとの失敗時の意味論（REQ-2.4）
# ---------------------------------------------------------------------------


def _make_image(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (4, 4)).save(path)
    return path


class _FakeTagger:
    tag_meta_lookup: dict = {}

    def infer_batch(self, images, thresholds=None, max_tags=None):
        import tagging_core as T

        return [T.TagResult(tags=[T.TagPrediction(name="1girl", score=0.99,
                                                  category=T.TagCategory.GENERAL)])]


def _tagging_settings(input_dir: Path):
    import tagging_core as T

    return {
        "INPUT_DIR": input_dir,
        "EXISTING_FILE_MODE": T.ExistingFileMode.OVERWRITE,
        "TAG_THRESHOLDS": {},
        "MAX_TAGS_PER_CATEGORY": {},
        "ENABLE_SOLO_LIMIT": False,
        "CONVERT_UNDERSCORE": True,
    }


def test_tagging_loop_write_failure_keeps_file_and_reports_failure(monkeypatch, tmp_path: Path):
    import tagging_core as T

    img = _make_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old, tags", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(T, "atomic_write_text", boom)
    failed: list[Path] = []
    changed = T.process_image_loop(
        tagger=_FakeTagger(), settings=_tagging_settings(tmp_path), image_paths=[img],
        decision_resolver=None, log_gui=lambda m, c: None, stop_checker=lambda: False,
        get_string=None, failed_paths=failed,
    )
    assert changed == []  # 失敗時は FileChange を積まない
    assert failed == [img]
    assert txt.read_text(encoding="utf-8") == "old, tags"
    assert [p.name for p in tmp_path.iterdir() if p.suffix != ".png"] == ["a.txt"]


def test_caption_loop_write_failure_keeps_file_and_reports_failure(monkeypatch, tmp_path: Path):
    import caption_core as C

    img = _make_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")

    class FakeCaptioner:
        config = SimpleNamespace(default_task="t", tasks={"t": "<t>"})

        def generate(self, image, task_prompt, stop_checker):
            return ("new caption", False)

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(C, "atomic_write_text", boom)
    failed: list[Path] = []
    changed = C.process_caption_loop(
        captioner=FakeCaptioner(),
        settings={"INPUT_DIR": tmp_path, "EXISTING_FILE_MODE": C.ExistingFileMode.OVERWRITE,
                  "CAPTION_PLACEMENT": "OVERWRITE", "TASK": "t"},
        image_paths=[img], decision_resolver=None, log_gui=lambda m, c: None,
        stop_checker=lambda: False, get_string=None, failed_paths=failed,
    )
    assert changed == []
    assert failed == [img]
    assert txt.read_text(encoding="utf-8") == "old caption"


def test_bulk_tag_worker_write_failure_keeps_file(monkeypatch, tmp_path: Path):
    import workers

    txt = tmp_path / "a.txt"
    txt.write_text("x, y", encoding="utf-8")

    worker = workers.BulkTagWorker()

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(workers, "atomic_write_text", boom)
    ok = worker._process_tag_file(txt, lambda tags: tags + ["z"])
    assert ok is False
    assert txt.read_text(encoding="utf-8") == "x, y"


def test_bulk_tag_worker_writes_atomically(monkeypatch, tmp_path: Path):
    import workers

    txt = tmp_path / "a.txt"
    txt.write_text("x, y", encoding="utf-8")
    seen: list[dict] = []

    def recorder(path, content, **kwargs):
        seen.append(kwargs)
        atomic_write_text(path, content, **kwargs)

    monkeypatch.setattr(workers, "atomic_write_text", recorder)
    assert workers.BulkTagWorker()._process_tag_file(txt, lambda tags: tags + ["z"]) is True
    assert txt.read_text(encoding="utf-8") == "x, y, z"
    assert seen[0]["fsync"] is False and seen[0]["verify"] is False  # (iii) 段


def test_undo_tag_action_write_failure_returns_false(monkeypatch, tmp_path: Path):
    txt = tmp_path / "a.txt"
    txt.write_text("a, b", encoding="utf-8")
    action = undo_manager.AddTagsAction(file_path=txt, added_tags=["b"])

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(undo_manager, "atomic_write_text", boom)
    assert action.undo() is False
    assert txt.read_text(encoding="utf-8") == "a, b"


def test_snapshot_undo_removes_file_created_by_the_action(tmp_path: Path):
    txt = tmp_path / "a.txt"
    txt.write_text("new content", encoding="utf-8")
    action = undo_manager.OverwriteFileAction(file_path=txt, previous_content=None,
                                             new_content="new content")
    assert action.undo() is True
    assert not txt.exists()
    assert action.redo() is True
    assert txt.read_text(encoding="utf-8") == "new content"


def test_edit_caption_action_does_not_use_long_path_prefix(monkeypatch, tmp_path: Path):
    """REQ-2.9 / U-3: EditCaptionAction の経路は従来どおり変換を通さない。"""
    txt = tmp_path / "a.txt"
    txt.write_text("old", encoding="utf-8")
    seen: list[dict] = []
    real = undo_manager.atomic_write_text

    def recorder(path, content, **kwargs):
        seen.append(kwargs)
        real(path, content, **kwargs)

    monkeypatch.setattr(undo_manager, "atomic_write_text", recorder)
    action = undo_manager.EditCaptionAction(file_path=txt, old_text="old", new_text="new")
    assert action.redo() is True
    assert seen[0]["use_long_path"] is False
    assert txt.read_text(encoding="utf-8") == "new"


def test_save_config_failure_keeps_existing_file(monkeypatch, tmp_path: Path):
    import app_settings

    config_path = tmp_path / "config.ini"
    config_path.write_text("[General]\nlanguage_code = en\n", encoding="utf-8")
    monkeypatch.setattr(app_settings, "CONFIG_PATH", config_path)

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(app_settings, "atomic_write_text", boom)
    settings = app_settings.load_settings(app_settings.load_config())
    assert app_settings.save_config(settings) is False
    assert config_path.read_text(encoding="utf-8") == "[General]\nlanguage_code = en\n"


def test_save_config_writes_atomically_with_full_durability(monkeypatch, tmp_path: Path):
    import app_settings

    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", config_path)
    seen: list[dict] = []
    real = app_settings.atomic_write_text

    def recorder(path, content, **kwargs):
        seen.append(kwargs)
        real(path, content, **kwargs)

    monkeypatch.setattr(app_settings, "atomic_write_text", recorder)
    assert app_settings.save_config(app_settings.load_settings(app_settings.load_config())) is True
    assert config_path.is_file()
    assert seen[0]["fsync"] is True and seen[0]["verify"] is True


def test_load_config_creates_default_atomically(monkeypatch, tmp_path: Path):
    import app_settings

    config_path = tmp_path / "config.ini"
    monkeypatch.setattr(app_settings, "CONFIG_PATH", config_path)
    app_settings.load_config()
    assert config_path.is_file()
    assert "[General]" in config_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# tag_utils の失敗伝播とログ（REQ-2.10）
# ---------------------------------------------------------------------------


@pytest.fixture()
def debug_log_on(tmp_path, monkeypatch):
    """debug log を有効化して、ファイルへ実際に記録される状態にする。"""
    log_path = tmp_path / "debug_log.txt"
    debug_settings = utils.get_debug_settings()
    original_enabled = debug_settings.debug_log_enabled
    debug_settings.debug_log_enabled = True
    monkeypatch.setattr(utils, "LOG_FILE_PATH", log_path)
    # ログはプロセス内で1ハンドルを使い回すため、テストごとに差し替える。
    monkeypatch.setattr(utils, "_LOG_FH", None)
    monkeypatch.setattr(utils, "_LOG_LAST_FLUSH", 0.0)
    try:
        yield log_path
    finally:
        utils._flush_debug_log()
        handle = utils._LOG_FH
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        utils._LOG_FH = None
        debug_settings.debug_log_enabled = original_enabled


def test_write_tags_returns_false_and_logs(monkeypatch, tmp_path: Path, debug_log_on: Path):
    import tag_utils

    txt = tmp_path / "a.txt"
    txt.write_text("old", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(tag_utils, "atomic_write_text", boom)
    assert tag_utils.write_tags(txt, ["a", "b"]) is False
    assert txt.read_text(encoding="utf-8") == "old"
    utils._flush_debug_log()
    assert "tag_utils.write_tags failed" in debug_log_on.read_text(encoding="utf-8", errors="replace")


def test_add_and_remove_propagate_write_failure(monkeypatch, tmp_path: Path):
    import tag_utils

    txt = tmp_path / "a.txt"
    txt.write_text("a, b", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(tag_utils, "atomic_write_text", boom)
    assert tag_utils.add_tags_to_file(txt, ["c"]) is False
    assert tag_utils.remove_tag_from_file(txt, "a") is False
    assert txt.read_text(encoding="utf-8") == "a, b"


def test_add_tags_returns_false_when_nothing_new(tmp_path: Path):
    import tag_utils

    txt = tmp_path / "a.txt"
    txt.write_text("a, b", encoding="utf-8")
    assert tag_utils.add_tags_to_file(txt, ["a"]) is False
    assert tag_utils.remove_tag_from_file(txt, "zzz") is False


def test_add_tags_writes_atomically(tmp_path: Path):
    import tag_utils

    txt = tmp_path / "a.txt"
    assert tag_utils.add_tags_to_file(txt, ["x", "y"]) is True
    assert txt.read_text(encoding="utf-8") == "x, y"
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def test_read_tags_failure_returns_empty_without_print(monkeypatch, tmp_path: Path, debug_log_on: Path):
    """windowed ビルド相当（print が例外）でもフォールバックの return [] に到達する。"""
    import builtins
    import tag_utils

    txt = tmp_path / "bad.txt"
    txt.write_bytes(b"\xff\xfe\x00bad")

    def exploding_print(*_args, **_kwargs):
        raise OSError("stdout is not available")

    monkeypatch.setattr(builtins, "print", exploding_print)
    assert tag_utils.read_tags(txt) == []
    utils._flush_debug_log()
    log = debug_log_on.read_text(encoding="utf-8", errors="replace")
    assert "tag_utils.read_tags failed" in log


def test_grid_view_guards_on_write_result():
    """REQ-2.10(b): グリッドは書き込み失敗時に成功前提の後処理（emit）をしない。"""
    source = Path(text_io.__file__).with_name("grid_view_widget.py").read_text(encoding="utf-8")
    assert "if tag_utils.add_tags_to_file(" in source
    assert "if tag_utils.remove_tag_from_file(" in source


# ---------------------------------------------------------------------------
# 一時ファイルの残骸が走査に混入しないこと（REQ-2.6）
# ---------------------------------------------------------------------------


def test_tmp_files_are_not_collected_as_images(tmp_path: Path):
    import tagging_core as T

    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.deadbeef.vlmtmp").write_text("partial", encoding="utf-8")
    found = T.get_image_paths_recursive(tmp_path)
    assert [p.name for p in found] == ["a.png"]
    assert not any(p.suffix == ".txt" for p in found)


def test_link_target_uses_direct_write_without_tmp(monkeypatch, tmp_path: Path):
    """U-7: リンク対象は os.replace を使わず、一時ファイルも作らずに直接書く。

    Windows の非管理者環境では symlink を作れないため、判定だけを差し替えて
    フォールバック経路そのものを固定する（実リンクの検証は上のテスト）。
    """
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")
    monkeypatch.setattr(text_io, "is_link_target", lambda _p: True)

    replaced: list[str] = []
    real_replace = text_io.os.replace
    text_io.os.replace = lambda *a, **k: replaced.append(str(a[0]))
    try:
        atomic_write_text(target, "new")
    finally:
        text_io.os.replace = real_replace

    assert replaced == []  # replace しない
    assert target.read_text(encoding="utf-8") == "new"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.txt"]  # 一時ファイル無し


# ---------------------------------------------------------------------------
# 実機相当スモーク: GUI 側の保存経路も共有部品を通ること（§2.3）
# ---------------------------------------------------------------------------


def _qt_app():
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_grid_caption_save_uses_atomic_writer(tmp_path: Path, monkeypatch):
    app = _qt_app()
    from locale_manager import LocaleManager

    import caption_core
    import grid_view_widget

    img = _make_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")

    locale = LocaleManager("en", Path(caption_core.__file__).parent.parent / "lang")
    cell = grid_view_widget.ImageEditCellWidget(locale)
    cell._image_path = img
    cell._caption_mode = True
    cell.caption_edit.setPlainText("new caption")

    seen: list[dict] = []
    real = grid_view_widget.atomic_write_text

    def recorder(path, content, **kwargs):
        seen.append(kwargs)
        real(path, content, **kwargs)

    monkeypatch.setattr(grid_view_widget, "atomic_write_text", recorder)
    emitted: list[tuple] = []
    cell.caption_edited.connect(lambda *args: emitted.append(args))
    cell._save_caption()
    app.processEvents()

    assert txt.read_text(encoding="utf-8") == "new caption"
    assert seen and "newline" not in seen[0]
    assert emitted and emitted[0][0] == txt  # 保存成功後にだけ undo 用シグナルが出る
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png", "a.txt"]


def test_grid_caption_save_failure_keeps_file_and_skips_signal(tmp_path: Path, monkeypatch):
    app = _qt_app()
    from locale_manager import LocaleManager

    import caption_core
    import grid_view_widget

    img = _make_image(tmp_path / "a.png")
    txt = img.with_suffix(".txt")
    txt.write_text("old caption", encoding="utf-8")

    locale = LocaleManager("en", Path(caption_core.__file__).parent.parent / "lang")
    cell = grid_view_widget.ImageEditCellWidget(locale)
    cell._image_path = img
    cell._caption_mode = True
    cell.caption_edit.setPlainText("new caption")

    def boom(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(grid_view_widget, "atomic_write_text", boom)
    emitted: list[tuple] = []
    failures: list[str] = []
    cell.caption_edited.connect(lambda *args: emitted.append(args))
    cell.caption_save_failed.connect(lambda msg: failures.append(msg))
    cell._save_caption()
    app.processEvents()

    assert txt.read_text(encoding="utf-8") == "old caption"
    assert emitted == []       # 失敗時は undo 用シグナルを出さない
    assert failures            # 失敗は GUI ログ経路へ


def test_main_window_caption_save_uses_atomic_writer(tmp_path: Path, monkeypatch):
    """main_window._save_current_caption も共有部品を通り、成功後にだけ Undo を積む。"""
    app = _qt_app()
    import app_settings as A
    import constants as C
    import main_window
    import onnx_providers as OP
    import tagging_core as TC
    import utils as U

    config_path = tmp_path / "config.ini"
    for module in (A, C, U, TC):
        monkeypatch.setattr(module, "CONFIG_PATH", config_path)
    monkeypatch.setattr(OP, "has_nvidia_gpu", lambda *a, **k: False)

    window = main_window.MainWindow()
    app.processEvents()
    try:
        img = _make_image(tmp_path / "a.png")
        txt = img.with_suffix(".txt")
        txt.write_text("old caption", encoding="utf-8")
        window._current_caption_image_path = img
        monkeypatch.setattr(window, "_is_text_output_mode", lambda: True)
        window.caption_text_edit.setPlainText("new caption")

        seen: list[dict] = []
        real = main_window.atomic_write_text

        def recorder(path, content, **kwargs):
            seen.append(kwargs)
            real(path, content, **kwargs)

        monkeypatch.setattr(main_window, "atomic_write_text", recorder)
        pushed: list[object] = []
        monkeypatch.setattr(window.undo_manager, "push", lambda action: pushed.append(action))
        monkeypatch.setattr(window, "_update_undo_redo_buttons", lambda: None)

        window._save_current_caption()
        app.processEvents()

        assert txt.read_text(encoding="utf-8") == "new caption"
        assert seen and "newline" not in seen[0]
        assert len(pushed) == 1 and type(pushed[0]).__name__ == "EditCaptionAction"

        # 失敗時: ファイルは元のまま、Undo も積まない
        window.caption_text_edit.setPlainText("third caption")
        pushed.clear()
        monkeypatch.setattr(main_window, "atomic_write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
        window._save_current_caption()
        app.processEvents()
        assert txt.read_text(encoding="utf-8") == "new caption"
        assert pushed == []
    finally:
        window.close()
