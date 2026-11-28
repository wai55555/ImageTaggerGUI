"""Phase H（docs/261127_plan-review-findings-fix.md §14）の回帰テスト。

対象: R27-1 / R27-2 / R18-1 / R19-1 / R5-7 / R17-2 / R10-4（REQ-H1〜H7）。

オフラインのみ。実 config.ini・実 models/ は触らず、フィクスチャはすべて
tmp_path に作る。GUI は QT_QPA_PLATFORM=offscreen で動かす。
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest
from PIL import Image
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QGridLayout, QPushButton

_APP = QApplication.instance() or QApplication([])

import main_window
import tag_utils
import utils
import workers
from grid_view_widget import ImageEditCellWidget
from locale_manager import LocaleManager


def _get_string(section, key, **kwargs):
    return key


def _make_image(folder: Path, stem: str, ext: str = ".png") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}{ext}"
    Image.new("RGB", (8, 8)).save(path)
    return path


def _run_tag_loader(folder: Path) -> list[tuple[str, int]]:
    loader = workers.TagLoader(folder, _get_string)
    emitted: list[list[tuple[str, int]]] = []
    loader.tags_loaded.connect(emitted.append)
    loader.run()
    return emitted[-1] if emitted else []


def _run_bulk_add(folder: Path, tags: str, prepend: bool = False) -> workers.BulkTagWorker:
    worker = workers.BulkTagWorker(_get_string)
    worker.run_bulk_add(folder, tags, prepend)
    return worker


# ---------------------------------------------------------------------------
# REQ-H1 / REQ-H2: entry-side identity and whitespace normalization
# ---------------------------------------------------------------------------


def test_parse_tag_input_normalizes_whitespace_for_every_entry():
    assert tag_utils.parse_tag_input("  long   hair , blue\u3000eyes ") == [
        "long hair", "blue eyes",
    ]
    assert tag_utils.parse_tag_input(",,, ,") == []


def test_filter_new_tags_is_case_insensitive_and_dedupes_input():
    existing = ["solo", "blue eyes"]
    assert tag_utils.filter_new_tags(existing, ["Solo", "SOLO", "new tag", "new tag"]) == [
        "new tag",
    ]
    # merge_tags itself is untouched (CUT-1): it already compares case-insensitively,
    # and the entry-side filter applies the same rule rather than changing it.
    import tagging_core

    assert tagging_core.merge_tags(["solo"], ["Solo"]) == ["solo"]


class _FakeListItem:
    def __init__(self, relative_name: str):
        self._relative_name = relative_name

    def data(self, role):
        return self._relative_name


class _FakeImageList:
    def __init__(self, relative_name: str):
        self._item = _FakeListItem(relative_name)

    def currentItem(self):
        return self._item


class _FakeLineEdit:
    def __init__(self, text: str):
        self._text = text
        self.cleared = False

    def text(self) -> str:
        return self._text

    def clear(self) -> None:
        self._text = ""
        self.cleared = True

    # QToolTip.showText is patched out in the grid test, but its argument is still
    # computed from the widget, so the fake exposes the two calls involved.
    def rect(self):
        from PySide6.QtCore import QRect
        return QRect(0, 0, 1, 1)

    def mapToGlobal(self, point):
        return point


def _fake_main_window(folder: Path, relative_name: str, text: str):
    logs: list[str] = []
    ns = SimpleNamespace(
        image_list=_FakeImageList(relative_name),
        add_single_tag_line=_FakeLineEdit(text),
        settings=SimpleNamespace(paths=SimpleNamespace(input_dir=str(folder))),
        undo_manager=SimpleNamespace(push=lambda action: None),
        # Phase 2 の一本化した判定（処理中は書き込み経路を開始しない）。
        _is_processing=lambda: False,
        update_log=lambda message, color: logs.append(message),
        _update_undo_redo_buttons=lambda: None,
        _load_image_tags=lambda image_path, preserve_page=True: None,
        _update_tag_cache_entry=lambda rel_path: None,
        reload_tags_only=lambda: None,
        locale_manager=SimpleNamespace(get_string=lambda section, key, **kw: key),
    )
    return ns, logs


def test_single_add_rejects_case_variant_and_keeps_new_tags(tmp_path: Path):
    image = _make_image(tmp_path, "a")
    txt = tag_utils.get_txt_path(image)
    txt.write_text("solo", encoding="utf-8")

    fake, logs = _fake_main_window(tmp_path, image.name, "Solo")
    main_window.MainWindow._add_single_tag(fake)
    assert txt.read_text(encoding="utf-8") == "solo"
    assert logs == ["Tag_Already_Exists"]

    fake, _ = _fake_main_window(tmp_path, image.name, "  long   hair ")
    main_window.MainWindow._add_single_tag(fake)
    assert txt.read_text(encoding="utf-8") == "solo, long hair"


def test_grid_add_rejects_case_variant_and_normalizes_whitespace(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("grid_view_widget.QToolTip.showText", lambda *a, **k: None)
    image = _make_image(tmp_path, "a")
    txt = tag_utils.get_txt_path(image)
    txt.write_text("solo", encoding="utf-8")

    added: list[tuple[Path, list[str]]] = []
    ns = SimpleNamespace(
        add_tag_line=_FakeLineEdit("Solo"),
        _image_path=image,
        _editing_enabled=True,  # Phase 2: 錠中はハンドラ側でも書かない
        locale_manager=SimpleNamespace(get_string=lambda section, key, **kw: key),
        _update_tag_display=lambda: None,
        tags_added=SimpleNamespace(emit=lambda path, tags: added.append((path, tags))),
    )
    ImageEditCellWidget._add_tag(ns)
    assert txt.read_text(encoding="utf-8") == "solo"
    assert added == []

    ns.add_tag_line = _FakeLineEdit("  blue   eyes ,, new   tag ")
    ImageEditCellWidget._add_tag(ns)
    assert txt.read_text(encoding="utf-8") == "solo, blue eyes, new tag"
    assert added[-1][1] == ["blue eyes", "new tag"]


def test_bulk_add_rejects_case_variant_and_normalizes_whitespace(tmp_path: Path):
    image = _make_image(tmp_path, "a")
    txt = tag_utils.get_txt_path(image)
    txt.write_text("solo", encoding="utf-8")

    _run_bulk_add(tmp_path, "Solo, blue   eyes")
    assert txt.read_text(encoding="utf-8") == "solo, blue eyes"


# ---------------------------------------------------------------------------
# REQ-H3: bulk operations and TagLoader only target image sidecars
# ---------------------------------------------------------------------------


def test_bulk_operations_and_tag_loader_ignore_unrelated_txt(tmp_path: Path):
    image = _make_image(tmp_path, "a")
    sidecar = tag_utils.get_txt_path(image)
    sidecar.write_text("cat", encoding="utf-8")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("dog", encoding="utf-8")

    _run_bulk_add(tmp_path, "bird")
    assert sidecar.read_text(encoding="utf-8") == "cat, bird"
    assert unrelated.read_text(encoding="utf-8") == "dog"

    tags = _run_tag_loader(tmp_path)
    assert dict(tags) == {"cat": 1, "bird": 1}, "unrelated .txt must not be aggregated"

    workers.BulkTagWorker(_get_string).run_bulk_delete(tmp_path, "dog")
    assert unrelated.read_text(encoding="utf-8") == "dog"
    assert sidecar.read_text(encoding="utf-8") == "cat, bird"


def test_regular_sidecars_still_behave_as_before(tmp_path: Path):
    """§15.3: normal (case-consistent) .txt sets keep the previous add/aggregate/delete."""
    a_image = _make_image(tmp_path, "a")
    b_image = _make_image(tmp_path, "b", ext=".jpg")
    a_txt = tag_utils.get_txt_path(a_image)
    b_txt = tag_utils.get_txt_path(b_image)
    a_txt.write_text("cat, dog", encoding="utf-8")
    b_txt.write_text("cat, bird", encoding="utf-8")

    _run_bulk_add(tmp_path, "eagle")
    assert a_txt.read_text(encoding="utf-8") == "cat, dog, eagle"
    assert b_txt.read_text(encoding="utf-8") == "cat, bird, eagle"

    assert _run_tag_loader(tmp_path) == [
        ("cat", 2), ("eagle", 2), ("dog", 1), ("bird", 1),
    ]

    workers.BulkTagWorker(_get_string).run_bulk_delete(tmp_path, "dog")
    assert a_txt.read_text(encoding="utf-8") == "cat, eagle"
    assert b_txt.read_text(encoding="utf-8") == "cat, bird, eagle"


# ---------------------------------------------------------------------------
# REQ-H4: stable aggregation order
# ---------------------------------------------------------------------------


def test_tag_loader_order_does_not_depend_on_directory_enumeration(tmp_path: Path, monkeypatch):
    for stem, text in (("a", "zebra"), ("b", "alpha")):
        image = _make_image(tmp_path, stem)
        tag_utils.get_txt_path(image).write_text(text, encoding="utf-8")

    expected = [("zebra", 1), ("alpha", 1)]

    assert _run_tag_loader(tmp_path) == expected

    original_rglob = Path.rglob

    def reversed_rglob(self, pattern):
        return iter(sorted(original_rglob(self, pattern), reverse=True))

    monkeypatch.setattr(Path, "rglob", reversed_rglob)
    assert _run_tag_loader(tmp_path) == expected
    assert tag_utils.get_sidecar_txt_paths(tmp_path) == sorted(
        tag_utils.get_sidecar_txt_paths(tmp_path)
    )


# ---------------------------------------------------------------------------
# REQ-H5: translation lookup falls back to the normalized map key
# ---------------------------------------------------------------------------


def test_translation_lookup_falls_back_to_normalized_key():
    mapping = {"long hair": ["長い髪"]}
    assert tag_utils.get_tag_translations(mapping, "long_hair") == ["長い髪"]
    assert tag_utils.get_tag_translations(mapping, "long hair") == ["長い髪"]
    assert tag_utils.get_tag_translations(mapping, "unknown") is None


def test_grid_display_uses_normalized_translation_key(tmp_path: Path):
    image = _make_image(tmp_path, "a")
    tag_utils.get_txt_path(image).write_text("long_hair", encoding="utf-8")

    locale = LocaleManager("en", ROOT / "lang")
    cell = ImageEditCellWidget(locale)
    try:
        cell.load_data(image, 0)
        cell.set_tag_display_language("日本語", {"long hair": ["長い髪"]})
        assert [button.text() for button in cell.tag_buttons] == ["長い髪"]
    finally:
        cell.deleteLater()


def test_main_window_image_view_uses_normalized_translation_key():
    class _Button(QPushButton):
        pass

    button = _Button()

    class _FakeWindow(QObject):
        pass

    ns = _FakeWindow()
    ns.tag_buttons_for_image = []
    ns.settings = SimpleNamespace(
        window=SimpleNamespace(tag_display_cols=4, tag_display_rows=5)
    )
    ns._get_image_tags_per_page = lambda: 20
    ns._tag_display_grid_dims = lambda: (4, 5)  # Phase 5（R28-1）で追加された入口
    ns._get_translation_index = lambda language: 0 if language == "日本語" else -1
    ns._delete_image_tag = lambda tag: None
    ns._current_image_tags = ["long_hair"]
    ns._current_image_tag_page = 0
    ns._tag_display_language = "日本語"
    ns.tag_translation_map = {"long hair": ["長い髪"]}
    ns._tag_button_min_width = 1
    ns._tag_button_min_height = 1
    ns.tag_display_grid = QGridLayout()
    ns.image_tag_prev_page_btn = button
    ns.image_tag_next_page_btn = button
    ns.update_all_button_alignments = lambda: None
    ns._controls_enabled = True  # Phase 2: チップは錠状態を引き継いで生成される

    main_window.MainWindow._display_image_tag_page(ns)
    assert [b.text() for b in ns.tag_buttons_for_image] == ["長い髪"]


# ---------------------------------------------------------------------------
# REQ-H6: translation CSVs are resolved by generation (directory)
# ---------------------------------------------------------------------------


def _write_english_csv(path: Path, names: list[str]) -> None:
    lines = ["id,tag_id,name,category,count,ips"]
    for i, name in enumerate(names, start=1):
        lines.append(f"{i},{i},{name},0,0,0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_translation_csv(path: Path, translations: list[str]) -> None:
    lines = ["translation"]
    lines.extend(f'"{value}"' if value else "" for value in translations)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_translation_csvs_prefer_the_english_csv_directory(tmp_path: Path):
    primary = tmp_path / "model"
    fallback = tmp_path / "bundled"
    primary.mkdir()
    fallback.mkdir()

    _write_english_csv(primary / "selected_tags.csv", ["1girl", "blue_eyes"])
    _write_translation_csv(primary / "selected_tags_jp.csv", ["いち", "あお"])
    _write_translation_csv(primary / "selected_tags_fr.csv", ["une", "bleus"])

    # Another generation in the fallback directory: same row count, different content.
    _write_english_csv(fallback / "selected_tags.csv", ["1girl", "smile"])
    _write_translation_csv(fallback / "selected_tags_jp.csv", ["べつ", "わらい"])
    _write_translation_csv(fallback / "selected_tags_ko.csv", ["하나", "미소"])

    mapping = tag_utils.load_tag_translation_map(primary, fallback)

    jp = tag_utils._TRANSLATION_LANGUAGE_SUFFIXES.index("jp")
    fr = tag_utils._TRANSLATION_LANGUAGE_SUFFIXES.index("fr")
    ko = tag_utils._TRANSLATION_LANGUAGE_SUFFIXES.index("ko")

    # Keys come from the primary English CSV; same-directory translations win.
    assert set(mapping) == {"1girl", "blue eyes"}
    assert mapping["1girl"][jp] == "いち"
    assert mapping["1girl"][fr] == "une"
    # Only the language missing from that directory is filled from the old search.
    assert mapping["1girl"][ko] == "하나"


def test_translation_generation_mixing_is_logged(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "debug_log.txt"
    debug_settings = utils.get_debug_settings()
    original_enabled = debug_settings.debug_log_enabled
    debug_settings.debug_log_enabled = True
    monkeypatch.setattr(utils, "LOG_FILE_PATH", log_path)
    monkeypatch.setattr(utils, "_LOG_FH", None)
    monkeypatch.setattr(utils, "_LOG_LAST_FLUSH", 0.0)
    try:
        primary = tmp_path / "model"
        fallback = tmp_path / "bundled"
        primary.mkdir()
        fallback.mkdir()
        _write_english_csv(primary / "selected_tags.csv", ["1girl"])
        _write_translation_csv(fallback / "selected_tags_ko.csv", ["하나"])

        tag_utils.load_tag_translation_map(primary, fallback)
        utils._flush_debug_log()
        log = log_path.read_text(encoding="utf-8", errors="replace")
        assert "generations may be mixed" in log
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


# ---------------------------------------------------------------------------
# REQ-H7: is_stopped() no longer logs on every poll
# ---------------------------------------------------------------------------


def test_is_stopped_does_not_log_each_poll(tmp_path: Path, monkeypatch):
    log_path = tmp_path / "debug_log.txt"
    debug_settings = utils.get_debug_settings()
    original_enabled = debug_settings.debug_log_enabled
    debug_settings.debug_log_enabled = True
    monkeypatch.setattr(utils, "LOG_FILE_PATH", log_path)
    monkeypatch.setattr(utils, "_LOG_FH", None)
    monkeypatch.setattr(utils, "_LOG_LAST_FLUSH", 0.0)
    try:
        for cls in (workers.TaggerThreadWorker, workers.TagLoader, workers.BulkTagWorker):
            fake = SimpleNamespace(_stop_event=threading.Event())
            fake._stop_event.set()
            for _ in range(5):
                assert cls.is_stopped(fake) is True

        utils._flush_debug_log()
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        assert "is_stopped() returning True" not in log
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
