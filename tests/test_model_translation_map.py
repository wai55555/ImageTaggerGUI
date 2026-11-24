"""Model-specific tag translation selection tests."""

from types import SimpleNamespace

import main_window
from model_registry import get_model_entry


def test_english_only_model_skips_shared_tag_translations():
    entry = SimpleNamespace(config={"tag_translation": "english_only"})
    window = SimpleNamespace(_current_model_entry=lambda: entry)

    assert main_window.MainWindow._load_tag_translation_map_for_model(window) == {}


def test_pixai_v09_model_still_loads_shared_tag_translations(monkeypatch):
    entry = get_model_entry("pixai-tagger-v0.9")
    assert entry is not None

    expected = {"1girl": ["1 Mädchen"]}
    calls = []

    def fake_load_tag_translation_map(*paths):
        calls.append(paths)
        return expected

    monkeypatch.setattr(main_window, "load_tag_translation_map", fake_load_tag_translation_map)
    window = SimpleNamespace(_current_model_entry=lambda: entry)

    assert main_window.MainWindow._load_tag_translation_map_for_model(window) == expected
    assert len(calls) == 1
