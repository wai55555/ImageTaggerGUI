"""Shared tag translation map tests."""

from types import SimpleNamespace

import main_window
from model_registry import get_model_entry


def test_pixai_v1_model_uses_shared_tag_translations(monkeypatch):
    entry = get_model_entry("pixai-tagger-v1.0")
    assert entry is not None
    assert "tag_translation" not in entry.config

    expected = {"1girl": ["1 Mädchen"]}
    calls = []

    def fake_load_tag_translation_map(*paths):
        calls.append(paths)
        return expected

    monkeypatch.setattr(main_window, "load_tag_translation_map", fake_load_tag_translation_map)
    window = SimpleNamespace()

    assert main_window.MainWindow._load_shared_tag_translation_map(window) == expected
    assert len(calls) == 1


def test_pixai_v09_model_still_loads_shared_tag_translations(monkeypatch):
    entry = get_model_entry("pixai-tagger-v0.9")
    assert entry is not None

    expected = {"1girl": ["1 Mädchen"]}
    calls = []

    def fake_load_tag_translation_map(*paths):
        calls.append(paths)
        return expected

    monkeypatch.setattr(main_window, "load_tag_translation_map", fake_load_tag_translation_map)
    window = SimpleNamespace()

    assert main_window.MainWindow._load_shared_tag_translation_map(window) == expected
    assert len(calls) == 1
