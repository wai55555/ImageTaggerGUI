"""vlm_secrets の keyring サービス名フォールバック（ImageTaggerGUI.VLM / 旧
PixaiTaggerOnnxGui.VLM）の単体テスト（260920_rename_to_ImageTaggerGUI_plan.md 3.3）。

Offline only - keyring は fake に monkeypatch する。
Run:  rtk pytest tests/test_vlm_secrets_service_fallback.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import vlm_secrets as VS


class _FakeKeyring:
    """`(service, ref) -> value` の dict を持つ最小限の keyring 相当。"""

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service, ref):
        return self._store.get((service, ref))

    def set_password(self, service, ref, value):
        self._store[(service, ref)] = value

    def delete_password(self, service, ref):
        try:
            del self._store[(service, ref)]
        except KeyError:
            raise Exception("not found")


@pytest.fixture
def fake_keyring(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setattr(VS, "keyring", fake)
    monkeypatch.setattr(VS, "_session_store", {})
    return fake


def test_get_secret_reads_legacy_only_key(fake_keyring):
    fake_keyring.set_password(VS._LEGACY_SERVICE, "vlm/gemini/api_key", "legacy-value")
    assert VS.get_secret("vlm/gemini/api_key") == "legacy-value"


def test_set_secret_on_legacy_ref_updates_legacy_not_new(fake_keyring):
    fake_keyring.set_password(VS._LEGACY_SERVICE, "vlm/gemini/api_key", "old-value")
    assert VS.set_secret("vlm/gemini/api_key", "new-value", persist=True) is True
    assert fake_keyring.get_password(VS._LEGACY_SERVICE, "vlm/gemini/api_key") == "new-value"
    assert fake_keyring.get_password(VS._SERVICE, "vlm/gemini/api_key") is None


def test_set_secret_on_new_ref_creates_only_new(fake_keyring):
    assert VS.set_secret("vlm/openai/api_key", "fresh-value", persist=True) is True
    assert fake_keyring.get_password(VS._SERVICE, "vlm/openai/api_key") == "fresh-value"
    assert fake_keyring.get_password(VS._LEGACY_SERVICE, "vlm/openai/api_key") is None


def test_delete_secret_removes_from_both_services(fake_keyring):
    fake_keyring.set_password(VS._SERVICE, "vlm/openai/api_key", "a")
    fake_keyring.set_password(VS._LEGACY_SERVICE, "vlm/openai/api_key", "b")
    assert VS.delete_secret("vlm/openai/api_key") is True
    assert fake_keyring.get_password(VS._SERVICE, "vlm/openai/api_key") is None
    assert fake_keyring.get_password(VS._LEGACY_SERVICE, "vlm/openai/api_key") is None


def test_secret_status_is_keyring_when_only_legacy_has_it(fake_keyring):
    fake_keyring.set_password(VS._LEGACY_SERVICE, "vlm/gemini/api_key", "legacy-value")
    assert VS.secret_status("vlm/gemini/api_key") == "keyring"


def test_get_secret_prefers_new_service_when_both_have_it(fake_keyring):
    fake_keyring.set_password(VS._SERVICE, "vlm/gemini/api_key", "new-value")
    fake_keyring.set_password(VS._LEGACY_SERVICE, "vlm/gemini/api_key", "legacy-value")
    assert VS.get_secret("vlm/gemini/api_key") == "new-value"
