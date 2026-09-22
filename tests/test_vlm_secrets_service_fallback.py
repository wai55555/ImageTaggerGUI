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


def test_set_secret_writes_where_get_secret_reads_when_both_have_it(fake_keyring):
    """両サービスに同じ ref の値があるとき、保存先と読み出し先が食い違ってはいけない。

    260922 PR#27 レビュー指摘: _service_for が旧を先に見ていたため、保存は旧へ
    行くのに get_secret は新を先に読み、保存した値が以後一切読まれなかった。
    """
    fake_keyring.set_password(VS._SERVICE, "vlm/gemini/api_key", "new-value")
    fake_keyring.set_password(VS._LEGACY_SERVICE, "vlm/gemini/api_key", "legacy-value")
    assert VS.set_secret("vlm/gemini/api_key", "typed-now", persist=True) is True
    # 読み出し側（新サービス優先）が、今保存した値を返す。
    assert VS.get_secret("vlm/gemini/api_key") == "typed-now"
    assert fake_keyring.get_password(VS._SERVICE, "vlm/gemini/api_key") == "typed-now"
    # 旧サービス側は触らない（delete_secret が両方から消すので残っていても害はない）。
    assert fake_keyring.get_password(VS._LEGACY_SERVICE, "vlm/gemini/api_key") == "legacy-value"


def test_set_secret_still_creates_in_new_service_when_legacy_lookup_raises(fake_keyring,
                                                                          monkeypatch):
    """keyring の読みが落ちても、default が _SERVICE なので新規作成経路は死なない。"""
    def _boom(service, ref):
        raise Exception("backend unavailable")
    monkeypatch.setattr(fake_keyring, "get_password", _boom)
    assert VS.set_secret("vlm/openai/api_key", "fresh", persist=True) is True
    monkeypatch.undo()
    assert fake_keyring.get_password(VS._SERVICE, "vlm/openai/api_key") == "fresh"
    assert fake_keyring.get_password(VS._LEGACY_SERVICE, "vlm/openai/api_key") is None
