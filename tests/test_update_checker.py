"""update_checker.check_for_update() の単体テスト（Qt非依存、offscreen不要）。

Run:  rtk pytest tests/test_update_checker.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

import update_checker as UC


class _FakeResp:
    def __init__(self, status_code=200, payload=None, raise_on_json=None):
        self.status_code = status_code
        self._payload = payload
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise self._raise_on_json
        return self._payload


# A plausible html_url that passes the RELEASE_URL_PREFIX allowlist.
_URL = UC.RELEASE_URL_PREFIX + "tag/v1.8.0"


def _http(status_code=200, payload=None, raise_on_json=None):
    def _get(url, *, headers=None, timeout=5):
        assert url == UC.API_URL
        return _FakeResp(status_code, payload, raise_on_json)
    return _get


def _http_raises(exc):
    def _get(url, *, headers=None, timeout=5):
        raise exc
    return _get


def test_returns_update_info_when_newer_version_available():
    http = _http(200, {"tag_name": "v1.8.0", "html_url": _URL,
                       "published_at": "2026-01-01T00:00:00Z"})
    info = UC.check_for_update("1.7.0", http_get=http)
    assert info is not None
    assert info.version == "v1.8.0"
    assert info.html_url == _URL
    assert info.published_at == "2026-01-01T00:00:00Z"


def test_returns_none_when_same_version():
    http = _http(200, {"tag_name": "v1.7.0", "html_url": _URL})
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_when_older_version():
    http = _http(200, {"tag_name": "v1.6.0", "html_url": _URL})
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_on_404():
    http = _http(404, {"message": "Not Found"})
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_on_500():
    http = _http(500, None)
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_on_network_error():
    import requests
    http = _http_raises(requests.exceptions.ConnectionError("no network"))
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_on_timeout():
    import requests
    http = _http_raises(requests.exceptions.Timeout("timed out"))
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_on_invalid_json():
    http = _http(200, None, raise_on_json=ValueError("bad json"))
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_when_tag_name_missing():
    http = _http(200, {"html_url": _URL})
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_when_tag_name_has_no_v_prefix_but_is_still_a_version():
    # "1.8.0" without a leading v is still parseable and still newer - only truly
    # unparseable tag names (e.g. a codename) should return None.
    http = _http(200, {"tag_name": "1.8.0", "html_url": _URL})
    info = UC.check_for_update("1.7.0", http_get=http)
    assert info is not None
    assert info.version == "1.8.0"


def test_returns_none_when_tag_name_is_empty():
    http = _http(200, {"tag_name": "", "html_url": _URL})
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_when_tag_name_is_not_a_version():
    http = _http(200, {"tag_name": "v1.7.0-beta", "html_url": _URL})
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_when_html_url_missing():
    http = _http(200, {"tag_name": "v1.8.0"})
    assert UC.check_for_update("1.7.0", http_get=http) is None


@pytest.mark.parametrize("bad_url", [
    "https://evil.example/releases/tag/v1.8.0",      # other host
    "http://github.com/wai55555/PixaiTaggerOnnxGui/releases/tag/v1.8.0",  # plain http
    "https://github.com/someone-else/repo/releases/tag/v1.8.0",           # other repo
    "file:///C:/Windows/System32/calc.exe",
    "javascript:alert(1)",
    "",
    None,
    123,
])
def test_returns_none_when_html_url_is_not_our_release_page(bad_url):
    """html_url goes straight to webbrowser.open() (os.startfile on Windows), so
    anything outside our own /releases/ pages is refused rather than opened."""
    http = _http(200, {"tag_name": "v1.8.0", "html_url": bad_url})
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_accepts_html_url_under_our_release_prefix():
    http = _http(200, {"tag_name": "v1.8.0", "html_url": UC.RELEASE_URL_PREFIX + "tag/v1.8.0"})
    info = UC.check_for_update("1.7.0", http_get=http)
    assert info is not None
    assert info.html_url.startswith("https://github.com/wai55555/PixaiTaggerOnnxGui/releases/")


def test_returns_none_when_payload_is_not_a_dict():
    http = _http(200, ["not", "a", "dict"])
    assert UC.check_for_update("1.7.0", http_get=http) is None


def test_returns_none_when_current_version_is_unparseable():
    http = _http(200, {"tag_name": "v1.8.0", "html_url": _URL})
    assert UC.check_for_update("dev", http_get=http) is None


def test_published_at_defaults_to_empty_string_when_not_a_string():
    http = _http(200, {"tag_name": "v1.8.0", "html_url": _URL, "published_at": None})
    info = UC.check_for_update("1.7.0", http_get=http)
    assert info is not None
    assert info.published_at == ""


def test_version_tuple_comparison_not_string_comparison():
    """"1.7.10" must be treated as newer than "1.7.2" - a naive string compare
    would get this backwards ("1.7.10" < "1.7.2" lexicographically)."""
    http = _http(200, {"tag_name": "v1.7.10", "html_url": _URL})
    info = UC.check_for_update("1.7.2", http_get=http)
    assert info is not None
    assert info.version == "v1.7.10"

    http2 = _http(200, {"tag_name": "v1.7.2", "html_url": _URL})
    assert UC.check_for_update("1.7.10", http_get=http2) is None


def test_parse_version():
    assert UC.parse_version("v1.7.0") == (1, 7, 0)
    assert UC.parse_version("V1.7.0") == (1, 7, 0)
    assert UC.parse_version("1.7.0") == (1, 7, 0)
    assert UC.parse_version("") is None
    assert UC.parse_version(None) is None
    assert UC.parse_version("v1.7.0-beta") is None
    assert UC.parse_version("not-a-version") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# app_settings: the three [Behavior] keys the updater persists
# --------------------------------------------------------------------------

def test_parse_update_check_states_and_aliases():
    import app_settings as A
    assert A.parse_update_check("ask") == "ask"
    assert A.parse_update_check("  DISMISSED ") == "dismissed"
    assert A.parse_update_check("") == "ask"
    assert A.parse_update_check("garbage") == "ask"
    # hand-edited "turn it off" spellings must not silently mean "keep asking"
    for off in ("false", "No", "OFF", "0", "never", "disabled", "disable", "none"):
        assert A.parse_update_check(off) == "dismissed", off


def test_update_settings_round_trip_through_config_file(monkeypatch, tmp_path):
    import configparser
    import app_settings as A
    cfg_path = tmp_path / "config.ini"
    monkeypatch.setattr(A, "CONFIG_PATH", cfg_path)

    s = A.load_settings(A.get_default_config())
    assert (s.behavior.update_check, s.behavior.update_check_last, s.behavior.update_skip_version) == ("ask", "", "")
    s.behavior.update_check = "dismissed"
    s.behavior.update_check_last = "2026-09-20T01:02:03.456789+00:00"
    s.behavior.update_skip_version = "v1.8.0"
    assert A.save_config(s) is True

    raw = configparser.ConfigParser(interpolation=None)
    raw.read(cfg_path, encoding="utf-8")
    assert raw["Behavior"]["update_check"] == "dismissed"
    assert raw["Behavior"]["update_check_last"] == "2026-09-20T01:02:03.456789+00:00"
    assert raw["Behavior"]["update_skip_version"] == "v1.8.0"

    reloaded = A.load_settings(A.load_config())
    assert reloaded.behavior.update_check == "dismissed"
    assert reloaded.behavior.update_check_last == "2026-09-20T01:02:03.456789+00:00"
    assert reloaded.behavior.update_skip_version == "v1.8.0"


def test_missing_update_keys_in_old_config_fall_back_to_defaults():
    """A config.ini written by a pre-updater release has none of the three keys."""
    import configparser
    import app_settings as A
    cfg = A.get_default_config()
    for key in ("update_check", "update_check_last", "update_skip_version"):
        cfg.remove_option("Behavior", key)
    s = A.load_settings(cfg)
    assert s.behavior.update_check == "ask"
    assert s.behavior.update_check_last == ""
    assert s.behavior.update_skip_version == ""
