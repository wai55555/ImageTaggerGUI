import sys
import locale
from pathlib import Path

import requests

if __name__ == '__main__':
    # If executed directly as a script, add the project root to the path.
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from constants import MODEL_POINTER_PATH, BASE_DIR
from utils import write_debug_log, GetString, default_get_string_fallback
from app_settings import load_config, load_settings
from locale_manager import LocaleManager, available_language_codes, normalize_language_code

def get_model_info_from_pointer(url: str, get_string: GetString | None = None) -> tuple[str | None, int | None]:
    """
        Retrieve the pointer file from the specified URL and extract the SHA256 OID and size.

    Args:
        url (str): model pointer file
        get_string (GetString): function to retrieve localized strings.

    Returns:
        A tuple containing the SHA256 OID (str) and the size (int). (None, None) if retrieval or parsing fails.
    """
    _get_string: GetString = get_string if get_string else default_get_string_fallback

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception if the status code is not in the 200 range.
        
        lines = response.text.strip().split('\n')
        oid_line = next((line for line in lines if line.startswith('oid sha256:')), None)
        size_line = next((line for line in lines if line.startswith('size ')), None)

        if not oid_line or not size_line:
            write_debug_log(_get_string("GetPointerHuggingface", "Error_OidOrSizeNotFound"), _get_string)
            return None, None

        sha256_oid = oid_line.split(':')[1].strip()
        size = int(size_line.split(' ')[1].strip())
        
        return sha256_oid, size

    except requests.exceptions.RequestException as e:
        write_debug_log(_get_string("GetPointerHuggingface", "Error_FetchFailed", e=e), _get_string)
        return None, None
    except (IndexError, ValueError) as e:
        write_debug_log(_get_string("GetPointerHuggingface", "Error_ParseFailed", e=e), _get_string)
        return None, None

if __name__ == '__main__':
    config = load_config()
    settings = load_settings(config)
    # `'_' の前だけを取る` と zh_CN / zh_TW が存在しない "zh" に潰れるので、
    # main_window と同じ正規化を通す（同 ini が無いコードは en へ落ちる）。
    os_lang = normalize_language_code(
        locale.getdefaultlocale()[0] or "", available_language_codes(BASE_DIR / "lang"))
    locale_manager = LocaleManager(settings.language_code or os_lang, BASE_DIR)
    _ = locale_manager.get_string

    sha256, size = get_model_info_from_pointer(str(MODEL_POINTER_PATH), get_string=_)
    if sha256 and size:
        print(f"Model SHA256: {sha256}")
        print(f"Model Size: {size} bytes")

