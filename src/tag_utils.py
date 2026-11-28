from pathlib import Path
from typing import Sequence
import csv

import constants
from text_io import atomic_write_text, read_text
from utils import write_debug_log

def get_txt_path(image_path: Path) -> Path:
    """Return the path to the txt file corresponding to the image path."""
    return image_path.with_suffix('.txt')

def normalize_tag_whitespace(value: str) -> str:
    """Collapse every whitespace run (including the full-width space) to one space.

    REQ-H2: the single-tag add used to normalize whitespace while the grid and bulk
    add did not, so "long  hair" and "long hair" were stored as different tags and
    one of them was invisible to search. Every add entry point now uses this helper.
    """
    return ' '.join(value.split())

def parse_tag_input(raw: str) -> list[str]:
    """Split comma-separated user input into whitespace-normalized, non-empty tags.

    REQ-H2: shared by the single add, the grid add and the bulk add so the same
    typed text becomes the same tag no matter which entry point is used.
    """
    tags: list[str] = []
    for part in normalize_tag_whitespace(raw).split(','):
        tag = normalize_tag_whitespace(part)
        if tag:
            tags.append(tag)
    return tags

def filter_new_tags(existing_tags: Sequence[str], tags_to_add: Sequence[str]) -> list[str]:
    """Return the tags that are not already present, comparing case-insensitively.

    REQ-H1: tag merging (tagging_core.merge_tags) already treats `solo` and `Solo` as
    the same tag; the manual entry points compared them exactly and could therefore
    create a duplicate that no later aggregation/deletion operation could remove
    (CUT-1 leaves those comparison rules alone). This is the shared entry-side gate.
    """
    seen = {tag.lower() for tag in existing_tags}
    new_tags: list[str] = []
    for tag in tags_to_add:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)          # duplicate spellings inside the input collapse to one
        new_tags.append(tag)
    return new_tags

def get_tag_translations(mapping: dict[str, list[str]], tag: str) -> list[str] | None:
    """Look up a tag's translations, retrying with the map's normalized key form.

    REQ-H5: the map keys are normalized (`_` -> ' ') by load_tag_translation_map(),
    but the display side looked tags up verbatim. With underscore conversion off the
    three tag views silently lost translations. The map normalization itself is
    intentionally unchanged; this is only the read-side fallback.
    """
    translations = mapping.get(tag)
    if translations is None:
        translations = mapping.get(tag.replace('_', ' '))
    return translations

def is_image_sidecar(txt_path: Path) -> bool:
    """Return True when `txt_path` is the .txt sidecar of an existing image.

    REQ-H3: bulk add/delete and TagLoader used to edit/aggregate every *.txt in the
    folder, including unrelated text files. This matches the app's existing target
    definition (constants.IMAGE_EXTENSIONS + same-stem sidecar).
    """
    if txt_path.suffix.lower() != '.txt':
        return False
    return any(txt_path.with_suffix(ext).is_file() for ext in constants.IMAGE_EXTENSIONS)

def get_sidecar_txt_paths(base_dir: Path) -> list[Path]:
    """Return the sorted .txt sidecars of images under `base_dir`.

    REQ-H3/REQ-H4: the bulk workers and TagLoader share this target set, and the
    sorted order makes aggregation independent of the OS directory enumeration order.
    """
    return sorted(path for path in base_dir.rglob('*.txt') if is_image_sidecar(path))

def read_tags(txt_path: Path) -> list[str]:
    """Read tags from a txt file and return them as a list."""
    if not txt_path.is_file():
        return []
    try:
        # REQ-B4: BOM 付きの .txt でも先頭タグが "\ufefffoo" にならないよう
        # utf-8-sig で読む（BOM 無しはそのまま）。REQ-B2: 長パス変換も通す。
        content = read_text(txt_path, encoding='utf-8-sig').strip()
        if not content:
            return []
        # Remove leading/trailing whitespace from tags and exclude empty tags
        return [tag.strip() for tag in content.split(',') if tag.strip()]
    except Exception as e:
        # REQ-2.10: print は windowed ビルドで sys.stdout が無いとそれ自体が例外になり、
        # フォールバックの return [] まで到達しない。本体側と同じ debug log へ残す
        # （write_debug_log は [Debug] debug_log 有効時のみ記録する既存方針のまま）。
        write_debug_log(f"tag_utils.read_tags failed for {txt_path}: {type(e).__name__}: {e}")
        return []

def read_tags_strict(txt_path: Path) -> tuple[bool, list[str]]:
    """タグを読む。**読めなかった場合と「空」を区別する**（NFR-11 / REQ-4.1）。

    戻り値は `(ok, tags)`。ファイルが無い場合は `(True, [])`（空と同義）。
    存在するのに読めない（権限・ロック・復号失敗）場合は `(False, [])` を返し、
    呼び出し側は書き込まずに中止する。読めない既存タグを「空だった」として
    上書きしないための入口。
    """
    if not txt_path.is_file():
        return True, []
    try:
        content = read_text(txt_path, encoding='utf-8-sig').strip()
    except (OSError, UnicodeError, ValueError) as e:
        write_debug_log(
            f"tag_utils.read_tags_strict failed for {txt_path}: {type(e).__name__}: {e}")
        return False, []
    if not content:
        return True, []
    return True, [tag.strip() for tag in content.split(',') if tag.strip()]


def write_tags(txt_path: Path, tags: list[str]) -> bool:
    """Write a list of tags to a txt file (overwriting existing content).

    REQ-2.10: 書き込みは一時ファイル + os.replace で原子的に行い、成否を戻り値で返す。
    失敗しても例外は上げない（呼び出し側の現行契約を維持しつつ、成功扱いだけを防ぐ）。
    """
    # Join tags with ", " as separator
    content = ', '.join(tags)
    try:
        atomic_write_text(txt_path, content, fsync=False, verify=False)
        return True
    except Exception as e:
        write_debug_log(f"tag_utils.write_tags failed for {txt_path}: {type(e).__name__}: {e}")
        return False

def add_tags_to_file(txt_path: Path, tags_to_add: list[str]) -> bool:
    """Add new tags to an existing tag file (avoiding duplicates).

    戻り値は「新規タグがあって、かつ書き込みに成功した」ときだけ True（REQ-2.10）。
    """
    existing_tags = read_tags(txt_path)
    added = False
    for tag in tags_to_add:
        if tag not in existing_tags:
            existing_tags.append(tag)
            added = True

    if not added:
        return False
    return write_tags(txt_path, existing_tags)

def remove_tag_from_file(txt_path: Path, tag_to_remove: str) -> bool:
    """Remove the specified tag from the tag file。

    戻り値は「削除対象が存在し、かつ書き込みに成功した」ときだけ True（REQ-2.10）。
    """
    existing_tags = read_tags(txt_path)
    if tag_to_remove not in existing_tags:
        return False
    existing_tags.remove(tag_to_remove)
    return write_tags(txt_path, existing_tags)

# Must match the shipped filenames exactly (selected_tags_zh_CN.csv / _zh_TW.csv) -
# on a case-sensitive filesystem the lowercase forms silently fail to load.
_TRANSLATION_LANGUAGE_SUFFIXES = ["jp", "fr", "de", "es", "ru", "zh_CN", "zh_TW", "ko"]

def load_tag_translation_map(model_dir: Path, *fallback_dirs: Path | None) -> dict[str, list[str]]:
    """
    指定されたディレクトリの `selected_tags.csv`（英語, PixAI形式）を基準に、
    `selected_tags_<lang>.csv` が存在する言語だけ翻訳を持たせた辞書を作る。
    戻り値の形式: { 'english_tag': ['Japanese', 'French', 'German', 'Spanish', 'Russian', 'Zh_CN', 'Zh_TW', 'Korean'] }

    キーはタグ名文字列そのものなので、この辞書はPixAI以外のモデルのタグにも
    そのままlookupできる（2026-08-31、design.md 6.7節改訂）。呼び出し側
    （main_window.py / model_mode_controller.py）は常にPixAI自身のディレクトリ
    （`constants.MODEL_PATH.parent`）を渡す運用にし、モデルごとに翻訳CSVを
    個別管理することはしない。danbooru系タグ語彙はモデル間でかなり重複するため
    大半のタグは翻訳が効き、PixAI側に無いタグ（他モデル固有の語彙）は英語表示に
    フォールバックする。

    以前は9ファイル全部が揃っていないと（1つでも欠けると）全言語分を空で返す
    「全部か無か」のゲートだったが、言語ごとに独立して存在チェックする方式に
    変更した。存在しない言語の翻訳は英語タグ名でフォールバックする。

    CSVファイルの構造:
    - English (selected_tags.csv): id,tag_id,name,category,count,ips (nameが3列目、インデックス2)
    - 各言語 (selected_tags_<lang>.csv): 1列目（インデックス0）に翻訳が格納されている
    """
    mapping: dict[str, list[str]] = {}

    # 探索先は「渡されたディレクトリ → fallback_dirs」の順。frozen ビルドでは同梱CSVが
    # models/pixai-tagger-v0.9 に配置される一方、旧レイアウトのインストールでは
    # MODEL_PATH.parent が pixai-tagger-v0.9-onnx/ を指すことがあり、両者が食い違う。
    # ファイル単位でフォールバックすることで、配置がどちらでも翻訳が効く。
    search_dirs: list[Path] = [model_dir]
    for extra in fallback_dirs:
        if extra is not None and extra not in search_dirs:
            search_dirs.append(extra)

    def _find(file_name: str) -> Path | None:
        for directory in search_dirs:
            candidate = directory / file_name
            if candidate.is_file():
                return candidate
        return None

    english_csv_path = _find("selected_tags.csv")
    if english_csv_path is None:
        return mapping

    # REQ-H6: resolve translations per generation (directory), not per file. The
    # English CSV defines the generation; each language is taken from the same
    # directory when present. Only a language missing there falls back to the old
    # file-level search, and that mixing is recorded because the row counts can
    # coincidentally match while the translations belong to another generation.
    english_dir = english_csv_path.parent

    def _find_language(file_name: str) -> Path | None:
        same_generation = english_dir / file_name
        if same_generation.is_file():
            return same_generation
        fallback = _find(file_name)
        if fallback is not None:
            write_debug_log(
                f"tag translations: {file_name} was not found in {english_dir}; using "
                f"{fallback.parent} instead (generations may be mixed)"
            )
        return fallback

    language_paths = [_find_language(f"selected_tags_{suffix}.csv") for suffix in _TRANSLATION_LANGUAGE_SUFFIXES]
    if not any(p is not None for p in language_paths):
        # このモデルには翻訳ファイルが一切無い（PixAI以外の全モデルはこちらに該当する）。
        # 呼び出し側は空辞書に対して英語タグへフォールバックするので、これ以上何もしなくてよい。
        return mapping

    def _read_data_records(path: Path) -> list[list[str]]:
        """ヘッダーを読み飛ばし、`csv.reader` の論理レコード単位で全行を返す。

        クォート内の改行を含むフィールドも 1 レコードとして数える（物理行で数えない）。
        """
        with open(path, 'r', encoding='utf-8') as handle:
            reader = csv.reader(handle)
            next(reader, None)  # ヘッダー
            return list(reader)

    try:
        # 英語版CSVを読み込む（ベースとなる）。REQ-5.13: 短い行・空行を「詰めて落とす」と
        # 以降の翻訳が全部 1 行ずつズレるため、空文字のプレースホルダで位置を保つ。
        english_tags: list[str] = []
        for row in _read_data_records(english_csv_path):
            if len(row) >= 3:
                # 3列目（インデックス2）がタグ名、アンダースコアをスペースに変換
                english_tags.append(row[2].strip().replace('_', ' '))
            else:
                english_tags.append('')

        # 各翻訳ファイルを読み込む（存在しないファイルは空リストのまま＝英語でフォールバック）
        all_translations: list[list[str]] = []
        for suffix, lang_path in zip(_TRANSLATION_LANGUAGE_SUFFIXES, language_paths):
            translations: list[str] = []
            if lang_path is not None:
                translations = [
                    (row[0].strip().rstrip(',') if row else '')
                    for row in _read_data_records(lang_path)
                ]
                # REQ-5.14: レコード数が英語と食い違う翻訳は位置合わせ自体が不可能。
                # ズレた翻訳を黙って出さず、その言語だけ英語へフォールバックする
                # （存在しないファイルと同じ扱い。他の言語は巻き込まない）。
                if len(translations) != len(english_tags):
                    write_debug_log(
                        f"tag translations: {lang_path.name} has {len(translations)} data "
                        f"records but selected_tags.csv has {len(english_tags)}; dropping "
                        f"{suffix} and falling back to English"
                    )
                    translations = []
            all_translations.append(translations)

        # 英語タグと翻訳を対応付ける（同じレコード位置どうし）
        for i, en_tag in enumerate(english_tags):
            if en_tag:
                trans_list = []
                for trans_data in all_translations:
                    if i < len(trans_data) and trans_data[i]:
                        trans_list.append(trans_data[i])
                    else:
                        # データがない場合は英語タグをそのまま使用
                        trans_list.append(en_tag)
                mapping[en_tag] = trans_list

    except Exception as e:
        # REQ-5.14: print / traceback.print_exc() は windowed ビルドで stdout/stderr が
        # 無いとそれ自体が落ちるうえ利用者の手元に残らない。例外情報ごと debug log へ。
        # 影響範囲は現行どおり全体（mapping は空のまま＝全言語が英語表示にフォールバック）。
        write_debug_log(
            f"tag translations: failed to load {english_csv_path}: "
            f"{type(e).__name__}: {e}"
        )

    return mapping
