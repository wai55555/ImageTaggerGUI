from pathlib import Path
import csv

from text_io import atomic_write_text
from utils import write_debug_log

def get_txt_path(image_path: Path) -> Path:
    """Return the path to the txt file corresponding to the image path."""
    return image_path.with_suffix('.txt')

def read_tags(txt_path: Path) -> list[str]:
    """Read tags from a txt file and return them as a list."""
    if not txt_path.is_file():
        return []
    try:
        content = txt_path.read_text(encoding='utf-8').strip()
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

    language_paths = [_find(f"selected_tags_{suffix}.csv") for suffix in _TRANSLATION_LANGUAGE_SUFFIXES]
    if not any(p is not None for p in language_paths):
        # このモデルには翻訳ファイルが一切無い（PixAI以外の全モデルはこちらに該当する）。
        # 呼び出し側は空辞書に対して英語タグへフォールバックするので、これ以上何もしなくてよい。
        return mapping

    try:
        # 英語版CSVを読み込む（ベースとなる）
        english_tags = []
        with open(english_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダーをスキップ
            for row in reader:
                if len(row) >= 3:
                    # 3列目（インデックス2）がタグ名、アンダースコアをスペースに変換
                    tag = row[2].strip().replace('_', ' ')
                    english_tags.append(tag)

        # 各翻訳ファイルを読み込む（存在しないファイルは空リストのまま＝全行フォールバック）
        all_translations: list[list[str]] = []
        for lang_path in language_paths:
            translations: list[str] = []
            if lang_path is not None:
                with open(lang_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # ヘッダーをスキップ
                    for row in reader:
                        if len(row) > 0:
                            # カンマで終わる場合は削除
                            trans = row[0].strip().rstrip(',')
                            translations.append(trans if trans else '')
                        else:
                            translations.append('')
            all_translations.append(translations)

        # 英語タグと翻訳を対応付ける
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
        print(f"Error loading tag translations: {e}")
        import traceback
        traceback.print_exc()

    return mapping
