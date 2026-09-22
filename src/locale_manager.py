import configparser
from pathlib import Path
from typing import Any

from utils import write_debug_log

# 同梱している翻訳ファイルの言語コード。lang ディレクトリを走査できなかった場合
# （frozen ビルドの配置失敗など）の最終候補にも使う。
SHIPPED_LANGUAGE_CODES: tuple[str, ...] = (
    "en", "ja", "de", "es", "fr", "ko", "ru", "zh_CN", "zh_TW",
)

# 地域違いの表記を、実在する翻訳ファイルへ寄せる表。中国語は「言語コードを
# 前から切り詰める」方式では絶対に解決できない（zh.ini は無く zh_CN.ini と
# zh_TW.ini がある）ので、ここで明示する。
_REGION_ALIASES: dict[str, str] = {
    # 繁体字圏
    "zh_tw": "zh_TW", "zh_hk": "zh_TW", "zh_mo": "zh_TW", "zh_hant": "zh_TW",
    "zh_hant_tw": "zh_TW", "zh_hant_hk": "zh_TW", "zh_hant_mo": "zh_TW",
    # 簡体字圏
    "zh_cn": "zh_CN", "zh_sg": "zh_CN", "zh_hans": "zh_CN",
    "zh_hans_cn": "zh_CN", "zh_hans_sg": "zh_CN",
    # 地域が付かない "zh" はどちらとも言えないので、話者数の多い簡体字へ寄せる。
    "zh": "zh_CN", "zh_chs": "zh_CN", "zh_cht": "zh_TW",
}

_DEFAULT_LANGUAGE_CODE = "en"


def available_language_codes(*dirs: Path | None) -> list[str]:
    """`dirs` に実在する `<code>.ini` の言語コード一覧を、優先度順に返す。

    走査できたファイルが1つも無ければ SHIPPED_LANGUAGE_CODES を返す
    （翻訳ファイルの配置に失敗していても言語選択を壊さないため）。
    """
    found: list[str] = []
    for directory in dirs:
        if directory is None:
            continue
        try:
            stems = sorted(path.stem for path in directory.glob("*.ini"))
        except OSError as e:
            write_debug_log(f"Cannot list language files in {directory}: {e}")
            continue
        for stem in stems:
            if stem and stem not in found:
                found.append(stem)
    return found or list(SHIPPED_LANGUAGE_CODES)


def normalize_language_code(raw: str, available: list[str] | None = None) -> str:
    """OS 由来の言語タグを、実在する翻訳ファイルの言語コードへ正規化する。

    `zh-TW` / `zh_TW.UTF-8` / `pt_BR@euro` のような表記を受け取り、
    次の順で解決する。どれにも当たらなければ英語へフォールバックする。

    1. 完全一致（大文字小文字と `-`/`_` の違いは無視）
    2. 地域エイリアス（`zh_HK` → `zh_TW`、`zh` → `zh_CN` など）
    3. 言語部分だけの完全一致（`de_DE` → `de`）
    4. その言語で始まる候補が1つだけならそれ（将来 `pt_BR.ini` だけを
       同梱した場合に `pt` を拾えるようにするための段）
    5. 候補が複数あって決められない場合は、`available` の先頭（＝走査順で
       安定）を使う

    以前は Windows の LANGID 表が en/ja/de/fr/ko/zh の6つしか返さず、実在する
    es.ini / ru.ini へ到達できないうえ、中国語は存在しない `zh.ini` を指して
    いた（非 Windows でも `zh_CN` が `zh` へ切り詰められていた）。判定結果は
    config.ini へ保存されるため、初回起動でそうなると以後ずっと英語表示に
    固定される。
    """
    codes = list(available) if available is not None else list(SHIPPED_LANGUAGE_CODES)
    if not codes:
        codes = list(SHIPPED_LANGUAGE_CODES)

    cleaned = str(raw or "").strip()
    # `ja_JP.UTF-8` の codeset、`@euro` 等の modifier を落として区切りを揃える。
    cleaned = cleaned.split(".")[0].split("@")[0].replace("-", "_").strip("_")
    if not cleaned:
        return _DEFAULT_LANGUAGE_CODE

    lowered = cleaned.lower()
    by_lower = {code.lower(): code for code in codes}

    if lowered in by_lower:
        return by_lower[lowered]

    alias = _REGION_ALIASES.get(lowered)
    if alias is not None and alias.lower() in by_lower:
        return by_lower[alias.lower()]

    language = lowered.split("_")[0]
    if language in by_lower:
        return by_lower[language]

    alias = _REGION_ALIASES.get(language)
    if alias is not None and alias.lower() in by_lower:
        return by_lower[alias.lower()]

    same_language = [code for code in codes if code.lower().split("_")[0] == language]
    if same_language:
        if len(same_language) > 1:
            write_debug_log(
                f"Ambiguous OS language '{raw}': using {same_language[0]} "
                f"out of {same_language}")
        return same_language[0]

    write_debug_log(f"No translation file for OS language '{raw}'; falling back to en")
    return _DEFAULT_LANGUAGE_CODE


class LocaleManager:
    def __init__(self, lang_code: str, base_dir: Path, *fallback_dirs: Path | None):
        self.lang_code = lang_code
        self.base_dir = base_dir
        # frozen ビルドでは同梱 .ini が _internal/lang に入り、base_dir（exe隣）とは
        # 別になる。通常は起動時に exe 隣へ配置されるが、書き込めなかった場合に備えて
        # リソース側も探索する。
        self.search_dirs: list[Path] = [base_dir]
        for extra in fallback_dirs:
            if extra is not None and extra not in self.search_dirs:
                self.search_dirs.append(extra)
        self.translations = self._load_translations()

    def _load_file_layered(self, config: configparser.ConfigParser, file_name: str) -> None:
        """`file_name` を search_dirs の優先度が低い順（末尾から）に読み込む。

        configparser.read() は後で読んだファイルのキーが先に読んだものを上書きするので、
        末尾（同梱リソース側）から読んで先頭（exe隣の書き込み可能ディレクトリ）を最後に
        読めば、exe隣のファイルが優先されつつ、そこに無いキーだけ同梱版の値へ「ファイル
        単位」ではなく「キー単位」でフォールバックする。

        exe隣の lang/*.ini は一度作られたら二度と上書きされない（models/ と同じ「既存
        ファイルは触らない」方針、ユーザーの手編集を保護するため）。ファイル単位で最初に
        見つかった1つだけを読む実装だと、アップデートで追加された翻訳キーが exe隣の古い
        ファイルには無いまま埋まらず raw key 表示になっていた（PR#16 レビュー指摘）。
        """
        for directory in reversed(self.search_dirs):
            path = directory / file_name
            if not path.is_file():
                continue
            try:
                config.read(path, encoding="utf-8")
            except Exception as e:
                write_debug_log(f"Failed to read language file {path}: {e}")

    def _load_translations(self) -> configparser.ConfigParser:
        config = configparser.ConfigParser()

        # Load English first as a per-key fallback, then overlay the selected language so
        # any key a translation file is missing (e.g. a whole new [CaptionCore] section)
        # still resolves to English instead of showing the raw key.
        self._load_file_layered(config, "en.ini")
        if self.lang_code != "en":
            self._load_file_layered(config, f"{self.lang_code}.ini")
        return config

    def get_string(self, section: str, key: str, **kwargs: Any) -> str:
        try:
            raw_string = self.translations.get(section, key, fallback=key)
            try:
                return raw_string.format(**kwargs)
            except (KeyError, ValueError) as e:
                # Log the formatting error but return the raw string to avoid crashing
                write_debug_log(f"LocaleManager format error for key '{key}' in section '{section}': {e}. Kwargs: {kwargs}")
                return raw_string
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Fallback to key if not found
            return key.replace("_", " ").capitalize()
