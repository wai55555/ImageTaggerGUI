"""lang/*.ini の整合性テスト（docs/260920_post_release_hardening_plan.md 3.）。

locale_manager は en.ini を読んでから対象言語を重ねるので、キーや
{placeholder} が欠けていても実行時には英語が静かに出るだけで気づけない。
このテストは en.ini を基準に他 8 言語のセクション/キー/プレースホルダの
整合と、値の非空・重複キー無しを固定する。

Offline only - no network, no GUI. Run:  rtk pytest tests/test_lang_parity.py -q
"""
import configparser
from pathlib import Path
from string import Formatter

import pytest

LANG_DIR = Path(__file__).resolve().parent.parent / "lang"

# 比較対象の言語は lang/*.ini の走査から得る（R11-7。テスト側に言語一覧を
# ハードコードしない——言語を追加したときに自動で対象へ入る）。
# en.ini は「基準」という役割があるので、比較対象からは外して別に扱う。
ALL_LANGS = sorted(path.stem for path in LANG_DIR.glob("*.ini"))
BASE_LANG = "en"
OTHER_LANGS = [lang for lang in ALL_LANGS if lang != BASE_LANG]

# 意図的に許容する差異（現時点では想定なし）。
ALLOWED_EXTRA: set[tuple[str, str, str]] = set()

_FORMATTER = Formatter()


def _placeholders(value: str) -> set[str]:
    """value 内の {name} 形式プレースホルダ集合を抽出する（{{ エスケープ対応）。"""
    names: set[str] = set()
    for _literal_text, field_name, _format_spec, _conversion in _FORMATTER.parse(value):
        if field_name:
            names.add(field_name)
    return names


def _load_ini(lang: str, problems: list[str]) -> configparser.ConfigParser | None:
    """1言語分の ini を読む。重複キー等のパースエラーはそのファイルの失敗として記録する。"""
    path = LANG_DIR / f"{lang}.ini"
    cfg = configparser.ConfigParser(interpolation=None)
    try:
        with open(path, encoding="utf-8") as f:
            cfg.read_file(f, source=str(path))
    except configparser.Error as exc:
        problems.append(f"{lang} / (parse error): {exc}")
        return None
    return cfg


def test_lang_files_are_consistent_with_en():
    problems: list[str] = []

    # 走査結果の健全性（空振りしていないこと）。
    assert len(ALL_LANGS) >= 2, f"lang/*.ini が見つからない: {LANG_DIR}"
    assert BASE_LANG in ALL_LANGS, "基準の en.ini が無い"
    assert len(OTHER_LANGS) == len(ALL_LANGS) - 1

    parsed: dict[str, configparser.ConfigParser] = {}
    for lang in ALL_LANGS:
        cfg = _load_ini(lang, problems)
        if cfg is not None:
            parsed[lang] = cfg

    assert BASE_LANG in parsed, "en.ini must parse cleanly to act as the baseline"
    en_cfg = parsed[BASE_LANG]
    en_sections = set(en_cfg.sections())

    # en.ini 自体の非空チェックも行う（基準側が壊れていたら比較が無意味になる）。
    for lang in OTHER_LANGS:
        if lang not in parsed:
            continue  # already recorded as a parse-error problem
        cfg = parsed[lang]
        lang_sections = set(cfg.sections())

        missing_sections = en_sections - lang_sections
        extra_sections = lang_sections - en_sections
        for section in sorted(missing_sections):
            if (lang, section, "") not in ALLOWED_EXTRA:
                problems.append(f"{lang} / {section} / (section missing)")
        for section in sorted(extra_sections):
            if (lang, section, "") not in ALLOWED_EXTRA:
                problems.append(f"{lang} / {section} / (unexpected extra section)")

        for section in sorted(en_sections & lang_sections):
            en_keys = set(en_cfg.options(section))
            lang_keys = set(cfg.options(section))

            missing_keys = en_keys - lang_keys
            extra_keys = lang_keys - en_keys
            for key in sorted(missing_keys):
                if (lang, section, key) not in ALLOWED_EXTRA:
                    problems.append(f"{lang} / {section} / {key}: missing key")
            for key in sorted(extra_keys):
                if (lang, section, key) not in ALLOWED_EXTRA:
                    problems.append(f"{lang} / {section} / {key}: unexpected extra key")

            for key in sorted(en_keys & lang_keys):
                en_value = en_cfg.get(section, key)
                lang_value = cfg.get(section, key)

                if lang_value == "":
                    problems.append(f"{lang} / {section} / {key}: empty value")

                en_placeholders = _placeholders(en_value)
                lang_placeholders = _placeholders(lang_value)
                if en_placeholders != lang_placeholders:
                    missing_ph = en_placeholders - lang_placeholders
                    extra_ph = lang_placeholders - en_placeholders
                    detail = []
                    if missing_ph:
                        detail.append(f"missing placeholders {sorted(missing_ph)}")
                    if extra_ph:
                        detail.append(f"extra placeholders {sorted(extra_ph)}")
                    problems.append(
                        f"{lang} / {section} / {key}: placeholder mismatch ({', '.join(detail)})"
                    )

    # en.ini 自身の非空チェック（欠落値のまま残っているケースを拾う）。
    for section in en_sections:
        for key in en_cfg.options(section):
            if en_cfg.get(section, key) == "":
                problems.append(f"{BASE_LANG} / {section} / {key}: empty value")

    if problems:
        pytest.fail("\n".join(problems))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
