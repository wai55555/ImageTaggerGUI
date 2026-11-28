# テスト基盤のメモ（二次改修 Phase 1）

このディレクトリのテストは、**実環境を読まない・書かない**ことを前提にしている。
隔離の実体は `tests/conftest.py` にあり、テストモジュール側で `CONFIG_PATH` 等を
import 時に書き換えない（書き換えると pytest セッション全体へ漏れて順序依存になる）。

## 隔離されるもの

| 対象 | 隔離先 | 仕組み |
|---|---|---|
| `config.ini`（app_settings / constants / utils / tagging_core） | テストごとの `tmp_path/config.ini` | autouse fixture `isolated_app_config`（`monkeypatch`） |
| 既定 `input_dir`（`app_settings.BASE_DIR`） | `tmp_path` | 同上 |
| `utils.LOG_FILE_PATH` / `ERROR_LOG_PATH` / `LOG_ROTATED_PATH`、`tagging_core.LOG_FILE_PATH` | セッション共通の temp | conftest の import 時 + セッション終了時に復元 |
| `vlm_config.VLM_CONNECTIONS_PATH` / `VLM_PROFILES_PATH` | セッション共通の temp | 同上 |

実データ（`config.ini`、`.env`、`vlm_*.json`、ログ、`inputs/`、`models/`）は
テストの前後で指紋（サイズ・mtime・SHA-256）を比較し、変化したらそのテストを失敗させる。
`inputs/` と `models/` は数 GB になるため直下エントリだけを見る。

## 順序依存の確認（並列・シャッフル）

```bash
# 既定では何もしない。指定したときだけ収集順を決定的に並べ替える。
PYTEST_SHUFFLE_SEED=1 .venv/Scripts/python.exe -m pytest -q
```

プラグイン（pytest-randomly / pytest-xdist）は入れていない。並列実行の確認は
テストファイルを 2 群に分けて 2 プロセスで同時に走らせ、合計件数が全体実行と
一致することで行う。

## モックに関する原則（REQ-1.8）

- **モックで置き換えた関数の内側はテストされていない。** 偽の戻り値が素通りする
  ことだけを assert しても、本物のパース処理が壊れたことは検出できない。
- 置き換えるなら「呼び出し引数を記録しつつ本物へ委譲する」形にして、実ロジックまで
  通すこと（例: `tests/test_model_translation_map.py`）。
- 設定値のパース（`order_list` / `verified_set` / `model_id_override_map` 等）を
  テスト側で再実装しない。本物の `app_settings.Vlm` を使う。

## i18n の検査（REQ-1.6）

`tests/test_i18n_code_to_ini.py` が `src/*.py` の `get_string(...)`（文字列リテラル
引数のみ）を AST で集め、`lang/*.ini` 全言語にキーが存在することを検査する。
現時点で全言語に無いキーは同ファイルの `_KNOWN_MISSING_SECTIONS` /
`_KNOWN_MISSING_KEYS` に理由付きで記録してあり、**Phase 8C（REQ-8C.1）が 9 言語へ
登録したら台帳を消すまで失敗する**。動的なキー（f-string・変数）は静的に解決できない
ため対象外で、`tests/test_localization_coverage.py` が実行時の解決を確認する。
