# 二次改修 Phase 3 — Undo / 履歴の意味論（REQ-3.1 の文書化）

対象ベースライン: `dev` `95326d7`（2026-11-29 実装）。本稿は REQ-3.1 が要求する
「実装前に決める意味論」の確定版であり、Phase 3 の全アクションはこれに従う。

## 1. 戻り値 — 3 値

複合アクション（`CompositeUndoAction`）と一括アクション（`BulkAddTagsAction`,
`BulkRemoveTagsAction`）の `undo()` / `redo()` は `UndoResult` を返す。

| 値 | 意味 |
|---|---|
| `UndoResult.FULL` | 対象すべてに成功した |
| `UndoResult.PARTIAL` | 一部だけ成功した。失敗対象は `action.failed_paths` に入る |
| `UndoResult.FAILED` | 1 つも成功しなかった |

単一アクション（`AddTagsAction` など）は既存 API 互換のため `bool` を返し続ける。
`UndoManager` は `as_undo_result()` で `True → FULL` / `False → FAILED` に正規化する。
`any(...)` / `success_count > 0` を成功判定に使わない（NFR-10）。

## 2. スタック遷移

`UndoManager.undo()`:

| 結果 | スタック | 戻り値 | 利用者への表示 |
|---|---|---|---|
| FULL | アクションを redo スタックへ移す | `True` | 従来どおり「元に戻しました」 |
| PARTIAL | アクションを redo スタックへ移す | `True` | 成功メッセージに続けて「{count} 件戻せませんでした」（オレンジ）＋失敗パスを debug log へ |
| FAILED（一時的） | アクションを undo スタックの先頭へ戻す | `False` | 従来どおり「失敗しました」 |
| FAILED（恒久） | アクションを履歴から外す | `False` | 「失敗しました」（再試行しても変わらないため積み直さない） |

`redo()` も対称（逆向きのスタック）。

**恒久失敗の判定**（U-2 の決定）: 対象 `.txt` も親フォルダも存在しない、または
`PermissionError`。それ以外（一時的な I/O 失敗など）は履歴に残して再試行できる。
恒久失敗だけを履歴から外すことで、R11-11 の「先頭に戻り続けて以降の Undo が
全部止まる」デッドロックを解消する。

**部分成功を履歴へ積む理由**: 戻せた分を活かし、取り残された対象は件数とパスで明示する
（NFR-10）。全失敗扱いにすると 1 ファイルの失敗で全体が永久に進まなくなる。

## 3. アクション別の記録内容（REQ-3.3）

| アクション | 記録するもの | undo | redo |
|---|---|---|---|
| `AddTagsAction` | 追加したタグ（単一ファイル） | そのタグだけを除く。**中身が追加タグだけ**ならファイルごと消す（R2-9） | 再追加（消えていれば作り直す） |
| `RemoveTagAction` | 削除したタグと元の位置 | 元の位置へ戻す | 再度除く |
| `EditCaptionAction` | 前後の全文と「作成前は無かったか」 | 作成したファイルは、**現在の内容が自分の書いた new_text と一致するときだけ**消す（R13-6） | new_text を書く |
| `_FileSnapshotAction` 系 | 前後の全文 | 同上（`previous_content is None` のときの unlink は一致時だけ） | new_content を書く |
| `BulkAddTagsAction` | **ファイルごとの「実際に追加されたタグ」**と位置 | そのタグだけを除く（元からあったタグは消さない。R22-4） | 同じ側へ再追加（順序は利用者の指定順。R1-4） |
| `BulkRemoveTagsAction` | **ファイルごとの全出現位置** | 全出現位置へ戻す（R5-5） | 全出現を除く |

どれも「既にその状態なら書かない」ことで冪等。undo→redo→undo で元のバイト列に戻る。

## 4. パス解決（REQ-3.6）

- 位置指定系（`AddTagsAction` / `RemoveTagAction` / `BulkAddTagsAction` /
  `BulkRemoveTagsAction`）は**生成時点の `os.path.abspath()` に固定**し、読み書きは
  `text_io.read_text()` / `text_io.atomic_write_text()`（長パス変換）を通す。
  正本は `_FileSnapshotAction` と同じ方式。
- `EditCaptionAction` は abspath 固定のみで、長パス変換を通さない（意図された例外。
  undo_manager 内コメントの REQ-2.9／U-3 は旧仕様書側の番号）。
- これにより、CWD を変えても undo/redo は登録時と同じファイルを指す（R9-8）。

## 5. フォルダ切替（REQ-3.5 / R20-2）

`_handle_folder_drop()`（D&D／Browse／パス入力の 3 経路が集約）で
`undo_manager.clear()` を呼び、履歴と redo スタックを破棄する。
`_batch_failed_paths` の破棄と同じ方針であり、切替経路ごとに差を作らない。

## 6. 保持量の上限（REQ-3.8 / R20-4）

`UndoManager.max_history` は従来どおり 50（履歴エントリ数）。
一括アクションは 1 エントリで数千ファイルを抱えうるため、**エントリ数が
`constants.UNDO_BATCH_SNAPSHOT_LIMIT`（500、バッチ側と同じ水準）を超えた場合は
その一括操作の Undo を作らない**。超過時の扱いは既存のバッチ側と同じで、
「今回の操作だけ諦め、既存履歴は保持する」。利用者へは
`MainWindow.Undo_Batch_Too_Large` で件数を出す。

## 7. GUI への反映（REQ-3.7 / R2-10）

`_refresh_ui_after_undo_redo()` は `_build_tag_cache()` の後で
`grid_view_widget.update_tag_cache(self._tag_cache)` を呼ぶ。`_build_tag_cache()` は
新しい dict を作るため、渡し直さないとグリッドの検索フィルタが古いキャッシュを
参照し続ける。

## 8. テスト

`tests/test_phase3_undo_integrity.py` が上記の各項目を固定する
（対称性、部分失敗、恒久失敗の履歴外し、R13-6 の unlink 条件、R2-9、R9-8、
R20-2、R2-10、R20-4、複合アクションの 3 値集計）。
