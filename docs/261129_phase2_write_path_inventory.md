# 二次改修 Phase 2 — `.txt` 書き込み経路の棚卸し表（REQ-2.1）

対象ベースライン: `dev` HEAD `95326d7`（2026-11-28）。行番号はこの HEAD 時点。
本書は Phase 2 の REQ-2.1 が要求する棚卸し表であり、**以後に新しい書き込み経路を
足すときはこの表へ 1 行追加する**運用とする（表に無い経路は「ロック漏れの候補」）。

## 0. 判定の正本

| 項目 | 実体 | 位置 |
|---|---|---|
| 処理中判定（唯一の正本） | `MainWindow._is_processing()`。`_is_downloading` / `_is_bulk_deleting` / `_tagger_thread.isRunning()` / `_bulk_tag_thread.isRunning()` を束ねる。メモリ参照のみ（ファイル I/O なし。REQ-2.7） | `src/main_window.py:1261` |
| 錠（コントロールの無効化） | `MainWindow._set_main_controls_enabled(False)`。本体の全書き込み系・グリッド・Undo/Redo・一括系・単体タグ追加・タグチップ・GPU トグルを閉じる | `src/main_window.py:1289` |
| 錠の解除 | 処理の終了経路のみ（`_on_tagger_finished` / `_on_bulk_tag_finished` / `_on_download_finished` / `closeEvent`）。表示更新・ローダ完了は解錠しない（REQ-2.4） | 同 |
| グリッド側の錠 | `GridViewWidget.set_editing_enabled(False)` → 各セルの `ImageEditCellWidget.set_editing_enabled(False)`（セルは `_editing_enabled` を保持し、ハンドラ側でも止める） | `src/grid_view_widget.py:756` / `:463` |

## 1. 棚卸し表（.txt を読み-変更-書き込みする全経路）

「見る状態」は、その経路が開始可否を判断するために参照する状態。
「止め方（錠）」は UI 側、「止め方（ガード）」はハンドラ側（REQ-2.3 は両方を要求）。

| # | 経路 | 実体（ファイル:行） | 書くもの | 見る状態 | 止め方（錠） | 止め方（ガード） | テスト |
|---|---|---|---|---|---|---|---|
| 1 | タグ付けバッチ（モデル） | `MainWindow._start_tagging_thread` (`main_window.py:1779`) → `workers.TaggerThreadWorker` → `batch_runner.run_batch` | 各画像の `.txt`（OVERWRITE/APPEND/PREPEND） | `_is_processing()` | 実行中は開始不可（`run_button` は停止ボタンとして動作） | `_is_processing()` で即 return | `test_phase2_write_path_locks.py::test_is_processing_reads_the_existing_state_owners` |
| 2 | キャプション／VLM バッチ | 同上（`use_vlm` / モデル種別で worker が変わる） | 同上 | 同上 | 同上 | 同上 | 同上 |
| 3 | VLM 単体テスト（1 枚） | `MainWindow._run_vlm_single_test` (`main_window.py:1028`) | 選択 1 枚の `.txt` | `_is_processing()` | `vlm_single_test_button` 無効化（R12-9） | `_is_processing()` で即 return | `::test_vlm_single_test_is_blocked_while_a_bulk_run_is_active` |
| 4 | 一括タグ追加 | `MainWindow.add_tag_all` (`main_window.py:1739`) → `BulkTagWorker.run_bulk_add` | 対象全 `.txt` | `_is_processing()` | `add_tag_line` / `add_tag_button` 無効化 | `_is_processing()` で即 return | `::test_bulk_add_delete_and_undo_are_blocked_while_processing` |
| 5 | 一括タグ削除（タグボタン） | `MainWindow.delete_tag_all` (`main_window.py:1752`) → `BulkTagWorker.run_bulk_delete` | 対象全 `.txt` | `_is_processing()` | `tag_buttons` 無効化（`_set_bulk_controls_enabled`） | `_is_processing()` で即 return | 同上 |
| 6 | 単体タグ追加（入力欄＋ボタン） | `MainWindow._add_single_tag` (`main_window.py:1626`) | 表示中画像の `.txt` | `_is_processing()` | `add_single_tag_line` / `add_single_tag_button` 無効化（R25-1） | `_is_processing()` で即 return | `::test_single_tag_add_and_chip_delete_are_blocked_while_processing` |
| 7 | 単体タグ削除（表示中画像のタグチップ） | `MainWindow._delete_image_tag` (`main_window.py:1682`) | 表示中画像の `.txt` | `_is_processing()` | `tag_buttons_for_image` を無効化し、再描画時も錠状態で生成（R19-6） | `_is_processing()` で即 return | `::test_single_tag_add_and_chip_delete_are_blocked_while_processing` / `::test_tag_chips_are_created_locked_while_processing` |
| 8 | キャプション保存（メイン画面） | `MainWindow._save_current_caption` (`main_window.py:897`) | 表示中画像の `.txt` | `_controls_enabled`（編集欄の enabled） | `caption_text_edit` 無効化。錠をかける直前に flush する | 編集欄が無効なので新規編集は発生しない（錠前 flush は意図的に通す） | 既存 `tests/test_phase2_atomic_writes.py::test_main_window_caption_save_uses_atomic_writer` |
| 9 | グリッド タグ追加 | `ImageEditCellWidget._add_tag` (`grid_view_widget.py:342`) | セルの `.txt` | セル `_editing_enabled` | `_tag_area_widget` 無効化 | `_editing_enabled` で即 return | `::test_grid_cell_ignores_writes_while_locked` |
| 10 | グリッド タグ削除（チップ） | `ImageEditCellWidget._remove_tag` (`grid_view_widget.py:369`) | セルの `.txt` | セル `_editing_enabled` | 同上 | `_editing_enabled` で即 return | 同上 |
| 11 | グリッド キャプション保存 | `ImageEditCellWidget._save_caption` (`grid_view_widget.py:243`) | セルの `.txt` | セル `_editing_enabled` | `caption_edit.setReadOnly(True)` | `_editing_enabled` で即 return（錠前 flush は `set_editing_enabled` が先に実行） | `::test_grid_cell_still_flushes_an_edit_before_locking` |
| 12 | Undo（メイン画面・グリッド共通） | `MainWindow._perform_undo` (`main_window.py:2530`) | 履歴対象の `.txt` | `_is_processing()` | `undo_button`（両画面）無効化 | `_is_processing()` で即 return | `::test_bulk_add_delete_and_undo_are_blocked_while_processing` |
| 13 | Redo（同上） | `MainWindow._perform_redo` (`main_window.py:2543`) | 履歴対象の `.txt` | `_is_processing()` | `redo_button`（両画面）無効化 | `_is_processing()` で即 return | 同上（同じ経路群） |
| 14 | フォルダ切替（D&D／Browse／パス入力） | `MainWindow._handle_folder_drop` (`main_window.py:1494`)。3 経路がここへ集約 | `.txt` を直接は書かないが、**書き込み先の対象集合を差し替える**開始経路 | `_is_processing()` | `input_line` / `image_list` 無効化 | `_is_processing()` で即 return（U-1: 保留しない） | `::test_folder_switch_is_rejected_while_processing` |
| 15 | グリッド表示への切替 | `MainWindow._show_grid_view` | 直接は書かない（セルの flush 経由で書きうる） | `_controls_enabled` | `grid_view_button` 無効化 | セル側の `_editing_enabled` が受ける | `::test_bulk_start_locks_every_write_path_and_finish_unlocks` |
| 16 | モデル DL | `MainWindow._start_download_thread` (`main_window.py:2013`) | `.txt` は書かない（`config.ini` / `models/` を書く） | 実行ボタンの状態 | 一括処理中は `run_button` を閉じる（一括に停止手段が無いため） | `_start_tagging_thread` 側の `_is_processing()` が兼ねる | `::test_bulk_start_locks_every_write_path_and_finish_unlocks` |

## 2. 錠を解く経路（REQ-2.4）

| # | 経路 | 位置 | 何をするか |
|---|---|---|---|
| a | タグ付け／VLM バッチ終了 | `_on_tagger_finished` → `_update_ui_for_processing(False)` | 解錠してモデル状態を再評価 |
| b | 一括処理終了 | `_on_bulk_tag_finished` (`main_window.py:2155`) | **スレッドを畳んで一括フラグを落としてから**解錠し、`_check_model_status_and_update_ui()` で実行ボタンを戻す |
| c | DL 終了 | `_on_download_finished` | 解錠 |
| d | アプリ終了 | `closeEvent` | 解錠はしない（終了する） |

**解錠してはいけない経路**（以前は無条件に戻していた。R7-4 / R12-6）:

- `display_current_tag_page()`（タグページの再描画）… `self._controls_enabled` を引き継ぐ
- `_update_bulk_tag_buttons()`（TagLoader の遅延完了）… 一括フラグを落とさない
- `_display_image_tag_page()`（チップ再生成）… 現在の錠状態で生成する

## 3. 後始末の再入（REQ-2.6 / R35-1）

`_cleanup_tagger_thread` (`main_window.py:1921`) は待機対象をローカルに固定し、
`processEvents()` を挟む待機（`_wait_for_tagger_thread_stop`）の後で
`self._tagger_thread is thread` を確認する。差し替わっていれば detach せず戻る。
待機タイムアウト（5 秒）と `terminate()` を使わない方針は変えない（NFR-12）。

再現テスト: `tests/test_phase2_write_path_locks.py::test_cleanup_leaves_a_run_that_started_during_the_wait_alone`
（待機中に新しいスレッドへ差し替え、外側の後始末が触らないこと）、
`::test_cleanup_detaches_a_stuck_thread_without_terminating`（停止しないスレッドは
detach され、`terminate()` は呼ばれないこと）。

## 4. 意図的に錠の対象外にしたもの

| 対象 | 理由 |
|---|---|
| `run_button`（タグ付け/DL 中） | 「停止」ボタンとして使うため。一括処理中だけは `_start_bulk_tag_worker` が明示的に閉じる |
| TagLoader（`tag_thread`：タグ集計の読み取り） | 読み取り専用で `.txt` を書かない。`_is_processing()` に含めない（R12-6 の原因だった） |
| グリッドのページ送り／メイン画面への復帰 | 読み取りのみ。閉じると利用者がグリッドから出られなくなる |
| ビューア（拡大表示） | 読み取りのみ |
