"""バッチ処理ループの骨格（REQ-3.1 / REQ-3.3 / REQ-3.7）。

`tagging_core.process_image_loop` と `caption_core.process_caption_loop` に重複していた
「反復・進捗間引き・停止判定と残件の failed 寄せ・既存ファイルの事前判定・失敗収集・
サマリ集計・FileChange 収集」をここへ集約する。生成と結合は各ループのコールバックに残す。

骨格へ寄せない（REQ-3.7。観測可能な挙動が変わるため現行のまま各ループ側に残す）:

- 進捗と停止の順序 … `progress_before_stop` / `progress_after_item` で選ぶ
  （tagging/caption は「処理後」＝ `progress_after_item=True`。REQ-C4 で現行の
  「処理前」から意図的に変えた。VLM は自前のループを持つ）
- 出力パスの作り方 … `output_path_for` を各ループが渡す
- 相対パス計算（`relative_to(INPUT_DIR)`）… `on_item_start` で各ループが計算する
- ASK で選ばれた決定の解釈（追記かどうか、placement への写像）… `process_item` 側

レビューループ Phase C の是正（§9）:

- REQ-C3: 停止・中断で抜けるとき、未処理ぶんを failed へ寄せつつ件数の会計を
  閉じる（`written+skipped+unchanged+errors == total`）。SKIP モードで既存 .txt が
  ある未処理分は failed ではなく skipped に数える。
- REQ-C4: `progress_after_item=True` で、各アイテムの処理が終わってから進捗を発行する。
- REQ-C5: 同一の .txt を共有する画像（a.png と a.jpg）を開始時に検出し、
  2 件目以降を理由付きでスキップする。

Qt 非依存。シグナル発行は呼び出し側（Worker）の責務のまま。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import constants
from batch_types import ExistingFileMode, OverwriteDecision

# 1 件処理の結果種別。
# - written:   書き込み成功（change に FileChange 相当を入れる）
# - unchanged: 生成結果が既存内容と同じで書かなかった
# - error:     生成・読み込み・書き込みの失敗（failed へ積む）
# - abort:     ループを抜ける（Florence の cancelled 専用。errors は増やさない）
ItemKind = str
WRITTEN = "written"
UNCHANGED = "unchanged"
ERROR = "error"
ABORT = "abort"


@dataclass
class ItemOutcome:
    """`process_item` の戻り値。骨格は kind だけで分岐する。"""
    kind: ItemKind
    change: Any = None

    @classmethod
    def written(cls, change: Any) -> "ItemOutcome":
        return cls(WRITTEN, change)

    @classmethod
    def unchanged(cls) -> "ItemOutcome":
        return cls(UNCHANGED)

    @classmethod
    def error(cls) -> "ItemOutcome":
        return cls(ERROR)

    @classmethod
    def abort(cls) -> "ItemOutcome":
        return cls(ABORT)


@dataclass
class ItemContext:
    """1 件ぶんの文脈。`on_item_start` で埋めた値を `process_item` が使える。"""
    index: int
    total: int
    image_path: Path
    output_path: Path
    mode: ExistingFileMode
    existed: bool = False
    decision: OverwriteDecision | None = None
    relative_path: Any = None


@dataclass
class BatchResult:
    """ループ全体の集計（サマリ表示と戻り値に使う）。

    `errors` は「処理して失敗した件数」に加え、停止・中断で抜けたときの未処理件数も
    含む（REQ-C3）。これにより `written + skipped + unchanged + errors == total` が
    常に成り立ち、要約だけを見て不足分が分かる。
    """
    total: int
    changes: list[Any] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)
    written: int = 0
    skipped: int = 0
    unchanged: int = 0
    errors: int = 0
    stopped_at: int | None = None

    @property
    def appended(self) -> int:
        """追記として書かれた件数（tagging/caption のサマリ用）。"""
        return sum(1 for change in self.changes if getattr(change, "was_append", False))


def duplicate_output_indices(
    image_paths: list[Path],
    output_path_for: Callable[[Path], Path],
) -> dict[int, int]:
    """同じ .txt を共有する画像のうち、2 件目以降の index → 先頭の index を返す（REQ-C5）。

    `a.png` と `a.jpg` は同じ `a.txt` を指す。先頭（列挙順で最初）だけを処理し、
    残りは黙って「処理済み」にせず理由付きでスキップするために使う。

    キーは出力 .txt のパス。大文字小文字は区別しない（対象環境の Windows では
    同じファイルを指すため。ホストが POSIX でも同じ判定に揃える）。
    """
    first_seen: dict[str, int] = {}
    duplicates: dict[int, int] = {}
    for index, image_path in enumerate(image_paths):
        key = str(output_path_for(image_path)).casefold()
        first = first_seen.setdefault(key, index)
        if first != index:
            duplicates[index] = first
    return duplicates


def resolve_existing(
    output_path: Path,
    mode: ExistingFileMode,
    decision_resolver: Callable[[Path], OverwriteDecision] | None,
    stop_checker: Callable[[], bool] | None,
) -> tuple[bool, OverwriteDecision | None, bool]:
    """既存 .txt の事前判定（REQ-3.1 の「既存ファイルの事前判定」）。

    Returns:
        (will_write, decision, stop_requested)。`stop_requested` が True のときは
        呼び出し側が残件を failed へ寄せてループを抜ける（ASK のダイアログ前に停止が
        入った場合。resolver は呼ばない）。
    """
    if not output_path.is_file():
        return True, None, False
    if mode is ExistingFileMode.SKIP:
        return False, OverwriteDecision.SKIP, False
    if mode is ExistingFileMode.OVERWRITE:
        return True, OverwriteDecision.OVERWRITE, False
    if mode is ExistingFileMode.APPEND:
        return True, OverwriteDecision.APPEND, False
    # ASK
    if decision_resolver is None:
        # 防御: resolver 未設定なら既存ファイルには触らない（現行どおりスキップ扱い）
        return False, OverwriteDecision.SKIP, False
    if stop_checker and stop_checker():
        return False, None, True
    decision = decision_resolver(output_path)
    if decision is OverwriteDecision.SKIP:
        return False, decision, False
    return True, decision, False


def run_batch(
    image_paths: list[Path],
    *,
    mode: ExistingFileMode,
    output_path_for: Callable[[Path], Path],
    process_item: Callable[[ItemContext], ItemOutcome],
    decision_resolver: Callable[[Path], OverwriteDecision] | None = None,
    stop_checker: Callable[[], bool] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    failed_paths: list[Path] | None = None,
    on_item_start: Callable[[ItemContext], None] | None = None,
    on_existing_skip: Callable[[ItemContext], None] | None = None,
    on_duplicate_skip: Callable[[ItemContext], None] | None = None,
    on_stop: Callable[[], None] | None = None,
    progress_before_stop: bool = True,
    progress_after_item: bool = False,
    dedupe_failed: bool = True,
    changes_out: list[Any] | None = None,
) -> BatchResult:
    """バッチループを回す（生成と結合は `process_item` の責務）。

    Args:
        image_paths: 処理対象（列挙済み）。
        mode: 既存 .txt の扱い。
        output_path_for: 画像パス → 出力 .txt パス（tagging/caption は splitext + ".txt"、
            VLM は with_suffix）。
        process_item: 1 件処理。生成・結合・書き込みを行い `ItemOutcome` を返す。
        decision_resolver: ASK モードの判断要求（GUI 往復しうる）。
        stop_checker: 停止要求の確認。
        progress_cb: 進捗通知（全体で最大 PROGRESS_SIGNAL_BUDGET 回＋最終 N/N）。
        failed_paths: 呼び出し側が持つ失敗リスト（あれば追記する）。
        on_item_start: 既存ファイル判定の**前**に呼ぶ。相対パス計算など、現行の順序を
            保つ必要がある処理を各ループがここで行う（`relative_to` の例外もここで出る）。
        on_existing_skip: 既存ファイルによりスキップしたときのログ。
        on_duplicate_skip: 同じ .txt を共有する 2 件目以降をスキップしたときのログ（REQ-C5）。
        on_stop: 停止でループを抜けるときのログ。
        progress_before_stop: True なら停止判定より前に進捗を発行する（旧 tagging/caption と
            VLM の現行順。`progress_after_item=True` のときは使わない）。
        progress_after_item: True なら各アイテムの処理が終わってから進捗を発行する
            （tagging/caption。REQ-C4 の意図的変更）。
        dedupe_failed: failed の重複を防ぐか（tagging/caption は True、VLM は False）。
        changes_out: 書き込んだ FileChange を逐次追加するリスト。致命的例外で
            ループが中断しても、呼び出し側が処理済み分の Undo 情報を失わないため
            （REQ-C1）。
    """
    result = BatchResult(total=len(image_paths))
    step = constants.progress_step_for(result.total)
    seen: set[Path] = set(failed_paths or ())
    # REQ-C5: 同一 .txt を共有する 2 件目以降は開始時点でスキップ確定として数える。
    # 停止で途中終了しても会計（REQ-C3）が閉じるよう、ループ到達前に計上する。
    duplicates = duplicate_output_indices(image_paths, output_path_for)
    result.skipped += len(duplicates)

    def mark_failed(path: Path) -> None:
        if dedupe_failed and path in seen:
            return
        seen.add(path)
        result.failed.append(path)
        if failed_paths is not None:
            failed_paths.append(path)

    def mark_remaining_failed(start: int) -> None:
        """未処理ぶんを failed へ寄せ、件数の会計を閉じる（REQ-C3）。

        既にスキップ確定した重複と、SKIP モードで既存 .txt がある未処理分は失敗に
        数えない（後者は skipped に数える）。残りは failed へ寄せ、errors にも数える
        ことで `written+skipped+unchanged+errors == total` を保つ。
        """
        for index in range(start, result.total):
            pending = image_paths[index]
            if index in duplicates:
                continue
            if mode is ExistingFileMode.SKIP and output_path_for(pending).is_file():
                result.skipped += 1
                continue
            mark_failed(pending)
            result.errors += 1

    def progress_due(index: int) -> bool:
        return bool(progress_cb) and (
            (index + 1) % step == 0 or index == result.total - 1
        )

    def emit_progress(index: int) -> None:
        if progress_cb and progress_due(index):
            progress_cb(index + 1, result.total)

    def handle_item(index: int, image_path: Path) -> str:
        """1 件処理する。'next' は継続、'break' はループを抜ける（未完了）。"""
        context = ItemContext(
            index=index,
            total=result.total,
            image_path=image_path,
            output_path=output_path_for(image_path),
            mode=mode,
        )
        if index in duplicates:
            # REQ-C5: a.png と a.jpg が同じ a.txt を共有する。先に処理した側の内容へ
            # 黙って混ぜたり、理由なく恒久的に処理済み扱いにしたりしない。
            if on_duplicate_skip:
                on_duplicate_skip(context)
            return "next"

        if on_item_start:
            # 相対パス計算（relative_to）など、現行どおり既存ファイル判定より前に走る。
            on_item_start(context)

        context.existed = context.output_path.is_file()
        will_write, decision, stop_requested = resolve_existing(
            context.output_path, mode, decision_resolver, stop_checker,
        )
        context.decision = decision
        if stop_requested:
            mark_remaining_failed(index)
            result.stopped_at = index
            return "break"
        if not will_write:
            result.skipped += 1
            if on_existing_skip:
                on_existing_skip(context)
            return "next"

        outcome = process_item(context)
        if outcome.kind == WRITTEN:
            result.written += 1
            result.changes.append(outcome.change)
            if changes_out is not None:
                changes_out.append(outcome.change)
        elif outcome.kind == UNCHANGED:
            result.unchanged += 1
        elif outcome.kind == ERROR:
            result.errors += 1
            mark_failed(image_path)
        elif outcome.kind == ABORT:
            # Florence の cancelled: errors は増やさず、残件を寄せて抜ける。
            mark_remaining_failed(index)
            result.stopped_at = index
            return "break"
        else:  # pragma: no cover - 呼び出し側の誤り
            raise ValueError(f"unknown item outcome: {outcome.kind!r}")
        return "next"

    for i, image_path in enumerate(image_paths):
        if not progress_after_item:
            if progress_before_stop:
                emit_progress(i)
        if stop_checker and stop_checker():
            if on_stop:
                on_stop()
            mark_remaining_failed(i)
            result.stopped_at = i
            break
        if not progress_after_item and not progress_before_stop:
            emit_progress(i)

        action = handle_item(i, image_path)
        if action == "break":
            break
        # REQ-C4: 進捗はアイテムの処理が終わってから発行する。停止判定で抜けた場合は
        # そのアイテムが未処理なので発行しない（結果として「常時1件先行」が消える）。
        if progress_after_item:
            emit_progress(i)

    return result
