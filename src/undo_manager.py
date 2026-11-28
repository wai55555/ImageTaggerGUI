"""
Undo/Redo Manager for Tag Operations
Manages the history of tag editing operations and provides undo/redo functionality.
"""

from __future__ import annotations
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from utils import write_debug_log
from text_io import atomic_write_text, long_path_str, read_text


def _desc(key: str, **kwargs: object) -> str:
    """Undo 説明文を lang/*.ini の [MainWindow] から引く。

    UndoAction は main_window / grid_view_widget / 各ワーカーなど多くの場所で
    生成され、その大半は LocaleManager を持っていない。そのため app_settings が
    公開している「後から差し込まれる」get_string（MainWindow が起動時に
    set_get_string_func() で配線する）を遅延インポートで使う
    ——utils._uncaught_error_strings() と同じやり方で、import 時の循環も避ける。

    description() が呼ばれるのは表示のタイミング（ツールチップ生成時）なので、
    履歴を積んだ後に言語を切り替えても次の再描画で追従する。ここで文字列を
    確定させてしまうと、積んだ時点の言語で固定されてしまう。

    配線前（QApplication も LocaleManager も無い単体テスト等）は
    app_settings._get_string が default_get_string_fallback のままでキーを
    そのまま返す。表示経路には乗らないので実害は無いが、description() が
    何も返さないより追跡しやすい。
    """
    try:
        from app_settings import _get_string
        return _get_string("MainWindow", key, **kwargs)
    except Exception:  # 配線前・import 失敗時でも description() は失敗させない
        return key


def _long_path_str(path: Path) -> str:
    """Windows の MAX_PATH（260文字）制限を避ける `\\\\?\\` プレフィックス付き絶対パス。

    tagging_core.process_image_loop / caption_core.process_caption_loop の書き込みと
    同じ狙い（長いパスでも書けるのに undo/redo だけ素の Path 経由で失敗する非対称を防ぐ）。

    - `os.path.abspath()` を使う（`Path.resolve()` ではない）: `\\\\?\\` パスは OS がそのまま
      使うので `..` の正規化は必要だが、**symlink は解決しない**。resolve() だと、新規
      作成した出力がその後 symlink に差し替えられていた場合、undo の unlink が
      リンク先の実体を消してしまう（PR#16 レビュー指摘）。
    - UNC パス（`\\\\server\\share\\...`）は `\\\\?\\UNC\\server\\share\\...` の形にする必要が
      あり、単純に `\\\\?\\` を前置すると不正なパスになる（PR#16 レビュー指摘）。
    - 既に `\\\\?\\` 付き（拡張長パス）が渡された場合はそのまま返す。二重に前置すると
      `\\\\?\\UNC\\?\\...` のような不正パスになる（PR#16 レビュー第2ラウンド指摘）。
    """
    # REQ-2.9: 実装は共通部品（src/text_io.py）へ一本化した。名前は既存テスト
    # （tests/test_pr16_review_fixes.py）が直接参照しているため薄い委譲として残す。
    return long_path_str(path)

class GetString(Protocol):
    """Protocol for localization function."""
    def __call__(self, section: str, key: str, **kwargs: str | int | float) -> str:
        ...


class UndoResult(Enum):
    """複合・一括アクションの undo/redo の結果（REQ-3.1 / NFR-10）。

    FULL:    対象全てに成功した。
    PARTIAL: 一部だけ成功した（失敗対象は `failed_paths`、件数は呼び出し側が明示する）。
    FAILED:  1 つも成功しなかった。
    """

    FULL = "full"
    PARTIAL = "partial"
    FAILED = "failed"


def as_undo_result(value: bool | UndoResult) -> UndoResult:
    """単一アクションの bool 戻り値を 3 値へ正規化する（既存 API を壊さない）。"""
    if isinstance(value, UndoResult):
        return value
    return UndoResult.FULL if value else UndoResult.FAILED


def _permanently_missing(path: Path) -> bool:
    """対象 .txt も親フォルダも無い = 再試行しても変わらない失敗か（U-2）。"""
    try:
        return not path.exists() and not path.parent.is_dir()
    except OSError:
        return False


@dataclass
class _Outcome:
    """一括・複合アクションの成否集計（3 値 + 失敗対象。REQ-3.1）。"""

    failures: list[Path] = field(default_factory=list)
    transient: bool = False
    succeeded: int = 0

    def ok(self) -> None:
        self.succeeded += 1

    def fail(self, path: Path | None, exc: BaseException | None = None) -> None:
        if path is not None:
            self.failures.append(path)
        permanent = isinstance(exc, PermissionError) or (
            path is not None and _permanently_missing(path))
        if not permanent:
            # 一時的な失敗が 1 つでもあれば、履歴からは外さない（再試行の余地を残す）。
            self.transient = True

    def result(self) -> UndoResult:
        if not self.failures:
            return UndoResult.FULL
        return UndoResult.PARTIAL if self.succeeded else UndoResult.FAILED

    @property
    def permanent(self) -> bool:
        return bool(self.failures) and not self.transient


class UndoAction(ABC):
    """Abstract base class for all undoable actions."""

    # REQ-3.1: 直近の undo/redo で失敗した対象（複合・一括アクションが埋める）。
    failed_paths: list[Path]
    # 再試行しても変わらない失敗だけだったか（U-2: 履歴から外す判断に使う）。
    permanent_failure: bool

    def _succeed(self) -> bool:
        self.failed_paths = []
        self.permanent_failure = False
        return True

    def _fail(self, path: Path | None, exc: BaseException | None = None) -> bool:
        self.failed_paths = [path] if path is not None else []
        self.permanent_failure = (
            isinstance(exc, PermissionError)
            or (path is not None and _permanently_missing(path))
        )
        return False

    def _apply(self, outcome: _Outcome) -> UndoResult:
        self.failed_paths = list(outcome.failures)
        self.permanent_failure = outcome.permanent
        result = outcome.result()
        if outcome.failures:
            write_debug_log(
                f"{type(self).__name__}: {result.value} "
                f"({len(outcome.failures)} failed: "
                f"{', '.join(p.name for p in outcome.failures[:5])})")
        return result

    @abstractmethod
    def undo(self) -> bool:
        """
        Undo this action.
        Returns True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def redo(self) -> bool:
        """
        Redo this action.
        Returns True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def description(self) -> str:
        """
        Returns a human-readable description of this action.
        """
        pass


@dataclass
class AddTagsAction(UndoAction):
    """Action for adding tags to a single image."""
    file_path: Path
    added_tags: list[str]

    def __post_init__(self) -> None:
        # REQ-3.6: 登録時点の絶対パスへ固定する（CWD 変更や相対入力フォルダで undo が
        # 別ファイルを叩かないように）。正本は _FileSnapshotAction と同じ。
        self.file_path = Path(os.path.abspath(self.file_path))

    def undo(self) -> bool:
        """Remove the added tags from the file."""
        path = self.file_path
        try:
            if not path.exists():
                write_debug_log(f"Undo failed: File not found: {path}")
                return self._fail(path)

            # Read current tags（REQ-B2: 長パス変換を通す）
            content = read_text(path, encoding='utf-8').strip()
            if not content:
                return self._succeed()

            tags = [tag.strip() for tag in content.split(',')]

            # Remove added tags
            remaining = list(tags)
            for tag in self.added_tags:
                if tag in remaining:
                    remaining.remove(tag)

            if not remaining and all(tag in self.added_tags for tag in tags):
                # REQ-3.7 / R2-9: この操作が作り出した（= 中身が追加タグだけの）ファイルは、
                # 空ファイルを残さず「無かった状態」へ戻す。
                os.unlink(_long_path_str(path))
                write_debug_log(f"Undo AddTags: removed the file the action created: {path.name}")
                return self._succeed()

            # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
            atomic_write_text(path, ', '.join(remaining), fsync=False, verify=False)

            write_debug_log(f"Undo AddTags: Removed {len(self.added_tags)} tags from {path.name}")
            return self._succeed()

        except Exception as e:
            write_debug_log(f"Undo AddTags failed: {e}")
            return self._fail(path, e)

    def redo(self) -> bool:
        """Re-add the tags to the file (recreates it when undo removed it)."""
        path = self.file_path
        try:
            tags: list[str] = []
            if path.exists():
                content = read_text(path, encoding='utf-8').strip()
                tags = [tag.strip() for tag in content.split(',')] if content else []

            # Add tags (avoid duplicates)
            for tag in self.added_tags:
                if tag not in tags:
                    tags.append(tag)

            # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
            atomic_write_text(path, ', '.join(tags), fsync=False, verify=False)

            write_debug_log(f"Redo AddTags: Added {len(self.added_tags)} tags to {path.name}")
            return self._succeed()

        except Exception as e:
            write_debug_log(f"Redo AddTags failed: {e}")
            return self._fail(path, e)

    def description(self) -> str:
        """Return a localized description of this action."""
        if len(self.added_tags) == 1:
            return _desc("Undo_Desc_Add_Tag", tag=self.added_tags[0])
        if len(self.added_tags) <= 3:
            return _desc("Undo_Desc_Add_Tags", tags=", ".join(self.added_tags))
        return _desc("Undo_Desc_Add_Tags_Many",
                     tags=", ".join(self.added_tags[:3]), count=len(self.added_tags))


@dataclass
class RemoveTagAction(UndoAction):
    """Action for removing a tag from a single image."""
    file_path: Path
    removed_tag: str
    original_index: int

    def __post_init__(self) -> None:
        # REQ-3.6: 登録時点の絶対パスへ固定する（CWD 変更に影響されない）。
        self.file_path = Path(os.path.abspath(self.file_path))

    def undo(self) -> bool:
        """Re-insert the removed tag at its original position."""
        path = self.file_path
        try:
            if not path.exists():
                write_debug_log(f"Undo failed: File not found: {path}")
                return self._fail(path)

            content = read_text(path, encoding='utf-8').strip()
            tags = [tag.strip() for tag in content.split(',')] if content else []

            # Insert tag at original position
            insert_pos = min(self.original_index, len(tags))
            tags.insert(insert_pos, self.removed_tag)

            # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
            atomic_write_text(path, ', '.join(tags), fsync=False, verify=False)

            write_debug_log(f"Undo RemoveTag: Re-inserted '{self.removed_tag}' at position {insert_pos} in {path.name}")
            return self._succeed()

        except Exception as e:
            write_debug_log(f"Undo RemoveTag failed: {e}")
            return self._fail(path, e)

    def redo(self) -> bool:
        """Remove the tag again."""
        path = self.file_path
        try:
            if not path.exists():
                write_debug_log(f"Redo failed: File not found: {path}")
                return self._fail(path)

            content = read_text(path, encoding='utf-8').strip()
            if not content:
                return self._succeed()

            tags = [tag.strip() for tag in content.split(',')]

            if self.removed_tag in tags:
                tags.remove(self.removed_tag)

            # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
            atomic_write_text(path, ', '.join(tags), fsync=False, verify=False)

            write_debug_log(f"Redo RemoveTag: Removed '{self.removed_tag}' from {path.name}")
            return self._succeed()

        except Exception as e:
            write_debug_log(f"Redo RemoveTag failed: {e}")
            return self._fail(path, e)

    def description(self) -> str:
        """Return a localized description of this action."""
        return _desc("Undo_Desc_Remove_Tag", tag=self.removed_tag)


@dataclass
class EditCaptionAction(UndoAction):
    """Action for editing an image's free-text caption (captioner models).

    Unlike the tag actions this stores the whole before/after text - captions are
    short and not comma-structured, so a positional diff would not help.
    """
    file_path: Path
    old_text: str
    new_text: str
    file_existed_before: bool = True

    def __post_init__(self) -> None:
        # 作成時に絶対パスへ固定する（理由は _FileSnapshotAction.__post_init__ と同じ。
        # このクラスは _long_path_str を通さず素のパスで書き込み／unlink するので、相対の
        # まま持つと undo 時点の CWD 基準で別ファイルに当たりうる）。
        self.file_path = Path(os.path.abspath(self.file_path))

    def _write(self, text: str) -> bool:
        try:
            # REQ-2.3: 一時ファイル + os.replace で原子的に（(iii) 段）。
            # REQ-2.9 / U-3: このクラスだけは従来どおり拡張長パス変換を通さない
            # （経路のパス解決セマンティクスを変えないため use_long_path=False）。
            atomic_write_text(self.file_path, text, fsync=False, verify=False, use_long_path=False)
            return self._succeed()
        except Exception as e:
            write_debug_log(f"EditCaption write failed for {self.file_path}: {e}")
            return self._fail(self.file_path, e)

    def undo(self) -> bool:
        if not self.file_existed_before:
            # The edit created the file; undo must restore its absence, otherwise
            # caption generation treats the leftover .txt as done and skips the image.
            # REQ-3.4 / R13-6: ただし現在の内容が自分の書いた new_text と一致するときだけ
            # 消す。他経路が書き換えていたら消さず、部分失敗として扱う。
            try:
                current = (read_text(self.file_path, encoding='utf-8')
                           if self.file_path.exists() else None)
            except (OSError, UnicodeError) as e:
                write_debug_log(f"EditCaption undo read failed for {self.file_path}: {e}")
                return self._fail(self.file_path, e)
            if current is None:
                return self._succeed()
            if current != self.new_text:
                write_debug_log(
                    "EditCaption undo skipped unlink: the file was changed after the "
                    f"action ({self.file_path.name})")
                return self._fail(self.file_path)
            try:
                self.file_path.unlink(missing_ok=True)
                return self._succeed()
            except Exception as e:
                write_debug_log(f"EditCaption undo unlink failed for {self.file_path}: {e}")
                return self._fail(self.file_path, e)
        return self._write(self.old_text)

    def redo(self) -> bool:
        return self._write(self.new_text)

    def description(self) -> str:
        return _desc("Undo_Desc_Edit_Caption")


def _detect_newline(path: Path) -> str | None:
    """書き戻し先ファイルの現行の改行規約を検出する（REQ-B3）。

    スナップショットの previous_content / new_content は LF 正規化済みで規約を運ばない。
    そのまま `newline=None` で書くと、Windows では元が LF のファイルが CRLF へ変換
    されてしまう（`vlm_persistence.save_caption` が保持している規約と食い違う）。

    `vlm_persistence.save_caption` と同じ判定（生読みで `\r\n` を含むか）を使う。
    読めない・存在しない場合は `None`（現行どおり `open()` の既定）を返す。
    アクション生成時点のファイルは直前に保存されたものなので、元の規約と一致する。
    """
    try:
        with open(long_path_str(path), "r", encoding="utf-8", newline="") as handle:
            raw = handle.read()
    except (OSError, UnicodeDecodeError) as e:
        write_debug_log(f"undo newline detection failed for {path}: {type(e).__name__}: {e}")
        return None
    return "\r\n" if "\r\n" in raw else "\n"


@dataclass
class _FileSnapshotAction(UndoAction):
    """全文スナップショットによる undo/redo の共通実装（PR#16 レビュー指摘: 従来
    OverwriteFileAction と AppendTagsActionV2 が完全に同じ実装を重複させていた）。

    previous_content が None のときは「変更前にファイルが存在しなかった」ことを表し、
    undo はファイル削除まで行う。undo / redo とも全文スナップショットの復元で対称。
    `description()` はサブクラスで実装する（抽象のまま = 直接インスタンス化はしない）。
    """
    file_path: Path
    previous_content: str | None
    new_content: str
    # REQ-B3: 生成時点で対象ファイルから検出した改行規約。None は現行どおり
    # `newline=None`（プラットフォーム既定）で書く。
    newline: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # 作成時に絶対パスへ固定する。undo/redo はずっと後に走りうるので、
        # 相対パス（利用者が相対の入力フォルダを指定した場合など）のまま持つと
        # `_long_path_str()` の `os.path.abspath()` が「undo した時点の CWD」基準で
        # 解決してしまい、別ファイルを書き換え／削除しかねない（PR#16 レビュー: coderabbit）。
        # `resolve()` ではなく `os.path.abspath()`: symlink を解決しない点を _long_path_str と揃える。
        self.file_path = Path(os.path.abspath(self.file_path))
        self.newline = _detect_newline(self.file_path)

    def _write(self, text: str) -> bool:
        try:
            # REQ-2.3: 一時ファイル + os.replace で原子的に（(iii) 段）。
            # REQ-B3: 生成時に検出した改行規約で書く（LF のファイルが Undo/Redo で
            # CRLF へ化けるのを防ぐ）。検出できなければ現行どおり None。
            atomic_write_text(self.file_path, text, fsync=False, verify=False,
                              newline=self.newline)
            return self._succeed()
        except Exception as e:
            write_debug_log(f"{type(self).__name__} write failed for {self.file_path}: {e}")
            return self._fail(self.file_path, e)

    def undo(self) -> bool:
        if self.previous_content is None:
            # REQ-3.4 / R13-6: 現在の内容が自分の書いた new_content と一致するときだけ
            # 消す。他経路が書き換えていたら消さず、部分失敗として扱う。
            try:
                current = (read_text(self.file_path, encoding='utf-8')
                           if self.file_path.exists() else None)
            except (OSError, UnicodeError) as e:
                write_debug_log(f"{type(self).__name__} undo read failed for {self.file_path}: {e}")
                return self._fail(self.file_path, e)
            if current is None:
                return self._succeed()
            if current != self.new_content:
                write_debug_log(
                    f"{type(self).__name__} undo skipped unlink: the file was changed "
                    f"after the action ({self.file_path.name})")
                return self._fail(self.file_path)
            try:
                os.unlink(_long_path_str(self.file_path))
            except FileNotFoundError:
                pass
            except Exception as e:
                write_debug_log(f"{type(self).__name__} undo unlink failed for {self.file_path}: {e}")
                return self._fail(self.file_path, e)
            return self._succeed()
        return self._write(self.previous_content)

    def redo(self) -> bool:
        return self._write(self.new_content)


@dataclass
class OverwriteFileAction(_FileSnapshotAction):
    """タグ付けバッチによる1ファイルの上書き / 新規作成（spec.md 4.1節）。"""

    def description(self) -> str:
        return _desc("Undo_Desc_Overwrite_File", name=self.file_path.name)


@dataclass
class AppendTagsActionV2(_FileSnapshotAction):
    """タグ付けバッチによる1ファイルへの追記（spec.md 4.1節）。

    undo/redo は _FileSnapshotAction 共通の全文スナップショット方式で行い、
    added_tags は説明文・ログ用のメタ情報として保持する。
    """
    added_tags: list[str]

    def description(self) -> str:
        return _desc("Undo_Desc_Append_Tags", name=self.file_path.name,
                     count=len(self.added_tags))


@dataclass
class CompositeUndoAction(UndoAction):
    """バッチ1回分の複数ファイル変更を、Undo履歴上の1エントリとして扱う（spec.md 4.2節）。

    undo は逆順、redo は元の順で実行する。max_history=50 が大量画像処理で
    即座に消費されるのを防ぐのが目的。
    """
    actions: list[UndoAction]
    label: str = ""

    def undo(self) -> UndoResult:
        # 逆順に戻す。1 つでも成功していれば「部分成功」として履歴を進める（REQ-3.1）。
        # any(...) を全成功の根拠にしない（NFR-10）。
        outcome = _Outcome()
        for action in reversed(self.actions):
            result = as_undo_result(action.undo())
            if result is UndoResult.FAILED:
                paths = getattr(action, "failed_paths", None) or []
                if paths:
                    for path in paths:
                        outcome.fail(path)
                else:
                    outcome.fail(None)
            else:
                outcome.ok()
        return self._apply(outcome)

    def redo(self) -> UndoResult:
        outcome = _Outcome()
        for action in self.actions:
            result = as_undo_result(action.redo())
            if result is UndoResult.FAILED:
                paths = getattr(action, "failed_paths", None) or []
                if paths:
                    for path in paths:
                        outcome.fail(path)
                else:
                    outcome.fail(None)
            else:
                outcome.ok()
        return self._apply(outcome)

    def description(self) -> str:
        # label は生成側（main_window）が [MainWindow] Undo_Batch_Tagging から
        # 翻訳して渡す。空のときだけここで既定の説明文を引く。
        if self.label:
            return self.label
        return _desc("Undo_Desc_Batch_Change", count=len(self.actions))


@dataclass
class BulkAddTagsAction(UndoAction):
    """Action for adding tags to multiple images.

    REQ-3.3（R22-4 / R1-4）: ファイルごとに「実際に追加されたタグ」を持ち、
    元からあったタグは undo で消さない。並び順は利用者の指定順を保つ（ソートしない）。
    """
    file_tags: list[tuple[Path, list[str]]]
    position: str  # "prepend" or "append"

    def __post_init__(self) -> None:
        # REQ-3.6: 登録時点の絶対パスへ固定する（CWD 変更に影響されない）。
        self.file_tags = [(Path(os.path.abspath(p)), list(t)) for p, t in self.file_tags]

    def undo(self) -> UndoResult:
        """Remove the tags this action actually added from every file."""
        outcome = _Outcome()
        for path, tags in self.file_tags:
            try:
                if not path.exists():
                    write_debug_log(f"Undo: File not found: {path}")
                    outcome.fail(path)
                    continue

                content = read_text(path, encoding='utf-8').strip()
                current = [t.strip() for t in content.split(',') if t.strip()] if content else []

                remaining = list(current)
                for tag in tags:
                    if tag in remaining:
                        remaining.remove(tag)

                # REQ-3.7: 中身がこのアクションの追加分だけなら、空ファイルを残さず
                # ファイルごと無かった状態へ戻す（冪等判定より先に評価する）。
                if not remaining and all(tag in tags for tag in current):
                    os.unlink(_long_path_str(path))
                    outcome.ok()
                    continue

                if remaining == current:
                    outcome.ok()  # 既に戻っている（冪等）
                    continue

                # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
                atomic_write_text(path, ', '.join(remaining), fsync=False, verify=False)
                outcome.ok()

            except Exception as e:
                write_debug_log(f"Undo BulkAddTags failed for {path}: {e}")
                outcome.fail(path, e)

        return self._apply(outcome)

    def redo(self) -> UndoResult:
        """Re-add the recorded tags to every file (recreates missing ones)."""
        outcome = _Outcome()
        for path, tags in self.file_tags:
            try:
                current: list[str] = []
                if path.exists():
                    content = read_text(path, encoding='utf-8').strip()
                    current = [t.strip() for t in content.split(',') if t.strip()] if content else []

                to_add = [tag for tag in tags if tag not in current]
                if not to_add:
                    outcome.ok()  # 既に追加済み（冪等）
                    continue

                if self.position == "prepend":
                    new_tags = to_add + current
                else:
                    new_tags = current + to_add

                # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
                atomic_write_text(path, ', '.join(new_tags), fsync=False, verify=False)
                outcome.ok()

            except Exception as e:
                write_debug_log(f"Redo BulkAddTags failed for {path}: {e}")
                outcome.fail(path, e)

        return self._apply(outcome)

    def description(self) -> str:
        """Return a localized description of this action."""
        tag_count = max((len(tags) for _, tags in self.file_tags), default=0)
        if tag_count == 1:
            first = next((tags[0] for _, tags in self.file_tags if tags), "")
            return _desc("Undo_Desc_Bulk_Add_Tag", tag=first,
                         count=len(self.file_tags))
        return _desc("Undo_Desc_Bulk_Add_Tags", tag_count=tag_count,
                     count=len(self.file_tags))


@dataclass
class BulkRemoveTagsAction(UndoAction):
    """Action for removing a tag from multiple images.

    REQ-3.3（R5-5）: ファイルごとに「全出現位置」を持つ。undo は全ての位置へ戻し、
    redo は全ての出現を消す。どちらも冪等（既にその状態なら書かない）。
    """
    removed_tag: str
    file_positions: list[tuple[Path, list[int]]]  # (file_path, original_indexes)

    def __post_init__(self) -> None:
        # REQ-3.6: 登録時点の絶対パスへ固定する。
        self.file_positions = [(Path(os.path.abspath(p)), list(i))
                               for p, i in self.file_positions]

    def undo(self) -> UndoResult:
        """Re-insert the removed tag at every original position."""
        outcome = _Outcome()
        for path, positions in self.file_positions:
            try:
                if not path.exists():
                    write_debug_log(f"Undo: File not found: {path}")
                    outcome.fail(path)
                    continue

                content = read_text(path, encoding='utf-8').strip()
                tags = [t.strip() for t in content.split(',')] if content else []

                if tags.count(self.removed_tag) >= len(positions):
                    outcome.ok()  # 既に戻っている（冪等）
                    continue

                for index in sorted(positions):
                    insert_pos = min(index, len(tags))
                    tags.insert(insert_pos, self.removed_tag)

                # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
                atomic_write_text(path, ', '.join(tags), fsync=False, verify=False)
                outcome.ok()

            except Exception as e:
                write_debug_log(f"Undo BulkRemoveTags failed for {path}: {e}")
                outcome.fail(path, e)

        return self._apply(outcome)

    def redo(self) -> UndoResult:
        """Remove every occurrence of the tag again."""
        outcome = _Outcome()
        for path, positions in self.file_positions:
            try:
                if not path.exists():
                    write_debug_log(f"Redo: File not found: {path}")
                    outcome.fail(path)
                    continue

                content = read_text(path, encoding='utf-8').strip()
                if not content:
                    outcome.ok()
                    continue

                tags = [t.strip() for t in content.split(',')]
                before = len(tags)
                for _ in positions:
                    if self.removed_tag in tags:
                        tags.remove(self.removed_tag)

                if len(tags) != before:
                    # Write back（REQ-2.3: 一時ファイル + os.replace で原子的に）
                    atomic_write_text(path, ', '.join(tags), fsync=False, verify=False)
                outcome.ok()

            except Exception as e:
                write_debug_log(f"Redo BulkRemoveTags failed for {path}: {e}")
                outcome.fail(path, e)

        return self._apply(outcome)

    def description(self) -> str:
        """Return a localized description of this action."""
        return _desc("Undo_Desc_Bulk_Remove_Tag", tag=self.removed_tag,
                     count=len(self.file_positions))


class UndoManager:
    """Manages the history of undoable actions."""
    
    def __init__(self, max_history: int = 50):
        """
        Initialize the UndoManager.
        
        Args:
            max_history: Maximum number of actions to keep in history.
        """
        self.undo_stack: list[UndoAction] = []
        self.redo_stack: list[UndoAction] = []
        self.max_history = max_history
        # REQ-3.1: 直近の undo/redo の結果と失敗対象（GUI が件数を出し、ログが対象を残す）。
        self.last_result: UndoResult = UndoResult.FULL
        self.last_failed_paths: list[Path] = []
        write_debug_log(f"UndoManager initialized with max_history={max_history}")
    
    def push(self, action: UndoAction) -> None:
        """
        Add a new action to the undo stack.
        This clears the redo stack.
        
        Args:
            action: The action to add.
        """
        self.undo_stack.append(action)
        self.redo_stack.clear()
        
        # Limit history size
        if len(self.undo_stack) > self.max_history:
            removed = self.undo_stack.pop(0)
            write_debug_log(f"History limit reached, removed oldest action: {removed.description()}")
        
        write_debug_log(f"Action pushed: {action.description()} (undo_stack size: {len(self.undo_stack)})")
    
    def can_undo(self) -> bool:
        """Check if there are actions that can be undone."""
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if there are actions that can be redone."""
        return len(self.redo_stack) > 0
    
    def _record(self, action: UndoAction, result: UndoResult) -> None:
        """実行結果を記録する（部分成功を全成功と偽らない。NFR-10）。"""
        self.last_result = result
        self.last_failed_paths = list(getattr(action, "failed_paths", None) or [])
        write_debug_log(
            f"{type(action).__name__} -> {result.value} "
            f"(failed: {len(self.last_failed_paths)})")

    def undo(self) -> bool:
        """
        Undo the most recent action.

        Returns:
            True if successful, False otherwise.
        """
        if not self.can_undo():
            write_debug_log("Undo failed: No actions to undo")
            return False

        action = self.undo_stack.pop()
        write_debug_log(f"Undoing: {action.description()}")

        result = as_undo_result(action.undo())
        self._record(action, result)

        if result is UndoResult.FAILED:
            if getattr(action, "permanent_failure", False):
                # U-2: 恒久失敗（対象と親フォルダの消失、権限拒否）だけなら履歴から外す。
                # 先頭に戻し続けると以降の Undo が全て同じアクションで止まる（R11-11）。
                write_debug_log(
                    "Undo failed permanently; dropping the action from history: "
                    f"{action.description()}")
                return False
            # A transient file-system failure must not consume the only record of the
            # action.  Keep it at the top so the user can fix the cause and retry.
            self.undo_stack.append(action)
            write_debug_log("Undo failed, action restored to undo stack")
            return False

        # FULL / PARTIAL とも履歴は進める（部分成功は「戻せた分」を活かす）。
        self.redo_stack.append(action)
        write_debug_log(f"Undo successful ({result.value}; redo_stack size: {len(self.redo_stack)})")
        return True

    def redo(self) -> bool:
        """
        Redo the most recently undone action.

        Returns:
            True if successful, False otherwise.
        """
        if not self.can_redo():
            write_debug_log("Redo failed: No actions to redo")
            return False

        action = self.redo_stack.pop()
        write_debug_log(f"Redoing: {action.description()}")

        result = as_undo_result(action.redo())
        self._record(action, result)

        if result is UndoResult.FAILED:
            if getattr(action, "permanent_failure", False):
                write_debug_log(
                    "Redo failed permanently; dropping the action from history: "
                    f"{action.description()}")
                return False
            self.redo_stack.append(action)
            write_debug_log("Redo failed, action restored to redo stack")
            return False

        self.undo_stack.append(action)
        write_debug_log(f"Redo successful ({result.value}; undo_stack size: {len(self.undo_stack)})")
        return True
    
    def get_undo_description(self) -> str:
        """
        Get a description of the next action that would be undone.
        
        Returns:
            Description string, or empty string if no actions to undo.
        """
        if self.can_undo():
            return self.undo_stack[-1].description()
        return ""
    
    def get_redo_description(self) -> str:
        """
        Get a description of the next action that would be redone.
        
        Returns:
            Description string, or empty string if no actions to redo.
        """
        if self.can_redo():
            return self.redo_stack[-1].description()
        return ""
    
    def clear(self) -> None:
        """Clear all undo/redo history."""
        self.undo_stack.clear()
        self.redo_stack.clear()
        write_debug_log("Undo/Redo history cleared")
