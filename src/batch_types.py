"""バッチ処理まわりで共有する型（REQ-3.3）。

`FileChange` / `ExistingFileMode` / `OverwriteDecision` は元々 `tagging_core` にあったが、
ループ骨格（`batch_runner.py`）が `tagging_core` を import すると循環するため、ここへ
移した。`tagging_core` は互換のため再エクスポートする（`from tagging_core import FileChange`
や `tagging_core.ExistingFileMode` は従来どおり動く）。

Qt 非依存・標準ライブラリのみ。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


@dataclass(frozen=True)
class FileChange:
    """1ファイル分の書き換え記録（design.md 2.2節）。

    previous_content が None なら「変更前にファイルが存在しなかった」ことを表し、
    Undo ではファイルを削除する。was_append は追記ログ / 説明用のメタ情報。
    """
    path: Path
    previous_content: str | None
    new_content: str
    was_append: bool
    added_tags: tuple[str, ...] = ()


class ExistingFileMode(Enum):
    """出力先の .txt が既に存在する場合の処理方針（spec.md 1.1節）。"""
    ASK = auto()        # 都度確認
    OVERWRITE = auto()  # 常に上書き
    SKIP = auto()       # 常にスキップ
    APPEND = auto()     # 常に追記


class OverwriteDecision(Enum):
    """ASK モードでユーザーが1ファイルに対して下した判断（spec.md 2.2節）。"""
    OVERWRITE = auto()
    SKIP = auto()
    APPEND = auto()
