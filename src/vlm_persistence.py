"""キャプション保存（260901_VLM_spec.md 11章 / design.md 8章）。

- 既存内容とキャプションの区切りは改行1つ（既存 caption_core の ", " 区切りとは別物）
- APPEND / PREPEND では完全一致の重複追加を防ぐ
- 一時ファイル + 検証 + os.replace の原子的書き込み。失敗時は既存 .txt を変えない
- 生成失敗時は保存処理を呼ばない（呼び出し側の責務）
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# REQ-2.2: 原子的書き込みと拡張長パス変換は VLM 非依存の共通部品（src/text_io.py）へ移した。
# ここは従来どおり (i) fsync + 読み直し検証 + replace、親ディレクトリ作成ありで呼ぶ。
from text_io import atomic_write_text, long_path_str


VLM_PLACEMENTS = ("PREPEND", "APPEND", "OVERWRITE")


def parse_placement(raw: str) -> str:
    v = str(raw).strip().upper()
    return v if v in VLM_PLACEMENTS else "OVERWRITE"


def combine_caption(existing: str, caption: str, placement: str) -> str:
    """既存内容と生成キャプションを改行1つで結合する（spec.md 11.1節）。

    片方が空なら余分な改行を足さない。OVERWRITE は既存を無視してキャプションのみ。
    """
    placement = parse_placement(placement)
    existing = existing.strip()
    caption = caption.strip()
    if placement == "OVERWRITE" or not existing:
        return caption
    if not caption:
        return existing
    if placement == "PREPEND":
        return f"{caption}\n{existing}"
    return f"{existing}\n{caption}"


def caption_already_present(existing: str, caption: str) -> bool:
    """既存内容の中に、追加しようとするキャプションと完全一致する塊があるか。

    意味的・表現的な類似は判定しない（spec.md 11.3節）。複数行キャプションにも対応する
    よう、改行を境界とした contiguous な完全一致（先頭 / 末尾 / 中間 / 全体）を見る。
    改行コードは正規化して比較する（CRLF の既存 .txt と LF のキャプション、その逆でも
    同一とみなす。Windows でメモ帳編集した .txt などが対象）。
    """
    def _lf(s: str) -> str:
        return s.replace("\r\n", "\n").replace("\r", "\n")
    target = _lf(caption.strip())
    ex = _lf(existing.strip())
    if not target:
        return True
    if ex == target:
        return True
    return (ex.startswith(target + "\n")
            or ex.endswith("\n" + target)
            or ("\n" + target + "\n") in ex)


@dataclass(frozen=True)
class SaveOutcome:
    written: bool
    path: Path
    previous_content: str | None      # None = 変更前にファイルが存在しなかった
    new_content: str
    skipped_reason: str = ""          # "duplicate" / "no_change" / ""（＝書き込んだ）


def save_caption(output_path: Path, caption: str, placement: str) -> SaveOutcome:
    """キャプションを output_path へ保存する。

    手順（spec.md 11.2節）:
      1. 既存ファイルを読み込む（読めない既存ファイルは触らず例外を送出）
      2. 結合後の内容をメモリで作る
      3. 一時ファイルへ書く
      4. 読み直して検証する
      5. os.replace で設置する
    """
    placement = parse_placement(placement)
    # 以降は改行を LF に統一して扱う。ファイルへ書くときだけ output_newline へ戻す
    # （生成キャプションに \r\n が混じっていても検証不一致で失敗しないように）。
    caption = caption.strip().replace("\r\n", "\n").replace("\r", "\n")
    if not caption:
        raise ValueError("empty caption")

    previous_content: str | None = None
    # 既存ファイルの改行コード。書き出し時にこの規約へ戻し、CRLF の .txt が APPEND で
    # 黙って LF 化されるのを防ぐ。undo スナップショット（previous_content / new_content）は
    # LF 正規化して持ち、_FileSnapshotAction._write（newline=None）の書き戻し翻訳と整合させる。
    output_newline = "\n"
    fs_output_path = Path(long_path_str(output_path))
    if fs_output_path.is_file():
        # 読めない既存ファイルは新規扱いにしない（undo での破壊防止。PR#16 と同方針）。
        with open(long_path_str(output_path), "r", encoding="utf-8", newline="") as f:
            raw_previous = f.read()
        output_newline = "\r\n" if "\r\n" in raw_previous else "\n"
        previous_content = raw_previous.replace("\r\n", "\n").replace("\r", "\n")

    if placement in ("APPEND", "PREPEND") and previous_content is not None:
        if caption_already_present(previous_content, caption):
            return SaveOutcome(False, output_path, previous_content,
                               previous_content, skipped_reason="duplicate")

    new_content = combine_caption(previous_content or "", caption, placement)
    if previous_content is not None and new_content == previous_content:
        return SaveOutcome(False, output_path, previous_content, previous_content,
                           skipped_reason="no_change")

    _atomic_write(output_path, new_content, newline=output_newline)
    return SaveOutcome(True, output_path, previous_content, new_content)


def _atomic_write(output_path: Path, content: str, *, newline: str = "\n") -> None:
    """`content`（LF 前提）を `newline` 規約で原子的に書く。検証は改行を正規化して比較する。

    （REQ-2.2 で共通部品へ移動。ここは従来と同じフル装備 = fsync + 検証 + 親ディレクトリ作成。）
    """
    atomic_write_text(
        output_path,
        content,
        newline=newline,
        fsync=True,
        verify=True,
        makedirs=True,
    )
