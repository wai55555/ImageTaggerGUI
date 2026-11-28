"""テキストファイルの書き込みヘルパ（原子的書き込み・拡張長パス変換）。

REQ-2.2: それまで `vlm_persistence` の中にあった `_atomic_write()` /
`_long_path_str()` を、VLM 固有でない共有部品として切り出したもの。Qt 非依存で、
標準ライブラリだけに依存する（src 内のフラット import 前提・循環 import なし）。

原子的書き込みの意味論:

- 一時ファイル（対象と同ディレクトリ・衝突しない名前・拡張子は .txt 以外）へ書き、
  `os.replace()` で設置する。失敗した場合、対象ファイルは**変更前の内容のまま**残る。
- 一時ファイルは `finally` で必ず消す（正常終了時は replace 済みなので存在しない）。
- `verify=True` のときは、書いた内容を読み直して一致を確認してから replace する。
  比較は改行を LF に正規化して行う（CRLF 翻訳の有無で誤検知しないため）。
- `fsync=True` のときは flush 後に `os.fsync()` を呼ぶ（電源断での消失耐性）。

呼び出し側ごとに現行挙動が違う 3 点は、すべて引数で選べるようにしてある:

- `newline`: `None` なら `open()` の既定（Windows では `\\n` → CRLF 翻訳）。VLM 経路だけが
  既存ファイルから検出した改行規約を明示する。**既定を `"\\n"` にしてはならない**
  （Windows で config.ini やキャプション .txt のバイト列が変わる。REQ-2.5）。
- `makedirs`: 正本（VLM 経路）は親ディレクトリを作るが、他の経路は「親が無ければ
  失敗する」のが現行挙動。既定 False で、VLM 経路だけ True を渡す。
- `fsync` / `verify`: 高頻度・大量の書き込みではコストになるため、経路ごとに
  計測して選ぶ（REQ-2.7）。既定は安全側（どちらも True）。

symlink / hardlink（§6 U-7・2026-11-27 決定で対象外）:

- 対象が symlink / hardlink のときは `os.replace()` を使わず、**リンクを辿って実体へ
  直接書く**（現行挙動の維持）。replace 方式だとリンク自体が通常ファイルに置き換わり、
  実体が更新されずリンクも切れる。この経路では verify も fsync も行わない
  （判定コストを負ケースに足さないため。U-7 の決定どおり対象外）。
"""
from __future__ import annotations

import os
import stat
import sys
import uuid
from pathlib import Path
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]

# 一時ファイルの拡張子。**`.txt` にしてはならない**: 異常終了で残った残骸が、次回の
# 画像走査・タグ集計に成果物として混入する（REQ-2.6）。VLM 経路の現行命名を踏襲する。
DEFAULT_TMP_SUFFIX = ".vlmtmp"


def long_path_str(path: PathLike) -> str:
    """Windows の MAX_PATH（260 文字）制限を避ける `\\\\?\\` 付き絶対パスを返す。

    - `os.path.abspath()` を使う（`Path.resolve()` ではない）: `\\\\?\\` パスは OS が
      そのまま使うので `..` の正規化は必要だが、**symlink は解決しない**。resolve() だと
      リンク先の実体を別ファイルとして扱ってしまう（PR#16 レビュー指摘）。
    - UNC パス（`\\\\server\\share\\...`）は `\\\\?\\UNC\\server\\share\\...` の形にする必要が
      あり、単純に `\\\\?\\` を前置すると不正なパスになる（PR#16 レビュー指摘）。
    - 既に `\\\\?\\` 付きならそのまま返す（二重前置は `\\\\?\\UNC\\?\\...` のような不正パスに
      なる。PR#16 レビュー第 2 ラウンド指摘）。
    - 非 Windows では恒等（`abspath` のみ）。
    """
    p = os.path.abspath(path)
    if sys.platform != "win32":
        return p
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC" + p[1:]
    return "\\\\?\\" + p


def is_link_target(path: PathLike) -> bool:
    """対象が symlink、または hardlink（link count > 1）か。存在しなければ False。"""
    try:
        if os.path.islink(path):
            return True
        return os.lstat(path).st_nlink > 1
    except OSError:
        return False


def _existing_mode(path: str) -> int | None:
    """既存ファイルのモード（誰が読めるか）を返す。無ければ None。"""
    try:
        return stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return None


def _apply_mode(path: str, mode: int) -> None:
    """モードを設定する。失敗しても書き込み自体は続ける（Windows では制限あり）。"""
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def atomic_write_text(
    path: PathLike,
    content: str,
    *,
    newline: str | None = None,
    encoding: str = "utf-8",
    fsync: bool = True,
    verify: bool = True,
    makedirs: bool = False,
    use_long_path: bool = True,
    tmp_suffix: str = DEFAULT_TMP_SUFFIX,
) -> None:
    """`content` を `path` へ原子的に書く。失敗時は `path` を変更前のまま残す。

    Args:
        path: 書き込み先。
        content: 書き込む文字列。
        newline: `open()` に渡す改行規約。既定 `None`（プラットフォーム既定）。
            既存ファイルの規約へ合わせたい経路だけ明示する。
        encoding: 文字コード。既定 UTF-8。
        fsync: flush 後に `os.fsync()` するか。
        verify: 書き込み後に読み直して内容を検証するか。
        makedirs: 親ディレクトリを作るか。既定 False（現行の非 VLM 経路は
            「親が無ければ失敗」なので、既定で挙動を変えない）。
        use_long_path: 拡張長パス変換を通すか。現行のまま変換を通さない経路
            （`undo_manager.EditCaptionAction`）だけ False を渡す。
        tmp_suffix: 一時ファイルの拡張子。`.txt` を渡してはならない。

    Raises:
        OSError: 書き込み・検証・置換のいずれかに失敗した場合。
    """
    target = Path(path)
    long_target = long_path_str(target) if use_long_path else str(target)

    if is_link_target(long_target):
        # U-7: リンクは辿って実体へ書く（現行挙動）。os.replace は使わない。
        with open(long_target, "w", encoding=encoding, newline=newline) as handle:
            handle.write(content)
        return

    if makedirs:
        os.makedirs(long_path_str(target.parent), exist_ok=True)

    # 既存ファイルを置換する場合、一時ファイルを既定モード（umask 次第で 0644）で作ると
    # 0600 だった内容が他ユーザーから読めるようになる。書き込み前から本人だけに絞り、
    # 置換直前に元のモードへ戻す（PR#38 レビュー指摘）。新規作成は従来の open() と同じ
    # モードのまま（広げるものが無いので触らない）。
    preserved_mode = _existing_mode(long_target)

    tmp = target.with_name(f"{target.stem}.{uuid.uuid4().hex}{tmp_suffix}")
    long_tmp = long_path_str(tmp) if use_long_path else str(tmp)
    try:
        with open(long_tmp, "w", encoding=encoding, newline=newline) as handle:
            if preserved_mode is not None:
                _apply_mode(long_tmp, 0o600)
            handle.write(content)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        if verify:
            # newline="" で生読みし、書き出し時の改行変換ぶんを正規化してから比較する
            # （正本 vlm_persistence と同じ判定。CRLF 環境でも誤検知しない）。
            with open(long_tmp, "r", encoding=encoding, newline="") as handle:
                written = handle.read()
            if written.replace("\r\n", "\n").replace("\r", "\n") != content:
                raise OSError("verification mismatch after write")
        if preserved_mode is not None:
            _apply_mode(long_tmp, preserved_mode)
        os.replace(long_tmp, long_target)
    finally:
        try:
            if os.path.exists(long_tmp):
                os.unlink(long_tmp)
        except OSError:
            pass
