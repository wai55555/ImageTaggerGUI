import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QProxyStyle, QStyle
from constants import APP_VERSION as __version__
from main_window import MainWindow


class _FastTooltipStyle(QProxyStyle):
    """Shortens the tooltip hover delay app-wide.

    Fusion's default SH_ToolTip_WakeUpDelay is 700 ms, which feels sluggish for the
    model-picker dropdown where the tooltips carry the per-model descriptions the user
    is deliberately hovering to read. 300 ms stays clear of accidental flicker while
    dragging the pointer across items.
    """

    WAKE_UP_DELAY_MS = 300

    def styleHint(self, hint, option=None, widget=None, return_data=None):
        if hint == QStyle.StyleHint.SH_ToolTip_WakeUpDelay:
            return self.WAKE_UP_DELAY_MS
        return super().styleHint(hint, option, widget, return_data)


def os_prefers_dark(app: QApplication, fallback_palette: QPalette | None = None) -> bool:
    """OS のカラースキームがダークかどうかを返す。

    アプリは Fusion スタイルを強制しており、Fusion は OS のテーマに追随しない固定
    パレットを持つ。そのため `QApplication.palette()` を見てもライト判定にしかならず、
    「システムテーマ感知が働かない」状態になっていた。Qt 6.5 以降の
    `QStyleHints.colorScheme()` はスタイルに依存せず OS 設定そのものを返すので、
    そちらを一次情報として使い、取得できない環境ではパレットの明度に戻す。

    `fallback_palette` は `setStyle(Fusion)` を適用する**前**に取得したパレットを渡すこと。
    `colorScheme()` が使えない環境こそこのフォールバックが必要な環境であり、そこで
    `app.palette()`（Fusion 適用後）を見ると常に Fusion の固定ライトパレットを見てしまい、
    このフォールバック自体が無意味になる（PR#16 レビュー指摘）。省略時は互換のため
    `app.palette()` を使うが、`main()` からは必ず退避済みのものを渡す。
    """
    try:
        scheme = app.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except Exception:
        pass
    palette = fallback_palette if fallback_palette is not None else app.palette()
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


def apply_dark_palette(app: QApplication) -> None:
    """Fusion 用のダークパレットを適用する。

    検出だけ直してもウィンドウ自体はライトのままなので、ログ色やハイライト色だけが
    ダーク向けになってちぐはぐになる。OS がダークなら見た目もダークに揃える。
    """
    p = QPalette()
    window = QColor(53, 53, 53)
    base = QColor(35, 35, 35)
    text = QColor(220, 220, 220)
    disabled = QColor(127, 127, 127)
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, window)
    p.setColor(QPalette.ColorRole.ToolTipBase, window)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, window)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    p.setColor(QPalette.ColorRole.PlaceholderText, disabled)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText, QPalette.ColorRole.HighlightedText):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    app.setPalette(p)


def main():
    """main entry point."""
    # 未捕捉例外が黙って消える問題への対処。windowed ビルド（console=False）では
    # sys.stderr/sys.stdout が None になり、既定の sys.excepthook は何も出力しない
    # まま GUI だけが動き続ける。QApplication 生成より前に一番最初にインストールし、
    # 以降のどの初期化コードで例外が出ても error_log.txt に残るようにする。
    from utils import install_excepthook

    install_excepthook()

    # GPU コンポーネント（gpu_runtime/）が入っていれば、最初の InferenceSession より
    # 前に CUDA/cuDNN DLL を明示ロードしておく（順序を誤るとシステム PATH 上の別
    # バージョン cuDNN を掴む）。未整備なら無害な no-op。例外は内部で握られる。
    try:
        from onnx_providers import preload_gpu_dlls

        preload_gpu_dlls()
    except Exception as exc:
        # preload_gpu_dlls() itself already catches and logs its own failures, so
        # reaching here means something outside that (e.g. the `from onnx_providers
        # import` line itself, if the module is missing from a broken build) went
        # wrong. In practice a missing onnx_providers would already have crashed the
        # process at import time via main_window's own top-level import chain, well
        # before main() runs - this handler is really just a last-resort diagnostic
        # for anything unexpected here, not the primary place that failure surfaces
        # (cubic review, PR #21).
        try:
            from utils import write_debug_log

            write_debug_log(f"main: preload_gpu_dlls unavailable ({exc!r})")
        except Exception:
            # sys.stderr (and sys.stdout) is None in a windowed PyInstaller build
            # (console=False in the spec) - print(..., file=None) falls back to
            # sys.stdout internally, which is also None there, so an unguarded
            # print() would itself raise and crash startup (cubic review, PR #22).
            if sys.stderr is not None:
                print(f"preload_gpu_dlls unavailable: {exc!r}", file=sys.stderr)

    app = QApplication(sys.argv)

    # colorScheme() が使えない環境向けのフォールバック判定は Fusion 適用前のプラット
    # フォーム既定パレットで行う必要がある（setStyle 後の palette() は Fusion の固定
    # ライトパレットになり判定できない。PR#16 レビュー指摘）。
    platform_palette = app.palette()
    app.setStyle(_FastTooltipStyle("Fusion"))

    # スタイルを適用した後に OS のテーマへ合わせる。MainWindow はこの時点のパレットから
    # ダーク判定を行うので、ウィンドウ生成より前に済ませておく必要がある。
    if os_prefers_dark(app, platform_palette):
        apply_dark_palette(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
