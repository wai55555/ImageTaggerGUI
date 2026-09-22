"""docs/260922_code_review_report.md の4件の指摘に対する回帰テスト。

いずれも「無言で壊れる」類（終了時クラッシュ／未検証モデルの検証済み扱い／
再起動で経路が黙って入れ替わる／子ウィジェットが画面外へ出る）なので、
再発したら落ちるように固定する。

Offline only - no network. Run:  rtk pytest tests/test_review_260922_fixes.py -q
"""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])


def _vlm_settings(profile_id="gemma-4-31b-it", connection_order="gemini,openai"):
    """config.ini を触らずに VlmSettings 相当を組み立てる。"""
    s = types.SimpleNamespace(
        model_profile_id=profile_id, cloudflare_account_id="", anthropic_workspace_id="",
        verified_bindings="", model_id_overrides="", vlm_capable_overrides="",
        connection_order=connection_order)
    s.verified_set = lambda: {t.strip() for t in s.verified_bindings.split(",") if t.strip()}
    s.order_list = lambda: [p for p in s.connection_order.split(",") if p]
    s.model_id_override_map = lambda: {
        k.strip(): v.strip()
        for k, _, v in (t.partition("=") for t in s.model_id_overrides.split(","))
        if k.strip() and v.strip()}
    return s


# --- [P1] 退避済みスレッドの寿命 ----------------------------------------------------

class _NeverStops(QThread):
    """quit() に応じないスレッド（_cleanup_tagger_thread が手放す状況）。"""

    def run(self):
        while not self.isInterruptionRequested():
            QThread.msleep(20)


def test_detached_threads_live_outside_the_main_window_instance():
    """退避先が MainWindow の属性だと、ウィンドウ破棄で参照ごと消えてしまう。

    稼働中の QThread への最後の Python 参照が消えると PySide6 が C++ 側を破棄し、
    Qt の「Deleting a running QThread」経路でプロセスが fail-fast する
    （最小再現で終了コード 0xC0000409 を確認）。置き場はモジュール変数であること。
    """
    import main_window as MW
    from main_window import MainWindow

    assert isinstance(MW._detached_threads, list)
    assert not hasattr(MainWindow, "_detached_threads"), \
        "インスタンス属性に戻すと、ウィンドウ破棄で参照が消える"

    MW._detached_threads.clear()
    thread = _NeverStops()
    thread.start()
    try:
        assert thread.isRunning()
        MainWindow._detach_running_thread(None, thread, None)
        assert thread in MW._detached_threads
        assert MW.detached_thread_count() == 1
        # 稼働中はイベントループを回しても破棄されない。
        for _ in range(3):
            _APP.processEvents()
        assert thread.isRunning()
    finally:
        thread.requestInterruption()
        thread.quit()
        assert thread.wait(5000)
        for _ in range(3):
            _APP.processEvents()
        MW._detached_threads.clear()


def test_wait_for_detached_threads_reports_and_keeps_references():
    """終了処理は退避済みスレッドを待ち、待ちきれなくても参照は手放さない。"""
    import main_window as MW
    from main_window import MainWindow

    MW._detached_threads.clear()
    assert MW.wait_for_detached_threads(10) is True   # 空なら即 True

    thread = _NeverStops()
    thread.start()
    try:
        MainWindow._detach_running_thread(None, thread, None)
        assert MW.wait_for_detached_threads(200) is False
        # 待ちきれなくても参照は残す（破棄するとクラッシュするため）。
        assert thread in MW._detached_threads
        assert thread.isRunning()
    finally:
        thread.requestInterruption()
        thread.quit()
        assert thread.wait(5000)
        for _ in range(3):
            _APP.processEvents()
        MW._detached_threads.clear()

    # 終了済みになれば待機は成功し、置き場からも外れる。
    assert MW.wait_for_detached_threads(10) is True
    assert MW.detached_thread_count() == 0


def _method_body(source: str, name: str) -> str:
    """`def <name>(` から次の同インデントの `def ` 直前までを返す。

    ファイル末尾まで切り出すと、後続メソッドの docstring 内の言及まで拾って
    しまい、呼び出しを消す変異を取り逃す。
    """
    after = source.split(f"def {name}(", 1)[1]
    end = after.find("\n    def ")
    return after if end < 0 else after[:end]


def test_close_event_waits_for_detached_threads():
    """closeEvent が退避済みスレッドを待っていること（ソース上の配線確認）。"""
    source = (ROOT / "src" / "main_window.py").read_text(encoding="utf-8")
    close_event = _method_body(source, "closeEvent")
    assert "wait_for_detached_threads(" in close_event, \
        "closeEvent が退避済みスレッドを待っていない"


# --- [P1] override のモデルを変えたら検証済みを失効させる ---------------------------

def test_changing_the_override_model_drops_the_verified_state():
    import vlm_config as CFG
    import vlm_models as M

    s = _vlm_settings()
    profile = CFG.resolve_model_profile(s)
    CFG.set_model_id_override(s, "openai", "gpt-5.6-sol", profile_id=s.model_profile_id)
    assert CFG.mark_binding_verified(s, "openai", profile_id=s.model_profile_id)
    bound = CFG.profile_with_override_bindings(s, profile).binding_for("openai")
    assert bound.identity_status is M.ModelIdentityStatus.VERIFIED

    # 別のモデルへ差し替えたら、検証済みは引き継がない。
    CFG.set_model_id_override(s, "openai", "gpt-5.6-luna", profile_id=s.model_profile_id)
    assert "gemma-4-31b-it:openai" not in s.verified_set()
    rebound = CFG.profile_with_override_bindings(s, profile).binding_for("openai")
    assert rebound.model_id == "gpt-5.6-luna"
    assert rebound.identity_status is M.ModelIdentityStatus.DECLARED, \
        "一度も試していないモデルが VERIFIED になっている"


def test_same_override_value_keeps_the_verified_state():
    """同じIDで上書きし直しただけなら検証済みは消さない。"""
    import vlm_config as CFG

    s = _vlm_settings()
    CFG.set_model_id_override(s, "openai", "gpt-5.6-sol", profile_id=s.model_profile_id)
    CFG.mark_binding_verified(s, "openai", profile_id=s.model_profile_id)
    CFG.set_model_id_override(s, "openai", "gpt-5.6-sol", profile_id=s.model_profile_id)
    assert "gemma-4-31b-it:openai" in s.verified_set()


def test_clearing_the_override_drops_both_verified_and_capability():
    import vlm_config as CFG

    s = _vlm_settings()
    CFG.set_model_id_override(s, "openai", "gpt-5.6-sol", profile_id=s.model_profile_id)
    CFG.mark_binding_verified(s, "openai", profile_id=s.model_profile_id)
    CFG.mark_override_vlm_capable(s, "openai", "gpt-5.6-sol",
                                  profile_id=s.model_profile_id)
    CFG.set_model_id_override(s, "openai", "", profile_id=s.model_profile_id)
    assert s.verified_set() == set()
    assert CFG.vlm_capable_override_map(s) == {}


# --- [P1] ライブ一覧で見つけた経路が再起動後も残る -----------------------------------

def test_discovered_override_capability_survives_a_restart():
    """出荷カタログに無いIDでも、保存しておけば再起動後に同じ判定になる。

    これが無いと、再起動後は is_vlm_model_id() が False になり、利用者が明示的に
    選んだ経路（connection_order=openai）が黙ってプロファイル既定の別経路群へ
    差し替わっていた。
    """
    import vlm_config as CFG
    import vlm_models as M

    new_id = "future/vendor-vision-1"
    s = _vlm_settings(connection_order="openai")
    profile = CFG.resolve_model_profile(s)

    saved = dict(M._DISCOVERED_VLM_MODEL_IDS)
    try:
        M._DISCOVERED_VLM_MODEL_IDS.clear()
        # 「モデル一覧を取得」で拾い、override として保存した状態。
        M.register_discovered_vlm_ids("openai", [new_id])
        CFG.set_model_id_override(s, "openai", new_id, profile_id=s.model_profile_id)
        CFG.mark_override_vlm_capable(s, "openai", new_id, profile_id=s.model_profile_id)
        in_session = CFG.ordered_builtin_provider_ids(s, profile)
        assert in_session == ["openai"], in_session

        # 再起動を模す: プロセス内登録だけ消す。
        M._DISCOVERED_VLM_MODEL_IDS.clear()
        assert M.is_vlm_model_id(profile, "openai", new_id) is False
        assert CFG.ordered_builtin_provider_ids(s, profile) != ["openai"]

        # 起動時の復元を通すと、セッション中と同じ判定へ戻る。
        assert CFG.restore_discovered_vlm_ids(s) == 1
        assert M.is_vlm_model_id(profile, "openai", new_id) is True
        assert CFG.ordered_builtin_provider_ids(s, profile) == ["openai"]
        assert CFG.build_connection_map(s, profile)["builtin-openai"].enabled is True
    finally:
        M._DISCOVERED_VLM_MODEL_IDS.clear()
        M._DISCOVERED_VLM_MODEL_IDS.update(saved)


def test_startup_restores_discovered_ids():
    """起動時（_initialize_settings_and_locale）に復元を通していること。"""
    source = (ROOT / "src" / "main_window.py").read_text(encoding="utf-8")
    init = _method_body(source, "_initialize_settings_and_locale")
    assert "restore_discovered_vlm_ids(" in init


def test_vlm_capable_overrides_round_trips_through_config(tmp_path, monkeypatch):
    """新しい設定項目が config.ini に保存・復元されること。"""
    import app_settings as A

    monkeypatch.setattr(A, "CONFIG_PATH", tmp_path / "config.ini")
    settings = A.load_settings(A.get_default_config())
    settings.vlm.vlm_capable_overrides = "gemma-4-31b-it:openai=future/vendor-vision-1"
    assert A.save_config(settings)
    reloaded = A.load_settings(A.load_config())
    assert (reloaded.vlm.vlm_capable_overrides
            == "gemma-4-31b-it:openai=future/vendor-vision-1")


# --- [P2] 経路欄が画面・ダイアログの内側に収まる -------------------------------------

def test_routes_area_stays_inside_the_dialog():
    """幅上限から親レイアウトの余白を引かないと、子だけが外へはみ出す。"""
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    settings = A.load_settings(A.get_default_config())
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        dialog.show()
        _APP.processEvents()
        for mode in ("recommended", "all"):
            getattr(dialog, f"routes_mode_{mode}").setChecked(True)
            _APP.processEvents()
            scroll = dialog._routes_scroll
            left = scroll.mapTo(dialog, scroll.rect().topLeft()).x()
            right = left + scroll.width()
            assert right <= dialog.width(), (mode, left, scroll.width(), dialog.width())
            cap = dialog._width_cap()
            if cap is not None:
                assert dialog.minimumWidth() <= cap, mode
                # 余白ぶんが引かれている（上限をそのまま子へ流していない）。
                assert scroll.minimumWidth() <= cap - dialog._routes_horizontal_inset() \
                    or scroll.minimumWidth() == max(dialog._routes_grid.sizeHint().width(), 1)
    finally:
        dialog.close()


def test_routes_horizontal_inset_is_measured_not_hardcoded():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    settings = A.load_settings(A.get_default_config())
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        dialog.show()
        _APP.processEvents()
        inset = dialog._routes_horizontal_inset()
        # 実レイアウトから求めるので 0 より大きく、ダイアログ幅未満。
        assert 0 < inset < dialog.width()
    finally:
        dialog.close()


def test_horizontal_scrollbar_appears_only_when_the_width_is_capped():
    import app_settings as A
    from vlm_settings_dialog import VlmSettingsDialog

    settings = A.load_settings(A.get_default_config())
    dialog = VlmSettingsDialog(settings, lambda sec, key, **kw: key)
    try:
        dialog.show()
        _APP.processEvents()
        dialog.routes_mode_all.setChecked(True)
        _APP.processEvents()
        scroll = dialog._routes_scroll
        natural = dialog._routes_grid.sizeHint().width()
        capped = scroll.minimumWidth() < natural
        expected = (Qt.ScrollBarPolicy.ScrollBarAlwaysOn if capped
                    else Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        assert scroll.horizontalScrollBarPolicy() is expected
    finally:
        dialog.close()


# --- MainWindow の未フラッシュ singleShot(0) がテストを跨いでクラッシュさせる件 ------

def test_main_window_flushes_its_own_initial_load_before_returning():
    """MainWindow.__init__ の QTimer.singleShot(0, self.initial_load) が、この
    テスト自身の中で発火させ切られていること。

    起動直後に app.exec() が続く実アプリでは、initial_load() は他の何より先に
    走るので安全。だが単体テストで MainWindow() を作って一度もイベントループを
    回さずに関数を抜けると、Qt が `w` を生かしたまま initial_load() を保留し続け、
    それが後で発火するのは「たまたま次に processEvents() を呼んだ、全く無関係な
    別テスト」の最中になる。そこで reload_image_list() などの重い処理が走り、
    Windows で fail-fast（終了コード 0xC0000409）を起こした
    （tests/test_pr16_review_fixes.py::test_main_window_locking_and_no_clear と
    tests/test_vlm_phase3.py の組み合わせで実際に確認、260923）。
    """
    source = (ROOT / "src" / "main_window.py").read_text(encoding="utf-8")
    assert "QTimer.singleShot(0, self.initial_load)" in source, \
        "この前提が崩れたら本テストの意義も無くなるので、まず確認する"

    import main_window

    calls: list[bool] = []
    original = main_window.MainWindow.initial_load
    main_window.MainWindow.initial_load = lambda self: (calls.append(True), original(self))[1]
    try:
        w = main_window.MainWindow()
        try:
            # __init__ の中では singleShot(0) が積まれるだけで、まだ呼ばれていない。
            assert calls == []
            for _ in range(5):
                _APP.processEvents()
            # ここまでで発火し切っている（そもそも遅延させている意味が無くなる）。
            assert calls == [True]
        finally:
            w.close()
    finally:
        main_window.MainWindow.initial_load = original


def test_the_actual_crashing_combination_now_exits_cleanly():
    """実際に fail-fast（exit 9 / 0xC0000409）を再現していた組み合わせを、
    サブプロセスの pytest で本当に再実行し、終了コードで確認する。

    上のテストのようにこのプロセス内だけで確認すると、「MainWindow を作って
    processEvents で流す」という直したい対象そのものを自分で正しくなぞって
    しまい、tests/test_pr16_review_fixes.py 側の修正が消えても気付けない。
    実際に壊れていたファイル・テストの組み合わせをサブプロセスで再実行するのが
    唯一確実な検証方法（このプロセス自身が既にクラッシュしていたら
    このテストの結果自体を報告できない、という事情のため隔離する）。
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_pr16_review_fixes.py::test_main_window_locking_and_no_clear",
         "tests/test_vlm_phase3.py::test_routes_all_tab_caps_scroll_height_to_about_four_rows"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"exit={result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")
    assert "2 passed" in result.stdout
