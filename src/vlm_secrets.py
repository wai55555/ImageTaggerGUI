"""API キーなど秘密情報の解決（260901_VLM_spec.md 15章 / implement_plan 13.1節）。

優先順位: セッション中だけ保持 → OS の秘密情報ストレージ（keyring） → 環境変数。
秘密値は config.ini / vlm_connections.json / 設定エクスポート / debug log / エラー
ダイアログのいずれにも出さない。この層だけが実体を扱う。
"""
from __future__ import annotations

import os
import threading

from utils import write_debug_log

try:  # keyring はオプション依存。無い環境ではセッション保持のみに落ちる。
    import keyring  # type: ignore
except Exception:  # pragma: no cover - 環境依存
    keyring = None  # type: ignore

_SERVICE = "ImageTaggerGUI.VLM"              # 2026-09 改名後の保存先（新規 ref はこちら）
_LEGACY_SERVICE = "PixaiTaggerOnnxGui.VLM"   # 改名前。既存エントリはここに残したまま使い続ける


def _service_for(secret_ref: str) -> str:
    """この ref を書き込む先。既に値がある方（新を優先）、どちらにも無ければ新。

    default は _SERVICE でなければならない（過去の CodeRabbit 指摘を却下した理由）:
    ここは「新規作成 or 既存更新のどちらか」を決める場であり、既存が無ければ新に
    作るのが目的。default を _LEGACY_SERVICE にすると _SERVICE への初回書き込み
    経路が無くなり（_SERVICE を返すには既に _SERVICE にある必要がある、という循環）、
    改名後もずっと新規接続が旧サービス名に書かれ続ける。

    チェック順は get_secret() と揃える（260922 PR#27 レビュー指摘）。両サービスに
    同じ ref の値があるとき旧を返すと、保存は旧へ行くのに get_secret() は新を先に
    読むため、保存した値が以後一切読まれない（古い新サービス値が返り続ける）。
    default を _SERVICE に保つ限り、順序を揃えても上記の初回書き込み経路は壊れない。
    """
    if keyring is not None:
        for service in (_SERVICE, _LEGACY_SERVICE):
            try:
                if keyring.get_password(service, secret_ref):
                    return service
            except Exception:
                pass
    return _SERVICE


def _unquote_dotenv_value(value: str) -> str:
    """.env の値をクォート解除する（REQ-5.8）。

    両端が**対応する同じクォート**で囲まれているときだけ外側 1 組を外す。
    旧実装の `val.strip().strip('"').strip("'")` はクォート文字を無条件に全部剥がすため、
    クォートしていない値の両端の `"` / `'` まで壊していた（`"abc'` はそのまま、
    `""x""` は `"x"` になる）。
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def _load_dotenv() -> None:
    """exe/スクリプトの隣、または作業ディレクトリの `.env` を環境変数へ読み込む。

    内蔵プロバイダーぶんのキーをダイアログで登録する代わりに `.env` 一枚で済ませられる。
    既に設定済みの環境変数は上書きしない。値の `"..."` / `'...'` は外す。

    REQ-D6: 取り込むのは**アプリが実際に参照するキーだけ**（許可リスト）。
    これが無いと、敵対的フォルダの `.env` が REQUESTS_CA_BUNDLE / SSL_CERT_FILE を
    設定して TLS 検証を無効化できてしまう。許可リストは鍵解決の定義
    (`_ENV_ALIASES`) と `PIXAI_VLM_*` の慣習名から生成する（別名が増えても自動追随）。

    REQ-D7: BOM 付きの `.env`（メモ帳等で保存）でも先頭行のキーが無効にならないよう
    `utf-8-sig` で読む。
    """
    try:
        from constants import BASE_DIR
        candidates = [BASE_DIR / ".env", os.path.join(os.getcwd(), ".env")]
    except Exception:  # pragma: no cover
        candidates = [os.path.join(os.getcwd(), ".env")]
    seen: set[str] = set()
    for path in candidates:
        p = str(path)
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        try:
            with open(p, "r", encoding="utf-8-sig") as f:
                lines = f.readlines()
        except OSError as e:  # pragma: no cover
            write_debug_log(f"vlm_secrets: cannot read {p}: {e}")
            continue
        skipped = 0
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:]
            key, _, val = line.partition("=")
            key = key.strip()
            val = _unquote_dotenv_value(val.strip())
            if not key:
                continue
            if not _dotenv_key_allowed(key):
                skipped += 1
                continue
            if key not in os.environ:
                os.environ[key] = val
        if skipped:
            write_debug_log(
                f"vlm_secrets: ignored {skipped} .env key(s) the app does not read "
                f"({os.path.basename(p)})")


# secret_ref -> 環境変数名 の対応（既知の内蔵接続用）。未知の ref は
# "PIXAI_VLM_" + ref を英大文字化した名前も試す。
_ENV_ALIASES = {
    "vlm/gemini/api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "vlm/openrouter/api_key": ("OPENROUTER_API_KEY",),
    "vlm/cloudflare/api_token": ("CLOUDFLARE_API_TOKEN", "CF_API_TOKEN"),
    "vlm/groq/api_key": ("GROQ_API_KEY",),
    "vlm/nvidia/api_key": ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
    "vlm/mistral/api_key": ("MISTRAL_API_KEY",),
    "vlm/huggingface/api_token": ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    "vlm/vercel/api_key": ("AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"),
    "vlm/openai/api_key": ("OPENAI_API_KEY",),
    "vlm/anthropic/api_key": ("ANTHROPIC_API_KEY",),
    # xai だけ慣習名が漏れていた（R18-5）。vlm_connections の secret_ref・設定
    # ダイアログの _PROVIDER_KEY_INFO・プロファイルエディタの4箇所でプロバイダ集合を
    # 列挙しており、ここだけが欠けていた。keyring 経路は secret_ref 直接なので正常。
    "vlm/xai/api_key": ("XAI_API_KEY",),
    # "vlm/ovhcloud/api_key": ("OVH_AI_ENDPOINTS_ACCESS_TOKEN", "OVHCLOUD_API_KEY"),
}

# 秘密値ではないが、アプリが直接 os.environ から読む設定（vlm_config.build_connection_map）。
# .env 利用者の既存運用を壊さないよう、これらも取り込み許可リストに入れる。
_DOTENV_NON_SECRET_KEYS: frozenset[str] = frozenset(
    {"CLOUDFLARE_ACCOUNT_ID", "ANTHROPIC_WORKSPACE_ID"})

# `_env_candidates()` が生成する慣習名の接頭辞。すべての secret_ref は `vlm/...`
# なので generic 名は `PIXAI_VLM_*` になる（内蔵接続もカスタム `vlm/custom/<id>` も同じ）。
_DOTENV_GENERIC_PREFIX = "PIXAI_VLM_"


def _dotenv_allowed_keys() -> frozenset[str]:
    """`.env` から取り込んでよいキー名（REQ-D6）。

    固定リストとして二重管理せず、鍵解決が参照する定義（`_ENV_ALIASES` の値）と、
    非秘密の参照キーから生成する。REQ-F8 で別名が増えても自動で追随する。
    """
    return frozenset(
        {name for names in _ENV_ALIASES.values() for name in names}
        | _DOTENV_NON_SECRET_KEYS)


def _dotenv_key_allowed(key: str) -> bool:
    """許可リスト本体。`PIXAI_VLM_*`（`_env_candidates()` の慣習名）は接頭辞で許可する。"""
    return key in _dotenv_allowed_keys() or key.startswith(_DOTENV_GENERIC_PREFIX)


# `_ENV_ALIASES` と許可リストの定義後に読み込む（許可リストを参照するため）。
_load_dotenv()

_lock = threading.Lock()
_session_store: dict[str, str] = {}


def _env_candidates(secret_ref: str) -> list[str]:
    names = list(_ENV_ALIASES.get(secret_ref, ()))
    generic = "PIXAI_" + secret_ref.replace("/", "_").replace("-", "_").upper()
    if generic not in names:
        names.append(generic)
    return names


def get_secret(secret_ref: str) -> str | None:
    """secret_ref に対応する秘密値を返す。見つからなければ None。"""
    if not secret_ref:
        return None
    # A value explicitly entered during this session is an intentional override of
    # keyring/.env values (for example when the user replaces an environment key
    # without checking persistent storage). It is still session-only after restart.
    with _lock:
        session_value = _session_store.get(secret_ref)
    if session_value:
        return session_value
    # 1. keyring
    if keyring is not None:
        try:
            v = keyring.get_password(_SERVICE, secret_ref)
            if v:
                return v
        except Exception as e:  # backend 無し等
            write_debug_log(f"vlm_secrets: keyring get failed for a ref: {type(e).__name__}")
        try:
            v = keyring.get_password(_LEGACY_SERVICE, secret_ref)
            if v:
                return v
        except Exception as e:
            write_debug_log(f"vlm_secrets: keyring get (legacy) failed for a ref: {type(e).__name__}")
    # 2. 環境変数
    for name in _env_candidates(secret_ref):
        v = os.environ.get(name)
        if v:
            return v
    return None


def set_secret(secret_ref: str, value: str, *, persist: bool) -> bool:
    """秘密値を保存する。persist=True なら keyring、False ならセッションのみ。

    戻り値は「keyring へ永続保存できたか」。persist=False のとき、および keyring 保存に
    失敗してセッション保持へフォールバックしたときは False。
    """
    if not secret_ref:
        return False
    if persist and keyring is not None:
        try:
            keyring.set_password(_service_for(secret_ref), secret_ref, value)
            with _lock:
                _session_store.pop(secret_ref, None)
            return True
        except Exception as e:
            write_debug_log(f"vlm_secrets: keyring set failed, falling back to session: {type(e).__name__}")
    with _lock:
        _session_store[secret_ref] = value
    return False


def _looks_like_missing_entry(exc: BaseException) -> bool:
    """「エントリが無い」ことを表す例外か（backend ごとに型も文言も違う）。"""
    if type(exc).__name__ in {"PasswordDeleteError", "PasswordGetError"}:
        return True
    text = str(exc).lower()
    return "not found" in text or "no such" in text or "not exist" in text


def delete_secret(secret_ref: str) -> bool:
    """秘密値を削除する。keyring から実際に消えていれば True。

    persist=False へ切り替えるときに、古い keyring 値が残っていないかを呼び出し側が
    確認できるよう bool を返す（残っていると再起動後に get_secret がそれを拾う）。
    """
    if not secret_ref:
        return True
    removed = True
    if keyring is not None:
        unconfirmed_delete = False
        for service in (_SERVICE, _LEGACY_SERVICE):
            try:
                keyring.delete_password(service, secret_ref)
            except Exception as e:  # backend 無し / エントリ未登録 / ロック中など
                write_debug_log(f"vlm_secrets: keyring delete failed for a ref: {type(e).__name__}")
                if not _looks_like_missing_entry(e):
                    # 「無い」ではなく「消せたか分からない」例外（KeyringLocked 等）。
                    # この状態ではキーが残っている可能性がある（PR#38 レビュー指摘）。
                    unconfirmed_delete = True
        # 実際に消えたかを両サービス名で確認する。「値が残っていることを確認できた」
        # ときだけ removed = False とする（REQ-5.5）。エントリ未登録のときに例外を返す
        # バックエンドでは、読み直しが例外になるのが正常なので「確認できなかった」を
        # そのまま「残っている」と判定はしない。
        for service in (_SERVICE, _LEGACY_SERVICE):
            try:
                if keyring.get_password(service, secret_ref):
                    removed = False
            except Exception as e:
                if unconfirmed_delete:
                    # 削除自体も確認できていない → 消えたとは言えないので失敗として返す。
                    removed = False
                    write_debug_log(
                        f"vlm_secrets: delete and verification both failed "
                        f"({type(e).__name__}); reporting it as not removed"
                    )
                else:
                    write_debug_log(
                        f"vlm_secrets: could not verify deletion ({type(e).__name__}); "
                        f"treating it as removed"
                    )
    with _lock:
        _session_store.pop(secret_ref, None)
    return removed


def secret_status(secret_ref: str) -> str:
    """UI 表示用: 秘密値の在り処。'keyring' / 'env' / 'session' / 'missing'。"""
    if not secret_ref:
        return "missing"
    with _lock:
        if secret_ref in _session_store and _session_store[secret_ref]:
            return "session"
    if keyring is not None:
        for service in (_SERVICE, _LEGACY_SERVICE):
            try:
                if keyring.get_password(service, secret_ref):
                    return "keyring"
            except Exception:
                pass
    for name in _env_candidates(secret_ref):
        if os.environ.get(name):
            return "env"
    return "missing"


_keyring_probe: bool | None = None


def persist_choice_from_checkbox(checked: bool, enabled: bool) -> tuple[bool, bool]:
    """「永続保存」チェックボックスの状態を `(persist, drop_existing)` に解釈する。

    REQ-D8(b): keyring プローブが失敗すると、ダイアログはチェックボックスを
    **未チェックかつ無効**にする。その False を「利用者がセッション限定へ切替えた」
    と解釈して delete_secret を呼ぶと、後で keyring が回復した環境で既存の永続キーが
    実際に削除される。無効化されている場合は利用者の意思ではないので、削除分岐へ入らない。
    """
    if not enabled:
        return checked, False
    return checked, not checked


def keyring_available() -> bool:
    """OS の秘密情報ストレージが「実際に使えるか」。

    `import keyring` が通っても、バックエンドが無い環境では
    `keyring.backends.fail.Keyring`（priority<=0）が返り、`set_password` は例外になる。
    モジュールの有無ではなく、使えるバックエンドがあるかを見る。

    REQ-D8(a): **失敗を固定しない**。成功だけをキャッシュし、失敗は次回の呼び出しで
    再プローブする。初回の一時的な失敗（バックエンドの起動遅れ、ロック等）が
    プロセス中ずっと「永続保存不可」として残り、回復後も保存できなくなるのを避ける
    （非正のキャッシュは保存操作の時点の再試行を潰す）。
    """
    global _keyring_probe
    if _keyring_probe:
        return True
    if keyring is None:
        return False
    try:
        backend = keyring.get_keyring()
        module = (type(backend).__module__ or "").lower()
        priority = float(getattr(backend, "priority", 1) or 0)
        available = ("fail" not in module) and priority > 0
    except Exception as e:  # pragma: no cover - 環境依存
        write_debug_log(f"vlm_secrets: keyring backend probe failed: {type(e).__name__}")
        return False
    if available:
        _keyring_probe = True
    return available
