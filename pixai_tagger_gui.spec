# -*- mode: python ; coding: utf-8 -*-

import ast
import glob
import os
import re

_project_root = os.path.abspath(SPECPATH)
_source_dir = os.path.join(_project_root, 'src')

# Ship every model's hand-authored model_config.json (NOT the multi-GB model.onnx files,
# which the app downloads at runtime) plus PixAI's curated tag-translation CSVs, keeping
# the models/<model_id>/ directory structure. constants._seed_bundled_model_files() copies
# them out of _internal/ into the user-visible models/ folder on first launch.
_model_datas = [
    (p, os.path.relpath(os.path.dirname(p), _project_root))
    for p in glob.glob(os.path.join(_project_root, 'models', '*', 'model_config.json'))
]


def _translation_suffixes():
    """The languages tag_utils actually loads, read straight from the source so a new
    language never silently misses the build. Falls back to the current list."""
    fallback = ["jp", "fr", "de", "es", "ru", "zh_CN", "zh_TW", "ko"]
    try:
        src = open(os.path.join(_source_dir, 'tag_utils.py'), encoding='utf-8').read()
        match = re.search(r'_TRANSLATION_LANGUAGE_SUFFIXES\s*=\s*(\[[^\]]*\])', src)
        return ast.literal_eval(match.group(1)) if match else fallback
    except Exception:
        return fallback


# Only the 8 hand-curated translation CSVs. A bare `selected_tags*.csv` glob would also
# sweep up selected_tags.csv (downloaded at runtime) and selected_tags_en.csv (redundant),
# both gitignored - that would make the build depend on the developer's local downloads.
_pixai_dir = os.path.join(_project_root, 'models', 'pixai-tagger-v0.9')
_model_datas += [
    (os.path.join(_pixai_dir, f'selected_tags_{suffix}.csv'),
     os.path.relpath(_pixai_dir, _project_root))
    for suffix in _translation_suffixes()
    if os.path.isfile(os.path.join(_pixai_dir, f'selected_tags_{suffix}.csv'))
]


# The runtime GPU-component manifest, if a release build has generated it
# (tools/gen_gpu_components.py). Lands at _internal/gpu_components.json, which
# constants.RESOURCE_DIR points at. Absent in dev/source builds -> no GPU prompt.
_gpu_components = os.path.join(_project_root, 'gpu_components.json')
_gpu_datas = [(_gpu_components, '.')] if os.path.isfile(_gpu_components) else []

a = Analysis(
    [os.path.join(_source_dir, 'pixai_tagger_gui.py')],
    pathex=[_source_dir],
    binaries=[],
    datas=[(os.path.join(_project_root, 'icons'), 'icons'),
           (os.path.join(_project_root, 'lang'), 'lang')] + _model_datas + _gpu_datas,
    hiddenimports=[
        'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets',
        # keyring はバックエンドを entry point 経由で探すため、凍結ビルドでは
        # 明示しないと 1つも見つからず、API キーが毎回セッション保持に落ちる。
        'keyring.backends.Windows',
        'keyring.backends.SecretService',
        'keyring.backends.kwallet',
        'keyring.backends.chainer',
        'keyring.backends.fail',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6.QtWebEngineCore', 'PySide6.QtMultimedia', 'PySide6.QtCharts', 'PySide6.QtSql', 'PySide6.QtPrintSupport', 'QtWebEngineCore', 'QtSql', 'QtNetwork', 'QtTest', 'tkinter'],
    noarchive=False,
    optimize=0,
)
# onnxruntime-gpu ships onnxruntime_providers_cuda.dll (~200MB+). GPU support is
# opt-in: the app downloads that DLL plus the NVIDIA runtime DLLs at runtime into
# gpu_runtime/ (docs/260910_gpu_acceleration_impl_plan.md). Strip it here so the
# distributed zip stays small; onnxruntime_providers_shared.dll (tiny) is kept.
_stripped_cuda_provider = any(
    os.path.basename(b[0]).lower() == 'onnxruntime_providers_cuda.dll' for b in a.binaries)
a.binaries = [b for b in a.binaries
              if os.path.basename(b[0]).lower() != 'onnxruntime_providers_cuda.dll']
if _stripped_cuda_provider and not _gpu_datas:
    # The strip above and bundling gpu_components.json are two independent steps;
    # forgetting `tools/gen_gpu_components.py` before this build would otherwise
    # silently ship a release where GPU acceleration can never be enabled - a
    # print()-only warning is easy to miss in build output, and a completed build
    # with exit code 0 reads as success (CodeRabbit review, PR #21). Fail the
    # build instead, so a mis-rolled release is caught here, not by a user.
    raise SystemExit(
        'pixai_tagger_gui.spec: onnxruntime_providers_cuda.dll was stripped from '
        'this build but gpu_components.json is missing, which would permanently '
        'disable GPU acceleration in this build. Run tools/gen_gpu_components.py '
        'first, or delete onnxruntime-gpu from this environment if a CPU-only '
        'build is actually what you want.')

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pixai_tagger_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[os.path.join(_project_root, 'icons', 'app_icon.ico')],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pixai_tagger_gui',
)
