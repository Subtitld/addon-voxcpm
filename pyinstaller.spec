# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Subtitld VoxCPM add-on.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Produces `dist/voxcpm-addon/` (launcher + `_internal/`): the add-on's own
code on CPython's standard library, and audio.cpp in `_internal/audiocpp/`.
The model is not in it; it downloads on the first synthesis. The release step
copies the folder to `bin/` inside the archive, next to manifest.json.

Run from the add-on root:

    python tools/build_native.py      # Linux: tools/build_native_linux.sh
    pyinstaller pyinstaller.spec --noconfirm
    dist/voxcpm-addon/voxcpm-addon --self-test
"""

# ruff: noqa: F821  # PyInstaller injects Analysis/PYZ/EXE/COLLECT at runtime.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SPEC_ROOT = Path(SPECPATH).resolve()
ASSETS = SPEC_ROOT / 'build' / 'assets'
PACKAGE = SPEC_ROOT / 'voxcpm_addon'
_PLATFORM = {'win32': 'windows', 'darwin': 'macos'}.get(sys.platform, 'linux')

# Fail the build, loudly, if the runtime is missing. Otherwise the frozen
# add-on would answer the handshake and then fail the first request.
try:
    _NATIVE = json.loads((ASSETS / 'native.json').read_text(encoding='utf-8'))
except OSError:
    raise SystemExit(f'FATAL: no {ASSETS / "native.json"}. Run `python tools/build_native.py` first.')
if _NATIVE['platform'] != _PLATFORM:
    raise SystemExit(f'FATAL: build/assets holds the {_NATIVE["platform"]} runtime, not {_PLATFORM}\'s')
_FILES = [*_NATIVE['files'], 'native.json']
_MISSING = [name for name in _FILES if not (ASSETS / name).is_file()]
if _MISSING:
    raise SystemExit(f'FATAL: build/assets lacks {", ".join(_MISSING)}. Run `python tools/build_native.py` again.')


# ---------------------------------------------------------------------------
# libexpat sanity check (from addon-whispercpp): a stale expat on the build
# host makes the frozen binary die before Python starts.
# ---------------------------------------------------------------------------
_EXPAT_SYMBOL = 'XML_SetReparseDeferralEnabled'


def _exports_symbol(path: str, symbol: str) -> bool:
    try:
        out = subprocess.run(['nm', '-D', '--defined-only', path],
                             capture_output=True, text=True, timeout=30)
        return symbol in out.stdout
    except Exception:
        return True  # no nm (macOS/Windows): assume fine rather than break the build


def _fix_libexpat(entries):
    if not sys.platform.startswith('linux'):
        return entries
    out = []
    for entry in entries:
        name = entry[0] if entry else ''
        src = entry[1] if len(entry) > 1 else ''
        if not os.path.basename(name).startswith('libexpat.so') or _exports_symbol(src, _EXPAT_SYMBOL):
            out.append(entry)
            continue
        for cand in ('/usr/lib/x86_64-linux-gnu/libexpat.so.1', '/usr/lib64/libexpat.so.1',
                     '/lib/x86_64-linux-gnu/libexpat.so.1'):
            if os.path.exists(cand) and _exports_symbol(cand, _EXPAT_SYMBOL):
                print(f'WARNING: {src} lacks {_EXPAT_SYMBOL}; substituting {cand}')
                out.append((name, cand) + tuple(entry[2:]))
                break
        else:
            print(f'ERROR: bundled libexpat lacks {_EXPAT_SYMBOL}; the frozen binary will not start.')
            out.append(entry)
    return out


a = Analysis(
    [str(PACKAGE / '__main__.py')],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PySide6', 'PyQt6', 'PyQt5', 'tkinter', 'Tkinter', '_tkinter',
        'numpy', 'torch', 'torchaudio', 'onnxruntime', 'setuptools', 'pkg_resources', 'unittest', 'pydoc_data',
        'pytest', 'huggingface_hub', 'requests', 'urllib3', 'certifi',
        # Standard modules the add-on never calls, reached only through
        # optional imports (shutil's archive formats, random's hash fallback,
        # uuid). Each would bring a native library, and with it a licence to
        # ship: bzip2, xz, zstd, libmpdec, libuuid, SQLite. ssl and hashlib
        # stay: the model downloads over HTTPS and is checked by sha256, so
        # OpenSSL ships. tests/test_installed.py runs the frozen binary end to
        # end, so a module that turns out to be needed fails the release
        # build, not the user.
        'bz2', '_bz2', 'lzma', '_lzma', 'compression', '_zstd',
        'decimal', '_decimal', '_pydecimal', '_uuid', 'sqlite3', '_sqlite3',
        # GPL (readline) and ncurses: for interactive terminals, which the
        # add-on never has.
        'readline', 'curses', '_curses', '_curses_panel',
    ],
    noarchive=False,
)

a.binaries = _fix_libexpat(a.binaries)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='voxcpm-addon',
    debug=False,
    bootloader_ignore_signals=False,
    # strip=False: stripping the native libraries saves little and breaks
    # macOS code signatures.
    strip=False,
    upx=False,
    console=True,  # stdio is the protocol channel
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='voxcpm-addon',
)

# audio.cpp is loaded by the add-on with ctypes from a path it knows, so it is
# copied in after COLLECT rather than handed to PyInstaller. Given to
# PyInstaller, even as data, the libraries are reclassified as binaries, their
# dependencies are collected beside them and on macOS they are re-signed.
# Copied here, they stay byte for byte as built, each finding the others in
# its own folder.
_out = Path(DISTPATH) / 'voxcpm-addon' / '_internal'
if (_out / 'audiocpp').exists():
    shutil.rmtree(_out / 'audiocpp')
for _name in _FILES:
    (_out / _name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ASSETS / _name, _out / _name)
