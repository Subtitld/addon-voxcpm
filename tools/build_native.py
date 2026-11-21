"""Build audio.cpp's C library for VoxCPM2 and put it into build/assets/audiocpp/.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The add-on calls audio.cpp's C ABI (include/audiocpp.h) through ctypes. This
script checks out 0xShug0/audio.cpp at a pinned commit, builds its C library
with only the VoxCPM2 family compiled in, and collects:

- build/assets/audiocpp/: the C library. On Linux and Windows, ggml beside it
  (libggml.so.0 and libggml-base.so.0; ggml.dll and ggml-base.dll) and one
  ggml CPU backend per x86-64 instruction set (libggml-cpu-*.so,
  ggml-cpu-*.dll), of which ggml loads the best the processor runs; each
  library finds the others in its own folder. On macOS one library,
  libaudiocpp.0.dylib, with ggml, its CPU backend for Apple Silicon and its
  Metal backend (the shaders embedded) linked in.
- build/assets/audiocpp/audiocpp.h: the header the ctypes bindings follow.
- build/assets/native.json: where it all came from, and each file's SHA-256.
- build/licences/audiocpp/: the licences of what the libraries compile in.

One patch is applied first, tools/patches/audiocpp-step-callback.patch: a
callback audio.cpp calls before each generated step, which reports progress
and stops a line between two steps (upstream's offline run cannot be stopped,
and its streaming run returns only when the line is done). The patch travels
with the licences, as the record of what was changed.

The model specs are compiled into the library (AUDIOCPP_DEPLOYMENT_BUILD), so
no data files travel with it. ggml runs on its own thread pool, without
OpenMP: as fast as with it on an i7-13700T, and no libgomp to ship.

Linux builds run in the AlmaLinux 8 container (tools/build_native_linux.sh),
so the libraries load on glibc 2.28. macOS builds are for Apple Silicon and
macOS 13.3 or newer, like the other add-ons. Windows builds need Visual
Studio's C++ tools, and link Microsoft's C runtime statically, so no
runtime DLL has to be on the user's machine.

    python tools/build_native.py [--jobs N]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / 'build'
ASSETS = BUILD / 'assets'
RUNTIME = ASSETS / 'audiocpp'
LICENCES = BUILD / 'licences' / 'audiocpp'
SRC = BUILD / 'audiocpp-src'
NATIVE_BUILD = BUILD / 'native'
PATCHES = [ROOT / 'tools' / 'patches' / 'audiocpp-step-callback.patch']

REPO = 'https://github.com/0xShug0/audio.cpp'
VERSION = '0.9.5'
COMMIT = '61c7838baf1f7f34769b5b7bd369c0d78edf9535'  # tag v0.9.5, 2026-11-14

CMAKE_OPTIONS = [
    '-DAUDIOCPP_MODEL_SET=custom',
    '-DAUDIOCPP_MODELS=voxcpm2',
    '-DAUDIOCPP_BUILD_C_API=ON',
    '-DAUDIOCPP_DEPLOYMENT_BUILD=ON',
    f'-DAUDIOCPP_VERSION={VERSION}',
    '-DENGINE_ENABLE_NATIVE_CPU=OFF',
    '-DENGINE_ENABLE_OPENMP=OFF',
    '-DGGML_OPENMP=OFF',
]
LINUX_OPTIONS = ['-DENGINE_ENABLE_CPU_ALL_VARIANTS=ON']
WINDOWS_OPTIONS = ['-A', 'x64', '-DENGINE_ENABLE_CPU_ALL_VARIANTS=ON',
                   '-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded', '-DCMAKE_POLICY_DEFAULT_CMP0091=NEW']
MACOS_OPTIONS = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_OSX_ARCHITECTURES=arm64',
                 '-DCMAKE_OSX_DEPLOYMENT_TARGET=13.3', '-DENGINE_ENABLE_METAL=ON',
                 '-DGGML_METAL_EMBED_LIBRARY=ON']
MACOS_MIN = (13, 3)

# The file the add-on loads, by platform.
ENTRY = {'linux': 'libaudiocpp.so.0', 'macos': 'libaudiocpp.0.dylib', 'windows': 'audiocpp.dll'}

# The library files to ship, by the name they are loaded under (their SONAME).
LINUX_LIBRARIES = {
    'libaudiocpp.so.0': 'libaudiocpp.so.0.1.0',
    'libggml.so.0': 'libggml.so.0.12.0',
    'libggml-base.so.0': 'libggml-base.so.0.12.0',
}

# Libraries every Linux desktop has; the build may link nothing else dynamically.
LINUX_SYSTEM = {
    'libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0', 'librt.so.1',
    'ld-linux-x86-64.so.2', 'libstdc++.so.6', 'libgcc_s.so.1',
}
# The newest symbol version each AlmaLinux 8 (glibc 2.28, GCC 8) library provides.
LINUX_CEILINGS = {'GLIBC': (2, 28), 'GLIBCXX': (3, 4, 25), 'CXXABI': (1, 3, 11), 'GCC': (8, 0, 0)}

# Third-party code the libraries may compile in: a symbol that marks it, and
# its licence files in the audio.cpp tree. A vendored library whose symbols show
# up without an entry here stops the build (see check_vendored).
VENDORED = {
    'ggml': (None, ['external/ggml/LICENSE']),
    'cJSON': (r'\bcJSON_', ['external/cJSON/LICENSE']),
    'libyaml': (r'\byaml_', ['external/libyaml/License']),
}
UNEXPECTED = {
    'sentencepiece': r'sentencepiece::',
    'protobuf-lite': r'google::protobuf::',
    'utf8proc': r'\butf8proc_',
    'lz4': r'\bLZ4_',
    'cpp-httplib': r'httplib::',
    'llama tokenizer': r'\b(unicode_cpt|llm_tokenizer)',
    'eSpeak NG (GPL)': r'\bespeak_',
}


def run(cmd, cwd=None, env=None):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def output(cmd, cwd=None):
    env = dict(os.environ, LC_ALL='C')
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True,
                          capture_output=True, text=True).stdout


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fresh(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


GIT = ['git', '-c', 'core.autocrlf=false', '-c', 'core.eol=lf']


def checkout():
    """The pinned commit, shallow, with PATCHES applied. Its one submodule (web front ends) isn't needed."""
    if not (SRC / '.git').exists():
        SRC.mkdir(parents=True, exist_ok=True)
        run([*GIT, 'init', '-q'], cwd=SRC)
        run([*GIT, 'remote', 'add', 'origin', REPO + '.git'], cwd=SRC)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SRC, capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        run([*GIT, 'fetch', '-q', '--depth', '1', 'origin', COMMIT], cwd=SRC)
    run([*GIT, 'checkout', '-q', '--force', COMMIT], cwd=SRC)
    run([*GIT, 'clean', '-q', '-fd'], cwd=SRC)
    for patch in PATCHES:
        run([*GIT, 'apply', '--whitespace=nowarn', patch], cwd=SRC)


def build(platform, jobs):
    configure = ['cmake', '-S', SRC, '-B', NATIVE_BUILD, *CMAKE_OPTIONS]
    configure += {'linux': ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', *LINUX_OPTIONS],
                  'windows': WINDOWS_OPTIONS, 'macos': MACOS_OPTIONS}[platform]
    run(configure)
    run(['cmake', '--build', NATIVE_BUILD, '--config', 'Release', '--parallel', str(jobs), '--target', 'audiocpp'])
    return NATIVE_BUILD


def built_files(built, pattern):
    """Files under the build tree matching `pattern`, outside CMake's own folders."""
    return sorted(p for p in built.rglob(pattern)
                  if p.is_file() and not p.is_symlink() and 'CMakeFiles' not in p.parts)


def collect_linux(built):
    built = built / 'bin'
    fresh(RUNTIME)
    for name, real in LINUX_LIBRARIES.items():
        if not (built / real).is_file():
            sys.exit(f'FATAL: the build made no {real} in {built}')
        shutil.copyfile(built / real, RUNTIME / name)
    variants = sorted(built.glob('libggml-cpu-*.so'))
    if not variants:
        sys.exit(f'FATAL: no ggml CPU backends in {built}')
    for variant in variants:
        shutil.copyfile(variant, RUNTIME / variant.name)
    shutil.copyfile(SRC / 'include' / 'audiocpp.h', RUNTIME / 'audiocpp.h')


def collect_macos(built):
    """libaudiocpp.0.dylib: ggml is linked in, so it is the only library."""
    fresh(RUNTIME)
    found = built_files(built, 'libaudiocpp*.dylib')
    if len(found) != 1:
        sys.exit(f'FATAL: expected one libaudiocpp dylib in {built}, found {[str(p) for p in found]}')
    shutil.copyfile(found[0], RUNTIME / ENTRY['macos'])
    stray = built_files(built, 'libggml*.dylib')
    if stray:
        sys.exit(f'FATAL: ggml was built as separate libraries ({stray[0].name}); it should be linked in')
    shutil.copyfile(SRC / 'include' / 'audiocpp.h', RUNTIME / 'audiocpp.h')


def collect_windows(built):
    """audiocpp.dll, ggml.dll, ggml-base.dll and the ggml-cpu-*.dll backends."""
    fresh(RUNTIME)
    names = ['audiocpp.dll', 'ggml.dll', 'ggml-base.dll']
    for name in names:
        found = built_files(built, name)
        if len(found) != 1:
            sys.exit(f'FATAL: expected one {name} in {built}, found {[str(p) for p in found]}')
        shutil.copyfile(found[0], RUNTIME / name)
    variants = built_files(built, 'ggml-cpu-*.dll')
    if not variants:
        sys.exit(f'FATAL: no ggml CPU backends in {built}')
    for variant in variants:
        shutil.copyfile(variant, RUNTIME / variant.name)
    shutil.copyfile(SRC / 'include' / 'audiocpp.h', RUNTIME / 'audiocpp.h')


# ---- Linux: refuse anything AlmaLinux 8 couldn't load ---------------------

def needed(lib):
    return re.findall(r'\(NEEDED\)\s+Shared library: \[([^\]]+)\]', output(['readelf', '-d', lib]))


def version_needs(lib):
    needs, current = {}, None
    for line in output(['readelf', '-V', '--wide', lib]).splitlines():
        m = re.search(r'File: (\S+)', line)
        if m:
            current = needs.setdefault(m.group(1), [])
            continue
        m = re.search(r'Name: (\S+)', line)
        if m and current is not None:
            current.append(m.group(1))
    return needs


def check_linux():
    shipped = {p.name for p in RUNTIME.glob('*.so*')}
    problems = []
    for lib in sorted(RUNTIME.glob('*.so*')):
        problems += [f'{lib.name} needs {dep}, which is neither shipped nor a system library'
                     for dep in needed(lib) if dep not in LINUX_SYSTEM | shipped]
        for dep, versions in version_needs(lib).items():
            for version in versions:
                m = re.fullmatch(r'([A-Z]+)_([0-9.]+)', version)
                if not m or m.group(1) not in LINUX_CEILINGS:
                    continue
                if tuple(int(x) for x in m.group(2).split('.')) > LINUX_CEILINGS[m.group(1)]:
                    problems.append(f'{lib.name} needs {version} from {dep}')
        runpath = re.findall(r'\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]+)\]',
                             output(['readelf', '-d', lib]))
        if runpath != ['$ORIGIN']:
            problems.append(f'{lib.name} has runpath {runpath}, not $ORIGIN')
    exported = [line.split()[-1] for line in output(['nm', '-D', '--defined-only', RUNTIME / 'libaudiocpp.so.0']).splitlines()
                if ' T ' in line and not line.split()[-1].startswith('audiocpp_')]
    if exported:
        problems.append(f'libaudiocpp.so.0 exports more than audiocpp_*: {exported[:5]}')
    if problems:
        sys.exit('not loadable on glibc 2.28 / GCC 8:\n  ' + '\n  '.join(sorted(set(problems))))
    print('the libraries load on glibc 2.28 with the GCC 8 runtime, from one folder; libaudiocpp exports only audiocpp_*')


# ---- macOS: refuse anything a Mac with macOS 13.3 couldn't load -----------

def check_macos():
    lib = RUNTIME / ENTRY['macos']
    problems = []
    for line in output(['otool', '-L', lib]).splitlines()[1:]:
        dep = line.strip().split(' (', 1)[0]
        if dep.startswith(('/usr/lib/', '/System/Library/')) or dep.endswith('/' + ENTRY['macos']):
            continue
        problems.append(f'{lib.name} needs {dep}, which is neither shipped nor part of macOS')
    minos = re.findall(r'^\s*minos (\d+)\.(\d+)', output(['otool', '-l', lib]), re.M)
    if not minos:
        problems.append(f'{lib.name} carries no minimum macOS version')
    for major, minor in minos:
        if (int(major), int(minor)) > MACOS_MIN:
            problems.append(f'{lib.name} needs macOS {major}.{minor}')
    exported = [line.split()[-1] for line in output(['nm', '-gU', lib]).splitlines()
                if line.split() and not line.split()[-1].startswith('_audiocpp_')]
    if exported:
        problems.append(f'{lib.name} exports more than audiocpp_*: {exported[:5]}')
    if 'arm64' not in output(['lipo', '-archs', lib]):
        problems.append(f'{lib.name} is not built for Apple Silicon')
    if problems:
        sys.exit('not loadable on macOS 13.3:\n  ' + '\n  '.join(problems))
    print(f'{lib.name} loads on macOS {MACOS_MIN[0]}.{MACOS_MIN[1]} and newer, on Apple Silicon; it exports only audiocpp_*')


# ---- Windows: only system DLLs, no C runtime DLL --------------------------

def _pe_imports(path):
    import pefile  # PyInstaller's own dependency on Windows
    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories([pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    return [entry.dll.decode() for entry in getattr(pe, 'DIRECTORY_ENTRY_IMPORT', [])]


def check_windows():
    shipped = {p.name.lower() for p in RUNTIME.glob('*.dll')}
    problems = []
    for lib in sorted(RUNTIME.glob('*.dll')):
        for dep in _pe_imports(lib):
            low = dep.lower()
            if low in shipped:
                continue
            if re.match(r'(msvcp|vcruntime|vcomp|concrt)\d+', low) or low.startswith('api-ms-win-crt-'):
                problems.append(f'{lib.name} needs {dep}: the C runtime should be linked in statically')
            elif not (low.startswith('api-ms-win-') or low in WINDOWS_SYSTEM):
                problems.append(f'{lib.name} needs {dep}, which is neither shipped nor a system DLL')
    if problems:
        sys.exit('not loadable on a bare Windows:\n  ' + '\n  '.join(sorted(set(problems))))
    print('the DLLs need only Windows system DLLs, and find each other in their own folder')


WINDOWS_SYSTEM = {'kernel32.dll', 'user32.dll', 'advapi32.dll', 'shell32.dll', 'ole32.dll', 'oleaut32.dll',
                  'ws2_32.dll', 'bcrypt.dll', 'ntdll.dll', 'shlwapi.dll', 'dbghelp.dll', 'psapi.dll',
                  'version.dll', 'winmm.dll', 'crypt32.dll', 'secur32.dll', 'userenv.dll'}


def check_vendored(platform):
    """Which third-party code is compiled in, from the library's own symbols.

    A DLL keeps no symbols but its exports, so on Windows every library in
    VENDORED is taken as present; the Linux and macOS builds check the same
    sources.
    """
    if platform == 'windows':
        present = list(VENDORED)
        print('compiled in (not checked on Windows):', ', '.join(['audio.cpp', *present]))
        return present
    symbols = output(['nm', '-C', RUNTIME / ENTRY[platform]])
    found = [name for name, pattern in UNEXPECTED.items() if re.search(pattern, symbols)]
    if found:
        sys.exit(f'FATAL: {ENTRY[platform]} now compiles in {", ".join(found)}: add its licence to VENDORED')
    present = [name for name, (pattern, _files) in VENDORED.items() if pattern is None or re.search(pattern, symbols)]
    print('compiled in:', ', '.join(['audio.cpp', *present]))
    return present


# ---- licences ---------------------------------------------------------------

def collect_licences(present):
    fresh(LICENCES)
    shutil.copyfile(SRC / 'LICENSE', LICENCES / 'LICENSE')
    for patch in PATCHES:
        shutil.copyfile(patch, LICENCES / patch.name)
    for name in present:
        folder = LICENCES / 'third_party' / name
        folder.mkdir(parents=True)
        for relative in VENDORED[name][1]:
            shutil.copyfile(SRC / relative, folder / Path(relative).name)


def main():
    host = {'linux': 'linux', 'win32': 'windows', 'darwin': 'macos'}[sys.platform]
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--jobs', type=int, default=int(os.environ.get('JOBS', os.cpu_count() or 4)))
    args = parser.parse_args()

    checkout()
    built = build(host, args.jobs)
    {'linux': collect_linux, 'macos': collect_macos, 'windows': collect_windows}[host](built)
    {'linux': check_linux, 'macos': check_macos, 'windows': check_windows}[host]()
    collect_licences(check_vendored(host))

    files = {p.relative_to(ASSETS).as_posix(): sha256(p) for p in sorted(ASSETS.rglob('*')) if p.is_file()}
    record = {'runtime': f'audio.cpp {VERSION} C API, VoxCPM2 only', 'repo': REPO, 'commit': COMMIT,
              'patches': {p.name: sha256(p) for p in PATCHES},
              'platform': host, 'entry': f'audiocpp/{ENTRY[host]}', 'files': files}
    (ASSETS / 'native.json').write_text(json.dumps(record, indent=1) + '\n')
    size = sum(p.stat().st_size for p in RUNTIME.iterdir()) / 2**20
    print(f'{len(files)} files in {ASSETS}, {size:.1f} MiB of runtime')


if __name__ == '__main__':
    main()
