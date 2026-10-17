#!/usr/bin/env python3
"""Gather THIRD_PARTY_LICENSES/ for a release archive from the build environment.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

The frozen bundle carries CPython, the PyInstaller bootloader and the Python
distributions installed in the build environment (PyTorch, VoxCPM, librosa,
NumPy...). Each distribution's licence files, as its wheel ships them, are
copied to THIRD_PARTY_LICENSES/<distribution>/; PyTorch's include the notices
of the libraries it bundles. A distribution that ships no licence file gets
its declared licence recorded instead, and the build fails if it declares
none either. Tools used only to build or test are left out.

    python tools/collect_licences.py --out release-staging/THIRD_PARTY_LICENSES
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, distribution, distributions
from pathlib import Path

BUILD_ONLY = {
    'pip', 'setuptools', 'wheel', 'uv', 'pyinstaller', 'pyinstaller-hooks-contrib', 'altgraph',
    'macholib', 'pefile', 'pywin32-ctypes', 'pytest', 'pluggy', 'iniconfig', 'pygments',
    'subtitld-addon-voxcpm',
}
_LICENCE_FILE = re.compile(r'(^|/)(licen[cs]e|copying|notice|authors|third[-_]party)[^/]*$', re.I)


def _normal(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def _licence_files(dist) -> list[tuple[Path, str]]:
    """(file, its path inside the .dist-info folder) for each licence file the wheel ships."""
    found = []
    for entry in dist.files or []:
        text = str(entry).replace('\\', '/')
        if '.dist-info/' in text and _LICENCE_FILE.search(text):
            path = Path(dist.locate_file(entry))
            if path.is_file():
                found.append((path, text.split('.dist-info/', 1)[1]))
    return found


def _declared(dist) -> str:
    meta = dist.metadata
    expression = meta.get('License-Expression') or ''
    if expression:
        return expression
    classifiers = [c.split('::')[-1].strip() for c in meta.get_all('Classifier') or []
                   if c.startswith('License ::')]
    text = (meta.get('License') or '').strip()
    if text and len(text) < 200:
        classifiers.insert(0, text)
    return '; '.join(classifiers)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    problems = []
    seen = set()
    for dist in sorted(distributions(), key=lambda d: _normal(d.metadata['Name'] or '')):
        name = _normal(dist.metadata['Name'] or '')
        if not name or name in BUILD_ONLY or name in seen:
            continue
        seen.add(name)
        target = out / name
        files = _licence_files(dist)
        declared = _declared(dist)
        target.mkdir(exist_ok=True)
        for path, relative in files:
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target / relative)
        if not files:
            if not declared:
                problems.append(name)
                continue
            (target / 'DECLARED-LICENSE.txt').write_text(
                f'{dist.metadata["Name"]} {dist.version} ships no licence file; its metadata declares: '
                f'{declared}\n', encoding='utf-8')
        rows.append((dist.metadata['Name'], dist.version, declared or 'see files', name))

    # CPython, which the bundle embeds, and the PyInstaller bootloader.
    stdlib = Path(sysconfig.get_paths()['stdlib'])
    for candidate in (Path(sys.base_prefix) / 'LICENSE.txt', stdlib / 'LICENSE.txt', Path(sys.base_prefix) / 'LICENSE'):
        if candidate.is_file():
            (out / 'python').mkdir(exist_ok=True)
            shutil.copyfile(candidate, out / 'python' / 'LICENSE.txt')
            rows.append(('CPython', sys.version.split()[0], 'PSF-2.0', 'python'))
            break
    else:
        problems.append('CPython (no LICENSE.txt next to the interpreter)')
    try:
        pyinstaller = [p for p, _rel in _licence_files(distribution('pyinstaller')) if p.name.upper().startswith('COPYING')]
    except PackageNotFoundError:
        pyinstaller = []
    if pyinstaller:
        (out / 'pyinstaller').mkdir(exist_ok=True)
        shutil.copyfile(pyinstaller[0], out / 'pyinstaller' / 'COPYING.txt')
        rows.append(('PyInstaller bootloader', distribution('pyinstaller').version,
                     'GPL-2.0-or-later WITH Bootloader-exception', 'pyinstaller'))
    else:
        problems.append('PyInstaller (no COPYING.txt)')

    if problems:
        print('FATAL: no licence found for: ' + ', '.join(problems), file=sys.stderr)
        return 1

    lines = ['# Third-party licences', '',
             'What this archive carries besides the add-on itself (Apache-2.0, LICENSE at the root).',
             'The VoxCPM2 weights are not in the archive; they download on first use and are Apache-2.0.', '',
             '| Component | Version | Licence | Files |', '|---|---|---|---|']
    lines += [f'| {name} | {version} | {declared} | `{folder}/` |' for name, version, declared, folder in rows]
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{out}: {len(rows)} components')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
