#!/usr/bin/env python3
"""Refuse a Linux bundle that needs a newer glibc than Subtitld add-ons may.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Subtitld's Linux add-ons run on glibc 2.28 and newer: AlmaLinux and RHEL 8,
Debian 10, Ubuntu 20.04 and everything since. A library linked on a newer
system asks the dynamic linker for that glibc's symbol versions (GLIBC_2.34,
say) and does not load on an older one: the add-on then dies before its first
line of Python, or, for a library loaded with ctypes, at the user's first
line of speech. tools/build_linux.sh builds in an AlmaLinux 8 container for
this reason, and this check proves the result.

Every ELF file in the given directories and zip archives, whatever its name,
is read, and each glibc version it requires (its .gnu.version_r section, which
the dynamic linker checks in full at load time, used or not) is compared with
the floor. A newer one fails the check and is reported with the symbols bound
to it. So does GLIBC_PRIVATE, which only glibc's own libraries use (a bundled
copy of one, libmvec say, works with the glibc it came from and no other), a
glibc version name this check does not know, and an ELF file it cannot read.

    python tools/check_glibc.py parakeet-nemotron-26.11.1-linux-x86_64.zip
    python tools/check_glibc.py --max 2.28 dist/parakeet-nemotron-addon
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import zipfile
from pathlib import Path
from typing import Iterator

FLOOR = '2.28'
SHT_DYNSYM = 11
SHT_GNU_VERNEED = 0x6ffffffe
SHT_GNU_VERSYM = 0x6fffffff
_GLIBC = re.compile(r'^GLIBC_(\d+(?:\.\d+)+)$')
# Versions glibc defines besides GLIBC_<release>, with the release that added
# each: linking with -z pack-relative-relocs makes a file need DT_RELR.
_MARKERS = {'GLIBC_ABI_DT_RELR': '2.36'}


class ElfError(Exception):
    """An ELF file this check cannot read."""


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split('.'))


def _string(data: bytes, offset: int) -> str:
    return data[offset:data.index(b'\0', offset)].decode('ascii', 'replace')


def glibc_needs(data: bytes) -> dict[str, set[str]]:
    """The GLIBC_* versions an ELF image requires, each with the undefined symbols bound to it."""
    if data[:4] != b'\x7fELF':
        raise ElfError('not an ELF file')
    wide, order = data[4] == 2, data[5]
    if data[4] not in (1, 2) or order not in (1, 2):
        raise ElfError(f'unknown ELF class {data[4]} or byte order {order}')
    end = '<' if order == 1 else '>'
    try:
        if wide:
            shoff, = struct.unpack_from(end + 'Q', data, 0x28)
            shentsize, shnum = struct.unpack_from(end + 'HH', data, 0x3A)
            section, symbol, shndx_field = end + 'IIQQQQIIQQ', end + 'IBBHQQ', 3
        else:
            shoff, = struct.unpack_from(end + 'I', data, 0x20)
            shentsize, shnum = struct.unpack_from(end + 'HH', data, 0x2E)
            section, symbol, shndx_field = end + 'IIIIIIIIII', end + 'IIIBBH', 5
        if not shnum:
            raise ElfError('no section headers, so its version needs cannot be read')
        # (name, type, flags, addr, offset, size, link, info, addralign, entsize)
        sections = [struct.unpack_from(section, data, shoff + i * shentsize) for i in range(shnum)]

        versions: dict[int, str] = {}  # version index -> name, from every Verneed's Vernaux
        for header in sections:
            if header[1] != SHT_GNU_VERNEED:
                continue
            strings, offset = sections[header[6]][4], header[4]
            for _ in range(header[7]):
                _, count, _, aux, following = struct.unpack_from(end + 'HHIII', data, offset)
                entry = offset + aux
                for _ in range(count):
                    _, _, index, name, next_aux = struct.unpack_from(end + 'IHHII', data, entry)
                    versions[index & 0x7fff] = _string(data, strings + name)
                    entry += next_aux
                offset += following

        needs = {name: set() for name in versions.values() if name.startswith('GLIBC_')}
        dynsym = next((h for h in sections if h[1] == SHT_DYNSYM), None)
        versym = next((h for h in sections if h[1] == SHT_GNU_VERSYM), None)
        if needs and dynsym and versym and dynsym[9]:
            strings = sections[dynsym[6]][4]
            for i in range(dynsym[5] // dynsym[9]):
                fields = struct.unpack_from(symbol, data, dynsym[4] + i * dynsym[9])
                if fields[shndx_field] != 0:
                    continue  # defined in this file, not a need
                index, = struct.unpack_from(end + 'H', data, versym[4] + 2 * i)
                name = versions.get(index & 0x7fff)
                if name in needs:
                    needs[name].add(_string(data, strings + fields[0]))
        return needs
    except (struct.error, ValueError, IndexError) as exc:
        raise ElfError(f'truncated or malformed ({exc})') from None


def elf_images(path: Path) -> Iterator[tuple[str, bytes]]:
    """(name, contents) of every ELF file in a directory, in a zip archive, or of a single file."""
    if path.is_dir():
        for file in sorted(p for p in path.rglob('*') if p.is_file()):
            with file.open('rb') as handle:
                if handle.read(4) == b'\x7fELF':
                    yield file.relative_to(path).as_posix(), file.read_bytes()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                with archive.open(info) as handle:
                    if handle.read(4) != b'\x7fELF':
                        continue
                yield info.filename, archive.read(info)
    else:
        yield path.name, path.read_bytes()


def check(paths: list[Path], floor: str = FLOOR) -> tuple[int, str, list[str]]:
    """(ELF files read, the newest glibc any of them needs, problems)."""
    limit = version_tuple(floor)
    problems: list[str] = []
    count, newest = 0, ''
    for path in paths:
        if not path.exists():
            problems.append(f'{path}: does not exist')
            continue
        for name, data in elf_images(path):
            count += 1
            try:
                needs = glibc_needs(data)
            except ElfError as exc:
                problems.append(f'{name}: {exc}')
                continue
            for version, symbols in sorted(needs.items()):
                match = _GLIBC.match(version)
                number = match.group(1) if match else _MARKERS.get(version)
                if number is None:
                    reason = ('a part of glibc itself, which works only with the glibc it came from'
                              if version == 'GLIBC_PRIVATE' else 'a glibc version this check does not know')
                    problems.append(f'{name} needs {version}: {reason}')
                    continue
                if not newest or version_tuple(number) > version_tuple(newest):
                    newest = number
                if version_tuple(number) > limit:
                    shown = ', '.join(sorted(symbols)[:8]) or 'no symbol, but the dynamic linker checks it'
                    more = f' and {len(symbols) - 8} more' if len(symbols) > 8 else ''
                    release = '' if match else f' (glibc {number})'
                    problems.append(f'{name} needs {version}{release}, newer than {floor}: {shown}{more}')
    if not count and not problems:
        problems.append('no ELF file found in ' + ', '.join(str(p) for p in paths))
    return count, newest, problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', type=Path, nargs='+', help='directories, zip archives or ELF files')
    parser.add_argument('--max', default=FLOOR, metavar='VERSION',
                        help=f'the newest glibc the files may need (default {FLOOR})')
    args = parser.parse_args(argv)
    count, newest, problems = check(args.paths, args.max)
    print(f'{count} ELF file(s) need at most glibc {newest or "(none)"}; the floor is {args.max}')
    if problems:
        print(f'FATAL: the bundle does not run on glibc {args.max}:', file=sys.stderr)
        for problem in problems:
            print('  ' + problem, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
