#!/usr/bin/env bash
# Build audio.cpp, the frozen add-on and its archive, inside the AlmaLinux 8
# container.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# The bundle carries the build interpreter's libpython and its OpenSSL, and
# audio.cpp is compiled here, so the add-on runs on any glibc at least as new
# as the build's. The image (tools/linux-build.Dockerfile) is AlmaLinux 8,
# glibc 2.28, as for the other Subtitld add-ons, with the distribution's
# Python 3.12. tools/check_glibc.py then refuses the archive if any file in it
# needs a newer glibc after all.
#
#     tools/build_linux.sh [--jobs N]
#
# writes, into the repository, owned by you:
#     voxcpm-<version>-linux-x86_64.zip   (+ .sha256)
#     build/assets/, dist/voxcpm-addon/   (as tools/build_native.py and PyInstaller do)
#
# The audio.cpp checkout stays in build/audiocpp-src/ and the build tree in
# build/native/ between runs. Needs Docker and the network.

set -euo pipefail

BUILD_IMAGE="${BUILD_IMAGE:-subtitld-voxcpm-build:almalinux8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = .\(.*\).$/\1/p' "$ROOT/voxcpm_addon/__init__.py")"
ARCHIVE="voxcpm-${VERSION}-linux-x86_64.zip"
# pip's cache survives between builds.
PIP_CACHE="${PIP_CACHE:-$HOME/.cache/subtitld-voxcpm-build-pip}"
mkdir -p "$PIP_CACHE"

# No build context: the Dockerfile copies nothing in.
docker build -q -t "$BUILD_IMAGE" - < "$ROOT/tools/linux-build.Dockerfile" >/dev/null

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$ROOT:/src" \
    -v "$PIP_CACHE:/tmp/.cache/pip" \
    -w /src \
    "$BUILD_IMAGE" \
    bash -euo pipefail -c "
        command -v nm >/dev/null || { echo 'nm is required for the libexpat check' >&2; exit 1; }
        # The distribution's interpreter, which has a shared libpython.
        /usr/bin/python3.12 -m venv /tmp/build-venv
        . /tmp/build-venv/bin/activate
        pip install -q --upgrade pip
        pip install -q '.[build,dev]'

        # A failed build must not leave older outputs that look current.
        rm -rf build/pyinstaller dist release-staging '$ARCHIVE' '$ARCHIVE.sha256'

        # CMake and Ninja come from /opt/build-tools, still on PATH.
        python tools/build_native.py $*
        python -m pytest -q -p no:cacheprovider -m 'not weights' \\
            tests/test_engine.py tests/test_audio.py tests/test_manifest.py tests/test_protocol.py tests/test_download.py \\
            tests/test_metal.py

        # /tmp/.cache belongs to root here, made by the pip mount.
        PYINSTALLER_CONFIG_DIR=/tmp/pyinstaller-config \\
            pyinstaller pyinstaller.spec --noconfirm --log-level WARN --workpath build/pyinstaller
        dist/voxcpm-addon/voxcpm-addon --self-test

        mkdir -p release-staging/bin
        cp -R dist/voxcpm-addon/. release-staging/bin/
        cp manifest.json LICENSE README.md release-staging/
        python tools/assemble_licences.py --bundle dist/voxcpm-addon \\
            --out release-staging/THIRD_PARTY_LICENSES
        python tools/make_archive.py release-staging '$ARCHIVE'
        python tools/check_glibc.py --max 2.28 '$ARCHIVE'
        sha256sum '$ARCHIVE' | cut -d' ' -f1 > '$ARCHIVE.sha256'
    "

echo "built $ARCHIVE, sha256 $(cat "$ROOT/$ARCHIVE.sha256")"
