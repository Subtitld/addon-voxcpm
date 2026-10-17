#!/usr/bin/env bash
# Build the Linux archive in ubuntu:22.04, as the release workflow does on its
# runner, so the bundle links against glibc 2.35 and runs on older systems.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
#     tools/build-linux.sh            # -> dist-linux/voxcpm-<version>-linux-x86_64.zip
#
# Downloads (Python, wheels) are kept in the `voxcpm-addon-cache` volume.
# Needs docker and uv on the host; the host's uv binary is mounted read-only.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
uv_bin=$(command -v uv)
version=$(python3 -c "import json; print(json.load(open('$repo/manifest.json'))['version'])")

docker run --rm \
  -v "$repo":/src:ro \
  -v "$repo/dist-linux":/out \
  -v voxcpm-addon-cache:/cache \
  -v "$uv_bin":/usr/local/bin/uv:ro \
  -e UV_CACHE_DIR=/cache/uv -e UV_PYTHON_INSTALL_DIR=/cache/python \
  -e VERSION="$version" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  ubuntu:22.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq --no-install-recommends ca-certificates binutils ffmpeg >/dev/null
    cp -r /src /build && cd /build && rm -rf build dist release-staging .spike
    uv venv -q --python 3.12 /venv && export VIRTUAL_ENV=/venv PATH=/venv/bin:$PATH
    uv pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchaudio==2.11.0
    uv pip install -q -c constraints.txt -e ".[build,dev]"
    uv pip install -q --no-deps -c constraints.txt voxcpm
    python -m pytest -q
    pyinstaller pyinstaller.spec --noconfirm --log-level WARN
    dist/voxcpm-addon/voxcpm-addon --self-test
    mkdir -p release-staging/bin
    cp -R dist/voxcpm-addon/. release-staging/bin/
    cp manifest.json LICENSE README.md release-staging/
    python tools/collect_licences.py --out release-staging/THIRD_PARTY_LICENSES
    archive=voxcpm-$VERSION-linux-x86_64.zip
    python tools/make_archive.py release-staging "/out/$archive"
    sha256sum "/out/$archive" | cut -d" " -f1 > "/out/$archive.sha256"
    du -sh dist/voxcpm-addon
    chown -R "$HOST_UID:$HOST_GID" /out
  '
