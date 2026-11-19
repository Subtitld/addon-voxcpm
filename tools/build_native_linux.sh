#!/usr/bin/env bash
# Build the Linux native libraries (tools/build_native.py) inside the
# AlmaLinux 8 container, so they load on glibc 2.28.
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
#     tools/build_native_linux.sh [--jobs N]
#
# writes build/assets/ and build/licences/, owned by you. The audio.cpp
# checkout stays in build/audiocpp-src/ and the build tree in build/native/ between runs.
# Needs Docker and the network.

set -euo pipefail

BUILD_IMAGE="${BUILD_IMAGE:-subtitld-voxcpm-build:almalinux8}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# No build context: the Dockerfile copies nothing in.
docker build -q -t "$BUILD_IMAGE" - < "$ROOT/tools/linux-build.Dockerfile" >/dev/null

docker run --rm \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$ROOT:/src" \
    -w /src \
    "$BUILD_IMAGE" \
    python tools/build_native.py "$@"
