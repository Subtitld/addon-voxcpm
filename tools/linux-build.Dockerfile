# Build environment for the Linux audio.cpp libraries and, later, the frozen
# add-on (tools/build_native_linux.sh runs it).
#
# Copyright 2026 Subtitld contributors
# SPDX-License-Identifier: Apache-2.0
#
# AlmaLinux 8.10, glibc 2.28: the base of manylinux_2_28 and the floor the
# Subtitld add-ons build for. Its gcc-toolset-14 links new C++ features
# statically and leaves only the system libstdc++ (GCC 8) as a runtime need.
#
# PyInstaller needs a Python with a shared libpython; the image's /opt/python
# interpreters are static, so the distribution's Python 3.12 is installed
# (/usr/bin/python3.12; /usr/local/bin/python3.12 is a static one).
#
# CMake 3.31.6 and Ninja are pinned in /opt/build-tools, ahead of the image's
# CMake 4 on PATH.
FROM quay.io/pypa/manylinux_2_28_x86_64:2025.11.09-2

RUN dnf install -y --setopt=install_weak_deps=False python3.12 python3.12-pip \
    && dnf clean all \
    && /usr/bin/python3.12 -c "import sysconfig, sys; sys.exit(0 if sysconfig.get_config_var('Py_ENABLE_SHARED') else '/usr/bin/python3.12 has no shared libpython')" \
    && /usr/bin/python3.12 -m venv /opt/build-tools \
    && /opt/build-tools/bin/pip install --no-cache-dir cmake==3.31.6 ninja==1.13.0

ENV PATH=/opt/build-tools/bin:$PATH
