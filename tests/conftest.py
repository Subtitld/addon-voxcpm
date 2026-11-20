"""Shared fixtures: the request handler with a stand-in for the model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from array import array

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxcpm_addon import __main__ as addon  # noqa: E402
from voxcpm_addon import audio, engine  # noqa: E402


class FakeModel:
    """Records each synthesis and returns a short tone; runs the step callback like the real loop."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.loads = 0
        self.steps = 8

    def load(self, path, threads=0):
        self.loads += 1
        return engine.Loaded(lib=None, registry=None, model=None, session=None, threads=threads or 4)

    def synthesize(self, loaded, text, **kwargs):
        self.calls.append((text, kwargs))
        on_step = kwargs.get('on_step')
        for index in range(self.steps):
            if on_step is not None:
                on_step(index)
        return array('f', (0.25 * math.sin(i / 10.0) for i in range(4800))), 48000


@pytest.fixture
def run(monkeypatch, tmp_path):
    """run(params) -> (frames, fake): one tts.synthesize request through the handler."""
    frames: list[dict] = []
    fake = FakeModel()
    monkeypatch.setattr(addon, '_emit', frames.append)
    monkeypatch.setattr(addon, '_loaded', None)
    monkeypatch.setattr(engine, 'weights_present', lambda folder: True)
    monkeypatch.setattr(engine, 'load', fake.load)
    monkeypatch.setattr(engine, 'synthesize', fake.synthesize)
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setenv('SUBTITLD_TEMP_DIR', str(tmp_path))
    for name in ('VOXCPM_VOICE_DESCRIPTION', 'VOXCPM_VOICE_REF_AUDIO', 'VOXCPM_VOICE_REF_TEXT',
                 'VOXCPM_MODEL_DIR', 'VOXCPM_STEPS', 'VOXCPM_CFG'):
        monkeypatch.delenv(name, raising=False)

    def _run(params: dict, cancel_at_step: int | None = None):
        state = addon._Request('req-1')
        if cancel_at_step is not None:
            original = fake.synthesize

            def cancelling(loaded, text, **kwargs):
                on_step = kwargs.get('on_step')

                def step(index):
                    if index == cancel_at_step:
                        state.cancel.set()
                    on_step(index)
                kwargs['on_step'] = step
                return original(loaded, text, **kwargs)
            monkeypatch.setattr(engine, 'synthesize', cancelling)
        frames.clear()
        addon._handle_synthesize(state, params)
        return list(frames), fake
    return _run


@pytest.fixture
def reference(tmp_path):
    """A one-second WAV standing in for the speaker's clip."""
    path = tmp_path / 'speaker.wav'
    audio.write_wav(path, array('f', bytes(4 * 16000)), 16000)
    return str(path)


def pytest_configure(config):
    # Also declared in pyproject.toml, which tools/test_archive.sh does not mount.
    config.addinivalue_line('markers', 'weights: needs the 2.96 GB VoxCPM2 model (local runs only)')
