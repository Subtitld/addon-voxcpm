"""Shared fixtures: the request handler with a stand-in for the model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxcpm_addon import __main__ as addon  # noqa: E402
from voxcpm_addon import engine  # noqa: E402


class FakeModel:
    """Records each synthesis and returns a short tone; runs the step callback like the real loop."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.loads = 0
        self.steps = 8

    def load(self, folder, device, dtype, threads=0, float32_head=False):
        self.loads += 1
        return engine.Loaded(model=None, device=device, dtype=dtype, sample_rate=48000)

    def synthesize(self, loaded, text, **kwargs):
        self.calls.append((text, kwargs))
        on_step = kwargs.get('on_step')
        for index in range(self.steps):
            if on_step is not None:
                on_step(index)
        return (0.25 * np.sin(np.arange(4800) / 10.0)).astype(np.float32)


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
    monkeypatch.setattr(engine, 'resolve_device', lambda choice: 'cpu')
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
    addon._write_wav(path, np.zeros(16000, dtype=np.float32), 16000)
    return str(path)
