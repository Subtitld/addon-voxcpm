"""audio.cpp as built by tools/build_native.py: loaded, typed, and with our patch in.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Needs build/assets/audiocpp/ (run tools/build_native.py first). The tests
marked `weights` also need the model: VOXCPM_TEST_MODEL names the GGUF file
(local runs only, 2.96 GB).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from voxcpm_addon import audio, engine

MODEL = (os.environ.get('VOXCPM_TEST_MODEL') or '').strip()
needs_model = pytest.mark.skipif(not MODEL, reason='VOXCPM_TEST_MODEL is not set')


def test_the_library_is_ours():
    engine.load_library()  # fails without audiocpp_session_set_step_callback
    assert engine.runtime_version().startswith('audio.cpp 0.9.5 (C ABI 0.')
    assert 'voxcpm2' in engine.families()


def test_a_missing_library_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, '_lib', None)
    monkeypatch.setenv('VOXCPM_AUDIOCPP_DIR', str(tmp_path))
    with pytest.raises(engine.EngineError, match='missing'):
        engine.load_library()


def test_a_model_that_is_not_there_fails_cleanly(tmp_path):
    with pytest.raises(engine.EngineError):
        engine.load(tmp_path / 'nothing.gguf', threads=1)


def test_threads():
    assert engine.default_threads() >= 1
    cpus = engine.performance_cpus()
    assert cpus == sorted(cpus)


@pytest.fixture(scope='module')
def loaded():
    model = engine.load(Path(MODEL))
    yield model
    engine.close(model)


@pytest.mark.weights
@needs_model
def test_speaks_steps_and_stops(loaded, tmp_path):
    steps = []
    samples, rate = engine.synthesize(loaded, 'We leave at dawn.', seed=3, on_step=steps.append)
    assert rate == 48000 and len(samples) / rate > 0.5
    assert steps[:3] == [0, 1, 2]
    assert len(steps) == pytest.approx(len(samples) / rate * engine.STEPS_PER_SECOND, abs=3)

    reference = tmp_path / 'voice.wav'
    audio.write_wav(reference, samples, rate)

    def stop(index):
        if index == 3:
            raise engine.Cancelled()

    started = time.monotonic()
    with pytest.raises(engine.Cancelled):
        engine.synthesize(loaded, 'A long line that is stopped after a few steps, well before its end.',
                          reference=str(reference), reference_text='We leave at dawn.', on_step=stop)
    assert time.monotonic() - started < 60

    samples, rate = engine.synthesize(loaded, 'Still here.', reference=str(reference),
                                      reference_text='We leave at dawn.')
    assert len(samples) / rate > 0.3
