"""Metal on Apple Silicon: checked once in a child process, the verdict kept, the CPU otherwise.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import subprocess

import pytest

from voxcpm_addon import engine


@pytest.fixture
def mac(monkeypatch, tmp_path):
    """An Apple Silicon Mac whose probe child answers what `answer` holds."""
    monkeypatch.setenv('SUBTITLD_MODELS_DIR', str(tmp_path))
    monkeypatch.delenv('VOXCPM_DEVICE', raising=False)
    monkeypatch.setattr(engine, 'apple_silicon', lambda: True)
    monkeypatch.setattr(engine, 'runtime_version', lambda: 'audio.cpp test')
    state = {'answer': (0, b'metal works\n'), 'runs': 0}

    def run(cmd, **kwargs):
        state['runs'] += 1
        assert '--probe-metal' in cmd and cmd[-1] == str(tmp_path / 'm.gguf')
        code, out = state['answer']
        if code is None:
            raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])
        return subprocess.CompletedProcess(cmd, code, out, b'')

    monkeypatch.setattr(engine.subprocess, 'run', run)
    state['model'] = tmp_path / 'm.gguf'
    return state


def test_elsewhere_it_is_the_cpu(monkeypatch):
    monkeypatch.setattr(engine, 'apple_silicon', lambda: False)
    monkeypatch.setenv('VOXCPM_DEVICE', 'metal')
    assert engine.choose_backend(engine.Path('x.gguf')) == 'cpu'


def test_metal_when_the_child_speaks_and_the_answer_is_kept(mac):
    assert engine.choose_backend(mac['model']) == 'metal'
    assert engine.choose_backend(mac['model']) == 'metal'
    assert mac['runs'] == 1
    saved = json.loads((engine.cache_root() / 'metal-check.json').read_text())
    assert saved['voxcpm2'] is True and saved['runtime'].startswith('audio.cpp test on macOS')


@pytest.mark.parametrize('answer', [(-11, b''), (1, b'Traceback'), (None, b'')])
def test_the_cpu_when_the_child_crashes_fails_or_hangs(mac, answer):
    mac['answer'] = answer
    assert engine.choose_backend(mac['model']) == 'cpu'
    assert engine.choose_backend(mac['model']) == 'cpu'
    assert mac['runs'] == 1


def test_a_new_runtime_checks_again(mac, monkeypatch):
    mac['answer'] = (-11, b'')
    assert engine.choose_backend(mac['model']) == 'cpu'
    monkeypatch.setattr(engine, 'runtime_version', lambda: 'audio.cpp newer')
    mac['answer'] = (0, b'metal works\n')
    assert engine.choose_backend(mac['model']) == 'metal'
    assert mac['runs'] == 2


def test_the_setting_skips_the_check(mac, monkeypatch):
    monkeypatch.setenv('VOXCPM_DEVICE', 'cpu')
    assert engine.choose_backend(mac['model']) == 'cpu'
    monkeypatch.setenv('VOXCPM_DEVICE', 'metal')
    assert engine.choose_backend(mac['model']) == 'metal'
    assert mac['runs'] == 0
