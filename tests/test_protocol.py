"""The tts.synthesize request: which mode each voice runs in, and the frames it answers with.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

from voxcpm_addon import voices

REPO = Path(__file__).resolve().parents[1]


def _terminal(frames):
    terminal = [f for f in frames if f.get('type') in ('result', 'error')]
    assert len(terminal) == 1, frames
    return terminal[0]


def test_clone_with_transcript_continues_from_the_clip(run, reference, tmp_path):
    out = tmp_path / 'line.wav'
    frames, fake = run({'text': '  Olá,   mundo ', 'voice': voices.CLONE, 'output_path': str(out),
                        'voice_ref_audio': reference, 'voice_ref_text': 'Hello there', 'seed': 7})
    result = _terminal(frames)
    assert result['type'] == 'result', result
    text, kwargs = fake.calls[0]
    assert text == 'Olá, mundo'
    assert kwargs['reference_wav'] == reference
    assert kwargs['prompt_wav'] == reference
    assert kwargs['prompt_text'] == 'Hello there'
    assert kwargs['seed'] == 7
    data = result['data']
    assert data['path'] == str(out) and data['sample_rate'] == 48000 and data['channels'] == 1
    with wave.open(str(out)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 48000)
        assert clip.getnframes() == 4800
    assert abs(data['duration_sec'] - 0.1) < 1e-6
    values = [f['value'] for f in frames if f['type'] == 'progress']
    assert values == sorted(values) and values[-1] == 1.0


def test_clone_without_transcript_clones_the_timbre_only(run, reference, tmp_path):
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav'),
                        'voice_ref_audio': reference})
    assert _terminal(frames)['type'] == 'result'
    kwargs = fake.calls[0][1]
    assert kwargs['reference_wav'] == reference
    assert 'prompt_wav' not in kwargs and 'prompt_text' not in kwargs


def test_clone_falls_back_to_the_configured_clip(run, reference, tmp_path, monkeypatch):
    monkeypatch.setenv('VOXCPM_VOICE_REF_AUDIO', reference)
    monkeypatch.setenv('VOXCPM_VOICE_REF_TEXT', 'configured words')
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav')})
    assert _terminal(frames)['type'] == 'result'
    assert fake.calls[0][1]['prompt_text'] == 'configured words'


def test_the_configured_transcript_never_pairs_with_the_hosts_clip(run, reference, tmp_path, monkeypatch):
    monkeypatch.setenv('VOXCPM_VOICE_REF_TEXT', 'words of another clip')
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav'),
                        'voice_ref_audio': reference})
    assert _terminal(frames)['type'] == 'result'
    assert 'prompt_text' not in fake.calls[0][1]


def test_clone_without_any_clip_is_refused(run, tmp_path):
    frames, fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(tmp_path / 'a.wav')})
    error = _terminal(frames)
    assert error['type'] == 'error' and error['code'] == 'bad_params'
    assert not fake.calls


def test_described_voice_makes_its_anchor_once_per_speaker(run, tmp_path):
    params = {'text': 'Bom dia', 'voice': 'voxcpm-calm-man', 'language': 'pt-br', 'seed': 11,
              'output_path': str(tmp_path / 'a.wav')}
    frames, fake = run(params)
    assert _terminal(frames)['type'] == 'result'
    (anchor_text, anchor_kwargs), (line_text, line_kwargs) = fake.calls
    description = voices.BY_ID['voxcpm-calm-man'].description
    assert anchor_text == f'({description}){voices.ANCHORS["pt"]}'
    assert anchor_kwargs['seed'] == 11
    assert line_text == 'Bom dia'
    assert line_kwargs['prompt_text'] == voices.ANCHORS['pt']
    assert line_kwargs['prompt_wav'] == line_kwargs['reference_wav']
    assert Path(line_kwargs['reference_wav']).is_file()

    frames, fake = run(dict(params, text='Boa noite'))
    assert _terminal(frames)['type'] == 'result'
    assert [text for text, _ in fake.calls[-1:]] == ['Boa noite']
    assert len(fake.calls) == 3  # the anchor was reused

    frames, fake = run(dict(params, seed=12))  # another speaker, another voice
    assert len(fake.calls) == 5


def test_described_voice_in_a_language_without_an_anchor_clones_the_timbre_only(run, tmp_path):
    frames, fake = run({'text': 'สวัสดี', 'voice': 'voxcpm-warm-woman', 'language': 'th',
                        'output_path': str(tmp_path / 'a.wav')})
    assert _terminal(frames)['type'] == 'result'
    anchor_text = fake.calls[0][0]
    assert anchor_text.endswith(voices.ANCHORS['en'])
    line_kwargs = fake.calls[1][1]
    assert 'prompt_text' not in line_kwargs and line_kwargs['reference_wav']


def test_the_voice_you_describe_reads_the_setting(run, tmp_path, monkeypatch):
    frames, fake = run({'text': 'Hi', 'voice': voices.DESCRIBED, 'output_path': str(tmp_path / 'a.wav')})
    error = _terminal(frames)
    assert error['type'] == 'error' and error['code'] == 'bad_params'

    monkeypatch.setenv('VOXCPM_VOICE_DESCRIPTION', 'A child, cheerful voice')
    frames, fake = run({'text': 'Hi', 'voice': voices.DESCRIBED, 'output_path': str(tmp_path / 'a.wav')})
    assert _terminal(frames)['type'] == 'result'
    assert fake.calls[0][0].startswith('(A child, cheerful voice)')


def test_unknown_voice_and_missing_fields(run, tmp_path):
    error = _terminal(run({'text': 'Hi', 'voice': 'qwen3-ryan', 'output_path': str(tmp_path / 'a.wav')})[0])
    assert error['code'] == 'unsupported_voice'
    assert _terminal(run({'text': '   ', 'voice': voices.CLONE, 'output_path': 'x.wav'})[0])['code'] == 'bad_params'
    assert _terminal(run({'text': 'Hi', 'voice': voices.CLONE})[0])['code'] == 'bad_params'


def test_cancel_stops_the_line_between_steps(run, reference, tmp_path):
    out = tmp_path / 'a.wav'
    frames, _fake = run({'text': 'Hi', 'voice': voices.CLONE, 'output_path': str(out),
                         'voice_ref_audio': reference}, cancel_at_step=3)
    error = _terminal(frames)
    assert error['type'] == 'error' and error['code'] == 'cancelled'
    assert not out.exists()


def test_settings_are_held_to_their_range(monkeypatch):
    from voxcpm_addon import __main__ as addon
    monkeypatch.setenv('VOXCPM_STEPS', '500')
    monkeypatch.setenv('VOXCPM_CFG', 'lots')
    options = addon._options()
    assert options['steps'] == 30 and options['cfg'] == 2.0


def test_the_process_says_hello_and_leaves_on_shutdown():
    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.Popen([sys.executable, '-m', 'voxcpm_addon'], cwd=REPO, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        hello = json.loads(proc.stdout.readline())
        proc.stdin.write(b'{"type":"ready","host":"test"}\n')
        proc.stdin.write(b'{"id":"x","type":"asr.transcribe","params":{}}\n')
        proc.stdin.flush()
        error = json.loads(proc.stdout.readline())
        proc.stdin.write(b'{"type":"shutdown"}\n')
        proc.stdin.flush()
        assert proc.wait(timeout=20) == 0
    finally:
        proc.kill()
    assert hello['type'] == 'hello' and hello['addon'] == 'voxcpm'
    capability = hello['capabilities'][0]
    assert capability['task'] == 'tts.synthesize' and capability['voice_clone'] is True
    assert {v['id'] for v in capability['voices']} == set(voices.BY_ID)
    assert error == {'id': 'x', 'type': 'error', 'code': 'bad_params',
                     'message': "unsupported frame type 'asr.transcribe'", 'retryable': False}
