"""WAV in and out with the standard library.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import math
import wave
from array import array

import pytest

from voxcpm_addon import audio


def _write(path, values, rate=16000, channels=1, width=2):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(array('h', values).tobytes() if width == 2 else bytes(len(values) * width))


def test_reads_16_bit_wav_as_floats_with_its_channels(tmp_path):
    path = tmp_path / 'stereo.wav'
    _write(path, [16384, -16384, 32767, -32768], rate=44100, channels=2)
    clip = audio.read_wav(path)
    assert (clip.rate, clip.channels, clip.frames) == (44100, 2, 2)
    assert list(clip.samples) == pytest.approx([0.5, -0.5, 32767 / 32768, -1.0])


def test_writes_16_bit_mono_and_mixes_channels_down(tmp_path):
    tone = array('f', (0.5 * math.sin(i / 7) for i in range(1600)))
    path = tmp_path / 'out' / 'line.wav'
    assert audio.write_wav(path, tone, 16000) == pytest.approx(0.1)
    with wave.open(str(path)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.getnframes()) == (1, 2, 16000, 1600)
    back = audio.read_wav(path).samples
    assert max(abs(a - b) for a, b in zip(back, tone)) < 1e-4
    assert not (tmp_path / 'out' / 'line.wav.part').exists()

    stereo = array('f', [0.2, 0.4, 2.0, 2.0])
    audio.write_wav(path, stereo, 8000, channels=2)
    assert list(audio.read_wav(path).samples) == pytest.approx([0.3, 32767 / 32768], abs=1e-4)


def test_what_it_cannot_read_is_reported(tmp_path):
    (tmp_path / 'text.wav').write_text('not audio')
    with pytest.raises(audio.UnreadableAudio):
        audio.read_wav(tmp_path / 'text.wav')
    _write(tmp_path / 'wide.wav', [0] * 10, width=3)
    with pytest.raises(audio.UnreadableAudio, match='24-bit'):
        audio.read_wav(tmp_path / 'wide.wav')
    _write(tmp_path / 'empty.wav', [])
    with pytest.raises(audio.UnreadableAudio, match='no audio'):
        audio.read_wav(tmp_path / 'empty.wav')
    with pytest.raises(audio.UnreadableAudio):
        audio.read_wav(tmp_path / 'missing.wav')
