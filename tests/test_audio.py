"""The librosa stand-in: WAV reading, resampling, and silence trimming as librosa trims.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from voxcpm_addon import audio

SR = 16000


def _tone(n, freq, amplitude):
    return (amplitude * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


# [start, end] from librosa 1.0.0's effects.trim(y, top_db, ref=np.max,
# frame_length=2048, hop_length=512) on the same signals.
SIGNALS = {
    'tone_in_silence': (lambda: np.concatenate([np.zeros(5000, np.float32), _tone(12000, 220, 0.3),
                                                np.zeros(7000, np.float32)]),
                        {20: [4096, 18432], 35: [4096, 18432], 60: [4096, 18432]}),
    'quiet_then_loud': (lambda: np.concatenate([_tone(4000, 440, 1e-4), _tone(3000, 300, 0.2),
                                                _tone(6000, 500, 1e-3), _tone(2000, 700, 0.05),
                                                np.zeros(3333, np.float32)]),
                        {20: [3072, 15872], 35: [3072, 16384], 60: [3072, 16384]}),
    'fade_out': (lambda: (_tone(20000, 330, 0.5) * np.linspace(1, 0, 20000, dtype=np.float32)).astype(np.float32),
                 {20: [0, 18432], 35: [0, 20000], 60: [0, 20000]}),
}


@pytest.mark.parametrize('name', sorted(SIGNALS))
def test_trim_finds_the_bounds_librosa_finds(name):
    make, expected = SIGNALS[name]
    y = make()
    for top_db, bounds in expected.items():
        trimmed, found = audio.effects.trim(y, top_db=top_db, ref=np.max, frame_length=2048, hop_length=512)
        assert found.tolist() == bounds, (name, top_db)
        assert np.array_equal(trimmed, y[bounds[0]:bounds[1]])


def _write(path, samples: np.ndarray, width: int, channels: int, rate: int) -> None:
    scaled = np.clip(samples, -1, 1)
    if width == 1:
        data = ((scaled * 127).round() + 128).astype(np.uint8).tobytes()
    elif width == 2:
        data = (scaled * 32767).round().astype('<i2').tobytes()
    elif width == 3:
        ints = (scaled * (2 ** 23 - 1)).round().astype('<i4')
        data = b''.join(int(v).to_bytes(4, 'little', signed=True)[:3] for v in ints.reshape(-1))
    else:
        data = (scaled * (2 ** 31 - 1)).round().astype('<i4').tobytes()
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(channels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(data)


@pytest.mark.parametrize('width', [1, 2, 3, 4])
def test_reads_pcm_wav_of_every_width(tmp_path, width):
    left, right = _tone(800, 200, 0.5), _tone(800, 300, -0.25)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / f'{width}.wav'
    _write(path, stereo, width, 2, SR)
    samples, rate = audio.read_wav(path)
    assert rate == SR and samples.shape == (800, 2) and samples.dtype == np.float32
    tolerance = 1.5 / 128 if width == 1 else 1e-4
    assert np.max(np.abs(samples - stereo)) < tolerance
    y, rate = audio.load(path, sr=None, mono=True)
    assert np.max(np.abs(y - stereo.mean(axis=1))) < tolerance


def test_load_resamples_to_the_asked_rate(tmp_path):
    path = tmp_path / '48k.wav'
    source = (0.5 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype(np.float32)
    _write(path, source[:, None], 2, 1, 48000)
    y, rate = audio.load(path, sr=SR, mono=True)
    assert rate == SR and abs(len(y) - SR) <= 1
    expected = _tone(len(y), 440, 0.5)
    middle = slice(200, len(y) - 200)  # away from the edges, where the filter has no past or future
    assert np.max(np.abs(y[middle] - expected[middle])) < 2e-3


def test_what_it_cannot_read_is_reported(tmp_path):
    path = tmp_path / 'not.wav'
    path.write_bytes(b'ID3 an mp3, not a wav')
    with pytest.raises(audio.UnreadableAudio):
        audio.read_wav(path)
