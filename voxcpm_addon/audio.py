"""The two things VoxCPM2 takes from librosa, without librosa.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

VoxCPM2 imports librosa for `librosa.load(path, sr=16000, mono=True)`, to read
a reference clip, and `librosa.effects.trim(...)`, to trim its silence. librosa
brings scipy, scikit-learn, numba with LLVM, soundfile with libsndfile, and
soxr: about 400 MB of the bundle, and every LGPL library in it but numpy's
libquadmath. So `engine.load` registers this module as `librosa` before
importing VoxCPM, and nothing else of librosa is used.

`load` reads 8/16/24/32-bit PCM WAV with the standard library; the add-on
converts anything else with ffmpeg first. `trim` is librosa's algorithm
(frame RMS, centred frames, decibels against the loudest frame), checked
against librosa in tests/test_audio.py.
"""

from __future__ import annotations

import types
import wave

import numpy as np


class UnreadableAudio(Exception):
    """Not a PCM WAV file this reader takes."""


def read_wav(path) -> tuple[np.ndarray, int]:
    """(samples as float32 in [-1, 1], shape (frames, channels); sample rate)."""
    try:
        with wave.open(str(path), 'rb') as clip:
            width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
            data = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError) as exc:
        raise UnreadableAudio(f'{path}: {exc}') from exc
    if width == 1:
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = (np.where(value >= 1 << 23, value - (1 << 24), value) / float(1 << 23)).astype(np.float32)
    elif width == 4:
        samples = (np.frombuffer(data, dtype='<i4').astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        raise UnreadableAudio(f'{path}: {8 * width}-bit samples')
    return samples.reshape(-1, channels), rate


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Band-limited sinc resampling, Kaiser window: torchaudio's "kaiser_best"."""
    if orig_sr == target_sr:
        return y
    import torch
    import torchaudio.functional as taf
    out = taf.resample(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), orig_sr, target_sr,
                       lowpass_filter_width=64, rolloff=0.9475937167399596,
                       resampling_method='sinc_interp_kaiser', beta=14.769656459379492)
    return out.numpy()


def load(path, sr: int | None = 22050, mono: bool = True, **_ignored) -> tuple[np.ndarray, int]:
    """librosa.load for PCM WAV: float32 samples (mono when asked) at `sr` (None: the file's)."""
    samples, rate = read_wav(path)
    y = samples.mean(axis=1) if mono else samples.T
    if sr is not None and sr != rate:
        y = resample(y, rate, sr)
        rate = sr
    return np.ascontiguousarray(y, dtype=np.float32), rate


def _frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """librosa.feature.rms(y=y, center=True, pad_mode='constant') for mono y."""
    padded = np.pad(y, frame_length // 2, mode='constant')
    count = 1 + (len(padded) - frame_length) // hop_length
    power = np.empty(count, dtype=np.float64)
    squares = np.abs(padded.astype(np.float64)) ** 2
    for index in range(count):
        start = index * hop_length
        power[index] = squares[start:start + frame_length].mean()
    return np.sqrt(power)


def trim(y, top_db: float = 60, ref=np.max, frame_length: int = 2048, hop_length: int = 512,
         **_ignored) -> tuple[np.ndarray, np.ndarray]:
    """librosa.effects.trim for mono y: (trimmed y, [start, end] in samples)."""
    y = np.asarray(y)
    rms = _frame_rms(y, frame_length, hop_length)
    ref_value = ref(np.abs(rms)) if callable(ref) else np.abs(ref)
    amin = 1e-10  # librosa's amplitude_to_db amin (1e-5), squared
    db = 10.0 * np.log10(np.maximum(amin, rms ** 2)) - 10.0 * np.log10(np.maximum(amin, ref_value ** 2))
    nonzero = np.flatnonzero(db > -top_db)
    if nonzero.size:
        start = int(nonzero[0] * hop_length)
        end = min(y.shape[-1], int((nonzero[-1] + 1) * hop_length))
    else:
        start = end = 0
    return y[start:end], np.asarray([start, end])


effects = types.SimpleNamespace(trim=trim)
