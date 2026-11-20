"""Read and write PCM WAV with the standard library.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

audio.cpp takes a reference clip as float samples at any rate and channel
count, and resamples it itself, so a 16-bit PCM WAV is read as it is.
Subtitld's own clips are 16 kHz mono 16-bit WAV; anything else is converted
with ffmpeg first (see __main__).
"""

from __future__ import annotations

import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path


class UnreadableAudio(Exception):
    """Not a PCM WAV file this reader takes."""


@dataclass
class Clip:
    samples: array      # float32 in [-1, 1], channels interleaved
    rate: int
    channels: int

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels


def read_wav(path) -> Clip:
    """A 16-bit PCM WAV file, as float samples."""
    try:
        with wave.open(str(path), 'rb') as clip:
            width, channels, rate = clip.getsampwidth(), clip.getnchannels(), clip.getframerate()
            data = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise UnreadableAudio(f'{path}: {exc}') from exc
    if width != 2:
        raise UnreadableAudio(f'{path}: {8 * width}-bit samples')
    pcm = array('h')
    pcm.frombytes(data[:len(data) - len(data) % (2 * channels)])
    if sys.byteorder != 'little':
        pcm.byteswap()
    if not pcm:
        raise UnreadableAudio(f'{path}: no audio')
    return Clip(array('f', (x / 32768.0 for x in pcm)), rate, channels)


def write_wav(path, samples: array, rate: int, channels: int = 1) -> float:
    """16-bit mono PCM, which the host's dubbing pipeline expects, written beside
    `path` and renamed into place. Several channels are mixed down. Returns seconds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels > 1:
        samples = array('f', (sum(samples[i:i + channels]) / channels
                              for i in range(0, len(samples) - channels + 1, channels)))
    pcm = array('h', (int(round(max(-1.0, min(1.0, x)) * 32767.0)) for x in samples))
    if sys.byteorder != 'little':
        pcm.byteswap()
    part = path.with_name(path.name + '.part')
    with wave.open(str(part), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(rate))
        out.writeframes(pcm.tobytes())
    part.replace(path)
    return len(pcm) / float(rate)
