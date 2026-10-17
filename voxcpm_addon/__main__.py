"""VoxCPM add-on for Subtitld — JSON-line protocol entry point.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by
applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.

What this add-on is
-------------------
Text-to-speech with OpenBMB's VoxCPM2 (Apache-2.0), in 30 languages, at
48 kHz. Serves `tts.synthesize`. A line is spoken in the speaker's own voice,
cloned from the video, or in a voice made from a description (see voices.py).
The model, 4.96 GB, downloads on the first synthesis (see engine.py); after
that nothing leaves the machine.

Wire format
-----------
One JSON object per line on stdin/stdout. Stderr is free-form logging, which
the host prefixes with `[addon:voxcpm]` and surfaces.

Options (the host sets them from Configure as VOXCPM_<KEY>)
-----------------------------------------------------------
device, precision, steps, cfg, threads; voice_description for the described
voice; voice_ref_audio and voice_ref_text, a clone reference to fall back on
when the host sends none.

Limits
------
One synthesis at a time: the model holds a single KV cache. A cancel stops a
line between two decoding steps, or stops the download (which resumes on the
next request).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path

from voxcpm_addon import ADDON_ID, PROTOCOL_VERSION, __version__
from voxcpm_addon import engine, voices

ADDON_VERSION = __version__

ERR_MODEL_MISSING = 'model_missing'
ERR_UNSUPPORTED_VOICE = 'unsupported_voice'
ERR_BAD_PARAMS = 'bad_params'
ERR_INTERNAL = 'internal'
ERR_CANCELLED = 'cancelled'
ERR_OOM = 'oom'
ERR_DISK_FULL = 'disk_full'
ERR_NETWORK = 'network_unavailable'

log = logging.getLogger('voxcpm_addon')


# ---------------------------------------------------------------------------
# stdio ownership + protocol frames
# ---------------------------------------------------------------------------
_PROTO_IN = None   # binary reader for host frames
_PROTO_OUT = None  # binary writer for our frames
_proto_lock = threading.Lock()


def _claim_protocol_stdio() -> None:
    """Give the protocol exclusive use of the real stdin/stdout.

    fd-level, not just `sys.stdout = sys.stderr`: native code (PyTorch, its
    OpenMP runtime) may write to fd 1 directly. Falls back to the
    Python-level swap if the fd dance fails.
    """
    global _PROTO_IN, _PROTO_OUT
    try:
        sys.stdout.flush()
        in_fd = os.dup(0)
        out_fd = os.dup(1)
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
        os.dup2(2, 1)
        _PROTO_IN = os.fdopen(in_fd, 'rb')
        _PROTO_OUT = os.fdopen(out_fd, 'wb')
    except (OSError, ValueError, AttributeError):
        _PROTO_IN = sys.stdin.buffer
        _PROTO_OUT = sys.stdout.buffer
    sys.stdin = open(os.devnull, encoding='utf-8')  # stray input() -> EOFError
    sys.stdout = sys.stderr                        # stray print() -> the log


def _emit(frame: dict) -> None:
    data = (json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    out = _PROTO_OUT if _PROTO_OUT is not None else sys.stdout.buffer
    with _proto_lock:
        try:
            out.write(data)
            out.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # host went away; the frame loop exits on the next empty read


def _emit_progress(req_id: str, value: float, message: str = '') -> None:
    _emit({'id': req_id, 'type': 'progress',
           'value': max(0.0, min(1.0, float(value))), 'message': message})


def _emit_result(req_id: str, data: dict) -> None:
    _emit({'id': req_id, 'type': 'result', 'data': data})


def _emit_error(req_id: str, code: str, message: str, retryable: bool = False) -> None:
    _emit({'id': req_id, 'type': 'error', 'code': code,
           'message': message, 'retryable': retryable})


class _AddonError(Exception):
    """An expected failure that already knows its protocol code."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def _env(*names: str) -> str:
    """First non-empty value among `names`, stripped. '' when none are set."""
    for name in names:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value
    return ''


def _env_number(name: str, default: float, low: float, high: float) -> float:
    raw = _env(name)
    try:
        value = float(raw) if raw else default
    except ValueError:
        log.warning('%s=%r is not a number; using %s', name, raw, default)
        value = default
    return max(low, min(high, value))


def _options() -> dict:
    return {
        'device': _env('VOXCPM_DEVICE') or 'auto',
        'precision': _env('VOXCPM_PRECISION') or 'auto',
        'steps': int(_env_number('VOXCPM_STEPS', 10, 4, 30)),
        'cfg': _env_number('VOXCPM_CFG', 2.0, 1.0, 4.0),
        'threads': int(_env_number('VOXCPM_THREADS', 0, 0, 256)),
        'voice_description': _env('VOXCPM_VOICE_DESCRIPTION'),
        'voice_ref_audio': _env('VOXCPM_VOICE_REF_AUDIO'),
        'voice_ref_text': _env('VOXCPM_VOICE_REF_TEXT'),
    }


# ---------------------------------------------------------------------------
# ffmpeg, for references in formats the model's reader does not take
# ---------------------------------------------------------------------------
_DIRECT_AUDIO = ('.wav', '.flac')


def _find_ffmpeg() -> str | None:
    for candidate in (_env('SUBTITLD_FFMPEG_EXECUTABLE'), _env('FFMPEG_EXECUTABLE')):
        if candidate and (os.path.isfile(candidate) or shutil.which(candidate)):
            return candidate
    return shutil.which('ffmpeg')


def _no_window_startupinfo():
    """Keep ffmpeg from flashing a console window on Windows GUI hosts."""
    if sys.platform != 'win32':
        return None
    si = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    si.wShowWindow = subprocess.SW_HIDE  # type: ignore[attr-defined]
    return si


def _child_env() -> dict | None:
    """Environment for ffmpeg, or None to inherit ours unchanged.

    The frozen Linux bootloader prepends the bundle's `_internal/` to
    LD_LIBRARY_PATH, and a system ffmpeg that inherits it loads the bundle's
    libraries instead of its own. PyInstaller saves the host value in
    LD_LIBRARY_PATH_ORIG, so restore it for the child.
    """
    if not (getattr(sys, 'frozen', False) and sys.platform.startswith('linux')):
        return None
    env = dict(os.environ)
    original = env.get('LD_LIBRARY_PATH_ORIG')
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    return env


def _readable_reference(path: str, scratch: Path) -> str:
    """`path` itself if the model reads it, else a 16 kHz WAV of it made by ffmpeg."""
    if Path(path).suffix.lower() in _DIRECT_AUDIO:
        return path
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise _AddonError(ERR_BAD_PARAMS,
                          f'The reference {Path(path).name} is not WAV or FLAC, and no ffmpeg was '
                          f'found to convert it. Use a WAV file.')
    out = scratch / 'reference.wav'
    cmd = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', path,
           '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(out)]
    done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, env=_child_env(),
                          startupinfo=_no_window_startupinfo())
    if done.returncode != 0 or not out.is_file():
        tail = ' / '.join(done.stderr.decode('utf-8', 'replace').strip().splitlines()[-3:])
        raise _AddonError(ERR_BAD_PARAMS, f'Could not read the reference {Path(path).name}: {tail}')
    return str(out)


# ---------------------------------------------------------------------------
# The model: loaded on first use, shared by every request
# ---------------------------------------------------------------------------
_loaded: engine.Loaded | None = None
_model_lock = threading.Lock()   # held for the whole synthesis: one KV cache


def _load_model(state: '_Request', options: dict) -> engine.Loaded:
    """Download (once) and load. The caller holds _model_lock."""
    global _loaded
    if _loaded is not None:
        return _loaded
    folder = engine.model_dir()
    if not engine.weights_present(folder):
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if now - last[0] < 1.0 and done < total:
                return
            last[0] = now
            share = done / total if total else 1.0
            _emit_progress(state.req_id, 0.2 * share,
                           f'Downloading VoxCPM2, once: {done / 1e9:.2f} of {total / 1e9:.2f} GB')

        log.info('weights missing in %s; downloading', folder)
        try:
            engine.ensure_weights(folder, progress, state.cancel.is_set)
        except engine.Cancelled:
            raise
        except engine.DownloadError as exc:
            raise _AddonError(ERR_NETWORK, f'Could not download VoxCPM2: {exc}', retryable=True) from exc
        except OSError as exc:
            if getattr(exc, 'errno', None) == 28:
                raise _AddonError(ERR_DISK_FULL,
                                  f'Out of disk space downloading VoxCPM2 to {folder} '
                                  f'(it needs {engine.MODEL_BYTES / 1e9:.1f} GB).') from exc
            raise
    _emit_progress(state.req_id, 0.2, 'Loading VoxCPM2...')
    try:
        device = engine.resolve_device(options['device'])
        dtype, float32_head = engine.resolve_precision(options['precision'], device)
    except Exception as exc:  # noqa: BLE001 — torch failing to import is reported, not raised
        raise _AddonError(ERR_INTERNAL, f'PyTorch is not usable: {exc}') from exc
    started = time.monotonic()
    try:
        _loaded = engine.load(folder, device, dtype, options['threads'], float32_head)
    except (ValueError, RuntimeError) as exc:
        if 'not available' in str(exc).lower():  # a device this build cannot use
            raise _AddonError(ERR_BAD_PARAMS, f'{exc}. Set Inference device to Automatic.') from exc
        raise
    log.info('VoxCPM2 loaded in %.1fs on %s (%s%s), %d Hz', time.monotonic() - started,
             _loaded.device, _loaded.dtype, ', diffusion in float32' if float32_head else '',
             _loaded.sample_rate)
    return _loaded


# ---------------------------------------------------------------------------
# Described voices: one anchor clip per speaker, cloned by every line
# ---------------------------------------------------------------------------
def _anchor(state: '_Request', loaded: engine.Loaded, description: str, seed: int,
            language_tag: str, options: dict) -> tuple[str, str, bool]:
    """(clip, its text, native) for this voice, making the clip if needed."""
    language, sentence, native = voices.anchor_for(language_tag)
    key = hashlib.sha256(
        f'{engine.MODEL_REVISION}\n{description}\n{seed}\n{language}'.encode('utf-8')).hexdigest()[:32]
    folder = engine.cache_root() / 'anchors'
    clip = folder / f'{key}.wav'
    if not clip.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        _emit_progress(state.req_id, 0.25, 'Making the voice...')
        log.info('anchor %s: %r in %s, seed %d', key, description, language, seed)
        wav = engine.synthesize(loaded, f'({description}){sentence}', cfg_value=options['cfg'],
                                steps=options['steps'], seed=seed, on_step=state.check)
        partial = clip.with_suffix('.part')
        _write_wav(partial, wav, loaded.sample_rate)
        os.replace(partial, clip)
    return str(clip), sentence, native


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------
class _Request:
    def __init__(self, req_id: str):
        self.req_id = req_id
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None

    def check(self, _step: int = 0) -> None:
        if self.cancel.is_set():
            raise engine.Cancelled()


_active: dict[str, _Request] = {}
_active_lock = threading.Lock()


def _write_wav(path: Path, wav, sample_rate: int) -> float:
    """16-bit mono WAV, which the host's dubbing pipeline expects. Returns seconds."""
    import numpy as np
    samples = np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (samples * 32767.0).round().astype('<i2')
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(pcm.tobytes())
    return len(samples) / float(sample_rate)


def _seed(params: dict) -> int:
    try:
        return int(params.get('seed', 0)) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return 0


def _handle_synthesize(state: _Request, params: dict) -> None:
    req_id = state.req_id
    text = ' '.join(str(params.get('text') or '').split())
    voice_id = str(params.get('voice') or '')
    output_path = params.get('output_path')
    if not text:
        _emit_error(req_id, ERR_BAD_PARAMS, 'text is empty')
        return
    if not isinstance(output_path, str) or not output_path.strip():
        _emit_error(req_id, ERR_BAD_PARAMS, 'output_path is required')
        return
    voice = voices.BY_ID.get(voice_id)
    if voice is None:
        _emit_error(req_id, ERR_UNSUPPORTED_VOICE, f'unknown voice {voice_id!r}')
        return
    options = _options()
    seed = _seed(params)

    # What the line clones: the reference the host sent, else the one in
    # Configure; or a described voice.
    reference = transcript = description = ''
    if voice.id == voices.CLONE:
        sent = str(params.get('voice_ref_audio') or '')
        if sent:
            reference, transcript = sent, ' '.join(str(params.get('voice_ref_text') or '').split())
        else:
            reference, transcript = options['voice_ref_audio'], options['voice_ref_text']
        if not reference:
            _emit_error(req_id, ERR_BAD_PARAMS,
                        "The speaker's own voice needs a clip of it, and none came with the request. "
                        'Pick a reference clip in Configure, or another voice.')
            return
        if not os.path.isfile(reference):
            _emit_error(req_id, ERR_BAD_PARAMS, f'The reference clip is missing: {reference}')
            return
    else:
        description = voice.description or options['voice_description']
        if not description:
            _emit_error(req_id, ERR_BAD_PARAMS,
                        'Describe the voice in Configure first (for example: '
                        '"An elderly woman, soft and warm voice").')
            return

    outcome: tuple = ('error', ERR_INTERNAL, 'no outcome', False)
    scratch = Path(tempfile.mkdtemp(prefix='voxcpm-', dir=_env('SUBTITLD_TEMP_DIR') or None))
    try:
        state.check()
        _emit_progress(req_id, 0.0, 'Waiting for VoxCPM2...')
        with _model_lock:
            state.check()
            loaded = _load_model(state, options)
            state.check()
            if voice.id == voices.CLONE:
                clip = _readable_reference(reference, scratch)
                mode = 'continuation' if transcript else 'reference'
                kwargs = {'reference_wav': clip}
                if transcript:
                    kwargs.update(prompt_wav=clip, prompt_text=transcript)
            else:
                clip, sentence, native = _anchor(state, loaded, description, seed,
                                                 str(params.get('language') or ''), options)
                mode = 'anchor-continuation' if native else 'anchor-reference'
                kwargs = {'reference_wav': clip}
                if native:
                    kwargs.update(prompt_wav=clip, prompt_text=sentence)

            # Progress: 0.3 -> 0.95 against the line's expected length, about
            # 14 characters a second of speech.
            expected = max(4.0, len(text) / 14.0 * engine.STEPS_PER_SECOND)

            def on_step(index: int) -> None:
                state.check()
                if index % 4 == 0:
                    _emit_progress(req_id, 0.3 + 0.65 * min(1.0, index / expected), 'Speaking...')

            log.info('synthesize %s: voice=%s mode=%s seed=%d steps=%d chars=%d',
                     req_id, voice.id, mode, seed, options['steps'], len(text))
            started = time.monotonic()
            wav = engine.synthesize(loaded, text, cfg_value=options['cfg'], steps=options['steps'],
                                    seed=seed, on_step=on_step, **kwargs)
            took = time.monotonic() - started
        state.check()
        duration = _write_wav(Path(output_path), wav, loaded.sample_rate)
        log.info('synthesize %s: %.2fs of audio in %.1fs', req_id, duration, took)
        outcome = ('result', {
            'path': output_path,
            'duration_sec': round(duration, 3),
            'sample_rate': loaded.sample_rate,
            'channels': 1,
            'voice': voice.id,
        })
    except engine.Cancelled:
        log.info('synthesize %s: cancelled', req_id)
        outcome = ('error', ERR_CANCELLED, 'cancelled', False)
    except _AddonError as exc:
        log.warning('synthesize %s failed: [%s] %s', req_id, exc.code, exc.message)
        outcome = ('error', exc.code, exc.message, exc.retryable)
    except MemoryError:
        outcome = ('error', ERR_OOM, 'Out of memory. VoxCPM2 needs about 6 GB free.', False)
    except Exception as exc:  # noqa: BLE001 — every path must end in one terminal frame
        log.exception('synthesize %s failed', req_id)
        if 'out of memory' in str(exc).lower():
            outcome = ('error', ERR_OOM, f'Out of memory: {exc}', False)
        else:
            outcome = ('error', ERR_INTERNAL, f'{type(exc).__name__}: {exc}', False)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if outcome[0] == 'result':
        _emit_progress(req_id, 1.0, '')
        _emit_result(req_id, outcome[1])
    else:
        _emit_error(req_id, *outcome[1:])


def _spawn(req_id: str, target, *args) -> None:
    state = _Request(req_id)
    with _active_lock:
        _active[req_id] = state

    def _run():
        try:
            target(state, *args)
        except BaseException:  # noqa: BLE001
            log.exception('worker crashed')
            tail = traceback.format_exc().strip().splitlines()[-1]
            _emit_error(req_id, ERR_INTERNAL, f'worker crashed: {tail}')
        finally:
            with _active_lock:
                _active.pop(req_id, None)

    thread = threading.Thread(target=_run, name=f'voxcpm-{req_id}', daemon=True)
    state.thread = thread
    thread.start()


# ---------------------------------------------------------------------------
# Handshake + frame loop
# ---------------------------------------------------------------------------
def _send_hello() -> None:
    _emit({
        'type': 'hello',
        'protocol': PROTOCOL_VERSION,
        'addon': ADDON_ID,
        'version': ADDON_VERSION,
        'capabilities': [{
            'task': 'tts.synthesize',
            'languages': list(voices.LANGUAGES),
            'voices': voices.manifest_voices(),
            'voice_clone': True,
        }],
    })


def _read_frame() -> dict | None:
    source = _PROTO_IN if _PROTO_IN is not None else sys.stdin.buffer
    line = source.readline()
    if not line:
        return None  # stdin closed: the host went away
    text = line.decode('utf-8', errors='replace').strip().lstrip('﻿')
    if not text:
        return {}
    try:
        frame = json.loads(text)
    except ValueError as exc:
        log.warning('malformed frame %r: %s', line[:200], exc)
        return {}
    return frame if isinstance(frame, dict) else {}


def _main_loop() -> int:
    while True:
        frame = _read_frame()
        if frame is None:
            log.info('stdin closed; exiting')
            return 0
        if not frame:
            continue
        ftype = frame.get('type')

        if ftype == 'shutdown':
            log.info('shutdown requested')
            return 0

        if ftype == 'ready':
            log.info('handshake complete; host=%s host_version=%s',
                     frame.get('host'), frame.get('host_version'))
            continue

        if ftype == 'cancel':
            target = frame.get('target')
            with _active_lock:
                state = _active.get(target) if isinstance(target, str) else None
            if state is not None:
                log.info('cancel %s', target)
                state.cancel.set()
            continue

        if ftype == 'tts.synthesize':
            req_id = frame.get('id')
            params = frame.get('params')
            if not isinstance(req_id, str) or not req_id:
                log.warning('tts.synthesize without id')
                continue
            if params is None:
                params = {}
            if not isinstance(params, dict):
                _emit_error(req_id, ERR_BAD_PARAMS, 'params must be an object')
                continue
            _spawn(req_id, _handle_synthesize, params)
            continue

        # Unknown task: still answer, so the host's future resolves instead of
        # hanging until its timeout.
        req_id = frame.get('id')
        if isinstance(req_id, str) and req_id:
            _emit_error(req_id, ERR_BAD_PARAMS, f'unsupported frame type {ftype!r}')


def _stop_workers(budget_sec: float = 3.5) -> bool:
    """Cancel and join every worker. True when none is left running.

    The host allows 5 s between `shutdown` and SIGTERM, so stay under it.
    """
    with _active_lock:
        states = list(_active.values())
    for state in states:
        state.cancel.set()
    deadline = time.monotonic() + budget_sec
    for state in states:
        if state.thread is not None:
            state.thread.join(timeout=max(0.0, deadline - time.monotonic()))
    return not any(s.thread is not None and s.thread.is_alive() for s in states)


def _self_test() -> int:
    """Exercise what the frozen bundle needs to synthesize, without the weights.

    Imports the whole inference stack and runs the reference-audio path
    (librosa's load, resampling and trim, which JIT-compiles through numba),
    so a module left out of the bundle fails the build instead of the user's
    first line.
    """
    import numpy as np
    import torch
    from voxcpm.model import voxcpm2 as v2
    from voxcpm.modules.audiovae import AudioVAEV2
    from voxcpm.modules.minicpm4.model import MiniCPMLongRoPE  # noqa: F401
    from safetensors.torch import load_file  # noqa: F401
    from transformers import LlamaTokenizerFast  # noqa: F401

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / 'tone.wav'
        tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(44100) / 44100)
        _write_wav(clip, np.concatenate([np.zeros(4410), tone, np.zeros(4410)]), 44100)
        audio, _sr = v2.librosa.load(str(clip), sr=16000, mono=True)
        trimmed = v2._trim_audio_silence_vad(torch.from_numpy(audio).unsqueeze(0), 16000)
    with torch.device('meta'):
        AudioVAEV2()
    report = {
        'ok': True,
        'version': ADDON_VERSION,
        'torch': torch.__version__,
        'threads': torch.get_num_threads(),
        'samples': int(trimmed.shape[-1]),
        'device': engine.resolve_device('auto'),
    }
    print(json.dumps(report), file=sys.__stdout__, flush=True)
    return 0


def main() -> int:
    if '--version' in sys.argv[1:]:
        print(f'voxcpm-addon {ADDON_VERSION}')
        return 0
    logging.basicConfig(
        level=(_env('VOXCPM_ADDON_LOG_LEVEL') or 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        stream=sys.stderr,
    )
    if '--self-test' in sys.argv[1:]:
        return _self_test()
    _claim_protocol_stdio()
    log.info('voxcpm-addon %s on %s %s, python %s',
             ADDON_VERSION, platform.system(), platform.machine(), platform.python_version())

    _send_hello()
    rc = 1
    try:
        rc = _main_loop()
    except KeyboardInterrupt:
        rc = 130
    except Exception:  # noqa: BLE001
        log.exception('main loop crashed')
        rc = 1
    finally:
        stopped = _stop_workers()
    if not stopped:
        # A worker is still inside a decoding step. Normal teardown would free
        # tensors under it, so leave without teardown.
        log.warning('a synthesis is still running; exiting without cleanup')
        try:
            sys.stderr.flush()
        finally:
            os._exit(rc)
    return rc


if __name__ == '__main__':
    sys.exit(main())
