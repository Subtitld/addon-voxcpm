"""Load build/assets/audiocpp with ctypes and speak one cloned line: proof the libraries work here.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

    python tools/smoke_native.py MODEL.gguf REFERENCE.wav OUT.wav [--threads N]

Standard library only, so it runs in the bare build container (glibc 2.28)
as well as on a desktop. The reference is a 16-bit mono WAV; its transcript is
optional (--reference-text). Exits non-zero unless the line has sound in it.
"""

import argparse
import array
import ctypes as C
import math
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = {'linux': 'libaudiocpp.so.0', 'darwin': 'libaudiocpp.0.dylib', 'win32': 'audiocpp.dll'}[sys.platform]


class BackendConfig(C.Structure):
    _fields_ = [('backend', C.c_char_p), ('device', C.c_int), ('threads', C.c_int)]


class ModelConfig(C.Structure):
    _fields_ = [('family_hint', C.c_char_p), ('config_id', C.c_char_p),
                ('weight_id', C.c_char_p), ('model_spec_override', C.c_char_p)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('model')
    parser.add_argument('reference')
    parser.add_argument('out')
    parser.add_argument('--reference-text', default='')
    parser.add_argument('--text', default='The robot hand works. We leave at dawn.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--runtime', default=str(ROOT / 'build' / 'assets' / 'audiocpp'))
    args = parser.parse_args()

    lib = C.CDLL(str(Path(args.runtime) / LIBRARY))
    lib.audiocpp_last_error.restype = C.c_char_p
    lib.audiocpp_build_version.restype = C.c_char_p
    lib.audiocpp_request_create.restype = C.c_void_p

    def ok(status):
        if status:
            sys.exit(f'audio.cpp: {lib.audiocpp_last_error().decode()}')

    print('audio.cpp', lib.audiocpp_build_version().decode(), 'ABI', lib.audiocpp_abi_version())
    started = time.monotonic()
    registry, model, session = C.c_void_p(), C.c_void_p(), C.c_void_p()
    ok(lib.audiocpp_registry_create(None, C.byref(registry)))
    ok(lib.audiocpp_model_load(registry, args.model.encode(), C.byref(ModelConfig(b'voxcpm2', None, None, None)),
                               None, C.byref(model)))
    ok(lib.audiocpp_session_create(model, b'tts', b'offline', C.byref(BackendConfig(b'cpu', 0, args.threads)),
                                   None, C.byref(session)))
    print(f'loaded in {time.monotonic() - started:.1f} s')

    with wave.open(args.reference) as ref:
        if ref.getsampwidth() != 2 or ref.getnchannels() != 1:
            sys.exit('the reference must be 16-bit mono')
        pcm = array.array('h', ref.readframes(ref.getnframes()))
        rate = ref.getframerate()
    samples = (C.c_float * len(pcm))(*(s / 32768 for s in pcm))

    request = C.c_void_p(lib.audiocpp_request_create())
    ok(lib.audiocpp_request_set_text(request, args.text.encode(), None))
    ok(lib.audiocpp_request_set_voice_audio(request, samples, C.c_size_t(len(pcm)), rate, 1))
    if args.reference_text:
        ok(lib.audiocpp_request_set_option(request, b'reference_text', args.reference_text.encode()))
        ok(lib.audiocpp_request_set_option(request, b'ultimate_clone', b'true'))
    ok(lib.audiocpp_request_set_option(request, b'seed', b'1234'))

    started = time.monotonic()
    result = C.c_void_p()
    ok(lib.audiocpp_session_run(session, request, C.byref(result)))
    took = time.monotonic() - started
    data, frames, out_rate, channels = C.POINTER(C.c_float)(), C.c_size_t(), C.c_int(), C.c_int()
    ok(lib.audiocpp_result_audio(result, C.byref(data), C.byref(frames), C.byref(out_rate), C.byref(channels)))
    values = data[:frames.value * channels.value]
    rms = math.sqrt(sum(v * v for v in values) / max(1, len(values)))
    with wave.open(args.out, 'wb') as out:
        out.setnchannels(channels.value)
        out.setsampwidth(2)
        out.setframerate(out_rate.value)
        out.writeframes(array.array('h', (int(max(-1.0, min(1.0, v)) * 32767) for v in values)).tobytes())
    seconds = frames.value / out_rate.value
    print(f'{seconds:.2f} s at {out_rate.value} Hz in {took:.1f} s (RTF {took / seconds:.2f}), rms {rms:.3f}')
    lib.audiocpp_result_free(result)
    lib.audiocpp_request_free(request)
    lib.audiocpp_session_free(session)
    lib.audiocpp_model_free(model)
    lib.audiocpp_registry_free(registry)
    if seconds < 1 or rms < 0.005:
        sys.exit('FAIL: no speech came out')
    print('OK')


if __name__ == '__main__':
    main()
