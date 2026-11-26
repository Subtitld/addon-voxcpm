"""VoxCPM2 for the add-on: fetch the model, load it in audio.cpp, synthesize.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Runtime
-------
Speech is made by audio.cpp (Apache-2.0), a ggml runtime, through its C API
with ctypes: build/assets/audiocpp/ from source, `audiocpp/` inside the frozen
bundle. tools/build_native.py builds it with only VoxCPM2 in, and with one
patch: a callback before each generated step, by which a line reports its
progress and is stopped between two steps. ggml picks the CPU backend for the
processor it runs on.

Model
-----
VoxCPM2 as GGUF, its weights in 8-bit (Q8_0): 2.96 GB, about half the size of
the original bfloat16 checkpoint and, in our listening and transcription
tests, as good. It is fetched on the first synthesis from Hugging Face at a
pinned revision of audio.cpp's model repository, checked against its sha256,
into the host's model cache ($SUBTITLD_MODELS_DIR/voxcpm/VoxCPM2-GGUF). An
interrupted download resumes. $HF_ENDPOINT points it at a mirror;
$VOXCPM_MODEL_DIR at a folder that already holds the file, which is then used
as it is. The PyTorch weights earlier versions downloaded (4.96 GB) are
removed once the new file checks out.

Metal
-----
On Apple Silicon, synthesis runs on the GPU through ggml's Metal backend,
once a check in a child process has shown it works there (see metal_works);
elsewhere, and when it does not, on the CPU. $VOXCPM_DEVICE=cpu or metal
skips the check.

Threads
-------
On a CPU with performance and efficiency cores, the slower cores hold every
step back: synthesis uses one thread per performance core, and on Linux the
process is kept on those cores. Elsewhere it uses one thread per core.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import platform
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from voxcpm_addon import __version__, audio

log = logging.getLogger('voxcpm_addon.engine')

_PACKAGE = Path(__file__).resolve().parent

MODEL_REPO = 'audio-cpp/audio.cpp-gguf'
MODEL_REVISION = '32c1ce1fea57a513eb9386aa62408d47b1cf7ab2'
MODEL_DIRNAME = 'VoxCPM2-GGUF'

# What audio.cpp reads, at MODEL_REVISION: (path in the repository, bytes,
# sha256). Each is saved under its file name.
MODEL_FILES = (
    ('VoxCPM2-GGUF/voxcpm2-q8_0.gguf', 2955007488,
     '8df1faf274e1bdeedef1fd9ddd6aba6632536f66f2e080991d2352af35ceef72'),
)
MODEL_FILE = 'voxcpm2-q8_0.gguf'
MODEL_BYTES = sum(size for _name, size, _sha in MODEL_FILES)

# The PyTorch weights of the add-on's versions before audio.cpp.
TORCH_DIRNAME = 'VoxCPM2'
TORCH_REVISION = '32279effe8c19989596f05d353d1447f51d9e915'

_CHUNK = 1 << 20
_VERIFIED = '.verified'


class Cancelled(Exception):
    """The request was cancelled."""


class DownloadError(Exception):
    """The model could not be fetched or did not check out."""


class EngineError(Exception):
    """audio.cpp failed, or could not be loaded."""


# ---------------------------------------------------------------------------
# Where the model lives
# ---------------------------------------------------------------------------
def cache_root() -> Path:
    """$SUBTITLD_MODELS_DIR/voxcpm: the model and the described voices' anchors."""
    base = (os.environ.get('SUBTITLD_MODELS_DIR') or '').strip()
    root = Path(os.path.expanduser(base)) if base else Path.home() / '.cache' / 'subtitld' / 'models'
    return root / 'voxcpm'


def _user_folder() -> bool:
    return bool((os.environ.get('VOXCPM_MODEL_DIR') or '').strip())


def model_dir() -> Path:
    if _user_folder():
        return Path(os.path.expanduser(os.environ['VOXCPM_MODEL_DIR'].strip()))
    return cache_root() / MODEL_DIRNAME


def _local(name: str) -> str:
    return name.rsplit('/', 1)[-1]


def weights_present(folder: Path) -> bool:
    """True when every file is there at its expected size, checked once by sha256."""
    if _user_folder():
        return all((folder / _local(name)).is_file() for name, _size, _sha in MODEL_FILES)
    marker = folder / _VERIFIED
    try:
        if marker.read_text(encoding='utf-8').strip() != MODEL_REVISION:
            return False
    except OSError:
        return False
    return all((folder / _local(name)).is_file() and (folder / _local(name)).stat().st_size == size
               for name, size, _sha in MODEL_FILES)


def missing_bytes(folder: Path) -> int:
    """Bytes still to download, counting partial files as progress."""
    total = 0
    for name, size, _sha in MODEL_FILES:
        path = folder / _local(name)
        if path.is_file() and path.stat().st_size == size:
            continue
        part = folder / (_local(name) + '.part')
        have = part.stat().st_size if part.is_file() else 0
        total += max(0, size - have)
    return total


def _ssl_context() -> ssl.SSLContext:
    # voxcpm_addon.tls has set SSL_CERT_FILE for the frozen Linux and macOS
    # builds; elsewhere this is the platform's own trust store.
    return ssl.create_default_context()


def _sha256(path: Path, cancel: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_CHUNK * 8):
            if cancel():
                raise Cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, size: int, sha256: str,
           on_bytes: Callable[[int], None], cancel: Callable[[], bool]) -> None:
    part = dest.with_name(dest.name + '.part')
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        headers = {'User-Agent': f'subtitld-voxcpm-addon/{__version__}'}
        if have:
            headers['Range'] = f'bytes={have}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60, context=_ssl_context()) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the range: start over
                on_bytes(have)
                with part.open('ab' if have else 'wb') as out:
                    while chunk := response.read(_CHUNK):
                        if cancel():
                            raise Cancelled()
                        out.write(chunk)
                        on_bytes(len(chunk))
        except urllib.error.HTTPError as exc:
            raise DownloadError(f'{dest.name}: HTTP {exc.code} from {url}') from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise DownloadError(f'{dest.name}: {getattr(exc, "reason", exc)}') from exc
    got = part.stat().st_size
    if got != size:
        raise DownloadError(f'{dest.name}: got {got:,} bytes, expected {size:,}. Try again to resume.')
    if _sha256(part, cancel) != sha256:
        part.unlink()
        raise DownloadError(f'{dest.name} does not match its pinned sha256; the download was discarded.')
    os.replace(part, dest)


def ensure_weights(folder: Path, progress: Callable[[int, int], None],
                   cancel: Callable[[], bool]) -> None:
    """Download whatever is missing. `progress(done, total)` is in bytes."""
    if weights_present(folder):
        return
    if _user_folder():
        missing = [_local(name) for name, _s, _h in MODEL_FILES if not (folder / _local(name)).is_file()]
        raise DownloadError(f'VOXCPM_MODEL_DIR={folder} lacks {", ".join(missing)}')
    folder.mkdir(parents=True, exist_ok=True)
    endpoint = (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
    total = missing_bytes(folder)
    done = 0

    def on_bytes(count: int) -> None:
        nonlocal done
        done += count
        progress(min(done, total), total)

    for name, size, sha256 in MODEL_FILES:
        dest = folder / _local(name)
        if dest.is_file() and dest.stat().st_size == size and _sha256(dest, cancel) == sha256:
            continue
        part = folder / (_local(name) + '.part')
        before = part.stat().st_size if part.is_file() else 0
        done -= before  # _fetch reports the bytes already on disk again
        url = f'{endpoint}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}'
        log.info('fetching %s (%s bytes)', url, f'{size:,}')
        _fetch(url, dest, size, sha256, on_bytes, cancel)
    (folder / _VERIFIED).write_text(MODEL_REVISION + '\n', encoding='utf-8')


def retire_torch_weights() -> list[Path]:
    """Remove what the PyTorch versions of the add-on downloaded. Returns what went.

    Only from the host's model cache, and only the weights folder that carries
    their own verified marker; and the anchor clips those versions made, which
    the new model does not reuse.
    """
    if _user_folder():
        return []
    root = cache_root()
    old = root / TORCH_DIRNAME
    try:
        if (old / _VERIFIED).read_text(encoding='utf-8').strip() != TORCH_REVISION:
            return []
    except OSError:
        return []
    removed = [old]
    shutil.rmtree(old, ignore_errors=True)
    anchors = root / 'anchors'
    if anchors.is_dir():
        shutil.rmtree(anchors, ignore_errors=True)
        removed.append(anchors)
    return removed


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------
def _cpu_list(text: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    cpus: list[int] = []
    for part in text.strip().split(','):
        if '-' in part:
            low, high = part.split('-', 1)
            cpus.extend(range(int(low), int(high) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def performance_cpus() -> list[int]:
    """The logical CPUs of a hybrid processor's performance cores, on Linux; [] otherwise."""
    if not sys.platform.startswith('linux'):
        return []
    try:
        return _cpu_list(Path('/sys/devices/cpu_core/cpus').read_text())
    except (OSError, ValueError):
        return []


def _physical(cpus: list[int]) -> int:
    """How many physical cores the logical CPUs `cpus` are, on Linux."""
    cores = set()
    for cpu in cpus:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            cores.add(((topology / 'physical_package_id').read_text().strip(),
                       (topology / 'core_id').read_text().strip()))
        except OSError:
            cores.add(('cpu', str(cpu)))
    return len(cores)


def _sysctl(*names: str) -> list[int]:
    out = subprocess.run(['sysctl', '-n', *names], capture_output=True, text=True, timeout=5)
    return [int(line) for line in out.stdout.split()]


def default_threads() -> int:
    """One thread per performance core of a hybrid CPU, else one per core."""
    try:
        if sys.platform.startswith('linux'):
            cpus = performance_cpus() or sorted(os.sched_getaffinity(0))
            return max(1, _physical(cpus))
        if sys.platform == 'darwin':
            counts = _sysctl('hw.perflevel0.physicalcpu')
            return max(1, counts[0]) if counts else max(1, _sysctl('hw.physicalcpu')[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2 if logical >= 4 else logical)


def keep_to_performance_cores() -> list[int]:
    """On Linux with a hybrid CPU, keep this process on its performance cores. Returns them, or []."""
    cpus = performance_cpus()
    if not cpus:
        return []
    try:
        allowed = os.sched_getaffinity(0)
        chosen = [cpu for cpu in cpus if cpu in allowed]
        if chosen and set(chosen) != allowed:
            os.sched_setaffinity(0, chosen)
        return chosen
    except OSError:
        return []


# ---------------------------------------------------------------------------
# audio.cpp's C API (include/audiocpp.h, with tools/patches/)
# ---------------------------------------------------------------------------
LIBRARY = {'linux': 'libaudiocpp.so.0', 'darwin': 'libaudiocpp.0.dylib', 'win32': 'audiocpp.dll'}
ABI_MAJOR = 0

OK = 0
ERR_OUT_OF_MEMORY = 5
ERR_CANCELLED = 8

_LOG_LEVELS = {1: logging.DEBUG, 2: logging.DEBUG, 3: logging.WARNING, 4: logging.ERROR}


class _BackendConfig(ctypes.Structure):
    _fields_ = [('backend', ctypes.c_char_p), ('device', ctypes.c_int), ('threads', ctypes.c_int)]


class _ModelConfig(ctypes.Structure):
    _fields_ = [('family_hint', ctypes.c_char_p), ('config_id', ctypes.c_char_p),
                ('weight_id', ctypes.c_char_p), ('model_spec_override', ctypes.c_char_p)]


_STEP_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int64, ctypes.c_void_p)
_LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)


def runtime_dir() -> Path:
    """Where audio.cpp is: $VOXCPM_AUDIOCPP_DIR, the frozen bundle's, or build/assets/audiocpp/."""
    override = (os.environ.get('VOXCPM_AUDIOCPP_DIR') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    if getattr(sys, 'frozen', False):
        return Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)) / 'audiocpp'
    return _PACKAGE.parent / 'build' / 'assets' / 'audiocpp'


_lib = None
_log_callback = None  # kept alive for as long as the library may call it


def _on_log(level: int, text: bytes, _user) -> None:
    message = (text or b'').decode('utf-8', 'replace').rstrip()
    if message:
        log.log(_LOG_LEVELS.get(level, logging.DEBUG), 'audio.cpp: %s', message)


def load_library():
    """audio.cpp's C library, loaded once, with its functions typed."""
    global _lib, _log_callback
    if _lib is not None:
        return _lib
    folder = runtime_dir()
    path = folder / LIBRARY.get(sys.platform, LIBRARY['linux'])
    if not path.is_file():
        raise EngineError(f'audio.cpp is missing: no {path}')
    if sys.platform == 'win32':
        os.add_dll_directory(str(folder))
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise EngineError(f'could not load {path.name}: {exc}') from exc

    c = ctypes
    handle = c.POINTER(c.c_void_p)
    signatures = {
        'audiocpp_abi_version': (c.c_uint32, []),
        'audiocpp_build_version': (c.c_char_p, []),
        'audiocpp_last_error': (c.c_char_p, []),
        'audiocpp_log_set': (None, [_LOG_CALLBACK, c.c_void_p]),
        'audiocpp_registry_create': (c.c_int, [c.c_char_p, handle]),
        'audiocpp_registry_free': (None, [c.c_void_p]),
        'audiocpp_registry_family_count': (c.c_size_t, [c.c_void_p]),
        'audiocpp_registry_family': (c.c_int, [c.c_void_p, c.c_size_t, c.POINTER(c.c_char_p)]),
        'audiocpp_options_create': (c.c_void_p, []),
        'audiocpp_options_set': (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p]),
        'audiocpp_options_free': (None, [c.c_void_p]),
        'audiocpp_model_load': (c.c_int, [c.c_void_p, c.c_char_p, c.POINTER(_ModelConfig), c.c_void_p, handle]),
        'audiocpp_model_free': (None, [c.c_void_p]),
        'audiocpp_session_create': (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p, c.POINTER(_BackendConfig),
                                              c.c_void_p, handle]),
        'audiocpp_session_free': (None, [c.c_void_p]),
        'audiocpp_session_set_step_callback': (c.c_int, [c.c_void_p, _STEP_CALLBACK, c.c_void_p]),
        'audiocpp_session_run': (c.c_int, [c.c_void_p, c.c_void_p, handle]),
        'audiocpp_request_create': (c.c_void_p, []),
        'audiocpp_request_free': (None, [c.c_void_p]),
        'audiocpp_request_set_text': (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p]),
        'audiocpp_request_set_voice_audio': (c.c_int, [c.c_void_p, c.POINTER(c.c_float), c.c_size_t,
                                                       c.c_int, c.c_int]),
        'audiocpp_request_set_option': (c.c_int, [c.c_void_p, c.c_char_p, c.c_char_p]),
        'audiocpp_result_free': (None, [c.c_void_p]),
        'audiocpp_result_audio': (c.c_int, [c.c_void_p, c.POINTER(c.POINTER(c.c_float)), c.POINTER(c.c_size_t),
                                            c.POINTER(c.c_int), c.POINTER(c.c_int)]),
    }
    for name, (restype, argtypes) in signatures.items():
        try:
            function = getattr(lib, name)
        except AttributeError as exc:
            # audiocpp_session_set_step_callback is ours: a library without it
            # was built without tools/patches/, and a line could not be stopped.
            raise EngineError(f'{path.name} has no {name}: not the build tools/build_native.py makes') from exc
        function.restype = restype
        function.argtypes = argtypes
    abi = lib.audiocpp_abi_version()
    if abi >> 16 != ABI_MAJOR:
        raise EngineError(f'{path.name} has C ABI {abi >> 16}.{(abi >> 8) & 0xFF}, the add-on needs {ABI_MAJOR}.x')
    _log_callback = _LOG_CALLBACK(_on_log)
    lib.audiocpp_log_set(_log_callback, None)
    _lib = lib
    return lib


def runtime_version() -> str:
    lib = load_library()
    abi = lib.audiocpp_abi_version()
    return f'audio.cpp {lib.audiocpp_build_version().decode()} (C ABI {abi >> 16}.{(abi >> 8) & 0xFF})'


def families() -> list[str]:
    """The model families this build of audio.cpp has; ['voxcpm2'] for ours."""
    lib = load_library()
    registry = ctypes.c_void_p()
    _check(lib, lib.audiocpp_registry_create(None, ctypes.byref(registry)), 'registry')
    try:
        names = []
        for index in range(lib.audiocpp_registry_family_count(registry)):
            name = ctypes.c_char_p()
            _check(lib, lib.audiocpp_registry_family(registry, index, ctypes.byref(name)), 'registry')
            names.append(name.value.decode())
        return names
    finally:
        lib.audiocpp_registry_free(registry)


def _check(lib, status: int, what: str) -> None:
    if status == OK:
        return
    detail = (lib.audiocpp_last_error() or b'').decode('utf-8', 'replace')
    if status == ERR_CANCELLED:
        raise Cancelled()
    if status == ERR_OUT_OF_MEMORY:
        raise MemoryError(f'{what}: {detail}')
    raise EngineError(f'{what}: {detail}')


# ---------------------------------------------------------------------------
# Loading and synthesis
# ---------------------------------------------------------------------------
# Decoding steps per second of audio: 25 latent frames a second, 4 per step.
STEPS_PER_SECOND = 6.25

# Encoded reference clips the session keeps, so a speaker's next line skips
# encoding its clip again. A dub alternates between a few speakers.
PROMPT_CACHE_SLOTS = 8

# audio.cpp's ggml contexts hold only tensor and graph metadata (under 3 MB
# here), but VoxCPM2 reserves up to 3 GB for each by default: 5.6 GB in all,
# never touched. Linux and macOS overcommit; Haiku doesn't, so there the
# reservations alone used up its memory. 64 MB each leaves ample room.
HAIKU_CONTEXT_MB = 64
CONTEXT_OPTIONS = (
    'voxcpm2.weight_context_mb', 'voxcpm2.text_embedding_graph_arena_mb', 'voxcpm2.lm_step_graph_arena_mb',
    'voxcpm2.projection_graph_arena_mb', 'voxcpm2.local_encoder_graph_arena_mb', 'voxcpm2.dit_graph_arena_mb',
    'voxcpm2.audiovae_weight_context_mb', 'voxcpm2.audiovae_graph_arena_mb',
    'voxcpm2.audiovae_encoder_graph_arena_mb',
)


def session_options() -> dict[str, str]:
    """audio.cpp session options for this platform."""
    options = {'voxcpm2.prompt_cache_slots': str(PROMPT_CACHE_SLOTS)}
    if sys.platform.startswith('haiku'):
        options.update((key, str(HAIKU_CONTEXT_MB)) for key in CONTEXT_OPTIONS)
    return options


@dataclass
class Loaded:
    lib: object
    registry: ctypes.c_void_p
    model: ctypes.c_void_p
    session: ctypes.c_void_p
    threads: int
    cpus: list[int] = field(default_factory=list)
    backend: str = 'cpu'


def load(path: Path, threads: int = 0, backend: str = 'cpu') -> Loaded:
    """VoxCPM2 from the GGUF file at `path`, in a session on `backend` ('cpu' or 'metal')."""
    lib = load_library()
    cpus = keep_to_performance_cores()
    threads = threads if threads > 0 else default_threads()
    registry, model, session = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    options = lib.audiocpp_options_create()
    try:
        _check(lib, lib.audiocpp_registry_create(None, ctypes.byref(registry)), 'registry')
        _check(lib, lib.audiocpp_model_load(registry, str(path).encode(),
                                            ctypes.byref(_ModelConfig(b'voxcpm2', None, None, None)),
                                            None, ctypes.byref(model)), 'loading VoxCPM2')
        for key, value in session_options().items():
            _check(lib, lib.audiocpp_options_set(options, key.encode(), value.encode()), 'session options')
        _check(lib, lib.audiocpp_session_create(model, b'tts', b'offline',
                                                ctypes.byref(_BackendConfig(backend.encode(), 0, threads)),
                                                options, ctypes.byref(session)), 'session')
    except BaseException:
        lib.audiocpp_session_free(session)
        lib.audiocpp_model_free(model)
        lib.audiocpp_registry_free(registry)
        raise
    finally:
        lib.audiocpp_options_free(options)
    return Loaded(lib=lib, registry=registry, model=model, session=session, threads=threads, cpus=cpus,
                  backend=backend)


def close(loaded: Loaded) -> None:
    loaded.lib.audiocpp_session_free(loaded.session)
    loaded.lib.audiocpp_model_free(loaded.model)
    loaded.lib.audiocpp_registry_free(loaded.registry)


def synthesize(loaded: Loaded, text: str, *, reference: str | None = None,
               reference_text: str | None = None, cfg_value: float = 2.0, steps: int = 10,
               seed: int = 0, on_step: Callable[[int], None] | None = None) -> tuple[array, int]:
    """(float32 mono samples, their sample rate) of `text` spoken.

    `reference`, a 16-bit WAV file, alone clones the voice in it. With its
    exact transcript `reference_text` the line continues from the clip, which
    comes closer to the speaker. A text that starts with a description in
    parentheses describes the voice to make. `on_step(index)` runs before each
    decoding step; raising from it stops the line, and the exception comes out
    of here.
    """
    lib = loaded.lib
    stopped: list[BaseException] = []

    def step(index: int, _user) -> int:
        if on_step is None:
            return 0
        try:
            on_step(int(index))
        except BaseException as exc:  # noqa: BLE001 — carried out of the C call, raised below
            stopped.append(exc)
            return 1
        return 0

    callback = _STEP_CALLBACK(step)
    request = lib.audiocpp_request_create()
    result = ctypes.c_void_p()
    try:
        _check(lib, lib.audiocpp_request_set_text(request, text.encode('utf-8'), None), 'text')
        if reference:
            clip = audio.read_wav(reference)
            samples = (ctypes.c_float * len(clip.samples)).from_buffer(clip.samples)
            _check(lib, lib.audiocpp_request_set_voice_audio(request, samples, clip.frames, clip.rate,
                                                             clip.channels), 'reference clip')
            if reference_text:
                _check(lib, lib.audiocpp_request_set_option(request, b'reference_text',
                                                            reference_text.encode('utf-8')), 'transcript')
                _check(lib, lib.audiocpp_request_set_option(request, b'ultimate_clone', b'true'), 'transcript')
        for key, value in (('guidance_scale', f'{float(cfg_value):g}'), ('num_inference_steps', str(int(steps))),
                           ('seed', str(int(seed) & 0xFFFFFFFF))):
            _check(lib, lib.audiocpp_request_set_option(request, key.encode(), value.encode()), key)
        _check(lib, lib.audiocpp_session_set_step_callback(loaded.session, callback, None), 'step callback')
        try:
            status = lib.audiocpp_session_run(loaded.session, request, ctypes.byref(result))
        finally:
            lib.audiocpp_session_set_step_callback(loaded.session, _STEP_CALLBACK(), None)
        if stopped:
            raise stopped[0]
        _check(lib, status, 'synthesis')
        data, frames = ctypes.POINTER(ctypes.c_float)(), ctypes.c_size_t()
        rate, channels = ctypes.c_int(), ctypes.c_int()
        _check(lib, lib.audiocpp_result_audio(result, ctypes.byref(data), ctypes.byref(frames),
                                              ctypes.byref(rate), ctypes.byref(channels)), 'synthesis')
        out = array('f')
        out.frombytes(ctypes.string_at(data, frames.value * channels.value * 4))
        if channels.value > 1:
            n = channels.value
            out = array('f', (sum(out[i:i + n]) / n for i in range(0, len(out) - n + 1, n)))
        return out, rate.value
    finally:
        lib.audiocpp_result_free(result)
        lib.audiocpp_request_free(request)


# ---------------------------------------------------------------------------
# Metal, on Apple Silicon: tried once in a child process
# ---------------------------------------------------------------------------
# Where Metal is not up to it (the paravirtual GPU of a macOS virtual machine,
# at least), the first graph run on it can kill the process with a
# segmentation fault, which nothing in-process can catch. So before the
# add-on uses Metal, it speaks one short line on it in a child process
# (`--probe-metal`), and uses the CPU when the child does not come back with
# an answer. The verdict is kept in the model cache, per runtime and macOS
# version, so this happens once.
PROBE_TIMEOUT_SEC = 600  # loading compiles Metal's shaders; the line itself is short
_probe_lock = threading.Lock()


def apple_silicon() -> bool:
    return sys.platform == 'darwin' and platform.machine() == 'arm64'


def _verdict_file() -> Path:
    return cache_root() / 'metal-check.json'


def _probe_command(path: Path) -> list[str]:
    if getattr(sys, 'frozen', False):
        return [sys.executable, '--probe-metal', str(path)]
    return [sys.executable, '-m', 'voxcpm_addon', '--probe-metal', str(path)]


def metal_works(path: Path) -> bool:
    """Whether VoxCPM2 runs on Metal here; checked once, in a child process."""
    with _probe_lock:
        key = f'{runtime_version()} on macOS {platform.mac_ver()[0]}'
        verdicts = _verdict_file()
        try:
            saved = json.loads(verdicts.read_text(encoding='utf-8'))
            if saved.get('runtime') == key and isinstance(saved.get('voxcpm2'), bool):
                return saved['voxcpm2']
        except (OSError, ValueError, AttributeError):
            pass
        started = time.monotonic()
        try:
            done = subprocess.run(_probe_command(path), stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=PROBE_TIMEOUT_SEC)
            works = done.returncode == 0 and b'metal works' in done.stdout
            detail = f'exit {done.returncode}'
        except (OSError, subprocess.SubprocessError) as exc:
            works, detail = False, str(exc)
        log.info('Metal: %s (%s, %.1fs)', 'works' if works else 'fails; using the CPU', detail,
                 time.monotonic() - started)
        try:
            verdicts.parent.mkdir(parents=True, exist_ok=True)
            verdicts.write_text(json.dumps({'runtime': key, 'voxcpm2': works}) + '\n', encoding='utf-8')
        except OSError as exc:
            log.warning('could not save the Metal check: %s', exc)
        return works


def probe_metal(path: Path) -> None:
    """The child's side of metal_works(): one short line on the GPU. A crash is the answer too."""
    loaded = load(path, threads=default_threads(), backend='metal')
    try:
        samples, rate = synthesize(loaded, 'Hello there.', steps=4, seed=1)
        if len(samples) < rate // 4:
            raise EngineError(f'Metal made {len(samples)} samples of speech')
    finally:
        close(loaded)


def choose_backend(path: Path) -> str:
    """'metal' on Apple Silicon when it works there (or $VOXCPM_DEVICE says so), else 'cpu'."""
    device = (os.environ.get('VOXCPM_DEVICE') or '').strip().lower()
    if device == 'cpu' or not apple_silicon():
        return 'cpu'
    if device in ('metal', 'gpu', 'mps') or metal_works(path):
        return 'metal'
    return 'cpu'
