"""VoxCPM2 for the add-on: fetch the weights, load them lean, synthesize.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Weights
-------
VoxCPM2 is not in the archive: 4.96 GB, fetched on the first synthesis from
Hugging Face at a pinned revision, each file checked against its sha256, into
the host's model cache ($SUBTITLD_MODELS_DIR/voxcpm/VoxCPM2). An interrupted
download resumes. $HF_ENDPOINT points it at a mirror; $VOXCPM_MODEL_DIR at a
folder that already holds the files, which is then used as it is.

Loading
-------
Upstream's loader builds the model in float32 and copies the checkpoint into
it, so loading the 2.4B parameters peaks near 10 GB. Here the model is built
on PyTorch's `meta` device, without memory, and takes the checkpoint's tensors
as they are: the peak is about the size of the weights. The only tensors the
checkpoint does not hold, the rotary position tables, are rebuilt the way
upstream builds them. Upstream's `optimize` (torch.compile) is CUDA-only and
needs a compiler at run time, so it is not used.
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from voxcpm_addon import __version__

log = logging.getLogger('voxcpm_addon.engine')

MODEL_REPO = 'openbmb/VoxCPM2'
MODEL_REVISION = '32279effe8c19989596f05d353d1447f51d9e915'
MODEL_DIRNAME = 'VoxCPM2'

# What the inference path reads, at MODEL_REVISION: (file, bytes, sha256).
# The repo's tokenization_voxcpm2.py is not fetched: the tokenizer is loaded
# as LlamaTokenizerFast from tokenizer.json, so no code from the repo runs.
MODEL_FILES = (
    ('config.json', 4336, '405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c'),
    ('tokenizer.json', 3676772, 'f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20'),
    ('tokenizer_config.json', 5059, 'e78a3ebb48a0b9437efd1823b6b726c823da89e49dd8bcc90c02419d9baa772b'),
    ('special_tokens_map.json', 1632, '068594063e37662c02b21acf42ebb334ef6a74fb810e68a2368f88f08351de76'),
    ('audiovae.pth', 376951122, '94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1'),
    ('model.safetensors', 4580080592, 'f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d'),
)
MODEL_BYTES = sum(size for _name, size, _sha in MODEL_FILES)

_CHUNK = 1 << 20
_VERIFIED = '.verified'


class Cancelled(Exception):
    """The request was cancelled."""


class DownloadError(Exception):
    """The weights could not be fetched or did not check out."""


# ---------------------------------------------------------------------------
# Where the weights live
# ---------------------------------------------------------------------------
def cache_root() -> Path:
    """$SUBTITLD_MODELS_DIR/voxcpm: the weights and the described voices' anchors."""
    base = (os.environ.get('SUBTITLD_MODELS_DIR') or '').strip()
    root = Path(os.path.expanduser(base)) if base else Path.home() / '.cache' / 'subtitld' / 'models'
    return root / 'voxcpm'


def model_dir() -> Path:
    override = (os.environ.get('VOXCPM_MODEL_DIR') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    return cache_root() / MODEL_DIRNAME


def weights_present(folder: Path) -> bool:
    """True when every file is there at its expected size, checked once by sha256."""
    if (os.environ.get('VOXCPM_MODEL_DIR') or '').strip():
        return all((folder / name).is_file() for name, _size, _sha in MODEL_FILES)
    marker = folder / _VERIFIED
    try:
        if marker.read_text(encoding='utf-8').strip() != MODEL_REVISION:
            return False
    except OSError:
        return False
    return all((folder / name).is_file() and (folder / name).stat().st_size == size
               for name, size, _sha in MODEL_FILES)


def missing_bytes(folder: Path) -> int:
    """Bytes still to download, counting partial files as progress."""
    total = 0
    for name, size, _sha in MODEL_FILES:
        path = folder / name
        if path.is_file() and path.stat().st_size == size:
            continue
        part = folder / (name + '.part')
        have = part.stat().st_size if part.is_file() else 0
        total += max(0, size - have)
    return total


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # frozen macOS builds have no system CA bundle Python can read
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
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
    if (os.environ.get('VOXCPM_MODEL_DIR') or '').strip():
        missing = [name for name, _s, _h in MODEL_FILES if not (folder / name).is_file()]
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
        dest = folder / name
        if dest.is_file() and dest.stat().st_size == size and _sha256(dest, cancel) == sha256:
            continue
        part = folder / (name + '.part')
        before = part.stat().st_size if part.is_file() else 0
        done -= before  # _fetch reports the bytes already on disk again
        url = f'{endpoint}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}'
        log.info('fetching %s (%s bytes)', url, f'{size:,}')
        _fetch(url, dest, size, sha256, on_bytes, cancel)
    (folder / _VERIFIED).write_text(MODEL_REVISION + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# VoxCPM2 runs its decoding loop as `for i in tqdm(range(max_len))`. The tqdm
# it imported is swapped for this, which reports each step to the thread's
# callback: progress, and a cancel that stops a line between two steps. Every
# generation refills the KV cache from its prompt, so stopping one leaves
# nothing behind for the next.
_step = threading.local()

# Decoding steps per second of audio: 25 latent frames a second, 4 per step.
STEPS_PER_SECOND = 6.25


def _stepping(iterable, *_args, **_kwargs):
    callback = getattr(_step, 'callback', None)
    for index, item in enumerate(iterable):
        if callback is not None:
            callback(index)
        yield item


@dataclass
class Loaded:
    model: object          # voxcpm.VoxCPM
    device: str
    dtype: str
    sample_rate: int


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


def performance_cores() -> int:
    """Physical performance cores of a hybrid CPU; 0 when not hybrid or not known.

    PyTorch's default, one thread per physical core, counts efficiency cores
    too, and the slower cores then hold every step back: on a 13th-gen Intel
    with 8 performance and 8 efficiency cores, 8 threads synthesize about a
    quarter faster than 16.
    """
    try:
        if sys.platform.startswith('linux'):
            cpus = _cpu_list(Path('/sys/devices/cpu_core/cpus').read_text())
            siblings = Path(f'/sys/devices/system/cpu/cpu{cpus[0]}/topology/thread_siblings_list')
            return max(1, len(cpus) // max(1, len(_cpu_list(siblings.read_text()))))
        if sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu'],
                                 capture_output=True, text=True, timeout=5)
            counts = [int(line) for line in out.stdout.split()]
            return counts[0] if len(counts) == 2 and counts[1] > 0 else 0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return 0


def resolve_device(choice: str) -> str:
    import torch
    choice = (choice or 'auto').strip().lower()
    if choice == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return 'mps'
        return 'cpu'
    return choice


def resolve_precision(choice: str, device: str) -> tuple[str, bool]:
    """(the model's dtype, whether its diffusion head and encoder run in float32).

    Automatic is bfloat16, which halves the memory, with the two small parts
    that do most of the work in float32 on a CPU (see _in_float32). MPS needs
    float32 throughout (upstream's finding).
    """
    choice = (choice or 'auto').strip().lower()
    if device == 'mps':
        return 'float32', False
    if choice in ('bfloat16', 'float32'):
        return choice, False
    return 'bfloat16', device == 'cpu'


def _in_float32(module):
    """`module` run in float32 inside a bfloat16 model: tensors in, cast up; out, cast back.

    Used, on a CPU, for the diffusion head (VoxCPM2's LocDiT and its Euler solver)
    and the local encoder, 0.2B parameters each. The head does about three
    quarters of the work of a step: 12 layers, run twice (guidance) for each
    diffusion step. On CPUs without native bfloat16 the bfloat16 kernels are
    several times slower than float32 ones, so this roughly halves the time of
    a line for 0.8 GB more memory, and is no less accurate.
    """
    import torch

    def up(value):
        return value.float() if torch.is_tensor(value) and value.is_floating_point() else value

    class InFloat32(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner.float()

        def forward(self, *args, **kwargs):
            dtype = next(a.dtype for a in (*args, *kwargs.values())
                         if torch.is_tensor(a) and a.is_floating_point())
            out = self.inner(*map(up, args), **{k: up(v) for k, v in kwargs.items()})
            return out.to(dtype)

    return InFloat32(module)


def load(folder: Path, device: str, dtype: str, threads: int = 0,
         float32_head: bool = False) -> Loaded:
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaTokenizerFast
    from voxcpm.core import VoxCPM
    from voxcpm.model import voxcpm2 as v2
    from voxcpm.modules.audiovae import AudioVAEV2
    from voxcpm.modules.minicpm4.model import MiniCPMLongRoPE

    v2.tqdm = _stepping
    threads = threads if threads > 0 else performance_cores()
    if threads > 0:
        torch.set_num_threads(threads)
    config = v2.VoxCPMConfig.model_validate_json((folder / 'config.json').read_text(encoding='utf-8'))
    config.dtype = dtype
    tokenizer = LlamaTokenizerFast.from_pretrained(str(folder))
    with torch.device('meta'):
        audio_vae = AudioVAEV2(config=config.audio_vae_config)
        model = v2.VoxCPM2Model(config, tokenizer, audio_vae, None, device=device)
    lm_dtype = v2.get_dtype(model.config.dtype)

    # Each tensor goes straight to its device and dtype, so no second copy of
    # the model is ever made. On a CPU in bfloat16 `.to()` changes nothing and
    # the weights stay memory-mapped from the file.
    state = load_file(str(folder / 'model.safetensors'))
    for key in list(state):
        tensor = state[key]
        state[key] = tensor.to(device=device, dtype=lm_dtype if tensor.is_floating_point() else tensor.dtype)
    vae = torch.load(folder / 'audiovae.pth', map_location='cpu', weights_only=True)
    vae = vae.get('state_dict', vae)
    for key, tensor in vae.items():
        # Upstream keeps the audio VAE in float32 whatever the model's dtype.
        state[f'audio_vae.{key}'] = tensor.to(
            device=device, dtype=torch.float32 if tensor.is_floating_point() else tensor.dtype)
    del vae
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    if unexpected:
        raise RuntimeError(f'checkpoint has tensors the model does not: {unexpected[:5]}')

    # The rotary tables are computed, not stored: rebuild them, in the
    # model's dtype as upstream's `model.to(dtype)` leaves them.
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, MiniCPMLongRoPE):
                setattr(parent, name, MiniCPMLongRoPE(child.config).to(device=device, dtype=lm_dtype))

    on_meta = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
               if tensor.is_meta]
    if on_meta:
        raise RuntimeError(f'not loaded: {", ".join(on_meta[:5])}')
    if float32_head and lm_dtype != torch.float32:
        model.feat_decoder = _in_float32(model.feat_decoder)
        model.feat_encoder = _in_float32(model.feat_encoder)
    model = model.to(device).eval()

    wrapper = VoxCPM.__new__(VoxCPM)  # upstream's __init__ loads the model its own way
    wrapper.tts_model = model
    wrapper.text_normalizer = None
    wrapper.denoiser = None
    return Loaded(model=wrapper, device=device, dtype=model.config.dtype, sample_rate=int(model.sample_rate))


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def synthesize(loaded: Loaded, text: str, *, prompt_wav: str | None = None,
               prompt_text: str | None = None, reference_wav: str | None = None,
               cfg_value: float = 2.0, steps: int = 10, seed: int | None = None,
               on_step: Callable[[int], None] | None = None):
    """A float32 waveform at `loaded.sample_rate`.

    `reference_wav` alone clones the voice in it. `prompt_wav` with its exact
    transcript `prompt_text` continues from that clip, which comes closer to
    the speaker. A text that starts with a description in parentheses
    describes the voice to make. `on_step(index)` runs before each decoding
    step; raising from it stops the line.
    """
    import torch
    if seed is not None:
        torch.manual_seed(int(seed) & 0xFFFFFFFF)
    _step.callback = on_step
    try:
        with torch.inference_mode():
            return loaded.model.generate(
                text=text,
                prompt_wav_path=prompt_wav,
                prompt_text=prompt_text,
                reference_wav_path=reference_wav,
                cfg_value=float(cfg_value),
                inference_timesteps=int(steps),
                normalize=False,
                denoise=False,
            )
    finally:
        _step.callback = None
