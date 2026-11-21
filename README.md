# VoxCPM add-on for Subtitld

Text-to-speech for [Subtitld](https://subtitld.org) dubbing with OpenBMB's
[VoxCPM2](https://github.com/OpenBMB/VoxCPM): 30 languages at 48 kHz, in the
speaker's own voice cloned from the video, or in a voice made from a
description. Apache-2.0, the model weights included, so dubs can be used
commercially.

## Voices

- **Speaker's own voice (cloned from the video).** Subtitld cuts a clip of the
  speaker from the video and sends it with the subtitle text spoken in it.
  VoxCPM2 then continues from the clip, which comes closest to the speaker.
  Turn on *Clone the timbre only* in Configure to clone the voice without the
  clip's delivery: try it when a dub into another language keeps the
  speaker's accent, or when the subtitles paraphrase the speech.
- **Described voices**: Warm woman, Bright young woman, Gentle older woman,
  Calm man, Energetic young man, Gravelly older man, Narrator, and *Voice you
  describe*, which takes your own words from Configure ("An elderly woman,
  soft and warm voice, slow pace"). VoxCPM2 invents a new voice every time it
  reads a description, so the add-on makes each speaker's voice once, as a
  fixed sentence in the line's language, and every line of that speaker clones
  it. Two speakers given the same described voice still sound like two people.

The voice is chosen per speaker in Subtitld's dubbing panel.

## Languages

Arabic, Burmese, Chinese, Danish, Dutch, English, Finnish, French, German,
Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay,
Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog,
Thai, Turkish and Vietnamese, plus several Chinese dialects. VoxCPM2 reads the
language from the text.

## What it needs

- **Disk:** about 3.2 GB. The model, 2.96 GB, downloads from Hugging Face on
  the first synthesis, at a pinned revision, checked by sha256, into
  Subtitld's model cache. An interrupted download resumes. `HF_ENDPOINT`
  points it at a mirror; `VOXCPM_MODEL_DIR` at a folder that already holds
  the file. The PyTorch weights that versions before 26.11.1 downloaded
  (4.96 GB) are removed once the new model checks out.
- **Memory:** about 4.5 GB free. The model is memory-mapped, so with less free
  memory the add-on still runs, but far slower, rereading it from disk.
- **Time:** VoxCPM2 has 2.4B parameters, and on a CPU each second of speech
  takes several seconds to make: about 4.5 seconds on an otherwise idle Intel
  Core i7-13700T (8 performance cores) with the defaults. *Quality steps* at 5 is
  faster, with audibly less clarity. The add-on runs on the CPU, or on the GPU
  of an Apple Silicon Mac.

After the download nothing leaves the machine.

## Settings

| Setting | Default | |
|---|---|---|
| Quality steps | 10 | diffusion steps; fewer is faster |
| Guidance | 2.0 | how closely speech follows the text and voice |
| CPU threads | 0 | the performance cores on a CPU that has efficiency cores too, else one per core |
| Voice description | | for *Voice you describe* |
| Fallback reference clip / transcript | | for the cloned voice when Subtitld has no clip to send |
| Clone the timbre only | off | see Voices |

## How it runs the model

Speech is made by [audio.cpp](https://github.com/0xShug0/audio.cpp)
(Apache-2.0), a C++ runtime on ggml, which the add-on calls through its C API
with ctypes. `tools/build_native.py` builds it from source at a pinned commit
with only VoxCPM2 compiled in, and applies one patch,
`tools/patches/audiocpp-step-callback.patch`: a callback before each generated
step, so a line reports its progress and stops between two steps when it is
cancelled. Upstream's C API runs a line to its end. ggml picks the build for
the processor it runs on, from SSE4.2 to AVX-512. On an Apple Silicon Mac it
runs on the GPU (Metal), once a short test line in a separate process has
shown that works there; `VOXCPM_DEVICE=cpu` or `metal` skips the test.

The model is VoxCPM2 with its weights in 8-bit (Q8_0), as audio.cpp's project
converted it: 2.96 GB, against 4.96 GB for the original bfloat16 checkpoint.
On the same lines, it was transcribed as accurately and sounded as close to
the speaker as the original. On the same machine it made them about a fifth
faster than the PyTorch build of this add-on, in half the memory.

The add-on itself needs nothing but Python's standard library.

## Building

Linux, in the AlmaLinux 8 container, so the archive runs on glibc 2.28 and
newer (needs Docker):

```bash
tools/build_linux.sh                      # audio.cpp, tests, the bundle, the archive
tools/test_archive.sh almalinux:8         # the archive, installed as Subtitld does, on AlmaLinux 8
```

`VOXCPM_TEST_MODELS_DIR` (a folder holding `voxcpm/VoxCPM2-GGUF/`, or an empty
one to download into) also has the archive speak and cancel for real. For the
engine tests with the model, from source, set `VOXCPM_TEST_MODEL` to the GGUF
file:

```bash
python tools/build_native.py              # Linux: tools/build_native_linux.sh
pip install -e '.[build,dev]'
VOXCPM_TEST_MODEL=/path/to/voxcpm2-q8_0.gguf pytest
pyinstaller pyinstaller.spec --noconfirm
dist/voxcpm-addon/voxcpm-addon --self-test
```

The release workflow builds, proves and publishes every platform on a tag
`v<version>` matching `manifest.json`; a manual run can build only some
platforms (`targets`). `tools/assemble_licences.py` fails the build when a
native file in the bundle has no notice.

## Licence

The add-on is Apache-2.0. audio.cpp is Apache-2.0, with ggml, cJSON and
libyaml (MIT) compiled in; the VoxCPM2 weights are Apache-2.0 (OpenBMB).
Each archive carries `THIRD_PARTY_LICENSES/` with the licence of everything
frozen into it, and our patch to audio.cpp.

Voice cloning makes speech that sounds like a real person. Clone voices you
have the right to use, and say that a dub is synthetic where it matters.
