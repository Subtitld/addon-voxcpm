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

- **Disk:** about 5.6 GB. The model, 4.96 GB, downloads from Hugging Face on
  the first synthesis, at a pinned revision, each file checked by sha256,
  into Subtitld's model cache. An interrupted download resumes. `HF_ENDPOINT`
  points it at a mirror; `VOXCPM_MODEL_DIR` at a folder that already holds
  the files.
- **Memory:** about 7 GB free with the default precision, 11 GB in float32.
  The weights are memory-mapped, so with less free memory the add-on still
  runs, but far slower, rereading them from disk.
- **Time:** VoxCPM2 has 2.4B parameters, and on a CPU each second of speech
  takes several seconds to make. On an Intel Core i7-13700T (8 performance
  cores) with the defaults it is 5 to 6 seconds, a little less in float32.
  *Quality steps* at 5 is about a quarter faster, with audibly less clarity.
  The add-on runs on the CPU, or on the GPU of an Apple Silicon Mac; this
  build does not use NVIDIA GPUs.

After the download nothing leaves the machine.

## Settings

| Setting | Default | |
|---|---|---|
| Inference device | Automatic | CPU, or the Apple GPU (MPS) |
| Weight precision | Automatic | bfloat16, with the diffusion in float32 on a CPU; float32 on the Apple GPU, which needs it |
| Quality steps | 10 | diffusion steps; fewer is faster |
| Guidance | 2.0 | how closely speech follows the text and voice |
| CPU threads | 0 | the performance cores on a CPU that has efficiency cores too, else one per core |
| Voice description | | for *Voice you describe* |
| Fallback reference clip / transcript | | for the cloned voice when Subtitld has no clip to send |
| Clone the timbre only | off | see Voices |

## Building

Python 3.10 to 3.12. VoxCPM itself is installed without its dependencies:
those add a web demo, training and a denoiser, none of which the add-on uses.

```bash
# Linux and Windows: the CPU build of PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchaudio==2.11.0
pip install -c constraints.txt -e '.[build,dev]'
pip install --no-deps -c constraints.txt voxcpm
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/voxcpm-addon/voxcpm-addon --self-test
```

`tools/build-linux.sh` makes the Linux archive in `ubuntu:22.04`, as the
release workflow does. A tag `v<version>` matching `manifest.json` builds and
publishes all three platforms.

## How it runs the model

Upstream's loader builds the 2.4B-parameter model in float32 before copying
the checkpoint in, which peaks near 10 GB. This add-on builds it on PyTorch's
`meta` device and hands it the memory-mapped checkpoint, so the peak is about
the size of the weights.

On a CPU the model stays in bfloat16 except its diffusion head and local
encoder, 0.2B parameters each, which run in float32. The head does about three
quarters of the work, and most CPUs have no native bfloat16: on the
i7-13700T it halves the time of a line, from 9 to 11 seconds per second of
speech in bfloat16 throughout to 5 to 6, for 0.8 GB more memory. See `voxcpm_addon/engine.py`.

## Licence

The add-on is Apache-2.0. VoxCPM's code and the VoxCPM2 weights are
Apache-2.0 (OpenBMB). Each archive carries `THIRD_PARTY_LICENSES/` with the
licence of everything frozen into it.

Voice cloning makes speech that sounds like a real person. Clone voices you
have the right to use, and say that a dub is synthetic where it matters.
