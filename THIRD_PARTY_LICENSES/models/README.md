# The model

The add-on downloads VoxCPM2 the first time a line is spoken; none of it is in
this archive. It is OpenBMB's [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2),
released under the Apache License 2.0 ([Apache-2.0.txt](Apache-2.0.txt)), as
converted to GGUF by the audio.cpp project and published in
[audio-cpp/audio.cpp-gguf](https://huggingface.co/audio-cpp/audio.cpp-gguf),
which lists it under the same licence. It is fetched at the revision
`engine.MODEL_REVISION` names (in the add-on's source, `voxcpm_addon/engine.py`)
and checked against its pinned sha256.

| File | What it is | When it downloads |
|------|------------|-------------------|
| `VoxCPM2-GGUF/voxcpm2-q8_0.gguf` (2.96 GB) | VoxCPM2 with its weights in 8-bit (Q8_0): the language models, the diffusion head, the audio VAE and the tokenizer | The first line |
