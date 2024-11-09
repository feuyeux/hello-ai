# [MusicLM](https://arxiv.org/abs/2301.11325)

## Models

- **SoundStream** AudioLM, self-supervised audio representations
- **w2v-BERT** acoustic tokens to enable high-fidelity synthesis
- **MuLan music embedding**[training], **MuLan text embedding**[inference time] semantic tokens to facilitate long-term coherent generation

Residual Vector Quantization(RVQ) 残差矢量量化

## Training & Inference

- semantic modeling stage (语义建模阶段)
- acoustic modeling stage (声学建模阶段)

![music_lm](../img/music_lm.png)

Left:

During **training** we extract the MuLan audio tokens, semantic tokens, and acoustic tokens from the *audio-only* training set.

In the **semantic modeling stage**, we predict semantic tokens using MuLan audio tokens as conditioning.

In the subsequent **acoustic modeling stage**, we predict acoustic tokens, given both MuLan audio tokens and semantic tokens. Each stage is modeled as a sequence-to-sequence task using decoder-only Transformers.

Right:

During **inference**, we use MuLan text tokens computed from the text prompt as conditioning signal and convert the generated audio tokens to waveforms using the SoundStream decoder.

## Evaluation Dataset

[MusicCaps](https://www.kaggle.com/datasets/googleai/musiccaps)

## Metrics

We compute different metrics to evaluate MusicLM, capturing two important aspects of music generation:

- the audio quality
- the adherence to the text description.

## Stuff

- <https://www.youtube.com/watch?v=N_s0Tc_Tfa0>
- <https://www.youtube.com/watch?v=JBqurD1KG_w>

## REPO

- <https://github.com/lucidrains/musiclm-pytorch>
