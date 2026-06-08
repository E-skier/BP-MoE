---
license: other
license_name: fair-noncommercial-research-license
license_link: https://huggingface.co/facebook/blt/blob/main/LICENSE
extra_gated_fields:
  First Name: text
  Last Name: text
  Date of birth: date_picker
  Country: country
  Affiliation: text
  I accept the terms and conditions: checkbox
  geo: ip_location
language:
  - en
tags:
  - facebook
  - meta-pytorch
  - blt
---

# Byte Latent Transformer (BLT)

This repository contains the model weights for our paper: "Byte Latent Transformer: Patches Scale Better Than Tokens"

- [Paper Link](https://dl.fbaipublicfiles.com/blt/BLT__Patches_Scale_Better_Than_Tokens.pdf)
- [HF Paper Link](https://huggingface.co/papers/2412.09871)

## Abstract

We introduce the Byte Latent Transformer architecture (BLTs), a new byte-level LLM architecture that
for the first time, matches tokenization-based LLM performance at scale, with significant improvements
in inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve
as the primary units of computation. Patches are segmented dynamically based on the entropy of the
next byte, allocating more compute and model capacity where there is more data complexity. The BLT
architecture includes new attention mechanisms to maximize the information flow between byte and
patch hidden representations and a new type of byte-sequence memory. We present the first scaling
study of byte-level models up to 8B parameters and 8T training bytes, showing for the first time
that we can train a model end-to-end at scale from bytes with no tokenization or other preprocessing.
Scaling trends reveal training and inference efficiency benefits from dynamically selecting very long
patches on average, along with qualitative improvements with reasoning and long tail generalization
from modeling byte-sequences.

To run the model, see the readme here: https://github.com/facebookresearch/blt

## Links

- Code: https://github.com/facebookresearch/blt
- BLT 1B Weights: https://huggingface.co/facebook/blt-1b
- BLT 7B Weights: https://huggingface.co/facebook/blt-7b
- BLT Weight Collection: https://huggingface.co/collections/facebook/blt-6801263d4ac1704702a192a6

