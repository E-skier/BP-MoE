# PatchMoE Internal Proposal

## Project Identity

**PatchMoE** is a conditional-computation architecture for tokenizer-free language
modeling. The goal is not to reproduce large-scale BLT pretraining, but to test
whether dynamic byte patches are a better sparse-routing unit than tokenizer
tokens.

The central claim is:

> When language models move from fixed tokens to dynamic byte patches, sparse
> expert routing and load balancing should also become patch-native.

This proposal follows the roadmap in
`Byte_Patch_MoE_A_Conference_Roadmap.md`. The first paper should emphasize
architecture and controlled evidence rather than industrial-scale model quality.

## Core Question

How should MoE routing change when each routed unit is a variable-length byte
patch with its own entropy, length, byte pattern, and compute cost?

The initial implementation will test the simplest version:

1. Route global patch representations instead of byte or token representations.
2. Compare hidden-only patch routing against dense patch FFNs.
3. Log expert usage and load imbalance at patch granularity.
4. Add entropy-/length-/cost-aware routing after the hidden-only path is stable.

## Method Sketch

The existing BLT pipeline already gives the right architectural split:

- local byte encoder;
- dynamic or fixed patch construction;
- global patch-level Transformer;
- local byte decoder.

PatchMoE replaces selected global Transformer FFNs with sparse MoE FFNs. In the
Phase-1 implementation, each patch hidden state is routed by a learned linear
router:

```text
patch hidden state -> router logits -> Top-K experts -> weighted expert output
```

This deliberately starts as hidden-only routing. Phase 2 will extend the router
with side information:

- patch entropy;
- patch byte length;
- byte/script/code-pattern features;
- expert congestion or cost price.

The first load-balancing objective is patch-count balancing. Later ablations
will replace it with byte-count, entropy-weighted, and latency/FLOP-weighted
definitions.

## Minimal Engineering Scope

The first milestone is a small, optional patch-level MoE path in the global
latent Transformer. It must be disabled by default and preserve all dense BLT
behavior when `moe_num_experts == 0`.

Required configuration knobs:

- `moe_num_experts`: number of FFN experts; `0` disables MoE;
- `moe_top_k`: number of experts selected per patch;
- `moe_layer_frequency`: replace every Nth global FFN with MoE;
- `moe_balance_loss_weight`: optional patch-count load-balancing loss weight;
- `moe_router_jitter`: optional training noise for routing exploration.

Required early metrics:

- mean router entropy;
- per-layer expert load imbalance;
- max expert load fraction;
- average selected expert load fraction;
- MoE auxiliary balance loss.

## Stage-0 Experiments

Stage-0 should be small enough to debug quickly:

- 50M-100M active parameters;
- 4-8 experts;
- Top-1 or Top-2 routing;
- fixed patching first, then existing entropy patching;
- 1B-5B training bytes once smoke tests are stable.

The first plots should be:

1. patch length and entropy distribution;
2. BPB training curve;
3. expert load distribution.

Only after these plots exist should the project move toward Stage-1 ablations.

## Exit Criteria

Phase 1 is complete when:

- dense BLT still runs unchanged;
- patch-level MoE forward/backward passes run;
- training loss decreases on a small run;
- expert load metrics are logged;
- router collapse can be detected from metrics.

Phase 2 begins after this by adding entropy-aware and length-aware routing to
the same MoE interface.
