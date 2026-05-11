# EdgeTAM batch=3 multi-session runtime experiment

## Goal

Original HF EdgeTAM weights plus a custom batch=3 multi-session runtime.
This is not a change to the HF public `model(inference_session=..., frame=...)` API.

## Source

- fork path: `/home/zhangxinjie/EdgeTAM-HF-batched`
- branch: `feat/edgetam-batched-multisession-runtime`
- upstream commit: `e9fd66d116c1dc03f85803931a1a3a940057488a`
- HF source touched: no
- generated `modeling_edgetam_video.py` touched: no
- wrapper files added: `edgetam_batched/*`

## Current Status

Implemented first-stage correctness scaffolding:

- source inspection report
- synthetic session state map
- stackability/mutation classification
- camera-order test helper
- state-leakage test helper
- compile config guard
- CUDA graph output ring-buffer helper
- correctness/profile entrypoint stubs

The real batch=3 multi-session runtime is not claimed complete yet. No real 100-frame RGB replay was found at:

```text
/home/zhangxinjie/proj-QQTT-v2/result/demo22_rgb_triplet_100frames_towel_stuffed_animal
```

Therefore no valid model performance or correctness claim is made from this fork yet.

## Session State Map

- session count: 3
- field count: 61
- tensor fields: 14
- stackable tensor fields: 14
- metadata fields: 47
- mutated fields: 10

Report:

```text
docs/generated/edgetam_batched_session_state_map.md
docs/generated/edgetam_batched_session_state_map.json
```

## Performance

Two profile commands were run in this fork:

```text
docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none_scaffold.json
docs/generated/edgetam_batched_profile_hf_ref_seq_public_synthetic.json
```

The `hf_ref_seq_public_synthetic` profile does load and run the HF model, but it uses synthetic RGB/masks and sequential public sessions. It is not a validated batched multi-session runtime result:

```text
synthetic hf_ref_seq_public sequential 3-session:
  stage_wall p50: 76.00 ms
  stage_wall p90: 80.39 ms
  cam0 p50: 24.08 ms
  cam1 p50: 24.50 ms
  cam2 p50: 25.60 ms
```

The current valid real-scene performance baseline remains the QQTT Demo 2.1.5 result:

```text
hf_batch_vision_seq_session + vision-reduce-overhead:
  stage_wall p50: 77.92 ms
  stage_wall p90: 86.24 ms
  complete group FPS: 12.48
```

## Decision

- usable as Demo 2.2 backend now: no
- recommended backend now: existing QQTT `hf_batch_vision_seq_session`
- next required step: create/load real 100-frame RGB triplet replay, implement reference runtime stepping, then run correctness before any compile/profile claim
