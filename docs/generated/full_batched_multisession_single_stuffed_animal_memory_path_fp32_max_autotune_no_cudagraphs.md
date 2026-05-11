# EdgeTAM Batched Correctness

- backend: `hf_batched_multisession`
- compile_mode: `max-autotune-no-cudagraphs`
- correctness_pass: `False`
- mask_correctness_pass: `False`
- strict_correctness_pass: `False`
- speed_first_acceptance_pass: `False`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `hf-public`
- init_source: `deterministic`
- prompt_source: `deterministic_replay_boxes`
- sam31_mask_root: `None`
- sam31_frame0_init_mask_root: `None`
- empty_reference_policy: `strict-empty-mismatch`
- correctness_gate: `strict`
- speed_first_gate: `False`
- evaluated_sample_count: `279`
- ignored_reference_empty_count: `0`
- candidate_nonempty_when_reference_empty_count: `0`
- strict_full_batched: `True`
- disallow_partial_backend_success: `True`

## Blockers

- mask correctness gate failed: gate=strict, evaluated=279/1, empty_mismatch=0/0, global_iou_avg=0.9296243356994457/0.98, global_iou_p50=0.9718191964285714/0.98

## Metrics

| key | iou_avg | iou_min | iou_p50 | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | ref_nonempty | cand_nonempty_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.9566273240604154 | 0.7687389770723104 | 0.9712280701754386 | 93 | 0 | 0 | 93 | 93 |
| cam1_obj0 | 0.8685225856103136 | 0.5999406880189798 | 0.947436133063775 | 93 | 0 | 0 | 93 | 93 |
| cam2_obj0 | 0.9637230974276078 | 0.7273298758447273 | 0.9822289156626506 | 93 | 0 | 0 | 93 | 93 |

## Empty SAM3.1 Reference Policy

When SAM3.1 reference is empty and the policy is `ignore-candidate`, candidate masks are ignored for IoU and empty-mismatch. Ignored samples are unevaluated, not correct.
