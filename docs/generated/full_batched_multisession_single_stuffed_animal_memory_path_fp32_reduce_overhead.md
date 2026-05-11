# EdgeTAM Batched Correctness

- backend: `hf_batched_multisession`
- compile_mode: `reduce-overhead`
- correctness_pass: `False`
- mask_correctness_pass: `False`
- strict_correctness_pass: `False`
- speed_first_acceptance_pass: `True`
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

- mask correctness gate failed: gate=strict, evaluated=279/1, empty_mismatch=0/0, global_iou_avg=0.9434346992365232/0.98, global_iou_p50=0.9767501277465508/0.98

## Metrics

| key | iou_avg | iou_min | iou_p50 | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | ref_nonempty | cand_nonempty_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.9726443469528675 | 0.7685578605348663 | 0.9784731469152241 | 93 | 0 | 0 | 93 | 93 |
| cam1_obj0 | 0.8928888387807805 | 0.5992886781268524 | 0.9465705765407555 | 93 | 0 | 0 | 93 | 93 |
| cam2_obj0 | 0.9647709119759218 | 0.6979373567608862 | 0.9851038218477279 | 93 | 0 | 0 | 93 | 93 |

## Empty SAM3.1 Reference Policy

When SAM3.1 reference is empty and the policy is `ignore-candidate`, candidate masks are ignored for IoU and empty-mismatch. Ignored samples are unevaluated, not correct.
