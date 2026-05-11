# EdgeTAM Batched Correctness

- backend: `hf_batched_multisession`
- compile_mode: `none`
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

- mask correctness gate failed: gate=strict, evaluated=279/1, empty_mismatch=0/0, global_iou_avg=0.9490332435244935/0.98, global_iou_p50=0.9864885380294519/0.98

## Metrics

| key | iou_avg | iou_min | iou_p50 | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | ref_nonempty | cand_nonempty_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.9882313432305199 | 0.9741626794258373 | 0.9872712808273667 | 93 | 0 | 0 | 93 | 93 |
| cam1_obj0 | 0.8761449490088216 | 0.5635206199833933 | 0.9648382559774965 | 93 | 0 | 0 | 93 | 93 |
| cam2_obj0 | 0.9827234383341388 | 0.7647345348270823 | 0.9891752577319588 | 93 | 0 | 0 | 93 | 93 |

## Empty SAM3.1 Reference Policy

When SAM3.1 reference is empty and the policy is `ignore-candidate`, candidate masks are ignored for IoU and empty-mismatch. Ignored samples are unevaluated, not correct.
