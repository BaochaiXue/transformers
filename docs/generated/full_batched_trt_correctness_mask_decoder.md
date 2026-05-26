# EdgeTAM Batched Correctness

- backend: `hf_batched_multisession_trt_components`
- compile_mode: `none`
- correctness_pass: `True`
- mask_correctness_pass: `True`
- strict_correctness_pass: `True`
- speed_first_acceptance_pass: `True`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `hf-public-seq`
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

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | ref_nonempty | cand_nonempty_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.9912242527037276 | 0.9796252927400468 | 0.9921046695240243 | 93 | 0 | 0 | 93 | 93 |
| cam1_obj0 | 0.9900760970215405 | 0.9628252788104089 | 0.9921813917122753 | 93 | 0 | 0 | 93 | 93 |
| cam2_obj0 | 0.9812876324695882 | 0.7627091115803151 | 0.988556041736789 | 93 | 0 | 0 | 93 | 93 |

## Empty SAM3.1 Reference Policy

When SAM3.1 reference is empty and the policy is `ignore-candidate`, candidate masks are ignored for IoU and empty-mismatch. Ignored samples are unevaluated, not correct.
