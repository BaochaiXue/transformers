# EdgeTAM Batched Correctness

- backend: `hf_ref_seq_public`
- compile_mode: `none`
- correctness_pass: `True`
- mask_correctness_pass: `True`
- strict_correctness_pass: `False`
- speed_first_acceptance_pass: `True`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `sam31-replay`
- init_source: `sam31-video-reference`
- prompt_source: `sam31_frame0_video_reference_masks`
- sam31_mask_root: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal/sam31_video_reference_masks`
- sam31_frame0_init_mask_root: `None`
- empty_reference_policy: `ignore-candidate`
- correctness_gate: `speed-first`
- speed_first_gate: `True`
- evaluated_sample_count: `279`
- ignored_reference_empty_count: `0`
- candidate_nonempty_when_reference_empty_count: `0`
- strict_full_batched: `False`
- disallow_partial_backend_success: `False`

## Blockers

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | ref_nonempty | cand_nonempty_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.9575342005157147 | 0.949313408723748 | 0.9575949856844386 | 93 | 0 | 0 | 93 | 93 |
| cam1_obj0 | 0.9667510728516525 | 0.9584172066731008 | 0.9675638371290545 | 93 | 0 | 0 | 93 | 93 |
| cam2_obj0 | 0.9660371199068802 | 0.9540648792985241 | 0.9665497507845671 | 93 | 0 | 0 | 93 | 93 |

## Empty SAM3.1 Reference Policy

When SAM3.1 reference is empty and the policy is `ignore-candidate`, candidate masks are ignored for IoU and empty-mismatch. Ignored samples are unevaluated, not correct.
