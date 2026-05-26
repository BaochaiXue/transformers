# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `none`
- correctness_pass: `False`
- mask_correctness_pass: `False`
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
- correctness_gate: `strict`
- speed_first_gate: `False`
- evaluated_sample_count: `279`
- ignored_reference_empty_count: `279`
- candidate_nonempty_when_reference_empty_count: `3`
- strict_full_batched: `False`
- disallow_partial_backend_success: `False`

## Blockers

- mask correctness gate failed: gate=strict, evaluated=279/1, empty_mismatch=0/0, global_iou_avg=0.963467973777417/0.98, global_iou_p50=0.9644944427553124/0.98

## Metrics

| key | iou_avg | iou_min | iou_p50 | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | ref_nonempty | cand_nonempty_eval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cam0_obj0 |  |  |  | 0 | 93 | 0 | 0 | 0 |
| cam0_obj1 | 0.9575342005157147 | 0.949313408723748 | 0.9575949856844386 | 93 | 0 | 0 | 93 | 93 |
| cam1_obj0 |  |  |  | 0 | 93 | 2 | 0 | 0 |
| cam1_obj1 | 0.9668657642239493 | 0.9582173469668991 | 0.9675188814043683 | 93 | 0 | 0 | 93 | 93 |
| cam2_obj0 |  |  |  | 0 | 93 | 1 | 0 | 0 |
| cam2_obj1 | 0.9660039565925866 | 0.9539005812064546 | 0.9667089173874143 | 93 | 0 | 0 | 93 | 93 |

## Empty SAM3.1 Reference Policy

When SAM3.1 reference is empty and the policy is `ignore-candidate`, candidate masks are ignored for IoU and empty-mismatch. Ignored samples are unevaluated, not correct.
