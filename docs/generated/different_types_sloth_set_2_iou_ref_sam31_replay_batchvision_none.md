# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `none`
- correctness_pass: `False`
- mask_correctness_pass: `False`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `sam31-replay`
- init_source: `sam31-video-reference`
- prompt_source: `sam31_frame0_video_reference_masks`
- sam31_mask_root: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal/sam31_video_reference_masks`
- sam31_frame0_init_mask_root: `None`
- strict_full_batched: `False`
- disallow_partial_backend_success: `False`

## Blockers

- mask correctness gate failed: full backend contract passes, but 100-frame IoU drifts below threshold

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam0_obj1 | 0.9575342005157147 | 0.949313408723748 | 0.9575949856844386 | 93 | 93 |
| cam1_obj0 | 0.978494623655914 | 0.0 | 1.0 | 0 | 2 |
| cam1_obj1 | 0.9668657642239493 | 0.9582173469668991 | 0.9675188814043683 | 93 | 93 |
| cam2_obj0 | 0.989247311827957 | 0.0 | 1.0 | 0 | 1 |
| cam2_obj1 | 0.9660039565925866 | 0.9539005812064546 | 0.9667089173874143 | 93 | 93 |
