# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `reduce-overhead`
- correctness_pass: `True`
- mask_correctness_pass: `True`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `hf-public`
- init_source: `sam31-image-frame0`
- prompt_source: `sam31_image_frame0_masks`
- sam31_mask_root: `None`
- sam31_frame0_init_mask_root: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal/sam31_image_frame0_init_masks_hand`

## Blockers

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.9973904583499003 | 0.981637337413925 | 0.9980821917808219 | 93 | 93 |
| cam1_obj0 | 0.9927393225047498 | 0.9794154619736015 | 0.9969706632653061 | 93 | 93 |
| cam2_obj0 | 0.9969611654158598 | 0.9855555555555555 | 0.9974082934609251 | 93 | 93 |
