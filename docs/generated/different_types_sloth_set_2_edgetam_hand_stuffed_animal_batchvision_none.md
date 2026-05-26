# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `none`
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
| cam0_obj0 | 1.0 | 1.0 | 1.0 | 93 | 93 |
| cam0_obj1 | 1.0 | 1.0 | 1.0 | 93 | 93 |
| cam1_obj0 | 0.9869489408125531 | 0.9131268840778295 | 0.9892924323542179 | 93 | 93 |
| cam1_obj1 | 0.9943448183126459 | 0.9789440603394092 | 0.9977474095209491 | 93 | 93 |
| cam2_obj0 | 0.9995337784484487 | 0.9952194697957409 | 0.999637943519189 | 93 | 93 |
| cam2_obj1 | 0.998869989382603 | 0.988110964332893 | 0.9993681550126369 | 93 | 93 |
