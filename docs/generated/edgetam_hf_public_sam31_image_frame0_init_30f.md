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
- sam31_frame0_init_mask_root: `/home/zhangxinjie/proj-QQTT-v2/result/demo22_rgb_triplet_100frames_towel_nonempty_stuffed_animal/sam31_image_frame0_init_masks_green_towel`

## Blockers

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 1.0 | 1.0 | 1.0 | 30 | 30 |
| cam0_obj1 | 1.0 | 1.0 | 1.0 | 30 | 30 |
| cam1_obj0 | 1.0 | 1.0 | 1.0 | 30 | 30 |
| cam1_obj1 | 1.0 | 1.0 | 1.0 | 30 | 30 |
| cam2_obj0 | 1.0 | 1.0 | 1.0 | 30 | 30 |
| cam2_obj1 | 1.0 | 1.0 | 1.0 | 30 | 30 |
