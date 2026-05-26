# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `reduce-overhead`
- correctness_pass: `True`
- mask_correctness_pass: `True`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `sam31-replay`
- prompt_source: `sam31_frame0_video_reference_masks`
- sam31_mask_root: `/home/zhangxinjie/proj-QQTT-v2/result/demo22_rgb_triplet_100frames_towel_nonempty_stuffed_animal/sam31_video_reference_masks`

## Blockers

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam0_obj1 | 0.976010181280046 | 0.9725846022041357 | 0.9762884345856812 | 100 | 100 |
| cam1_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam1_obj1 | 0.9623791850271133 | 0.9591876208897485 | 0.9623375689322131 | 100 | 100 |
| cam2_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam2_obj1 | 0.958190413602625 | 0.951823695225079 | 0.9582234078177219 | 100 | 100 |
