# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `none`
- correctness_pass: `True`
- mask_correctness_pass: `True`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `sam31-replay`
- prompt_source: `sam31_frame0_video_reference_masks`
- sam31_mask_root: `/home/zhangxinjie/proj-QQTT-v2/result/demo22_rgb_triplet_100frames_towel_stuffed_animal/sam31_video_reference_masks`

## Blockers

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam0_obj1 | 0.9757815294145089 | 0.9729482513671837 | 0.9760671377988359 | 100 | 100 |
| cam1_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam1_obj1 | 0.962383071897625 | 0.9579934485194939 | 0.9624091632778202 | 100 | 100 |
| cam2_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam2_obj1 | 0.9431770907480069 | 0.9403201089732675 | 0.9426344355191447 | 100 | 100 |
