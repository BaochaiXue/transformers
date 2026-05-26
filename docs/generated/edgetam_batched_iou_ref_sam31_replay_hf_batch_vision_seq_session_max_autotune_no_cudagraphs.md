# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `max-autotune-no-cudagraphs`
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
| cam0_obj1 | 0.9758185395112144 | 0.9733034758056665 | 0.976110742365584 | 100 | 100 |
| cam1_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam1_obj1 | 0.9629748377147347 | 0.9581514762516046 | 0.9629796259399704 | 100 | 100 |
| cam2_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam2_obj1 | 0.9424180894747494 | 0.9388694688379231 | 0.9419574400819849 | 100 | 100 |
