# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `max-autotune-no-cudagraphs`
- correctness_pass: `False`
- mask_correctness_pass: `False`
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
| cam0_obj0 | 0.9276431708835573 | 0.5554749818709209 | 0.9835766423357665 | 93 | 93 |
| cam0_obj1 | 0.9973461957361621 | 0.9835878732619102 | 0.9979450295402004 | 93 | 93 |
| cam1_obj0 | 0.9837704077307765 | 0.9164137931034483 | 0.9872924187725631 | 93 | 93 |
| cam1_obj1 | 0.9932661606619054 | 0.9808404443728869 | 0.9970109101778508 | 93 | 93 |
| cam2_obj0 | 0.9906964110277803 | 0.6915384615384615 | 0.9979389942291839 | 93 | 93 |
| cam2_obj1 | 0.997583670331675 | 0.9878210806023029 | 0.997791519434629 | 93 | 93 |
