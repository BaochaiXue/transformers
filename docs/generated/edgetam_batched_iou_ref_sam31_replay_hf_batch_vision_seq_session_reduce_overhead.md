# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `reduce-overhead`
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
| cam0_obj1 | 0.9747222236286013 | 0.970013037809648 | 0.9746943366418392 | 100 | 100 |
| cam1_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam1_obj1 | 0.9626904310869283 | 0.9590158668979251 | 0.962640259434445 | 100 | 100 |
| cam2_obj0 | 1.0 | 1.0 | 1.0 | 0 | 0 |
| cam2_obj1 | 0.9437309635169875 | 0.9383450973722255 | 0.942975178851374 | 100 | 100 |
