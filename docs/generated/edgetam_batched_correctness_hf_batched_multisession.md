# EdgeTAM Batched Correctness

- backend: `hf_batched_multisession`
- compile_mode: `none`
- correctness_pass: `False`
- mask_correctness_pass: `True`
- candidate_partial: `True`
- fallback_backend: `hf_batch_vision_seq_session`

## Blockers

- hf_batched_multisession requires explicit HF session memory/object-pointer tensorization; current implementation falls back to batch vision + sequential session decode

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 1.0 | 1.0 | 1.0 | 100 | 100 |
| cam0_obj1 | 1.0 | 1.0 | 1.0 | 100 | 100 |
| cam1_obj0 | 1.0 | 1.0 | 1.0 | 100 | 100 |
| cam1_obj1 | 1.0 | 1.0 | 1.0 | 100 | 100 |
| cam2_obj0 | 1.0 | 1.0 | 1.0 | 100 | 100 |
| cam2_obj1 | 1.0 | 1.0 | 1.0 | 100 | 100 |
