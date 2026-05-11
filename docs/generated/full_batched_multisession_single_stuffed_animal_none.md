# EdgeTAM Batched Correctness

- backend: `hf_batched_multisession`
- compile_mode: `none`
- correctness_pass: `False`
- mask_correctness_pass: `False`
- candidate_partial: `False`
- fallback_backend: `None`
- reference_source: `hf-public`
- init_source: `deterministic`
- prompt_source: `deterministic_replay_boxes`
- sam31_mask_root: `None`
- sam31_frame0_init_mask_root: `None`
- strict_full_batched: `True`
- disallow_partial_backend_success: `True`

## Blockers

- mask correctness gate failed: full backend contract passes, but 100-frame IoU drifts below threshold

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.9799253844552651 | 0.9297407912687585 | 0.9788418708240535 | 93 | 93 |
| cam1_obj0 | 0.9599975483281635 | 0.674901185770751 | 0.985007072135785 | 93 | 93 |
| cam2_obj0 | 0.974707358767951 | 0.7350816582914573 | 0.9906890130353817 | 93 | 93 |
