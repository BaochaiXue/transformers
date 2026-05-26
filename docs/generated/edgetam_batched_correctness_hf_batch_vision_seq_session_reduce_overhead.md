# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `reduce-overhead`
- correctness_pass: `False`
- mask_correctness_pass: `False`
- candidate_partial: `False`
- fallback_backend: `None`

## Blockers

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.6213235304951561 | 0.14622720428308517 | 0.6918678538954522 | 100 | 100 |
| cam0_obj1 | 0.8480335089675325 | 0.4762826718296225 | 0.9362974265997133 | 100 | 100 |
| cam1_obj0 | 0.9762894429143499 | 0.882587064676617 | 0.9796241903727965 | 100 | 100 |
| cam1_obj1 | 0.9664198286651594 | 0.823599931495119 | 0.9689816738600197 | 100 | 100 |
| cam2_obj0 | 0.9351395332889267 | 0.8064729194187582 | 0.9502452950755468 | 100 | 100 |
| cam2_obj1 | 0.9722986107935496 | 0.8266152934202726 | 0.9770560736285311 | 100 | 100 |
