# Full Batched EdgeTAM Current-Frame Isolation

| field | value |
| --- | --- |
| frame_idx | 47 |
| camera | cam1 |
| replace | object_pointer_with_reference |
| precision_mode | all_bf16 |
| effective_dtype | bfloat16 |
| force_reference_state_before_frame | True |
| inferred_issue | replacement_level_requires_runtime_debug_hook |

| variant | status | raw_iou | reported_iou | pred_p95 | ptr_p95 | mem_p95 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_batch | ok | 0.73297 | 0.73297 | 6.8125 | 0.855957 | 1.07617 | runtime does not yet expose an in-frame component replacement hook for this level |
