# Full Batched EdgeTAM Current-Frame Isolation

| field | value |
| --- | --- |
| frame_idx | 47 |
| camera | cam1 |
| replace | postprocess_output_with_reference |
| precision_mode | all_bf16 |
| effective_dtype | bfloat16 |
| force_reference_state_before_frame | True |
| inferred_issue | posthoc_replacement_reaches_reference_mask; rerun-level hook still needed |

| variant | status | raw_iou | reported_iou | pred_p95 | ptr_p95 | mem_p95 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_batch | ok | 0.73297 | 1 | 6.8125 | 0.855957 | 1.07617 | replacement is applied to the reported current mask after the candidate step; it confirms downstream scoring sensitivity but does not rerun upstream components |
