# Full Batched EdgeTAM Current-Frame Isolation

| field | value |
| --- | --- |
| frame_idx | 47 |
| camera | cam1 |
| replace | none |
| precision_mode | all_fp32 |
| effective_dtype | float32 |
| force_reference_state_before_frame | True |
| inferred_issue | precision_sensitive_current_frame_path |

| variant | status | raw_iou | reported_iou | pred_p95 | ptr_p95 | mem_p95 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_batch | ok | 0.99908 | 0.99908 | 0.01009 | 0.00219432 | 0.0104502 |  |
| repeated_target | ok | 0.99908 | 0.99908 | 0.01009 | 0.00219432 | 0.0104502 |  |
| black_neighbors | ok | 0.99908 | 0.99908 | 0.01009 | 0.00219432 | 0.0104502 |  |
| shuffled_target_first | ok | 0.99908 | 0.99908 | 0.01009 | 0.00219432 | 0.0104502 |  |
| batch1_wrapper | ok | 1 | 1 | 0 | 0 | 0 |  |
