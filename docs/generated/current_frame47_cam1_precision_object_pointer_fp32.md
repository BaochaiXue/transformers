# Full Batched EdgeTAM Current-Frame Isolation

| field | value |
| --- | --- |
| frame_idx | 47 |
| camera | cam1 |
| replace | none |
| precision_mode | object_pointer_fp32 |
| effective_dtype | bfloat16 |
| force_reference_state_before_frame | True |
| inferred_issue | batch3_dimension_handling_or_diagonal_slicing_suspect |

| variant | status | raw_iou | reported_iou | pred_p95 | ptr_p95 | mem_p95 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_batch | ok | 0.734722 | 0.734722 | 6.8125 | 0.849121 | 1.06216 |  |
| repeated_target | ok | 0.734722 | 0.734722 | 6.8125 | 0.849121 | 1.06216 |  |
| black_neighbors | ok | 0.734722 | 0.734722 | 6.8125 | 0.849121 | 1.06216 |  |
| shuffled_target_first | ok | 0.734722 | 0.734722 | 6.8125 | 0.849121 | 1.06216 |  |
| batch1_wrapper | ok | 1 | 1 | 0.0625 | 0.0078125 | 0.128906 |  |
