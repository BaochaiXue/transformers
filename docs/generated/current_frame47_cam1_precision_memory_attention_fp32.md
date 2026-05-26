# Full Batched EdgeTAM Current-Frame Isolation

| field | value |
| --- | --- |
| frame_idx | 47 |
| camera | cam1 |
| replace | none |
| precision_mode | memory_attention_fp32 |
| effective_dtype | bfloat16 |
| force_reference_state_before_frame | True |
| inferred_issue | no_current_frame_divergence_under_probe |

| variant | status | raw_iou | reported_iou | pred_p95 | ptr_p95 | mem_p95 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_batch | ok | 0.999548 | 0.999548 | 0.0625 | 0.0109863 | 0.148438 |  |
| repeated_target | ok | 0.999548 | 0.999548 | 0.0625 | 0.0109863 | 0.148438 |  |
| black_neighbors | ok | 0.999548 | 0.999548 | 0.0625 | 0.0109863 | 0.148438 |  |
| shuffled_target_first | ok | 0.999548 | 0.999548 | 0.0625 | 0.0109863 | 0.148438 |  |
| batch1_wrapper | ok | 0.999548 | 0.999548 | 0.0625 | 0.0078125 | 0.140625 |  |
