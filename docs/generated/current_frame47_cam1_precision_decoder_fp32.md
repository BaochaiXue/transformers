# Full Batched EdgeTAM Current-Frame Isolation

| field | value |
| --- | --- |
| frame_idx | 47 |
| camera | cam1 |
| replace | none |
| precision_mode | decoder_fp32 |
| effective_dtype | bfloat16 |
| force_reference_state_before_frame | True |
| inferred_issue | no_current_frame_divergence_under_probe |

| variant | status | raw_iou | reported_iou | pred_p95 | ptr_p95 | mem_p95 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_batch | ok | 1 | 1 | 0.0575497 | 0.00592553 | 0.143213 |  |
| repeated_target | ok | 1 | 1 | 0.0575497 | 0.00592553 | 0.143213 |  |
| black_neighbors | ok | 1 | 1 | 0.0575497 | 0.00592553 | 0.143213 |  |
| shuffled_target_first | ok | 1 | 1 | 0.0575497 | 0.00592553 | 0.143213 |  |
| batch1_wrapper | ok | 1 | 1 | 0.0574124 | 0.00510922 | 0.140625 |  |
