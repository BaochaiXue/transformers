# Full Batched EdgeTAM Decoder Diff Probe

| field | value |
| --- | --- |
| frame_idx | 47 |
| camera | cam1 |
| raw_iou_vs_hf_public | 0.73297 |
| first_divergent_tensor | {'field': 'pred_masks', 'shape_match': True, 'max_abs_diff': 8.09375, 'mean_abs_diff': 5.0433125495910645, 'p95_abs_diff': 6.8125} |
| threshold_flip_count | 784 |
| reference_area | 2212 |
| candidate_area | 2876 |
| bbox_center_distance | 22.4722 |
| bbox_area_ratio | 2.11834 |

| tensor | shape_match | max_abs_diff | mean_abs_diff | p95_abs_diff |
| --- | --- | --- | --- | --- |
| pred_masks | True | 8.09375 | 5.04331 | 6.8125 |
| object_pointer | True | 1.79297 | 0.350936 | 0.855957 |
| object_score_logits | True | 0.03125 | 0.03125 | 0.03125 |
| maskmem_features | True | 5.4375 | 0.274955 | 1.07617 |
| maskmem_pos_enc | True | 0 | 0 | 0 |

## Note

This probe compares stored HF session outputs. Internal decoder tensors need a deeper runtime debug hook before they can be compared directly.
