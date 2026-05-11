# Full Batched EdgeTAM State Drift Curve

- backend: `hf_batched_multisession`
- dtype: `bfloat16`
- replay: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal`
- global_iou_avg: `0.97984`
- global_iou_p50: `0.98993`
- first_iou_lt_0.90: `{'frame_idx': 43, 'camera': 'cam1', 'value': 0.5920526014865638, 'threshold': 0.9}`
- first_tensor_drift: `{'frame_idx': 0, 'camera': 'cam0', 'field': 'maskmem_features', 'p95_abs_diff': 0.015625, 'threshold': 0.001}`

| field | p95_diff_p50 | p95_diff_p90 | first >1e-3 | first >0.1 | first >1.0 |
| --- | --- | --- | --- | --- | --- |
| pred_masks | 0.4375 | 1.38125 | {'frame_idx': 1, 'camera': 'cam0', 'value': 0.0625, 'threshold': 0.001} | {'frame_idx': 2, 'camera': 'cam0', 'value': 0.125, 'threshold': 0.1} | {'frame_idx': 33, 'camera': 'cam1', 'value': 1.25, 'threshold': 1.0} |
| maskmem_features | 0.414062 | 0.90585 | {'frame_idx': 0, 'camera': 'cam0', 'value': 0.015625, 'threshold': 0.001} | {'frame_idx': 1, 'camera': 'cam0', 'value': 0.18626709282398224, 'threshold': 0.1} | {'frame_idx': 38, 'camera': 'cam1', 'value': 1.107421875, 'threshold': 1.0} |
| maskmem_pos_enc | 0 | 0 |  |  |  |
| object_pointer | 0.119263 | 0.429517 | {'frame_idx': 0, 'camera': 'cam0', 'value': 0.0068359375, 'threshold': 0.001} | {'frame_idx': 9, 'camera': 'cam0', 'value': 0.10595703125, 'threshold': 0.1} |  |
| object_score_logits | 0.125 | 1.09375 | {'frame_idx': 1, 'camera': 'cam0', 'value': 0.0625, 'threshold': 0.001} | {'frame_idx': 7, 'camera': 'cam2', 'value': 0.125, 'threshold': 0.1} | {'frame_idx': 60, 'camera': 'cam0', 'value': 1.03125, 'threshold': 1.0} |
