# Full Batched EdgeTAM State Drift Curve

- backend: `hf_batched_multisession`
- dtype: `bfloat16`
- replay: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal`
- global_iou_avg: `0.987242`
- global_iou_p50: `0.990599`
- first_iou_lt_0.90: `{'frame_idx': 50, 'camera': 'cam2', 'value': 0.8460878517501715, 'threshold': 0.9}`
- first_tensor_drift: `{'frame_idx': 0, 'camera': 'cam0', 'field': 'maskmem_features', 'p95_abs_diff': 0.02552780508995056, 'threshold': 0.001}`

| field | p95_diff_p50 | p95_diff_p90 | first >1e-3 | first >0.1 | first >1.0 |
| --- | --- | --- | --- | --- | --- |
| pred_masks | 0.41962 | 0.930768 | {'frame_idx': 1, 'camera': 'cam0', 'value': 0.07201862335205078, 'threshold': 0.001} | {'frame_idx': 2, 'camera': 'cam0', 'value': 0.14121675491333008, 'threshold': 0.1} | {'frame_idx': 20, 'camera': 'cam2', 'value': 6.9928202629089355, 'threshold': 1.0} |
| maskmem_features | 0.423245 | 0.584294 | {'frame_idx': 0, 'camera': 'cam0', 'value': 0.02552780508995056, 'threshold': 0.001} | {'frame_idx': 1, 'camera': 'cam0', 'value': 0.3041016459465027, 'threshold': 0.1} | {'frame_idx': 66, 'camera': 'cam1', 'value': 1.3342756032943726, 'threshold': 1.0} |
| maskmem_pos_enc | 0.00128418 | 0.00128418 | {'frame_idx': 0, 'camera': 'cam0', 'value': 0.001284182071685791, 'threshold': 0.001} |  |  |
| object_pointer | 0.0790241 | 0.405921 | {'frame_idx': 0, 'camera': 'cam0', 'value': 0.005421601235866547, 'threshold': 0.001} | {'frame_idx': 16, 'camera': 'cam2', 'value': 0.11493729054927826, 'threshold': 0.1} | {'frame_idx': 77, 'camera': 'cam2', 'value': 1.0763732194900513, 'threshold': 1.0} |
| object_score_logits | 0.0831947 | 0.268903 | {'frame_idx': 1, 'camera': 'cam0', 'value': 0.035308837890625, 'threshold': 0.001} | {'frame_idx': 2, 'camera': 'cam1', 'value': 0.10554218292236328, 'threshold': 0.1} |  |
