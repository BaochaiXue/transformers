# Full Batched EdgeTAM Batch Order Ablation

| field | value |
| --- | --- |
| order_dependent | False |
| diagonal_slicing_bug | False |

| order | first_bad | global_avg | global_min | cam1_avg | cam1_min |
| --- | --- | --- | --- | --- | --- |
| 0,1,2 | {'frame_idx': 43, 'output_index': 1, 'reference_camera': 'cam1', 'iou': 0.5920526014865638} | 0.97984 | 0.592053 | 0.96296 | 0.592053 |
| 1,0,2 | {'frame_idx': 43, 'output_index': 0, 'reference_camera': 'cam1', 'iou': 0.5920526014865638} | 0.97984 | 0.592053 | 0.96296 | 0.592053 |
| 2,1,0 | {'frame_idx': 43, 'output_index': 1, 'reference_camera': 'cam1', 'iou': 0.5920526014865638} | 0.97984 | 0.592053 | 0.96296 | 0.592053 |
| 1,1,1 | {'frame_idx': 43, 'output_index': 0, 'reference_camera': 'cam1', 'iou': 0.5920526014865638} | 0.96296 | 0.592053 | 0.96296 | 0.592053 |
