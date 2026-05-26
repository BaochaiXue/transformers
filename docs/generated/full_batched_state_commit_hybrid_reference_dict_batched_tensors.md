# Full Batched EdgeTAM Teacher Forcing

| field | value |
| --- | --- |
| teacher_force_mode | after_memory_encoder |
| status | ok |
| first_bad_frame | {'frame_idx': 47, 'camera': 'cam1', 'iou': 0.7351535836177474} |
| min_iou | 0.7351535836177474 |
| global_iou_avg | 0.995299918865833 |
| global_iou_p50 | 0.9977257220832386 |
| component_replaced | memory_encoder_outputs |
| drift_source_hypothesis | drift_not_eliminated_by_after_memory_encoder |
| reason |  |

## State Commit Interpretation

drift remains after forcing memory outputs; decoder/object pointer remains suspect
