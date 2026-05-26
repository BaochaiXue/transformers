# Different-types sloth_set_2 EdgeTAM batch=3 summary

- replay: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal`

## Primary Single-object Result

| field | value |
| --- | --- |
| object | stuffed animal |
| backend | hf_batch_vision_seq_session |
| compile | reduce-overhead |
| correctness_pass | True |
| empty_mismatch | 0 |
| stage_wall_p50_ms | 31.30548 |
| stage_wall_p90_ms | 32.98849 |
| stage_wall_p95_ms | 33.75403 |
| complete_group_fps_from_p50 | 31.94329 |
| p50_30fps_gate | True |
| p90_30fps_gate | True |
| p95_30fps_gate | False |

## Single-object Per-camera IoU

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.99739 | 0.98164 | 0.99808 | 93 | 93 |
| cam1_obj0 | 0.99274 | 0.97942 | 0.99697 | 93 | 93 |
| cam2_obj0 | 0.99696 | 0.98556 | 0.99741 | 93 | 93 |

## Two-object Compile Matrix

| compile | pass | global_avg | global_min | global_p50 | empty_mismatch | path |
| --- | --- | --- | --- | --- | --- | --- |
| max-autotune-no-cudagraphs | False | 0.9746 | 0.55547 | 0.99794 | 0 | docs/generated/different_types_sloth_set_2_edgetam_hand_stuffed_animal_batchvision_max_autotune_no_cudagraphs.json |
| none | True | 0.9961 | 0.91313 | 1.0 | 0 | docs/generated/different_types_sloth_set_2_edgetam_hand_stuffed_animal_batchvision_none.json |
| reduce-overhead | False | 0.97544 | 0.56741 | 0.99811 | 0 | docs/generated/different_types_sloth_set_2_edgetam_hand_stuffed_animal_batchvision_reduce_overhead.json |

## Hand Low-IoU Status

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.93087 | 0.56741 | 0.98018 | 93 | 93 |
| cam1_obj0 | 0.98933 | 0.93792 | 0.99584 | 93 | 93 |
| cam2_obj0 | 0.98172 | 0.68996 | 0.99732 | 93 | 93 |

## Stuffed Animal In Two-object Run

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj1 | 0.99739 | 0.98164 | 0.99808 | 93 | 93 |
| cam1_obj1 | 0.99274 | 0.97942 | 0.99697 | 93 | 93 |
| cam2_obj1 | 0.99696 | 0.98556 | 0.99741 | 93 | 93 |

## Decision

| field | value |
| --- | --- |
| single_object_stuffed_animal_validated | True |
| controller_hand_validated | False |
| controller_hand_reason | low IoU outliers on cam0/cam2 |
| best_single_object_backend | hf_batch_vision_seq_session |
| recommended_compile_mode | reduce-overhead |
| hf_batched_multisession_usable | False |
