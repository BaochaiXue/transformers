# EdgeTAM batch=3 multi-session final report

## Goal

Original weights + custom batch=3 multi-session runtime.

## Source

| field | value |
| --- | --- |
| github_repo | https://github.com/BaochaiXue/transformers/tree/feat/edgetam-batched-multisession-runtime |
| fork_path | /home/zhangxinjie/EdgeTAM-HF-batched |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 9521446b1c8b555ff020a3024280cb64af1ae58a |
| modeling_edgetam_video_touched | False |

## Correctness

| backend | compile | pass | mask_pass | partial | fallback | contract_pass | path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hf_batched_multisession | none | False | False | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_precision_all_bf16.json |
| hf_batched_multisession | none | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_precision_all_fp32.json |
| hf_batched_multisession | none | False | False | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_precision_decoder_fp32.json |
| hf_batched_multisession | none | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_precision_memory_attention_fp32.json |
| hf_batched_multisession | none | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_precision_memory_path_fp32.json |
| hf_batched_multisession | max-autotune-no-cudagraphs | False | False | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_max_autotune_no_cudagraphs.json |
| hf_batched_multisession | reduce-overhead | False | False | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_reduce_overhead.json |

## Profiles

| backend | compile | p50 | p90 | partial | path |
| --- | --- | --- | --- | --- | --- |

## SAM3.1 replay reference correctness

No SAM3.1 replay IoU summary provided.

## Non-empty object quality: stuffed animal only

No stuffed animal IoU rows available.

## Controller/towel caveat

Controller/towel reference is not empty in all cameras for the provided summary.

## Different-types sloth_set_2 result

No different-types summary provided.

## Empty SAM3.1 reference policy

When SAM3.1 reference is empty, EdgeTAM candidate output is ignored for IoU and empty-mismatch. The sample is marked reference_absent_ignored. This is not counted as correct; it is unevaluated.

## Original vs compiled delta on evaluated samples

No original-vs-compiled delta report provided.

## Full batched first bad frame

| field | value |
| --- | --- |
| frame_idx | 43 |
| camera | cam1 |
| iou | 0.59205 |
| first_diverging_component | mask_decoder_or_accumulated_state |
| backend_contract_pass | True |

| tensor | max_abs_diff | mean_abs_diff | p95_abs_diff |
| --- | --- | --- | --- |
| maskmem_features | 5.46875 | 0.21898 | 1.04263 |
| maskmem_pos_enc | 0.0 | 0.0 | 0.0 |
| object_pointer | 0.95508 | 0.20126 | 0.49171 |
| object_score_logits | 0.21875 | 0.21875 | 0.21875 |
| pred_masks | 7.98438 | 1.93154 | 3.25 |

## Full batched recurrent drift localization

### Component equivalence fixtures

| component | groups | passed | pass | max_abs_diff | p95_abs_diff | path |
| --- | --- | --- | --- | --- | --- | --- |
| mask_decoder | 30 | 30 | True | 0.3125 | 0.0625 | docs/generated/component_equivalence_mask_decoder_single_object.json |
| memory_attention | 29 | 29 | True | 0.0625 | 0.00781 | docs/generated/component_equivalence_memory_attention_single_object.json |
| memory_encoder | 30 | 30 | True | 0.11719 | 0.00391 | docs/generated/component_equivalence_memory_encoder_single_object.json |
|  |  |  | True |  |  | docs/generated/component_equivalence_vision_encoder_single_object.json |

### State drift curve

| field | value |
| --- | --- |
| mask_iou_avg | 0.98724 |
| mask_iou_min | 0.76316 |
| first_iou_lt_0_90 | {'camera': 'cam2', 'frame_idx': 50, 'threshold': 0.9, 'value': 0.8460878517501715} |
| first_iou_lt_0_98 | {'camera': 'cam0', 'frame_idx': 3, 'threshold': 0.98, 'value': 0.9799414018480955} |
| first_tensor_drift | {'camera': 'cam0', 'field': 'maskmem_features', 'frame_idx': 0, 'p95_abs_diff': 0.02552780508995056, 'threshold': 0.001} |
| maskmem_features_first_p95_gt_1 | {'camera': 'cam1', 'frame_idx': 66, 'threshold': 1.0, 'value': 1.3342756032943726} |
| object_pointer_first_p95_gt_1e_1 | {'camera': 'cam2', 'frame_idx': 16, 'threshold': 0.1, 'value': 0.11493729054927826} |
| maskmem_pos_enc_p95 | 0.00128 |


## Current-frame divergence probes

### Current-frame isolation

| frame | camera | replace | precision | raw_iou | inferred_issue | path |
| --- | --- | --- | --- | --- | --- | --- |
| 47 | cam1 | none | all_bf16 | 0.73472 | batch3_dimension_handling_or_diagonal_slicing_suspect | docs/generated/current_frame47_cam1_precision_all_bf16.json |
| 47 | cam1 | none | all_fp32 | 0.99908 | precision_sensitive_current_frame_path | docs/generated/current_frame47_cam1_precision_all_fp32.json |
| 47 | cam1 | none | decoder_fp32 | 1.0 | no_current_frame_divergence_under_probe | docs/generated/current_frame47_cam1_precision_decoder_fp32.json |
| 47 | cam1 |  |  |  |  | docs/generated/current_frame47_cam1_precision_grid_summary.json |
| 47 | cam1 | none | mask_logits_fp32 | 0.73472 | batch3_dimension_handling_or_diagonal_slicing_suspect | docs/generated/current_frame47_cam1_precision_mask_logits_fp32.json |
| 47 | cam1 | none | maskmem_features_fp32 | 0.73472 | batch3_dimension_handling_or_diagonal_slicing_suspect | docs/generated/current_frame47_cam1_precision_maskmem_features_fp32.json |
| 47 | cam1 | none | memory_attention_fp32 | 0.99955 | no_current_frame_divergence_under_probe | docs/generated/current_frame47_cam1_precision_memory_attention_fp32.json |
| 47 | cam1 | none | memory_encoder_fp32 | 0.73472 | batch3_dimension_handling_or_diagonal_slicing_suspect | docs/generated/current_frame47_cam1_precision_memory_encoder_fp32.json |
| 47 | cam1 | none | memory_path_fp32 | 0.99955 | no_current_frame_divergence_under_probe | docs/generated/current_frame47_cam1_precision_memory_path_fp32.json |
| 47 | cam1 | none | object_pointer_fp32 | 0.73472 | batch3_dimension_handling_or_diagonal_slicing_suspect | docs/generated/current_frame47_cam1_precision_object_pointer_fp32.json |

### Batch order ablation

| field | value |
| --- | --- |
| order_dependent | False |
| diagonal_slicing_bug | False |
| path | docs/generated/full_batched_batch_order_ablation.json |

### Storage alias audit

| field | value |
| --- | --- |
| alias_found | False |
| alias_record_count | 0 |
| path | docs/generated/full_batched_storage_alias_audit.json |

## Decision

| field | value |
| --- | --- |
| hf_batch_vision_seq_session_usable | False |
| single_object_stuffed_animal_validated | False |
| single_object_30fps_p50_gate | False |
| single_object_30fps_p90_gate | False |
| single_object_30fps_p95_gate |  |
| hf_batched_multisession_usable | False |
| hf_batched_multisession_failure_stage | compile_correctness |
| hf_batched_multisession_reason | memory_path_fp32 eager strict correctness passes, but compiled strict correctness fails for max-autotune-no-cudagraphs and reduce-overhead; keep ring_buffer and debug compiled numeric/state lifetime path before profiling. |
| hf_batched_multisession_blockers | memory_path_fp32 eager strict correctness passes, but compiled strict correctness fails for max-autotune-no-cudagraphs and reduce-overhead; keep ring_buffer and debug compiled numeric/state lifetime path before profiling. |
| full_batched_bf16_strict_pass | False |
| full_batched_memory_attention_fp32_strict_pass | True |
| full_batched_decoder_fp32_strict_pass | False |
| full_batched_memory_path_fp32_strict_pass | True |
| full_batched_all_fp32_strict_pass | True |
| recommended_precision_mode | memory_path_fp32 |
| recommended_precision_mode_reason | memory_path_fp32 is the best non-all-fp32 eager strict pass by global IoU |
| full_batched_compile_max_autotune_no_cudagraphs_pass | False |
| full_batched_compile_reduce_overhead_pass | False |
| full_batched_vs_sam31_not_worse |  |
| full_batched_vs_sam31_delta |  |
| recurrent_memory_slot_order_pass |  |
| current_frame_inferred_issue | batch3_dimension_handling_or_diagonal_slicing_suspect |
| batch_order_dependent | False |
| storage_alias_found | False |
| recurrent_drift_first_tensor | {'camera': 'cam0', 'field': 'maskmem_features', 'frame_idx': 0, 'p95_abs_diff': 0.02552780508995056, 'threshold': 0.001} |
| faster_than_77_92_ms_baseline | False |
| recommended_backend | hf_batch_vision_seq_session |
| recommended_compile_mode |  |
| fallback_backend | hf_batch_vision_seq_session |
| controller_hand_validated | False |
| controller_hand_status |  |
| controller_towel_validated | False |
| speed_first_usable | False |
| strict_validated_objects | ['stuffed animal'] |
| reference_uncertain_objects | [] |
| empty_reference_policy | strict-empty-mismatch |
| demo22_final_fps_pending | True |
| demo22_final_fps_source | pending full Demo 2.2 profile; replay/component FPS is not final FPS |
| controller_towel_caveat | SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras; current quality claim is for stuffed animal only. |
