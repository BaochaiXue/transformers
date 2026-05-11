# EdgeTAM batch=3 multi-session final report

## Goal

Original weights + custom batch=3 multi-session runtime.

## Source

| field | value |
| --- | --- |
| github_repo | https://github.com/BaochaiXue/transformers/tree/feat/edgetam-batched-multisession-runtime |
| fork_path | /home/zhangxinjie/EdgeTAM-HF-batched |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 97fba4dabd9a7b9d8730f6a157dd53e3179cba9c |
| modeling_edgetam_video_touched | False |

## Correctness

| backend | compile | pass | mask_pass | partial | fallback | contract_pass | path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hf_batched_multisession | none | False | False | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_none.json |
| hf_batched_multisession | none | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_all_fp32_none.json |
| hf_batched_multisession | none | True | True | False |  | True | docs/generated/different_types_sam31ref_full_batched_single_stuffed_animal_ignore_ref_empty.json |

## Profiles

| backend | compile | p50 | p90 | partial | path |
| --- | --- | --- | --- | --- | --- |
| hf_batched_multisession | none | 63.66823450662196 | 65.51955243339762 | True | docs/generated/edgetam_batched_profile_hf_batched_multisession_none.json |
| hf_batch_vision_seq_session | reduce-overhead | 58.43767599435523 | 65.8067935204599 | False | docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_reduce_overhead.json |

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

- baseline_backend: `hf_ref_seq_public`
- candidate_backend: `hf_batched_multisession`
- candidate_compile_mode: `none`
- empty_reference_policy: `ignore-candidate`
- evaluated_subset_mismatch: `False`

Compiled batch vision is compared against original HF public only on SAM3.1 reference-nonempty samples.

| object | baseline_iou | candidate_iou | delta | not_worse | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | reference_uncertain | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stuffed animal | 0.96344 | 0.96342 | -2e-05 | True | 279 | 0 | 0 | False |  |

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
| memory_attention | 59 | 59 | True | 0.0625 | 0.00781 | docs/generated/component_equivalence_memory_attention_single_object_60f.json |
| mask_decoder | 60 | 60 | True | 0.375 | 0.0625 | docs/generated/component_equivalence_mask_decoder_single_object_60f.json |
| memory_encoder | 60 | 60 | True | 0.14062 | 0.00391 | docs/generated/component_equivalence_memory_encoder_single_object_60f.json |

### Teacher forcing

| mode | status | first_bad_frame | camera | iou | avg | p50 | hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after_mask_decoder | ok | 47 | cam1 | 0.73297 | 0.99802 | 0.99913 | drift_not_eliminated_by_after_mask_decoder |
| after_memory_attention | unsupported |  |  |  |  |  | not_evaluated |
| after_memory_encoder | ok | 47 | cam1 | 0.73515 | 0.9953 | 0.99773 | drift_not_eliminated_by_after_memory_encoder |
| after_state_scatter | ok | 47 | cam1 | 0.73297 | 0.99802 | 0.99913 | drift_not_eliminated_by_after_state_scatter |
| before_step_state | ok | 47 | cam1 | 0.73297 | 0.99802 | 0.99913 | drift_not_eliminated_by_before_step_state |
| none | ok | 43 | cam1 | 0.59205 | 0.97984 | 0.98993 | baseline_full_batched_closed_loop_drift |

### State drift curve

| field | value |
| --- | --- |
| mask_iou_avg | 0.97984 |
| mask_iou_min | 0.59205 |
| first_iou_lt_0_90 | {'camera': 'cam1', 'frame_idx': 43, 'threshold': 0.9, 'value': 0.5920526014865638} |
| first_iou_lt_0_98 | {'camera': 'cam0', 'frame_idx': 5, 'threshold': 0.98, 'value': 0.9757200666507975} |
| first_tensor_drift | {'camera': 'cam0', 'field': 'maskmem_features', 'frame_idx': 0, 'p95_abs_diff': 0.015625, 'threshold': 0.001} |
| maskmem_features_first_p95_gt_1 | {'camera': 'cam1', 'frame_idx': 38, 'threshold': 1.0, 'value': 1.107421875} |
| object_pointer_first_p95_gt_1e_1 | {'camera': 'cam0', 'frame_idx': 9, 'threshold': 0.1, 'value': 0.10595703125} |
| maskmem_pos_enc_p95 | 0.0 |

### State commit ablation

| mode | first_bad_frame | camera | iou | avg | p50 | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| batched_scatter | 43 | cam1 | 0.59205 | 0.97984 | 0.98993 | current full-batched state commit path |
| hybrid_reference_dict_batched_tensors | 47 | cam1 | 0.73515 | 0.9953 | 0.99773 | drift remains after forcing memory outputs; decoder/object pointer remains suspect |
| reference_store_output | 47 | cam1 | 0.73297 | 0.99802 | 0.99913 | drift remains even when current outputs are forced to reference after decoder |

### Memory slot audit

| field | value |
| --- | --- |
| pass | True |
| first_mismatch |  |
| path | docs/generated/full_batched_memory_slot_audit.json |

## Current-frame divergence probes

### Current-frame isolation

| frame | camera | replace | precision | raw_iou | inferred_issue | path |
| --- | --- | --- | --- | --- | --- | --- |
| 47 | cam1 | none | all_bf16 | 0.73297 | batch3_dimension_handling_or_diagonal_slicing_suspect | docs/generated/full_batched_current_frame_isolation_frame47_cam1.json |
| 47 | cam1 | decoder_inputs_with_reference | all_bf16 | 0.73297 | replacement_level_requires_runtime_debug_hook | docs/generated/current_frame47_cam1_replace_decoder_inputs_with_reference.json |
| 47 | cam1 | mask_decoder_output_with_reference | all_bf16 | 0.73297 | posthoc_replacement_reaches_reference_mask; rerun-level hook still needed | docs/generated/current_frame47_cam1_replace_mask_decoder_output_with_reference.json |
| 47 | cam1 | memory_encoder_output_with_reference | all_bf16 | 0.73297 | replacement_level_requires_runtime_debug_hook | docs/generated/current_frame47_cam1_replace_memory_encoder_output_with_reference.json |
| 47 | cam1 | object_pointer_with_reference | all_bf16 | 0.73297 | replacement_level_requires_runtime_debug_hook | docs/generated/current_frame47_cam1_replace_object_pointer_with_reference.json |
| 47 | cam1 | postprocess_output_with_reference | all_bf16 | 0.73297 | posthoc_replacement_reaches_reference_mask; rerun-level hook still needed | docs/generated/current_frame47_cam1_replace_postprocess_output_with_reference.json |

### Decoder diff probe

| field | value |
| --- | --- |
| raw_iou | 0.73297 |
| first_divergent_tensor | {'field': 'pred_masks', 'max_abs_diff': 8.09375, 'mean_abs_diff': 5.0433125495910645, 'p95_abs_diff': 6.8125, 'shape_match': True} |
| threshold_flip_count | 784 |
| reference_area | 2212 |
| candidate_area | 2876 |
| bbox_center_distance | 22.47221 |

### Precision ladder

| mode | implemented | dtype | raw_iou | conclusion | reason |
| --- | --- | --- | --- | --- | --- |
| all_bf16 | True | bfloat16 | 0.73297 | current-frame divergence remains under this precision |  |
| all_fp32 | True | float32 | 0.99908 | current-frame pass under this precision |  |
| memory_path_fp32 | False | bfloat16 |  | not implemented; add selective dtype hooks before using this mode as evidence | selective component fp32 requires runtime-level dtype hooks; current probe records this as a pending patch |

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
| hf_batched_multisession_failure_stage | precision |
| hf_batched_multisession_reason | first bad frame 43 cam1 IoU=0.59205, component=mask_decoder_or_accumulated_state; bf16 closed-loop strict fails, while diagnostic all-fp32 eager strict passes; next patch must implement selective mixed memory/decoder path and compiled correctness |
| hf_batched_multisession_blockers | first bad frame 43 cam1 IoU=0.59205, component=mask_decoder_or_accumulated_state; bf16 closed-loop strict fails, while diagnostic all-fp32 eager strict passes; next patch must implement selective mixed memory/decoder path and compiled correctness |
| full_batched_bf16_strict_pass | False |
| full_batched_all_fp32_strict_pass | True |
| full_batched_vs_sam31_not_worse | True |
| full_batched_vs_sam31_delta | -1.81535062957483e-05 |
| recurrent_memory_slot_order_pass | True |
| current_frame_inferred_issue | batch3_dimension_handling_or_diagonal_slicing_suspect |
| batch_order_dependent | False |
| storage_alias_found | False |
| recurrent_drift_first_tensor | {'camera': 'cam0', 'field': 'maskmem_features', 'frame_idx': 0, 'p95_abs_diff': 0.015625, 'threshold': 0.001} |
| faster_than_77_92_ms_baseline | True |
| recommended_backend | hf_batch_vision_seq_session |
| recommended_compile_mode | reduce-overhead |
| fallback_backend | hf_batch_vision_seq_session |
| controller_hand_validated | False |
| controller_hand_status |  |
| controller_towel_validated | False |
| speed_first_usable | True |
| strict_validated_objects | ['stuffed animal'] |
| reference_uncertain_objects | [] |
| empty_reference_policy | strict-empty-mismatch |
| demo22_final_fps_pending | True |
| demo22_final_fps_source | pending full Demo 2.2 profile; replay/component FPS is not final FPS |
| controller_towel_caveat | SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras; current quality claim is for stuffed animal only. |
