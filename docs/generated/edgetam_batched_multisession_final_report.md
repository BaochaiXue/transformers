# EdgeTAM batch=3 multi-session final report

## Goal

Original weights + custom batch=3 multi-session runtime.

## Source

| field | value |
| --- | --- |
| github_repo | https://github.com/BaochaiXue/transformers/tree/feat/edgetam-batched-multisession-runtime |
| fork_path | /home/zhangxinjie/EdgeTAM-HF-batched |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 90bb25a043b31fc9867f65ccfe9e0c4a871626f4 |
| modeling_edgetam_video_touched | False |

## Correctness

| backend | compile | pass | mask_pass | partial | fallback | contract_pass | path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hf_batched_multisession | none | False | False | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_none.json |
| hf_batched_multisession | none | True | True | False |  | True | docs/generated/different_types_sam31ref_full_batched_single_stuffed_animal_ignore_ref_empty.json |
| hf_ref_seq_public | none | True | True | False |  | False | docs/generated/different_types_sam31ref_original_hf_seq_single_stuffed_animal_ignore_ref_empty.json |

## Profiles

| backend | compile | p50 | p90 | partial | path |
| --- | --- | --- | --- | --- | --- |
| hf_batch_vision_seq_session | max-autotune-no-cudagraphs | 59.9900099914521 | 66.65191479842179 | False | docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_max_autotune_no_cudagraphs.json |
| hf_batch_vision_seq_session | none | 63.74551501357928 | 66.34561588289216 | False | docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none.json |
| hf_batch_vision_seq_session | reduce-overhead | 58.43767599435523 | 65.8067935204599 | False | docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_reduce_overhead.json |
| hf_batched_multisession | none | 63.66823450662196 | 65.51955243339762 | True | docs/generated/edgetam_batched_profile_hf_batched_multisession_none.json |

## SAM3.1 replay reference correctness

- reference_source: `sam31-replay`
- sam31_mask_root: `/home/zhangxinjie/proj-QQTT-v2/result/demo22_rgb_triplet_100frames_towel_stuffed_animal/sam31_video_reference_masks`
- backend: `hf_batch_vision_seq_session`

| compile_mode | correctness_pass | global_iou_avg | global_iou_min | global_iou_p50 | empty_mismatch |
| --- | --- | --- | --- | --- | --- |
| max-autotune-no-cudagraphs | True | 0.98122 | 0.93887 | 0.99073 | 0 |
| none | True | 0.98137 | 0.94032 | 0.99073 | 0 |
| reduce-overhead | True | 0.98138 | 0.93835 | 0.99073 | 0 |

## Non-empty object quality: stuffed animal only

| compile_mode | cam0 stuffed animal IoU | cam1 stuffed animal IoU | cam2 stuffed animal IoU |
| --- | --- | --- | --- |
| max-autotune-no-cudagraphs | 0.97582 | 0.96297 | 0.94242 |
| none | 0.97578 | 0.96238 | 0.94318 |
| reduce-overhead | 0.97472 | 0.96269 | 0.94373 |

## Controller/towel caveat

SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras.
Therefore obj0 IoU=1.0 is empty-vs-empty and does not validate controller tracking.
The current replay validates stuffed animal quality, not successful towel tracking.
A new replay with non-empty towel masks is required before claiming controller-object correctness.

## Different-types sloth_set_2 result

- replay: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal`

| field | value |
| --- | --- |
| single_object_stuffed_animal_validated | True |
| controller_hand_validated | False |
| controller_hand_reason | low IoU outliers on cam0/cam2 |
| backend | hf_batch_vision_seq_session |
| compile | reduce-overhead |
| stage_wall_p50_ms | 31.30548 |
| stage_wall_p90_ms | 32.98849 |
| stage_wall_p95_ms | 33.75403 |
| complete_group_fps_from_p50 | 31.94329 |
| p50_30fps_gate | True |
| p90_30fps_gate | True |
| p95_30fps_gate | False |

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

## Decision

| field | value |
| --- | --- |
| hf_batch_vision_seq_session_usable | True |
| single_object_stuffed_animal_validated | True |
| single_object_30fps_p50_gate | True |
| single_object_30fps_p90_gate | True |
| single_object_30fps_p95_gate | borderline/fail |
| hf_batched_multisession_usable | False |
| hf_batched_multisession_failure_stage | correctness |
| hf_batched_multisession_reason | first bad frame 43 cam1 IoU=0.59205, component=mask_decoder_or_accumulated_state; mask correctness gate failed: gate=strict, evaluated=279/1, empty_mismatch=0/0, global_iou_avg=0.97983960550254/0.98, global_iou_p50=0.9899300221880867/0.98 |
| hf_batched_multisession_blockers | first bad frame 43 cam1 IoU=0.59205, component=mask_decoder_or_accumulated_state; mask correctness gate failed: gate=strict, evaluated=279/1, empty_mismatch=0/0, global_iou_avg=0.97983960550254/0.98, global_iou_p50=0.9899300221880867/0.98 |
| full_batched_vs_sam31_not_worse | True |
| full_batched_vs_sam31_delta | -1.81535062957483e-05 |
| recurrent_memory_slot_order_pass | True |
| recurrent_drift_first_tensor | {'camera': 'cam0', 'field': 'maskmem_features', 'frame_idx': 0, 'p95_abs_diff': 0.015625, 'threshold': 0.001} |
| faster_than_77_92_ms_baseline | True |
| recommended_backend | hf_batch_vision_seq_session |
| recommended_compile_mode | reduce-overhead |
| fallback_backend | hf_batch_vision_seq_session |
| controller_hand_validated | False |
| controller_hand_status | low IoU outliers on cam0/cam2 |
| controller_towel_validated | False |
| speed_first_usable | True |
| strict_validated_objects | ['stuffed animal'] |
| reference_uncertain_objects | [] |
| empty_reference_policy | strict-empty-mismatch |
| demo22_final_fps_pending | True |
| demo22_final_fps_source | pending full Demo 2.2 profile; replay/component FPS is not final FPS |
| controller_towel_caveat | SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras; current quality claim is for stuffed animal only. |
