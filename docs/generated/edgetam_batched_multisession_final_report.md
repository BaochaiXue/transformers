# EdgeTAM batch=3 multi-session final report

## Goal

Original weights + custom batch=3 multi-session runtime.

## Source

| field | value |
| --- | --- |
| github_repo | https://github.com/BaochaiXue/transformers/tree/feat/edgetam-batched-multisession-runtime |
| fork_path | /home/zhangxinjie/EdgeTAM-HF-batched |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 57b37fefbace53720cff482103cedc538f1b6419 |
| modeling_edgetam_video_touched | False |

## Correctness

| backend | compile | pass | mask_pass | partial | fallback | contract_pass | path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hf_batched_multisession | none | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_precision_memory_path_fp32.json |
| hf_batched_multisession | default | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_default.json |
| hf_batched_multisession | max-autotune-no-cudagraphs | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_max_autotune_no_cudagraphs.json |
| hf_batched_multisession | reduce-overhead | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_reduce_overhead.json |

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

No original-vs-compiled delta report provided.

## Full batched first bad frame

No first-bad-frame report provided.

## Full batched recurrent drift localization

No recurrent drift localization reports provided.

## Current-frame divergence probes

No current-frame divergence probes provided.

## BatchTam ONNX/TRT component runtime

BatchTam ONNX/TRT component runtime:

| component | runtime | status | closed_loop |
| --- | --- | --- | --- |
| memory_attention | bucketed static TRT engines | 16/16 validated | True |
| mask_decoder | single TRT component | validated | True |
| memory_encoder | single TRT component | validated | True |
| memory_path_all | closed-loop scheduler | strict correctness | True |
| vision_encoder | PyTorch/compiled vision path | not exported to TRT in this phase | not in recommended TRT scope |

Legacy single-static component diagnostics are shown below only for debugging; they do not define the `memory_path_all` Demo 2.2 gate.

| component | onnx_export | onnx_validate | trt_build | trt_validate | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| vision_encoder | False | False | False | False | onnx_export | onnx_export has not passed |
| memory_attention | False | False | False | True | onnx_export |  |
| mask_decoder | False | False | False | True | onnx_export |  |
| memory_encoder | False | False | False | True | onnx_export |  |

| field | value |
| --- | --- |
| memory_attention_shape_strategy | bucketed_static_engines |
| memory_attention_bucket_count | 16 |
| memory_attention_buckets_exported | 16 |
| memory_attention_buckets_built | 16 |
| memory_attention_buckets_validated | 16 |
| component_validation_usable | True |
| closed_loop_strict_pass | True |
| trt_components_usable | True |
| recommended_trt_scope | memory_path_all |
| demo22_trt_integration_allowed | True |
| failure_stage |  |
| exact_blocker |  |

## Decision

| field | value |
| --- | --- |
| hf_batch_vision_seq_session_usable | False |
| single_object_stuffed_animal_validated | True |
| single_object_30fps_p50_gate | True |
| single_object_30fps_p90_gate | True |
| single_object_30fps_p95_gate | borderline/fail |
| hf_batched_multisession_usable | True |
| hf_batched_multisession_failure_stage |  |
| hf_batched_multisession_reason | contract pass + strict compiled closed-loop correctness pass |
| hf_batched_multisession_blockers |  |
| full_batched_bf16_strict_pass | False |
| full_batched_memory_attention_fp32_strict_pass | False |
| full_batched_decoder_fp32_strict_pass | False |
| full_batched_memory_path_fp32_strict_pass | True |
| full_batched_all_fp32_strict_pass | False |
| recommended_precision_mode | memory_path_fp32 |
| recommended_precision_mode_reason | memory_path_fp32 is the best non-all-fp32 eager strict pass by global IoU |
| full_batched_compile_max_autotune_no_cudagraphs_pass | True |
| full_batched_compile_default_pass | True |
| full_batched_compile_reduce_overhead_pass | True |
| full_batched_vs_sam31_not_worse |  |
| full_batched_vs_sam31_delta |  |
| recurrent_memory_slot_order_pass |  |
| current_frame_inferred_issue |  |
| batch_order_dependent |  |
| storage_alias_found |  |
| recurrent_drift_first_tensor |  |
| faster_than_77_92_ms_baseline | False |
| recommended_backend | hf_batched_multisession |
| recommended_compile_mode | reduce-overhead |
| fallback_backend | hf_batch_vision_seq_session |
| controller_hand_validated | False |
| controller_hand_status | low IoU outliers on cam0/cam2 |
| controller_towel_validated | False |
| speed_first_usable | False |
| strict_validated_objects | ['stuffed animal'] |
| reference_uncertain_objects | [] |
| empty_reference_policy | strict-empty-mismatch |
| demo22_final_fps_pending | True |
| demo22_final_fps_source | pending full Demo 2.2 profile; replay/component FPS is not final FPS |
| batchtam_trt_component_validation_usable | True |
| batchtam_trt_components_usable | True |
| batchtam_trt_demo22_integration_allowed | True |
| controller_towel_caveat | SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras; current quality claim is for stuffed animal only. |
