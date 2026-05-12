# EdgeTAM batch=3 multi-session final report

## Goal

Original weights + custom batch=3 multi-session runtime.

## Source

| field | value |
| --- | --- |
| github_repo | https://github.com/BaochaiXue/transformers/tree/feat/edgetam-batched-multisession-runtime |
| fork_path | /home/zhangxinjie/EdgeTAM-HF-batched |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 4a41c4a45fa6f0541f3d62aff95867de06d90c92 |
| modeling_edgetam_video_touched | False |

## Correctness

| backend | compile | pass | mask_pass | partial | fallback | contract_pass | path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hf_batch_vision_seq_session | none | True | True | False |  |  | docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session.json |
| hf_batch_vision_seq_session | max-autotune-no-cudagraphs | False | False | False |  |  | docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_max_autotune_no_cudagraphs.json |
| hf_batch_vision_seq_session | reduce-overhead | False | False | False |  |  | docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_reduce_overhead.json |
| hf_batched_multisession | none | False | True | True | hf_batch_vision_seq_session |  | docs/generated/edgetam_batched_correctness_hf_batched_multisession.json |
| hf_batched_multisession | default | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_default.json |
| hf_batched_multisession | max-autotune-no-cudagraphs | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_max_autotune_no_cudagraphs.json |
| hf_batched_multisession | reduce-overhead | True | True | False |  | True | docs/generated/full_batched_multisession_single_stuffed_animal_memory_path_fp32_reduce_overhead.json |

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

No first-bad-frame report provided.

## Full batched recurrent drift localization

No recurrent drift localization reports provided.

## Current-frame divergence probes

No current-frame divergence probes provided.

## BatchTam ONNX/TRT component runtime

| component | onnx_export | onnx_validate | trt_build | trt_validate | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| vision_encoder | False | False | False | False | onnx_export | onnx_export has not passed |
| memory_attention | True | True | True | True |  |  |
| mask_decoder | True | True | True | True |  |  |
| memory_encoder | True | True | True | True |  |  |

| field | value |
| --- | --- |
| component_validation_usable | True |
| closed_loop_strict_pass | False |
| trt_components_usable | False |
| recommended_trt_scope | memory_path_all |
| demo22_integration_allowed | False |
| failure_stage | closed_loop_correctness |
| exact_blocker | closed-loop strict correctness has not passed |

## Decision

| field | value |
| --- | --- |
| hf_batch_vision_seq_session_usable | True |
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
| full_batched_memory_path_fp32_strict_pass | False |
| full_batched_all_fp32_strict_pass | False |
| recommended_precision_mode |  |
| recommended_precision_mode_reason |  |
| full_batched_compile_max_autotune_no_cudagraphs_pass | False |
| full_batched_compile_default_pass | False |
| full_batched_compile_reduce_overhead_pass | False |
| full_batched_vs_sam31_not_worse | True |
| full_batched_vs_sam31_delta | -1.81535062957483e-05 |
| recurrent_memory_slot_order_pass |  |
| current_frame_inferred_issue |  |
| batch_order_dependent |  |
| storage_alias_found |  |
| recurrent_drift_first_tensor |  |
| faster_than_77_92_ms_baseline | True |
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
| batchtam_trt_components_usable | False |
| batchtam_trt_demo22_integration_allowed | False |
| controller_towel_caveat | SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras; current quality claim is for stuffed animal only. |
