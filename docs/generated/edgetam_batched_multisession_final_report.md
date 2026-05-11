# EdgeTAM batch=3 multi-session final report

## Goal

Original weights + custom batch=3 multi-session runtime.

## Source

| field | value |
| --- | --- |
| fork_path | /home/zhangxinjie/EdgeTAM-HF-batched |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 83110b8dd4820ba22d288993e7a24060f2ddf494 |
| modeling_edgetam_video_touched | False |

## Correctness

| backend | compile | pass | mask_pass | partial | fallback | path |
| --- | --- | --- | --- | --- | --- | --- |
| hf_batch_vision_seq_session | none | True | True | False |  | docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session.json |
| hf_batch_vision_seq_session | max-autotune-no-cudagraphs | False | False | False |  | docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_max_autotune_no_cudagraphs.json |
| hf_batch_vision_seq_session | reduce-overhead | False | False | False |  | docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_reduce_overhead.json |
| hf_batched_multisession | none | False | True | True | hf_batch_vision_seq_session | docs/generated/edgetam_batched_correctness_hf_batched_multisession.json |

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

## Decision

| field | value |
| --- | --- |
| hf_batch_vision_seq_session_usable | True |
| single_object_stuffed_animal_validated | True |
| single_object_30fps_p50_gate | True |
| single_object_30fps_p90_gate | True |
| single_object_30fps_p95_gate | borderline/fail |
| hf_batched_multisession_usable | False |
| hf_batched_multisession_reason | true batched session/memory/object-pointer tensorization is not complete |
| faster_than_77_92_ms_baseline | True |
| recommended_backend | hf_batch_vision_seq_session |
| recommended_compile_mode | reduce-overhead |
| fallback_backend | hf_batch_vision_seq_session |
| controller_hand_validated | False |
| controller_hand_status | low IoU outliers on cam0/cam2 |
| controller_towel_validated | False |
| controller_towel_caveat | SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras; current quality claim is for stuffed animal only. |
