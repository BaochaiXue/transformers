# EdgeTAM batch=3 multi-session final report

## Goal

Original weights + custom batch=3 multi-session runtime.

## Source

| field | value |
| --- | --- |
| fork_path | /home/zhangxinjie/EdgeTAM-HF-batched |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 80359782959f26fab60a18e28ed86020f8064402 |
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

## Decision

| field | value |
| --- | --- |
| hf_batched_multisession_usable | False |
| faster_than_77_92_ms_baseline | True |
| recommended_backend | hf_batch_vision_seq_session |
| fallback_backend | hf_batch_vision_seq_session |
