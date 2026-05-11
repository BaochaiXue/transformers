# EdgeTAM Batched Source Inspection

## Repo

| field | value |
| --- | --- |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 80359782959f26fab60a18e28ed86020f8064402 |
| status_short | M docs/generated/edgetam_batched_multisession_final_report.json<br> M docs/generated/edgetam_batched_multisession_final_report.md<br> M edgetam_batched/batched_multisession_runtime.py<br> M edgetam_batched/camera_order.py<br> M edgetam_batched/compare_multisession.py<br> M edgetam_batched/component_adapter.py<br> M edgetam_batched/profile_multisession.py<br> M edgetam_batched/reference_runtime.py<br> M edgetam_batched/rgb_replay.py<br> M scripts/harness/check_all.py<br>?? docs/generated/edgetam_batched_component_access.json<br>?? docs/generated/edgetam_batched_component_access.md<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session.json<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session.md<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_max_autotune_no_cudagraphs.json<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_max_autotune_no_cudagraphs.md<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_reduce_overhead.json<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_reduce_overhead.md<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_smoke.json<br>?? docs/generated/edgetam_batched_correctness_hf_batch_vision_seq_session_smoke.md<br>?? docs/generated/edgetam_batched_correctness_hf_batched_multisession.json<br>?? docs/generated/edgetam_batched_correctness_hf_batched_multisession.md<br>?? docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_max_autotune_no_cudagraphs.json<br>?? docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_max_autotune_no_cudagraphs.md<br>?? docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none.json<br>?? docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none.md<br>?? docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_reduce_overhead.json<br>?? docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_reduce_overhead.md<br>?? docs/generated/edgetam_batched_profile_hf_batched_multisession_none.json<br>?? docs/generated/edgetam_batched_profile_hf_batched_multisession_none.md<br>?? docs/generated/edgetam_reference_runtime_smoke.json<br>?? docs/generated/edgetam_reference_runtime_smoke.md<br>?? docs/generated/reference_outputs_smoke/<br>?? edgetam_batched/final_report.py<br>?? scripts/run_edgetam_batched_multisession_e2e.sh<br>?? tests/test_batched_runtime_shapes.py<br>?? tests/test_compare_multisession.py<br>?? tests/test_component_adapter.py<br>?? tests/test_profile_multisession.py |

## Files

| name | path | exists | generated | size |
| --- | --- | --- | --- | --- |
| edgetam_modular | /home/zhangxinjie/EdgeTAM-HF-batched/src/transformers/models/edgetam_video/modular_edgetam_video.py | True | False | 66587 |
| edgetam_modeling | /home/zhangxinjie/EdgeTAM-HF-batched/src/transformers/models/edgetam_video/modeling_edgetam_video.py | True | True | 146530 |
| sam2_video_modeling | /home/zhangxinjie/EdgeTAM-HF-batched/src/transformers/models/sam2_video/modeling_sam2_video.py | True | True | 133217 |
| sam2_video_processing | /home/zhangxinjie/EdgeTAM-HF-batched/src/transformers/models/sam2_video/processing_sam2_video.py | True | True | 37447 |

## Findings

| finding | value |
| --- | --- |
| edgetam_subclasses_sam2_video | True |
| public_forward_has_inference_session | True |
| get_image_features_defined | True |
| session_class_defined | True |
| processor_init_video_session | True |
| processor_add_inputs | True |

## Component Token Counts

| component | count |
| --- | --- |
| vision_encoder | 6 |
| memory_attention | 116 |
| memory_encoder | 15 |
| mask_decoder | 33 |
| prompt_encoder | 31 |
| object_pointer | 189 |

## Signatures

| symbol | signature |
| --- | --- |
| EdgeTamVideoModel.forward | def forward(self, inference_session: EdgeTamVideoInferenceSession, frame_idx: int \| None = None, frame: torch.Tensor \| None = None, reverse: bool = False, **kwargs, ) -> EdgeTamVideoSegmentationOutput: r""" inference_session (`EdgeTamVideoInferenceSession`) |
| Sam2VideoModel.get_image_features | def get_image_features(self, pixel_values: torch.FloatTensor, **kwargs: Unpack[TransformersKwargs], ) -> tuple \| Sam2VideoVisionEncoderOutput: r""" pixel_values (`torch.FloatTensor`) |
| EdgeTamVideoInferenceSession | class EdgeTamVideoInferenceSession(Sam2VideoInferenceSession): |

## Boundary

- Do not edit `modeling_edgetam_video.py` directly; it is generated.
- The research runtime should use wrapper modules first and only patch modular source for accessors if blocked.
