# EdgeTAM Batched Source Inspection

## Repo

| field | value |
| --- | --- |
| branch | feat/edgetam-batched-multisession-runtime |
| commit | 52f51bb8daccc5264d4b1dddcbc52968bf4e2553 |
| status_short | M  .gitignore<br>M  docs/generated/component_equivalence_mask_decoder_single_object.json<br>M  docs/generated/component_equivalence_mask_decoder_single_object.md<br>M  docs/generated/component_equivalence_memory_attention_single_object.json<br>M  docs/generated/component_equivalence_memory_attention_single_object.md<br>M  docs/generated/component_equivalence_memory_encoder_single_object.json<br>M  docs/generated/component_equivalence_memory_encoder_single_object.md<br>A  docs/generated/component_fixtures_single_object_30f.json<br>A  docs/generated/component_fixtures_single_object_30f.md<br>A  docs/generated/different_types_sloth_set_2_iou_ref_sam31_replay_batchvision_none.json<br>A  docs/generated/different_types_sloth_set_2_iou_ref_sam31_replay_batchvision_none.md<br>A  docs/generated/different_types_sloth_set_2_iou_ref_sam31_replay_batchvision_reduce_overhead.json<br>A  docs/generated/different_types_sloth_set_2_iou_ref_sam31_replay_batchvision_reduce_overhead.md<br>A  docs/generated/different_types_sloth_set_2_iou_ref_sam31_replay_hf_public_none.json<br>A  docs/generated/different_types_sloth_set_2_iou_ref_sam31_replay_hf_public_none.md<br>A  docs/generated/different_types_sloth_set_2_profile_batchvision_reduce_overhead_sam31_frame0.json<br>A  docs/generated/different_types_sloth_set_2_profile_batchvision_reduce_overhead_sam31_frame0.md<br>A  docs/generated/different_types_sloth_set_2_profile_hf_public_none_sam31_frame0.json<br>A  docs/generated/different_types_sloth_set_2_profile_hf_public_none_sam31_frame0.md<br>A  docs/generated/edgetam_batched_iou_ref_sam31_replay_green_towel_reduce_overhead.json<br>A  docs/generated/edgetam_batched_iou_ref_sam31_replay_green_towel_reduce_overhead.md<br>A  docs/generated/edgetam_batched_iou_ref_sam31_replay_towel_nonempty_reduce_overhead.json<br>A  docs/generated/edgetam_batched_iou_ref_sam31_replay_towel_nonempty_reduce_overhead.md<br>M  docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none_scaffold.json<br>M  docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none_scaffold.md<br>M  docs/generated/edgetam_batched_source_inspection.json<br>M  docs/generated/edgetam_batched_source_inspection.md<br>A  docs/generated/edgetam_hf_public_sam31_image_frame0_init_30f.json<br>A  docs/generated/edgetam_hf_public_sam31_image_frame0_init_30f.md<br>A  docs/generated/full_batched_multisession_single_stuffed_animal_none.json<br>A  docs/generated/full_batched_multisession_single_stuffed_animal_none.md<br>A  docs/generated/hf_edgetam_component_trace_single_object_30f.json<br>A  docs/generated/sam31_frame0_controller_prompt_sweep.json<br>A  docs/generated/sam31_image_frame0_init_green_towel_summary.json<br>A  docs/generated/sam31_image_frame0_init_green_towel_summary.md<br>A  docs/generated/sam31_video_reference_green_towel_on_table_summary.json<br>A  docs/generated/sam31_video_reference_green_towel_on_table_summary.md<br>A  docs/generated/sam31_video_text_box_reference_green_towel_summary.json<br>A  docs/generated/sam31_video_text_box_reference_green_towel_summary.md<br>M  edgetam_batched/batched_multisession_runtime.py<br>M  edgetam_batched/compare_multisession.py<br>M  edgetam_batched/component_batch_equivalence.py<br>A  edgetam_batched/component_fixtures.py<br>A  edgetam_batched/sam31_video_text_box_reference.py<br>A  tests/test_component_fixtures.py<br>A  tests/test_sam31_video_text_box_reference.py |

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
