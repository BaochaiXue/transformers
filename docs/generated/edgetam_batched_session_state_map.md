# EdgeTAM Batched Session State Map

This report maps the session fields that a batch=3 multi-session runtime must preserve.

## Summary

| metric | value |
| --- | --- |
| session_count | 3 |
| field_count | 61 |
| tensor_field_count | 14 |
| stackable_tensor_field_count | 14 |
| metadata_field_count | 47 |
| mutated_field_count | 10 |

## Required Field Presence

| field group | present |
| --- | --- |
| frame_idx_or_processed_frames | True |
| video_size | True |
| obj_ids | True |
| prompt_inputs | True |
| vision_feature_cache | True |
| memory_features | True |
| object_pointers | True |
| mask_logits_or_pred_masks | True |
| conditioning_outputs | True |

## Fields

| path | type | shape | dtype | mutated | can_stack | needs_padding | metadata | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| session[*] | SimpleNamespace |  |  | False | False | False | True |  |
| session[*]._obj_id_to_idx | dict |  |  | False | False | False | True | len=2 |
| session[*]._obj_id_to_idx.[1] | int |  |  | False | False | False | True | value=0 |
| session[*]._obj_id_to_idx.[2] | int |  |  | False | False | False | True | value=1 |
| session[*]._obj_idx_to_id | dict |  |  | False | False | False | True | len=2 |
| session[*]._obj_idx_to_id.[0] | int |  |  | False | False | False | True | value=1 |
| session[*]._obj_idx_to_id.[1] | int |  |  | False | False | False | True | value=2 |
| session[*].cache | dict |  |  | False | False | False | True | len=1 |
| session[*].cache.vision_features | dict |  |  | False | False | False | True | len=1 |
| session[*].cache.vision_features.[0] | dict |  |  | True | False | False | True | len=2 |
| session[*].cache.vision_features.[0].high_res_feats | list |  |  | True | False | False | True | len=2 |
| session[*].cache.vision_features.[0].high_res_feats[0] | Tensor | [1, 32, 256, 256] | torch.bfloat16 | True | True | False | False |  |
| session[*].cache.vision_features.[0].high_res_feats[1] | Tensor | [1, 64, 128, 128] | torch.bfloat16 | True | True | False | False |  |
| session[*].cache.vision_features.[0].image_embed | Tensor | [1, 256, 64, 64] | torch.bfloat16 | True | True | False | False |  |
| session[*].camera_idx | int |  |  | False | False | False | True | value=0 |
| session[*].dtype | str |  |  | False | False | False | True | value='torch.bfloat16' |
| session[*].frames_tracked_per_obj | dict |  |  | False | False | False | True | len=2 |
| session[*].frames_tracked_per_obj.[0] | dict |  |  | True | False | False | True | len=1 |
| session[*].frames_tracked_per_obj.[0].[0] | dict |  |  | False | False | False | True | len=1 |
| session[*].frames_tracked_per_obj.[0].[0].reverse | bool |  |  | False | False | False | True | value=False |
| session[*].frames_tracked_per_obj.[1] | dict |  |  | True | False | False | True | len=1 |
| session[*].frames_tracked_per_obj.[1].[0] | dict |  |  | False | False | False | True | len=1 |
| session[*].frames_tracked_per_obj.[1].[0].reverse | bool |  |  | False | False | False | True | value=False |
| session[*].inference_device | str |  |  | False | False | False | True | value='cuda:0' |
| session[*].inference_state_device | str |  |  | False | False | False | True | value='cuda:0' |
| session[*].mask_inputs_per_obj | dict |  |  | False | False | False | True | len=2 |
| session[*].mask_inputs_per_obj.[0] | dict |  |  | False | False | False | True | len=1 |
| session[*].mask_inputs_per_obj.[0].[0] | Tensor | [1, 480, 848] | torch.bool | False | True | False | False |  |
| session[*].mask_inputs_per_obj.[1] | dict |  |  | False | False | False | True | len=1 |
| session[*].mask_inputs_per_obj.[1].[0] | Tensor | [1, 480, 848] | torch.bool | False | True | False | False |  |
| session[*].max_vision_features_cache_size | int |  |  | False | False | False | True | value=1 |
| session[*].memory_features | dict |  |  | False | False | False | True | len=2 |
| session[*].memory_features.maskmem_features | Tensor | [2, 64, 64, 64] | torch.bfloat16 | False | True | False | False |  |
| session[*].memory_features.maskmem_pos_enc | Tensor | [2, 64, 64, 64] | torch.bfloat16 | False | True | False | False |  |
| session[*].obj_ids | list |  |  | False | False | False | True | len=2 |
| session[*].obj_ids[0] | int |  |  | False | False | False | True | value=1 |
| session[*].obj_ids[1] | int |  |  | False | False | False | True | value=2 |
| session[*].obj_with_new_inputs | set |  |  | False | False | False | True | len=0 |
| session[*].output_dict_per_obj | dict |  |  | False | False | False | True | len=2 |
| session[*].output_dict_per_obj.[0] | dict |  |  | False | False | False | True | len=2 |
| session[*].output_dict_per_obj.[0].cond_frame_outputs | dict |  |  | False | False | False | True | len=1 |
| session[*].output_dict_per_obj.[0].cond_frame_outputs.[0] | dict |  |  | False | False | False | True | len=3 |
| session[*].output_dict_per_obj.[0].cond_frame_outputs.[0].object_pointer | Tensor | [1, 256] | torch.bfloat16 | False | True | False | False |  |
| session[*].output_dict_per_obj.[0].cond_frame_outputs.[0].object_score_logits | Tensor | [1] | torch.float32 | False | True | False | False |  |
| session[*].output_dict_per_obj.[0].cond_frame_outputs.[0].pred_masks | Tensor | [1, 1, 480, 848] | torch.bfloat16 | False | True | False | False |  |
| session[*].output_dict_per_obj.[0].non_cond_frame_outputs | dict |  |  | True | False | False | True | len=0 |
| session[*].output_dict_per_obj.[1] | dict |  |  | False | False | False | True | len=2 |
| session[*].output_dict_per_obj.[1].cond_frame_outputs | dict |  |  | False | False | False | True | len=1 |
| session[*].output_dict_per_obj.[1].cond_frame_outputs.[0] | dict |  |  | False | False | False | True | len=3 |
| session[*].output_dict_per_obj.[1].cond_frame_outputs.[0].object_pointer | Tensor | [1, 256] | torch.bfloat16 | False | True | False | False |  |
| session[*].output_dict_per_obj.[1].cond_frame_outputs.[0].object_score_logits | Tensor | [1] | torch.float32 | False | True | False | False |  |
| session[*].output_dict_per_obj.[1].cond_frame_outputs.[0].pred_masks | Tensor | [1, 1, 480, 848] | torch.bfloat16 | False | True | False | False |  |
| session[*].output_dict_per_obj.[1].non_cond_frame_outputs | dict |  |  | True | False | False | True | len=0 |
| session[*].point_inputs_per_obj | dict |  |  | False | False | False | True | len=2 |
| session[*].point_inputs_per_obj.[0] | dict |  |  | False | False | False | True | len=0 |
| session[*].point_inputs_per_obj.[1] | dict |  |  | False | False | False | True | len=0 |
| session[*].previous_masks | Tensor | [2, 1, 480, 848] | torch.bool | False | True | False | False |  |
| session[*].processed_frames | int |  |  | True | False | False | True | value=0 |
| session[*].video_height | int |  |  | False | False | False | True | value=480 |
| session[*].video_storage_device | str |  |  | False | False | False | True | value='cpu' |
| session[*].video_width | int |  |  | False | False | False | True | value=848 |
