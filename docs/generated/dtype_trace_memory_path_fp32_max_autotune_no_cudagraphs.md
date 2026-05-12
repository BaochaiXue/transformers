# Full Batched EdgeTAM Dtype Trace

| field | value |
| --- | --- |
| precision_mode | memory_path_fp32 |
| compile_mode | max-autotune-no-cudagraphs |
| compile_scope | memory_path_all |
| precision_policy_applied_before_compile | True |
| compiled_after_component_dtype_conversion | True |

## Component Parameter Dtypes

| component | dtype |
| --- | --- |
| mask_decoder | torch.float32 |
| mask_downsample | torch.float32 |
| memory_attention | torch.float32 |
| memory_encoder | torch.float32 |
| object_pointer_proj | torch.float32 |
| prompt_encoder | torch.float32 |
| shared_image_embedding | torch.float32 |
| spatial_perceiver | torch.float32 |

## Stored State Dtypes

| path | dtypes |
| --- | --- |
| maskmem_features | torch.float32 |
| maskmem_pos_enc | torch.float32 |
| object_pointer | torch.float32 |
| object_score_logits | torch.float32 |
| pred_masks | torch.float32 |

## Gate

| check | value |
| --- | --- |
| pass | True |
| stored_mask_logits_fp32 | True |
| stored_maskmem_features_fp32 | True |
| stored_object_pointer_fp32 | True |
| stored_object_score_logits_fp32 | True |
