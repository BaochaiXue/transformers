# BatchTam ONNX/TRT Component Export Report

- trt_components_usable: `True`
- batchtam_component_engines_usable: `True`
- batchtam_closed_loop_usable: `True`
- recommended_trt_scope: `memory_path_all`
- demo22_trt_integration_allowed: `True`
- failure_stage: `None`
- exact_blocker: `None`

## Recommended BatchTam Runtime

The recommended scope is `memory_path_all`: memory_attention, mask_decoder, and memory_encoder run as TRT components. Vision encoder is not exported to TRT in this phase and remains on the PyTorch/compiled vision path.

| field | value |
| --- | --- |
| shape_strategy | bucketed_static_engines |
| bucket_count | 16 |
| buckets_exported | 16 |
| buckets_built | 16 |
| buckets_validated | 16 |
| memory_attention_bucketed_closed_loop_pass | True |
| mask_decoder_trt_closed_loop_pass | True |
| memory_encoder_trt_closed_loop_pass | True |
| memory_path_all_trt_closed_loop_pass | True |

## Legacy Single-Component Diagnostics

These rows are retained for diagnostics. They do not define the Demo 2.2 gate when `recommended_trt_scope=memory_path_all`; the memory_attention gate is the bucketed static engine set above.

| component | onnx_export | onnx_validate | trt_build | trt_validate | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| vision_encoder | False | False | False | False | onnx_export | onnx_export has not passed |
| memory_attention | False | False | False | True | onnx_export |  |
| mask_decoder | False | False | False | True | onnx_export |  |
| memory_encoder | False | False | False | True | onnx_export |  |
