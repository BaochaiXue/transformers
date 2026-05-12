# BatchTam ONNX/TRT Component Export Report

- trt_components_usable: `True`
- batchtam_component_engines_usable: `True`
- batchtam_closed_loop_usable: `True`
- recommended_trt_scope: `memory_path_all`
- demo22_integration_allowed: `True`
- failure_stage: `None`
- exact_blocker: `None`

| component | onnx_export | onnx_validate | trt_build | trt_validate | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| vision_encoder | False | False | False | False | onnx_export | onnx_export has not passed |
| memory_attention | False | False | False | True | onnx_export |  |
| mask_decoder | False | False | False | True | onnx_export |  |
| memory_encoder | False | False | False | True | onnx_export |  |
