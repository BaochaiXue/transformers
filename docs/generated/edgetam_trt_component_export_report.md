# BatchTam ONNX/TRT Component Export Report

- trt_components_usable: `False`
- batchtam_component_engines_usable: `True`
- batchtam_closed_loop_usable: `False`
- recommended_trt_scope: `memory_path_all`
- demo22_integration_allowed: `False`
- failure_stage: `closed_loop_correctness`
- exact_blocker: `RuntimeError: BatchTam memory_attention engine was built for the fixed single-object tracking shape (num_object_pointer_tokens=4, num_spatial_memory_tokens=1), got 8 and 2`

| component | onnx_export | onnx_validate | trt_build | trt_validate | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| vision_encoder | False | False | False | False | onnx_export | onnx_export has not passed |
| memory_attention | False | False | False | True | onnx_export |  |
| mask_decoder | False | False | False | True | onnx_export |  |
| memory_encoder | False | False | False | True | onnx_export |  |
