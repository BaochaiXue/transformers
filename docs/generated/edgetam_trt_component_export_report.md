# BatchTam ONNX/TRT Component Export Report

- trt_components_usable: `False`
- recommended_trt_scope: `memory_path_all`
- demo22_integration_allowed: `False`
- failure_stage: `closed_loop_correctness`
- exact_blocker: `closed-loop strict correctness has not passed`

| component | onnx_export | onnx_validate | trt_build | trt_validate | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- | --- |
| vision_encoder | False | False | False | False | onnx_export | onnx_export has not passed |
| memory_attention | True | True | True | True |  |  |
| mask_decoder | True | True | True | True |  |  |
| memory_encoder | True | True | True | True |  |  |
