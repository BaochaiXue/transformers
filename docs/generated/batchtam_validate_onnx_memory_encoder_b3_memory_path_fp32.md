# BatchTam ONNX Validation: memory_encoder

| field | value |
| --- | --- |
| onnx_path | artifacts/edgetam_trt_b3_memory_path_fp32/onnx/memory_encoder_b3.onnx |
| providers | ['CUDAExecutionProvider', 'CPUExecutionProvider'] |
| onnx_validation_pass | True |
| failure_stage |  |
| exact_blocker |  |

| output | shape | pass | max_abs | p95_abs | mean_abs |
| --- | --- | --- | --- | --- | --- |
| output_0 | [3, 64, 64, 64] | True | 0.0267849 | 0.00396337 | 0.00140644 |
| output_1 | [3, 64, 64, 64] | True | 5.96046e-08 | 0 | 2.15368e-09 |
