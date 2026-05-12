# BatchTam ONNX Validation: memory_attention

| field | value |
| --- | --- |
| onnx_path | artifacts/edgetam_trt_b3_memory_path_fp32/onnx/memory_attention_b3.onnx |
| providers | ['CUDAExecutionProvider', 'CPUExecutionProvider'] |
| onnx_validation_pass | True |
| failure_stage |  |
| exact_blocker |  |

| output | shape | pass | max_abs | p95_abs | mean_abs |
| --- | --- | --- | --- | --- | --- |
| output | [1, 3, 4096, 256] | True | 0.041045 | 0.00536609 | 0.00189595 |
