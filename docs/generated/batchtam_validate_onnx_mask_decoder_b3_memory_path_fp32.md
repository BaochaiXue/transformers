# BatchTam ONNX Validation: mask_decoder

| field | value |
| --- | --- |
| onnx_path | artifacts/edgetam_trt_b3_memory_path_fp32/onnx/mask_decoder_b3.onnx |
| providers | ['CUDAExecutionProvider', 'CPUExecutionProvider'] |
| onnx_validation_pass | True |
| failure_stage |  |
| exact_blocker |  |

| output | shape | pass | max_abs | p95_abs | mean_abs |
| --- | --- | --- | --- | --- | --- |
| output_0 | [3, 1, 3, 256, 256] | True | 0.226999 | 0.0647678 | 0.023207 |
| output_1 | [3, 1, 3] | True | 0.00371039 | 0.00346465 | 0.00163566 |
| output_2 | [3, 1, 3, 256] | True | 0.0610151 | 0.0278351 | 0.0102796 |
| output_3 | [3, 1, 1] | True | 0.0161438 | 0.0158638 | 0.0133657 |
