# BatchTam TensorRT Validation: memory_encoder

| field | value |
| --- | --- |
| engine_path | artifacts/edgetam_trt_b3_memory_path_fp32/engines/memory_encoder_b3.engine |
| trt_validation_pass | True |
| hot_path_zero_copy | True |
| failure_stage |  |
| exact_blocker |  |

| output | shape | pass | max_abs | p95_abs | mean_abs |
| --- | --- | --- | --- | --- | --- |
| output_0 | [3, 64, 64, 64] | True | 0.0270824 | 0.00395363 | 0.00140514 |
| output_1 | [3, 64, 64, 64] | True | 2.38419e-07 | 1.49012e-08 | 3.64629e-09 |
