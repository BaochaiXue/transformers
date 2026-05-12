# BatchTam TensorRT Validation: mask_decoder

| field | value |
| --- | --- |
| engine_path | artifacts/edgetam_trt_b3_memory_path_fp32/engines/mask_decoder_b3.engine |
| trt_validation_pass | True |
| hot_path_zero_copy | True |
| failure_stage |  |
| exact_blocker |  |

| output | shape | pass | max_abs | p95_abs | mean_abs |
| --- | --- | --- | --- | --- | --- |
| output_0 | [3, 1, 3, 256, 256] | True | 0.224461 | 0.0643654 | 0.0233586 |
| output_1 | [3, 1, 3] | True | 0.00371501 | 0.00351065 | 0.00164784 |
| output_2 | [3, 1, 3, 256] | True | 0.0626865 | 0.0281509 | 0.0102337 |
| output_3 | [3, 1, 1] | True | 0.0150161 | 0.0148366 | 0.0134617 |
