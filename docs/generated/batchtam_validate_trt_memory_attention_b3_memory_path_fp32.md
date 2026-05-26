# BatchTam TensorRT Validation: memory_attention

| field | value |
| --- | --- |
| engine_path | artifacts/edgetam_trt_b3_memory_path_fp32/engines/memory_attention_b3.engine |
| trt_validation_pass | True |
| hot_path_zero_copy | True |
| failure_stage |  |
| exact_blocker |  |

| output | shape | pass | max_abs | p95_abs | mean_abs |
| --- | --- | --- | --- | --- | --- |
| output | [1, 3, 4096, 256] | True | 0.0413014 | 0.00536614 | 0.0018949 |
