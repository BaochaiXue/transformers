# Compiled EdgeTAM Component Equivalence

| field | value |
| --- | --- |
| component | memory_attention |
| precision_mode | memory_path_fp32 |
| component_dtype | torch.float32 |
| compile_mode | max-autotune-no-cudagraphs |
| groups | 29 |
| compiled_component_equivalence_pass | True |
| compile_error |  |

| frame | eager | compiled | eager_vs_compiled | compiled_max | compiled_p95 | eager_compiled_max |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | True | True | True | 0.0413018 | 0.00536595 | 3.30806e-06 |
| 2 | True | True | True | 0.0465004 | 0.00547206 | 2.99141e-06 |
| 3 | True | True | True | 0.0380467 | 0.00554205 | 3.51667e-06 |
| 4 | True | True | True | 0.0389546 | 0.00532717 | 2.563e-06 |
| 5 | True | True | True | 0.0382869 | 0.00531149 | 2.6226e-06 |
| 6 | True | True | True | 0.0357666 | 0.00525974 | 2.38419e-06 |
| 7 | True | True | True | 0.0428095 | 0.00521934 | 2.5034e-06 |
| 8 | True | True | True | 0.0445999 | 0.00519593 | 2.38419e-06 |
| 9 | True | True | True | 0.0490522 | 0.00517595 | 2.92063e-06 |
| 10 | True | True | True | 0.0420477 | 0.00517541 | 2.38419e-06 |
| 11 | True | True | True | 0.0440668 | 0.00511682 | 2.31713e-06 |
| 12 | True | True | True | 0.0372427 | 0.005146 | 2.19047e-06 |
| 13 | True | True | True | 0.043353 | 0.00507033 | 2.68221e-06 |
| 14 | True | True | True | 0.0379677 | 0.00505972 | 2.02656e-06 |
| 15 | True | True | True | 0.0363665 | 0.00508577 | 2.38419e-06 |
| 16 | True | True | True | 0.0376656 | 0.00505984 | 2.26498e-06 |
| 17 | True | True | True | 0.0385967 | 0.00503397 | 2.38419e-06 |
| 18 | True | True | True | 0.0438895 | 0.00499511 | 2.38419e-06 |
| 19 | True | True | True | 0.0387676 | 0.00502455 | 2.5332e-06 |
| 20 | True | True | True | 0.0373912 | 0.00499225 | 3.03984e-06 |

## Blockers

- none
