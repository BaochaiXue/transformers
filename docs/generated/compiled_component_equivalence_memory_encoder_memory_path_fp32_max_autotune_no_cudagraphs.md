# Compiled EdgeTAM Component Equivalence

| field | value |
| --- | --- |
| component | memory_encoder |
| precision_mode | memory_path_fp32 |
| component_dtype | torch.float32 |
| compile_mode | max-autotune-no-cudagraphs |
| groups | 30 |
| compiled_component_equivalence_pass | True |
| compile_error |  |

| frame | eager | compiled | eager_vs_compiled | compiled_max | compiled_p95 | eager_compiled_max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | True | True | True | 0.0270019 | 0.00301504 | 0.000541449 |
| 1 | True | True | True | 0.182646 | 0.00445428 | 0.000509501 |
| 2 | True | True | True | 0.0943168 | 0.00440753 | 0.000555754 |
| 3 | True | True | True | 0.170426 | 0.00466813 | 0.000569731 |
| 4 | True | True | True | 0.16701 | 0.00416577 | 0.000497222 |
| 5 | True | True | True | 0.235589 | 0.00426328 | 0.000561953 |
| 6 | True | True | True | 0.110769 | 0.00423759 | 0.000509262 |
| 7 | True | True | True | 0.14753 | 0.00425053 | 0.000509918 |
| 8 | True | True | True | 0.13582 | 0.00437807 | 0.000473022 |
| 9 | True | True | True | 0.236644 | 0.00429922 | 0.000639439 |
| 10 | True | True | True | 0.147436 | 0.00422466 | 0.000488877 |
| 11 | True | True | True | 0.206869 | 0.00432041 | 0.000646293 |
| 12 | True | True | True | 0.204385 | 0.00425636 | 0.000508964 |
| 13 | True | True | True | 0.13446 | 0.00427771 | 0.000549793 |
| 14 | True | True | True | 0.265433 | 0.00433707 | 0.000542283 |
| 15 | True | True | True | 0.196602 | 0.00434232 | 0.000571132 |
| 16 | True | True | True | 0.142778 | 0.00426311 | 0.00062418 |
| 17 | True | True | True | 0.157324 | 0.00437152 | 0.0005036 |
| 18 | True | True | True | 0.135789 | 0.00446546 | 0.000565231 |
| 19 | True | True | True | 0.193532 | 0.00492377 | 0.000558734 |

## Blockers

- none
