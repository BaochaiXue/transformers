# Compiled EdgeTAM Component Equivalence

| field | value |
| --- | --- |
| component | memory_encoder |
| precision_mode | memory_path_fp32 |
| component_dtype | torch.float32 |
| compile_mode | reduce-overhead |
| groups | 30 |
| compiled_component_equivalence_pass | True |
| compile_error |  |

| frame | eager | compiled | eager_vs_compiled | compiled_max | compiled_p95 | eager_compiled_max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | True | True | True | 0.0268917 | 0.0030064 | 0.000227928 |
| 1 | True | True | True | 0.182749 | 0.00445123 | 0.000621796 |
| 2 | True | True | True | 0.0944635 | 0.00440443 | 0.000840366 |
| 3 | True | True | True | 0.170431 | 0.00466609 | 0.000583827 |
| 4 | True | True | True | 0.166965 | 0.00416153 | 0.000563622 |
| 5 | True | True | True | 0.235567 | 0.00426072 | 0.000422478 |
| 6 | True | True | True | 0.11071 | 0.0042318 | 0.000355184 |
| 7 | True | True | True | 0.147541 | 0.00424677 | 0.000535488 |
| 8 | True | True | True | 0.135853 | 0.0043745 | 0.000472516 |
| 9 | True | True | True | 0.236372 | 0.00429516 | 0.000455856 |
| 10 | True | True | True | 0.14731 | 0.00422096 | 0.00043267 |
| 11 | True | True | True | 0.206902 | 0.00431824 | 0.000455856 |
| 12 | True | True | True | 0.204537 | 0.00425398 | 0.000552952 |
| 13 | True | True | True | 0.134407 | 0.00427425 | 0.000455856 |
| 14 | True | True | True | 0.265484 | 0.00433172 | 0.000442505 |
| 15 | True | True | True | 0.196743 | 0.00433742 | 0.000511527 |
| 16 | True | True | True | 0.142841 | 0.00426098 | 0.000265062 |
| 17 | True | True | True | 0.157178 | 0.00436735 | 0.000468016 |
| 18 | True | True | True | 0.135823 | 0.00446141 | 0.000590086 |
| 19 | True | True | True | 0.19347 | 0.00492036 | 0.000492796 |

## Blockers

- none
