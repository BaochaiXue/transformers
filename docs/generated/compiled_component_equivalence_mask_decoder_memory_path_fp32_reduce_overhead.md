# Compiled EdgeTAM Component Equivalence

| field | value |
| --- | --- |
| component | mask_decoder |
| precision_mode | memory_path_fp32 |
| component_dtype | torch.float32 |
| compile_mode | reduce-overhead |
| groups | 30 |
| compiled_component_equivalence_pass | True |
| compile_error |  |

| frame | eager | compiled | eager_vs_compiled | compiled_max | compiled_p95 | eager_compiled_max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | True | True | True | 0.225245 | 0.064291 | 0.00397301 |
| 1 | True | True | True | 0.199657 | 0.0575829 | 0.00327778 |
| 2 | True | True | True | 0.207403 | 0.0628452 | 0.00370598 |
| 3 | True | True | True | 0.200344 | 0.0602093 | 0.00438404 |
| 4 | True | True | True | 0.191889 | 0.0633354 | 0.00357628 |
| 5 | True | True | True | 0.194578 | 0.0601015 | 0.00430679 |
| 6 | True | True | True | 0.19066 | 0.0586291 | 0.0052042 |
| 7 | True | True | True | 0.222616 | 0.0607338 | 0.00339413 |
| 8 | True | True | True | 0.191174 | 0.0616513 | 0.00502586 |
| 9 | True | True | True | 0.171881 | 0.0580158 | 0.00390816 |
| 10 | True | True | True | 0.196251 | 0.0662174 | 0.00474644 |
| 11 | True | True | True | 0.191818 | 0.0665923 | 0.0041666 |
| 12 | True | True | True | 0.236715 | 0.0643225 | 0.00775909 |
| 13 | True | True | True | 0.200621 | 0.0674562 | 0.00558281 |
| 14 | True | True | True | 0.213051 | 0.075225 | 0.0067749 |
| 15 | True | True | True | 0.223231 | 0.0646086 | 0.00316429 |
| 16 | True | True | True | 0.186342 | 0.0579215 | 0.00326347 |
| 17 | True | True | True | 0.239925 | 0.0598356 | 0.00416279 |
| 18 | True | True | True | 0.237804 | 0.0621605 | 0.00447559 |
| 19 | True | True | True | 0.232981 | 0.0664606 | 0.00407314 |

## Blockers

- none
