# Compiled EdgeTAM Component Equivalence

| field | value |
| --- | --- |
| component | mask_decoder |
| precision_mode | memory_path_fp32 |
| component_dtype | torch.float32 |
| compile_mode | max-autotune-no-cudagraphs |
| groups | 30 |
| compiled_component_equivalence_pass | True |
| compile_error |  |

| frame | eager | compiled | eager_vs_compiled | compiled_max | compiled_p95 | eager_compiled_max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | True | True | True | 0.22525 | 0.064292 | 0.00397491 |
| 1 | True | True | True | 0.199654 | 0.0575819 | 0.00490379 |
| 2 | True | True | True | 0.207403 | 0.0628453 | 0.00289726 |
| 3 | True | True | True | 0.200342 | 0.0602112 | 0.00438499 |
| 4 | True | True | True | 0.191891 | 0.0633373 | 0.00451088 |
| 5 | True | True | True | 0.194578 | 0.0601025 | 0.00478554 |
| 6 | True | True | True | 0.190662 | 0.0586281 | 0.00394821 |
| 7 | True | True | True | 0.222614 | 0.0607328 | 0.00299263 |
| 8 | True | True | True | 0.191177 | 0.061657 | 0.00320816 |
| 9 | True | True | True | 0.171883 | 0.058015 | 0.00352955 |
| 10 | True | True | True | 0.196251 | 0.0662184 | 0.00310898 |
| 11 | True | True | True | 0.19182 | 0.0665953 | 0.00416756 |
| 12 | True | True | True | 0.236716 | 0.0643178 | 0.00775909 |
| 13 | True | True | True | 0.200624 | 0.0674534 | 0.00562191 |
| 14 | True | True | True | 0.213053 | 0.075224 | 0.00678253 |
| 15 | True | True | True | 0.223231 | 0.0646057 | 0.00330162 |
| 16 | True | True | True | 0.186342 | 0.0579243 | 0.00326252 |
| 17 | True | True | True | 0.23992 | 0.0598379 | 0.00507355 |
| 18 | True | True | True | 0.237801 | 0.0621595 | 0.00447464 |
| 19 | True | True | True | 0.232981 | 0.0664597 | 0.00409985 |

## Blockers

- none
