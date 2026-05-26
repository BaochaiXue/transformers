# Compiled EdgeTAM Component Equivalence

| field | value |
| --- | --- |
| component | memory_attention |
| precision_mode | memory_path_fp32 |
| component_dtype | torch.float32 |
| compile_mode | reduce-overhead |
| groups | 29 |
| compiled_component_equivalence_pass | True |
| compile_error |  |

| frame | eager | compiled | eager_vs_compiled | compiled_max | compiled_p95 | eager_compiled_max |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | True | True | True | 0.041302 | 0.00536597 | 2.98023e-06 |
| 2 | True | True | True | 0.0465004 | 0.00547202 | 3.21865e-06 |
| 3 | True | True | True | 0.0380472 | 0.0055421 | 3.60608e-06 |
| 4 | True | True | True | 0.0389541 | 0.00532712 | 2.6226e-06 |
| 5 | True | True | True | 0.0382869 | 0.00531149 | 2.6226e-06 |
| 6 | True | True | True | 0.0357664 | 0.00525975 | 2.14577e-06 |
| 7 | True | True | True | 0.042809 | 0.00521934 | 2.5034e-06 |
| 8 | True | True | True | 0.0445991 | 0.00519598 | 2.29478e-06 |
| 9 | True | True | True | 0.0490522 | 0.00517596 | 2.5034e-06 |
| 10 | True | True | True | 0.0420487 | 0.00517532 | 2.38419e-06 |
| 11 | True | True | True | 0.0440667 | 0.00511676 | 2.44379e-06 |
| 12 | True | True | True | 0.0372423 | 0.00514603 | 2.80142e-06 |
| 13 | True | True | True | 0.0433524 | 0.00507034 | 2.563e-06 |
| 14 | True | True | True | 0.0379679 | 0.00505972 | 2.14577e-06 |
| 15 | True | True | True | 0.0363665 | 0.00508572 | 3.55393e-06 |
| 16 | True | True | True | 0.0376656 | 0.00505984 | 2.23517e-06 |
| 17 | True | True | True | 0.0385965 | 0.00503397 | 2.44379e-06 |
| 18 | True | True | True | 0.0438895 | 0.00499511 | 2.38419e-06 |
| 19 | True | True | True | 0.0387677 | 0.00502457 | 2.14577e-06 |
| 20 | True | True | True | 0.0373919 | 0.00499219 | 2.20537e-06 |

## Blockers

- none
