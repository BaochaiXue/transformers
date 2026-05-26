# Full batched multisession precision closed-loop summary

| precision_mode | compile_mode | strict | speed_first | avg IoU | p50 IoU | min IoU | empty mismatch | contract |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| all_bf16 | none | False | True | 0.952253 | 0.987773 | 0.578695 | 0 | True |
| all_fp32 | none | True | True | 0.986656 | 0.998423 | 0.587013 | 0 | True |
| decoder_fp32 | none | False | True | 0.949033 | 0.986489 | 0.563521 | 0 | True |
| memory_attention_fp32 | none | True | True | 0.982829 | 0.988694 | 0.724921 | 0 | True |
| memory_path_fp32 | none | True | True | 0.987242 | 0.990599 | 0.763158 | 0 | True |
| memory_path_fp32 | max-autotune-no-cudagraphs | False | False | 0.929624 | 0.971819 | 0.599941 | 0 | True |
| memory_path_fp32 | reduce-overhead | False | True | 0.943435 | 0.97675 | 0.599289 | 0 | True |
