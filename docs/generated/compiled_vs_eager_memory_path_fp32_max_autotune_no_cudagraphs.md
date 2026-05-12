# Compiled vs Eager Full Batched EdgeTAM Debug

| field | value |
| --- | --- |
| precision_mode | memory_path_fp32 |
| compile_mode | max-autotune-no-cudagraphs |
| compile_scope | memory_path_all |
| first_diverging_component | memory_path_component_or_state_scatter |
| first_bad_frame | 82 |
| first_bad_camera | cam1 |
| IoU HF/eager | 0.984961 |
| IoU HF/compiled | 0.979231 |
| IoU eager/compiled | 0.991807 |

| comparison | avg | min | p50 |
| --- | --- | --- | --- |
| hf_public_vs_eager | 0.987242 | 0.763158 | 0.990599 |
| hf_public_vs_compiled | 0.987332 | 0.762748 | 0.990709 |
| eager_vs_compiled | 0.999212 | 0.985824 | 0.999778 |
