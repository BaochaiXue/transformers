# Compiled vs Eager Full Batched EdgeTAM Debug

| field | value |
| --- | --- |
| precision_mode | memory_path_fp32 |
| compile_mode | reduce-overhead |
| compile_scope | memory_path_all |
| first_diverging_component |  |
| first_bad_frame |  |
| first_bad_camera |  |
| IoU HF/eager |  |
| IoU HF/compiled |  |
| IoU eager/compiled |  |

| comparison | avg | min | p50 |
| --- | --- | --- | --- |
| hf_public_vs_eager | 0.987242 | 0.763158 | 0.990599 |
| hf_public_vs_compiled | 0.987236 | 0.762995 | 0.990446 |
| eager_vs_compiled | 0.99946 | 0.984788 | 1 |
