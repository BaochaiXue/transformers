# BatchTam Export Fixtures

- fixtures_dir: ``
- out_dir: `artifacts/edgetam_trt_b3_memory_path_fp32/export_fixtures`
- shape_sequence_json: `docs/generated/batchtam_memory_attention_shape_sequence.json`
- batch_size: `3`
- object_count: `1`
- precision_mode: `memory_path_fp32`
- all_components_pass: `True`

| component | bucket | shape_key | pass | inputs | outputs | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| memory_attention | shape_002_objptr12_spmem3 | objptr12_spmem3_cur4096_mem1548 | True | 4 | 1 |  |  |
| memory_attention | shape_003_objptr16_spmem4 | objptr16_spmem4_cur4096_mem2064 | True | 4 | 1 |  |  |
| memory_attention | shape_004_objptr20_spmem5 | objptr20_spmem5_cur4096_mem2580 | True | 4 | 1 |  |  |
| memory_attention | shape_005_objptr24_spmem6 | objptr24_spmem6_cur4096_mem3096 | True | 4 | 1 |  |  |
| memory_attention | shape_006_objptr28_spmem7 | objptr28_spmem7_cur4096_mem3612 | True | 4 | 1 |  |  |
| memory_attention | shape_007_objptr32_spmem7 | objptr32_spmem7_cur4096_mem3616 | True | 4 | 1 |  |  |
| memory_attention | shape_008_objptr36_spmem7 | objptr36_spmem7_cur4096_mem3620 | True | 4 | 1 |  |  |
| memory_attention | shape_009_objptr40_spmem7 | objptr40_spmem7_cur4096_mem3624 | True | 4 | 1 |  |  |
| memory_attention | shape_010_objptr44_spmem7 | objptr44_spmem7_cur4096_mem3628 | True | 4 | 1 |  |  |
| memory_attention | shape_011_objptr48_spmem7 | objptr48_spmem7_cur4096_mem3632 | True | 4 | 1 |  |  |
| memory_attention | shape_000_objptr4_spmem1 | objptr4_spmem1_cur4096_mem516 | True | 4 | 1 |  |  |
| memory_attention | shape_012_objptr52_spmem7 | objptr52_spmem7_cur4096_mem3636 | True | 4 | 1 |  |  |
| memory_attention | shape_013_objptr56_spmem7 | objptr56_spmem7_cur4096_mem3640 | True | 4 | 1 |  |  |
| memory_attention | shape_014_objptr60_spmem7 | objptr60_spmem7_cur4096_mem3644 | True | 4 | 1 |  |  |
| memory_attention | shape_015_objptr64_spmem7 | objptr64_spmem7_cur4096_mem3648 | True | 4 | 1 |  |  |
| memory_attention | shape_001_objptr8_spmem2 | objptr8_spmem2_cur4096_mem1032 | True | 4 | 1 |  |  |
