# BatchTam Export Fixtures

- fixtures_dir: `docs/generated/fixtures/single_object`
- out_dir: `artifacts/edgetam_trt_b3_memory_path_fp32/export_fixtures`
- batch_size: `3`
- object_count: `1`
- precision_mode: `memory_path_fp32`
- all_components_pass: `True`

| component | pass | inputs | outputs | failure_stage | blocker |
| --- | --- | --- | --- | --- | --- |
| memory_attention | True | 4 | 1 |  |  |
| mask_decoder | True | 6 | 4 |  |  |
| memory_encoder | True | 2 | 2 |  |  |
