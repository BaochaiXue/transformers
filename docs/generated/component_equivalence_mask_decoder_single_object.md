# EdgeTAM Component Batch Equivalence

- trace: `docs/generated/hf_edgetam_component_trace_single_object_30f.json`
- all_components_pass: `False`

| component | trace_records | pass | blockers |
| --- | --- | --- | --- |
| mask_decoder | 90 | False | requires tensorized prompted/empty-tracking decoder inputs across camera sessions; raw tensor equivalence is not proven by shape-only trace |
