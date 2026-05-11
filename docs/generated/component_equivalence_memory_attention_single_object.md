# EdgeTAM Component Batch Equivalence

- trace: `docs/generated/hf_edgetam_component_trace_single_object_30f.json`
- all_components_pass: `False`

| component | trace_records | pass | blockers |
| --- | --- | --- | --- |
| memory_attention | 87 | False | requires tensorized memory/object-pointer inputs across camera sessions; raw tensor equivalence is not proven by shape-only trace |
