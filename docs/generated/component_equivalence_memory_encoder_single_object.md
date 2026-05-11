# EdgeTAM Component Batch Equivalence

- trace: `docs/generated/hf_edgetam_component_trace_single_object_30f.json`
- all_components_pass: `False`

| component | trace_records | pass | blockers |
| --- | --- | --- | --- |
| memory_encoder | 90 | False | EdgeTamVideoModel._batch_encode_memories is NotImplemented; full state update cannot be batched yet |
