# EdgeTAM Original vs Compiled Delta

- baseline_backend: `hf_ref_seq_public`
- candidate_backend: `hf_batch_vision_seq_session`
- candidate_compile_mode: `reduce-overhead`
- empty_reference_policy: `ignore-candidate`
- evaluated_subset_mismatch: `False`

Delta is computed only on SAM3.1 reference-nonempty evaluated samples. Reference-empty samples are reported as uncertainty, not correctness evidence.

| object | baseline_iou | candidate_iou | delta | not_worse | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | reference_uncertain | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hand |  |  |  |  | 0 | 279 | 3 | True | no evaluated SAM3.1 reference samples |
| stuffed animal | 0.9634407977580824 | 0.9634821772065768 | 4.1379448494360815e-05 | True | 279 | 0 | 0 | False |  |
