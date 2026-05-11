# EdgeTAM Original vs Compiled Delta

- baseline_backend: `hf_ref_seq_public`
- candidate_backend: `hf_batched_multisession`
- candidate_compile_mode: `none`
- empty_reference_policy: `ignore-candidate`
- evaluated_subset_mismatch: `False`

Delta is computed only on SAM3.1 reference-nonempty evaluated samples. Reference-empty samples are reported as uncertainty, not correctness evidence.

| object | baseline_iou | candidate_iou | delta | not_worse | evaluated | ref_empty_ignored | cand_nonempty_ref_empty | reference_uncertain | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stuffed animal | 0.9634407977580824 | 0.9634226442517867 | -1.81535062957483e-05 | True | 279 | 0 | 0 | False |  |
