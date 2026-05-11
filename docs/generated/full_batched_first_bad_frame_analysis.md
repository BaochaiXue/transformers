# Full Batched EdgeTAM First Bad Frame

- backend: `hf_batched_multisession`
- dtype: `bfloat16`
- replay: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal`
- threshold: `0.9`

## First Bad Sample

| frame_idx | camera | object | IoU | first_diverging_component |
| --- | --- | --- | --- | --- |
| 43 | cam1 | 0 | 0.592053 | mask_decoder_or_accumulated_state |

## Field Diffs

| field | shape_match | max_abs_diff | mean_abs_diff | p95_abs_diff |
| --- | --- | --- | --- | --- |
| maskmem_features | True | 5.46875 | 0.218978 | 1.04263 |
| maskmem_pos_enc | True | 0 | 0 | 0 |
| object_pointer | True | 0.955078 | 0.201257 | 0.491714 |
| object_score_logits | True | 0.21875 | 0.21875 | 0.21875 |
| pred_masks | True | 7.98438 | 1.93154 | 3.25 |

## Interpretation

The strict full backend contract can be true while current masks still drift. Component fixture equivalence should be read together with this report: if isolated component equivalence passes but this report fails after many frames, the blocker is the recurrent session memory/state update path rather than a single isolated call.
