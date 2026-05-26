# EdgeTAM IoU vs SAM3.1 Replay Reference

- reference_source: `sam31-replay`
- backend: `hf_batch_vision_seq_session`
- prompt/init source: `sam31_frame0_video_reference_masks`

| compile_mode | pass | global_iou_avg | global_iou_min | global_iou_p50 | empty_mismatch |
| --- | --- | --- | --- | --- | --- |
| none | True | 0.9813712332600867 | 0.9403201089732675 | 0.9907300229542289 | 0 |
| max-autotune-no-cudagraphs | True | 0.9812234233519859 | 0.9388694688379231 | 0.9907300229542289 | 0 |
| reduce-overhead | True | 0.9813799883990333 | 0.9383450973722255 | 0.9907300229542289 | 0 |

## Notes

- `obj0` is controller/towel. In this replay SAM3.1 produced empty controller masks on all cameras, so obj0 IoU is 1.0 because both sides are empty.
- `obj1` is stuffed animal and is the meaningful object mask quality row.
