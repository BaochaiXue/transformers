# HF EdgeTAM Component Trace

- rgb_replay: `/home/zhangxinjie/proj-QQTT-v2/result/different_types_sloth_set_2_motion_ffs_replay_hand_stuffed_animal`
- frames: `30`
- object_count: `1`

## Component Calls

| component | calls |
| --- | --- |
| mask_decoder | 90 |
| memory_attention | 87 |
| memory_encoder | 90 |
| prompt_encoder | 90 |
| vision_encoder | 90 |

## Full Batching Blockers

- `EdgeTamVideoModel.forward` loops over objects and calls `_run_single_frame_inference` per object.
- `EdgeTamVideoModel._batch_encode_memories` is currently `NotImplemented` in the modular source.
- Memory selection/object pointer gathering is driven by nested `inference_session.output_dict_per_obj` Python dictionaries.
