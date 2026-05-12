# EdgeTAM Strict Full-Batched Contract Failure

- backend: `hf_batched_multisession_trt_components`
- compile_mode: `none`
- strict_full_batched: `True`
- correctness_pass: `False`
- failure_stage: `runtime_execution`

## Contract

- contract_pass: `False`
- batch_vision: `None`
- batch_memory_attention: `None`
- batch_mask_decoder: `None`
- batch_memory_encoder: `None`
- batched_state_scatter: `None`
- used_public_session_step_in_hot_path: `None`
- partial_fallback_used: `None`

## Blockers

- RuntimeError: BatchTam memory_attention engine was built for the fixed single-object tracking shape (num_object_pointer_tokens=4, num_spatial_memory_tokens=1), got 8 and 2
