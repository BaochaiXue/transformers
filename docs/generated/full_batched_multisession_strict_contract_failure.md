# EdgeTAM Strict Full-Batched Backend Contract

| field | value |
| --- | --- |
| backend | hf_batched_multisession |
| contract_pass | False |
| batch_vision | True |
| batch_memory_attention | False |
| batch_mask_decoder | False |
| batch_memory_encoder | False |
| batched_state_scatter | False |
| used_public_session_step_in_hot_path | True |
| partial_fallback_used | True |
| missing_requirements | batch_memory_attention, batch_mask_decoder, batch_memory_encoder, batched_state_scatter, used_public_session_step_in_hot_path, partial_fallback_used |

## Blockers

- memory attention is not batched across camera sessions
- mask decoder is not batched across camera sessions
- memory encoder/state update is not batched across camera sessions
- session state scatter is not implemented
- current code still calls model(inference_session=..., frame=...) per camera
