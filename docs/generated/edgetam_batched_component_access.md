# EdgeTAM Component Access

| component | found | path | class | can_batch | notes |
| --- | --- | --- | --- | --- | --- |
| vision_encoder | True | vision_encoder | EdgeTamVisionModel | True | batch path uses model.get_image_features |
| memory_attention | True | memory_attention | EdgeTamVideoMemoryAttention | False | requires explicit state tensorization |
| mask_decoder | True | mask_decoder | EdgeTamVideoMaskDecoder | False | requires explicit state tensorization |
| memory_encoder | True | memory_encoder | EdgeTamVideoMemoryEncoder | False | requires explicit state tensorization |
| prompt_encoder | True | prompt_encoder | EdgeTamVideoPromptEncoder | False | requires explicit state tensorization |
