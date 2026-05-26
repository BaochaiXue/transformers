# Current-frame precision grid: frame47 cam1

| precision_mode | cam1 IoU | fixes current frame | pred_masks p95 | object_pointer p95 | maskmem_features p95 |
| --- | ---: | --- | ---: | ---: | ---: |
| all_bf16 | 0.734722 | no | 6.8125 | 0.849121 | 1.06216 |
| object_pointer_fp32 | 0.734722 | no | 6.8125 | 0.849121 | 1.06216 |
| maskmem_features_fp32 | 0.734722 | no | 6.8125 | 0.849121 | 1.0617 |
| mask_logits_fp32 | 0.734722 | no | 6.8125 | 0.849121 | 1.06216 |
| decoder_fp32 | 1 | yes | 0.0575497 | 0.00592553 | 0.143213 |
| memory_attention_fp32 | 0.999548 | yes | 0.0625 | 0.0109863 | 0.148438 |
| memory_encoder_fp32 | 0.734722 | no | 6.8125 | 0.849121 | 1.04066 |
| memory_path_fp32 | 0.999548 | yes | 0.0585098 | 0.00737936 | 0.388029 |
| all_fp32 | 0.99908 | yes | 0.01009 | 0.00219432 | 0.0104502 |

Passing current-frame modes: decoder_fp32, memory_attention_fp32, memory_path_fp32, all_fp32
