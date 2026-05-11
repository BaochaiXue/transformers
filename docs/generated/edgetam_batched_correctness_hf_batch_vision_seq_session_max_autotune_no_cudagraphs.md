# EdgeTAM Batched Correctness

- backend: `hf_batch_vision_seq_session`
- compile_mode: `max-autotune-no-cudagraphs`
- correctness_pass: `False`
- mask_correctness_pass: `False`
- candidate_partial: `False`
- fallback_backend: `None`

## Blockers

- none

## Metrics

| key | iou_avg | iou_min | iou_p50 | ref_nonempty | cand_nonempty |
| --- | --- | --- | --- | --- | --- |
| cam0_obj0 | 0.7044512399706053 | 0.1549553208773355 | 0.8903178829030252 | 100 | 100 |
| cam0_obj1 | 0.9511634320412559 | 0.6999554168524298 | 0.9770590366434339 | 100 | 100 |
| cam1_obj0 | 0.9565806231060868 | 0.9127218934911243 | 0.9563710116070467 | 100 | 100 |
| cam1_obj1 | 0.9730193909563201 | 0.835529693650522 | 0.9748648117746637 | 100 | 100 |
| cam2_obj0 | 0.9216048121735908 | 0.7600502512562815 | 0.9349777014411176 | 100 | 100 |
| cam2_obj1 | 0.977231214620709 | 0.8332593578931795 | 0.9821293393941053 | 100 | 100 |
