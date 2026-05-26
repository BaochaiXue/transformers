# Harness Engineering Policy

The EdgeTAM batched fork is managed through deterministic harness commands.

## Principle

Manual experiments can inform debugging, but reusable results must be produced
by scripts under `scripts/harness/` and recorded under `docs/generated/`.

## Required Flow

1. Add or update code under `edgetam_batched/`.
2. Add deterministic tests under `tests/`.
3. Run:

   ```bash
   conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py
   ```

4. For HF smoke performance, run:

   ```bash
   conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py --with-synthetic-profile
   ```

5. Do not claim a real backend performance result unless it uses real RGB replay
   and a passing correctness report.

## Current Baseline

The current validated real-scene baseline still lives in QQTT Demo 2.1.5:

```text
hf_batch_vision_seq_session + vision-reduce-overhead
stage_wall p50: 77.92 ms
stage_wall p90: 86.24 ms
complete group FPS: 12.48
```

This fork currently provides the harness and state-map scaffold for the next
batched multi-session backend. It does not yet replace the QQTT backend.
