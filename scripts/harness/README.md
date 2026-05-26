# EdgeTAM Batched Harness

This fork is managed as a harness-driven research backend. Do not treat ad-hoc
interactive runs as validation.

## Quick Check

```bash
cd /home/zhangxinjie/EdgeTAM-HF-batched
conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py
```

This runs:

- Python compile check for `edgetam_batched/*.py`
- deterministic unit tests
- source inspection report generation
- synthetic session state-map report generation
- scaffold-only profile report generation

## Optional Synthetic HF Profile

```bash
conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py --with-synthetic-profile
```

This loads `yonigozlan/EdgeTAM-hf` and runs a synthetic public-session profile.
It is useful for a smoke test, but it is not a validated batched multi-session
runtime benchmark.

## Validation Rule

`hf_batched_multisession` is not usable as a Demo 2.2 backend until a real
100-frame RGB triplet replay correctness report passes:

- camera order check
- state leakage check
- non-empty mask parity
- IoU/logit drift metrics
- performance profile after correctness
