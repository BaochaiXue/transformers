"""Small deterministic statistics helpers used by reports and tests."""

from __future__ import annotations

import math
from typing import Iterable


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    if q <= 0:
        return vals[0]
    if q >= 100:
        return vals[-1]
    rank = (len(vals) - 1) * (q / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return vals[lo]
    weight = rank - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "count": 0,
            "avg": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(vals),
        "avg": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
    }
