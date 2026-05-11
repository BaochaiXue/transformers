"""Camera order validation for batched multi-session masks."""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def mask_iou(a, b) -> float:
    if torch is not None and isinstance(a, torch.Tensor):
        aa = a.bool()
        bb = b.bool()
        inter = torch.logical_and(aa, bb).sum().item()
        union = torch.logical_or(aa, bb).sum().item()
    else:
        aa = np.asarray(a).astype(bool)
        bb = np.asarray(b).astype(bool)
        inter = int(np.logical_and(aa, bb).sum())
        union = int(np.logical_or(aa, bb).sum())
    return 1.0 if union == 0 else float(inter) / float(union)


def iou_matrix(candidate_masks: Sequence, reference_masks: Sequence) -> list[list[float]]:
    return [[mask_iou(candidate, reference) for reference in reference_masks] for candidate in candidate_masks]


def diagonal_best(matrix: Sequence[Sequence[float]]) -> dict:
    failures = []
    for row_idx, row in enumerate(matrix):
        best_value = max(row)
        if row[row_idx] < best_value:
            best_idx = max(range(len(row)), key=lambda idx: row[idx])
            failures.append({"candidate": row_idx, "best_reference": best_idx, "row": list(row)})
    return {
        "pass": len(failures) == 0,
        "failures": failures,
        "matrix": [list(row) for row in matrix],
    }
