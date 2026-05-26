"""State leakage checks for batched camera sessions."""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .camera_order import mask_iou


def compare_cam0_stability(original_mask, perturbed_mask, logit_a=None, logit_b=None) -> dict:
    result = {
        "cam0_iou_original_vs_perturbed": mask_iou(original_mask, perturbed_mask),
        "pass": False,
    }
    if logit_a is not None and logit_b is not None and torch is not None:
        diff = (logit_a.detach().float() - logit_b.detach().float()).abs().flatten()
        result["cam0_logit_abs_diff_p95"] = float(torch.quantile(diff, 0.95).item()) if diff.numel() else 0.0
    result["pass"] = result["cam0_iou_original_vs_perturbed"] >= 0.99
    return result
