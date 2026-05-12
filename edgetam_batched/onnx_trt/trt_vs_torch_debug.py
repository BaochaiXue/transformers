"""Debug first TensorRT-vs-PyTorch component divergence for BatchTam."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edgetam_batched.report_utils import write_json, write_markdown


def summarize_tensor_diff(torch_tensor: Any, trt_tensor: Any) -> dict[str, Any]:
    import torch

    if not isinstance(torch_tensor, torch.Tensor) or not isinstance(trt_tensor, torch.Tensor):
        return {"comparable": False, "reason": "non-tensor"}
    if tuple(torch_tensor.shape) != tuple(trt_tensor.shape):
        return {
            "comparable": False,
            "reason": "shape_mismatch",
            "torch_shape": list(torch_tensor.shape),
            "trt_shape": list(trt_tensor.shape),
        }
    diff = (torch_tensor.float() - trt_tensor.float()).abs().flatten()
    return {
        "comparable": True,
        "shape": list(torch_tensor.shape),
        "dtype_torch": str(torch_tensor.dtype),
        "dtype_trt": str(trt_tensor.dtype),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "p95_abs_diff": float(torch.quantile(diff, 0.95).item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        f"- component: `{payload.get('component')}`",
        f"- failure_stage: `{payload.get('failure_stage')}`",
        f"- exact_blocker: `{payload.get('exact_blocker')}`",
    ]
    return "\n".join(["# BatchTam TRT vs PyTorch Debug", "", *rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay")
    parser.add_argument("--component", required=True)
    parser.add_argument("--trt-engine-dir", required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--precision-mode", default="memory_path_fp32")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Full frame-by-frame TRT-vs-torch debugging is driven by compare_multisession
    # reports. This command exists as a structured failure-report entry point so
    # a failed closed-loop scope can record a precise next diagnostic without
    # silently rerunning PyTorch fallback in the hot path.
    payload = {
        "component": args.component,
        "trt_engine_dir": str(Path(args.trt_engine_dir)),
        "precision_mode": args.precision_mode,
        "failure_stage": "not_run",
        "exact_blocker": "run closed-loop scope first, then compare component tensors from the failed frame",
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
