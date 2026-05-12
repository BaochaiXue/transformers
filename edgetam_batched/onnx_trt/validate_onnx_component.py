"""Validate a BatchTam ONNX component against eager fixture outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from . import COMPONENTS
from .trt_engine_utils import load_io_spec


def tensor_to_numpy(tensor: Any):
    import torch

    value = tensor.detach().cpu()
    if value.dtype is torch.bfloat16:
        value = value.float()
    return value.numpy()


def diff_stats(expected, actual) -> dict[str, float | bool]:
    import numpy as np

    exp = expected.astype("float32", copy=False)
    act = actual.astype("float32", copy=False)
    if exp.shape != act.shape:
        return {"pass": False, "max_abs_diff": float("inf"), "mean_abs_diff": float("inf"), "p95_abs_diff": float("inf")}
    diff = np.abs(exp - act).reshape(-1)
    if diff.size == 0:
        return {"pass": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "p95_abs_diff": 0.0}
    return {
        "pass": bool(float(diff.max()) <= 5e-1 and float(np.quantile(diff, 0.95)) <= 1.25e-1 and float(diff.mean()) <= 4e-2),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(np.quantile(diff, 0.95)),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    fixture_dir = Path(args.fixtures_dir) / args.component
    sample_inputs = torch.load(fixture_dir / "sample_inputs.pt", map_location="cpu", weights_only=False)
    sample_outputs = torch.load(fixture_dir / "sample_outputs_eager.pt", map_location="cpu", weights_only=False)
    io_spec = load_io_spec(fixture_dir / "io_spec.json")
    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        return failure(args, "onnx_validation", f"missing ONNX file: {onnx_path}")
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        return failure(args, "onnx_validation", f"onnxruntime unavailable: {exc!r}")

    providers = [provider for provider in args.providers if provider in ort.get_available_providers()]
    if not providers:
        providers = ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        feed = {
            spec.name: tensor_to_numpy(tensor)
            for spec, tensor in zip(io_spec.inputs, sample_inputs["flat_inputs"], strict=False)
        }
        outputs = session.run([spec.name for spec in io_spec.outputs[: len(session.get_outputs())]], feed)
    except Exception as exc:  # noqa: BLE001
        return failure(args, "onnx_validation", repr(exc), providers=providers)

    rows = []
    pass_all = True
    for idx, (spec, expected, actual) in enumerate(zip(io_spec.outputs, sample_outputs["flat_outputs"], outputs, strict=False)):
        stats = diff_stats(tensor_to_numpy(expected), actual)
        pass_all = pass_all and bool(stats["pass"])
        rows.append({"name": spec.name, "index": idx, "shape": list(actual.shape), **stats})
    return {
        "component": args.component,
        "onnx_path": str(onnx_path),
        "providers": providers,
        "onnx_validation_pass": pass_all,
        "rows": rows,
        "failure_stage": None if pass_all else "onnx_validation",
        "exact_blocker": None if pass_all else "one or more ONNX outputs exceed tolerance",
    }


def failure(args: argparse.Namespace, stage: str, blocker: str, **extra) -> dict[str, Any]:
    return {
        "component": args.component,
        "onnx_path": args.onnx_path,
        "onnx_validation_pass": False,
        "failure_stage": stage,
        "exact_blocker": blocker,
        **extra,
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        [
            row.get("name"),
            row.get("shape"),
            row.get("pass"),
            f"{row.get('max_abs_diff', 0):.6g}",
            f"{row.get('p95_abs_diff', 0):.6g}",
            f"{row.get('mean_abs_diff', 0):.6g}",
        ]
        for row in payload.get("rows", [])
    ]
    return "\n".join(
        [
            f"# BatchTam ONNX Validation: {payload.get('component')}",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["onnx_path", payload.get("onnx_path")],
                    ["providers", payload.get("providers")],
                    ["onnx_validation_pass", payload.get("onnx_validation_pass")],
                    ["failure_stage", payload.get("failure_stage")],
                    ["exact_blocker", payload.get("exact_blocker")],
                ],
            ),
            "",
            markdown_table(["output", "shape", "pass", "max_abs", "p95_abs", "mean_abs"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--providers", nargs="+", default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = validate(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload.get("onnx_validation_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
