"""Validate BatchTam TensorRT component outputs against eager fixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from . import COMPONENTS
from .trt_component_runner import TrtComponentRunner
from .trt_engine_utils import load_io_spec
from .validate_onnx_component import diff_stats, tensor_to_numpy


def validate(args: argparse.Namespace) -> dict:
    import torch

    fixture_dir = Path(args.fixtures_dir) / args.component
    engine_path = Path(args.engine_path)
    if not engine_path.exists():
        return failure(args, "trt_component_validation", f"missing engine file: {engine_path}")
    if not torch.cuda.is_available():
        return failure(args, "trt_component_validation", "torch CUDA is unavailable")
    sample_inputs = torch.load(fixture_dir / "sample_inputs.pt", map_location="cpu", weights_only=False)
    sample_outputs = torch.load(fixture_dir / "sample_outputs_eager.pt", map_location="cpu", weights_only=False)
    io_spec = load_io_spec(fixture_dir / "io_spec.json")
    try:
        runner = TrtComponentRunner(engine_path, io_spec, name=args.component)
        inputs = [tensor.to(device=args.device).contiguous() for tensor in sample_inputs["flat_inputs"]]
        with torch.inference_mode():
            outputs = runner(*inputs)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
    except Exception as exc:  # noqa: BLE001
        return failure(args, "trt_component_validation", repr(exc))

    rows = []
    pass_all = True
    for idx, (spec, expected, actual) in enumerate(zip(io_spec.outputs, sample_outputs["flat_outputs"], outputs, strict=False)):
        stats = diff_stats(tensor_to_numpy(expected), tensor_to_numpy(actual))
        pass_all = pass_all and bool(stats["pass"])
        rows.append({"name": spec.name, "index": idx, "shape": list(actual.shape), **stats})
    return {
        "component": args.component,
        "engine_path": str(engine_path),
        "trt_validation_pass": pass_all,
        "rows": rows,
        "failure_stage": None if pass_all else "trt_component_validation",
        "exact_blocker": None if pass_all else "one or more TRT outputs exceed tolerance",
        "hot_path_zero_copy": True,
    }


def failure(args: argparse.Namespace, stage: str, blocker: str) -> dict:
    return {
        "component": args.component,
        "engine_path": args.engine_path,
        "trt_validation_pass": False,
        "failure_stage": stage,
        "exact_blocker": blocker,
    }


def render(payload: dict) -> str:
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
            f"# BatchTam TensorRT Validation: {payload.get('component')}",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["engine_path", payload.get("engine_path")],
                    ["trt_validation_pass", payload.get("trt_validation_pass")],
                    ["hot_path_zero_copy", payload.get("hot_path_zero_copy")],
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
    parser.add_argument("--engine-path", required=True)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = validate(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload.get("trt_validation_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
