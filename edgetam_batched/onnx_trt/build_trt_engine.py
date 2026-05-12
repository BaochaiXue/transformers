"""Build fixed-shape BatchTam TensorRT engines with trtexec."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from . import COMPONENTS
from .trt_engine_utils import build_trtexec_command, load_io_spec, run_command, trtexec_path


def build(args: argparse.Namespace) -> dict:
    io_spec = load_io_spec(args.io_spec)
    engine_path = Path(args.engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.onnx_path).exists():
        return failure(args, "trt_build", f"missing ONNX file: {args.onnx_path}")
    if trtexec_path() is None:
        return failure(args, "trt_build", "trtexec not found in PATH")
    cmd = build_trtexec_command(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        io_spec=io_spec,
        builder_optimization_level=args.builder_optimization_level,
        timing_cache=args.timing_cache,
        no_tf32=args.no_tf32_for_memory_path,
        export_layer_info=args.export_layer_info,
        export_profile=args.export_profile,
        skip_inference=args.skip_inference,
        include_shape_profiles=args.shape_profiles,
    )
    result = run_command(cmd, timeout=args.timeout_s)
    passed = result["returncode"] == 0 and engine_path.exists()
    return {
        "component": args.component,
        "onnx_path": args.onnx_path,
        "engine_path": args.engine_path,
        "builderOptimizationLevel": args.builder_optimization_level,
        "precision_policy": args.precision_policy,
        "noTF32": bool(args.no_tf32_for_memory_path),
        "timing_cache": args.timing_cache,
        "command": result["command"],
        "returncode": result["returncode"],
        "stdout_tail": (result.get("stdout") or "")[-8000:],
        "trt_build_pass": passed,
        "failure_stage": None if passed else "trt_build",
        "exact_blocker": None if passed else result.get("error") or "trtexec returned nonzero or did not write engine",
    }


def failure(args: argparse.Namespace, stage: str, blocker: str) -> dict:
    return {
        "component": args.component,
        "onnx_path": args.onnx_path,
        "engine_path": args.engine_path,
        "trt_build_pass": False,
        "failure_stage": stage,
        "exact_blocker": blocker,
    }


def render(payload: dict) -> str:
    return "\n".join(
        [
            f"# BatchTam TensorRT Build: {payload.get('component')}",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["onnx_path", payload.get("onnx_path")],
                    ["engine_path", payload.get("engine_path")],
                    ["builderOptimizationLevel", payload.get("builderOptimizationLevel")],
                    ["precision_policy", payload.get("precision_policy")],
                    ["noTF32", payload.get("noTF32")],
                    ["trt_build_pass", payload.get("trt_build_pass")],
                    ["failure_stage", payload.get("failure_stage")],
                    ["exact_blocker", payload.get("exact_blocker")],
                ],
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--engine-path", required=True)
    parser.add_argument("--io-spec", required=True)
    parser.add_argument("--precision-policy", default="memory_path_fp32")
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    parser.add_argument("--timing-cache", default=None)
    parser.add_argument("--no-tf32-for-memory-path", action="store_true")
    parser.add_argument("--export-layer-info", default=None)
    parser.add_argument("--export-profile", default=None)
    parser.add_argument("--skip-inference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--shape-profiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass min/opt/max shape profiles. Leave disabled for static-shape ONNX exports.",
    )
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = build(args)
    if args.output_json:
        write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload.get("trt_build_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
