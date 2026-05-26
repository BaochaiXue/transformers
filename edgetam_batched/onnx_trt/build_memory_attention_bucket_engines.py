"""Build TensorRT engines for memory-attention static shape buckets."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from .build_trt_engine import build


def build_bucket_engines(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_root = Path(args.fixtures_root)
    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for fixture_dir in sorted(path for path in fixtures_root.iterdir() if path.is_dir()):
        name = fixture_dir.name
        onnx_path = onnx_dir / f"memory_attention_{name}.onnx"
        payload = build(
            SimpleNamespace(
                component="memory_attention",
                onnx_path=str(onnx_path),
                engine_path=str(engine_dir / f"{name}.engine"),
                io_spec=str(fixture_dir / "io_spec.json"),
                precision_policy=args.precision_policy,
                builder_optimization_level=args.builder_optimization_level,
                timing_cache=args.timing_cache,
                no_tf32_for_memory_path=args.no_tf32,
                export_layer_info=None,
                export_profile=None,
                skip_inference=True,
                shape_profiles=args.shape_profiles,
                timeout_s=args.timeout_s,
            )
        )
        payload["bucket_dir_name"] = name
        buckets[name] = payload
    return {
        "component": "memory_attention",
        "onnx_dir": str(onnx_dir),
        "fixtures_root": str(fixtures_root),
        "engine_dir": str(engine_dir),
        "buckets": buckets,
        "all_buckets_pass": all(item.get("trt_build_pass") for item in buckets.values()),
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        [
            name,
            item.get("trt_build_pass"),
            item.get("engine_path"),
            item.get("returncode"),
            item.get("failure_stage"),
            item.get("exact_blocker"),
        ]
        for name, item in payload["buckets"].items()
    ]
    return "\n".join(
        [
            "# BatchTam Memory-Attention Bucket TensorRT Build",
            "",
            f"- all_buckets_pass: `{payload['all_buckets_pass']}`",
            "",
            markdown_table(["bucket", "pass", "engine", "returncode", "failure_stage", "blocker"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--fixtures-root", required=True)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    parser.add_argument("--timing-cache", default=None)
    parser.add_argument("--precision-policy", default="memory_path_fp32")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--shape-profiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass min/opt/max shape profiles. Static bucket ONNX exports already encode exact shapes, "
            "so this is disabled by default."
        ),
    )
    parser.add_argument("--timeout-s", type=int, default=None)
    parser.add_argument("--output-md", default="docs/generated/batchtam_memory_attention_bucket_trt_build.md")
    parser.add_argument("--output-json", default="docs/generated/batchtam_memory_attention_bucket_trt_build.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = build_bucket_engines(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload["all_buckets_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
