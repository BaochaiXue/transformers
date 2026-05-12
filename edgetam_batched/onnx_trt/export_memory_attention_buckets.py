"""Export static ONNX files for each observed memory-attention shape bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from .export_component_onnx import export_component


def export_buckets(args: argparse.Namespace) -> dict[str, Any]:
    sequence = json.loads(Path(args.shape_sequence_json).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for _shape_key, bucket in sequence.get("shape_buckets", {}).items():
        name = bucket["bucket_dir_name"]
        fixture_dir = Path(args.fixtures_root) / name
        onnx_name = f"memory_attention_{name}.onnx"
        payload = export_component(
            SimpleNamespace(
                component="memory_attention",
                fixtures_dir=str(fixture_dir),
                out_dir=str(out_dir),
                onnx_name=onnx_name,
                opset=args.opset,
                dynamo=args.dynamo,
                allow_legacy_retry=args.allow_legacy_retry,
                batch_size=args.batch_size,
                object_count=args.object_count,
                precision_mode=args.precision_mode,
                dtype=args.dtype,
                device=args.device,
                model_id=args.model_id,
            )
        )
        payload["bucket_dir_name"] = name
        buckets[name] = payload
    return {
        "component": "memory_attention",
        "shape_sequence_json": args.shape_sequence_json,
        "fixtures_root": args.fixtures_root,
        "out_dir": str(out_dir),
        "buckets": buckets,
        "all_buckets_pass": all(item.get("onnx_export_pass") for item in buckets.values()),
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        [
            name,
            item.get("onnx_export_pass"),
            item.get("onnx_path"),
            item.get("failure_stage"),
            item.get("exact_blocker"),
        ]
        for name, item in payload["buckets"].items()
    ]
    return "\n".join(
        [
            "# BatchTam Memory-Attention Bucket ONNX Export",
            "",
            f"- all_buckets_pass: `{payload['all_buckets_pass']}`",
            "",
            markdown_table(["bucket", "pass", "onnx", "failure_stage", "blocker"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-sequence-json", required=True)
    parser.add_argument("--fixtures-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--allow-legacy-retry", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--precision-mode", default="memory_path_fp32")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--output-md", default="docs/generated/batchtam_memory_attention_bucket_onnx_export.md")
    parser.add_argument("--output-json", default="docs/generated/batchtam_memory_attention_bucket_onnx_export.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = export_buckets(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload["all_buckets_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
