"""Validate every BatchTam memory-attention bucket TensorRT engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from .validate_trt_component import validate


def validate_buckets(args: argparse.Namespace) -> dict[str, Any]:
    fixtures_root = Path(args.fixtures_root)
    engine_dir = Path(args.engine_dir)
    buckets = {}
    for fixture_dir in sorted(path for path in fixtures_root.iterdir() if path.is_dir()):
        name = fixture_dir.name
        payload = validate(
            SimpleNamespace(
                component="memory_attention",
                engine_path=str(engine_dir / f"{name}.engine"),
                fixtures_dir=str(fixture_dir),
                device=args.device,
            )
        )
        payload["bucket_dir_name"] = name
        buckets[name] = payload
    return {
        "component": "memory_attention",
        "fixtures_root": str(fixtures_root),
        "engine_dir": str(engine_dir),
        "buckets": buckets,
        "all_buckets_pass": all(item.get("trt_validation_pass") for item in buckets.values()),
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        [
            name,
            item.get("trt_validation_pass"),
            item.get("failure_stage"),
            item.get("exact_blocker"),
        ]
        for name, item in payload["buckets"].items()
    ]
    return "\n".join(
        [
            "# BatchTam Memory-Attention Bucket TensorRT Validation",
            "",
            f"- all_buckets_pass: `{payload['all_buckets_pass']}`",
            "",
            markdown_table(["bucket", "pass", "failure_stage", "blocker"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--fixtures-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = validate_buckets(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload["all_buckets_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
