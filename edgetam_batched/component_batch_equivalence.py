"""Summarize whether traced EdgeTAM components have a proven batch equivalence path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .report_utils import markdown_table, write_json, write_markdown


COMPONENT_REQUIREMENTS = {
    "vision_encoder": "batch path exists through model.get_image_features(pixel_values[B,...])",
    "memory_attention": "requires tensorized memory/object-pointer inputs across camera sessions",
    "mask_decoder": "requires tensorized prompted/empty-tracking decoder inputs across camera sessions",
    "memory_encoder": "EdgeTamVideoModel._batch_encode_memories is NotImplemented in modular source",
}


def analyze_component(trace: dict[str, Any], component: str) -> dict[str, Any]:
    records = [r for r in trace.get("records", []) if r.get("component") == component]
    if component == "vision_encoder":
        passed = bool(records)
        blockers = [] if passed else ["no vision_encoder calls found in trace"]
    elif component == "memory_encoder":
        passed = False
        blockers = ["EdgeTamVideoModel._batch_encode_memories is NotImplemented; full state update cannot be batched yet"]
    else:
        passed = False
        blockers = [COMPONENT_REQUIREMENTS[component], "raw tensor equivalence is not proven by shape-only trace"]
    return {
        "component": component,
        "records": len(records),
        "equivalence_pass": passed,
        "requirement": COMPONENT_REQUIREMENTS.get(component, ""),
        "blockers": blockers,
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        [row["component"], row["records"], row["equivalence_pass"], "; ".join(row["blockers"])]
        for row in payload["components"]
    ]
    return "\n".join(
        [
            "# EdgeTAM Component Batch Equivalence",
            "",
            f"- trace: `{payload['trace_json']}`",
            f"- all_components_pass: `{payload['all_components_pass']}`",
            "",
            markdown_table(["component", "trace_records", "pass", "blockers"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--component", default=None)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    trace = json.loads(Path(args.trace_json).read_text())
    components = [args.component] if args.component else list(COMPONENT_REQUIREMENTS)
    rows = [analyze_component(trace, component) for component in components]
    payload = {
        "trace_json": args.trace_json,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "components": rows,
        "all_components_pass": all(row["equivalence_pass"] for row in rows),
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload["all_components_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
