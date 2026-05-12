"""Aggregate BatchTam ONNX/TensorRT component reports."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from . import COMPONENTS, MEMORY_PATH_COMPONENTS
from .trt_engine_utils import component_status_table, trt_components_usable


def load_reports(patterns: list[str]) -> list[dict[str, Any]]:
    reports = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                payload = {"load_error": repr(exc)}
            payload["_path"] = path
            reports.append(payload)
    return reports


def merge_component_reports(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    components = {name: {"component": name} for name in COMPONENTS}
    for report in reports:
        name = report.get("component")
        if name not in components:
            continue
        components[name].update(report)
    for name, item in components.items():
        item.setdefault("onnx_export_pass", False)
        item.setdefault("onnx_validation_pass", False)
        item.setdefault("trt_build_pass", False)
        item.setdefault("trt_validation_pass", False)
        if not item.get("failure_stage"):
            for key, stage in (
                ("onnx_export_pass", "onnx_export"),
                ("onnx_validation_pass", "onnx_validation"),
                ("trt_build_pass", "trt_build"),
                ("trt_validation_pass", "trt_component_validation"),
            ):
                if not item.get(key):
                    item["failure_stage"] = stage
                    item.setdefault("exact_blocker", f"{stage} has not passed")
                    break
    return components


def render(payload: dict[str, Any]) -> str:
    rows = component_status_table(payload["components"].values())
    return "\n".join(
        [
            "# BatchTam ONNX/TRT Component Export Report",
            "",
            f"- trt_components_usable: `{payload['decision']['trt_components_usable']}`",
            f"- recommended_trt_scope: `{payload['decision']['recommended_trt_scope']}`",
            f"- demo22_integration_allowed: `{payload['decision']['demo22_integration_allowed']}`",
            f"- failure_stage: `{payload['decision'].get('failure_stage')}`",
            f"- exact_blocker: `{payload['decision'].get('exact_blocker')}`",
            "",
            markdown_table(
                ["component", "onnx_export", "onnx_validate", "trt_build", "trt_validate", "failure_stage", "blocker"],
                rows,
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-json", nargs="*", default=[])
    parser.add_argument("--closed-loop-json", nargs="*", default=[])
    parser.add_argument("--profile-json", nargs="*", default=[])
    parser.add_argument("--recommended-scope", default="memory_path_all")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    component_reports = load_reports(args.component_json)
    components = merge_component_reports(component_reports)
    required = MEMORY_PATH_COMPONENTS if args.recommended_scope == "memory_path_all" else COMPONENTS
    usable = trt_components_usable({"components": components}, scope=args.recommended_scope)
    missing = [name for name in required if not components[name].get("trt_validation_pass")]
    closed_loop = load_reports(args.closed_loop_json)
    closed_loop_pass = any(item.get("strict_correctness_pass") or item.get("correctness_pass") for item in closed_loop)
    decision_usable = usable and closed_loop_pass
    payload = {
        "name": "BatchTam ONNX/TRT",
        "components": components,
        "closed_loop": closed_loop,
        "profiles": load_reports(args.profile_json),
        "decision": {
            "trt_components_usable": decision_usable,
            "component_validation_usable": usable,
            "closed_loop_strict_pass": closed_loop_pass,
            "recommended_trt_scope": args.recommended_scope,
            "demo22_integration_allowed": decision_usable,
            "failure_stage": None if decision_usable else ("closed_loop_correctness" if usable else "trt_component_validation"),
            "exact_blocker": None if decision_usable else (
                "closed-loop strict correctness has not passed" if usable else f"components missing TRT validation: {missing}"
            ),
        },
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    return 0 if decision_usable else 2


if __name__ == "__main__":
    raise SystemExit(main())
