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
    decision = payload["decision"]
    bucket_rows = [
        ["shape_strategy", decision.get("memory_attention_shape_strategy")],
        ["bucket_count", decision.get("memory_attention_bucket_count")],
        ["buckets_exported", decision.get("memory_attention_buckets_exported")],
        ["buckets_built", decision.get("memory_attention_buckets_built")],
        ["buckets_validated", decision.get("memory_attention_buckets_validated")],
        ["memory_attention_bucketed_closed_loop_pass", decision.get("memory_attention_bucketed_closed_loop_pass")],
        ["mask_decoder_trt_closed_loop_pass", decision.get("mask_decoder_trt_closed_loop_pass")],
        ["memory_encoder_trt_closed_loop_pass", decision.get("memory_encoder_trt_closed_loop_pass")],
        ["memory_path_all_trt_closed_loop_pass", decision.get("memory_path_all_trt_closed_loop_pass")],
    ]
    return "\n".join(
        [
            "# BatchTam ONNX/TRT Component Export Report",
            "",
            f"- trt_components_usable: `{decision['trt_components_usable']}`",
            f"- batchtam_component_engines_usable: `{decision['batchtam_component_engines_usable']}`",
            f"- batchtam_closed_loop_usable: `{decision['batchtam_closed_loop_usable']}`",
            f"- recommended_trt_scope: `{decision['recommended_trt_scope']}`",
            f"- demo22_trt_integration_allowed: `{decision['demo22_trt_integration_allowed']}`",
            f"- failure_stage: `{decision.get('failure_stage')}`",
            f"- exact_blocker: `{decision.get('exact_blocker')}`",
            "",
            "## Recommended BatchTam Runtime",
            "",
            "The recommended scope is `memory_path_all`: memory_attention, mask_decoder, and memory_encoder run as TRT components. Vision encoder is not exported to TRT in this phase and remains on the PyTorch/compiled vision path.",
            "",
            markdown_table(["field", "value"], bucket_rows),
            "",
            "## Legacy Single-Component Diagnostics",
            "",
            "These rows are retained for diagnostics. They do not define the Demo 2.2 gate when `recommended_trt_scope=memory_path_all`; the memory_attention gate is the bucketed static engine set above.",
            "",
            markdown_table(
                ["component", "onnx_export", "onnx_validate", "trt_build", "trt_validate", "failure_stage", "blocker"],
                rows,
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-json", "--component-report", nargs="*", default=[])
    parser.add_argument("--closed-loop-json", "--closed-loop-report", nargs="*", default=[])
    parser.add_argument("--profile-json", nargs="*", default=[])
    parser.add_argument("--memory-attention-shape-json", default=None)
    parser.add_argument("--memory-attention-bucket-validation-json", default=None)
    parser.add_argument("--memory-attention-bucket-export-json", default=None)
    parser.add_argument("--memory-attention-bucket-build-json", default=None)
    parser.add_argument("--recommended-scope", default="memory_path_all")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    component_reports = load_reports(args.component_json)
    shape_report = load_optional_report(
        args.memory_attention_shape_json,
        default_path="docs/generated/batchtam_memory_attention_shape_sequence.json",
    )
    bucket_validation = load_optional_report(
        args.memory_attention_bucket_validation_json,
        default_path="docs/generated/validate_trt_memory_attention_buckets.json",
    )
    bucket_export = load_optional_report(
        args.memory_attention_bucket_export_json,
        default_path="docs/generated/batchtam_memory_attention_bucket_onnx_export.json",
    )
    bucket_build = load_optional_report(
        args.memory_attention_bucket_build_json,
        default_path="docs/generated/batchtam_memory_attention_bucket_trt_build.json",
    )
    components = merge_component_reports(component_reports)
    required = MEMORY_PATH_COMPONENTS if args.recommended_scope == "memory_path_all" else COMPONENTS
    usable = trt_components_usable({"components": components}, scope=args.recommended_scope)
    missing = [name for name in required if not components[name].get("trt_validation_pass")]
    closed_loop = load_reports(args.closed_loop_json)
    closed_loop_for_scope = [item for item in closed_loop if item.get("trt_scope") == args.recommended_scope]
    closed_loop_pass = any(_closed_loop_pass(item) for item in closed_loop_for_scope)
    memory_attention_bucketed_closed_loop_pass = any(
        _closed_loop_pass(item)
        and item.get("trt_scope") == "memory_attention"
        and bool(item.get("trt_memory_attention_bucket_dir"))
        for item in closed_loop
    )
    mask_decoder_trt_closed_loop_pass = _component_closed_loop_pass(closed_loop, "trt_mask_decoder")
    memory_encoder_trt_closed_loop_pass = _component_closed_loop_pass(closed_loop, "trt_memory_encoder")
    memory_path_all_trt_closed_loop_pass = any(
        _closed_loop_pass(item) and item.get("trt_scope") == "memory_path_all" for item in closed_loop
    )
    bucket_count = _bucket_count(shape_report, bucket_validation)
    buckets_exported = _count_bucket_pass(bucket_export, "onnx_export_pass")
    buckets_built = _count_bucket_pass(bucket_build, "trt_build_pass")
    buckets_validated = _count_bucket_pass(bucket_validation, "trt_validation_pass")
    scope_blocker = _first_scope_blocker(closed_loop_for_scope)
    decision_usable = usable and closed_loop_pass
    integration_allowed = decision_usable and memory_path_all_trt_closed_loop_pass
    if args.recommended_scope == "memory_path_all":
        integration_allowed = bool(
            integration_allowed
            and bucket_count
            and buckets_exported == bucket_count
            and buckets_built == bucket_count
            and buckets_validated == bucket_count
            and memory_attention_bucketed_closed_loop_pass
            and mask_decoder_trt_closed_loop_pass
            and memory_encoder_trt_closed_loop_pass
        )
    payload = {
        "name": "BatchTam ONNX/TRT",
        "components": components,
        "closed_loop": closed_loop,
        "profiles": load_reports(args.profile_json),
        "decision": {
            "trt_components_usable": decision_usable,
            "batchtam_component_engines_usable": usable,
            "batchtam_closed_loop_usable": closed_loop_pass,
            "memory_attention_shape_strategy": "bucketed_static_engines"
            if bucket_validation
            else "single_static_engine",
            "memory_attention_bucket_count": bucket_count,
            "memory_attention_buckets_exported": buckets_exported,
            "memory_attention_buckets_built": buckets_built,
            "memory_attention_buckets_validated": buckets_validated,
            "memory_attention_bucketed_closed_loop_pass": memory_attention_bucketed_closed_loop_pass,
            "mask_decoder_trt_closed_loop_pass": mask_decoder_trt_closed_loop_pass,
            "memory_encoder_trt_closed_loop_pass": memory_encoder_trt_closed_loop_pass,
            "memory_path_all_trt_closed_loop_pass": memory_path_all_trt_closed_loop_pass,
            "memory_attention_observed_shape_keys": (shape_report or {}).get("observed_shape_keys"),
            "component_validation_usable": usable,
            "closed_loop_strict_pass": closed_loop_pass,
            "recommended_trt_scope": args.recommended_scope,
            "demo22_integration_allowed": integration_allowed,
            "demo22_trt_integration_allowed": integration_allowed,
            "failure_stage": None if decision_usable else ("closed_loop_correctness" if usable else "trt_component_validation"),
            "exact_blocker": None if decision_usable else (
                scope_blocker or "closed-loop strict correctness has not passed"
                if usable
                else f"components missing TRT validation: {missing}"
            ),
        },
    }
    payload.update(
        {
            "batchtam_component_engines_usable": payload["decision"]["batchtam_component_engines_usable"],
            "batchtam_closed_loop_usable": payload["decision"]["batchtam_closed_loop_usable"],
            "demo22_trt_integration_allowed": payload["decision"]["demo22_trt_integration_allowed"],
            "demo22_integration_allowed": payload["decision"]["demo22_trt_integration_allowed"],
            "recommended_trt_scope": payload["decision"]["recommended_trt_scope"],
            "memory_attention_shape_strategy": payload["decision"]["memory_attention_shape_strategy"],
            "memory_attention_bucket_count": bucket_count,
            "memory_attention_buckets_exported": buckets_exported,
            "memory_attention_buckets_built": buckets_built,
            "memory_attention_buckets_validated": buckets_validated,
            "memory_attention_bucketed_closed_loop_pass": memory_attention_bucketed_closed_loop_pass,
            "mask_decoder_trt_closed_loop_pass": mask_decoder_trt_closed_loop_pass,
            "memory_encoder_trt_closed_loop_pass": memory_encoder_trt_closed_loop_pass,
            "memory_path_all_trt_closed_loop_pass": memory_path_all_trt_closed_loop_pass,
        }
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    return 0 if integration_allowed else 2


def load_optional_report(path: str | None, *, default_path: str | None = None) -> dict[str, Any] | None:
    candidate = path or default_path
    if not candidate:
        return None
    report_path = Path(candidate)
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _closed_loop_pass(payload: dict[str, Any]) -> bool:
    if payload.get("strict_correctness_pass") is True:
        return True
    if payload.get("correctness_pass") is True:
        return True
    metrics = payload.get("metrics") or {}
    correctness = metrics.get("correctness_pass")
    if isinstance(correctness, dict):
        return bool(correctness.get("strict"))
    if correctness is True:
        return True
    return bool(metrics.get("strict_correctness_pass"))


def _first_scope_blocker(reports: list[dict[str, Any]]) -> str | None:
    for payload in reports:
        if payload.get("exact_blocker"):
            return str(payload["exact_blocker"])
        metrics = payload.get("metrics") or {}
        blockers = metrics.get("blockers") or []
        if blockers:
            return str(blockers[0])
    return None


def _component_closed_loop_pass(reports: list[dict[str, Any]], contract_field: str) -> bool:
    for payload in reports:
        if not _closed_loop_pass(payload):
            continue
        metrics = payload.get("metrics") or {}
        contract = metrics.get("backend_contract") or payload.get("backend_contract") or {}
        if contract.get(contract_field) is True:
            return True
    return False


def _bucket_count(shape_report: dict[str, Any] | None, bucket_validation: dict[str, Any] | None) -> int:
    if shape_report and isinstance(shape_report.get("observed_shape_keys"), list):
        return len(shape_report["observed_shape_keys"])
    if bucket_validation and isinstance(bucket_validation.get("buckets"), dict):
        return len(bucket_validation["buckets"])
    return 0


def _count_bucket_pass(report: dict[str, Any] | None, field: str) -> int:
    if not report:
        return 0
    buckets = report.get("buckets")
    if not isinstance(buckets, dict):
        return 0
    return sum(1 for item in buckets.values() if isinstance(item, dict) and item.get(field) is True)


if __name__ == "__main__":
    raise SystemExit(main())
