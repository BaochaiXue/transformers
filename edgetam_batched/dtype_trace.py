"""Trace precision policy, compile order, and persistent state dtypes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .config import COMPILE_SCOPES
from .find_first_bad_frame import _session_output
from .precision_policy import PRECISION_POLICY_NAMES, resolve_precision_policy
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames


TRACE_FIELDS = ("pred_masks", "object_pointer", "maskmem_features", "maskmem_pos_enc", "object_score_logits")


def collect_nested_dtypes(value: Any, path: str = "") -> list[dict[str, Any]]:
    rows = []
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        rows.append({"path": path, "dtype": str(value.dtype), "shape": list(value.shape), "device": str(value.device)})
    elif isinstance(value, dict):
        for key, item in value.items():
            rows.extend(collect_nested_dtypes(item, f"{path}.{key}" if path else str(key)))
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            rows.extend(collect_nested_dtypes(item, f"{path}[{idx}]"))
    return rows


def summarize_dtype_rows(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        out[str(row.get("path") or "")].add(str(row.get("dtype")))
    return {key: sorted(value) for key, value in sorted(out.items())}


def run_dtype_trace(args: argparse.Namespace) -> dict[str, Any]:
    frames = load_replay_frames(args.rgb_replay, args.frames)
    config = ReferenceRuntimeConfig(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        object_count=args.object_count,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    reference_runtime.init_sessions(frames[0])

    runtime = BatchedEdgeTamMultiSessionRuntime(
        reference_runtime.model,
        reference_runtime.processor,
        backend=args.backend,
        batch_size=len(frames[0].images),
        object_count=args.object_count,
        dtype=reference_runtime.dtype,
        device=args.device,
        compile_mode=args.compile_mode,
        compile_scope=args.compile_scope,
        graph_output_policy=args.graph_output_policy,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
        precision_mode=args.precision_mode,
    )
    runtime.init_from_reference_sessions(reference_runtime.sessions)
    runtime.prepare_compile(reference_runtime.torch)

    stored_rows = []
    last_result = None
    for step_idx, frame in enumerate(frames):
        last_result = runtime.step(frame.images, int(frame.frame_idx))
        for cam_idx, session in enumerate(runtime.sessions):
            output, bucket = _session_output(session, int(frame.frame_idx))
            for field in TRACE_FIELDS:
                for row in collect_nested_dtypes(output.get(field), field):
                    row.update({"frame_idx": step_idx, "camera": f"cam{cam_idx}", "bucket": bucket})
                    stored_rows.append(row)

    policy = resolve_precision_policy(args.precision_mode)
    stored_summary = summarize_dtype_rows(stored_rows)
    gates = dtype_gates(policy.name, stored_summary)
    payload = {
        "backend": args.backend,
        "rgb_replay": args.rgb_replay,
        "frames": len(frames),
        "dtype": args.dtype,
        "precision_mode": args.precision_mode,
        "compile_mode": args.compile_mode,
        "compile_scope": args.compile_scope or ("none" if args.compile_mode == "none" else "vision_encoder"),
        "graph_output_policy": args.graph_output_policy,
        "component_dtype_table": dict(runtime.component_dtype_table),
        "compile_order": dict(runtime.compile_order),
        "backend_contract": runtime.contract.to_json() if runtime.contract is not None else None,
        "last_step_backend_contract": (last_result or {}).get("backend_contract") if last_result else None,
        "stored_state_dtype_summary": stored_summary,
        "stored_state_dtype_rows": stored_rows[:200],
        "dtype_gate": gates,
    }
    return payload


def dtype_gates(precision_mode: str, stored_summary: dict[str, list[str]]) -> dict[str, Any]:
    if precision_mode != "memory_path_fp32":
        return {"checked": False, "reason": "gate currently defined for memory_path_fp32"}
    checks = {
        "stored_maskmem_features_fp32": _contains_only(stored_summary, "maskmem_features", "torch.float32"),
        "stored_object_pointer_fp32": _contains_only(stored_summary, "object_pointer", "torch.float32"),
        "stored_mask_logits_fp32": _contains_only(stored_summary, "pred_masks", "torch.float32"),
        "stored_object_score_logits_fp32": _contains_only(stored_summary, "object_score_logits", "torch.float32"),
    }
    checks["pass"] = all(checks.values())
    return checks


def _contains_only(summary: dict[str, list[str]], path_prefix: str, dtype: str) -> bool:
    rows = [dtypes for path, dtypes in summary.items() if path.startswith(path_prefix)]
    return bool(rows) and all(dtypes == [dtype] for dtypes in rows)


def render(payload: dict[str, Any]) -> str:
    component_rows = [[key, value] for key, value in sorted((payload.get("component_dtype_table") or {}).items())]
    stored_rows = [
        [path, ", ".join(dtypes)] for path, dtypes in sorted((payload.get("stored_state_dtype_summary") or {}).items())
    ]
    gate_rows = [[key, value] for key, value in sorted((payload.get("dtype_gate") or {}).items())]
    compile_order = payload.get("compile_order") or {}
    return "\n".join(
        [
            "# Full Batched EdgeTAM Dtype Trace",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["precision_mode", payload.get("precision_mode")],
                    ["compile_mode", payload.get("compile_mode")],
                    ["compile_scope", payload.get("compile_scope")],
                    [
                        "precision_policy_applied_before_compile",
                        compile_order.get("precision_policy_applied_before_compile"),
                    ],
                    [
                        "compiled_after_component_dtype_conversion",
                        compile_order.get("compiled_after_component_dtype_conversion"),
                    ],
                ],
            ),
            "",
            "## Component Parameter Dtypes",
            "",
            markdown_table(["component", "dtype"], component_rows),
            "",
            "## Stored State Dtypes",
            "",
            markdown_table(["path", "dtypes"], stored_rows),
            "",
            "## Gate",
            "",
            markdown_table(["check", "value"], gate_rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--backend", default="hf_batched_multisession")
    parser.add_argument("--reference-source", choices=("hf-public", "hf-public-seq"), default="hf-public-seq")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="hand")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--precision-mode", choices=PRECISION_POLICY_NAMES, default="memory_path_fp32")
    parser.add_argument("--compile-mode", required=True)
    parser.add_argument("--compile-scope", choices=COMPILE_SCOPES, default=None)
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("dtype_trace currently supports only HF public reference")
    payload = run_dtype_trace(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if (payload.get("dtype_gate") or {}).get("pass", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
