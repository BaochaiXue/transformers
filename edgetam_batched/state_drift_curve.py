"""Measure closed-loop state drift between HF public and full-batched EdgeTAM."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .camera_order import mask_iou
from .compare_multisession import _resize_mask_like
from .find_first_bad_frame import _clone_output_fields, _compare_output_fields, _session_output
from .precision_policy import PRECISION_POLICY_NAMES, reference_dtype_for_precision_mode
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames
from .stats import summarize


DRIFT_FIELDS = ("pred_masks", "maskmem_features", "maskmem_pos_enc", "object_pointer", "object_score_logits")


def run_state_drift_curve(
    *,
    rgb_replay: str,
    backend: str,
    object_count: int,
    object_prompt: str,
    controller_prompt: str,
    frames: int,
    dtype: str,
    device: str,
    precision_mode: str = "all_bf16",
    compile_mode: str = "none",
    graph_output_policy: str = "ring_buffer",
) -> dict[str, Any]:
    reference_dtype = reference_dtype_for_precision_mode(dtype, precision_mode)
    replay_frames = load_replay_frames(rgb_replay, frames)
    config = ReferenceRuntimeConfig(
        dtype=reference_dtype,
        device=device,
        object_prompt=object_prompt,
        controller_prompt=controller_prompt,
        object_count=object_count,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    reference_runtime.init_sessions(replay_frames[0])

    reference_rows = []
    for step_idx, frame in enumerate(replay_frames):
        masks, _logits, _scores, _timings = reference_runtime.step_public(frame)
        session_outputs = []
        for session in reference_runtime.sessions:
            output, bucket = _session_output(session, step_idx)
            session_outputs.append({"bucket": bucket, "output": _clone_output_fields(output)})
        reference_rows.append({"masks": masks, "session_outputs": session_outputs})

    reference_runtime.init_sessions(replay_frames[0])
    runtime = BatchedEdgeTamMultiSessionRuntime(
        reference_runtime.model,
        reference_runtime.processor,
        backend=backend,
        batch_size=len(replay_frames[0].images),
        object_count=object_count,
        dtype=reference_runtime.dtype,
        device=device,
        compile_mode=compile_mode,
        graph_output_policy=graph_output_policy,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
        precision_mode=precision_mode,
    )
    runtime.init_from_reference_sessions(reference_runtime.sessions)
    runtime.prepare_compile(reference_runtime.torch)

    rows = []
    started = time.perf_counter()
    for step_idx, frame in enumerate(replay_frames):
        result = runtime.step(frame.images, step_idx)
        for cam_idx, cand_masks in enumerate(result["masks_b3"]):
            ref_mask = reference_rows[step_idx]["masks"][cam_idx][0]
            cand_mask = _resize_mask_like(cand_masks[0], ref_mask)
            iou = mask_iou(ref_mask, cand_mask)
            cand_output, cand_bucket = _session_output(runtime.sessions[cam_idx], step_idx)
            field_diffs = _compare_output_fields(
                reference_rows[step_idx]["session_outputs"][cam_idx]["output"],
                _clone_output_fields(cand_output),
            )
            row = {
                "frame_idx": step_idx,
                "camera": f"cam{cam_idx}",
                "mask_iou": float(iou),
                "reference_bucket": reference_rows[step_idx]["session_outputs"][cam_idx]["bucket"],
                "candidate_bucket": cand_bucket,
            }
            for field in DRIFT_FIELDS:
                diff = field_diffs.get(field) or {}
                row[f"{field}_max_abs_diff"] = diff.get("max_abs_diff")
                row[f"{field}_mean_abs_diff"] = diff.get("mean_abs_diff")
                row[f"{field}_p95_abs_diff"] = diff.get("p95_abs_diff")
            rows.append(row)

    summary = summarize_state_drift(rows)
    return {
        "backend": backend,
        "dtype": dtype,
        "effective_reference_dtype": reference_dtype,
        "precision_mode": precision_mode,
        "compile_mode": compile_mode,
        "rgb_replay": str(rgb_replay),
        "object_count": object_count,
        "object_prompt": object_prompt,
        "frame_count": len(replay_frames),
        "rows": rows,
        "summary": summary,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "backend_contract": runtime.contract.to_json() if runtime.contract is not None else None,
    }


def summarize_state_drift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [float(row["mask_iou"]) for row in rows]
    summary = {"mask_iou": summarize(ious)}
    for field in DRIFT_FIELDS:
        p95_values = [
            float(row[f"{field}_p95_abs_diff"])
            for row in rows
            if row.get(f"{field}_p95_abs_diff") is not None
        ]
        max_values = [
            float(row[f"{field}_max_abs_diff"])
            for row in rows
            if row.get(f"{field}_max_abs_diff") is not None
        ]
        summary[field] = {
            "p95_abs_diff": summarize(p95_values),
            "max_abs_diff": summarize(max_values),
            "first_frame_p95_gt_1e_3": first_threshold_crossing(rows, f"{field}_p95_abs_diff", 1e-3),
            "first_frame_p95_gt_1e_1": first_threshold_crossing(rows, f"{field}_p95_abs_diff", 1e-1),
            "first_frame_p95_gt_1": first_threshold_crossing(rows, f"{field}_p95_abs_diff", 1.0),
        }
    summary["first_iou_lt_0_90"] = first_iou_below(rows, 0.90)
    summary["first_iou_lt_0_98"] = first_iou_below(rows, 0.98)
    summary["first_tensor_drift"] = first_tensor_drift(rows)
    return summary


def first_threshold_crossing(rows: list[dict[str, Any]], key: str, threshold: float) -> dict[str, Any] | None:
    for row in rows:
        value = row.get(key)
        if value is not None and float(value) > float(threshold):
            return {
                "frame_idx": row["frame_idx"],
                "camera": row["camera"],
                "value": float(value),
                "threshold": float(threshold),
            }
    return None


def first_iou_below(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    for row in rows:
        if float(row["mask_iou"]) < float(threshold):
            return {
                "frame_idx": row["frame_idx"],
                "camera": row["camera"],
                "value": float(row["mask_iou"]),
                "threshold": float(threshold),
            }
    return None


def first_tensor_drift(rows: list[dict[str, Any]], threshold: float = 1e-3) -> dict[str, Any] | None:
    for row in rows:
        for field in DRIFT_FIELDS:
            value = row.get(f"{field}_p95_abs_diff")
            if value is not None and float(value) > float(threshold):
                return {
                    "frame_idx": row["frame_idx"],
                    "camera": row["camera"],
                    "field": field,
                    "p95_abs_diff": float(value),
                    "threshold": float(threshold),
                }
    return None


def render_state_drift_curve(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = []
    for field in DRIFT_FIELDS:
        item = summary[field]
        rows.append(
            [
                field,
                _fmt((item["p95_abs_diff"] or {}).get("p50")),
                _fmt((item["p95_abs_diff"] or {}).get("p90")),
                item.get("first_frame_p95_gt_1e_3"),
                item.get("first_frame_p95_gt_1e_1"),
                item.get("first_frame_p95_gt_1"),
            ]
        )
    return "\n".join(
        [
            "# Full Batched EdgeTAM State Drift Curve",
            "",
            f"- backend: `{payload.get('backend')}`",
            f"- dtype: `{payload.get('dtype')}`",
            f"- replay: `{payload.get('rgb_replay')}`",
            f"- global_iou_avg: `{_fmt(summary['mask_iou'].get('avg'))}`",
            f"- global_iou_p50: `{_fmt(summary['mask_iou'].get('p50'))}`",
            f"- first_iou_lt_0.90: `{summary.get('first_iou_lt_0_90')}`",
            f"- first_tensor_drift: `{summary.get('first_tensor_drift')}`",
            "",
            markdown_table(
                ["field", "p95_diff_p50", "p95_diff_p90", "first >1e-3", "first >0.1", "first >1.0"],
                rows,
            ),
        ]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--backend", default="hf_batched_multisession")
    parser.add_argument("--reference-source", choices=("hf-public", "hf-public-seq"), default="hf-public-seq")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="hand")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--precision-mode", choices=PRECISION_POLICY_NAMES, default="all_bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("state_drift_curve currently supports only HF public reference")
    payload = run_state_drift_curve(
        rgb_replay=args.rgb_replay,
        backend=args.backend,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        frames=args.frames,
        dtype=args.dtype,
        precision_mode=args.precision_mode,
        device=args.device,
        compile_mode=args.compile_mode,
        graph_output_policy=args.graph_output_policy,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_state_drift_curve(payload))
    if args.debug:
        print(payload)
    return 0 if payload["summary"].get("first_iou_lt_0_90") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
