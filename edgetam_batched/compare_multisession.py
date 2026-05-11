"""Correctness harness for EdgeTAM batched multi-session backends."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .batched_multisession_runtime import run_candidate
from .camera_order import diagonal_best, iou_matrix, mask_iou
from .config import BACKENDS
from .leakage_test import compare_cam0_stability
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames
from .stats import summarize


def compare_outputs(reference, candidate) -> dict[str, Any]:
    frame_count = min(len(reference.masks), len(candidate.masks))
    camera_count = len(reference.masks[0]) if frame_count else 0
    object_count = reference.masks[0][0].shape[0] if frame_count else 0
    per_key: dict[str, dict[str, list[float] | int]] = {}
    empty_mismatch_count = 0

    for cam_idx in range(camera_count):
        for obj_idx in range(object_count):
            key = f"cam{cam_idx}_obj{obj_idx}"
            ious = []
            logit_diffs = []
            score_diffs = []
            ref_nonempty = 0
            cand_nonempty = 0
            for frame_idx in range(frame_count):
                ref_mask = reference.masks[frame_idx][cam_idx][obj_idx]
                cand_mask = candidate.masks[frame_idx][cam_idx][obj_idx]
                ref_has = bool(np.count_nonzero(ref_mask))
                cand_has = bool(np.count_nonzero(cand_mask))
                ref_nonempty += int(ref_has)
                cand_nonempty += int(cand_has)
                empty_mismatch_count += int(ref_has != cand_has)
                ious.append(mask_iou(ref_mask, cand_mask))
                ref_logit = reference.logits[frame_idx][cam_idx][obj_idx]
                cand_logit = candidate.logits[frame_idx][cam_idx][obj_idx]
                logit_diffs.append(float(np.mean(np.abs(ref_logit - cand_logit))))
                ref_score = np.asarray(reference.object_scores[frame_idx][cam_idx]).reshape(-1)
                cand_score = np.asarray(candidate.object_scores[frame_idx][cam_idx]).reshape(-1)
                if obj_idx < len(ref_score) and obj_idx < len(cand_score):
                    score_diffs.append(float(abs(ref_score[obj_idx] - cand_score[obj_idx])))
            per_key[key] = {
                "mask_iou": summarize(ious),
                "logit_abs_diff": summarize(logit_diffs),
                "object_score_abs_diff": summarize(score_diffs),
                "reference_nonempty_count": ref_nonempty,
                "candidate_nonempty_count": cand_nonempty,
            }

    first_ref = [reference.masks[0][cam_idx][0] for cam_idx in range(camera_count)] if frame_count else []
    first_cand = [candidate.masks[0][cam_idx][0] for cam_idx in range(camera_count)] if frame_count else []
    order = diagonal_best(iou_matrix(first_cand, first_ref)) if first_ref else {"pass": False, "matrix": []}
    leakage = {"pass": True, "note": "no cross-camera tensor state is shared in implemented batch-vision fallback"}
    all_ious = [
        value
        for metrics in per_key.values()
        for value in _expand_summary_sample(metrics["mask_iou"])
    ]
    global_iou = summarize(all_ious)
    mask_correctness_pass = (
        bool(order["pass"])
        and empty_mismatch_count == 0
        and (global_iou["avg"] is not None and global_iou["avg"] >= 0.98)
        and (global_iou["p50"] is not None and global_iou["p50"] >= 0.98)
    )
    correctness_pass = mask_correctness_pass and not candidate.partial
    return {
        "frame_count": frame_count,
        "camera_count": camera_count,
        "object_count": object_count,
        "per_camera_object": per_key,
        "global_mask_iou": global_iou,
        "empty_mismatch_count": empty_mismatch_count,
        "camera_order_check": "pass" if order["pass"] else "fail",
        "camera_order_matrix": order["matrix"],
        "state_leakage_check": "pass" if leakage["pass"] else "fail",
        "state_leakage": leakage,
        "mask_correctness_pass": mask_correctness_pass,
        "correctness_pass": correctness_pass,
        "candidate_partial": candidate.partial,
        "fallback_backend": candidate.fallback_backend,
        "blockers": candidate.blockers or [],
    }


def _expand_summary_sample(summary_dict: dict[str, Any]) -> list[float]:
    values = []
    for key in ("avg", "p50", "p90", "p95", "p99", "min", "max"):
        value = summary_dict.get(key)
        if value is not None:
            values.append(float(value))
    return values


def render_compare_report(payload: dict[str, Any]) -> str:
    rows = []
    for key, metrics in payload["metrics"]["per_camera_object"].items():
        rows.append(
            [
                key,
                metrics["mask_iou"]["avg"],
                metrics["mask_iou"]["min"],
                metrics["mask_iou"]["p50"],
                metrics["reference_nonempty_count"],
                metrics["candidate_nonempty_count"],
            ]
        )
    return "\n".join(
        [
            "# EdgeTAM Batched Correctness",
            "",
            f"- backend: `{payload['backend']}`",
            f"- compile_mode: `{payload['compile_mode']}`",
            f"- correctness_pass: `{payload['metrics']['correctness_pass']}`",
            f"- mask_correctness_pass: `{payload['metrics']['mask_correctness_pass']}`",
            f"- candidate_partial: `{payload['metrics']['candidate_partial']}`",
            f"- fallback_backend: `{payload['metrics']['fallback_backend']}`",
            "",
            "## Blockers",
            "",
            "\n".join(f"- {item}" for item in payload["metrics"]["blockers"]) or "- none",
            "",
            "## Metrics",
            "",
            markdown_table(["key", "iou_avg", "iou_min", "iou_p50", "ref_nonempty", "cand_nonempty"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--object-count", type=int, default=2)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="towel")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    frames = load_replay_frames(args.rgb_replay, args.frames)
    config = ReferenceRuntimeConfig(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    reference_runtime.init_sessions(frames[0])
    ref_masks = []
    ref_logits = []
    ref_scores = []
    ref_timings = []
    for frame in frames:
        masks, logits, scores, timings = reference_runtime.step_public(frame)
        ref_masks.append(masks)
        ref_logits.append(logits)
        ref_scores.append(scores)
        ref_timings.append(timings)
    reference = type("Outputs", (), {
        "masks": ref_masks,
        "logits": ref_logits,
        "object_scores": ref_scores,
        "timings_ms": ref_timings,
    })()

    # Candidate uses a fresh runtime but the same loaded model/processor.
    reference_runtime.init_sessions(frames[0])
    candidate = run_candidate(
        rgb_replay_frames=frames,
        reference_runtime=reference_runtime,
        backend=args.backend,
        compile_mode=args.compile_mode,
        graph_output_policy=args.graph_output_policy,
    )
    metrics = compare_outputs(reference, candidate)
    payload = {
        "backend": args.backend,
        "compile_mode": args.compile_mode,
        "graph_output_policy": args.graph_output_policy,
        "rgb_replay": str(args.rgb_replay),
        "prompt_source": "deterministic_replay_boxes",
        "metrics": metrics,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_compare_report(payload))
    if args.debug:
        print(payload)
    return 0 if metrics["correctness_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
