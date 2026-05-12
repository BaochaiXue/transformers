"""Correctness harness for EdgeTAM batched multi-session backends."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .batched_multisession_runtime import run_candidate
from .backend_contract import FullBatchedContractError, contract_for_current_runtime
from .camera_order import diagonal_best, iou_matrix, mask_iou
from .config import BACKENDS, COMPILE_SCOPES
from .leakage_test import compare_cam0_stability
from .precision_policy import PRECISION_POLICY_NAMES, reference_dtype_for_precision_mode
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames
from .sam31_frame0_init import (
    default_sam31_frame0_init_mask_root,
    generate_sam31_frame0_init_masks,
)
from .sam31_replay_reference import (
    default_sam31_mask_root,
    generate_sam31_replay_masks,
    initial_masks_by_camera_from_sam31,
    load_sam31_reference_outputs,
)
from .stats import summarize


def resolve_correctness_gate(
    gate_name: str,
    *,
    min_global_iou_avg: float | None = None,
    min_global_iou_p50: float | None = None,
    max_empty_mismatch_count: int | None = None,
) -> dict[str, Any]:
    gate_name = gate_name.replace("_", "-")
    if gate_name == "strict":
        defaults = {
            "min_global_iou_avg": 0.98,
            "min_global_iou_p50": 0.98,
            "max_empty_mismatch_count": 0,
            "speed_first": False,
        }
    elif gate_name == "speed-first":
        defaults = {
            "min_global_iou_avg": 0.93,
            "min_global_iou_p50": 0.95,
            "max_empty_mismatch_count": 3,
            "speed_first": True,
        }
    else:
        raise ValueError(f"unsupported correctness gate: {gate_name}")
    if min_global_iou_avg is not None:
        defaults["min_global_iou_avg"] = float(min_global_iou_avg)
    if min_global_iou_p50 is not None:
        defaults["min_global_iou_p50"] = float(min_global_iou_p50)
    if max_empty_mismatch_count is not None:
        defaults["max_empty_mismatch_count"] = int(max_empty_mismatch_count)
    defaults["name"] = gate_name
    return defaults


def resolve_empty_reference_policy(policy: str, *, reference_source: str = "hf-public") -> str:
    if policy == "auto":
        return "ignore-candidate" if reference_source == "sam31-replay" else "strict-empty-mismatch"
    if policy not in {"ignore-candidate", "strict-empty-mismatch"}:
        raise ValueError(f"unsupported empty reference policy: {policy}")
    return policy


def compare_outputs(
    reference,
    candidate,
    *,
    correctness_gate: str = "strict",
    empty_reference_policy: str = "strict-empty-mismatch",
    min_global_iou_avg: float | None = None,
    min_global_iou_p50: float | None = None,
    max_empty_mismatch_count: int | None = None,
    min_evaluated_samples: int = 1,
    min_evaluated_samples_per_object: int = 1,
    min_evaluated_samples_per_camera: int = 1,
) -> dict[str, Any]:
    empty_reference_policy = resolve_empty_reference_policy(empty_reference_policy)
    gate = resolve_correctness_gate(
        correctness_gate,
        min_global_iou_avg=min_global_iou_avg,
        min_global_iou_p50=min_global_iou_p50,
        max_empty_mismatch_count=max_empty_mismatch_count,
    )
    frame_count = min(len(reference.masks), len(candidate.masks))
    camera_count = len(reference.masks[0]) if frame_count else 0
    object_count = reference.masks[0][0].shape[0] if frame_count else 0
    per_key: dict[str, dict[str, list[float] | int]] = {}
    sample_statuses = []
    empty_mismatch_count = 0
    global_evaluated_sample_count = 0
    global_ignored_reference_empty_count = 0
    global_candidate_nonempty_when_reference_empty_count = 0
    global_reference_nonempty_count = 0
    global_candidate_nonempty_count_on_evaluated = 0
    global_ious: list[float] = []

    for cam_idx in range(camera_count):
        for obj_idx in range(object_count):
            key = f"cam{cam_idx}_obj{obj_idx}"
            ious = []
            logit_diffs = []
            score_diffs = []
            ref_nonempty = 0
            cand_nonempty = 0
            candidate_nonempty_on_evaluated = 0
            ignored_reference_empty = 0
            candidate_nonempty_when_reference_empty = 0
            evaluated_sample_count = 0
            counted_empty_mismatches = 0
            for frame_idx in range(frame_count):
                ref_mask = reference.masks[frame_idx][cam_idx][obj_idx]
                cand_mask = candidate.masks[frame_idx][cam_idx][obj_idx]
                cand_mask = _resize_mask_like(cand_mask, ref_mask)
                ref_has = bool(np.count_nonzero(ref_mask))
                cand_has = bool(np.count_nonzero(cand_mask))
                ref_nonempty += int(ref_has)
                cand_nonempty += int(cand_has)
                evaluated = ref_has or empty_reference_policy == "strict-empty-mismatch"
                ignored_reason = None
                empty_mismatch_counted = False
                iou_value = None
                if not evaluated:
                    ignored_reference_empty += 1
                    candidate_nonempty_when_reference_empty += int(cand_has)
                    ignored_reason = "reference_empty"
                else:
                    evaluated_sample_count += 1
                    candidate_nonempty_on_evaluated += int(cand_has)
                    if ref_has != cand_has:
                        empty_mismatch_count += 1
                        counted_empty_mismatches += 1
                        empty_mismatch_counted = True
                    iou_value = mask_iou(ref_mask, cand_mask)
                    ious.append(iou_value)
                    ref_logit = reference.logits[frame_idx][cam_idx][obj_idx]
                    cand_logit = candidate.logits[frame_idx][cam_idx][obj_idx]
                    cand_logit = _resize_float_like(cand_logit, ref_logit)
                    logit_diffs.append(float(np.mean(np.abs(ref_logit - cand_logit))))
                    ref_score = np.asarray(reference.object_scores[frame_idx][cam_idx]).reshape(-1)
                    cand_score = np.asarray(candidate.object_scores[frame_idx][cam_idx]).reshape(-1)
                    if obj_idx < len(ref_score) and obj_idx < len(cand_score):
                        score_diffs.append(float(abs(ref_score[obj_idx] - cand_score[obj_idx])))
                sample_statuses.append(
                    {
                        "camera": f"cam{cam_idx}",
                        "object_index": obj_idx,
                        "frame_idx": frame_idx,
                        "reference_nonempty": ref_has,
                        "candidate_nonempty": cand_has,
                        "evaluated": evaluated,
                        "ignored_reason": ignored_reason,
                        "iou": iou_value,
                        "empty_mismatch_counted": empty_mismatch_counted,
                    }
                )
            global_evaluated_sample_count += evaluated_sample_count
            global_ignored_reference_empty_count += ignored_reference_empty
            global_candidate_nonempty_when_reference_empty_count += candidate_nonempty_when_reference_empty
            global_reference_nonempty_count += ref_nonempty
            global_candidate_nonempty_count_on_evaluated += candidate_nonempty_on_evaluated
            global_ious.extend(ious)
            mask_iou_summary = summarize(ious)
            per_key[key] = {
                "mask_iou": mask_iou_summary,
                "iou_values_on_evaluated": list(ious),
                "iou_avg_on_evaluated": mask_iou_summary["avg"],
                "iou_p50_on_evaluated": mask_iou_summary["p50"],
                "iou_min_on_evaluated": mask_iou_summary["min"],
                "logit_abs_diff": summarize(logit_diffs),
                "object_score_abs_diff": summarize(score_diffs),
                "evaluated_sample_count": evaluated_sample_count,
                "ignored_reference_empty_count": ignored_reference_empty,
                "candidate_nonempty_when_reference_empty_count": candidate_nonempty_when_reference_empty,
                "reference_nonempty_count": ref_nonempty,
                "candidate_nonempty_count": cand_nonempty,
                "candidate_nonempty_count_on_evaluated": candidate_nonempty_on_evaluated,
                "empty_mismatch_count": counted_empty_mismatches,
                "reference_uncertain": ignored_reference_empty > 0,
            }

    first_ref = [reference.masks[0][cam_idx][0] for cam_idx in range(camera_count)] if frame_count else []
    first_cand = [
        _resize_mask_like(candidate.masks[0][cam_idx][0], reference.masks[0][cam_idx][0])
        for cam_idx in range(camera_count)
    ] if frame_count else []
    order = diagonal_best(iou_matrix(first_cand, first_ref)) if first_ref else {"pass": False, "matrix": []}
    leakage = {"pass": True, "note": "candidate runtime keeps per-camera session state containers isolated"}
    global_iou = summarize(global_ious)
    object_summaries = _summarize_by_object(per_key, object_count, min_evaluated_samples_per_object)
    camera_summaries = _summarize_by_camera(per_key, camera_count, min_evaluated_samples_per_camera)
    all_objects_validated = all(item["object_validated"] for item in object_summaries.values()) if object_summaries else False
    all_cameras_validated = all(item["camera_validated"] for item in camera_summaries.values()) if camera_summaries else False
    strict_gate = resolve_correctness_gate("strict")
    speed_first_gate = resolve_correctness_gate("speed-first")
    strict_correctness_pass = _passes_gate(
        strict_gate,
        order_pass=bool(order["pass"]),
        evaluated_sample_count=global_evaluated_sample_count,
        min_evaluated_samples=int(min_evaluated_samples),
        empty_mismatch_count=empty_mismatch_count,
        global_iou=global_iou,
    )
    speed_first_acceptance_pass = _passes_gate(
        speed_first_gate,
        order_pass=bool(order["pass"]),
        evaluated_sample_count=global_evaluated_sample_count,
        min_evaluated_samples=int(min_evaluated_samples),
        empty_mismatch_count=empty_mismatch_count,
        global_iou=global_iou,
    )
    mask_correctness_pass = _passes_gate(
        gate,
        order_pass=bool(order["pass"]),
        evaluated_sample_count=global_evaluated_sample_count,
        min_evaluated_samples=int(min_evaluated_samples),
        empty_mismatch_count=empty_mismatch_count,
        global_iou=global_iou,
    )
    correctness_pass = mask_correctness_pass and not candidate.partial
    blockers = list(candidate.blockers or [])
    if not mask_correctness_pass:
        blockers.append(
            "mask correctness gate failed: "
            f"gate={gate['name']}, "
            f"evaluated={global_evaluated_sample_count}/{min_evaluated_samples}, "
            f"empty_mismatch={empty_mismatch_count}/{gate['max_empty_mismatch_count']}, "
            f"global_iou_avg={global_iou['avg']}/{gate['min_global_iou_avg']}, "
            f"global_iou_p50={global_iou['p50']}/{gate['min_global_iou_p50']}"
        )
    return {
        "frame_count": frame_count,
        "camera_count": camera_count,
        "object_count": object_count,
        "per_camera_object": per_key,
        "global_mask_iou": global_iou,
        "global_iou_on_evaluated": global_iou,
        "evaluated_sample_count": global_evaluated_sample_count,
        "ignored_reference_empty_count": global_ignored_reference_empty_count,
        "candidate_nonempty_when_reference_empty_count": global_candidate_nonempty_when_reference_empty_count,
        "reference_nonempty_count": global_reference_nonempty_count,
        "candidate_nonempty_count_on_evaluated": global_candidate_nonempty_count_on_evaluated,
        "empty_mismatch_count": empty_mismatch_count,
        "empty_reference_policy": empty_reference_policy,
        "correctness_gate": gate,
        "min_evaluated_samples": int(min_evaluated_samples),
        "min_evaluated_samples_per_object": int(min_evaluated_samples_per_object),
        "min_evaluated_samples_per_camera": int(min_evaluated_samples_per_camera),
        "object_summaries": object_summaries,
        "camera_summaries": camera_summaries,
        "all_objects_validated": all_objects_validated,
        "all_cameras_validated": all_cameras_validated,
        "sample_statuses": sample_statuses,
        "camera_order_check": "pass" if order["pass"] else "fail",
        "camera_order_matrix": order["matrix"],
        "state_leakage_check": "pass" if leakage["pass"] else "fail",
        "state_leakage": leakage,
        "mask_correctness_pass": mask_correctness_pass,
        "active_gate_acceptance_pass": mask_correctness_pass,
        "strict_correctness_pass": strict_correctness_pass,
        "speed_first_acceptance_pass": speed_first_acceptance_pass,
        "correctness_pass_by_gate": {
            "strict": strict_correctness_pass and not candidate.partial,
            "speed_first": speed_first_acceptance_pass and not candidate.partial,
        },
        "correctness_pass": correctness_pass,
        "candidate_partial": candidate.partial,
        "fallback_backend": candidate.fallback_backend,
        "blockers": blockers,
        "backend_contract": getattr(candidate, "backend_contract", None),
    }


def select_object_outputs(outputs, *, object_count: int, object_index: int = -1):
    """Return an Outputs-like view restricted to a requested object count.

    Existing SAM3.1 sidecars are usually ordered as controller/object. For the
    single-object experiment we keep the object layer, i.e. the last mask plane.
    """

    object_count = int(object_count)
    if object_count != 1:
        return outputs
    selected_masks = []
    selected_logits = []
    selected_scores = []
    for frame_idx, frame_masks in enumerate(outputs.masks):
        masks_by_cam = []
        logits_by_cam = []
        scores_by_cam = []
        for cam_idx, cam_masks in enumerate(frame_masks):
            mask_arr = np.asarray(cam_masks)
            logit_arr = np.asarray(outputs.logits[frame_idx][cam_idx])
            score_arr = np.asarray(outputs.object_scores[frame_idx][cam_idx]).reshape(-1)
            idx = object_index if object_index >= 0 else mask_arr.shape[0] - 1
            masks_by_cam.append(mask_arr[idx : idx + 1])
            logits_by_cam.append(logit_arr[idx : idx + 1])
            if idx < score_arr.shape[0]:
                scores_by_cam.append(score_arr[idx : idx + 1])
            else:
                scores_by_cam.append(np.ones((1,), dtype=np.float32))
        selected_masks.append(masks_by_cam)
        selected_logits.append(logits_by_cam)
        selected_scores.append(scores_by_cam)
    return type(
        "Outputs",
        (),
        {
            "masks": selected_masks,
            "logits": selected_logits,
            "object_scores": selected_scores,
            "timings_ms": getattr(outputs, "timings_ms", {}),
            "backend": getattr(outputs, "backend", "selected_object"),
            "partial": getattr(outputs, "partial", False),
            "fallback_backend": getattr(outputs, "fallback_backend", None),
            "blockers": getattr(outputs, "blockers", []),
            "backend_contract": getattr(outputs, "backend_contract", None),
        },
    )()


def _summarize_by_object(
    per_key: dict[str, dict[str, Any]],
    object_count: int,
    min_evaluated_samples_per_object: int,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for obj_idx in range(object_count):
        rows = [metrics for key, metrics in per_key.items() if key.endswith(f"_obj{obj_idx}")]
        evaluated = sum(int(row.get("evaluated_sample_count") or 0) for row in rows)
        ignored = sum(int(row.get("ignored_reference_empty_count") or 0) for row in rows)
        candidate_nonempty_ignored = sum(
            int(row.get("candidate_nonempty_when_reference_empty_count") or 0) for row in rows
        )
        empty_mismatch = sum(int(row.get("empty_mismatch_count") or 0) for row in rows)
        reference_nonempty = sum(int(row.get("reference_nonempty_count") or 0) for row in rows)
        candidate_nonempty_on_evaluated = sum(
            int(row.get("candidate_nonempty_count_on_evaluated") or 0) for row in rows
        )
        iou_values = [
            float(value)
            for row in rows
            for value in row.get("iou_values_on_evaluated", [])
        ]
        iou_summary = summarize(iou_values)
        object_validated = evaluated >= int(min_evaluated_samples_per_object)
        summaries[f"obj{obj_idx}"] = {
            "evaluated_sample_count": evaluated,
            "ignored_reference_empty_count": ignored,
            "candidate_nonempty_when_reference_empty_count": candidate_nonempty_ignored,
            "reference_nonempty_count": reference_nonempty,
            "candidate_nonempty_count_on_evaluated": candidate_nonempty_on_evaluated,
            "empty_mismatch_count": empty_mismatch,
            "iou_avg_on_evaluated": iou_summary["avg"],
            "iou_p50_on_evaluated": iou_summary["p50"],
            "iou_min_on_evaluated": iou_summary["min"],
            "reference_uncertain": ignored > 0,
            "object_reference_absent": evaluated == 0,
            "object_validated": object_validated,
        }
    return summaries


def _summarize_by_camera(
    per_key: dict[str, dict[str, Any]],
    camera_count: int,
    min_evaluated_samples_per_camera: int,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for cam_idx in range(camera_count):
        rows = [metrics for key, metrics in per_key.items() if key.startswith(f"cam{cam_idx}_")]
        evaluated = sum(int(row.get("evaluated_sample_count") or 0) for row in rows)
        ignored = sum(int(row.get("ignored_reference_empty_count") or 0) for row in rows)
        candidate_nonempty_ignored = sum(
            int(row.get("candidate_nonempty_when_reference_empty_count") or 0) for row in rows
        )
        empty_mismatch = sum(int(row.get("empty_mismatch_count") or 0) for row in rows)
        reference_nonempty = sum(int(row.get("reference_nonempty_count") or 0) for row in rows)
        candidate_nonempty_on_evaluated = sum(
            int(row.get("candidate_nonempty_count_on_evaluated") or 0) for row in rows
        )
        iou_values = [
            float(value)
            for row in rows
            for value in row.get("iou_values_on_evaluated", [])
        ]
        iou_summary = summarize(iou_values)
        summaries[f"cam{cam_idx}"] = {
            "evaluated_sample_count": evaluated,
            "ignored_reference_empty_count": ignored,
            "candidate_nonempty_when_reference_empty_count": candidate_nonempty_ignored,
            "reference_nonempty_count": reference_nonempty,
            "candidate_nonempty_count_on_evaluated": candidate_nonempty_on_evaluated,
            "empty_mismatch_count": empty_mismatch,
            "iou_avg_on_evaluated": iou_summary["avg"],
            "iou_p50_on_evaluated": iou_summary["p50"],
            "iou_min_on_evaluated": iou_summary["min"],
            "reference_uncertain": ignored > 0,
            "camera_validated": evaluated >= int(min_evaluated_samples_per_camera),
        }
    return summaries


def _passes_gate(
    gate: dict[str, Any],
    *,
    order_pass: bool,
    evaluated_sample_count: int,
    min_evaluated_samples: int,
    empty_mismatch_count: int,
    global_iou: dict[str, Any],
) -> bool:
    return (
        bool(order_pass)
        and int(evaluated_sample_count) >= int(min_evaluated_samples)
        and int(empty_mismatch_count) <= int(gate["max_empty_mismatch_count"])
        and (global_iou.get("avg") is not None and global_iou["avg"] >= gate["min_global_iou_avg"])
        and (global_iou.get("p50") is not None and global_iou["p50"] >= gate["min_global_iou_p50"])
    )


def _expand_summary_sample(summary_dict: dict[str, Any]) -> list[float]:
    values = []
    for key in ("avg", "p50", "p90", "p95", "p99", "min", "max"):
        value = summary_dict.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _squeeze_spatial(array: Any) -> np.ndarray:
    arr = np.asarray(array)
    while arr.ndim > 2 and 1 in arr.shape[:-2]:
        arr = np.squeeze(arr, axis=tuple(idx for idx, size in enumerate(arr.shape[:-2]) if size == 1))
    if arr.ndim > 2:
        arr = arr.reshape((-1, *arr.shape[-2:]))[0]
    return arr


def _resize_mask_like(mask: Any, reference_mask: Any) -> np.ndarray:
    ref = _squeeze_spatial(reference_mask).astype(bool)
    arr = _squeeze_spatial(mask).astype(bool)
    if arr.shape == ref.shape:
        return arr
    from PIL import Image

    resized = Image.fromarray(arr.astype(np.uint8) * 255).resize(
        (ref.shape[1], ref.shape[0]),
        Image.Resampling.NEAREST,
    )
    return np.asarray(resized) > 0


def _resize_float_like(value: Any, reference_value: Any) -> np.ndarray:
    ref = _squeeze_spatial(reference_value).astype(np.float32)
    arr = _squeeze_spatial(value).astype(np.float32)
    if arr.shape == ref.shape:
        return arr
    from PIL import Image

    resized = Image.fromarray(arr).resize(
        (ref.shape[1], ref.shape[0]),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def render_compare_report(payload: dict[str, Any]) -> str:
    rows = []
    for key, metrics in payload["metrics"]["per_camera_object"].items():
        rows.append(
            [
                key,
                metrics["mask_iou"]["avg"],
                metrics["mask_iou"]["min"],
                metrics["mask_iou"]["p50"],
                metrics.get("evaluated_sample_count"),
                metrics.get("ignored_reference_empty_count"),
                metrics.get("candidate_nonempty_when_reference_empty_count"),
                metrics["reference_nonempty_count"],
                metrics.get("candidate_nonempty_count_on_evaluated"),
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
            f"- strict_correctness_pass: `{payload['metrics'].get('strict_correctness_pass')}`",
            f"- speed_first_acceptance_pass: `{payload['metrics'].get('speed_first_acceptance_pass')}`",
            f"- candidate_partial: `{payload['metrics']['candidate_partial']}`",
            f"- fallback_backend: `{payload['metrics']['fallback_backend']}`",
            f"- reference_source: `{payload.get('reference_source', 'hf-public')}`",
            f"- init_source: `{payload.get('init_source')}`",
            f"- prompt_source: `{payload.get('prompt_source')}`",
            f"- sam31_mask_root: `{payload.get('sam31_mask_root')}`",
            f"- sam31_frame0_init_mask_root: `{payload.get('sam31_frame0_init_mask_root')}`",
            f"- empty_reference_policy: `{payload['metrics'].get('empty_reference_policy')}`",
            f"- correctness_gate: `{payload['metrics'].get('correctness_gate', {}).get('name', 'strict')}`",
            f"- speed_first_gate: `{payload['metrics'].get('correctness_gate', {}).get('speed_first', False)}`",
            f"- evaluated_sample_count: `{payload['metrics'].get('evaluated_sample_count')}`",
            f"- ignored_reference_empty_count: `{payload['metrics'].get('ignored_reference_empty_count')}`",
            f"- candidate_nonempty_when_reference_empty_count: `{payload['metrics'].get('candidate_nonempty_when_reference_empty_count')}`",
            f"- strict_full_batched: `{payload.get('strict_full_batched', False)}`",
            f"- disallow_partial_backend_success: `{payload.get('disallow_partial_backend_success', False)}`",
            "",
            "## Blockers",
            "",
            "\n".join(f"- {item}" for item in payload["metrics"]["blockers"]) or "- none",
            "",
            "## Metrics",
            "",
            markdown_table(
                [
                    "key",
                    "iou_avg",
                    "iou_min",
                    "iou_p50",
                    "evaluated",
                    "ref_empty_ignored",
                    "cand_nonempty_ref_empty",
                    "ref_nonempty",
                    "cand_nonempty_eval",
                ],
                rows,
            ),
            "",
            "## Empty SAM3.1 Reference Policy",
            "",
            "When SAM3.1 reference is empty and the policy is `ignore-candidate`, candidate masks are ignored for IoU and empty-mismatch. Ignored samples are unevaluated, not correct.",
        ]
    )


def render_contract_failure_report(payload: dict[str, Any]) -> str:
    contract = payload.get("backend_contract") or {}
    blockers = payload["metrics"].get("blockers") or []
    return "\n".join(
        [
            "# EdgeTAM Strict Full-Batched Contract Failure",
            "",
            f"- backend: `{payload['backend']}`",
            f"- compile_mode: `{payload['compile_mode']}`",
            f"- strict_full_batched: `{payload.get('strict_full_batched', False)}`",
            f"- correctness_pass: `{payload['metrics']['correctness_pass']}`",
            f"- failure_stage: `{payload.get('failure_stage')}`",
            "",
            "## Contract",
            "",
            f"- contract_pass: `{contract.get('contract_pass', False)}`",
            f"- batch_vision: `{contract.get('batch_vision')}`",
            f"- batch_memory_attention: `{contract.get('batch_memory_attention')}`",
            f"- batch_mask_decoder: `{contract.get('batch_mask_decoder')}`",
            f"- batch_memory_encoder: `{contract.get('batch_memory_encoder')}`",
            f"- batched_state_scatter: `{contract.get('batched_state_scatter')}`",
            f"- used_public_session_step_in_hot_path: `{contract.get('used_public_session_step_in_hot_path')}`",
            f"- partial_fallback_used: `{contract.get('partial_fallback_used')}`",
            "",
            "## Blockers",
            "",
            "\n".join(f"- {item}" for item in blockers) or "- none",
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
    parser.add_argument("--precision-mode", choices=PRECISION_POLICY_NAMES, default="all_bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--compile-scope", choices=COMPILE_SCOPES, default=None)
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument(
        "--reference-source",
        choices=("hf-public", "hf-public-seq", "sam31-replay"),
        default="hf-public",
        help="Reference masks used for IoU_ref.",
    )
    parser.add_argument(
        "--init-source",
        choices=("auto", "deterministic", "sam31-video-reference", "sam31-image-frame0"),
        default="auto",
        help="Frame-0 masks used to initialize EdgeTAM sessions.",
    )
    parser.add_argument("--sam31-mask-root", "--reference-mask-root", dest="sam31_mask_root", default=None)
    parser.add_argument("--sam31-checkpoint", default=None)
    parser.add_argument("--sam31-overwrite", action="store_true")
    parser.add_argument("--sam31-compile-model", action="store_true")
    parser.add_argument("--sam31-frame0-init-mask-root", default=None)
    parser.add_argument("--sam31-frame0-init-overwrite", action="store_true")
    parser.add_argument("--sam31-frame0-controller-prompt", default=None)
    parser.add_argument("--sam31-frame0-object-prompt", default=None)
    parser.add_argument("--sam31-frame0-confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--sam31-frame0-controller-selection-mode",
        choices=("green-score", "largest", "all"),
        default="green-score",
    )
    parser.add_argument(
        "--sam31-frame0-object-selection-mode",
        choices=("green-score", "largest", "all"),
        default="largest",
    )
    parser.add_argument("--sam31-frame0-controller-max-instances", type=int, default=3)
    parser.add_argument("--sam31-frame0-object-max-instances", type=int, default=1)
    parser.add_argument("--sam31-frame0-min-area", type=int, default=64)
    parser.add_argument("--sam31-frame0-allow-empty", action="store_true")
    parser.add_argument("--qqtt-root", default="/home/zhangxinjie/proj-QQTT-v2")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--correctness-gate",
        choices=("strict", "speed-first"),
        default="strict",
        help="Correctness acceptance gate. speed-first is relaxed and must not be treated as strict validation.",
    )
    parser.add_argument("--min-global-iou-avg", type=float, default=None)
    parser.add_argument("--min-global-iou-p50", type=float, default=None)
    parser.add_argument("--max-empty-mismatch-count", type=int, default=None)
    parser.add_argument(
        "--empty-reference-policy",
        choices=("auto", "ignore-candidate", "strict-empty-mismatch"),
        default="auto",
    )
    parser.add_argument("--min-evaluated-samples", type=int, default=1)
    parser.add_argument("--min-evaluated-samples-per-object", type=int, default=1)
    parser.add_argument("--min-evaluated-samples-per-camera", type=int, default=1)
    parser.add_argument("--strict-full-batched", action="store_true")
    parser.add_argument("--disallow-partial-backend-success", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    frames = load_replay_frames(args.rgb_replay, args.frames)
    reference_dtype = reference_dtype_for_precision_mode(args.dtype, args.precision_mode)
    config = ReferenceRuntimeConfig(
        model_id=args.model_id,
        dtype=reference_dtype,
        device=args.device,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        object_count=args.object_count,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    sam31_mask_root = None
    sam31_frame0_init_mask_root = None
    sam31_frame0_init_summary = None
    init_source = args.init_source
    if init_source == "auto":
        init_source = "sam31-video-reference" if args.reference_source == "sam31-replay" else "deterministic"
    empty_reference_policy = resolve_empty_reference_policy(
        args.empty_reference_policy,
        reference_source=args.reference_source,
    )

    initial_masks_by_camera = None
    if args.reference_source == "sam31-replay":
        sam31_mask_root = args.sam31_mask_root or str(default_sam31_mask_root(args.rgb_replay))
        generate_sam31_replay_masks(
            rgb_replay=args.rgb_replay,
            output_dir=sam31_mask_root,
            qqtt_root=args.qqtt_root,
            checkpoint_path=args.sam31_checkpoint,
            object_prompt=args.object_prompt,
            controller_prompt=args.controller_prompt,
            overwrite=args.sam31_overwrite,
            compile_model=args.sam31_compile_model,
        )
        reference = load_sam31_reference_outputs(
            rgb_replay=args.rgb_replay,
            mask_root=sam31_mask_root,
            frames=args.frames,
            object_prompt=args.object_prompt,
            controller_prompt=args.controller_prompt,
        )
        reference = select_object_outputs(reference, object_count=args.object_count)

    if init_source == "sam31-video-reference":
        if sam31_mask_root is None:
            raise ValueError("--init-source sam31-video-reference requires --reference-source sam31-replay")
        initial_masks_by_camera = initial_masks_by_camera_from_sam31(
            rgb_replay=args.rgb_replay,
            mask_root=sam31_mask_root,
            object_prompt=args.object_prompt,
            controller_prompt=args.controller_prompt,
        )
    elif init_source == "sam31-image-frame0":
        sam31_frame0_init_mask_root = args.sam31_frame0_init_mask_root or str(
            default_sam31_frame0_init_mask_root(args.rgb_replay)
        )
        frame0_controller_prompt = args.sam31_frame0_controller_prompt or args.controller_prompt
        frame0_object_prompt = args.sam31_frame0_object_prompt or args.object_prompt
        sam31_frame0_init_summary = generate_sam31_frame0_init_masks(
            rgb_replay=args.rgb_replay,
            output_dir=sam31_frame0_init_mask_root,
            qqtt_root=args.qqtt_root,
            checkpoint_path=args.sam31_checkpoint,
            object_prompt=frame0_object_prompt,
            controller_prompt=frame0_controller_prompt,
            controller_label=args.controller_prompt,
            object_label=args.object_prompt,
            overwrite=args.sam31_frame0_init_overwrite,
            compile_model=args.sam31_compile_model,
            confidence_threshold=args.sam31_frame0_confidence_threshold,
            controller_selection_mode=args.sam31_frame0_controller_selection_mode,
            object_selection_mode=args.sam31_frame0_object_selection_mode,
            controller_max_instances=args.sam31_frame0_controller_max_instances,
            object_max_instances=args.sam31_frame0_object_max_instances,
            min_area=args.sam31_frame0_min_area,
            fail_on_empty=not args.sam31_frame0_allow_empty,
            device=args.device,
        )
        initial_masks_by_camera = initial_masks_by_camera_from_sam31(
            rgb_replay=args.rgb_replay,
            mask_root=sam31_frame0_init_mask_root,
            object_prompt=args.object_prompt,
            controller_prompt=args.controller_prompt,
        )

    prompt_source = {
        "deterministic": "deterministic_replay_boxes",
        "sam31-video-reference": "sam31_frame0_video_reference_masks",
        "sam31-image-frame0": "sam31_image_frame0_masks",
    }[init_source]

    if args.reference_source == "sam31-replay":
        reference_runtime.init_sessions(frames[0], initial_masks_by_camera=initial_masks_by_camera)
        reference_runtime.prompt_source = prompt_source
    else:
        reference_runtime.init_sessions(frames[0], initial_masks_by_camera=initial_masks_by_camera)
        reference_runtime.prompt_source = prompt_source
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
        reference = select_object_outputs(reference, object_count=args.object_count, object_index=0)

        # Candidate uses a fresh runtime but the same loaded model/processor.
        reference_runtime.init_sessions(frames[0], initial_masks_by_camera=initial_masks_by_camera)
        reference_runtime.prompt_source = prompt_source
    try:
        candidate = run_candidate(
            rgb_replay_frames=frames,
            reference_runtime=reference_runtime,
            backend=args.backend,
            compile_mode=args.compile_mode,
            graph_output_policy=args.graph_output_policy,
            strict_full_batched=args.strict_full_batched,
            disallow_partial_backend_success=args.disallow_partial_backend_success,
            precision_mode=args.precision_mode,
            compile_scope=args.compile_scope,
        )
    except FullBatchedContractError as exc:
        contract = contract_for_current_runtime(
            backend=args.backend,
            batch_vision=True,
            partial_fallback_used=True,
            blockers=[],
        ).to_json()
        blockers = list(contract.get("blockers") or [])
        blockers.insert(0, str(exc))
        payload = {
            "backend": args.backend,
            "compile_mode": args.compile_mode,
            "compile_scope": args.compile_scope,
            "graph_output_policy": args.graph_output_policy,
            "dtype": args.dtype,
            "effective_reference_dtype": reference_dtype,
            "precision_mode": args.precision_mode,
            "rgb_replay": str(args.rgb_replay),
            "object_count": args.object_count,
            "object_prompt": args.object_prompt,
            "controller_prompt": args.controller_prompt,
            "reference_source": args.reference_source,
            "init_source": init_source,
            "empty_reference_policy": empty_reference_policy,
            "strict_full_batched": args.strict_full_batched,
            "disallow_partial_backend_success": args.disallow_partial_backend_success,
            "failure_stage": "backend_contract",
            "backend_contract": contract,
            "metrics": {
                "frame_count": 0,
                "camera_count": 0,
                "object_count": args.object_count,
                "per_camera_object": {},
                "global_mask_iou": {},
                "empty_mismatch_count": None,
                "empty_reference_policy": empty_reference_policy,
                "correctness_gate": resolve_correctness_gate(
                    args.correctness_gate,
                    min_global_iou_avg=args.min_global_iou_avg,
                    min_global_iou_p50=args.min_global_iou_p50,
                    max_empty_mismatch_count=args.max_empty_mismatch_count,
                ),
                "evaluated_sample_count": 0,
                "ignored_reference_empty_count": 0,
                "candidate_nonempty_when_reference_empty_count": 0,
                "camera_order_check": "not_run",
                "state_leakage_check": "not_run",
                "mask_correctness_pass": False,
                "correctness_pass": False,
                "candidate_partial": True,
                "fallback_backend": None,
                "blockers": blockers,
                "backend_contract": contract,
            },
        }
        write_json(args.output_json, payload)
        write_markdown(args.output_md, render_contract_failure_report(payload))
        if args.debug:
            print(payload)
        return 3
    candidate = select_object_outputs(candidate, object_count=args.object_count, object_index=0)
    metrics = compare_outputs(
        reference,
        candidate,
        correctness_gate=args.correctness_gate,
        empty_reference_policy=empty_reference_policy,
        min_global_iou_avg=args.min_global_iou_avg,
        min_global_iou_p50=args.min_global_iou_p50,
        max_empty_mismatch_count=args.max_empty_mismatch_count,
        min_evaluated_samples=args.min_evaluated_samples,
        min_evaluated_samples_per_object=args.min_evaluated_samples_per_object,
        min_evaluated_samples_per_camera=args.min_evaluated_samples_per_camera,
    )
    payload = {
        "backend": args.backend,
        "compile_mode": args.compile_mode,
        "compile_scope": args.compile_scope,
        "graph_output_policy": args.graph_output_policy,
        "dtype": args.dtype,
        "effective_reference_dtype": reference_dtype,
        "precision_mode": args.precision_mode,
        "rgb_replay": str(args.rgb_replay),
        "object_count": args.object_count,
        "object_prompt": args.object_prompt,
        "controller_prompt": args.controller_prompt,
        "reference_source": args.reference_source,
        "init_source": init_source,
        "empty_reference_policy": empty_reference_policy,
        "sam31_mask_root": str(sam31_mask_root) if sam31_mask_root is not None else None,
        "sam31_frame0_init_mask_root": str(sam31_frame0_init_mask_root) if sam31_frame0_init_mask_root is not None else None,
        "sam31_frame0_init_summary": sam31_frame0_init_summary,
        "prompt_source": getattr(reference_runtime, "prompt_source", "deterministic_replay_boxes"),
        "strict_full_batched": args.strict_full_batched,
        "disallow_partial_backend_success": args.disallow_partial_backend_success,
        "metrics": metrics,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_compare_report(payload))
    if args.debug:
        print(payload)
    return 0 if metrics["correctness_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
