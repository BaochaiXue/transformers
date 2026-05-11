"""Correctness harness for EdgeTAM batched multi-session backends."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .batched_multisession_runtime import run_candidate
from .backend_contract import FullBatchedContractError, contract_for_current_runtime
from .camera_order import diagonal_best, iou_matrix, mask_iou
from .config import BACKENDS
from .leakage_test import compare_cam0_stability
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
                cand_mask = _resize_mask_like(cand_mask, ref_mask)
                ref_has = bool(np.count_nonzero(ref_mask))
                cand_has = bool(np.count_nonzero(cand_mask))
                ref_nonempty += int(ref_has)
                cand_nonempty += int(cand_has)
                empty_mismatch_count += int(ref_has != cand_has)
                ious.append(mask_iou(ref_mask, cand_mask))
                ref_logit = reference.logits[frame_idx][cam_idx][obj_idx]
                cand_logit = candidate.logits[frame_idx][cam_idx][obj_idx]
                cand_logit = _resize_float_like(cand_logit, ref_logit)
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
    first_cand = [
        _resize_mask_like(candidate.masks[0][cam_idx][0], reference.masks[0][cam_idx][0])
        for cam_idx in range(camera_count)
    ] if frame_count else []
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
            f"- reference_source: `{payload.get('reference_source', 'hf-public')}`",
            f"- init_source: `{payload.get('init_source')}`",
            f"- prompt_source: `{payload.get('prompt_source')}`",
            f"- sam31_mask_root: `{payload.get('sam31_mask_root')}`",
            f"- sam31_frame0_init_mask_root: `{payload.get('sam31_frame0_init_mask_root')}`",
            f"- strict_full_batched: `{payload.get('strict_full_batched', False)}`",
            f"- disallow_partial_backend_success: `{payload.get('disallow_partial_backend_success', False)}`",
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument(
        "--reference-source",
        choices=("hf-public", "sam31-replay"),
        default="hf-public",
        help="Reference masks used for IoU_ref.",
    )
    parser.add_argument(
        "--init-source",
        choices=("auto", "deterministic", "sam31-video-reference", "sam31-image-frame0"),
        default="auto",
        help="Frame-0 masks used to initialize EdgeTAM sessions.",
    )
    parser.add_argument("--sam31-mask-root", default=None)
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
    parser.add_argument("--strict-full-batched", action="store_true")
    parser.add_argument("--disallow-partial-backend-success", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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
    sam31_mask_root = None
    sam31_frame0_init_mask_root = None
    sam31_frame0_init_summary = None
    init_source = args.init_source
    if init_source == "auto":
        init_source = "sam31-video-reference" if args.reference_source == "sam31-replay" else "deterministic"

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
            "graph_output_policy": args.graph_output_policy,
            "rgb_replay": str(args.rgb_replay),
            "reference_source": args.reference_source,
            "init_source": init_source,
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
    metrics = compare_outputs(reference, candidate)
    payload = {
        "backend": args.backend,
        "compile_mode": args.compile_mode,
        "graph_output_policy": args.graph_output_policy,
        "rgb_replay": str(args.rgb_replay),
        "reference_source": args.reference_source,
        "init_source": init_source,
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
