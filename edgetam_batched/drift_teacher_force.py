"""Teacher-forcing probes for full-batched EdgeTAM closed-loop drift."""

from __future__ import annotations

import argparse
import copy
from typing import Any

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .camera_order import mask_iou
from .compare_multisession import _resize_mask_like
from .find_first_bad_frame import _session_output
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames
from .stats import summarize


TEACHER_FORCE_MODES = {
    "none",
    "before_step_state",
    "after_memory_attention",
    "after_mask_decoder",
    "after_memory_encoder",
    "after_state_scatter",
}


def run_teacher_force(
    *,
    rgb_replay: str,
    backend: str,
    object_count: int,
    object_prompt: str,
    controller_prompt: str,
    frames: int,
    dtype: str,
    device: str,
    teacher_force: str,
    iou_threshold: float,
) -> dict[str, Any]:
    if teacher_force not in TEACHER_FORCE_MODES:
        raise ValueError(f"unsupported teacher-force mode: {teacher_force}")
    if teacher_force == "after_memory_attention":
        return {
            "teacher_force_mode": teacher_force,
            "status": "unsupported",
            "reason": "reference memory-attention output is not stored by HF public session; use component fixtures for isolated memory_attention equivalence",
            "drift_source_hypothesis": "not_evaluated",
        }

    replay_frames = load_replay_frames(rgb_replay, frames)
    config = ReferenceRuntimeConfig(
        dtype=dtype,
        device=device,
        object_prompt=object_prompt,
        controller_prompt=controller_prompt,
        object_count=object_count,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    reference_runtime.init_sessions(replay_frames[0])

    runtime = BatchedEdgeTamMultiSessionRuntime(
        reference_runtime.model,
        reference_runtime.processor,
        backend=backend,
        batch_size=len(replay_frames[0].images),
        object_count=object_count,
        dtype=reference_runtime.dtype,
        device=device,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
    )
    runtime.init_from_reference_sessions(copy.deepcopy(reference_runtime.sessions))
    runtime.prepare_compile(reference_runtime.torch)

    ious = []
    rows = []
    first_bad = None
    for step_idx, frame in enumerate(replay_frames):
        if teacher_force == "before_step_state":
            runtime.sessions = copy.deepcopy(reference_runtime.sessions)

        ref_masks, _ref_logits, _ref_scores, _ref_timings = reference_runtime.step_public(frame)
        result = runtime.step(frame.images, step_idx)

        if teacher_force == "after_memory_encoder":
            _force_current_outputs(reference_runtime.sessions, runtime.sessions, step_idx, fields=("maskmem_features", "maskmem_pos_enc"))
        elif teacher_force == "after_mask_decoder":
            _force_current_outputs(
                reference_runtime.sessions,
                runtime.sessions,
                step_idx,
                fields=("pred_masks", "object_pointer", "object_score_logits", "maskmem_features", "maskmem_pos_enc"),
            )
        elif teacher_force == "after_state_scatter":
            runtime.sessions = copy.deepcopy(reference_runtime.sessions)

        for cam_idx, cand_masks in enumerate(result["masks_b3"]):
            ref_mask = ref_masks[cam_idx][0]
            cand_mask = _resize_mask_like(cand_masks[0], ref_mask)
            iou = mask_iou(ref_mask, cand_mask)
            ious.append(iou)
            row = {"frame_idx": step_idx, "camera": f"cam{cam_idx}", "iou": float(iou)}
            rows.append(row)
            if first_bad is None and iou < iou_threshold:
                first_bad = row

    iou_summary = summarize(ious)
    return {
        "teacher_force_mode": teacher_force,
        "status": "ok",
        "backend": backend,
        "dtype": dtype,
        "rgb_replay": str(rgb_replay),
        "object_count": object_count,
        "object_prompt": object_prompt,
        "first_bad_frame": first_bad,
        "min_iou": iou_summary.get("min"),
        "global_iou_avg": iou_summary.get("avg"),
        "global_iou_p50": iou_summary.get("p50"),
        "component_replaced": _component_replaced(teacher_force),
        "drift_source_hypothesis": infer_teacher_force_source(teacher_force, first_bad),
        "rows": rows,
        "backend_contract": runtime.contract.to_json() if runtime.contract is not None else None,
    }


def _component_replaced(mode: str) -> str | None:
    return {
        "none": None,
        "before_step_state": "session_state_before_step",
        "after_memory_attention": "memory_attention_output",
        "after_mask_decoder": "decoder_and_memory_outputs",
        "after_memory_encoder": "memory_encoder_outputs",
        "after_state_scatter": "whole_session_after_scatter",
    }.get(mode)


def infer_teacher_force_source(mode: str, first_bad: dict[str, Any] | None) -> str:
    if first_bad is None:
        if mode == "before_step_state":
            return "closed_loop_accumulation_or_state_writeback"
        if mode == "after_memory_encoder":
            return "memory_encoder_or_maskmem_writeback"
        if mode == "after_mask_decoder":
            return "decoder_object_pointer_or_downstream_state"
        if mode == "after_state_scatter":
            return "scatter_or_session_store_output_semantics"
        return "no_drift_under_this_mode"
    if mode == "none":
        return "baseline_full_batched_closed_loop_drift"
    return f"drift_not_eliminated_by_{mode}"


def _force_current_outputs(reference_sessions: list[Any], candidate_sessions: list[Any], frame_idx: int, *, fields: tuple[str, ...]) -> None:
    for ref_session, cand_session in zip(reference_sessions, candidate_sessions):
        ref_output, _ref_bucket = _session_output(ref_session, frame_idx)
        cand_output, _cand_bucket = _session_output(cand_session, frame_idx)
        if not ref_output or not cand_output:
            continue
        for field in fields:
            if field in ref_output:
                cand_output[field] = _clone_tensor_tree(ref_output[field])


def _clone_tensor_tree(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    return copy.deepcopy(value)


def render_teacher_force(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Batched EdgeTAM Teacher Forcing",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["teacher_force_mode", payload.get("teacher_force_mode")],
                    ["status", payload.get("status")],
                    ["first_bad_frame", payload.get("first_bad_frame")],
                    ["min_iou", payload.get("min_iou")],
                    ["global_iou_avg", payload.get("global_iou_avg")],
                    ["global_iou_p50", payload.get("global_iou_p50")],
                    ["component_replaced", payload.get("component_replaced")],
                    ["drift_source_hypothesis", payload.get("drift_source_hypothesis")],
                    ["reason", payload.get("reason")],
                ],
            ),
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
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-force", choices=sorted(TEACHER_FORCE_MODES), default="none")
    parser.add_argument("--iou-threshold", type=float, default=0.90)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("drift_teacher_force currently supports only HF public reference")
    payload = run_teacher_force(
        rgb_replay=args.rgb_replay,
        backend=args.backend,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        frames=args.frames,
        dtype=args.dtype,
        device=args.device,
        teacher_force=args.teacher_force,
        iou_threshold=args.iou_threshold,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_teacher_force(payload))
    if args.debug:
        print(payload)
    return 0 if payload.get("status") == "unsupported" or payload.get("first_bad_frame") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
