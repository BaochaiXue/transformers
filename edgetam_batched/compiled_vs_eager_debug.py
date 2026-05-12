"""Compare eager and compiled full-batched EdgeTAM against HF public reference."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .batched_multisession_runtime import run_candidate
from .camera_order import mask_iou
from .compare_multisession import _resize_mask_like, select_object_outputs
from .config import COMPILE_SCOPES
from .precision_policy import PRECISION_POLICY_NAMES, reference_dtype_for_precision_mode
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames
from .stats import summarize


def find_first_compiled_regression(
    reference: Any,
    eager: Any,
    compiled: Any,
    *,
    threshold: float,
) -> dict[str, Any] | None:
    frame_count = min(len(reference.masks), len(eager.masks), len(compiled.masks))
    if frame_count == 0:
        return None
    camera_count = len(reference.masks[0])
    object_count = reference.masks[0][0].shape[0]
    for frame_idx in range(frame_count):
        for cam_idx in range(camera_count):
            for obj_idx in range(object_count):
                ref_mask = reference.masks[frame_idx][cam_idx][obj_idx]
                eager_mask = _resize_mask_like(eager.masks[frame_idx][cam_idx][obj_idx], ref_mask)
                compiled_mask = _resize_mask_like(compiled.masks[frame_idx][cam_idx][obj_idx], ref_mask)
                eager_iou = mask_iou(ref_mask, eager_mask)
                compiled_iou = mask_iou(ref_mask, compiled_mask)
                eager_compiled_iou = mask_iou(eager_mask, compiled_mask)
                if eager_iou >= threshold and compiled_iou < threshold:
                    return {
                        "frame_idx": frame_idx,
                        "camera": f"cam{cam_idx}",
                        "object_index": obj_idx,
                        "iou_hf_public_eager": float(eager_iou),
                        "iou_hf_public_compiled": float(compiled_iou),
                        "iou_eager_compiled": float(eager_compiled_iou),
                    }
    return None


def summarize_three_way(reference: Any, eager: Any, compiled: Any) -> dict[str, Any]:
    frame_count = min(len(reference.masks), len(eager.masks), len(compiled.masks))
    if frame_count == 0:
        return {
            "hf_public_vs_eager": summarize([]),
            "hf_public_vs_compiled": summarize([]),
            "eager_vs_compiled": summarize([]),
        }
    camera_count = len(reference.masks[0])
    object_count = reference.masks[0][0].shape[0]
    eager_ious = []
    compiled_ious = []
    eager_compiled_ious = []
    for frame_idx in range(frame_count):
        for cam_idx in range(camera_count):
            for obj_idx in range(object_count):
                ref_mask = reference.masks[frame_idx][cam_idx][obj_idx]
                eager_mask = _resize_mask_like(eager.masks[frame_idx][cam_idx][obj_idx], ref_mask)
                compiled_mask = _resize_mask_like(compiled.masks[frame_idx][cam_idx][obj_idx], ref_mask)
                eager_ious.append(mask_iou(ref_mask, eager_mask))
                compiled_ious.append(mask_iou(ref_mask, compiled_mask))
                eager_compiled_ious.append(mask_iou(eager_mask, compiled_mask))
    return {
        "hf_public_vs_eager": summarize(eager_ious),
        "hf_public_vs_compiled": summarize(compiled_ious),
        "eager_vs_compiled": summarize(eager_compiled_ious),
    }


def infer_diverging_component(compile_scope: str | None, compile_mode: str) -> str:
    if compile_mode == "none" or compile_scope == "none":
        return "none"
    scope = compile_scope or "vision_encoder"
    if scope == "vision_encoder":
        return "vision_encoder_or_cached_features"
    if scope == "memory_attention":
        return "memory_attention"
    if scope == "mask_decoder":
        return "mask_decoder"
    if scope == "memory_encoder":
        return "memory_encoder"
    if scope == "memory_attention_mask_decoder":
        return "memory_attention_or_mask_decoder"
    if scope == "memory_path_all":
        return "memory_path_component_or_state_scatter"
    if scope == "vision_memory_path_all":
        return "vision_or_memory_path_component"
    return scope


def run_debug(args: argparse.Namespace) -> dict[str, Any]:
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
    runtime = HfEdgeTamReferenceRuntime(config)
    runtime.load()
    runtime.init_sessions(frames[0])
    ref_masks = []
    ref_logits = []
    ref_scores = []
    for frame in frames:
        masks, logits, scores, _timings = runtime.step_public(frame)
        ref_masks.append(masks)
        ref_logits.append(logits)
        ref_scores.append(scores)
    reference = type("Outputs", (), {"masks": ref_masks, "logits": ref_logits, "object_scores": ref_scores})()
    reference = select_object_outputs(reference, object_count=args.object_count, object_index=0)

    runtime.init_sessions(frames[0])
    eager = run_candidate(
        rgb_replay_frames=frames,
        reference_runtime=runtime,
        backend=args.backend,
        compile_mode="none",
        graph_output_policy=args.graph_output_policy,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
        precision_mode=args.precision_mode,
        compile_scope="none",
    )
    eager = select_object_outputs(eager, object_count=args.object_count, object_index=0)

    runtime.init_sessions(frames[0])
    compiled = run_candidate(
        rgb_replay_frames=frames,
        reference_runtime=runtime,
        backend=args.backend,
        compile_mode=args.compile_mode,
        graph_output_policy=args.graph_output_policy,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
        precision_mode=args.precision_mode,
        compile_scope=args.compile_scope,
    )
    compiled = select_object_outputs(compiled, object_count=args.object_count, object_index=0)

    first = find_first_compiled_regression(reference, eager, compiled, threshold=args.iou_threshold)
    summary = summarize_three_way(reference, eager, compiled)
    return {
        "backend": args.backend,
        "rgb_replay": args.rgb_replay,
        "frames": args.frames,
        "object_count": args.object_count,
        "object_prompt": args.object_prompt,
        "dtype": args.dtype,
        "effective_reference_dtype": reference_dtype,
        "precision_mode": args.precision_mode,
        "compile_mode": args.compile_mode,
        "compile_scope": args.compile_scope or "vision_encoder",
        "graph_output_policy": args.graph_output_policy,
        "iou_threshold": args.iou_threshold,
        "eager_backend_contract": eager.backend_contract,
        "compiled_backend_contract": compiled.backend_contract,
        "first_compiled_regression": first,
        "first_diverging_component": infer_diverging_component(args.compile_scope, args.compile_mode) if first else None,
        "summary": summary,
        "compiled_strict_regression_found": first is not None,
    }


def render(payload: dict[str, Any]) -> str:
    first = payload.get("first_compiled_regression") or {}
    summary = payload.get("summary") or {}
    rows = []
    for key, value in summary.items():
        rows.append([key, _fmt(value.get("avg")), _fmt(value.get("min")), _fmt(value.get("p50"))])
    return "\n".join(
        [
            "# Compiled vs Eager Full Batched EdgeTAM Debug",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["precision_mode", payload.get("precision_mode")],
                    ["compile_mode", payload.get("compile_mode")],
                    ["compile_scope", payload.get("compile_scope")],
                    ["first_diverging_component", payload.get("first_diverging_component")],
                    ["first_bad_frame", first.get("frame_idx")],
                    ["first_bad_camera", first.get("camera")],
                    ["IoU HF/eager", _fmt(first.get("iou_hf_public_eager"))],
                    ["IoU HF/compiled", _fmt(first.get("iou_hf_public_compiled"))],
                    ["IoU eager/compiled", _fmt(first.get("iou_eager_compiled"))],
                ],
            ),
            "",
            markdown_table(["comparison", "avg", "min", "p50"], rows),
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
    parser.add_argument("--precision-mode", choices=PRECISION_POLICY_NAMES, default="memory_path_fp32")
    parser.add_argument("--compile-mode", required=True)
    parser.add_argument("--compile-scope", choices=COMPILE_SCOPES, default=None)
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--iou-threshold", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("compiled_vs_eager_debug currently supports only HF public reference")
    payload = run_debug(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 2 if payload.get("compiled_strict_regression_found") else 0


if __name__ == "__main__":
    raise SystemExit(main())
