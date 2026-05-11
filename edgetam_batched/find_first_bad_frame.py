"""Locate the first full-batched EdgeTAM frame that diverges from HF public reference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .camera_order import mask_iou
from .compare_multisession import _resize_mask_like
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames


def tensor_diff_summary(reference: Any, candidate: Any) -> dict[str, Any]:
    """Return shape and absolute-difference stats for two tensor-like values."""

    try:
        import torch
    except Exception:  # pragma: no cover - torch is always available in integration envs.
        torch = None

    if reference is None or candidate is None:
        return {
            "present_in_reference": reference is not None,
            "present_in_candidate": candidate is not None,
            "shape_match": reference is None and candidate is None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "p95_abs_diff": None,
        }

    if torch is not None and hasattr(reference, "detach"):
        ref_arr = reference.detach().float().cpu().numpy()
    else:
        ref_arr = np.asarray(reference, dtype=np.float32)
    if torch is not None and hasattr(candidate, "detach"):
        cand_arr = candidate.detach().float().cpu().numpy()
    else:
        cand_arr = np.asarray(candidate, dtype=np.float32)

    shape_match = tuple(ref_arr.shape) == tuple(cand_arr.shape)
    if not shape_match:
        return {
            "present_in_reference": True,
            "present_in_candidate": True,
            "reference_shape": list(ref_arr.shape),
            "candidate_shape": list(cand_arr.shape),
            "shape_match": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "p95_abs_diff": None,
        }

    diff = np.abs(ref_arr - cand_arr).reshape(-1)
    if diff.size == 0:
        max_abs = mean_abs = p95_abs = 0.0
    else:
        max_abs = float(np.max(diff))
        mean_abs = float(np.mean(diff))
        p95_abs = float(np.percentile(diff, 95))
    return {
        "present_in_reference": True,
        "present_in_candidate": True,
        "reference_shape": list(ref_arr.shape),
        "candidate_shape": list(cand_arr.shape),
        "shape_match": True,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "p95_abs_diff": p95_abs,
    }


def classify_first_divergence(field_diffs: dict[str, dict[str, Any]]) -> str:
    """Infer the first likely component from current-frame output diffs."""

    priority = [
        ("pred_masks", "mask_decoder_or_accumulated_state"),
        ("object_score_logits", "mask_decoder_or_accumulated_state"),
        ("object_pointer", "mask_decoder_object_pointer_or_state_scatter"),
        ("maskmem_features", "memory_encoder_or_memory_state_recurrence"),
        ("maskmem_pos_enc", "memory_encoder_position_or_state_scatter"),
    ]
    for field, label in priority:
        diff = field_diffs.get(field) or {}
        p95 = diff.get("p95_abs_diff")
        max_abs = diff.get("max_abs_diff")
        if p95 is not None and float(p95) > 1e-3:
            return label
        if max_abs is not None and float(max_abs) > 1e-3:
            return label
        if diff.get("shape_match") is False:
            return label
    return "unknown_or_mask_threshold_only"


def _session_output(session: Any, frame_idx: int, *, obj_idx: int = 0) -> tuple[dict[str, Any], str | None]:
    output_dict = session.output_dict_per_obj[obj_idx]
    for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
        if frame_idx in output_dict[bucket]:
            return output_dict[bucket][frame_idx], bucket
    return {}, None


def _clone_output_fields(output: dict[str, Any]) -> dict[str, Any]:
    cloned = {}
    for key in ("pred_masks", "object_pointer", "object_score_logits", "maskmem_features", "maskmem_pos_enc"):
        value = output.get(key)
        cloned[key] = _clone_nested(value)
    return cloned


def _clone_nested(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().clone()
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    return value


def _flatten_first_tensor(value: Any) -> Any:
    if hasattr(value, "detach") or value is None:
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _flatten_first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _flatten_first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    return value


def _compare_output_fields(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields = sorted(set(reference) | set(candidate))
    return {
        field: tensor_diff_summary(
            _flatten_first_tensor(reference.get(field)),
            _flatten_first_tensor(candidate.get(field)),
        )
        for field in fields
    }


def _render_report(payload: dict[str, Any]) -> str:
    bad = payload.get("first_bad_frame")
    lines = [
        "# Full Batched EdgeTAM First Bad Frame",
        "",
        f"- backend: `{payload.get('backend')}`",
        f"- dtype: `{payload.get('dtype')}`",
        f"- replay: `{payload.get('rgb_replay')}`",
        f"- threshold: `{payload.get('iou_threshold')}`",
        "",
    ]
    if bad is None:
        lines.extend(["No frame below threshold was found.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## First Bad Sample",
            "",
            markdown_table(
                ["frame_idx", "camera", "object", "IoU", "first_diverging_component"],
                [[bad["frame_idx"], bad["camera"], bad["object_index"], f"{bad['iou']:.6f}", bad["first_diverging_component"]]],
            ),
            "",
            "## Field Diffs",
            "",
            markdown_table(
                ["field", "shape_match", "max_abs_diff", "mean_abs_diff", "p95_abs_diff"],
                [
                    [
                        field,
                        diff.get("shape_match"),
                        _fmt(diff.get("max_abs_diff")),
                        _fmt(diff.get("mean_abs_diff")),
                        _fmt(diff.get("p95_abs_diff")),
                    ]
                    for field, diff in sorted((bad.get("field_diffs") or {}).items())
                ],
            ),
            "",
            "## Interpretation",
            "",
            payload.get("interpretation", ""),
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--backend", default="hf_batched_multisession")
    parser.add_argument("--strict-full-batched", action="store_true")
    parser.add_argument("--reference-source", choices=("hf-public", "hf-public-seq"), default="hf-public-seq")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="hand")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--iou-threshold", type=float, default=0.90)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dump-component-diffs", action="store_true")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("find_first_bad_frame currently supports only HF public reference")

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

    reference_rows = []
    for step_idx, frame in enumerate(frames):
        masks, logits, scores, _timings = reference_runtime.step_public(frame)
        session_rows = []
        for cam_idx, session in enumerate(reference_runtime.sessions):
            frame_key = int(frame.frame_idx)
            output, bucket = _session_output(session, frame_key)
            if not output and frame_key != step_idx:
                output, bucket = _session_output(session, step_idx)
            session_rows.append(
                {
                    "bucket": bucket,
                    "output": _clone_output_fields(output),
                }
            )
        reference_rows.append(
            {
                "masks": masks,
                "logits": logits,
                "scores": scores,
                "session_outputs": session_rows,
            }
        )

    # Re-initialize the reference runtime as a state holder for the candidate,
    # matching compare_multisession.py and avoiding patching the public reference
    # path before it has been recorded.
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
        graph_output_policy=args.graph_output_policy,
        strict_full_batched=args.strict_full_batched,
        disallow_partial_backend_success=args.strict_full_batched,
    )
    runtime.init_from_reference_sessions(reference_runtime.sessions)
    runtime.prepare_compile(reference_runtime.torch)

    started = time.perf_counter()
    first_bad = None
    checked_samples = 0
    for step_idx, frame in enumerate(frames):
        result = runtime.step(frame.images, step_idx)
        for cam_idx, cand_masks in enumerate(result["masks_b3"]):
            ref_mask = reference_rows[step_idx]["masks"][cam_idx][0]
            cand_mask = _resize_mask_like(cand_masks[0], ref_mask)
            iou = mask_iou(ref_mask, cand_mask)
            checked_samples += 1
            if iou >= args.iou_threshold:
                continue

            frame_key = step_idx
            cand_output, cand_bucket = _session_output(runtime.sessions[cam_idx], frame_key)
            ref_session_output = reference_rows[step_idx]["session_outputs"][cam_idx]
            field_diffs = _compare_output_fields(ref_session_output["output"], _clone_output_fields(cand_output))
            first_bad = {
                "frame_idx": step_idx,
                "camera": f"cam{cam_idx}",
                "object_index": 0,
                "iou": float(iou),
                "reference_bucket": ref_session_output["bucket"],
                "candidate_bucket": cand_bucket,
                "first_diverging_component": classify_first_divergence(field_diffs),
                "field_diffs": field_diffs,
            }
            break
        if first_bad is not None:
            break

    interpretation = (
        "The strict full backend contract can be true while current masks still drift. "
        "Component fixture equivalence should be read together with this report: if isolated "
        "component equivalence passes but this report fails after many frames, the blocker is "
        "the recurrent session memory/state update path rather than a single isolated call."
    )
    payload = {
        "backend": args.backend,
        "dtype": args.dtype,
        "compile_mode": args.compile_mode,
        "rgb_replay": str(args.rgb_replay),
        "object_count": args.object_count,
        "object_prompt": args.object_prompt,
        "iou_threshold": args.iou_threshold,
        "checked_samples": checked_samples,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "first_bad_frame": first_bad,
        "backend_contract": runtime.contract.to_json() if runtime.contract is not None else None,
        "interpretation": interpretation,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, _render_report(payload))
    if args.debug:
        print(payload)
    return 0 if first_bad is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
