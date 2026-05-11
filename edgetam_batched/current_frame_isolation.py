"""Current-frame probes for full-batched EdgeTAM divergence.

This module focuses on one frame/camera with reference state aligned before
the frame. It is intentionally diagnostic: replacement modes that cannot be
rerun inside the current HF helper path are reported as unsupported rather
than treated as proof.
"""

from __future__ import annotations

import argparse
import copy
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .camera_order import mask_iou
from .compare_multisession import _resize_mask_like
from .find_first_bad_frame import _clone_output_fields, _compare_output_fields, _session_output
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames


VARIANT_NORMAL = "normal_batch"
VARIANT_REPEATED_TARGET = "repeated_target"
VARIANT_BLACK_NEIGHBORS = "black_neighbors"
VARIANT_SHUFFLED_TARGET_FIRST = "shuffled_target_first"
VARIANT_BATCH1_WRAPPER = "batch1_wrapper"

REPLACEMENT_MODES = {
    "none",
    "vision_features_with_reference",
    "high_res_features_with_reference",
    "memory_attention_output_with_reference",
    "decoder_inputs_with_reference",
    "mask_decoder_output_with_reference",
    "object_pointer_with_reference",
    "memory_encoder_output_with_reference",
    "postprocess_output_with_reference",
}

PRECISION_MODES = {
    "all_bf16",
    "vision_fp32",
    "memory_attention_fp32",
    "mask_decoder_fp32",
    "memory_encoder_fp32",
    "object_pointer_fp32",
    "maskmem_features_fp32",
    "memory_path_fp32",
    "all_fp32",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    image_indices: tuple[int | None, ...]
    session_indices: tuple[int, ...]
    target_output_index: int
    reference_camera_index: int


def parse_camera_index(camera: str) -> int:
    text = str(camera).strip().lower()
    if text.startswith("cam"):
        text = text[3:]
    idx = int(text)
    if idx < 0:
        raise ValueError(f"camera index must be non-negative: {camera}")
    return idx


def dtype_for_precision(dtype: str, precision_mode: str) -> str:
    if precision_mode == "all_fp32":
        return "float32"
    return dtype


def replacement_status(replace: str) -> dict[str, Any]:
    if replace == "none":
        return {"supported": True, "posthoc": False, "reason": None}
    if replace in {"mask_decoder_output_with_reference", "postprocess_output_with_reference"}:
        return {
            "supported": True,
            "posthoc": True,
            "reason": (
                "replacement is applied to the reported current mask after the candidate step; "
                "it confirms downstream scoring sensitivity but does not rerun upstream components"
            ),
        }
    return {
        "supported": False,
        "posthoc": False,
        "reason": "runtime does not yet expose an in-frame component replacement hook for this level",
    }


def build_variant_specs(camera_idx: int, camera_count: int, *, include_batch1: bool = True) -> list[VariantSpec]:
    if camera_count < 3:
        raise ValueError("current-frame isolation expects a 3-camera replay")
    specs = [
        VariantSpec(VARIANT_NORMAL, tuple(range(camera_count)), tuple(range(camera_count)), camera_idx, camera_idx),
        VariantSpec(
            VARIANT_REPEATED_TARGET,
            tuple(camera_idx for _ in range(camera_count)),
            tuple(camera_idx for _ in range(camera_count)),
            camera_idx,
            camera_idx,
        ),
        VariantSpec(
            VARIANT_BLACK_NEIGHBORS,
            tuple(camera_idx if pos == camera_idx else None for pos in range(camera_count)),
            tuple(camera_idx for _ in range(camera_count)),
            camera_idx,
            camera_idx,
        ),
        VariantSpec(
            VARIANT_SHUFFLED_TARGET_FIRST,
            tuple([camera_idx] + [idx for idx in range(camera_count) if idx != camera_idx]),
            tuple([camera_idx] + [idx for idx in range(camera_count) if idx != camera_idx]),
            0,
            camera_idx,
        ),
    ]
    if include_batch1:
        specs.append(VariantSpec(VARIANT_BATCH1_WRAPPER, (camera_idx,), (camera_idx,), 0, camera_idx))
    return specs


def run_current_frame_isolation(
    *,
    rgb_replay: str,
    backend: str,
    object_count: int,
    object_prompt: str,
    controller_prompt: str,
    frame_idx: int,
    camera: str,
    dtype: str,
    device: str,
    force_reference_state_before_frame: bool,
    replace: str,
    precision_mode: str,
    compile_mode: str = "none",
    graph_output_policy: str = "ring_buffer",
    include_batch1: bool = True,
    only_normal_variant: bool = False,
) -> dict[str, Any]:
    if replace not in REPLACEMENT_MODES:
        raise ValueError(f"unsupported replacement mode: {replace}")
    if precision_mode not in PRECISION_MODES:
        raise ValueError(f"unsupported precision mode: {precision_mode}")
    effective_dtype = dtype_for_precision(dtype, precision_mode)
    target_camera_idx = parse_camera_index(camera)
    replay_frames = load_replay_frames(rgb_replay, frame_idx + 1)
    camera_count = len(replay_frames[0].images)
    if target_camera_idx >= camera_count:
        raise ValueError(f"{camera} is outside replay camera count {camera_count}")

    config = ReferenceRuntimeConfig(
        dtype=effective_dtype,
        device=device,
        object_prompt=object_prompt,
        controller_prompt=controller_prompt,
        object_count=object_count,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    reference_runtime.init_sessions(replay_frames[0])

    for step_idx in range(frame_idx):
        reference_runtime.step_public(replay_frames[step_idx])
    reference_pre_sessions = copy.deepcopy(reference_runtime.sessions)
    ref_masks, _ref_logits, _ref_scores, _ = reference_runtime.step_public(replay_frames[frame_idx])
    ref_output, ref_bucket = _session_output(reference_runtime.sessions[target_camera_idx], frame_idx)
    ref_output_clone = _clone_output_fields(ref_output)
    ref_mask = ref_masks[target_camera_idx][0]

    replacement = replacement_status(replace)
    specs = build_variant_specs(target_camera_idx, camera_count, include_batch1=include_batch1)
    if only_normal_variant or replace != "none":
        specs = [spec for spec in specs if spec.name == VARIANT_NORMAL]

    rows = []
    for spec in specs:
        row = run_variant(
            reference_runtime=reference_runtime,
            reference_pre_sessions=reference_pre_sessions,
            replay_frame=replay_frames[frame_idx],
            backend=backend,
            object_count=object_count,
            dtype=effective_dtype,
            device=device,
            compile_mode=compile_mode,
            graph_output_policy=graph_output_policy,
            force_reference_state_before_frame=force_reference_state_before_frame,
            spec=spec,
            ref_mask=ref_mask,
            ref_output=ref_output_clone,
            ref_bucket=ref_bucket,
            replace=replace,
            replacement=replacement,
        )
        rows.append(row)

    normal = next((row for row in rows if row["variant"] == VARIANT_NORMAL), rows[0] if rows else {})
    payload = {
        "backend": backend,
        "rgb_replay": str(rgb_replay),
        "frame_idx": frame_idx,
        "camera": f"cam{target_camera_idx}",
        "object_count": object_count,
        "object_prompt": object_prompt,
        "dtype": dtype,
        "effective_dtype": effective_dtype,
        "precision_mode": precision_mode,
        "compile_mode": compile_mode,
        "force_reference_state_before_frame": force_reference_state_before_frame,
        "replace": replace,
        "replacement": replacement,
        "variants": rows,
        "inferred_issue": infer_issue(rows, replace, precision_mode),
        "normal_variant": normal,
    }
    return payload


def run_variant(
    *,
    reference_runtime: HfEdgeTamReferenceRuntime,
    reference_pre_sessions: list[Any],
    replay_frame: Any,
    backend: str,
    object_count: int,
    dtype: str,
    device: str,
    compile_mode: str,
    graph_output_policy: str,
    force_reference_state_before_frame: bool,
    spec: VariantSpec,
    ref_mask: Any,
    ref_output: dict[str, Any],
    ref_bucket: str | None,
    replace: str,
    replacement: dict[str, Any],
) -> dict[str, Any]:
    from PIL import Image

    images = []
    for image_idx in spec.image_indices:
        if image_idx is None:
            width, height = replay_frame.images[0].size
            images.append(Image.new("RGB", (width, height), (0, 0, 0)))
        else:
            images.append(replay_frame.images[int(image_idx)])
    if force_reference_state_before_frame:
        sessions = [copy.deepcopy(reference_pre_sessions[idx]) for idx in spec.session_indices]
    else:
        sessions = [copy.deepcopy(reference_runtime.sessions[idx]) for idx in spec.session_indices]

    runtime = BatchedEdgeTamMultiSessionRuntime(
        reference_runtime.model,
        reference_runtime.processor,
        backend=backend,
        batch_size=len(images),
        object_count=object_count,
        dtype=reference_runtime.dtype if dtype != "float32" else reference_runtime.torch.float32,
        device=device,
        compile_mode=compile_mode,
        graph_output_policy=graph_output_policy,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
    )
    runtime.init_from_reference_sessions(sessions)
    runtime.prepare_compile(reference_runtime.torch)

    try:
        result = runtime.step(images, int(replay_frame.frame_idx))
        target_idx = spec.target_output_index
        cand_mask = _resize_mask_like(result["masks_b3"][target_idx][0], ref_mask)
        cand_output, cand_bucket = _session_output(runtime.sessions[target_idx], int(replay_frame.frame_idx))
        cand_output_clone = _clone_output_fields(cand_output)
        field_diffs = _compare_output_fields(ref_output, cand_output_clone)
        reported_mask = cand_mask
        if replacement.get("supported") and replacement.get("posthoc"):
            reported_mask = ref_mask
        iou = mask_iou(ref_mask, reported_mask)
        geometry = mask_geometry(ref_mask, cand_mask)
        row = {
            "variant": spec.name,
            "image_indices": list(spec.image_indices),
            "session_indices": list(spec.session_indices),
            "target_output_index": target_idx,
            "reference_bucket": ref_bucket,
            "candidate_bucket": cand_bucket,
            "status": "ok",
            "iou_vs_hf_public": float(iou),
            "raw_iou_vs_hf_public": float(mask_iou(ref_mask, cand_mask)),
            "geometry": geometry,
            "replacement_applied": bool(replacement.get("supported") and replacement.get("posthoc")),
            "replacement_supported": bool(replacement.get("supported")),
            "replacement_reason": replacement.get("reason"),
            "field_diffs": field_diffs,
            "summary_diffs": summarize_core_diffs(field_diffs),
            "backend_contract": result.get("backend_contract"),
        }
        return row
    except Exception as exc:
        return {
            "variant": spec.name,
            "image_indices": list(spec.image_indices),
            "session_indices": list(spec.session_indices),
            "target_output_index": spec.target_output_index,
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "replacement_supported": bool(replacement.get("supported")),
            "replacement_reason": replacement.get("reason"),
        }


def summarize_core_diffs(field_diffs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for name in ("pred_masks", "object_pointer", "object_score_logits", "maskmem_features", "maskmem_pos_enc"):
        diff = field_diffs.get(name) or {}
        result[f"{name}_p95_abs_diff"] = diff.get("p95_abs_diff")
        result[f"{name}_max_abs_diff"] = diff.get("max_abs_diff")
    return result


def mask_geometry(reference_mask: Any, candidate_mask: Any) -> dict[str, Any]:
    ref = _as_2d_mask(reference_mask)
    cand = _as_2d_mask(candidate_mask)
    if ref.shape != cand.shape:
        cand = _resize_mask_like(cand, ref)
    intersection = np.logical_and(ref, cand)
    union = np.logical_or(ref, cand)
    ref_area = int(np.count_nonzero(ref))
    cand_area = int(np.count_nonzero(cand))
    intersection_area = int(np.count_nonzero(intersection))
    union_area = int(np.count_nonzero(union))
    xor_area = int(np.count_nonzero(np.logical_xor(ref, cand)))
    ref_bbox = _bbox(ref)
    cand_bbox = _bbox(cand)
    return {
        "reference_area": ref_area,
        "candidate_area": cand_area,
        "intersection_area": intersection_area,
        "union_area": union_area,
        "threshold_flip_count": xor_area,
        "boundary_flip_count": xor_area,
        "reference_bbox": ref_bbox,
        "candidate_bbox": cand_bbox,
        "bbox_center_distance_px": _bbox_center_distance(ref_bbox, cand_bbox),
        "bbox_area_ratio": _bbox_area(cand_bbox) / max(float(_bbox_area(ref_bbox)), 1.0),
    }


def _as_2d_mask(value: Any) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    while mask.ndim > 2 and 1 in mask.shape:
        mask = np.squeeze(mask, axis=next(idx for idx, size in enumerate(mask.shape) if size == 1))
    if mask.ndim != 2:
        # Fall back to the last two spatial dimensions for defensive diagnostics.
        mask = np.reshape(mask, (-1, *mask.shape[-2:]))[-1]
    return mask


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_area(box: list[int] | None) -> int:
    if box is None:
        return 0
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _bbox_center_distance(a: list[int] | None, b: list[int] | None) -> float | None:
    if a is None or b is None:
        return None
    ax = (float(a[0]) + float(a[2])) / 2.0
    ay = (float(a[1]) + float(a[3])) / 2.0
    bx = (float(b[0]) + float(b[2])) / 2.0
    by = (float(b[1]) + float(b[3])) / 2.0
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def infer_issue(rows: list[dict[str, Any]], replace: str, precision_mode: str) -> str:
    if not rows:
        return "no_variants_ran"
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return "all_variants_failed_to_run"
    if replace != "none":
        supported = ok_rows[0].get("replacement_supported")
        if not supported:
            return "replacement_level_requires_runtime_debug_hook"
        return "posthoc_replacement_reaches_reference_mask; rerun-level hook still needed"
    by_name = {row["variant"]: row for row in ok_rows}
    normal = by_name.get(VARIANT_NORMAL)
    repeated = by_name.get(VARIANT_REPEATED_TARGET)
    black = by_name.get(VARIANT_BLACK_NEIGHBORS)
    shuffled = by_name.get(VARIANT_SHUFFLED_TARGET_FIRST)
    batch1 = by_name.get(VARIANT_BATCH1_WRAPPER)
    threshold = 0.90
    normal_bad = normal is not None and float(normal.get("raw_iou_vs_hf_public", 0.0)) < threshold
    repeated_pass = repeated is not None and float(repeated.get("raw_iou_vs_hf_public", 0.0)) >= threshold
    black_pass = black is not None and float(black.get("raw_iou_vs_hf_public", 0.0)) >= threshold
    shuffled_bad = shuffled is not None and float(shuffled.get("raw_iou_vs_hf_public", 1.0)) < threshold
    batch1_pass = batch1 is not None and float(batch1.get("raw_iou_vs_hf_public", 0.0)) >= threshold
    if normal_bad and batch1_pass:
        return "batch3_dimension_handling_or_diagonal_slicing_suspect"
    if normal_bad and repeated_pass:
        return "neighbor_dependent_batch_contamination_or_stack_split_order"
    if normal_bad and black_pass:
        return "neighbor_content_or_shared_batch_state_affects_target"
    if normal_bad and shuffled_bad:
        return "batch_order_or_diagonal_slicing_suspect"
    if precision_mode == "all_fp32" and not normal_bad:
        return "precision_sensitive_current_frame_path"
    if normal_bad:
        return "current_frame_hot_path_differs_from_hf_public_even_with_reference_state"
    return "no_current_frame_divergence_under_probe"


def render_current_frame_isolation(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload.get("variants") or []:
        diffs = row.get("summary_diffs") or {}
        rows.append(
            [
                row.get("variant"),
                row.get("status"),
                _fmt(row.get("raw_iou_vs_hf_public")),
                _fmt(row.get("iou_vs_hf_public")),
                _fmt(diffs.get("pred_masks_p95_abs_diff")),
                _fmt(diffs.get("object_pointer_p95_abs_diff")),
                _fmt(diffs.get("maskmem_features_p95_abs_diff")),
                row.get("error") or row.get("replacement_reason"),
            ]
        )
    return "\n".join(
        [
            "# Full Batched EdgeTAM Current-Frame Isolation",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["frame_idx", payload.get("frame_idx")],
                    ["camera", payload.get("camera")],
                    ["replace", payload.get("replace")],
                    ["precision_mode", payload.get("precision_mode")],
                    ["effective_dtype", payload.get("effective_dtype")],
                    ["force_reference_state_before_frame", payload.get("force_reference_state_before_frame")],
                    ["inferred_issue", payload.get("inferred_issue")],
                ],
            ),
            "",
            markdown_table(
                [
                    "variant",
                    "status",
                    "raw_iou",
                    "reported_iou",
                    "pred_p95",
                    "ptr_p95",
                    "mem_p95",
                    "note",
                ],
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
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--force-reference-state-before-frame", action="store_true")
    parser.add_argument("--dump-all-component-diffs", action="store_true")
    parser.add_argument("--replace", choices=sorted(REPLACEMENT_MODES), default="none")
    parser.add_argument("--precision-mode", choices=sorted(PRECISION_MODES), default="all_bf16")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("current_frame_isolation currently supports only HF public reference")
    payload = run_current_frame_isolation(
        rgb_replay=args.rgb_replay,
        backend=args.backend,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        frame_idx=args.frame_idx,
        camera=args.camera,
        dtype=args.dtype,
        device=args.device,
        force_reference_state_before_frame=args.force_reference_state_before_frame,
        replace=args.replace,
        precision_mode=args.precision_mode,
        compile_mode=args.compile_mode,
        graph_output_policy=args.graph_output_policy,
        only_normal_variant=args.replace != "none",
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_current_frame_isolation(payload))
    if args.debug:
        print(payload)
    ok = all(row.get("status") == "ok" for row in payload.get("variants") or [])
    divergent = any(float(row.get("raw_iou_vs_hf_public") or 0.0) < 0.90 for row in payload.get("variants") or [] if row.get("status") == "ok")
    return 0 if ok and not divergent else 2


if __name__ == "__main__":
    raise SystemExit(main())
