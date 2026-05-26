"""Analyze and render low-IoU frames from an EdgeTAM correctness run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .batched_multisession_runtime import run_candidate
from .camera_order import mask_iou
from .compare_multisession import _resize_mask_like, _squeeze_spatial
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames
from .sam31_replay_reference import initial_masks_by_camera_from_sam31
from .stats import summarize


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_area(box: list[int] | None) -> int:
    if box is None:
        return 0
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def bbox_center(box: list[int] | None) -> tuple[float, float] | None:
    if box is None:
        return None
    return ((float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5)


def center_distance(a: list[int] | None, b: list[int] | None) -> float | None:
    ca = bbox_center(a)
    cb = bbox_center(b)
    if ca is None or cb is None:
        return None
    return float(((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5)


def area_ratio(ref_area: int, cand_area: int) -> float | None:
    if ref_area <= 0 or cand_area <= 0:
        return None
    return float(cand_area) / float(ref_area)


def possible_reasons(
    *,
    ref_area: int,
    candidate_area: int,
    center_distance_px: float | None,
    bbox_area_ratio: float | None,
) -> list[str]:
    reasons = []
    if ref_area < 1000 or candidate_area < 1000:
        reasons.append("small mask IoU sensitivity")
    if center_distance_px is not None and center_distance_px > 20:
        reasons.append("tracking drift / location shift")
    if bbox_area_ratio is not None and (bbox_area_ratio < 0.5 or bbox_area_ratio > 2.0):
        reasons.append("scale mismatch")
    if not reasons:
        reasons.append("boundary/detail mismatch")
    return reasons


def make_overlay(rgb: Image.Image, ref_mask: np.ndarray, cand_mask: np.ndarray) -> Image.Image:
    base = np.asarray(rgb.convert("RGB")).astype(np.float32)
    base = (base * 0.45).astype(np.uint8)
    ref = resize_bool_to_image(ref_mask, rgb).astype(bool)
    cand = resize_bool_to_image(cand_mask, rgb).astype(bool)
    union = ref | cand
    original = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    base[union] = original[union]
    ref_only = ref & ~cand
    cand_only = cand & ~ref
    base[ref_only] = np.array([255, 32, 32], dtype=np.uint8)
    base[cand_only] = np.array([0, 220, 255], dtype=np.uint8)
    return Image.fromarray(base, mode="RGB")


def resize_bool_to_image(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    arr = np.asarray(mask).astype(bool)
    height, width = arr.shape
    if (width, height) == image.size:
        return arr
    resized = Image.fromarray(arr.astype(np.uint8) * 255, mode="L").resize(
        image.size,
        Image.Resampling.NEAREST,
    )
    return np.asarray(resized) > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def run_outputs(args: argparse.Namespace, correctness: dict[str, Any]):
    frames = load_replay_frames(args.replay, correctness.get("metrics", {}).get("frame_count"))
    init_root = correctness.get("sam31_frame0_init_mask_root")
    if not init_root:
        raise ValueError("correctness JSON must include sam31_frame0_init_mask_root")
    initial_masks_by_camera = initial_masks_by_camera_from_sam31(
        rgb_replay=args.replay,
        mask_root=init_root,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
    )
    config = ReferenceRuntimeConfig(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        object_count=int(correctness.get("metrics", {}).get("object_count", 2)),
    )
    runtime = HfEdgeTamReferenceRuntime(config)
    runtime.load()
    runtime.init_sessions(frames[0], initial_masks_by_camera=initial_masks_by_camera)

    ref_masks = []
    ref_logits = []
    ref_scores = []
    for frame in frames:
        masks, logits, scores, _timings = runtime.step_public(frame)
        ref_masks.append(masks)
        ref_logits.append(logits)
        ref_scores.append(scores)

    runtime.init_sessions(frames[0], initial_masks_by_camera=initial_masks_by_camera)
    candidate = run_candidate(
        rgb_replay_frames=frames,
        reference_runtime=runtime,
        backend=correctness.get("backend", "hf_batch_vision_seq_session"),
        compile_mode=correctness.get("compile_mode", args.compile_mode),
        graph_output_policy=correctness.get("graph_output_policy", "ring_buffer"),
    )
    reference = type(
        "Outputs",
        (),
        {
            "masks": ref_masks,
            "logits": ref_logits,
            "object_scores": ref_scores,
            "backend": "hf_public_reference",
            "partial": False,
            "fallback_backend": None,
            "blockers": [],
        },
    )()
    return frames, reference, candidate


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    correctness = json.loads(Path(args.correctness_json).read_text(encoding="utf-8"))
    if args.object_prompt is None:
        args.object_prompt = (
            correctness.get("sam31_frame0_init_summary", {}).get("object_prompt")
            or "stuffed animal"
        )
    if args.controller_prompt is None:
        args.controller_prompt = (
            correctness.get("sam31_frame0_init_summary", {}).get("controller_prompt")
            or "hand"
        )
    frames, reference, candidate = run_outputs(args, correctness)
    object_index = int(args.object_index)
    rows = []
    for frame_idx, frame in enumerate(frames):
        for cam_idx, image in enumerate(frame.images):
            ref_mask = _squeeze_spatial(reference.masks[frame_idx][cam_idx][object_index]).astype(bool)
            cand_mask = _squeeze_spatial(candidate.masks[frame_idx][cam_idx][object_index]).astype(bool)
            cand_mask = _resize_mask_like(cand_mask, ref_mask)
            intersection = int(np.count_nonzero(ref_mask & cand_mask))
            union = int(np.count_nonzero(ref_mask | cand_mask))
            iou = mask_iou(ref_mask, cand_mask)
            ref_area = int(np.count_nonzero(ref_mask))
            candidate_area = int(np.count_nonzero(cand_mask))
            ref_bbox = bbox(ref_mask)
            candidate_bbox = bbox(cand_mask)
            dist = center_distance(ref_bbox, candidate_bbox)
            ratio = area_ratio(bbox_area(ref_bbox), bbox_area(candidate_bbox))
            row = {
                "camera": f"cam{cam_idx}",
                "camera_idx": cam_idx,
                "frame_idx": int(frame.frame_idx),
                "iou": iou,
                "ref_area": ref_area,
                "candidate_area": candidate_area,
                "intersection_area": intersection,
                "union_area": union,
                "ref_bbox": ref_bbox,
                "candidate_bbox": candidate_bbox,
                "bbox_center_distance_px": dist,
                "bbox_area_ratio": ratio,
                "empty_ref": ref_area == 0,
                "empty_candidate": candidate_area == 0,
                "possible_reason": possible_reasons(
                    ref_area=ref_area,
                    candidate_area=candidate_area,
                    center_distance_px=dist,
                    bbox_area_ratio=ratio,
                ),
            }
            if iou < args.iou_threshold:
                rows.append(row)
    rows.sort(key=lambda item: (float(item["iou"]), item["camera"], item["frame_idx"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows[: args.top_k]:
        frame = frames[row["frame_idx"]]
        cam_idx = int(row["camera_idx"])
        stem = f"cam{cam_idx}_frame{int(row['frame_idx']):06d}"
        ref_mask = _squeeze_spatial(reference.masks[row["frame_idx"]][cam_idx][object_index]).astype(bool)
        cand_mask = _squeeze_spatial(candidate.masks[row["frame_idx"]][cam_idx][object_index]).astype(bool)
        cand_mask = _resize_mask_like(cand_mask, ref_mask)
        frame.images[cam_idx].save(output_dir / f"{stem}_rgb.png")
        save_mask(output_dir / f"{stem}_ref_mask.png", ref_mask)
        save_mask(output_dir / f"{stem}_candidate_mask.png", cand_mask)
        make_overlay(frame.images[cam_idx], ref_mask, cand_mask).save(
            output_dir / f"{stem}_overlay_ref_candidate.png"
        )
        row["rgb_path"] = str(output_dir / f"{stem}_rgb.png")
        row["ref_mask_path"] = str(output_dir / f"{stem}_ref_mask.png")
        row["candidate_mask_path"] = str(output_dir / f"{stem}_candidate_mask.png")
        row["overlay_path"] = str(output_dir / f"{stem}_overlay_ref_candidate.png")

    return {
        "correctness_json": args.correctness_json,
        "replay": args.replay,
        "object_name": args.object_name,
        "object_index": object_index,
        "iou_threshold": args.iou_threshold,
        "top_k": args.top_k,
        "low_iou_count": len(rows),
        "low_iou_iou_stats": summarize(row["iou"] for row in rows),
        "low_iou_frames": rows,
        "output_dir": str(output_dir),
    }


def render(payload: dict[str, Any]) -> str:
    rows = []
    for item in payload["low_iou_frames"][: payload["top_k"]]:
        rows.append(
            [
                item["camera"],
                item["frame_idx"],
                _round(item["iou"]),
                item["ref_area"],
                item["candidate_area"],
                _round(item["bbox_center_distance_px"]),
                _round(item["bbox_area_ratio"]),
                ", ".join(item["possible_reason"]),
                item.get("overlay_path"),
            ]
        )
    return "\n".join(
        [
            "# Low-IoU Frame Analysis",
            "",
            f"- replay: `{payload['replay']}`",
            f"- object: `{payload['object_name']}`",
            f"- object_index: `{payload['object_index']}`",
            f"- iou_threshold: `{payload['iou_threshold']}`",
            f"- low_iou_count: `{payload['low_iou_count']}`",
            f"- output_dir: `{payload['output_dir']}`",
            "",
            markdown_table(
                [
                    "camera",
                    "frame",
                    "IoU",
                    "ref_area",
                    "candidate_area",
                    "center_dist_px",
                    "bbox_area_ratio",
                    "possible_reason",
                    "overlay",
                ],
                rows,
            ),
        ]
    )


def _round(value: Any, digits: int = 5) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-json", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--object-index", type=int, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--object-prompt", default=None)
    parser.add_argument("--controller-prompt", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = analyze(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(json.dumps({"low_iou_count": payload["low_iou_count"], "output_dir": payload["output_dir"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
