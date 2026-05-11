"""SAM3.1 image-mode frame-0 masks for EdgeTAM video initialization."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import frame_path, inspect_replay, load_manifest
from .sam31_replay_reference import DEFAULT_QQTT_ROOT


def default_sam31_frame0_init_mask_root(rgb_replay: str | Path) -> Path:
    return Path(rgb_replay).resolve() / "sam31_image_frame0_init_masks"


def generate_sam31_frame0_init_masks(
    *,
    rgb_replay: str | Path,
    output_dir: str | Path | None = None,
    qqtt_root: str | Path = DEFAULT_QQTT_ROOT,
    checkpoint_path: str | Path | None = None,
    object_prompt: str = "stuffed animal",
    controller_prompt: str = "green towel on wooden table",
    controller_label: str | None = None,
    object_label: str | None = None,
    overwrite: bool = False,
    compile_model: bool = False,
    confidence_threshold: float = 0.25,
    controller_selection_mode: str = "green-score",
    object_selection_mode: str = "largest",
    controller_max_instances: int = 3,
    object_max_instances: int = 1,
    min_area: int = 64,
    fail_on_empty: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """Generate frame-0 init masks with SAM3.1 image segmentation.

    The written mask tree intentionally matches the existing QQTT/SAM3.1 video
    sidecar schema so ``initial_masks_by_camera_from_sam31`` can load it.
    """

    replay_root = Path(rgb_replay).resolve()
    report = inspect_replay(replay_root)
    if not report["valid"]:
        raise FileNotFoundError(f"RGB replay is incomplete: {report['missing_sample'][:3]}")
    manifest = report["manifest"]
    output_root = Path(output_dir).resolve() if output_dir is not None else default_sam31_frame0_init_mask_root(replay_root)
    summary_path = output_root / "summary.json"
    if summary_path.exists() and not overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["reused_existing"] = True
        return summary
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    qqtt_root = Path(qqtt_root).resolve()
    if str(qqtt_root) not in sys.path:
        sys.path.insert(0, str(qqtt_root))
    from scripts.harness.sam31_mask_helper import run_image_segmentation  # noqa: PLC0415

    controller_label = controller_label or controller_prompt
    object_label = object_label or object_prompt
    camera_count = int(manifest["camera_count"])
    started = time.perf_counter()
    selected: dict[int, dict[str, np.ndarray]] = {}
    camera_summaries = []
    errors: list[str] = []

    for cam_idx in range(camera_count):
        image = Image.open(frame_path(replay_root, cam_idx, 0)).convert("RGB")
        controller_result = run_image_segmentation(
            image=image,
            text_prompt=controller_prompt,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
            confidence_threshold=confidence_threshold,
            device=device,
            reuse_model=True,
        )
        object_result = run_image_segmentation(
            image=image,
            text_prompt=object_prompt,
            checkpoint_path=checkpoint_path,
            compile_model=compile_model,
            confidence_threshold=confidence_threshold,
            device=device,
            reuse_model=True,
        )
        controller_candidates = _masks_for_prompt(controller_result, controller_prompt)
        object_candidates = _masks_for_prompt(object_result, object_prompt)
        controller = select_and_union_masks(
            controller_candidates,
            image,
            mode=controller_selection_mode,
            min_area=min_area,
            max_instances=controller_max_instances,
        )
        obj = select_and_union_masks(
            object_candidates,
            image,
            mode=object_selection_mode,
            min_area=min_area,
            max_instances=object_max_instances,
            avoid_mask=controller["mask"],
        )

        selected[cam_idx] = {"controller": controller["mask"], "object": obj["mask"]}
        cam_summary = {
            "camera_idx": cam_idx,
            "controller_prompt": controller_prompt,
            "object_prompt": object_prompt,
            "controller_candidate_count": len(controller_candidates),
            "object_candidate_count": len(object_candidates),
            "controller_selected_indices": controller["indices"],
            "object_selected_indices": obj["indices"],
            "controller_area": int(controller["mask"].sum()),
            "object_area": int(obj["mask"].sum()),
            "controller_selected_stats": controller["selected_stats"],
            "object_selected_stats": obj["selected_stats"],
            "controller_timing_ms": controller_result.get("timing_ms", {}),
            "object_timing_ms": object_result.get("timing_ms", {}),
        }
        if cam_summary["controller_area"] == 0:
            errors.append(f"cam{cam_idx}: empty controller frame0 mask for prompt `{controller_prompt}`")
        if cam_summary["object_area"] == 0:
            errors.append(f"cam{cam_idx}: empty object frame0 mask for prompt `{object_prompt}`")
        camera_summaries.append(cam_summary)

    if fail_on_empty and errors:
        write_json(
            output_root / "summary_failed.json",
            {
                "rgb_replay": str(replay_root),
                "output_dir": str(output_root),
                "errors": errors,
                "camera_summaries": camera_summaries,
            },
        )
        raise RuntimeError("; ".join(errors))

    write_frame0_init_mask_root(
        rgb_replay=replay_root,
        output_dir=output_root,
        selected_masks=selected,
        controller_label=controller_label,
        object_label=object_label,
    )
    _write_overlays(replay_root, output_root, selected)
    summary = {
        "rgb_replay": str(replay_root),
        "output_dir": str(output_root),
        "mask_root": str(output_root),
        "inference_mode": "sam31-image-frame0-init",
        "controller_prompt": controller_prompt,
        "object_prompt": object_prompt,
        "controller_label": controller_label,
        "object_label": object_label,
        "role_order": [controller_label, object_label],
        "controller_selection_mode": controller_selection_mode,
        "object_selection_mode": object_selection_mode,
        "controller_max_instances": int(controller_max_instances),
        "object_max_instances": int(object_max_instances),
        "confidence_threshold": float(confidence_threshold),
        "min_area": int(min_area),
        "camera_summaries": camera_summaries,
        "errors": errors,
        "frame0_init_pass": not errors,
        "generation_wall_ms": float((time.perf_counter() - started) * 1000.0),
        "reused_existing": False,
    }
    write_json(summary_path, summary)
    write_markdown(output_root / "summary.md", render_frame0_init_report(summary))
    return summary


def _masks_for_prompt(result: dict[str, Any], prompt: str) -> list[np.ndarray]:
    masks_by_label = result.get("masks_by_label", {})
    if prompt in masks_by_label:
        return [np.asarray(mask, dtype=bool) for mask in masks_by_label[prompt]]
    normalized = " ".join(prompt.strip().lower().split())
    for label, masks in masks_by_label.items():
        if " ".join(str(label).strip().lower().split()) == normalized:
            return [np.asarray(mask, dtype=bool) for mask in masks]
    return []


def select_and_union_masks(
    masks: list[np.ndarray],
    image: Image.Image,
    *,
    mode: str,
    min_area: int,
    max_instances: int,
    avoid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if mode not in {"green-score", "largest", "all"}:
        raise ValueError(f"unsupported selection mode: {mode}")
    if max_instances < 1:
        raise ValueError("max_instances must be >= 1")
    image_arr = np.asarray(image.convert("RGB"))
    scored = []
    for idx, mask in enumerate(masks):
        mask = _resize_mask_to_image(np.asarray(mask, dtype=bool), image_arr)
        stats = mask_stats(mask, image_arr)
        if stats["area"] < int(min_area):
            continue
        overlap = 0.0
        if avoid_mask is not None and np.count_nonzero(avoid_mask):
            overlap = mask_iou(mask, _resize_mask_to_image(np.asarray(avoid_mask, dtype=bool), image_arr))
        if mode == "largest":
            score = float(stats["area"]) * (1.0 - 0.5 * overlap)
        elif mode == "green-score":
            score = (
                float(np.sqrt(stats["area"]))
                * (1.0 + 4.0 * float(stats["green_fraction"]) + max(0.0, float(stats["green_delta"])) / 32.0)
                * (1.0 - 0.5 * overlap)
            )
        else:
            score = 1.0
        scored.append({"idx": idx, "mask": mask, "stats": {**stats, "overlap_iou": overlap, "score": score}})

    if mode == "all":
        selected = scored[:max_instances]
    else:
        selected = sorted(scored, key=lambda item: float(item["stats"]["score"]), reverse=True)[:max_instances]

    if not selected:
        empty = np.zeros(image_arr.shape[:2], dtype=bool)
        return {"mask": empty, "indices": [], "selected_stats": []}
    union = np.zeros(image_arr.shape[:2], dtype=bool)
    for item in selected:
        union |= item["mask"]
    return {
        "mask": union,
        "indices": [int(item["idx"]) for item in selected],
        "selected_stats": [item["stats"] for item in selected],
    }


def mask_stats(mask: np.ndarray, image_arr: np.ndarray) -> dict[str, Any]:
    mask = _resize_mask_to_image(mask.astype(bool), image_arr)
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    if area == 0:
        return {
            "area": 0,
            "bbox": None,
            "centroid": None,
            "mean_rgb": None,
            "green_fraction": 0.0,
            "green_delta": 0.0,
        }
    pixels = image_arr[mask].astype(np.float32)
    mean_rgb = pixels.mean(axis=0)
    red = pixels[:, 0]
    green = pixels[:, 1]
    blue = pixels[:, 2]
    green_fraction = float(np.mean((green > red * 1.05) & (green > blue * 1.05)))
    green_delta = float(mean_rgb[1] - 0.5 * (mean_rgb[0] + mean_rgb[2]))
    return {
        "area": area,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid": [float(xs.mean()), float(ys.mean())],
        "mean_rgb": [float(item) for item in mean_rgb.tolist()],
        "green_fraction": green_fraction,
        "green_delta": green_delta,
    }


def write_frame0_init_mask_root(
    *,
    rgb_replay: str | Path,
    output_dir: str | Path,
    selected_masks: dict[int, dict[str, np.ndarray]],
    controller_label: str,
    object_label: str,
) -> None:
    replay_root = Path(rgb_replay).resolve()
    output_root = Path(output_dir).resolve()
    manifest = load_manifest(replay_root)
    camera_count = int(manifest["camera_count"])
    mask_root = output_root / "mask"
    mask_root.mkdir(parents=True, exist_ok=True)
    for cam_idx in range(camera_count):
        cam_masks = selected_masks[cam_idx]
        (mask_root / str(cam_idx) / "0").mkdir(parents=True, exist_ok=True)
        (mask_root / str(cam_idx) / "1").mkdir(parents=True, exist_ok=True)
        (mask_root / f"mask_info_{cam_idx}.json").write_text(
            json.dumps({"0": controller_label, "1": object_label}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        Image.fromarray(cam_masks["controller"].astype(np.uint8) * 255).save(
            mask_root / str(cam_idx) / "0" / "frame_000000.png"
        )
        Image.fromarray(cam_masks["object"].astype(np.uint8) * 255).save(
            mask_root / str(cam_idx) / "1" / "frame_000000.png"
        )


def _write_overlays(replay_root: Path, output_root: Path, selected_masks: dict[int, dict[str, np.ndarray]]) -> None:
    overlay_root = output_root / "first_frame_overlay"
    overlay_root.mkdir(parents=True, exist_ok=True)
    for cam_idx, masks in selected_masks.items():
        image = Image.open(frame_path(replay_root, cam_idx, 0)).convert("RGB")
        base = np.asarray(image).astype(np.float32)
        overlay = base.copy()
        controller = _resize_mask_to_image(masks["controller"], base)
        obj = _resize_mask_to_image(masks["object"], base)
        overlay[controller] = 0.45 * overlay[controller] + 0.55 * np.array([255.0, 40.0, 40.0])
        overlay[obj] = 0.45 * overlay[obj] + 0.55 * np.array([20.0, 220.0, 255.0])
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(overlay_root / f"cam{cam_idx}_frame0_overlay.png")


def _resize_mask_to_image(mask: np.ndarray, image_arr: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    height, width = image_arr.shape[:2]
    if mask.shape == (height, width):
        return mask
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
    ) > 0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def render_frame0_init_report(summary: dict[str, Any]) -> str:
    rows = []
    for cam in summary.get("camera_summaries", []):
        rows.append(
            [
                cam.get("camera_idx"),
                cam.get("controller_candidate_count"),
                cam.get("controller_area"),
                cam.get("controller_selected_indices"),
                cam.get("object_candidate_count"),
                cam.get("object_area"),
                cam.get("object_selected_indices"),
            ]
        )
    return "\n".join(
        [
            "# SAM3.1 Frame0 Image Init Masks",
            "",
            f"- rgb_replay: `{summary.get('rgb_replay')}`",
            f"- output_dir: `{summary.get('output_dir')}`",
            f"- controller_prompt: `{summary.get('controller_prompt')}`",
            f"- object_prompt: `{summary.get('object_prompt')}`",
            f"- frame0_init_pass: `{summary.get('frame0_init_pass')}`",
            "",
            markdown_table(
                [
                    "camera",
                    "controller_candidates",
                    "controller_area",
                    "controller_indices",
                    "object_candidates",
                    "object_area",
                    "object_indices",
                ],
                rows,
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--qqtt-root", default=str(DEFAULT_QQTT_ROOT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="green towel on wooden table")
    parser.add_argument("--controller-label", default=None)
    parser.add_argument("--object-label", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--controller-selection-mode", choices=("green-score", "largest", "all"), default="green-score")
    parser.add_argument("--object-selection-mode", choices=("green-score", "largest", "all"), default="largest")
    parser.add_argument("--controller-max-instances", type=int, default=3)
    parser.add_argument("--object-max-instances", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=64)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    summary = generate_sam31_frame0_init_masks(
        rgb_replay=args.rgb_replay,
        output_dir=args.output_dir,
        qqtt_root=args.qqtt_root,
        checkpoint_path=args.checkpoint,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        controller_label=args.controller_label,
        object_label=args.object_label,
        overwrite=args.overwrite,
        compile_model=args.compile_model,
        confidence_threshold=args.confidence_threshold,
        controller_selection_mode=args.controller_selection_mode,
        object_selection_mode=args.object_selection_mode,
        controller_max_instances=args.controller_max_instances,
        object_max_instances=args.object_max_instances,
        min_area=args.min_area,
        fail_on_empty=not args.allow_empty,
        device=args.device,
    )
    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_md:
        write_markdown(args.output_md, render_frame0_init_report(summary))
    print(json.dumps({"output_dir": summary["output_dir"], "frame0_init_pass": summary["frame0_init_pass"]}, indent=2))
    return 0 if summary["frame0_init_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
