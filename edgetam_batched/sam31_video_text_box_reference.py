"""SAM3.1 video masks from separate text+box prompt sessions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import inspect_replay, load_manifest
from .sam31_frame0_init import default_sam31_frame0_init_mask_root
from .sam31_replay_reference import DEFAULT_QQTT_ROOT, build_sam31_replay_case


def default_sam31_text_box_mask_root(rgb_replay: str | Path) -> Path:
    return Path(rgb_replay).resolve() / "sam31_video_text_box_reference_masks"


def generate_sam31_video_text_box_masks(
    *,
    rgb_replay: str | Path,
    frame0_init_mask_root: str | Path,
    output_dir: str | Path | None = None,
    qqtt_root: str | Path = DEFAULT_QQTT_ROOT,
    checkpoint_path: str | Path | None = None,
    controller_prompt: str = "green towel on wooden table",
    object_prompt: str = "stuffed animal",
    controller_label: str = "towel",
    object_label: str = "stuffed animal",
    overwrite: bool = False,
    compile_model: bool = False,
    async_loading_frames: bool = False,
    max_num_objects: int = 16,
) -> dict[str, Any]:
    """Generate SAM3.1 video masks by running one session per role.

    SAM3.1 multiplex video resets state when a semantic text prompt is added.
    Running controller/object as separate sessions avoids losing the first text
    prompt, then this writer merges them back into a two-object sidecar tree.
    """

    replay_root = Path(rgb_replay).resolve()
    report = inspect_replay(replay_root)
    if not report["valid"]:
        raise FileNotFoundError(f"RGB replay is incomplete: {report['missing_sample'][:3]}")
    output_root = Path(output_dir).resolve() if output_dir is not None else default_sam31_text_box_mask_root(replay_root)
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

    from scripts.harness.sam31_mask_helper import (  # noqa: PLC0415
        ColorSource,
        _collect_frame_segments,
        _prepare_session_frames,
        _select_output_indices,
        build_sam31_video_predictor,
    )

    manifest = load_manifest(replay_root)
    case_root = build_sam31_replay_case(rgb_replay=replay_root, overwrite=False)
    camera_count = int(manifest["camera_count"])
    frame0_root = Path(frame0_init_mask_root).resolve()
    roles = [
        {"role_idx": 0, "label": controller_label, "prompt": controller_prompt},
        {"role_idx": 1, "label": object_label, "prompt": object_prompt},
    ]
    started = time.perf_counter()
    camera_summaries = []
    resolved_checkpoint = None
    for cam_idx in range(camera_count):
        source = ColorSource(
            camera_idx=cam_idx,
            mode="frames",
            path=case_root / "color" / str(cam_idx),
            frame_paths=sorted((case_root / "color" / str(cam_idx)).glob("*.png")),
        )
        predictor, resolved_checkpoint = build_sam31_video_predictor(
            checkpoint_path=checkpoint_path,
            async_loading_frames=async_loading_frames,
            compile_model=compile_model,
            max_num_objects=max_num_objects,
        )
        session_root = Path(tempfile.mkdtemp(prefix=f"sam31_text_box_cam{cam_idx}_"))
        frame_token_by_index = _prepare_session_frames(source, session_dir=session_root)
        (output_root / "mask" / str(cam_idx)).mkdir(parents=True, exist_ok=True)
        role_summaries = []
        try:
            for role in roles:
                role_idx = int(role["role_idx"])
                init_mask = _load_frame0_role_mask(frame0_root, cam_idx=cam_idx, role_idx=role_idx)
                box = mask_to_normalized_xywh(init_mask)
                session_id = predictor.handle_request(
                    {"type": "start_session", "resource_path": str(session_root)}
                )["session_id"]
                try:
                    response = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            "text": role["prompt"],
                            "bounding_boxes": [box],
                            "bounding_box_labels": [1],
                        }
                    )
                    outputs = response["outputs"]
                    selected_indices = set(_select_output_indices(outputs, keep_all_instances=False))
                    candidate_ids = np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64).reshape(-1)
                    selected_ids = {int(candidate_ids[idx]) for idx in selected_indices if idx < candidate_ids.size}
                    if not selected_ids and candidate_ids.size:
                        selected_ids = {int(candidate_ids[0])}
                    initial_segments = _collect_frame_segments(outputs, allowed_obj_ids=selected_ids)
                    video_segments: dict[int, dict[int, np.ndarray]] = {}
                    for stream_response in predictor.handle_stream_request(
                        {
                            "type": "propagate_in_video",
                            "session_id": session_id,
                            "start_frame_index": 0,
                            "propagation_direction": "forward",
                        }
                    ):
                        frame_idx = int(stream_response["frame_index"])
                        collected = _collect_frame_segments(stream_response["outputs"], allowed_obj_ids=selected_ids)
                        if collected:
                            # Merge any selected id into the fixed role id.
                            union = np.zeros_like(next(iter(collected.values())), dtype=bool)
                            for mask in collected.values():
                                union |= np.asarray(mask, dtype=bool)
                            video_segments[frame_idx] = {role_idx: union}
                    if 0 not in video_segments and initial_segments:
                        union = np.zeros_like(next(iter(initial_segments.values())), dtype=bool)
                        for mask in initial_segments.values():
                            union |= np.asarray(mask, dtype=bool)
                        video_segments[0] = {role_idx: union}
                    saved = _write_role_video_segments(
                        output_root=output_root,
                        camera_idx=cam_idx,
                        role_idx=role_idx,
                        frame_token_by_index=frame_token_by_index,
                        video_segments=video_segments,
                    )
                    role_summaries.append(
                        {
                            "role_idx": role_idx,
                            "label": role["label"],
                            "prompt": role["prompt"],
                            "box_xywh_norm": box,
                            "selected_source_ids": sorted(selected_ids),
                            "add_prompt_area": int(sum(np.asarray(mask, dtype=bool).sum() for mask in initial_segments.values())),
                            "saved_frame_count": int(saved["saved_frame_count"]),
                            "nonempty_frame_count": int(saved["nonempty_frame_count"]),
                            "frame0_area": int(saved["frame0_area"]),
                            "mean_area": float(saved["mean_area"]),
                        }
                    )
                finally:
                    predictor.handle_request(
                        {
                            "type": "close_session",
                            "session_id": session_id,
                            "run_gc_collect": True,
                        }
                    )
        finally:
            shutil.rmtree(session_root, ignore_errors=True)
            if hasattr(predictor, "shutdown"):
                predictor.shutdown()

        (output_root / "mask" / f"mask_info_{cam_idx}.json").write_text(
            json.dumps({str(role["role_idx"]): role["label"] for role in roles}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        camera_summaries.append(
            {
                "camera_idx": cam_idx,
                "frame_count": len(frame_token_by_index),
                "tracked_object_count": len(roles),
                "tracked_object_labels": {str(role["role_idx"]): role["label"] for role in roles},
                "role_summaries": role_summaries,
                "output_mask_root": str((output_root / "mask" / str(cam_idx)).resolve()),
            }
        )

    summary = {
        "rgb_replay": str(replay_root),
        "output_dir": str(output_root),
        "frame0_init_mask_root": str(frame0_root),
        "case_root": str(case_root),
        "checkpoint_path": resolved_checkpoint,
        "inference_mode": "sam31-video-text-box-separate-sessions",
        "controller_prompt": controller_prompt,
        "object_prompt": object_prompt,
        "sam31_reference_role_order": [controller_label, object_label],
        "camera_ids": list(range(camera_count)),
        "camera_summaries": camera_summaries,
        "generation_wall_ms": float((time.perf_counter() - started) * 1000.0),
        "reused_existing": False,
    }
    write_json(summary_path, summary)
    write_markdown(output_root / "summary.md", render_text_box_reference_report(summary))
    return summary


def _load_frame0_role_mask(root: Path, *, cam_idx: int, role_idx: int) -> np.ndarray:
    path = root / "mask" / str(int(cam_idx)) / str(int(role_idx)) / "frame_000000.png"
    if not path.exists():
        raise FileNotFoundError(f"missing frame0 init mask: {path}")
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if not np.count_nonzero(mask):
        raise ValueError(f"empty frame0 init mask: {path}")
    return mask


def mask_to_normalized_xywh(mask: np.ndarray) -> list[float]:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("cannot compute box for an empty mask")
    height, width = mask.shape
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [
        float(x0 / width),
        float(y0 / height),
        float((x1 - x0) / width),
        float((y1 - y0) / height),
    ]


def _write_role_video_segments(
    *,
    output_root: Path,
    camera_idx: int,
    role_idx: int,
    frame_token_by_index: dict[int, str],
    video_segments: dict[int, dict[int, np.ndarray]],
) -> dict[str, Any]:
    if not video_segments:
        raise RuntimeError(f"no SAM3.1 video segments for cam{camera_idx} obj{role_idx}")
    empty_shape = np.asarray(next(iter(next(iter(video_segments.values())).values())), dtype=bool).shape
    areas = []
    for frame_idx in sorted(frame_token_by_index):
        frame_token = frame_token_by_index[frame_idx]
        mask = video_segments.get(frame_idx, {}).get(role_idx)
        if mask is None:
            mask = np.zeros(empty_shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        areas.append(int(mask.sum()))
        output_path = output_root / "mask" / str(int(camera_idx)) / str(int(role_idx)) / f"{frame_token}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_path)
    return {
        "saved_frame_count": len(areas),
        "nonempty_frame_count": sum(area > 0 for area in areas),
        "frame0_area": areas[0] if areas else 0,
        "mean_area": float(sum(areas) / len(areas)) if areas else 0.0,
    }


def render_text_box_reference_report(summary: dict[str, Any]) -> str:
    rows = []
    for cam in summary.get("camera_summaries", []):
        for role in cam.get("role_summaries", []):
            rows.append(
                [
                    cam.get("camera_idx"),
                    role.get("role_idx"),
                    role.get("label"),
                    role.get("prompt"),
                    role.get("frame0_area"),
                    role.get("nonempty_frame_count"),
                    role.get("mean_area"),
                ]
            )
    return "\n".join(
        [
            "# SAM3.1 Video Text+Box Reference",
            "",
            f"- rgb_replay: `{summary.get('rgb_replay')}`",
            f"- output_dir: `{summary.get('output_dir')}`",
            f"- frame0_init_mask_root: `{summary.get('frame0_init_mask_root')}`",
            f"- inference_mode: `{summary.get('inference_mode')}`",
            "",
            markdown_table(
                ["camera", "obj", "label", "prompt", "frame0_area", "nonempty_frames", "mean_area"],
                rows,
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--frame0-init-mask-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--qqtt-root", default=str(DEFAULT_QQTT_ROOT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--controller-prompt", default="green towel on wooden table")
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-label", default="towel")
    parser.add_argument("--object-label", default="stuffed animal")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--async-loading-frames", action="store_true")
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    frame0_root = args.frame0_init_mask_root or default_sam31_frame0_init_mask_root(args.rgb_replay)
    summary = generate_sam31_video_text_box_masks(
        rgb_replay=args.rgb_replay,
        frame0_init_mask_root=frame0_root,
        output_dir=args.output_dir,
        qqtt_root=args.qqtt_root,
        checkpoint_path=args.checkpoint,
        controller_prompt=args.controller_prompt,
        object_prompt=args.object_prompt,
        controller_label=args.controller_label,
        object_label=args.object_label,
        overwrite=args.overwrite,
        compile_model=args.compile_model,
        async_loading_frames=args.async_loading_frames,
        max_num_objects=args.max_num_objects,
    )
    if args.output_json:
        write_json(args.output_json, summary)
    if args.output_md:
        write_markdown(args.output_md, render_text_box_reference_report(summary))
    print(json.dumps({"output_dir": summary["output_dir"], "reused_existing": summary["reused_existing"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
