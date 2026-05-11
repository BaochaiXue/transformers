"""SAM3.1 replay reference masks for EdgeTAM batched comparisons."""

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

from .reference_runtime import RuntimeOutputs, summarize_timing_rows
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import frame_path, inspect_replay, load_manifest, load_replay_frames


DEFAULT_QQTT_ROOT = Path("/home/zhangxinjie/proj-QQTT-v2")


def default_sam31_mask_root(rgb_replay: str | Path) -> Path:
    return Path(rgb_replay).resolve() / "sam31_video_reference_masks"


def default_sam31_case_root(rgb_replay: str | Path) -> Path:
    return Path(rgb_replay).resolve() / "_sam31_case"


def build_sam31_replay_case(
    *,
    rgb_replay: str | Path,
    case_root: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Create a QQTT case-like color/ tree from a replay folder."""

    replay_root = Path(rgb_replay).resolve()
    report = inspect_replay(replay_root)
    if not report["valid"]:
        raise FileNotFoundError(f"RGB replay is incomplete: {report['missing_sample'][:3]}")
    manifest = report["manifest"]
    camera_count = int(manifest["camera_count"])
    frame_count = int(manifest["frame_count"])
    root = Path(case_root).resolve() if case_root is not None else default_sam31_case_root(replay_root)
    color_root = root / "color"
    if color_root.exists() and overwrite:
        shutil.rmtree(color_root)
    color_root.mkdir(parents=True, exist_ok=True)

    for cam_idx in range(camera_count):
        cam_dir = color_root / str(cam_idx)
        if cam_dir.exists() and overwrite:
            shutil.rmtree(cam_dir)
        cam_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(frame_count):
            src = frame_path(replay_root, cam_idx, frame_idx)
            dst = cam_dir / src.name
            if dst.exists():
                continue
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)

    (root / "sam31_replay_case_manifest.json").write_text(
        json.dumps({"rgb_replay": str(replay_root), "manifest": manifest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def generate_sam31_replay_masks(
    *,
    rgb_replay: str | Path,
    output_dir: str | Path | None = None,
    qqtt_root: str | Path = DEFAULT_QQTT_ROOT,
    checkpoint_path: str | Path | None = None,
    object_prompt: str = "stuffed animal",
    controller_prompt: str = "towel",
    overwrite: bool = False,
    compile_model: bool = False,
    async_loading_frames: bool = False,
    max_num_objects: int = 16,
) -> dict[str, Any]:
    """Generate SAM3.1 video masks for the replay using QQTT's helper."""

    replay_root = Path(rgb_replay).resolve()
    manifest = load_manifest(replay_root)
    output_root = Path(output_dir).resolve() if output_dir is not None else default_sam31_mask_root(replay_root)
    summary_path = output_root / "summary.json"
    if summary_path.exists() and not overwrite:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["reused_existing"] = True
        return summary

    qqtt_root = Path(qqtt_root).resolve()
    if str(qqtt_root) not in sys.path:
        sys.path.insert(0, str(qqtt_root))
    from scripts.harness.sam31_mask_helper import run_case_segmentation  # noqa: PLC0415

    case_root = build_sam31_replay_case(rgb_replay=replay_root, overwrite=overwrite)
    camera_ids = list(range(int(manifest["camera_count"])))
    # Keep controller first to match the EdgeTAM session object order used in this fork:
    # obj0 = controller, obj1 = object.
    text_prompt = f"{controller_prompt}.{object_prompt}"
    started = time.perf_counter()
    result = run_case_segmentation(
        case_root=case_root,
        text_prompt=text_prompt,
        camera_ids=camera_ids,
        output_dir=output_root,
        source_mode="frames",
        checkpoint_path=checkpoint_path,
        ann_frame_index=0,
        keep_session_frames=False,
        overwrite=True,
        async_loading_frames=async_loading_frames,
        compile_model=compile_model,
        max_num_objects=max_num_objects,
    )
    result.update(
        {
            "rgb_replay": str(replay_root),
            "sam31_reference_role_order": [controller_prompt, object_prompt],
            "controller_prompt": controller_prompt,
            "object_prompt": object_prompt,
            "generation_wall_ms": (time.perf_counter() - started) * 1000.0,
            "reused_existing": False,
        }
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


def load_sam31_reference_outputs(
    *,
    rgb_replay: str | Path,
    mask_root: str | Path,
    frames: int,
    object_prompt: str = "stuffed animal",
    controller_prompt: str = "towel",
) -> RuntimeOutputs:
    """Load SAM3.1 sidecar masks into RuntimeOutputs-compatible shape."""

    replay_frames = load_replay_frames(rgb_replay, frames)
    root = Path(mask_root).resolve()
    role_order = [controller_prompt, object_prompt]
    masks: list[list[Any]] = []
    logits: list[list[Any]] = []
    scores: list[list[Any]] = []
    timing_rows: list[dict[str, float]] = []

    if not replay_frames:
        raise ValueError("RGB replay contains no frames")
    width, height = replay_frames[0].images[0].size

    label_maps = _load_label_maps(root, camera_count=len(replay_frames[0].images))
    for replay_frame in replay_frames:
        frame_masks_by_cam = []
        frame_logits_by_cam = []
        frame_scores_by_cam = []
        for cam_idx, image in enumerate(replay_frame.images):
            image_width, image_height = image.size
            per_role = []
            for role in role_order:
                per_role.append(
                    _load_union_mask_for_label(
                        root,
                        camera_idx=cam_idx,
                        frame_token=f"frame_{int(replay_frame.frame_idx):06d}",
                        label=role,
                        label_map=label_maps.get(cam_idx, {}),
                        height=image_height,
                        width=image_width,
                    )
                )
            mask_arr = np.stack(per_role, axis=0).astype(bool)
            frame_masks_by_cam.append(mask_arr)
            frame_logits_by_cam.append(mask_arr.astype(np.float32))
            frame_scores_by_cam.append(np.ones((len(role_order),), dtype=np.float32))
        masks.append(frame_masks_by_cam)
        logits.append(frame_logits_by_cam)
        scores.append(frame_scores_by_cam)
        timing_rows.append({"sam31_load_reference_ms": 0.0})

    return RuntimeOutputs(
        masks=masks,
        logits=logits,
        object_scores=scores,
        timings_ms=summarize_timing_rows(timing_rows),
        backend="sam31_video_replay_ref",
    )


def initial_masks_by_camera_from_sam31(
    *,
    rgb_replay: str | Path,
    mask_root: str | Path,
    object_prompt: str = "stuffed animal",
    controller_prompt: str = "towel",
) -> list[tuple[np.ndarray, np.ndarray]]:
    reference = load_sam31_reference_outputs(
        rgb_replay=rgb_replay,
        mask_root=mask_root,
        frames=1,
        object_prompt=object_prompt,
        controller_prompt=controller_prompt,
    )
    output = []
    for cam_masks in reference.masks[0]:
        output.append((cam_masks[0].copy(), cam_masks[1].copy()))
    return output


def _load_label_maps(root: Path, *, camera_count: int) -> dict[int, dict[int, str]]:
    label_maps: dict[int, dict[int, str]] = {}
    for cam_idx in range(camera_count):
        path = root / "mask" / f"mask_info_{cam_idx}.json"
        if not path.exists():
            label_maps[cam_idx] = {}
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        label_maps[cam_idx] = {int(obj_id): str(label) for obj_id, label in raw.items()}
    return label_maps


def _load_union_mask_for_label(
    root: Path,
    *,
    camera_idx: int,
    frame_token: str,
    label: str,
    label_map: dict[int, str],
    height: int,
    width: int,
) -> np.ndarray:
    matches = [obj_id for obj_id, obj_label in label_map.items() if obj_label == label]
    union = np.zeros((height, width), dtype=bool)
    for obj_id in matches:
        path = root / "mask" / str(int(camera_idx)) / str(int(obj_id)) / f"{frame_token}.png"
        if not path.exists():
            continue
        arr = np.asarray(Image.open(path).convert("L")) > 0
        if arr.shape != union.shape:
            arr = np.asarray(Image.fromarray(arr.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)) > 0
        union |= arr
    return union


def render_sam31_reference_report(payload: dict[str, Any]) -> str:
    rows = []
    for summary in payload.get("camera_summaries", []):
        rows.append(
            [
                summary.get("camera_idx"),
                summary.get("frame_count"),
                summary.get("saved_frame_count"),
                summary.get("tracked_object_count"),
                summary.get("tracked_object_labels"),
            ]
        )
    return "\n".join(
        [
            "# SAM3.1 Replay Reference",
            "",
            f"- rgb_replay: `{payload.get('rgb_replay')}`",
            f"- output_dir: `{payload.get('output_dir')}`",
            f"- text_prompt: `{payload.get('text_prompt')}`",
            f"- role_order: `{payload.get('sam31_reference_role_order')}`",
            f"- reused_existing: `{payload.get('reused_existing')}`",
            "",
            markdown_table(["camera", "frames", "saved_frames", "objects", "labels"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--qqtt-root", default=str(DEFAULT_QQTT_ROOT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="towel")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--async-loading-frames", action="store_true")
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payload = generate_sam31_replay_masks(
        rgb_replay=args.rgb_replay,
        output_dir=args.output_dir,
        qqtt_root=args.qqtt_root,
        checkpoint_path=args.checkpoint,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        overwrite=args.overwrite,
        compile_model=args.compile_model,
        async_loading_frames=args.async_loading_frames,
        max_num_objects=args.max_num_objects,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_sam31_reference_report(payload))
    print(json.dumps({"output_dir": payload.get("output_dir"), "reused_existing": payload.get("reused_existing")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
