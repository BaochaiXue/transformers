"""RGB triplet replay creation and loading utilities."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .report_utils import write_json


@dataclass(frozen=True)
class ReplayFrame:
    frame_idx: int
    images: list[Image.Image]


def load_manifest(replay_root: str | Path) -> dict:
    root = Path(replay_root)
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing RGB replay manifest: {manifest}")
    return json.loads(manifest.read_text(encoding="utf-8"))


def inspect_replay(replay_root: str | Path) -> dict[str, Any]:
    root = Path(replay_root)
    manifest = load_manifest(root)
    camera_count = int(manifest["camera_count"])
    frame_count = int(manifest["frame_count"])
    missing: list[str] = []
    for cam_idx in range(camera_count):
        for frame_idx in range(frame_count):
            path = frame_path(root, cam_idx, frame_idx)
            if not path.exists():
                missing.append(str(path))
    return {
        "path": str(root),
        "manifest": manifest,
        "camera_count": camera_count,
        "frame_count": frame_count,
        "missing_count": len(missing),
        "missing_sample": missing[:20],
        "valid": len(missing) == 0,
    }


def load_replay_frames(replay_root: str | Path, frames: int | None = None) -> list[ReplayFrame]:
    root = Path(replay_root)
    report = inspect_replay(root)
    if not report["valid"]:
        raise FileNotFoundError(f"RGB replay is incomplete: {report['missing_sample'][:3]}")
    manifest = report["manifest"]
    frame_count = int(manifest["frame_count"])
    camera_count = int(manifest["camera_count"])
    limit = frame_count if frames is None else min(int(frames), frame_count)
    output = []
    for frame_idx in range(limit):
        images = [Image.open(frame_path(root, cam_idx, frame_idx)).convert("RGB") for cam_idx in range(camera_count)]
        output.append(ReplayFrame(frame_idx=frame_idx, images=images))
    return output


def frame_path(root: Path, camera_idx: int, frame_idx: int) -> Path:
    return root / f"cam{int(camera_idx)}" / f"frame_{int(frame_idx):06d}.png"


def record_realsense_triplet(
    *,
    out_dir: str | Path,
    frames: int,
    camera_count: int,
    width: int,
    height: int,
    fps: int,
    object_prompt: str,
    controller_prompt: str,
    debug: bool = False,
) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for cam_idx in range(camera_count):
        (root / f"cam{cam_idx}").mkdir(parents=True, exist_ok=True)

    qqtt_root = Path("/home/zhangxinjie/proj-QQTT-v2")
    if str(qqtt_root) not in sys.path:
        sys.path.insert(0, str(qqtt_root))

    try:
        import cv2
        from qqtt.env.camera import CameraSystem
    except Exception as exc:
        raise RuntimeError(f"failed to import QQTT CameraSystem/cv2: {exc}") from exc

    started = time.perf_counter()
    system = CameraSystem(
        WH=(int(width), int(height)),
        fps=int(fps),
        num_cam=int(camera_count),
        capture_mode="stereo_ir",
        enable_keyboard_listener=False,
    )
    written = 0
    frame_records = []
    try:
        last_steps: tuple[int, ...] | None = None
        while written < int(frames):
            obs = system.get_observation()
            steps = tuple(int(obs[cam_idx]["step_idx"]) for cam_idx in range(camera_count))
            if steps == last_steps:
                time.sleep(0.002)
                continue
            last_steps = steps
            for cam_idx in range(camera_count):
                color = obs[cam_idx]["color"]
                cv2.imwrite(str(frame_path(root, cam_idx, written)), color)
            frame_records.append(
                {
                    "frame_idx": written,
                    "camera_steps": list(steps),
                    "timestamps": [float(obs[cam_idx]["timestamp"]) for cam_idx in range(camera_count)],
                }
            )
            written += 1
            if debug and (written == 1 or written % 10 == 0):
                print(f"[rgb_replay] recorded {written}/{frames} frame groups", flush=True)
    finally:
        try:
            system.realsense.stop()
        except Exception:
            pass
        try:
            system.shm_manager.shutdown()
        except Exception:
            pass

    manifest = {
        "camera_count": int(camera_count),
        "frame_count": int(written),
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "object_prompt": object_prompt,
        "controller_prompt": controller_prompt,
        "source": "realsense",
        "elapsed_s": time.perf_counter() - started,
    }
    write_json(root / "manifest.json", manifest)
    frames_jsonl = root / "frames.jsonl"
    frames_jsonl.write_text("\n".join(json.dumps(row, sort_keys=True) for row in frame_records) + "\n", encoding="utf-8")
    return inspect_replay(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "record-realsense-triplet"), required=True)
    parser.add_argument("--rgb-replay")
    parser.add_argument("--out-dir")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="towel")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mode == "inspect":
        if not args.rgb_replay:
            parser.error("--rgb-replay is required for inspect")
        report = inspect_replay(args.rgb_replay)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 2

    if not args.out_dir:
        parser.error("--out-dir is required for record-realsense-triplet")
    report = record_realsense_triplet(
        out_dir=args.out_dir,
        frames=args.frames,
        camera_count=args.camera_count,
        width=args.width,
        height=args.height,
        fps=args.fps,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        debug=args.debug,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
