"""Reference runtime for original HF public EdgeTAM video API."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .report_utils import write_json, write_markdown
from .rgb_replay import ReplayFrame, load_manifest, load_replay_frames
from .stats import summarize


@dataclass
class ReferenceRuntimeConfig:
    model_id: str = "yonigozlan/EdgeTAM-hf"
    dtype: str = "bfloat16"
    device: str = "cuda"
    object_prompt: str = "stuffed animal"
    controller_prompt: str = "towel"
    object_count: int = 2


@dataclass
class RuntimeOutputs:
    masks: list[list[Any]]
    logits: list[list[Any]]
    object_scores: list[list[Any]]
    timings_ms: dict[str, Any]
    backend: str
    partial: bool = False
    fallback_backend: str | None = None
    blockers: list[str] | None = None
    component_timings_ms: dict[str, Any] | None = None


class HfEdgeTamReferenceRuntime:
    def __init__(self, config: ReferenceRuntimeConfig | None = None):
        self.config = config or ReferenceRuntimeConfig()
        self.model = None
        self.processor = None
        self.sessions: list[Any] = []
        self.torch = None
        self.dtype = None

    def load(self) -> None:
        import torch
        from transformers import EdgeTamVideoModel, Sam2VideoProcessor

        self.torch = torch
        self.dtype = torch_dtype(torch, self.config.dtype)
        started = time.perf_counter()
        self.model = EdgeTamVideoModel.from_pretrained(self.config.model_id).to(
            device=self.config.device,
            dtype=self.dtype,
        )
        self.model.eval()
        self.processor = Sam2VideoProcessor.from_pretrained(self.config.model_id)
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def init_sessions(
        self,
        first_frame: ReplayFrame,
        *,
        initial_masks_by_camera: list[tuple[np.ndarray, ...]] | None = None,
    ) -> list[Any]:
        if self.model is None or self.processor is None or self.torch is None:
            raise RuntimeError("call load() before init_sessions()")
        from transformers import EdgeTamVideoInferenceSession

        width, height = first_frame.images[0].size
        object_count = int(self.config.object_count)
        if object_count not in {1, 2}:
            raise ValueError(f"unsupported object_count: {object_count}")
        obj_ids = list(range(1, object_count + 1))
        controller_mask, object_mask = make_initial_prompt_masks(height, width)
        self.initial_masks = (controller_mask, object_mask)
        if initial_masks_by_camera is None:
            if object_count == 1:
                self.initial_masks_by_camera = [(object_mask.copy(),) for _ in range(len(first_frame.images))]
            else:
                self.initial_masks_by_camera = [
                    (controller_mask.copy(), object_mask.copy()) for _ in range(len(first_frame.images))
                ]
            self.prompt_source = "deterministic_replay_boxes"
        else:
            if len(initial_masks_by_camera) != len(first_frame.images):
                raise ValueError(
                    f"expected {len(first_frame.images)} camera initial mask pairs, "
                    f"got {len(initial_masks_by_camera)}"
                )
            self.initial_masks_by_camera = []
            for masks in initial_masks_by_camera:
                normalized = tuple(np.asarray(mask, dtype=bool) for mask in masks)
                if object_count == 1:
                    self.initial_masks_by_camera.append((_select_single_object_mask(normalized),))
                else:
                    if len(normalized) != 2:
                        raise ValueError(f"expected controller/object mask pair, got {len(normalized)} masks")
                    self.initial_masks_by_camera.append(normalized)
            self.prompt_source = "sam31_frame0_video_reference_masks"
        self.sessions = []
        for camera_idx in range(len(first_frame.images)):
            cam_masks = self.initial_masks_by_camera[camera_idx]
            session = EdgeTamVideoInferenceSession(
                video=None,
                video_height=height,
                video_width=width,
                inference_device=self.config.device,
                inference_state_device=self.config.device,
                video_storage_device=self.config.device,
                dtype=self.dtype,
            )
            self.processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=0,
                obj_ids=list(obj_ids),
                input_masks=[mask.copy() for mask in cam_masks],
            )
            self.sessions.append(session)
        return self.sessions

    def preprocess_frames(self, frame: ReplayFrame):
        assert self.processor is not None
        started = time.perf_counter()
        inputs = self.processor(images=frame.images, device=self.config.device, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device=self.config.device, dtype=self.dtype)
        return pixel_values, (time.perf_counter() - started) * 1000.0

    def step_public(self, frame: ReplayFrame) -> tuple[list[Any], list[Any], list[Any], dict[str, float]]:
        if self.model is None or not self.sessions or self.torch is None:
            raise RuntimeError("runtime is not initialized")
        pixel_values, preprocess_ms = self.preprocess_frames(frame)
        masks = []
        logits = []
        scores = []
        cam_times = {}
        stage_started = time.perf_counter()
        with self.torch.inference_mode(), autocast_context(self.torch, self.config.device, self.dtype):
            for cam_idx, session in enumerate(self.sessions):
                started = time.perf_counter()
                output = self.model(
                    inference_session=session,
                    frame_idx=int(frame.frame_idx),
                    frame=pixel_values[cam_idx],
                )
                if str(self.config.device).startswith("cuda"):
                    self.torch.cuda.synchronize()
                cam_times[f"cam{cam_idx}_ms"] = (time.perf_counter() - started) * 1000.0
                logits_tensor = output.pred_masks.detach()
                masks.append((logits_tensor > 0).detach().cpu().numpy().astype(bool))
                logits.append(logits_tensor.detach().float().cpu().numpy())
                score = getattr(output, "object_score_logits", None)
                if score is None:
                    scores.append(np.zeros((logits_tensor.shape[0],), dtype=np.float32))
                else:
                    scores.append(score.detach().float().cpu().numpy())
        timings = {
            "preprocess_ms": preprocess_ms,
            "stage_wall_ms": (time.perf_counter() - stage_started) * 1000.0,
            **cam_times,
        }
        return masks, logits, scores, timings


def run_reference(
    *,
    rgb_replay: str | Path,
    frames: int,
    config: ReferenceRuntimeConfig,
) -> RuntimeOutputs:
    replay_frames = load_replay_frames(rgb_replay, frames)
    runtime = HfEdgeTamReferenceRuntime(config)
    runtime.load()
    runtime.init_sessions(replay_frames[0])
    all_masks = []
    all_logits = []
    all_scores = []
    timing_rows = []
    for frame in replay_frames:
        masks, logits, scores, timings = runtime.step_public(frame)
        all_masks.append(masks)
        all_logits.append(logits)
        all_scores.append(scores)
        timing_rows.append(timings)
    return RuntimeOutputs(
        masks=all_masks,
        logits=all_logits,
        object_scores=all_scores,
        timings_ms=summarize_timing_rows(timing_rows),
        backend="hf_ref_seq_public",
    )


def save_reference_outputs(output_dir: str | Path, outputs: RuntimeOutputs) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame_masks in enumerate(outputs.masks):
        for cam_idx, cam_masks in enumerate(frame_masks):
            cam_dir = root / f"cam{cam_idx}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cam_dir / f"frame_{frame_idx:06d}_masks.npz", masks=cam_masks)


def make_initial_prompt_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    controller = np.zeros((height, width), dtype=bool)
    obj = np.zeros((height, width), dtype=bool)
    controller[int(height * 0.15) : int(height * 0.32), int(width * 0.16) : int(width * 0.34)] = True
    obj[int(height * 0.34) : int(height * 0.66), int(width * 0.35) : int(width * 0.65)] = True
    return controller, obj


def _select_single_object_mask(masks: tuple[np.ndarray, ...]) -> np.ndarray:
    if not masks:
        raise ValueError("cannot initialize single-object EdgeTAM without an object mask")
    if len(masks) == 1:
        return masks[0]
    # Existing two-object mask roots use obj0=controller and obj1=stuffed animal.
    return masks[1]


def summarize_timing_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted({key for row in rows for key in row})
    return {key: summarize(row[key] for row in rows if key in row) for key in keys}


def torch_dtype(torch_module, name: str):
    lowered = name.lower().replace("torch.", "")
    if lowered in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch_module.float16
    if lowered in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"unsupported dtype: {name}")


class autocast_context:
    def __init__(self, torch_module, device: str, dtype):
        self.torch = torch_module
        self.device = device
        self.dtype = dtype
        self.ctx = None

    def __enter__(self):
        if str(self.device).startswith("cuda"):
            self.ctx = self.torch.autocast("cuda", dtype=self.dtype)
            return self.ctx.__enter__()
        return None

    def __exit__(self, exc_type, exc, tb):
        if self.ctx is not None:
            return self.ctx.__exit__(exc_type, exc, tb)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="towel")
    parser.add_argument("--object-count", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.rgb_replay)
    config = ReferenceRuntimeConfig(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        object_count=args.object_count,
    )
    outputs = run_reference(rgb_replay=args.rgb_replay, frames=args.frames, config=config)
    save_reference_outputs(args.output_dir, outputs)
    nonempty = {}
    for cam_idx in range(int(manifest["camera_count"])):
        count = 0
        for frame_masks in outputs.masks:
            count += int(np.count_nonzero(frame_masks[cam_idx]) > 0)
        nonempty[f"cam{cam_idx}"] = count
    payload = {
        "reference_pass": True,
        "rgb_replay": str(args.rgb_replay),
        "frames_processed": len(outputs.masks),
        "mask_nonempty_count": nonempty,
        "timings_ms": outputs.timings_ms,
        "prompt_source": "deterministic_replay_boxes",
        "object_prompt": args.object_prompt,
        "controller_prompt": args.controller_prompt,
    }
    write_json(args.output_json, payload)
    write_markdown(
        Path(args.output_json).with_suffix(".md"),
        "# EdgeTAM Reference Runtime\n\n"
        f"- reference_pass: `{payload['reference_pass']}`\n"
        f"- frames_processed: `{payload['frames_processed']}`\n"
        "- prompt_source: `deterministic_replay_boxes`\n",
    )
    if args.debug:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
