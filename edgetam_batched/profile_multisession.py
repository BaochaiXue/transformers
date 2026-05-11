"""Profile scaffolding for batched multi-session EdgeTAM experiments."""

from __future__ import annotations

import argparse
import time

from .config import BACKENDS
from .report_utils import write_json, write_markdown
from .stats import summarize


def run_scaffold_profile(frames: int, warmup: int) -> dict:
    timings = []
    for idx in range(frames + warmup):
        start = time.perf_counter()
        _ = idx * idx
        elapsed = (time.perf_counter() - start) * 1000.0
        if idx >= warmup:
            timings.append(elapsed)
    return {
        "stage_wall_ms": summarize(timings),
        "note": "scaffold-only timing; does not measure EdgeTAM model inference",
    }


def run_synthetic_hf_public_profile(args: argparse.Namespace) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from transformers import EdgeTamVideoInferenceSession, EdgeTamVideoModel, Sam2VideoProcessor

    dtype = _torch_dtype(torch, args.dtype)
    device = args.device
    model_start = time.perf_counter()
    model = EdgeTamVideoModel.from_pretrained(args.model_id).to(device=device, dtype=dtype).eval()
    processor = Sam2VideoProcessor.from_pretrained(args.model_id)
    model_load_ms = (time.perf_counter() - model_start) * 1000.0

    image_np = np.zeros((args.synthetic_height, args.synthetic_width, 3), dtype=np.uint8)
    image = Image.fromarray(image_np, mode="RGB")
    object_mask = np.zeros((args.synthetic_height, args.synthetic_width), dtype=bool)
    object_mask[
        int(args.synthetic_height * 0.34) : int(args.synthetic_height * 0.66),
        int(args.synthetic_width * 0.35) : int(args.synthetic_width * 0.65),
    ] = True
    controller_mask = np.zeros((args.synthetic_height, args.synthetic_width), dtype=bool)
    controller_mask[
        int(args.synthetic_height * 0.15) : int(args.synthetic_height * 0.32),
        int(args.synthetic_width * 0.16) : int(args.synthetic_width * 0.34),
    ] = True
    obj_ids = [1, 2]

    sessions = [
        EdgeTamVideoInferenceSession(
            video=None,
            video_height=args.synthetic_height,
            video_width=args.synthetic_width,
            inference_device=device,
            inference_state_device=device,
            video_storage_device=device,
            dtype=dtype,
        )
        for _ in range(3)
    ]

    preprocess_start = time.perf_counter()
    inputs = processor(images=image, device=device, return_tensors="pt")
    pixel_values = inputs.pixel_values[0].to(device=device, dtype=dtype)
    pixel_values_by_cam = [pixel_values.clone() for _ in range(3)]
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

    prompt_start = time.perf_counter()
    for session in sessions:
        processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=list(obj_ids),
            input_masks=[controller_mask.copy(), object_mask.copy()],
        )
    prompt_ms = (time.perf_counter() - prompt_start) * 1000.0

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    stage_times: list[float] = []
    per_cam_times: dict[str, list[float]] = {f"cam{idx}": [] for idx in range(3)}
    output_shape = None
    with torch.inference_mode():
        autocast = torch.autocast("cuda", dtype=dtype) if device.startswith("cuda") else _nullcontext()
        with autocast:
            for idx in range(args.frames + args.warmup):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                stage_start = time.perf_counter()
                for cam_idx, session in enumerate(sessions):
                    cam_start = time.perf_counter()
                    output = model(
                        inference_session=session,
                        frame_idx=idx,
                        frame=pixel_values_by_cam[cam_idx],
                    )
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - cam_start) * 1000.0
                    if idx >= args.warmup:
                        per_cam_times[f"cam{cam_idx}"].append(elapsed)
                    if output_shape is None:
                        output_shape = list(output.pred_masks.shape)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                if idx >= args.warmup:
                    stage_times.append((time.perf_counter() - stage_start) * 1000.0)

    return {
        "model_load_ms": model_load_ms,
        "preprocess_ms": preprocess_ms,
        "prompt_ms": prompt_ms,
        "stage_wall_ms": summarize(stage_times),
        "per_camera_model_ms": {cam: summarize(values) for cam, values in per_cam_times.items()},
        "output_pred_masks_shape": output_shape,
        "note": "synthetic HF public API sequential three-session profile; not batched multi-session runtime",
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _torch_dtype(torch_module, name: str):
    lowered = name.lower().replace("torch.", "")
    if lowered in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch_module.float16
    if lowered in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--object-count", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--gpu-sampling", action="store_true")
    parser.add_argument("--profile-cuda-events", action="store_true")
    parser.add_argument("--profile-nvtx", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--synthetic-hf-public", action="store_true")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-width", type=int, default=848)
    parser.add_argument("--synthetic-height", type=int, default=480)
    args = parser.parse_args()

    if args.synthetic_hf_public:
        profile = run_synthetic_hf_public_profile(args)
        status = "synthetic_hf_public"
    else:
        profile = run_scaffold_profile(args.frames, args.warmup)
        status = "scaffold_only"
    payload = {
        "backend": args.backend,
        "frames": args.frames,
        "warmup": args.warmup,
        "compile_mode": args.compile_mode,
        "graph_output_policy": args.graph_output_policy,
        "profile": profile,
        "correctness_report_used": None,
        "correctness_pass": False,
        "status": status,
    }
    write_json(args.output_json, payload)
    write_markdown(
        args.output_md,
        "\n".join(
            [
                "# EdgeTAM Batched Profile",
                "",
                f"- backend: `{args.backend}`",
                f"- status: `{payload['status']}`",
                "- this profile is not a validated batched multi-session runtime measurement",
                "",
                "Real performance requires the reference correctness harness and RGB replay.",
            ]
        ),
    )
    if args.debug:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
