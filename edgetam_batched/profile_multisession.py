"""Profile scaffolding for batched multi-session EdgeTAM experiments."""

from __future__ import annotations

import argparse
import time

from .backend_contract import FullBatchedContractError, contract_for_current_runtime
from .config import BACKENDS
from .precision_policy import PRECISION_POLICY_NAMES, reference_dtype_for_precision_mode
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
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="towel")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--precision-mode", choices=PRECISION_POLICY_NAMES, default="all_bf16")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--component-runtime", choices=("torch", "trt"), default="torch")
    parser.add_argument("--trt-engine-dir", default=None)
    parser.add_argument("--trt-memory-attention-bucket-dir", default=None)
    parser.add_argument("--trt-scope", default="memory_path_all")
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument(
        "--init-source",
        choices=("deterministic", "sam31-image-frame0"),
        default="deterministic",
    )
    parser.add_argument("--sam31-checkpoint", default=None)
    parser.add_argument("--sam31-compile-model", action="store_true")
    parser.add_argument("--sam31-frame0-init-mask-root", default=None)
    parser.add_argument("--sam31-frame0-init-overwrite", action="store_true")
    parser.add_argument("--sam31-frame0-controller-prompt", default=None)
    parser.add_argument("--sam31-frame0-object-prompt", default=None)
    parser.add_argument("--sam31-frame0-confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--sam31-frame0-controller-selection-mode",
        choices=("green-score", "largest", "all"),
        default="green-score",
    )
    parser.add_argument(
        "--sam31-frame0-object-selection-mode",
        choices=("green-score", "largest", "all"),
        default="largest",
    )
    parser.add_argument("--sam31-frame0-controller-max-instances", type=int, default=3)
    parser.add_argument("--sam31-frame0-object-max-instances", type=int, default=1)
    parser.add_argument("--sam31-frame0-min-area", type=int, default=64)
    parser.add_argument("--sam31-frame0-allow-empty", action="store_true")
    parser.add_argument("--qqtt-root", default="/home/zhangxinjie/proj-QQTT-v2")
    parser.add_argument("--gpu-sampling", action="store_true")
    parser.add_argument("--profile-cuda-events", action="store_true")
    parser.add_argument("--profile-nvtx", action="store_true")
    parser.add_argument("--strict-full-batched", action="store_true")
    parser.add_argument("--disallow-partial-backend-success", action="store_true")
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
    elif args.rgb_replay:
        try:
            profile = run_replay_profile(args)
            status = "replay_profile"
        except FullBatchedContractError as exc:
            contract = contract_for_current_runtime(
                backend=args.backend,
                batch_vision=True,
                partial_fallback_used=True,
                blockers=[],
            ).to_json()
            blockers = list(contract.get("blockers") or [])
            blockers.insert(0, str(exc))
            profile = {
                "backend": args.backend,
                "partial": True,
                "fallback_backend": None,
                "blockers": blockers,
                "backend_contract": contract,
                "timings_ms": {},
                "complete_group_fps_from_p50": None,
                "compiled_module_count": 0,
                "cuda_graph_enabled": args.compile_mode == "reduce-overhead",
                "ring_buffer_size": 8 if args.graph_output_policy == "ring_buffer" else 0,
                "correctness_pass": False,
                "failure_stage": "backend_contract",
                "note": "strict full hf_batched_multisession contract failed before profiling",
            }
            status = "strict_contract_failed"
    else:
        profile = run_scaffold_profile(args.frames, args.warmup)
        status = "scaffold_only"
    payload = {
        "backend": args.backend,
        "frames": args.frames,
        "warmup": args.warmup,
        "compile_mode": args.compile_mode,
        "graph_output_policy": args.graph_output_policy,
        "precision_mode": args.precision_mode,
        "profile": profile,
        "correctness_report_used": None,
        "correctness_pass": False,
        "strict_full_batched": args.strict_full_batched,
        "disallow_partial_backend_success": args.disallow_partial_backend_success,
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
                f"- strict_full_batched: `{args.strict_full_batched}`",
                "- this profile is not a validated batched multi-session runtime measurement",
                "",
                "Real performance requires the reference correctness harness and RGB replay.",
            ]
        ),
    )
    if args.debug:
        print(payload)
    if status == "strict_contract_failed":
        return 3
    return 0


def run_replay_profile(args: argparse.Namespace) -> dict:
    from .batched_multisession_runtime import run_candidate
    from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
    from .rgb_replay import load_replay_frames
    from .sam31_frame0_init import (
        default_sam31_frame0_init_mask_root,
        generate_sam31_frame0_init_masks,
    )
    from .sam31_replay_reference import initial_masks_by_camera_from_sam31

    replay_frames = load_replay_frames(args.rgb_replay)
    config = ReferenceRuntimeConfig(
        model_id=args.model_id,
        dtype=reference_dtype_for_precision_mode(args.dtype, args.precision_mode),
        device=args.device,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    initial_masks_by_camera = None
    sam31_frame0_init_summary = None
    sam31_frame0_init_mask_root = None
    if args.init_source == "sam31-image-frame0":
        sam31_frame0_init_mask_root = args.sam31_frame0_init_mask_root or str(
            default_sam31_frame0_init_mask_root(args.rgb_replay)
        )
        frame0_controller_prompt = args.sam31_frame0_controller_prompt or args.controller_prompt
        frame0_object_prompt = args.sam31_frame0_object_prompt or args.object_prompt
        sam31_frame0_init_summary = generate_sam31_frame0_init_masks(
            rgb_replay=args.rgb_replay,
            output_dir=sam31_frame0_init_mask_root,
            qqtt_root=args.qqtt_root,
            checkpoint_path=args.sam31_checkpoint,
            object_prompt=frame0_object_prompt,
            controller_prompt=frame0_controller_prompt,
            controller_label=args.controller_prompt,
            object_label=args.object_prompt,
            overwrite=args.sam31_frame0_init_overwrite,
            compile_model=args.sam31_compile_model,
            confidence_threshold=args.sam31_frame0_confidence_threshold,
            controller_selection_mode=args.sam31_frame0_controller_selection_mode,
            object_selection_mode=args.sam31_frame0_object_selection_mode,
            controller_max_instances=args.sam31_frame0_controller_max_instances,
            object_max_instances=args.sam31_frame0_object_max_instances,
            min_area=args.sam31_frame0_min_area,
            fail_on_empty=not args.sam31_frame0_allow_empty,
            device=args.device,
        )
        initial_masks_by_camera = initial_masks_by_camera_from_sam31(
            rgb_replay=args.rgb_replay,
            mask_root=sam31_frame0_init_mask_root,
            object_prompt=args.object_prompt,
            controller_prompt=args.controller_prompt,
        )
    reference_runtime.init_sessions(replay_frames[0], initial_masks_by_camera=initial_masks_by_camera)
    result = run_candidate(
        rgb_replay_frames=replay_frames,
        reference_runtime=reference_runtime,
        backend=args.backend,
        compile_mode=args.compile_mode,
        graph_output_policy=args.graph_output_policy,
        warmup=args.warmup,
        profile_frames=args.frames,
        strict_full_batched=args.strict_full_batched,
        disallow_partial_backend_success=args.disallow_partial_backend_success,
        precision_mode=args.precision_mode,
        component_runtime=args.component_runtime,
        trt_engine_dir=args.trt_engine_dir,
        trt_scope=args.trt_scope,
        trt_memory_attention_bucket_dir=args.trt_memory_attention_bucket_dir,
    )
    stage = result.timings_ms.get("stage_wall_ms", {})
    p50 = stage.get("p50")
    fps = None if not p50 else 1000.0 / float(p50)
    return {
        "backend": result.backend,
        "partial": result.partial,
        "fallback_backend": result.fallback_backend,
        "blockers": result.blockers or [],
        "backend_contract": result.backend_contract,
        "timings_ms": result.timings_ms,
        "complete_group_fps_from_p50": fps,
        "compiled_module_count": 1 if args.compile_mode != "none" else 0,
        "cuda_graph_enabled": args.compile_mode == "reduce-overhead",
        "ring_buffer_size": 8 if args.graph_output_policy == "ring_buffer" else 0,
        "correctness_pass": False,
        "init_source": args.init_source,
        "sam31_frame0_init_mask_root": sam31_frame0_init_mask_root,
        "sam31_frame0_init_summary": sam31_frame0_init_summary,
        "note": "profile-only run; use compare_multisession correctness JSON for acceptance",
    }


if __name__ == "__main__":
    raise SystemExit(main())
