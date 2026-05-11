"""Custom runtime candidates for batch=3 EdgeTAM multi-session work."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .backend_contract import (
    BackendContractResult,
    FullBatchedContractError,
    contract_for_current_runtime,
)
from .component_adapter import EdgeTamComponentAdapter
from .config import (
    BACKEND_BATCH_VISION_SEQ_SESSION,
    BACKEND_BATCHED_MEMORY_ATTENTION_DECODER,
    BACKEND_BATCHED_MEMORY_ATTENTION_SEQ_DECODER,
    BACKEND_BATCHED_MULTISESSION,
    BACKEND_HF_REF_SEQ_PUBLIC,
    BACKENDS,
    COMPILE_NONE,
)
from .reference_runtime import RuntimeOutputs, autocast_context, summarize_timing_rows


class BatchedEdgeTamMultiSessionRuntime:
    def __init__(
        self,
        hf_model,
        processor,
        *,
        backend: str,
        batch_size: int = 3,
        object_count: int = 2,
        dtype: Any = None,
        device: str = "cuda",
        compile_mode: str = "none",
        graph_output_policy: str = "ring_buffer",
        ring_size: int = 8,
        strict_full_batched: bool = False,
        disallow_partial_backend_success: bool = False,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"unsupported backend: {backend}")
        self.model = hf_model
        self.processor = processor
        self.backend = backend
        self.batch_size = int(batch_size)
        self.object_count = int(object_count)
        self.dtype = dtype
        self.device = device
        self.compile_mode = compile_mode
        self.graph_output_policy = graph_output_policy
        self.ring_size = int(ring_size)
        self.strict_full_batched = bool(strict_full_batched)
        self.disallow_partial_backend_success = bool(disallow_partial_backend_success)
        self.sessions: list[Any] = []
        self.adapter = EdgeTamComponentAdapter(hf_model)
        self.partial = False
        self.fallback_backend: str | None = None
        self.blockers: list[str] = []
        self.contract: BackendContractResult | None = None
        self.torch = None
        self._compiled_get_image_features = None

        if backend in {
            BACKEND_BATCHED_MEMORY_ATTENTION_SEQ_DECODER,
            BACKEND_BATCHED_MEMORY_ATTENTION_DECODER,
            BACKEND_BATCHED_MULTISESSION,
        }:
            self.partial = True
            self.fallback_backend = BACKEND_BATCH_VISION_SEQ_SESSION
            self.blockers.append(
                f"{backend} requires explicit HF session memory/object-pointer tensorization; "
                "current implementation falls back to batch vision + sequential session decode"
            )
        self.contract = contract_for_current_runtime(
            backend=backend,
            batch_vision=backend != BACKEND_HF_REF_SEQ_PUBLIC,
            partial_fallback_used=self.partial,
            blockers=self.blockers,
        )
        if (
            self.strict_full_batched
            or (self.disallow_partial_backend_success and backend == BACKEND_BATCHED_MULTISESSION)
        ):
            self.contract.assert_full_batched()

    def init_from_reference_sessions(self, sessions: list[Any]) -> None:
        if len(sessions) != self.batch_size:
            raise ValueError(f"expected {self.batch_size} sessions, got {len(sessions)}")
        self.sessions = sessions

    def prepare_compile(self, torch_module) -> None:
        self.torch = torch_module
        if self.compile_mode == COMPILE_NONE:
            return
        if self.compile_mode == "reduce-overhead" and self.graph_output_policy != "ring_buffer":
            raise ValueError("reduce-overhead requires graph_output_policy=ring_buffer")

        def get_image_features(pixel_values):
            return self.model.get_image_features(pixel_values, return_dict=True)

        self._compiled_get_image_features = torch_module.compile(get_image_features, mode=self.compile_mode)

    def step(self, frame_images: list[Any], frame_idx: int) -> dict[str, Any]:
        if self.backend == BACKEND_HF_REF_SEQ_PUBLIC:
            raise ValueError("hf_ref_seq_public is handled by reference_runtime")
        if not self.sessions:
            raise RuntimeError("sessions are not initialized")
        if self.torch is None:
            import torch

            self.prepare_compile(torch)

        started = time.perf_counter()
        inputs = self.processor(images=frame_images, device=self.device, return_tensors="pt")
        pixel_values_batch = inputs.pixel_values.to(device=self.device, dtype=self.dtype)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        frame_indices = []
        state_stack_started = time.perf_counter()
        for pos, session in enumerate(self.sessions):
            frame_indices.append(int(session.add_new_frame(pixel_values_batch[pos], frame_idx=frame_idx)))
        state_stack_ms = (time.perf_counter() - state_stack_started) * 1000.0

        with self.torch.inference_mode(), autocast_context(self.torch, self.device, self.dtype):
            vision_started = time.perf_counter()
            if self._compiled_get_image_features is not None:
                if hasattr(self.torch, "compiler") and hasattr(self.torch.compiler, "cudagraph_mark_step_begin"):
                    self.torch.compiler.cudagraph_mark_step_begin()
                image_outputs = self._compiled_get_image_features(pixel_values_batch)
            else:
                image_outputs = self.model.get_image_features(pixel_values_batch, return_dict=True)
            if str(self.device).startswith("cuda"):
                self.torch.cuda.synchronize()
            vision_ms = (time.perf_counter() - vision_started) * 1000.0

            split_started = time.perf_counter()
            for pos, session in enumerate(self.sessions):
                session.cache.cache_vision_features(
                    frame_indices[pos],
                    split_hf_vision_features_for_session(image_outputs, pos),
                )
            feature_split_ms = (time.perf_counter() - split_started) * 1000.0

            masks = []
            logits = []
            scores = []
            decoder_rows = {}
            decoder_started = time.perf_counter()
            for cam_idx, session in enumerate(self.sessions):
                cam_started = time.perf_counter()
                output = self.model(
                    inference_session=session,
                    frame_idx=frame_indices[cam_idx],
                    frame=pixel_values_batch[cam_idx],
                )
                if str(self.device).startswith("cuda"):
                    self.torch.cuda.synchronize()
                decoder_rows[f"cam{cam_idx}_decode_ms"] = (time.perf_counter() - cam_started) * 1000.0
                logits_tensor = output.pred_masks.detach()
                masks.append((logits_tensor > 0).detach().cpu().numpy().astype(bool))
                logits.append(logits_tensor.detach().float().cpu().numpy())
                score = getattr(output, "object_score_logits", None)
                scores.append(
                    np.zeros((logits_tensor.shape[0],), dtype=np.float32)
                    if score is None
                    else score.detach().float().cpu().numpy()
                )
            decoder_ms = (time.perf_counter() - decoder_started) * 1000.0

        timings = {
            "stage_wall_ms": (time.perf_counter() - started) * 1000.0,
            "preprocess_ms": preprocess_ms,
            "state_stack_ms": state_stack_ms,
            "vision_encoder_ms": vision_ms,
            "feature_split_ms": feature_split_ms,
            "mask_decoder_ms": decoder_ms,
            "memory_attention_ms": 0.0,
            "memory_encoder_ms": 0.0,
            "state_scatter_ms": 0.0,
            "mask_postprocess_ms": 0.0,
            **decoder_rows,
        }
        return {
            "masks_b3": masks,
            "logits_b3": logits,
            "object_scores_b3": scores,
            "updated_sessions": self.sessions,
            "timings_ms": timings,
            "backend": self.backend,
            "partial": self.partial,
            "fallback_backend": self.fallback_backend,
            "blockers": list(self.blockers),
            "backend_contract": self.contract.to_json() if self.contract is not None else None,
        }


def run_candidate(
    *,
    rgb_replay_frames,
    reference_runtime,
    backend: str,
    compile_mode: str,
    graph_output_policy: str,
    warmup: int = 0,
    profile_frames: int | None = None,
    strict_full_batched: bool = False,
    disallow_partial_backend_success: bool = False,
) -> RuntimeOutputs:
    import torch
    from transformers import EdgeTamVideoInferenceSession

    first_frame = rgb_replay_frames[0]
    width, height = first_frame.images[0].size
    object_count = int(getattr(reference_runtime.config, "object_count", 2))
    initial_masks_by_camera = getattr(reference_runtime, "initial_masks_by_camera", None)
    if initial_masks_by_camera is None:
        controller_mask, object_mask = reference_runtime.initial_masks
        if object_count == 1:
            initial_masks_by_camera = [(object_mask.copy(),) for _ in range(len(first_frame.images))]
        else:
            initial_masks_by_camera = [
                (controller_mask.copy(), object_mask.copy()) for _ in range(len(first_frame.images))
            ]
    sessions = []
    for cam_idx, _ in enumerate(first_frame.images):
        cam_masks = tuple(np.asarray(mask, dtype=bool) for mask in initial_masks_by_camera[cam_idx])
        session = EdgeTamVideoInferenceSession(
            video=None,
            video_height=height,
            video_width=width,
            inference_device=reference_runtime.config.device,
            inference_state_device=reference_runtime.config.device,
            video_storage_device=reference_runtime.config.device,
            dtype=reference_runtime.dtype,
        )
        reference_runtime.processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=0,
            obj_ids=list(range(1, object_count + 1)),
            input_masks=[mask.copy() for mask in cam_masks],
        )
        sessions.append(session)

    runtime = BatchedEdgeTamMultiSessionRuntime(
        reference_runtime.model,
        reference_runtime.processor,
        backend=backend,
        batch_size=len(first_frame.images),
        object_count=object_count,
        dtype=reference_runtime.dtype,
        device=reference_runtime.config.device,
        compile_mode=compile_mode,
        graph_output_policy=graph_output_policy,
        strict_full_batched=strict_full_batched,
        disallow_partial_backend_success=disallow_partial_backend_success,
    )
    runtime.init_from_reference_sessions(sessions)
    runtime.prepare_compile(torch)

    all_masks = []
    all_logits = []
    all_scores = []
    timing_rows = []
    backend_contract = runtime.contract.to_json() if runtime.contract is not None else None
    total_steps = (profile_frames or len(rgb_replay_frames)) + int(warmup)
    for step_idx in range(total_steps):
        frame = rgb_replay_frames[step_idx % len(rgb_replay_frames)]
        result = runtime.step(frame.images, step_idx)
        backend_contract = result.get("backend_contract") or backend_contract
        if step_idx >= warmup:
            all_masks.append(result["masks_b3"])
            all_logits.append(result["logits_b3"])
            all_scores.append(result["object_scores_b3"])
            timing_rows.append(result["timings_ms"])
    return RuntimeOutputs(
        masks=all_masks,
        logits=all_logits,
        object_scores=all_scores,
        timings_ms=summarize_timing_rows(timing_rows),
        backend=backend,
        partial=runtime.partial,
        fallback_backend=runtime.fallback_backend,
        blockers=list(runtime.blockers),
        backend_contract=backend_contract,
    )


def split_hf_vision_features_for_session(image_outputs: Any, batch_idx: int) -> dict[str, Any]:
    idx = int(batch_idx)
    return {
        "vision_feats": [feature[:, idx : idx + 1, :].contiguous() for feature in image_outputs.fpn_hidden_states],
        "vision_pos_embeds": [
            pos_embed[:, idx : idx + 1, :].contiguous() for pos_embed in image_outputs.fpn_position_encoding
        ],
    }
