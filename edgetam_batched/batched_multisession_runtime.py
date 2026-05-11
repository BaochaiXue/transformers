"""Custom runtime candidates for batch=3 EdgeTAM multi-session work."""

from __future__ import annotations

import time
import types
from typing import Any

import numpy as np

from .backend_contract import (
    BackendContractResult,
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
        patch_edgetam_spatial_perceiver_batch_view(hf_model)
        self.partial = False
        self.fallback_backend: str | None = None
        self.blockers: list[str] = []
        self.contract: BackendContractResult | None = None
        self.torch = None
        self._compiled_get_image_features = None

        if backend in {
            BACKEND_BATCHED_MEMORY_ATTENTION_SEQ_DECODER,
            BACKEND_BATCHED_MEMORY_ATTENTION_DECODER,
        }:
            self.partial = True
            self.fallback_backend = BACKEND_BATCH_VISION_SEQ_SESSION
            self.blockers.append(
                f"{backend} requires explicit HF session memory/object-pointer tensorization; "
                "current implementation falls back to batch vision + sequential session decode"
            )
        if backend == BACKEND_BATCHED_MULTISESSION:
            if self.object_count != 1:
                self.partial = True
                self.fallback_backend = BACKEND_BATCH_VISION_SEQ_SESSION
                self.blockers.append("full batched multisession V1 supports only object_count=1")
                self.contract = contract_for_current_runtime(
                    backend=backend,
                    batch_vision=True,
                    partial_fallback_used=True,
                    blockers=self.blockers,
                )
            else:
                self.contract = BackendContractResult(
                    backend=backend,
                    batch_vision=True,
                    batch_memory_attention=True,
                    batch_mask_decoder=True,
                    batch_memory_encoder=True,
                    batched_state_scatter=True,
                    used_public_session_step_in_hot_path=False,
                    partial_fallback_used=False,
                    blockers=[],
                )
        else:
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
        if self.backend == BACKEND_BATCHED_MULTISESSION and not self.partial:
            return self._step_full_batched_single_object(frame_images, frame_idx)

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

    def _step_full_batched_single_object(self, frame_images: list[Any], frame_idx: int) -> dict[str, Any]:
        started = time.perf_counter()
        inputs = self.processor(images=frame_images, device=self.device, return_tensors="pt")
        pixel_values_batch = inputs.pixel_values.to(device=self.device, dtype=self.dtype)
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        frame_indices = []
        state_started = time.perf_counter()
        for pos, session in enumerate(self.sessions):
            frame_indices.append(int(session.add_new_frame(pixel_values_batch[pos], frame_idx=frame_idx)))
        _assert_same(frame_indices, "frame_idx")
        frame_idx_current = frame_indices[0]
        state_stack_ms = (time.perf_counter() - state_started) * 1000.0

        with self.torch.inference_mode(), autocast_context(self.torch, self.device, self.dtype):
            vision_started = time.perf_counter()
            if self._compiled_get_image_features is not None:
                image_outputs = self._compiled_get_image_features(pixel_values_batch)
            else:
                image_outputs = self.model.get_image_features(pixel_values_batch, return_dict=True)
            if str(self.device).startswith("cuda"):
                self.torch.cuda.synchronize()
            vision_ms = (time.perf_counter() - vision_started) * 1000.0

            split_started = time.perf_counter()
            for pos, session in enumerate(self.sessions):
                session.cache.cache_vision_features(
                    frame_idx_current,
                    split_hf_vision_features_for_session(image_outputs, pos),
                )
            feature_split_ms = (time.perf_counter() - split_started) * 1000.0

            object_contexts = [self._single_object_context(session, frame_idx_current) for session in self.sessions]
            current_vision_feats = image_outputs.fpn_hidden_states
            current_vision_pos = image_outputs.fpn_position_encoding
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], self.model.backbone_feature_sizes[:-1])
            ]

            memory_attention_ms = 0.0
            decoder_started = time.perf_counter()
            if any(ctx["mask_inputs"] is not None for ctx in object_contexts):
                mask_inputs_b3 = self.torch.cat([ctx["mask_inputs"] for ctx in object_contexts], dim=0)
                pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(
                    len(self.sessions),
                    self.model.hidden_dim,
                    *self.model.backbone_feature_sizes[-1],
                )
                sam_outputs = self.model._use_mask_as_output(pix_feat, high_res_features, mask_inputs_b3)
                is_mask_from_pts = True
            else:
                mem_started = time.perf_counter()
                conditioned_features = self._batch_memory_conditioned_features(
                    object_contexts,
                    current_vision_feats[-1],
                    current_vision_pos[-1],
                    frame_idx_current,
                )
                if str(self.device).startswith("cuda"):
                    self.torch.cuda.synchronize()
                memory_attention_ms = (time.perf_counter() - mem_started) * 1000.0
                sam_outputs = self.model._single_frame_forward(
                    pixel_values=None,
                    input_points=None,
                    input_labels=None,
                    input_masks=None,
                    image_embeddings=high_res_features + [conditioned_features],
                    multimask_output=self.model._use_multimask(False, None),
                )
                is_mask_from_pts = False
            _normalize_single_object_batch_outputs(sam_outputs, len(self.sessions), self.torch)
            if str(self.device).startswith("cuda"):
                self.torch.cuda.synchronize()
            decoder_ms = (time.perf_counter() - decoder_started) * 1000.0

            memenc_started = time.perf_counter()
            maskmem_features, maskmem_pos_enc = self._batch_encode_new_memory_v1(
                current_vision_feats=current_vision_feats[-1],
                pred_masks_high_res=sam_outputs.high_res_masks,
                object_score_logits=sam_outputs.object_score_logits,
                is_mask_from_pts=is_mask_from_pts,
            )
            if str(self.device).startswith("cuda"):
                self.torch.cuda.synchronize()
            memory_encoder_ms = (time.perf_counter() - memenc_started) * 1000.0

            scatter_started = time.perf_counter()
            per_cam_masks = []
            per_cam_logits = []
            per_cam_scores = []
            for cam_idx, session in enumerate(self.sessions):
                ctx = object_contexts[cam_idx]
                current_out = {
                    "pred_masks": _slice_batch(sam_outputs.pred_masks, cam_idx),
                    "object_pointer": _slice_object_pointer_batch(
                        sam_outputs.object_pointer,
                        cam_idx,
                        batch_size=len(self.sessions),
                    ),
                    "maskmem_features": _slice_batch(maskmem_features, cam_idx),
                    "maskmem_pos_enc": _slice_batch(maskmem_pos_enc, cam_idx),
                    "object_score_logits": _slice_batch(sam_outputs.object_score_logits, cam_idx),
                }
                session.store_output(
                    0,
                    frame_idx_current,
                    output_value=current_out,
                    is_conditioning_frame=ctx["is_init_cond_frame"],
                )
                if not ctx["is_init_cond_frame"]:
                    session.frames_tracked_per_obj[0][frame_idx_current] = {"reverse": False}
                per_cam_logits.append(current_out["pred_masks"].detach().float().cpu().numpy())
                per_cam_masks.append((current_out["pred_masks"] > 0).detach().cpu().numpy().astype(bool))
                per_cam_scores.append(current_out["object_score_logits"].detach().float().cpu().numpy())
            state_scatter_ms = (time.perf_counter() - scatter_started) * 1000.0

        timings = {
            "stage_wall_ms": (time.perf_counter() - started) * 1000.0,
            "preprocess_ms": preprocess_ms,
            "state_stack_ms": state_stack_ms,
            "vision_encoder_ms": vision_ms,
            "feature_split_ms": feature_split_ms,
            "memory_attention_ms": memory_attention_ms,
            "mask_decoder_ms": decoder_ms,
            "memory_encoder_ms": memory_encoder_ms,
            "state_scatter_ms": state_scatter_ms,
            "mask_postprocess_ms": 0.0,
        }
        return {
            "masks_b3": per_cam_masks,
            "logits_b3": per_cam_logits,
            "object_scores_b3": per_cam_scores,
            "updated_sessions": self.sessions,
            "timings_ms": timings,
            "backend": self.backend,
            "partial": False,
            "fallback_backend": None,
            "blockers": [],
            "backend_contract": self.contract.to_json() if self.contract is not None else None,
        }

    def _single_object_context(self, session: Any, frame_idx: int) -> dict[str, Any]:
        obj_id = session.obj_idx_to_id(0)
        has_new_inputs = obj_id in session.obj_with_new_inputs
        has_cond_output = frame_idx in session.output_dict_per_obj[0]["cond_frame_outputs"]
        is_init_cond_frame = False
        point_inputs = None
        mask_inputs = None
        if (not has_new_inputs) and has_cond_output:
            is_init_cond_frame = True
        elif has_new_inputs:
            is_init_cond_frame = frame_idx not in session.frames_tracked_per_obj[0]
            point_inputs = session.point_inputs_per_obj[0].get(frame_idx, None)
            mask_inputs = session.mask_inputs_per_obj[0].get(frame_idx, None)
            if point_inputs is not None or mask_inputs is not None:
                session.obj_with_new_inputs.remove(obj_id)
        if point_inputs is not None:
            raise NotImplementedError("full batched V1 supports mask-init tracking only, not point prompts")
        return {
            "is_init_cond_frame": is_init_cond_frame,
            "mask_inputs": mask_inputs,
        }

    def _batch_memory_conditioned_features(
        self,
        contexts: list[dict[str, Any]],
        current_vision_features: Any,
        current_vision_positional_embeddings: Any,
        frame_idx: int,
    ) -> Any:
        memories = []
        memory_pos = []
        spatial_counts = []
        pointer_counts = []
        for session in self.sessions:
            temporal = self.model._gather_memory_frame_outputs(session, 0, frame_idx, False)
            mem_list, pos_list = self.model._build_memory_attention_inputs(temporal, current_vision_features.device)
            spatial_counts.append(len(mem_list))
            temporal_offsets, pointer_tokens, max_ptr = self.model._get_object_pointers(
                session,
                0,
                frame_idx,
                session.num_frames,
                current_vision_features.device,
                False,
                True,
            )
            ptr_count = 0
            if pointer_tokens:
                obj_ptrs, obj_pos = self.model._process_object_pointers(
                    temporal_offsets,
                    pointer_tokens,
                    max_ptr,
                    1,
                    self.model.hidden_dim,
                    current_vision_features.device,
                )
                if obj_ptrs is not None:
                    mem_list.append(obj_ptrs)
                    pos_list.append(obj_pos)
                    ptr_count = obj_ptrs.shape[0]
            pointer_counts.append(ptr_count)
            memories.append(self.torch.cat(mem_list, dim=0))
            memory_pos.append(self.torch.cat(pos_list, dim=0))
        _assert_same(spatial_counts, "num_spatial_memory_tokens")
        _assert_same(pointer_counts, "num_object_pointer_tokens")
        combined_memory = self.torch.cat(memories, dim=1)
        combined_memory_pos = self.torch.cat(memory_pos, dim=1)
        conditioned_flat = self.model.memory_attention(
            current_vision_features=current_vision_features,
            current_vision_position_embeddings=current_vision_positional_embeddings,
            memory=combined_memory,
            memory_posision_embeddings=combined_memory_pos,
            num_object_pointer_tokens=pointer_counts[0],
            num_spatial_memory_tokens=spatial_counts[0],
        )
        batch_size = len(self.sessions)
        height, width = self.model.backbone_feature_sizes[-1]
        if conditioned_flat.shape[0] == 1 and conditioned_flat.shape[1] == batch_size:
            flat = conditioned_flat.squeeze(0)
        elif conditioned_flat.shape[0] == batch_size:
            flat = conditioned_flat.squeeze(1)
        else:
            raise RuntimeError(f"unexpected memory_attention output shape {tuple(conditioned_flat.shape)}")
        return flat.permute(0, 2, 1).view(batch_size, self.model.hidden_dim, height, width)

    def _batch_encode_new_memory_v1(
        self,
        *,
        current_vision_feats: Any,
        pred_masks_high_res: Any,
        object_score_logits: Any,
        is_mask_from_pts: bool,
    ) -> tuple[Any, Any]:
        batch_size = current_vision_feats.size(1)
        channels = self.model.hidden_dim
        height, width = self.model.backbone_feature_sizes[-1]
        pix_feat = current_vision_feats.permute(1, 2, 0).view(batch_size, channels, height, width)
        if is_mask_from_pts and not self.model.training:
            mask_for_mem = (pred_masks_high_res > 0).to(pred_masks_high_res.dtype)
        else:
            mask_for_mem = self.torch.sigmoid(pred_masks_high_res)
        mask_for_mem = mask_for_mem * self.model.config.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.model.config.sigmoid_bias_for_mem_enc

        maskmem_features, maskmem_pos_enc = self.model.memory_encoder(pix_feat, mask_for_mem)
        if self.model.occlusion_spatial_embedding_parameter is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (1 - is_obj_appearing[..., None]) * self.model.occlusion_spatial_embedding_parameter[
                ..., None, None
            ].expand(*maskmem_features.shape)

        maskmem_features, maskmem_pos_enc = self.model.spatial_perceiver(maskmem_features, maskmem_pos_enc)
        return maskmem_features.to(pred_masks_high_res.dtype), maskmem_pos_enc.to(pred_masks_high_res.dtype)


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

    if backend == BACKEND_HF_REF_SEQ_PUBLIC:
        all_masks = []
        all_logits = []
        all_scores = []
        timing_rows = []
        total_steps = (profile_frames or len(rgb_replay_frames)) + int(warmup)
        for step_idx in range(total_steps):
            frame = rgb_replay_frames[step_idx % len(rgb_replay_frames)]
            masks, logits, scores, timings = reference_runtime.step_public(frame)
            if step_idx >= warmup:
                all_masks.append(masks)
                all_logits.append(logits)
                all_scores.append(scores)
                timing_rows.append(timings)
        return RuntimeOutputs(
            masks=all_masks,
            logits=all_logits,
            object_scores=all_scores,
            timings_ms=summarize_timing_rows(timing_rows),
            backend=backend,
            partial=False,
            fallback_backend=None,
            blockers=[],
            backend_contract=contract_for_current_runtime(
                backend=backend,
                batch_vision=False,
                partial_fallback_used=False,
                blockers=[],
            ).to_json(),
        )

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


def _slice_batch(value: Any, batch_idx: int) -> Any:
    return value[batch_idx : batch_idx + 1].contiguous()


def _slice_object_pointer_batch(value: Any, batch_idx: int, *, batch_size: int) -> Any:
    """Slice a per-camera object pointer from a batched EdgeTAM output.

    HF EdgeTAM's mask-init path can return an object pointer shaped like
    ``[camera_batch, camera_batch, channels]`` after multimask selection. The
    public single-camera session stores ``[1, 1, channels]``. For strict
    batch=3 session scatter we keep the diagonal entry, i.e. camN receives only
    the pointer produced from camN's frame/mask.
    """

    if (
        getattr(value, "ndim", 0) == 3
        and value.shape[0] == batch_size
        and value.shape[1] == batch_size
        and batch_size > 1
    ):
        return value[batch_idx : batch_idx + 1, batch_idx : batch_idx + 1, :].contiguous()
    return _slice_batch(value, batch_idx)


def _normalize_single_object_batch_outputs(outputs: Any, batch_size: int, torch_module: Any) -> None:
    """Normalize HF batched single-object outputs to ``[B, 1, ...]`` tensors.

    The current HF EdgeTAM implementation has indexing paths that are harmless
    for ``B=1`` but produce ``[B, B, ...]`` tensors for batch=3 multimask/object
    pointer outputs. Full multi-session scatter needs one independent object
    result per camera, so we keep the camera/object diagonal.
    """

    batch = int(batch_size)
    for attr in ("pred_masks", "high_res_masks"):
        value = getattr(outputs, attr, None)
        if (
            getattr(value, "ndim", 0) == 4
            and value.shape[0] == batch
            and value.shape[1] == batch
            and batch > 1
        ):
            idx = torch_module.arange(batch, device=value.device)
            setattr(outputs, attr, value[idx, idx].unsqueeze(1).contiguous())

    value = getattr(outputs, "object_pointer", None)
    if (
        getattr(value, "ndim", 0) == 3
        and value.shape[0] == batch
        and value.shape[1] == batch
        and batch > 1
    ):
        idx = torch_module.arange(batch, device=value.device)
        setattr(outputs, "object_pointer", value[idx, idx].unsqueeze(1).contiguous())


def _assert_same(values: list[Any], name: str) -> None:
    if not values:
        return
    first = values[0]
    if any(value != first for value in values):
        raise RuntimeError(f"{name} differs across cameras: {values}")


def patch_edgetam_spatial_perceiver_batch_view(model: Any) -> None:
    perceiver = getattr(model, "spatial_perceiver", None)
    if perceiver is None or getattr(perceiver, "_qqtt_batch_view_patch", False):
        return

    def _forward_2d_patched(self, hidden_states):
        import math
        import sys

        module = sys.modules[self.__class__.__module__]
        window_partition = getattr(module, "window_partition")
        batch_size, channels, height, width = hidden_states.shape
        latents_2d = self.latents_2d.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, 1, channels)
        num_windows_per_dim = int(math.sqrt(self.num_latents_2d))
        window_size = height // num_windows_per_dim
        windowed_input = hidden_states.permute(0, 2, 3, 1)
        windowed_features, _ = window_partition(windowed_input, window_size)
        windowed_features = windowed_features.flatten(1, 2)
        for layer in self.layers:
            latents_2d = layer(latents_2d, windowed_features, positional_encoding=None)
        latents_2d = latents_2d.reshape(batch_size, num_windows_per_dim, num_windows_per_dim, channels).permute(
            0, 3, 1, 2
        )
        positional_encoding_2d = self.positional_encoding(
            latents_2d.shape,
            latents_2d.device,
            latents_2d.dtype,
        ).to(dtype=hidden_states.dtype)
        positional_encoding_2d = positional_encoding_2d.permute(0, 2, 3, 1).flatten(1, 2)
        latents_2d = latents_2d.permute(0, 2, 3, 1).flatten(1, 2)
        latents_2d = self.layer_norm(latents_2d)
        return latents_2d, positional_encoding_2d

    perceiver._forward_2d = types.MethodType(_forward_2d_patched, perceiver)
    perceiver._qqtt_batch_view_patch = True
