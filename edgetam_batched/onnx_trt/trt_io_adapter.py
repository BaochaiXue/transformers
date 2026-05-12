"""Map full-batched scheduler tensors to fixed BatchTam TensorRT IO specs."""

from __future__ import annotations

from typing import Any

from .trt_engine_utils import ComponentIOSpec


class TrtIoAdapter:
    def __init__(self, torch_module: Any):
        self.torch = torch_module

    def memory_attention_inputs(
        self,
        *,
        io_spec: ComponentIOSpec,
        current_vision_features: Any,
        current_vision_position_embeddings: Any,
        memory: Any,
        memory_posision_embeddings: Any,
    ) -> tuple[Any, ...]:
        tensors = (
            current_vision_features,
            current_vision_position_embeddings,
            memory,
            memory_posision_embeddings,
        )
        return self._coerce_inputs("memory_attention", io_spec, tensors)

    def mask_decoder_inputs(
        self,
        *,
        io_spec: ComponentIOSpec,
        dense_prompt_embeddings: Any,
        high_resolution_features: list[Any],
        image_embeddings: Any,
        image_positional_embeddings: Any,
        sparse_prompt_embeddings: Any,
    ) -> tuple[Any, ...]:
        tensors = (
            dense_prompt_embeddings,
            high_resolution_features[0],
            high_resolution_features[1],
            image_embeddings,
            image_positional_embeddings,
            sparse_prompt_embeddings,
        )
        return self._coerce_inputs("mask_decoder", io_spec, tensors)

    def memory_encoder_inputs(self, *, io_spec: ComponentIOSpec, pix_feat: Any, mask_for_mem: Any) -> tuple[Any, ...]:
        return self._coerce_inputs("memory_encoder", io_spec, (pix_feat, mask_for_mem))

    def _coerce_inputs(self, component: str, io_spec: ComponentIOSpec, tensors: tuple[Any, ...]) -> tuple[Any, ...]:
        if len(tensors) != len(io_spec.inputs):
            raise RuntimeError(f"{component}: expected {len(io_spec.inputs)} TRT inputs, got {len(tensors)}")
        out = []
        for tensor, spec in zip(tensors, io_spec.inputs, strict=True):
            if not isinstance(tensor, self.torch.Tensor):
                raise TypeError(f"{component}.{spec.name}: expected torch.Tensor, got {type(tensor).__name__}")
            expected_shape = tuple(int(dim) for dim in spec.shape)
            actual_shape = tuple(int(dim) for dim in tensor.shape)
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"{component}.{spec.name}: shape mismatch, expected {expected_shape}, got {actual_shape}"
                )
            expected_dtype = _torch_dtype(self.torch, spec.dtype)
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            out.append(tensor)
        return tuple(out)


def _torch_dtype(torch_module: Any, dtype_name: str):
    lowered = dtype_name.lower().replace("torch.", "")
    if lowered in {"float32", "fp32"}:
        return torch_module.float32
    if lowered in {"float16", "fp16"}:
        return torch_module.float16
    if lowered in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if lowered == "int32":
        return torch_module.int32
    if lowered == "int64":
        return torch_module.int64
    if lowered == "bool":
        return torch_module.bool
    raise ValueError(f"unsupported dtype in TRT io spec: {dtype_name}")
