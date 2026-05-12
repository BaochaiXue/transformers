"""Shared configuration constants for the EdgeTAM batched runtime experiment."""

from __future__ import annotations

from dataclasses import dataclass

BACKEND_HF_REF_SEQ_PUBLIC = "hf_ref_seq_public"
BACKEND_BATCH_VISION_SEQ_SESSION = "hf_batch_vision_seq_session"
BACKEND_BATCHED_MEMORY_ATTENTION_SEQ_DECODER = "hf_batched_memory_attention_seq_decoder"
BACKEND_BATCHED_MEMORY_ATTENTION_DECODER = "hf_batched_memory_attention_decoder"
BACKEND_BATCHED_MULTISESSION = "hf_batched_multisession"
BACKEND_BATCHED_MULTISESSION_TRT = "hf_batched_multisession_trt_components"

BACKENDS = (
    BACKEND_HF_REF_SEQ_PUBLIC,
    BACKEND_BATCH_VISION_SEQ_SESSION,
    BACKEND_BATCHED_MEMORY_ATTENTION_SEQ_DECODER,
    BACKEND_BATCHED_MEMORY_ATTENTION_DECODER,
    BACKEND_BATCHED_MULTISESSION,
    BACKEND_BATCHED_MULTISESSION_TRT,
)

COMPILE_NONE = "none"
COMPILE_DEFAULT = "default"
COMPILE_MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"
COMPILE_REDUCE_OVERHEAD = "reduce-overhead"

COMPILE_MODES = (
    COMPILE_NONE,
    COMPILE_DEFAULT,
    COMPILE_MAX_AUTOTUNE_NO_CUDAGRAPHS,
    COMPILE_REDUCE_OVERHEAD,
)

COMPILE_SCOPE_NONE = "none"
COMPILE_SCOPE_VISION_ENCODER = "vision_encoder"
COMPILE_SCOPE_MEMORY_ATTENTION = "memory_attention"
COMPILE_SCOPE_MASK_DECODER = "mask_decoder"
COMPILE_SCOPE_MEMORY_ENCODER = "memory_encoder"
COMPILE_SCOPE_MEMORY_ATTENTION_MASK_DECODER = "memory_attention_mask_decoder"
COMPILE_SCOPE_MEMORY_PATH_ALL = "memory_path_all"
COMPILE_SCOPE_VISION_MEMORY_PATH_ALL = "vision_memory_path_all"

COMPILE_SCOPES = (
    COMPILE_SCOPE_NONE,
    COMPILE_SCOPE_VISION_ENCODER,
    COMPILE_SCOPE_MEMORY_ATTENTION,
    COMPILE_SCOPE_MASK_DECODER,
    COMPILE_SCOPE_MEMORY_ENCODER,
    COMPILE_SCOPE_MEMORY_ATTENTION_MASK_DECODER,
    COMPILE_SCOPE_MEMORY_PATH_ALL,
    COMPILE_SCOPE_VISION_MEMORY_PATH_ALL,
)

GRAPH_OUTPUT_CLONE = "clone"
GRAPH_OUTPUT_RING_BUFFER = "ring_buffer"
GRAPH_OUTPUT_POLICIES = (GRAPH_OUTPUT_CLONE, GRAPH_OUTPUT_RING_BUFFER)


@dataclass(frozen=True)
class CompileConfig:
    mode: str = COMPILE_NONE
    graph_output_policy: str = GRAPH_OUTPUT_CLONE
    ring_size: int = 8

    def validate(self) -> None:
        if self.mode not in COMPILE_MODES:
            raise ValueError(f"unsupported compile mode: {self.mode}")
        if self.graph_output_policy not in GRAPH_OUTPUT_POLICIES:
            raise ValueError(f"unsupported graph output policy: {self.graph_output_policy}")
        if self.mode == COMPILE_REDUCE_OVERHEAD and self.graph_output_policy != GRAPH_OUTPUT_RING_BUFFER:
            raise ValueError("reduce-overhead requires graph_output_policy=ring_buffer")
        if self.graph_output_policy == GRAPH_OUTPUT_RING_BUFFER and self.ring_size < 3:
            raise ValueError("ring_buffer policy requires ring_size >= 3")


def normalize_dtype_name(name: str) -> str:
    normalized = name.strip().lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return "bfloat16"
    if normalized in {"fp16", "float16", "half"}:
        return "float16"
    if normalized in {"fp32", "float32"}:
        return "float32"
    raise ValueError(f"unsupported dtype: {name}")
