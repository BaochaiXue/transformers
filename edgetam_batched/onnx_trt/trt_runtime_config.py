"""Runtime configuration for BatchTam TensorRT component execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TRT_SCOPE_MEMORY_ATTENTION = "memory_attention"
TRT_SCOPE_MASK_DECODER = "mask_decoder"
TRT_SCOPE_MEMORY_ENCODER = "memory_encoder"
TRT_SCOPE_MEMORY_ATTENTION_MASK_DECODER = "memory_attention_mask_decoder"
TRT_SCOPE_MEMORY_PATH_ALL = "memory_path_all"

TRT_SCOPES = (
    TRT_SCOPE_MEMORY_ATTENTION,
    TRT_SCOPE_MASK_DECODER,
    TRT_SCOPE_MEMORY_ENCODER,
    TRT_SCOPE_MEMORY_ATTENTION_MASK_DECODER,
    TRT_SCOPE_MEMORY_PATH_ALL,
)


@dataclass(frozen=True)
class TrtRuntimeConfig:
    engine_dir: Path
    trt_scope: str
    precision_mode: str = "memory_path_fp32"
    allow_torch_fallback: bool = False
    require_all_scope_engines: bool = True
    use_cuda_graph_safe_outputs: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_dir", Path(self.engine_dir))
        if self.trt_scope not in TRT_SCOPES:
            raise ValueError(f"unsupported TRT scope: {self.trt_scope}")


def components_for_scope(scope: str) -> tuple[str, ...]:
    if scope == TRT_SCOPE_MEMORY_ATTENTION:
        return ("memory_attention",)
    if scope == TRT_SCOPE_MASK_DECODER:
        return ("mask_decoder",)
    if scope == TRT_SCOPE_MEMORY_ENCODER:
        return ("memory_encoder",)
    if scope == TRT_SCOPE_MEMORY_ATTENTION_MASK_DECODER:
        return ("memory_attention", "mask_decoder")
    if scope == TRT_SCOPE_MEMORY_PATH_ALL:
        return ("memory_attention", "mask_decoder", "memory_encoder")
    raise ValueError(f"unsupported TRT scope: {scope}")
