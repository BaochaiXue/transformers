"""Research runtime skeleton for batch=3 multi-session EdgeTAM.

The only backend implemented at this milestone is the explicit baseline contract.
Deeper batched memory/decoder backends intentionally raise NotImplementedError
until state-map correctness is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    BACKEND_BATCH_VISION_SEQ_SESSION,
    BACKEND_BATCHED_MEMORY_ATTENTION_DECODER,
    BACKEND_BATCHED_MEMORY_ATTENTION_SEQ_DECODER,
    BACKEND_BATCHED_MULTISESSION,
    BACKEND_HF_REF_SEQ_PUBLIC,
    BACKENDS,
    CompileConfig,
)


@dataclass
class BatchedStepResult:
    backend: str
    frame_idx: int
    masks_by_camera: dict[str, Any]
    timings_ms: dict[str, float]
    correctness_reference_required: bool = True


class BatchedEdgeTamMultiSessionRuntime:
    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        *,
        batch_size: int = 3,
        object_count: int = 2,
        dtype: Any = None,
        device: str = "cuda",
        backend: str = BACKEND_BATCH_VISION_SEQ_SESSION,
        compile_config: CompileConfig | None = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"unsupported backend: {backend}")
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.object_count = object_count
        self.dtype = dtype
        self.device = device
        self.backend = backend
        self.compile_config = compile_config or CompileConfig()
        self.sessions: list[Any] = []

    def init_from_reference_sessions(self, sessions: list[Any]) -> None:
        if len(sessions) != self.batch_size:
            raise ValueError(f"expected {self.batch_size} sessions, got {len(sessions)}")
        self.sessions = sessions

    def step(self, frames_b3: Any, frame_idx: int) -> BatchedStepResult:
        if self.backend in {BACKEND_HF_REF_SEQ_PUBLIC, BACKEND_BATCH_VISION_SEQ_SESSION}:
            raise NotImplementedError(
                "baseline runtime requires HF model/processor integration and real RGB replay; "
                "this milestone provides the state contract and validation scaffolding"
            )
        if self.backend in {
            BACKEND_BATCHED_MEMORY_ATTENTION_SEQ_DECODER,
            BACKEND_BATCHED_MEMORY_ATTENTION_DECODER,
            BACKEND_BATCHED_MULTISESSION,
        }:
            raise NotImplementedError(f"{self.backend} is blocked on real session state-map validation")
        raise AssertionError(self.backend)
