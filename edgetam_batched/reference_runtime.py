"""Reference runtime boundary for the original HF public EdgeTAM API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ReferenceRuntimeConfig:
    model_id: str = "yonigozlan/EdgeTAM-hf"
    dtype: str = "bfloat16"
    device: str = "cuda"
    object_prompt: str = "stuffed animal"
    controller_prompt: str = "towel"


class HfEdgeTamReferenceRuntime:
    """Small wrapper documenting the reference API boundary.

    Full reference execution requires the HF model weights and real RGB replay.
    This class stays deliberately thin so reports can distinguish reference
    behavior from the custom batched scheduler.
    """

    def __init__(self, config: ReferenceRuntimeConfig | None = None):
        self.config = config or ReferenceRuntimeConfig()
        self.model = None
        self.processor = None
        self.sessions: list[Any] = []

    def load(self) -> None:
        from transformers import EdgeTamVideoModel, Sam2VideoProcessor

        self.model = EdgeTamVideoModel.from_pretrained(self.config.model_id)
        self.processor = Sam2VideoProcessor.from_pretrained(self.config.model_id)

    def init_sessions(self, first_frames: list[Any], prompts: dict[str, Any]) -> list[Any]:
        if self.model is None or self.processor is None:
            raise RuntimeError("call load() before init_sessions()")
        raise NotImplementedError("real prompt/session initialization will be implemented after RGB replay is fixed")

    def step(self, frames_by_cam: list[Any], frame_idx: int) -> dict[str, Any]:
        if not self.sessions:
            raise RuntimeError("sessions are not initialized")
        raise NotImplementedError("reference stepping requires real HF sessions")
