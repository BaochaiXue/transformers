"""Component access wrapper for HF EdgeTAM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ComponentAccessReport:
    vision_encoder: bool
    memory_attention: bool
    memory_encoder: bool
    mask_decoder: bool
    prompt_encoder: bool


class EdgeTamComponentAdapter:
    def __init__(self, model: Any):
        self.model = model
        self.vision_encoder = getattr(model, "vision_encoder", None)
        self.memory_attention = getattr(model, "memory_attention", None)
        self.memory_encoder = getattr(model, "memory_encoder", None)
        self.mask_decoder = getattr(model, "mask_decoder", None)
        self.prompt_encoder = getattr(model, "prompt_encoder", None)

    def report(self) -> ComponentAccessReport:
        return ComponentAccessReport(
            vision_encoder=self.vision_encoder is not None,
            memory_attention=self.memory_attention is not None,
            memory_encoder=self.memory_encoder is not None,
            mask_decoder=self.mask_decoder is not None,
            prompt_encoder=self.prompt_encoder is not None,
        )
