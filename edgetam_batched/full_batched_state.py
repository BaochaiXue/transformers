"""Explicit state contract for a future full batched EdgeTAM runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FullBatchedEdgeTamState:
    batch_size: int
    object_count: int
    frame_idx: int
    memory_features: Any = None
    memory_pos_enc: Any = None
    object_pointers: Any = None
    previous_masks: Any = None
    mask_logits_cache: Any = None
    conditioning_outputs: Any = None
    non_conditioning_outputs: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    unsupported_fields: list[str] = field(default_factory=list)

    @property
    def stack_complete(self) -> bool:
        return not self.unsupported_fields


def summarize_full_state_requirements() -> dict[str, Any]:
    return {
        "required_tensor_state": [
            "memory_features",
            "memory_pos_enc",
            "object_pointers",
            "previous_masks",
            "mask_logits_cache",
            "conditioning_outputs",
            "non_conditioning_outputs",
        ],
        "required_metadata": ["frame_idx", "object_ids", "original_sizes", "video_height", "video_width"],
        "status": "contract only; extraction/scatter must be implemented before hf_batched_multisession can pass",
    }
