"""Batched state containers for the future multi-session scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchedMultiSessionState:
    batch_size: int
    object_count: int
    frame_idx: int
    camera_indices: Any = None
    original_sizes: Any = None
    memory_features: dict[str, Any] = field(default_factory=dict)
    memory_pos: dict[str, Any] = field(default_factory=dict)
    object_pointers: Any = None
    mask_logits_cache: dict[str, Any] = field(default_factory=dict)
    previous_masks: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
