"""Session view dataclasses used by future state tensorization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraSessionView:
    camera_idx: int
    session: object
    obj_ids: list[int]
    frame_idx: int
    original_size: tuple[int, int]
    tensor_fields: dict[str, Any] = field(default_factory=dict)
    metadata_fields: dict[str, Any] = field(default_factory=dict)
