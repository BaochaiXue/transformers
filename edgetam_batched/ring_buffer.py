"""Tensor ring buffer for CUDA graph output lifetime safety."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: object
    device: object = "cpu"


class TensorRingBuffer:
    def __init__(self, specs: Sequence[TensorSpec], ring_size: int = 8):
        if torch is None:
            raise RuntimeError("torch is required for TensorRingBuffer")
        if ring_size < 3:
            raise ValueError("ring_size must be >= 3")
        self.specs = list(specs)
        self.ring_size = int(ring_size)
        self.index = 0
        self.buffers = [
            [torch.empty(spec.shape, dtype=spec.dtype, device=spec.device) for spec in self.specs]
            for _ in range(self.ring_size)
        ]

    def copy_from(self, outputs: Sequence) -> list:
        if len(outputs) != len(self.specs):
            raise ValueError(f"expected {len(self.specs)} outputs, got {len(outputs)}")
        slot = self.buffers[self.index]
        for dst, src in zip(slot, outputs, strict=True):
            dst.copy_(src, non_blocking=True)
            if dst.is_cuda:
                dst.record_stream(torch.cuda.current_stream())
        self.index = (self.index + 1) % self.ring_size
        return slot
