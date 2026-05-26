"""State scatter helpers for future batched runtime outputs."""

from __future__ import annotations

from .batched_state import BatchedMultiSessionState


def validate_scatter_target(state: BatchedMultiSessionState, target_count: int = 3) -> None:
    if state.batch_size != target_count:
        raise ValueError(f"expected batch_size={target_count}, got {state.batch_size}")
