"""Full-state scatter gate for strict hf_batched_multisession."""

from __future__ import annotations

from typing import Any

from .backend_contract import FullBatchedContractError
from .full_batched_state import FullBatchedEdgeTamState


def scatter_full_state(state: FullBatchedEdgeTamState, sessions: list[Any]) -> None:
    if not state.stack_complete:
        raise FullBatchedContractError(
            "cannot scatter incomplete full batched state: " + "; ".join(state.unsupported_fields)
        )
    if len(sessions) != state.batch_size:
        raise FullBatchedContractError(f"scatter target count {len(sessions)} != batch_size {state.batch_size}")
