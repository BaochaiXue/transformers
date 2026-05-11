"""Full-state stack gate for strict hf_batched_multisession."""

from __future__ import annotations

from typing import Any

from .full_batched_state import FullBatchedEdgeTamState


def stack_full_state(sessions: list[Any], frame_idx: int) -> FullBatchedEdgeTamState:
    object_counts = {int(session.get_obj_num()) if hasattr(session, "get_obj_num") else len(getattr(session, "obj_ids", [])) for session in sessions}
    object_count = object_counts.pop() if len(object_counts) == 1 else 0
    unsupported = [
        "HF EdgeTAM session output_dict_per_obj is a nested per-object/per-frame Python dict",
        "memory frame selection depends on frame_idx/object history and is not tensorized",
        "object pointer history is gathered through Python session methods",
    ]
    return FullBatchedEdgeTamState(
        batch_size=len(sessions),
        object_count=object_count,
        frame_idx=int(frame_idx),
        metadata={"session_count": len(sessions)},
        unsupported_fields=unsupported,
    )
