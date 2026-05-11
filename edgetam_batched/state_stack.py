"""State stacking helpers.

Only metadata validation is implemented at this milestone; real tensor stacking
is blocked on the generated session state map.
"""

from __future__ import annotations

from .session_view import CameraSessionView


def validate_stack_inputs(views: list[CameraSessionView], batch_size: int = 3) -> None:
    if len(views) != batch_size:
        raise ValueError(f"expected {batch_size} session views, got {len(views)}")
    object_counts = {len(view.obj_ids) for view in views}
    if len(object_counts) != 1:
        raise ValueError(f"object count mismatch across cameras: {sorted(object_counts)}")
