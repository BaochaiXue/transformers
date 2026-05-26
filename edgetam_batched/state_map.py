"""Inspect HF EdgeTAM video session state and classify stackable tensors.

This module is intentionally correctness-first.  It does not implement the
batched runtime; it maps the state contract that a later runtime must preserve.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - tests run with torch, but keep importable.
    torch = None  # type: ignore[assignment]

from .report_utils import markdown_table, write_json, write_markdown

PRIMITIVES = (str, int, float, bool, type(None))


@dataclass
class FieldEntry:
    path: str
    python_type: str
    is_tensor: bool = False
    shape: list[int] | None = None
    dtype: str | None = None
    device: str | None = None
    requires_grad: bool | None = None
    mutated_per_frame: bool = False
    can_stack: bool = False
    stack_dim: int | None = None
    needs_padding: bool = False
    is_metadata: bool = False
    present_in_sessions: int = 0
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def inspect_object(root: Any, prefix: str = "session", max_depth: int = 10) -> list[FieldEntry]:
    entries: list[FieldEntry] = []
    seen: set[int] = set()
    _walk(root, prefix, 0, max_depth, seen, entries)
    return entries


def build_state_map(
    sessions_before: Sequence[Any],
    sessions_after: Sequence[Any] | None = None,
    *,
    max_depth: int = 10,
) -> dict[str, Any]:
    if not sessions_before:
        raise ValueError("at least one session is required")

    before_by_session = [
        {entry.path: entry for entry in inspect_object(session, f"session[{idx}]", max_depth)}
        for idx, session in enumerate(sessions_before)
    ]
    after_by_session = None
    if sessions_after is not None:
        after_by_session = [
            {entry.path: entry for entry in inspect_object(session, f"session[{idx}]", max_depth)}
            for idx, session in enumerate(sessions_after)
        ]

    grouped: dict[str, list[FieldEntry]] = {}
    for session_entries in before_by_session:
        for path, entry in session_entries.items():
            grouped.setdefault(_normalize_session_path(path), []).append(entry)

    output_entries: list[FieldEntry] = []
    for normalized_path in sorted(grouped):
        entries = grouped[normalized_path]
        representative = copy.deepcopy(entries[0])
        representative.path = normalized_path
        representative.present_in_sessions = len(entries)
        representative.mutated_per_frame = _is_mutated(normalized_path, before_by_session, after_by_session)
        _classify_stackability(representative, entries, len(sessions_before))
        output_entries.append(representative)

    summary = {
        "session_count": len(sessions_before),
        "field_count": len(output_entries),
        "tensor_field_count": sum(1 for entry in output_entries if entry.is_tensor),
        "stackable_tensor_field_count": sum(1 for entry in output_entries if entry.can_stack),
        "metadata_field_count": sum(1 for entry in output_entries if entry.is_metadata),
        "mutated_field_count": sum(1 for entry in output_entries if entry.mutated_per_frame),
    }
    required = _required_field_presence(output_entries)
    return {
        "summary": summary,
        "required_field_presence": required,
        "fields": [entry.to_json() for entry in output_entries],
    }


def create_synthetic_sessions(
    *,
    camera_count: int = 3,
    object_count: int = 2,
    frame_count: int = 0,
    height: int = 480,
    width: int = 848,
) -> list[Any]:
    """Create realistic session-like objects for state-map and unit tests.

    The objects model HF session shape and mutability without requiring model
    weights.  Final correctness tests must still use real RGB replay.
    """
    if torch is None:
        raise RuntimeError("torch is required for synthetic session creation")

    sessions = []
    for camera_idx in range(camera_count):
        obj_ids = list(range(1, object_count + 1))
        memory_features = {
            "maskmem_features": torch.full((object_count, 64, 64, 64), camera_idx, dtype=torch.bfloat16),
            "maskmem_pos_enc": torch.zeros((object_count, 64, 64, 64), dtype=torch.bfloat16),
        }
        output_dict_per_obj = {}
        frames_tracked_per_obj = {}
        point_inputs_per_obj = {}
        mask_inputs_per_obj = {}
        for obj_idx, obj_id in enumerate(obj_ids):
            output_dict_per_obj[obj_idx] = {
                "cond_frame_outputs": {
                    0: {
                        "pred_masks": torch.zeros((1, 1, height, width), dtype=torch.bfloat16),
                        "object_pointer": torch.full((1, 256), obj_id, dtype=torch.bfloat16),
                        "object_score_logits": torch.tensor([1.0], dtype=torch.float32),
                    }
                },
                "non_cond_frame_outputs": {},
            }
            if frame_count:
                output_dict_per_obj[obj_idx]["non_cond_frame_outputs"][frame_count] = {
                    "pred_masks": torch.full((1, 1, height, width), frame_count, dtype=torch.bfloat16),
                    "object_pointer": torch.full((1, 256), obj_id + frame_count, dtype=torch.bfloat16),
                    "object_score_logits": torch.tensor([1.0 + frame_count], dtype=torch.float32),
                }
            frames_tracked_per_obj[obj_idx] = {idx: {"reverse": False} for idx in range(frame_count + 1)}
            point_inputs_per_obj[obj_idx] = {}
            mask_inputs_per_obj[obj_idx] = {
                0: torch.zeros((1, height, width), dtype=torch.bool),
            }

        session = SimpleNamespace(
            processed_frames=frame_count,
            video_height=height,
            video_width=width,
            inference_device="cuda:0",
            inference_state_device="cuda:0",
            video_storage_device="cpu",
            dtype="torch.bfloat16",
            max_vision_features_cache_size=1,
            cache={
                "vision_features": {
                    frame_count: {
                        "image_embed": torch.zeros((1, 256, 64, 64), dtype=torch.bfloat16),
                        "high_res_feats": [
                            torch.zeros((1, 32, 256, 256), dtype=torch.bfloat16),
                            torch.zeros((1, 64, 128, 128), dtype=torch.bfloat16),
                        ],
                    }
                }
            },
            _obj_id_to_idx={obj_id: idx for idx, obj_id in enumerate(obj_ids)},
            _obj_idx_to_id={idx: obj_id for idx, obj_id in enumerate(obj_ids)},
            obj_ids=obj_ids,
            point_inputs_per_obj=point_inputs_per_obj,
            mask_inputs_per_obj=mask_inputs_per_obj,
            output_dict_per_obj=output_dict_per_obj,
            frames_tracked_per_obj=frames_tracked_per_obj,
            obj_with_new_inputs=set(),
            memory_features=memory_features,
            previous_masks=torch.zeros((object_count, 1, height, width), dtype=torch.bool),
            camera_idx=camera_idx,
        )
        sessions.append(session)
    return sessions


def render_state_map_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    required = payload["required_field_presence"]
    rows = []
    for entry in payload["fields"]:
        rows.append(
            [
                entry["path"],
                entry["python_type"],
                entry["shape"],
                entry["dtype"],
                entry["mutated_per_frame"],
                entry["can_stack"],
                entry["needs_padding"],
                entry["is_metadata"],
                entry["notes"],
            ]
        )
    summary_lines = [
        "# EdgeTAM Batched Session State Map",
        "",
        "This report maps the session fields that a batch=3 multi-session runtime must preserve.",
        "",
        "## Summary",
        "",
        markdown_table(["metric", "value"], summary.items()),
        "",
        "## Required Field Presence",
        "",
        markdown_table(["field group", "present"], required.items()),
        "",
        "## Fields",
        "",
        markdown_table(
            [
                "path",
                "type",
                "shape",
                "dtype",
                "mutated",
                "can_stack",
                "needs_padding",
                "metadata",
                "notes",
            ],
            rows,
        ),
    ]
    return "\n".join(summary_lines)


def _walk(
    value: Any,
    path: str,
    depth: int,
    max_depth: int,
    seen: set[int],
    entries: list[FieldEntry],
) -> None:
    if depth > max_depth:
        entries.append(FieldEntry(path=path, python_type=type(value).__name__, notes="max_depth"))
        return
    if _is_tensor(value):
        entries.append(_tensor_entry(path, value))
        return
    if isinstance(value, PRIMITIVES):
        entries.append(
            FieldEntry(
                path=path,
                python_type=type(value).__name__,
                is_metadata=True,
                notes=f"value={value!r}"[:120],
            )
        )
        return
    obj_id = id(value)
    if obj_id in seen:
        entries.append(FieldEntry(path=path, python_type=type(value).__name__, notes="cycle"))
        return
    seen.add(obj_id)

    if isinstance(value, Mapping):
        entries.append(FieldEntry(path=path, python_type=type(value).__name__, is_metadata=True, notes=f"len={len(value)}"))
        for key in sorted(value.keys(), key=lambda item: str(item)):
            _walk(value[key], f"{path}.{_safe_key(key)}", depth + 1, max_depth, seen, entries)
        return
    if isinstance(value, (list, tuple)):
        entries.append(FieldEntry(path=path, python_type=type(value).__name__, is_metadata=True, notes=f"len={len(value)}"))
        for idx, item in enumerate(value):
            _walk(item, f"{path}[{idx}]", depth + 1, max_depth, seen, entries)
        return
    if isinstance(value, set):
        entries.append(FieldEntry(path=path, python_type="set", is_metadata=True, notes=f"len={len(value)}"))
        return
    if hasattr(value, "__dict__"):
        entries.append(FieldEntry(path=path, python_type=type(value).__name__, is_metadata=True))
        for key, item in sorted(vars(value).items()):
            if key.startswith("__"):
                continue
            _walk(item, f"{path}.{key}", depth + 1, max_depth, seen, entries)
        return

    entries.append(FieldEntry(path=path, python_type=type(value).__name__, is_metadata=True))


def _tensor_entry(path: str, value: Any) -> FieldEntry:
    return FieldEntry(
        path=path,
        python_type=type(value).__name__,
        is_tensor=True,
        shape=list(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        requires_grad=bool(value.requires_grad),
        is_metadata=False,
    )


def _is_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _normalize_session_path(path: str) -> str:
    return re.sub(r"^session\[\d+\]", "session[*]", path)


def _safe_key(key: Any) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return key
    return f"[{repr(key)}]"


def _classify_stackability(entry: FieldEntry, entries: list[FieldEntry], session_count: int) -> None:
    entry.present_in_sessions = len(entries)
    if entry.present_in_sessions != session_count:
        entry.can_stack = False
        entry.notes = _append_note(entry.notes, "missing_in_some_sessions")
        return
    if not entry.is_tensor:
        entry.can_stack = False
        entry.is_metadata = True
        return
    shape_key = {(tuple(item.shape or []), item.dtype) for item in entries}
    if len(shape_key) == 1:
        entry.can_stack = True
        entry.stack_dim = 0
    else:
        entry.can_stack = False
        entry.needs_padding = True
        entry.notes = _append_note(entry.notes, "shape_or_dtype_mismatch")


def _is_mutated(
    normalized_path: str,
    before_by_session: list[dict[str, FieldEntry]],
    after_by_session: list[dict[str, FieldEntry]] | None,
) -> bool:
    if after_by_session is None:
        return False
    for before_entries, after_entries in zip(before_by_session, after_by_session, strict=False):
        concrete_path = next((path for path in before_entries if _normalize_session_path(path) == normalized_path), None)
        after_path = next((path for path in after_entries if _normalize_session_path(path) == normalized_path), None)
        if concrete_path is None or after_path is None:
            return True
        before = before_entries[concrete_path]
        after = after_entries[after_path]
        if before.python_type != after.python_type or before.shape != after.shape or before.dtype != after.dtype:
            return True
        if before.notes != after.notes and (before.is_metadata or after.is_metadata):
            return True
    return False


def _required_field_presence(entries: list[FieldEntry]) -> dict[str, bool]:
    path_blob = "\n".join(entry.path.lower() for entry in entries)
    groups = {
        "frame_idx_or_processed_frames": ("frame_idx", "processed_frames"),
        "video_size": ("video_height", "video_width", "original_size"),
        "obj_ids": ("obj_ids", "_obj_id_to_idx", "_obj_idx_to_id"),
        "prompt_inputs": ("point_inputs", "mask_inputs"),
        "vision_feature_cache": ("vision_features", "image_embed", "high_res_feats"),
        "memory_features": ("memory", "maskmem"),
        "object_pointers": ("object_pointer", "object_ptr"),
        "mask_logits_or_pred_masks": ("mask_logits", "pred_masks", "previous_masks"),
        "conditioning_outputs": ("cond_frame_outputs", "non_cond_frame_outputs"),
    }
    return {name: any(token in path_blob for token in tokens) for name, tokens in groups.items()}


def _append_note(notes: str, addition: str) -> str:
    return addition if not notes else f"{notes};{addition}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-count", type=int, default=3)
    parser.add_argument("--object-count", type=int, default=2)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="towel")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--frames-for-mutation-check", type=int, default=3)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    before = create_synthetic_sessions(
        camera_count=args.camera_count,
        object_count=args.object_count,
        frame_count=0,
    )
    after = create_synthetic_sessions(
        camera_count=args.camera_count,
        object_count=args.object_count,
        frame_count=args.frames_for_mutation_check,
    )
    payload = build_state_map(before, after)
    payload["experiment"] = {
        "camera_count": args.camera_count,
        "object_count": args.object_count,
        "object_prompt": args.object_prompt,
        "controller_prompt": args.controller_prompt,
        "dtype": args.dtype,
        "frames_for_mutation_check": args.frames_for_mutation_check,
        "source": "synthetic_session_contract",
        "note": "State contract map only. Final correctness must use real RGB replay.",
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_state_map_markdown(payload))
    if args.debug:
        print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
