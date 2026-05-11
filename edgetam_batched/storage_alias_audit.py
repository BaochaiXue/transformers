"""Audit tensor storage aliasing in full-batched EdgeTAM state scatter."""

from __future__ import annotations

import argparse
import copy
from typing import Any

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .find_first_bad_frame import _session_output
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames


AUDIT_FIELDS = ("pred_masks", "object_pointer", "maskmem_features", "maskmem_pos_enc", "object_score_logits")


def tensor_storage_ids(value: Any, path: str = "") -> list[dict[str, Any]]:
    ids = []
    if hasattr(value, "detach") and hasattr(value, "untyped_storage"):
        try:
            ptr = int(value.untyped_storage().data_ptr())
        except Exception:
            ptr = None
        ids.append(
            {
                "path": path,
                "storage_ptr": ptr,
                "data_ptr": int(value.data_ptr()) if hasattr(value, "data_ptr") else None,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "storage_offset": int(value.storage_offset()) if hasattr(value, "storage_offset") else None,
            }
        )
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            ids.extend(tensor_storage_ids(item, f"{path}[{idx}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            ids.extend(tensor_storage_ids(item, f"{path}.{key}" if path else str(key)))
    return ids


def run_storage_alias_audit(
    *,
    rgb_replay: str,
    backend: str,
    object_count: int,
    object_prompt: str,
    controller_prompt: str,
    frames: int,
    dtype: str,
    device: str,
) -> dict[str, Any]:
    replay_frames = load_replay_frames(rgb_replay, frames)
    config = ReferenceRuntimeConfig(
        dtype=dtype,
        device=device,
        object_prompt=object_prompt,
        controller_prompt=controller_prompt,
        object_count=object_count,
    )
    reference_runtime = HfEdgeTamReferenceRuntime(config)
    reference_runtime.load()
    reference_runtime.init_sessions(replay_frames[0])
    runtime = BatchedEdgeTamMultiSessionRuntime(
        reference_runtime.model,
        reference_runtime.processor,
        backend=backend,
        batch_size=len(replay_frames[0].images),
        object_count=object_count,
        dtype=reference_runtime.dtype,
        device=device,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
    )
    runtime.init_from_reference_sessions(copy.deepcopy(reference_runtime.sessions))
    runtime.prepare_compile(reference_runtime.torch)

    alias_records = []
    previous_by_cam_field: dict[tuple[int, str], set[int]] = {}
    for step_idx, frame in enumerate(replay_frames):
        runtime.step(frame.images, step_idx)
        current_by_cam_field: dict[tuple[int, str], set[int]] = {}
        for cam_idx, session in enumerate(runtime.sessions):
            output, _bucket = _session_output(session, step_idx)
            for field in AUDIT_FIELDS:
                ptrs = {
                    item["storage_ptr"]
                    for item in tensor_storage_ids(output.get(field), field)
                    if item.get("storage_ptr") is not None
                }
                current_by_cam_field[(cam_idx, field)] = ptrs
        for field in AUDIT_FIELDS:
            seen: dict[int, int] = {}
            for cam_idx in range(len(runtime.sessions)):
                for ptr in current_by_cam_field.get((cam_idx, field), set()):
                    if ptr in seen:
                        alias_records.append(
                            {
                                "kind": "cross_camera_storage_alias",
                                "frame_idx": step_idx,
                                "field": field,
                                "camera_a": f"cam{seen[ptr]}",
                                "camera_b": f"cam{cam_idx}",
                                "storage_ptr": ptr,
                            }
                        )
                    seen[ptr] = cam_idx
        for key, ptrs in current_by_cam_field.items():
            prev = previous_by_cam_field.get(key, set())
            overlap = sorted(prev & ptrs)
            if overlap:
                alias_records.append(
                    {
                        "kind": "storage_reused_across_frames",
                        "frame_idx": step_idx,
                        "camera": f"cam{key[0]}",
                        "field": key[1],
                        "storage_ptrs": overlap,
                    }
                )
        previous_by_cam_field = current_by_cam_field
    return {
        "backend": backend,
        "rgb_replay": str(rgb_replay),
        "dtype": dtype,
        "frames": len(replay_frames),
        "alias_found": bool(alias_records),
        "alias_records": alias_records[:100],
        "alias_record_count": len(alias_records),
        "backend_contract": runtime.contract.to_json() if runtime.contract is not None else None,
    }


def render_storage_alias_audit(payload: dict[str, Any]) -> str:
    rows = []
    for item in payload.get("alias_records") or []:
        rows.append(
            [
                item.get("kind"),
                item.get("frame_idx"),
                item.get("camera") or f"{item.get('camera_a')}->{item.get('camera_b')}",
                item.get("field"),
                item.get("storage_ptr") or item.get("storage_ptrs"),
            ]
        )
    return "\n".join(
        [
            "# Full Batched EdgeTAM Storage Alias Audit",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["alias_found", payload.get("alias_found")],
                    ["alias_record_count", payload.get("alias_record_count")],
                    ["frames", payload.get("frames")],
                ],
            ),
            "",
            markdown_table(["kind", "frame", "camera", "field", "storage"], rows[:20]) if rows else "No alias records found.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--backend", default="hf_batched_multisession")
    parser.add_argument("--reference-source", choices=("hf-public", "hf-public-seq"), default="hf-public-seq")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="hand")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("storage_alias_audit currently supports only HF public reference")
    payload = run_storage_alias_audit(
        rgb_replay=args.rgb_replay,
        backend=args.backend,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        frames=args.frames,
        dtype=args.dtype,
        device=args.device,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_storage_alias_audit(payload))
    if args.debug:
        print(payload)
    return 2 if payload.get("alias_found") else 0


if __name__ == "__main__":
    raise SystemExit(main())
