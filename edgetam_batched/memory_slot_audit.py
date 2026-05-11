"""Audit session memory slot/key ordering between HF public and full-batched EdgeTAM."""

from __future__ import annotations

import argparse
from typing import Any

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames


def _session_obj_ids(session: Any) -> list[Any]:
    obj_ids = getattr(session, "obj_ids", [])
    if callable(obj_ids):
        try:
            obj_ids = obj_ids()
        except TypeError:
            return []
    return list(obj_ids or [])


def _session_obj_id_to_idx(session: Any, obj_ids: list[Any]) -> dict[str, Any]:
    mapping = getattr(session, "obj_id_to_idx", {})
    if callable(mapping):
        out = {}
        for obj_id in obj_ids:
            try:
                out[str(obj_id)] = int(mapping(obj_id))
            except Exception as exc:  # pragma: no cover - defensive for HF internals.
                out[str(obj_id)] = f"error:{type(exc).__name__}"
        return out
    try:
        return {str(key): value for key, value in dict(mapping).items()}
    except TypeError:
        return {"repr": repr(mapping)}


def session_slot_digest(session: Any, *, obj_idx: int = 0) -> dict[str, Any]:
    output_dict = session.output_dict_per_obj[obj_idx]
    cond_keys = sorted(int(key) for key in output_dict["cond_frame_outputs"].keys())
    noncond_keys = sorted(int(key) for key in output_dict["non_cond_frame_outputs"].keys())
    tracked_keys = sorted(int(key) for key in session.frames_tracked_per_obj[obj_idx].keys())
    latest_key = None
    latest_bucket = None
    if noncond_keys:
        latest_key = noncond_keys[-1]
        latest_bucket = "non_cond_frame_outputs"
    elif cond_keys:
        latest_key = cond_keys[-1]
        latest_bucket = "cond_frame_outputs"
    latest_output = output_dict[latest_bucket][latest_key] if latest_bucket is not None else {}
    obj_ids = _session_obj_ids(session)
    return {
        "cond_keys": cond_keys,
        "noncond_keys": noncond_keys,
        "tracked_keys": tracked_keys,
        "latest_key": latest_key,
        "latest_bucket": latest_bucket,
        "latest_fields": sorted(latest_output.keys()) if latest_output else [],
        "obj_ids": obj_ids,
        "obj_id_to_idx": _session_obj_id_to_idx(session, obj_ids),
    }


def compare_slot_digests(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for key in ("cond_keys", "noncond_keys", "tracked_keys", "latest_key", "latest_bucket", "latest_fields", "obj_ids"):
        if reference.get(key) != candidate.get(key):
            mismatches.append(
                {
                    "field": key,
                    "reference": reference.get(key),
                    "candidate": candidate.get(key),
                }
            )
    return {
        "pass": not mismatches,
        "mismatches": mismatches,
    }


def run_memory_slot_audit(
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

    reference_digests = []
    for step_idx, frame in enumerate(replay_frames):
        reference_runtime.step_public(frame)
        reference_digests.append(
            [session_slot_digest(session, obj_idx=0) for session in reference_runtime.sessions]
        )

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
    runtime.init_from_reference_sessions(reference_runtime.sessions)
    runtime.prepare_compile(reference_runtime.torch)

    rows = []
    first_mismatch = None
    for step_idx, frame in enumerate(replay_frames):
        runtime.step(frame.images, step_idx)
        for cam_idx, session in enumerate(runtime.sessions):
            candidate = session_slot_digest(session, obj_idx=0)
            reference = reference_digests[step_idx][cam_idx]
            comparison = compare_slot_digests(reference, candidate)
            row = {
                "frame_idx": step_idx,
                "camera": f"cam{cam_idx}",
                "pass": comparison["pass"],
                "mismatches": comparison["mismatches"],
            }
            rows.append(row)
            if first_mismatch is None and not comparison["pass"]:
                first_mismatch = row

    return {
        "backend": backend,
        "dtype": dtype,
        "rgb_replay": str(rgb_replay),
        "object_count": object_count,
        "object_prompt": object_prompt,
        "pass": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "rows": rows,
        "backend_contract": runtime.contract.to_json() if runtime.contract is not None else None,
    }


def render_memory_slot_audit(payload: dict[str, Any]) -> str:
    first = payload.get("first_mismatch")
    rows = []
    if first:
        for item in first.get("mismatches", []):
            rows.append([item["field"], item["reference"], item["candidate"]])
    return "\n".join(
        [
            "# Full Batched EdgeTAM Memory Slot Audit",
            "",
            f"- backend: `{payload.get('backend')}`",
            f"- dtype: `{payload.get('dtype')}`",
            f"- pass: `{payload.get('pass')}`",
            f"- first_mismatch_frame: `{None if first is None else first.get('frame_idx')}`",
            f"- first_mismatch_camera: `{None if first is None else first.get('camera')}`",
            "",
            markdown_table(["field", "reference", "candidate"], rows) if rows else "No memory slot/key mismatch found.",
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
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("memory_slot_audit currently supports only HF public reference")
    payload = run_memory_slot_audit(
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
    write_markdown(args.output_md, render_memory_slot_audit(payload))
    if args.debug:
        print(payload)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
