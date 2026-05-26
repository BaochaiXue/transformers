"""Ablate session state commit semantics for full-batched EdgeTAM drift."""

from __future__ import annotations

import argparse
from typing import Any

from .drift_teacher_force import render_teacher_force, run_teacher_force
from .report_utils import write_json, write_markdown


COMMIT_TO_FORCE = {
    "batched_scatter": "none",
    "reference_store_output": "after_mask_decoder",
    "hybrid_reference_dict_batched_tensors": "after_memory_encoder",
}


def run_state_commit_ablation(
    *,
    state_commit_mode: str,
    **kwargs: Any,
) -> dict[str, Any]:
    if state_commit_mode not in COMMIT_TO_FORCE:
        raise ValueError(f"unsupported state commit mode: {state_commit_mode}")
    payload = run_teacher_force(teacher_force=COMMIT_TO_FORCE[state_commit_mode], **kwargs)
    payload["state_commit_mode"] = state_commit_mode
    payload["teacher_force_mode"] = COMMIT_TO_FORCE[state_commit_mode]
    payload["interpretation"] = _interpret(state_commit_mode, payload)
    return payload


def _interpret(mode: str, payload: dict[str, Any]) -> str:
    passed = payload.get("first_bad_frame") is None and payload.get("status") == "ok"
    if mode == "batched_scatter":
        return "current full-batched state commit path"
    if mode == "reference_store_output":
        return (
            "reference-style complete current-output writeback eliminates drift"
            if passed
            else "drift remains even when current outputs are forced to reference after decoder"
        )
    if mode == "hybrid_reference_dict_batched_tensors":
        return (
            "forcing memory outputs eliminates drift; memory encoder/writeback is likely source"
            if passed
            else "drift remains after forcing memory outputs; decoder/object pointer remains suspect"
        )
    return ""


def render_state_commit_ablation(payload: dict[str, Any]) -> str:
    return render_teacher_force(payload) + "\n\n## State Commit Interpretation\n\n" + payload.get("interpretation", "")


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
    parser.add_argument(
        "--state-commit-mode",
        choices=sorted(COMMIT_TO_FORCE),
        default="batched_scatter",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.90)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("state_commit_ablation currently supports only HF public reference")
    payload = run_state_commit_ablation(
        state_commit_mode=args.state_commit_mode,
        rgb_replay=args.rgb_replay,
        backend=args.backend,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        frames=args.frames,
        dtype=args.dtype,
        device=args.device,
        iou_threshold=args.iou_threshold,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_state_commit_ablation(payload))
    if args.debug:
        print(payload)
    return 0 if payload.get("first_bad_frame") is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
