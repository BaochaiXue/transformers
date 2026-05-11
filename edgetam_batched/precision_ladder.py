"""Precision mode helpers for full-batched EdgeTAM drift probes."""

from __future__ import annotations

import argparse
from typing import Any

from .current_frame_isolation import PRECISION_MODES, dtype_for_precision, run_current_frame_isolation
from .report_utils import markdown_table, write_json, write_markdown


def precision_mode_status(mode: str) -> dict[str, Any]:
    if mode not in PRECISION_MODES:
        raise ValueError(f"unsupported precision mode: {mode}")
    if mode in {"all_bf16", "all_fp32"}:
        return {"mode": mode, "implemented": True, "reason": None}
    return {
        "mode": mode,
        "implemented": False,
        "reason": "selective component fp32 requires runtime-level dtype hooks; current probe records this as a pending patch",
    }


def render_precision_ladder(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Batched EdgeTAM Precision Ladder",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["mode", payload.get("precision_mode")],
                    ["implemented", payload.get("implemented")],
                    ["effective_dtype", payload.get("effective_dtype")],
                    ["frame_idx", payload.get("frame_idx")],
                    ["camera", payload.get("camera")],
                    ["raw_iou", _fmt(payload.get("raw_iou_vs_hf_public"))],
                    ["conclusion", payload.get("conclusion")],
                    ["reason", payload.get("reason")],
                ],
            ),
        ]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--backend", default="hf_batched_multisession")
    parser.add_argument("--reference-source", choices=("hf-public", "hf-public-seq"), default="hf-public-seq")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="hand")
    parser.add_argument("--frame-idx", type=int, default=47)
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision-mode", choices=sorted(PRECISION_MODES), required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("precision_ladder currently supports only HF public reference")
    status = precision_mode_status(args.precision_mode)
    payload = {
        "precision_mode": args.precision_mode,
        "implemented": status["implemented"],
        "reason": status["reason"],
        "effective_dtype": dtype_for_precision(args.dtype, args.precision_mode),
        "frame_idx": args.frame_idx,
        "camera": args.camera,
    }
    if status["implemented"]:
        isolation = run_current_frame_isolation(
            rgb_replay=args.rgb_replay,
            backend=args.backend,
            object_count=args.object_count,
            object_prompt=args.object_prompt,
            controller_prompt=args.controller_prompt,
            frame_idx=args.frame_idx,
            camera=args.camera,
            dtype=args.dtype,
            device=args.device,
            force_reference_state_before_frame=True,
            replace="none",
            precision_mode=args.precision_mode,
            only_normal_variant=True,
        )
        normal = isolation.get("normal_variant") or {}
        payload.update(
            {
                "raw_iou_vs_hf_public": normal.get("raw_iou_vs_hf_public"),
                "conclusion": (
                    "current-frame pass under this precision"
                    if (normal.get("raw_iou_vs_hf_public") or 0.0) >= 0.90
                    else "current-frame divergence remains under this precision"
                ),
                "isolation": isolation,
            }
        )
    else:
        payload["conclusion"] = "not implemented; add selective dtype hooks before using this mode as evidence"
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_precision_ladder(payload))
    if args.debug:
        print(payload)
    return 0 if (not payload["implemented"]) or (payload.get("raw_iou_vs_hf_public") or 0.0) >= 0.90 else 2


if __name__ == "__main__":
    raise SystemExit(main())
