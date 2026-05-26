"""Batch-order ablation for full-batched EdgeTAM camera mixing bugs."""

from __future__ import annotations

import argparse
import copy
from typing import Any

from .batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from .camera_order import mask_iou
from .compare_multisession import _resize_mask_like
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames
from .stats import summarize


def parse_order(text: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in text.split(",") if part.strip() != "")
    if not parts:
        raise ValueError("empty order")
    return parts


def run_batch_order_ablation(
    *,
    rgb_replay: str,
    backend: str,
    object_count: int,
    object_prompt: str,
    controller_prompt: str,
    frames: int,
    dtype: str,
    device: str,
    orders: list[str],
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
    reference_masks = []
    for frame in replay_frames:
        masks, _logits, _scores, _timings = reference_runtime.step_public(frame)
        reference_masks.append(masks)

    order_rows = []
    for text in orders:
        order = parse_order(text)
        order_rows.append(
            run_one_order(
                reference_runtime=reference_runtime,
                replay_frames=replay_frames,
                reference_masks=reference_masks,
                backend=backend,
                object_count=object_count,
                dtype=dtype,
                device=device,
                order=order,
            )
        )
    normal = next((row for row in order_rows if row["order"] == [0, 1, 2]), None)
    normal_iou = (normal or {}).get("cam1_iou_summary", {}).get("avg")
    order_dependent = any(
        row.get("cam1_iou_summary", {}).get("avg") is not None
        and normal_iou is not None
        and abs(float(row["cam1_iou_summary"]["avg"]) - float(normal_iou)) > 1e-3
        for row in order_rows
    )
    return {
        "backend": backend,
        "rgb_replay": str(rgb_replay),
        "dtype": dtype,
        "object_count": object_count,
        "object_prompt": object_prompt,
        "orders": order_rows,
        "order_dependent": bool(order_dependent),
        "diagonal_slicing_bug": bool(order_dependent),
    }


def run_one_order(
    *,
    reference_runtime: HfEdgeTamReferenceRuntime,
    replay_frames: list[Any],
    reference_masks: list[list[Any]],
    backend: str,
    object_count: int,
    dtype: str,
    device: str,
    order: tuple[int, ...],
) -> dict[str, Any]:
    reference_runtime.init_sessions(replay_frames[0])
    base_sessions = copy.deepcopy(reference_runtime.sessions)
    ordered_sessions = [copy.deepcopy(base_sessions[idx]) for idx in order]
    runtime = BatchedEdgeTamMultiSessionRuntime(
        reference_runtime.model,
        reference_runtime.processor,
        backend=backend,
        batch_size=len(order),
        object_count=object_count,
        dtype=reference_runtime.dtype,
        device=device,
        strict_full_batched=True,
        disallow_partial_backend_success=True,
    )
    runtime.init_from_reference_sessions(ordered_sessions)
    runtime.prepare_compile(reference_runtime.torch)

    all_ious = []
    cam1_ious = []
    first_bad = None
    for step_idx, frame in enumerate(replay_frames):
        ordered_images = [frame.images[idx] for idx in order]
        result = runtime.step(ordered_images, step_idx)
        for out_idx, ref_cam_idx in enumerate(order):
            ref_mask = reference_masks[step_idx][ref_cam_idx][0]
            cand_mask = _resize_mask_like(result["masks_b3"][out_idx][0], ref_mask)
            iou = mask_iou(ref_mask, cand_mask)
            all_ious.append(iou)
            if ref_cam_idx == 1:
                cam1_ious.append(iou)
            if first_bad is None and iou < 0.90:
                first_bad = {
                    "frame_idx": step_idx,
                    "output_index": out_idx,
                    "reference_camera": f"cam{ref_cam_idx}",
                    "iou": float(iou),
                }
    return {
        "order": list(order),
        "first_bad_frame": first_bad,
        "global_iou_summary": summarize(all_ious),
        "cam1_iou_summary": summarize(cam1_ious),
        "object_pointer_order_check": "not_checked",
        "maskmem_features_order_check": "not_checked",
        "backend_contract": runtime.contract.to_json() if runtime.contract is not None else None,
    }


def render_batch_order_ablation(payload: dict[str, Any]) -> str:
    rows = []
    for item in payload.get("orders") or []:
        rows.append(
            [
                ",".join(str(v) for v in item.get("order", [])),
                item.get("first_bad_frame"),
                _fmt((item.get("global_iou_summary") or {}).get("avg")),
                _fmt((item.get("global_iou_summary") or {}).get("min")),
                _fmt((item.get("cam1_iou_summary") or {}).get("avg")),
                _fmt((item.get("cam1_iou_summary") or {}).get("min")),
            ]
        )
    return "\n".join(
        [
            "# Full Batched EdgeTAM Batch Order Ablation",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["order_dependent", payload.get("order_dependent")],
                    ["diagonal_slicing_bug", payload.get("diagonal_slicing_bug")],
                ],
            ),
            "",
            markdown_table(["order", "first_bad", "global_avg", "global_min", "cam1_avg", "cam1_min"], rows),
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
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--orders", nargs="+", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("batch_order_ablation currently supports only HF public reference")
    payload = run_batch_order_ablation(
        rgb_replay=args.rgb_replay,
        backend=args.backend,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        frames=args.frames,
        dtype=args.dtype,
        device=args.device,
        orders=args.orders,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_batch_order_ablation(payload))
    if args.debug:
        print(payload)
    return 2 if payload.get("order_dependent") else 0


if __name__ == "__main__":
    raise SystemExit(main())
