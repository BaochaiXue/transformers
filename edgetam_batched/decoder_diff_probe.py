"""Decoder-focused current-frame diff report for full-batched EdgeTAM."""

from __future__ import annotations

import argparse
from typing import Any

from .current_frame_isolation import run_current_frame_isolation
from .report_utils import markdown_table, write_json, write_markdown


DECODER_FIELDS = (
    "pred_masks",
    "object_pointer",
    "object_score_logits",
    "maskmem_features",
    "maskmem_pos_enc",
)


def extract_decoder_probe(isolation_payload: dict[str, Any]) -> dict[str, Any]:
    normal = isolation_payload.get("normal_variant") or {}
    field_diffs = normal.get("field_diffs") or {}
    geometry = normal.get("geometry") or {}
    rows = []
    for field in DECODER_FIELDS:
        diff = field_diffs.get(field) or {}
        rows.append(
            {
                "field": field,
                "shape_match": diff.get("shape_match"),
                "max_abs_diff": diff.get("max_abs_diff"),
                "mean_abs_diff": diff.get("mean_abs_diff"),
                "p95_abs_diff": diff.get("p95_abs_diff"),
            }
        )
    first = first_divergent_tensor(rows)
    return {
        "backend": isolation_payload.get("backend"),
        "rgb_replay": isolation_payload.get("rgb_replay"),
        "frame_idx": isolation_payload.get("frame_idx"),
        "camera": isolation_payload.get("camera"),
        "dtype": isolation_payload.get("dtype"),
        "effective_dtype": isolation_payload.get("effective_dtype"),
        "precision_mode": isolation_payload.get("precision_mode"),
        "raw_iou_vs_hf_public": normal.get("raw_iou_vs_hf_public"),
        "first_divergent_tensor": first,
        "tensor_diffs": rows,
        "logit_diff_causes_threshold_flip_count": geometry.get("threshold_flip_count"),
        "mask_boundary_flip_count": geometry.get("boundary_flip_count"),
        "candidate_area": geometry.get("candidate_area"),
        "reference_area": geometry.get("reference_area"),
        "bbox_center_distance": geometry.get("bbox_center_distance_px"),
        "bbox_area_ratio": geometry.get("bbox_area_ratio"),
        "unsupported_internal_tensors": [
            "image_embed",
            "high_res_features",
            "prompt embeddings",
            "memory_attention output",
            "decoder input tokens",
            "decoder low_res_logits",
            "decoder high_res_logits",
        ],
        "note": (
            "This probe compares stored HF session outputs. Internal decoder tensors need a deeper "
            "runtime debug hook before they can be compared directly."
        ),
    }


def first_divergent_tensor(rows: list[dict[str, Any]], threshold: float = 1e-3) -> dict[str, Any] | None:
    for row in rows:
        if row.get("shape_match") is False:
            return row
        for key in ("p95_abs_diff", "max_abs_diff"):
            value = row.get(key)
            if value is not None and float(value) > threshold:
                return row
    return None


def render_decoder_probe(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full Batched EdgeTAM Decoder Diff Probe",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["frame_idx", payload.get("frame_idx")],
                    ["camera", payload.get("camera")],
                    ["raw_iou_vs_hf_public", _fmt(payload.get("raw_iou_vs_hf_public"))],
                    ["first_divergent_tensor", payload.get("first_divergent_tensor")],
                    ["threshold_flip_count", payload.get("logit_diff_causes_threshold_flip_count")],
                    ["reference_area", payload.get("reference_area")],
                    ["candidate_area", payload.get("candidate_area")],
                    ["bbox_center_distance", _fmt(payload.get("bbox_center_distance"))],
                    ["bbox_area_ratio", _fmt(payload.get("bbox_area_ratio"))],
                ],
            ),
            "",
            markdown_table(
                ["tensor", "shape_match", "max_abs_diff", "mean_abs_diff", "p95_abs_diff"],
                [
                    [
                        row.get("field"),
                        row.get("shape_match"),
                        _fmt(row.get("max_abs_diff")),
                        _fmt(row.get("mean_abs_diff")),
                        _fmt(row.get("p95_abs_diff")),
                    ]
                    for row in payload.get("tensor_diffs") or []
                ],
            ),
            "",
            "## Note",
            "",
            payload.get("note", ""),
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
    parser.add_argument("--frame-idx", type=int, required=True)
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("decoder_diff_probe currently supports only HF public reference")
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
        precision_mode="all_bf16",
        only_normal_variant=True,
    )
    payload = extract_decoder_probe(isolation)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_decoder_probe(payload))
    if args.debug:
        print(payload)
    return 0 if (payload.get("raw_iou_vs_hf_public") or 0.0) >= 0.90 else 2


if __name__ == "__main__":
    raise SystemExit(main())
