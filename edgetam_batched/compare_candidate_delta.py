"""Compare original-vs-candidate EdgeTAM reports on evaluated samples only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .report_utils import markdown_table, write_json, write_markdown


def build_delta_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    not_worse_tolerance: float = 0.01,
) -> dict[str, Any]:
    base_metrics = baseline.get("metrics", {})
    cand_metrics = candidate.get("metrics", {})
    base_per = base_metrics.get("per_camera_object", {})
    cand_per = cand_metrics.get("per_camera_object", {})
    keys = sorted(set(base_per) | set(cand_per))
    per_key = {}
    object_rows: dict[str, list[dict[str, Any]]] = {}
    evaluated_subset_mismatch = False
    for key in keys:
        b = base_per.get(key, {})
        c = cand_per.get(key, {})
        b_eval = int(b.get("evaluated_sample_count") or 0)
        c_eval = int(c.get("evaluated_sample_count") or 0)
        if b_eval != c_eval:
            evaluated_subset_mismatch = True
        b_avg = _metric_avg(b)
        c_avg = _metric_avg(c)
        delta = None if b_avg is None or c_avg is None else float(c_avg) - float(b_avg)
        row = {
            "key": key,
            "baseline_iou_avg_on_evaluated": b_avg,
            "candidate_iou_avg_on_evaluated": c_avg,
            "delta": delta,
            "not_worse": None if delta is None else delta >= -float(not_worse_tolerance),
            "baseline_evaluated_sample_count": b_eval,
            "candidate_evaluated_sample_count": c_eval,
            "baseline_ignored_reference_empty_count": int(b.get("ignored_reference_empty_count") or 0),
            "candidate_ignored_reference_empty_count": int(c.get("ignored_reference_empty_count") or 0),
            "baseline_candidate_nonempty_when_reference_empty_count": int(
                b.get("candidate_nonempty_when_reference_empty_count") or 0
            ),
            "candidate_nonempty_when_reference_empty_count": int(
                c.get("candidate_nonempty_when_reference_empty_count") or 0
            ),
            "reference_uncertain": bool(c.get("reference_uncertain") or b.get("reference_uncertain")),
        }
        per_key[key] = row
        object_rows.setdefault(_object_name(key, baseline, candidate), []).append(row)

    per_object = {}
    for obj_name, rows in object_rows.items():
        base_vals = [row["baseline_iou_avg_on_evaluated"] for row in rows if row["baseline_iou_avg_on_evaluated"] is not None]
        cand_vals = [row["candidate_iou_avg_on_evaluated"] for row in rows if row["candidate_iou_avg_on_evaluated"] is not None]
        evaluated = sum(row["candidate_evaluated_sample_count"] for row in rows)
        ignored = sum(row["candidate_ignored_reference_empty_count"] for row in rows)
        candidate_nonempty_ignored = sum(row["candidate_nonempty_when_reference_empty_count"] for row in rows)
        base_avg = sum(base_vals) / len(base_vals) if base_vals else None
        cand_avg = sum(cand_vals) / len(cand_vals) if cand_vals else None
        delta = None if base_avg is None or cand_avg is None else cand_avg - base_avg
        per_object[obj_name] = {
            "baseline_iou_avg_on_evaluated": base_avg,
            "candidate_iou_avg_on_evaluated": cand_avg,
            "delta": delta,
            "not_worse": None if delta is None else delta >= -float(not_worse_tolerance),
            "evaluated_sample_count": evaluated,
            "ignored_reference_empty_count": ignored,
            "candidate_nonempty_when_reference_empty_count": candidate_nonempty_ignored,
            "reference_uncertain": ignored > 0,
            "strict_validated": evaluated > 0 and delta is not None,
            "delta_valid": evaluated > 0 and delta is not None,
            "reason": "no evaluated SAM3.1 reference samples" if evaluated == 0 else None,
        }

    return {
        "baseline_path": baseline.get("_path"),
        "candidate_path": candidate.get("_path"),
        "baseline_backend": baseline.get("backend"),
        "candidate_backend": candidate.get("backend"),
        "candidate_compile_mode": candidate.get("compile_mode"),
        "empty_reference_policy": cand_metrics.get("empty_reference_policy") or candidate.get("empty_reference_policy"),
        "evaluated_subset_mismatch": evaluated_subset_mismatch,
        "not_worse_tolerance": float(not_worse_tolerance),
        "per_key": per_key,
        "per_object": per_object,
    }


def render_delta_report(payload: dict[str, Any]) -> str:
    object_rows = []
    for obj, row in payload.get("per_object", {}).items():
        object_rows.append(
            [
                obj,
                row.get("baseline_iou_avg_on_evaluated"),
                row.get("candidate_iou_avg_on_evaluated"),
                row.get("delta"),
                row.get("not_worse"),
                row.get("evaluated_sample_count"),
                row.get("ignored_reference_empty_count"),
                row.get("candidate_nonempty_when_reference_empty_count"),
                row.get("reference_uncertain"),
                row.get("reason"),
            ]
        )
    return "\n".join(
        [
            "# EdgeTAM Original vs Compiled Delta",
            "",
            f"- baseline_backend: `{payload.get('baseline_backend')}`",
            f"- candidate_backend: `{payload.get('candidate_backend')}`",
            f"- candidate_compile_mode: `{payload.get('candidate_compile_mode')}`",
            f"- empty_reference_policy: `{payload.get('empty_reference_policy')}`",
            f"- evaluated_subset_mismatch: `{payload.get('evaluated_subset_mismatch')}`",
            "",
            "Delta is computed only on SAM3.1 reference-nonempty evaluated samples. Reference-empty samples are reported as uncertainty, not correctness evidence.",
            "",
            markdown_table(
                [
                    "object",
                    "baseline_iou",
                    "candidate_iou",
                    "delta",
                    "not_worse",
                    "evaluated",
                    "ref_empty_ignored",
                    "cand_nonempty_ref_empty",
                    "reference_uncertain",
                    "reason",
                ],
                object_rows,
            ),
        ]
    )


def _metric_avg(row: dict[str, Any]) -> float | None:
    if row.get("iou_avg_on_evaluated") is not None:
        return row.get("iou_avg_on_evaluated")
    return (row.get("mask_iou") or {}).get("avg")


def _object_name(key: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    if key.endswith("_obj0"):
        object_count = int(candidate.get("object_count") or baseline.get("object_count") or 0)
        if object_count == 1:
            return candidate.get("object_prompt") or baseline.get("object_prompt") or "obj0"
        return candidate.get("controller_prompt") or baseline.get("controller_prompt") or "obj0"
    if key.endswith("_obj1"):
        return candidate.get("object_prompt") or baseline.get("object_prompt") or "obj1"
    return key.rsplit("_", 1)[-1]


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["_path"] = str(path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--not-worse-tolerance", type=float, default=0.01)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = build_delta_report(
        _load(args.baseline_json),
        _load(args.candidate_json),
        not_worse_tolerance=args.not_worse_tolerance,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_delta_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
