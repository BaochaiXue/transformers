"""Summarize different-types EdgeTAM batch-vision validation results."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from .report_utils import markdown_table, write_json, write_markdown


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expand_paths(patterns: list[str] | None) -> list[str]:
    if not patterns:
        return []
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def per_object_rows(payload: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for key, metrics in sorted(payload.get("metrics", {}).get("per_camera_object", {}).items()):
        iou = metrics.get("mask_iou", {})
        rows.append(
            [
                key,
                _round(iou.get("avg")),
                _round(iou.get("min")),
                _round(iou.get("p50")),
                metrics.get("reference_nonempty_count"),
                metrics.get("candidate_nonempty_count"),
            ]
        )
    return rows


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    single = load_json(args.single_correctness_json)
    profile = load_json(args.single_profile_json)
    two_paths = expand_paths(args.two_object_correctness_json)
    two_results = [load_json(path) | {"_path": path} for path in two_paths]

    single_metrics = single.get("metrics", {})
    single_profile = profile.get("profile", {})
    profile_timings = single_profile.get("timings_ms", {})
    stage = profile_timings.get("stage_wall_ms", {})
    p50 = stage.get("p50")
    p90 = stage.get("p90")
    p95 = stage.get("p95")
    fps = single_profile.get("complete_group_fps_from_p50")

    two_reduce = next(
        (
            item
            for item in two_results
            if item.get("compile_mode") == "reduce-overhead"
            or str(item.get("_path", "")).endswith("reduce_overhead.json")
        ),
        two_results[0] if two_results else None,
    )
    hand_rows = []
    object_rows = []
    if two_reduce:
        for key, metrics in sorted(two_reduce.get("metrics", {}).get("per_camera_object", {}).items()):
            row = {
                "key": key,
                "avg": metrics.get("mask_iou", {}).get("avg"),
                "min": metrics.get("mask_iou", {}).get("min"),
                "p50": metrics.get("mask_iou", {}).get("p50"),
                "reference_nonempty_count": metrics.get("reference_nonempty_count"),
                "candidate_nonempty_count": metrics.get("candidate_nonempty_count"),
            }
            if key.endswith("_obj0"):
                hand_rows.append(row)
            elif key.endswith("_obj1"):
                object_rows.append(row)

    payload = {
        "replay": args.replay,
        "single_object": {
            "object": "stuffed animal",
            "backend": single.get("backend"),
            "compile_mode": single.get("compile_mode"),
            "correctness_pass": single_metrics.get("correctness_pass"),
            "empty_mismatch_count": single_metrics.get("empty_mismatch_count"),
            "per_camera_object": single_metrics.get("per_camera_object", {}),
            "stage_wall_ms": stage,
            "complete_group_fps_from_p50": fps,
            "p50_30fps_gate": bool(p50 is not None and p50 <= 33.3333333333),
            "p90_30fps_gate": bool(p90 is not None and p90 <= 33.3333333333),
            "p95_30fps_gate": bool(p95 is not None and p95 <= 33.3333333333),
        },
        "two_object": {
            "controller": "hand",
            "object": "stuffed animal",
            "results": two_results,
            "reduce_overhead_hand_rows": hand_rows,
            "reduce_overhead_object_rows": object_rows,
        },
        "decision": {
            "single_object_stuffed_animal_validated": bool(single_metrics.get("correctness_pass") is True),
            "controller_hand_validated": False,
            "controller_hand_reason": "low IoU outliers on cam0/cam2",
            "best_single_object_backend": "hf_batch_vision_seq_session",
            "recommended_compile_mode": "reduce-overhead",
            "hf_batched_multisession_usable": False,
        },
        "inputs": {
            "single_correctness_json": args.single_correctness_json,
            "two_object_correctness_json": two_paths,
            "single_profile_json": args.single_profile_json,
            "sam31_init_md": args.sam31_init_md,
        },
    }
    return payload


def render(payload: dict[str, Any]) -> str:
    single = payload["single_object"]
    stage = single["stage_wall_ms"]
    two = payload["two_object"]
    compile_rows = []
    for item in two.get("results", []):
        metrics = item.get("metrics", {})
        global_iou = metrics.get("global_mask_iou", {})
        compile_rows.append(
            [
                item.get("compile_mode"),
                metrics.get("correctness_pass"),
                _round(global_iou.get("avg")),
                _round(global_iou.get("min")),
                _round(global_iou.get("p50")),
                metrics.get("empty_mismatch_count"),
                item.get("_path"),
            ]
        )
    hand_rows = [
        [
            item["key"],
            _round(item["avg"]),
            _round(item["min"]),
            _round(item["p50"]),
            item["reference_nonempty_count"],
            item["candidate_nonempty_count"],
        ]
        for item in two.get("reduce_overhead_hand_rows", [])
    ]
    object_rows = [
        [
            item["key"],
            _round(item["avg"]),
            _round(item["min"]),
            _round(item["p50"]),
            item["reference_nonempty_count"],
            item["candidate_nonempty_count"],
        ]
        for item in two.get("reduce_overhead_object_rows", [])
    ]
    return "\n".join(
        [
            "# Different-types sloth_set_2 EdgeTAM batch=3 summary",
            "",
            f"- replay: `{payload['replay']}`",
            "",
            "## Primary Single-object Result",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["object", single["object"]],
                    ["backend", single["backend"]],
                    ["compile", single["compile_mode"]],
                    ["correctness_pass", single["correctness_pass"]],
                    ["empty_mismatch", single["empty_mismatch_count"]],
                    ["stage_wall_p50_ms", _round(stage.get("p50"))],
                    ["stage_wall_p90_ms", _round(stage.get("p90"))],
                    ["stage_wall_p95_ms", _round(stage.get("p95"))],
                    ["complete_group_fps_from_p50", _round(single["complete_group_fps_from_p50"])],
                    ["p50_30fps_gate", single["p50_30fps_gate"]],
                    ["p90_30fps_gate", single["p90_30fps_gate"]],
                    ["p95_30fps_gate", single["p95_30fps_gate"]],
                ],
            ),
            "",
            "## Single-object Per-camera IoU",
            "",
            markdown_table(
                ["key", "iou_avg", "iou_min", "iou_p50", "ref_nonempty", "cand_nonempty"],
                per_object_rows({"metrics": {"per_camera_object": single["per_camera_object"]}}),
            ),
            "",
            "## Two-object Compile Matrix",
            "",
            markdown_table(
                ["compile", "pass", "global_avg", "global_min", "global_p50", "empty_mismatch", "path"],
                compile_rows,
            ),
            "",
            "## Hand Low-IoU Status",
            "",
            markdown_table(
                ["key", "iou_avg", "iou_min", "iou_p50", "ref_nonempty", "cand_nonempty"],
                hand_rows,
            ),
            "",
            "## Stuffed Animal In Two-object Run",
            "",
            markdown_table(
                ["key", "iou_avg", "iou_min", "iou_p50", "ref_nonempty", "cand_nonempty"],
                object_rows,
            ),
            "",
            "## Decision",
            "",
            markdown_table(["field", "value"], payload["decision"].items()),
        ]
    )


def _round(value: Any, digits: int = 5) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--single-correctness-json", required=True)
    parser.add_argument("--two-object-correctness-json", nargs="+", required=True)
    parser.add_argument("--single-profile-json", required=True)
    parser.add_argument("--sam31-init-md", default=None)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    payload = build_summary(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
