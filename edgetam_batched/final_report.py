"""Aggregate EdgeTAM batched correctness/profile reports."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path
from typing import Any

from .report_utils import markdown_table, write_json, write_markdown


def load_jsons(patterns: list[str]) -> list[dict[str, Any]]:
    payloads = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
                payload["_path"] = path
                payloads.append(payload)
            except Exception as exc:
                payloads.append({"_path": path, "load_error": repr(exc)})
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correctness-json", nargs="*", default=[])
    parser.add_argument("--profile-json", nargs="*", default=[])
    parser.add_argument("--iou-ref-summary", default=None)
    parser.add_argument("--different-types-summary", default=None)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    correctness = load_jsons(args.correctness_json)
    profiles = load_jsons(args.profile_json)
    iou_ref_summary = load_optional_json(args.iou_ref_summary)
    different_types_summary = load_optional_json(args.different_types_summary)
    repo = {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
    }
    best_profile = choose_best_profile(profiles)
    sam31_best = choose_sam31_compile_mode(iou_ref_summary)
    full = next((c for c in correctness if c.get("backend") == "hf_batched_multisession"), None)
    payload = {
        "goal": "original weights + custom batch=3 multi-session runtime",
        "source": {
            "fork_path": "/home/zhangxinjie/EdgeTAM-HF-batched",
            **repo,
            "modeling_edgetam_video_touched": False,
        },
        "correctness": correctness,
        "profiles": profiles,
        "sam31_replay_iou": iou_ref_summary,
        "different_types_summary": different_types_summary,
        "best_profile": best_profile,
        "decision": {
            "hf_batch_vision_seq_session_usable": bool(sam31_best),
            "single_object_stuffed_animal_validated": bool(
                (different_types_summary or {})
                .get("decision", {})
                .get("single_object_stuffed_animal_validated")
            ),
            "single_object_30fps_p50_gate": bool(
                (different_types_summary or {})
                .get("single_object", {})
                .get("p50_30fps_gate")
            ),
            "single_object_30fps_p90_gate": bool(
                (different_types_summary or {})
                .get("single_object", {})
                .get("p90_30fps_gate")
            ),
            "single_object_30fps_p95_gate": (
                "pass"
                if (different_types_summary or {})
                .get("single_object", {})
                .get("p95_30fps_gate")
                else "borderline/fail"
            )
            if different_types_summary
            else None,
            "hf_batched_multisession_usable": bool(
                full and full.get("metrics", {}).get("correctness_pass") is True
            ),
            "hf_batched_multisession_reason": (
                "true batched session/memory/object-pointer tensorization is not complete"
            ),
            "faster_than_77_92_ms_baseline": bool(
                best_profile and (best_profile.get("stage_wall_p50_ms") or 1e9) < 77.92
            ),
            "recommended_backend": "hf_batch_vision_seq_session",
            "recommended_compile_mode": (sam31_best or {}).get("compile_mode") or (best_profile or {}).get("compile_mode"),
            "fallback_backend": "hf_batch_vision_seq_session",
            "controller_hand_validated": bool(
                (different_types_summary or {}).get("decision", {}).get("controller_hand_validated")
            ),
            "controller_hand_status": (
                (different_types_summary or {}).get("decision", {}).get("controller_hand_reason")
            ),
            "controller_towel_validated": False,
            "controller_towel_caveat": (
                "SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras; "
                "current quality claim is for stuffed animal only."
            ),
        },
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    print(args.output_md)
    return 0


def load_optional_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["_path"] = path
    return payload


def choose_best_profile(profiles: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for profile in profiles:
        if profile.get("status") != "replay_profile":
            continue
        backend = profile.get("backend")
        timings = profile.get("profile", {}).get("timings_ms") or profile.get("profile", {})
        if profile.get("profile", {}).get("partial"):
            continue
        stage = timings.get("stage_wall_ms") if isinstance(timings, dict) else None
        if not isinstance(stage, dict) or stage.get("p50") is None:
            continue
        candidates.append(
            {
                "backend": backend,
                "compile_mode": profile.get("compile_mode"),
                "stage_wall_p50_ms": stage.get("p50"),
                "stage_wall_p90_ms": stage.get("p90"),
                "path": profile.get("_path"),
            }
        )
    return min(candidates, key=lambda item: item["stage_wall_p50_ms"]) if candidates else None


def choose_sam31_compile_mode(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    results = summary.get("results", {})
    passing = []
    for mode, result in results.items():
        if result.get("correctness_pass") is not True:
            continue
        global_iou = result.get("global_mask_iou", {})
        passing.append(
            {
                "compile_mode": mode,
                "global_iou_avg": global_iou.get("avg"),
                "global_iou_min": global_iou.get("min"),
                "global_iou_p50": global_iou.get("p50"),
            }
        )
    if not passing:
        return None
    for preferred in ("reduce-overhead", "max-autotune-no-cudagraphs", "none"):
        for item in passing:
            if item["compile_mode"] == preferred:
                return item
    return passing[0]


def render(payload: dict[str, Any]) -> str:
    correctness_rows = []
    for item in payload["correctness"]:
        metrics = item.get("metrics", {})
        correctness_rows.append(
            [
                item.get("backend"),
                item.get("compile_mode"),
                metrics.get("correctness_pass"),
                metrics.get("mask_correctness_pass"),
                metrics.get("candidate_partial"),
                metrics.get("fallback_backend"),
                item.get("_path"),
            ]
        )
    profile_rows = []
    for item in payload["profiles"]:
        if item.get("status") != "replay_profile":
            continue
        profile = item.get("profile", {})
        timings = profile.get("timings_ms") or profile
        stage = timings.get("stage_wall_ms") if isinstance(timings, dict) else {}
        profile_rows.append(
            [
                item.get("backend"),
                item.get("compile_mode"),
                stage.get("p50") if isinstance(stage, dict) else None,
                stage.get("p90") if isinstance(stage, dict) else None,
                profile.get("partial"),
                item.get("_path"),
            ]
        )
    sam31_summary = payload.get("sam31_replay_iou")
    different_types_summary = payload.get("different_types_summary")
    sam31_rows = []
    stuffed_rows = []
    controller_empty = False
    if sam31_summary:
        for mode, result in sam31_summary.get("results", {}).items():
            global_iou = result.get("global_mask_iou", {})
            sam31_rows.append(
                [
                    mode,
                    result.get("correctness_pass"),
                    _round(global_iou.get("avg")),
                    _round(global_iou.get("min")),
                    _round(global_iou.get("p50")),
                    result.get("empty_mismatch_count"),
                ]
            )
            stuffed = {item["key"]: item for item in result.get("per_camera_object", [])}
            stuffed_rows.append(
                [
                    mode,
                    _round(stuffed.get("cam0_obj1", {}).get("avg")),
                    _round(stuffed.get("cam1_obj1", {}).get("avg")),
                    _round(stuffed.get("cam2_obj1", {}).get("avg")),
                ]
            )
            controller_rows = [
                stuffed.get("cam0_obj0", {}),
                stuffed.get("cam1_obj0", {}),
                stuffed.get("cam2_obj0", {}),
            ]
            if controller_rows and all(int(row.get("reference_nonempty_count", -1)) == 0 for row in controller_rows):
                controller_empty = True
    return "\n".join(
        [
            "# EdgeTAM batch=3 multi-session final report",
            "",
            "## Goal",
            "",
            "Original weights + custom batch=3 multi-session runtime.",
            "",
            "## Source",
            "",
            markdown_table(["field", "value"], payload["source"].items()),
            "",
            "## Correctness",
            "",
            markdown_table(
                ["backend", "compile", "pass", "mask_pass", "partial", "fallback", "path"],
                correctness_rows,
            ),
            "",
            "## Profiles",
            "",
            markdown_table(["backend", "compile", "p50", "p90", "partial", "path"], profile_rows),
            "",
            "## SAM3.1 replay reference correctness",
            "",
            *(render_sam31_section(sam31_summary, sam31_rows) if sam31_summary else ["No SAM3.1 replay IoU summary provided."]),
            "",
            "## Non-empty object quality: stuffed animal only",
            "",
            *(render_stuffed_section(stuffed_rows) if stuffed_rows else ["No stuffed animal IoU rows available."]),
            "",
            "## Controller/towel caveat",
            "",
            *(
                [
                    "SAM3.1 replay reference marks obj0/controller/towel as empty for all three cameras.",
                    "Therefore obj0 IoU=1.0 is empty-vs-empty and does not validate controller tracking.",
                    "The current replay validates stuffed animal quality, not successful towel tracking.",
                    "A new replay with non-empty towel masks is required before claiming controller-object correctness.",
                ]
                if controller_empty
                else ["Controller/towel reference is not empty in all cameras for the provided summary."]
            ),
            "",
            "## Different-types sloth_set_2 result",
            "",
            *(
                render_different_types_section(different_types_summary)
                if different_types_summary
                else ["No different-types summary provided."]
            ),
            "",
            "## Decision",
            "",
            markdown_table(["field", "value"], payload["decision"].items()),
        ]
    )


def render_sam31_section(summary: dict[str, Any], rows: list[list[Any]]) -> list[str]:
    return [
        f"- reference_source: `{summary.get('reference_source')}`",
        f"- sam31_mask_root: `{summary.get('sam31_mask_root')}`",
        f"- backend: `{summary.get('backend')}`",
        "",
        markdown_table(
            ["compile_mode", "correctness_pass", "global_iou_avg", "global_iou_min", "global_iou_p50", "empty_mismatch"],
            rows,
        ),
    ]


def render_stuffed_section(rows: list[list[Any]]) -> list[str]:
    return [
        markdown_table(
            ["compile_mode", "cam0 stuffed animal IoU", "cam1 stuffed animal IoU", "cam2 stuffed animal IoU"],
            rows,
        )
    ]


def render_different_types_section(summary: dict[str, Any]) -> list[str]:
    single = summary.get("single_object", {})
    stage = single.get("stage_wall_ms", {})
    decision = summary.get("decision", {})
    return [
        f"- replay: `{summary.get('replay')}`",
        "",
        markdown_table(
            ["field", "value"],
            [
                ["single_object_stuffed_animal_validated", decision.get("single_object_stuffed_animal_validated")],
                ["controller_hand_validated", decision.get("controller_hand_validated")],
                ["controller_hand_reason", decision.get("controller_hand_reason")],
                ["backend", single.get("backend")],
                ["compile", single.get("compile_mode")],
                ["stage_wall_p50_ms", _round(stage.get("p50"))],
                ["stage_wall_p90_ms", _round(stage.get("p90"))],
                ["stage_wall_p95_ms", _round(stage.get("p95"))],
                ["complete_group_fps_from_p50", _round(single.get("complete_group_fps_from_p50"))],
                ["p50_30fps_gate", single.get("p50_30fps_gate")],
                ["p90_30fps_gate", single.get("p90_30fps_gate")],
                ["p95_30fps_gate", single.get("p95_30fps_gate")],
            ],
        ),
    ]


def _round(value: Any, digits: int = 5) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
