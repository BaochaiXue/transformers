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
    parser.add_argument("--candidate-delta", default=None)
    parser.add_argument("--first-bad-frame", default=None)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    correctness = [compact_correctness_payload(item) for item in load_jsons(args.correctness_json)]
    profiles = load_jsons(args.profile_json)
    iou_ref_summary = load_optional_json(args.iou_ref_summary)
    different_types_summary = load_optional_json(args.different_types_summary)
    candidate_delta = load_optional_json(args.candidate_delta)
    first_bad_frame = load_optional_json(args.first_bad_frame)
    repo = {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
    }
    best_profile = choose_best_profile(profiles)
    sam31_best = choose_sam31_compile_mode(iou_ref_summary)
    full_candidates = [c for c in correctness if c.get("backend") == "hf_batched_multisession"]
    full = choose_full_batched_report(full_candidates)
    full_contract = (full or {}).get("backend_contract") or (full or {}).get("metrics", {}).get("backend_contract") or {}
    if not full_contract and "contract_pass" in (full or {}):
        full_contract = full or {}
    full_usable = bool(
        full
        and full.get("metrics", {}).get("correctness_pass") is True
        and full_contract.get("contract_pass") is True
    )
    full_failure_stage = full_failure_stage_for(full, full_contract)
    full_blocker = full_blocker_for(full, full_contract, first_bad_frame)
    payload = {
        "goal": "original weights + custom batch=3 multi-session runtime",
        "source": {
            "github_repo": "https://github.com/BaochaiXue/transformers/tree/feat/edgetam-batched-multisession-runtime",
            "fork_path": "/home/zhangxinjie/EdgeTAM-HF-batched",
            **repo,
            "modeling_edgetam_video_touched": False,
        },
        "correctness": correctness,
        "profiles": profiles,
        "sam31_replay_iou": iou_ref_summary,
        "different_types_summary": different_types_summary,
        "candidate_delta": candidate_delta,
        "first_bad_frame": first_bad_frame,
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
            "hf_batched_multisession_usable": full_usable,
            "hf_batched_multisession_failure_stage": None if full_usable else full_failure_stage,
            "hf_batched_multisession_reason": "contract pass + correctness pass" if full_usable else full_blocker,
            "hf_batched_multisession_blockers": full_blocker,
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
            "speed_first_usable": any(
                item.get("metrics", {}).get("correctness_gate", {}).get("speed_first")
                and item.get("metrics", {}).get("correctness_pass") is True
                for item in correctness
            ),
            "strict_validated_objects": strict_validated_objects(correctness),
            "reference_uncertain_objects": reference_uncertain_objects(correctness),
            "empty_reference_policy": first_empty_reference_policy(correctness),
            "demo22_final_fps_pending": True,
            "demo22_final_fps_source": "pending full Demo 2.2 profile; replay/component FPS is not final FPS",
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


def compact_correctness_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return payload
    metrics.pop("sample_statuses", None)
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


def choose_full_batched_report(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    def score(item: dict[str, Any]) -> tuple[int, int, int, str]:
        metrics = item.get("metrics") or {}
        contract = item.get("backend_contract") or metrics.get("backend_contract") or {}
        strict = bool(item.get("strict_full_batched"))
        contract_pass = bool(contract.get("contract_pass"))
        correctness_pass = bool(metrics.get("correctness_pass"))
        path = str(item.get("_path") or "")
        return (int(strict), int(contract_pass), int(correctness_pass), path)

    return max(candidates, key=score)


def render(payload: dict[str, Any]) -> str:
    correctness_rows = []
    for item in payload["correctness"]:
        metrics = item.get("metrics", {})
        contract = item.get("backend_contract") or metrics.get("backend_contract") or {}
        if not contract and "contract_pass" in item:
            contract = item
        correctness_rows.append(
            [
                item.get("backend"),
                item.get("compile_mode"),
                metrics.get("correctness_pass"),
                metrics.get("mask_correctness_pass"),
                metrics.get("candidate_partial"),
                metrics.get("fallback_backend"),
                contract.get("contract_pass"),
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
                ["backend", "compile", "pass", "mask_pass", "partial", "fallback", "contract_pass", "path"],
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
            "## Empty SAM3.1 reference policy",
            "",
            "When SAM3.1 reference is empty, EdgeTAM candidate output is ignored for IoU and empty-mismatch. The sample is marked reference_absent_ignored. This is not counted as correct; it is unevaluated.",
            "",
            "## Original vs compiled delta on evaluated samples",
            "",
            *(
                render_candidate_delta_section(payload.get("candidate_delta"))
                if payload.get("candidate_delta")
                else ["No original-vs-compiled delta report provided."]
            ),
            "",
            "## Full batched first bad frame",
            "",
            *(
                render_first_bad_frame_section(payload.get("first_bad_frame"))
                if payload.get("first_bad_frame")
                else ["No first-bad-frame report provided."]
            ),
            "",
            "## Decision",
            "",
            markdown_table(["field", "value"], payload["decision"].items()),
        ]
    )


def full_failure_stage_for(full: dict[str, Any] | None, contract: dict[str, Any]) -> str:
    if not full:
        return "missing_full_batched_report"
    if contract.get("contract_pass") is not True:
        return "backend_contract"
    metrics = full.get("metrics") or {}
    if metrics.get("correctness_pass") is not True:
        return "correctness"
    return full.get("failure_stage") or "unknown"


def full_blocker_for(
    full: dict[str, Any] | None,
    contract: dict[str, Any],
    first_bad_frame: dict[str, Any] | None,
) -> str:
    if not full:
        return "no strict full-batched correctness report"
    contract_blockers = contract.get("blockers") or []
    if contract.get("contract_pass") is not True:
        return "; ".join(contract_blockers or ["strict full backend contract failed"])
    metrics = full.get("metrics") or {}
    blockers = list(metrics.get("blockers") or [])
    if first_bad_frame and first_bad_frame.get("first_bad_frame"):
        bad = first_bad_frame["first_bad_frame"]
        blockers.insert(
            0,
            "first bad frame "
            f"{bad.get('frame_idx')} {bad.get('camera')} "
            f"IoU={_round(bad.get('iou'))}, "
            f"component={bad.get('first_diverging_component')}",
        )
    if blockers:
        return "; ".join(str(item) for item in blockers)
    if metrics.get("correctness_pass") is not True:
        global_iou = metrics.get("global_mask_iou") or metrics.get("global_iou_on_evaluated") or {}
        return (
            "strict correctness failed: "
            f"global_iou_avg={_round(global_iou.get('avg'))}, "
            f"global_iou_p50={_round(global_iou.get('p50'))}"
        )
    return "no blocker"


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


def render_candidate_delta_section(summary: dict[str, Any]) -> list[str]:
    rows = []
    for object_name, item in (summary.get("per_object") or {}).items():
        rows.append(
            [
                object_name,
                _round(item.get("baseline_iou_avg_on_evaluated")),
                _round(item.get("candidate_iou_avg_on_evaluated")),
                _round(item.get("delta")),
                item.get("not_worse"),
                item.get("evaluated_sample_count"),
                item.get("ignored_reference_empty_count"),
                item.get("candidate_nonempty_when_reference_empty_count"),
                item.get("reference_uncertain"),
                item.get("reason"),
            ]
        )
    return [
        f"- baseline_backend: `{summary.get('baseline_backend')}`",
        f"- candidate_backend: `{summary.get('candidate_backend')}`",
        f"- candidate_compile_mode: `{summary.get('candidate_compile_mode')}`",
        f"- empty_reference_policy: `{summary.get('empty_reference_policy')}`",
        f"- evaluated_subset_mismatch: `{summary.get('evaluated_subset_mismatch')}`",
        "",
        "Compiled batch vision is compared against original HF public only on SAM3.1 reference-nonempty samples.",
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
            rows,
        ),
    ]


def render_first_bad_frame_section(summary: dict[str, Any]) -> list[str]:
    bad = summary.get("first_bad_frame") if summary else None
    if not bad:
        return ["No frame below the requested IoU threshold was found."]
    diffs = bad.get("field_diffs") or {}
    return [
        markdown_table(
            ["field", "value"],
            [
                ["frame_idx", bad.get("frame_idx")],
                ["camera", bad.get("camera")],
                ["iou", _round(bad.get("iou"))],
                ["first_diverging_component", bad.get("first_diverging_component")],
                ["backend_contract_pass", (summary.get("backend_contract") or {}).get("contract_pass")],
            ],
        ),
        "",
        markdown_table(
            ["tensor", "max_abs_diff", "mean_abs_diff", "p95_abs_diff"],
            [
                [
                    name,
                    _round(diff.get("max_abs_diff")),
                    _round(diff.get("mean_abs_diff")),
                    _round(diff.get("p95_abs_diff")),
                ]
                for name, diff in sorted(diffs.items())
            ],
        ),
    ]


def first_empty_reference_policy(correctness: list[dict[str, Any]]) -> str | None:
    for item in correctness:
        metrics = item.get("metrics") or {}
        policy = metrics.get("empty_reference_policy") or item.get("empty_reference_policy")
        if policy:
            return policy
    return None


def strict_validated_objects(correctness: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for item in correctness:
        metrics = item.get("metrics") or {}
        for obj_key, summary in (metrics.get("object_summaries") or {}).items():
            if summary.get("object_reference_absent"):
                continue
            if summary.get("reference_uncertain"):
                continue
            if int(summary.get("evaluated_sample_count") or 0) <= 0:
                continue
            labels.add(_object_label(item, obj_key))
    return sorted(labels)


def reference_uncertain_objects(correctness: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for item in correctness:
        metrics = item.get("metrics") or {}
        for obj_key, summary in (metrics.get("object_summaries") or {}).items():
            if summary.get("reference_uncertain") or summary.get("object_reference_absent"):
                labels.add(_object_label(item, obj_key))
    return sorted(labels)


def _object_label(item: dict[str, Any], obj_key: str) -> str:
    try:
        obj_idx = int(str(obj_key).replace("obj", ""))
    except ValueError:
        return str(obj_key)
    object_count = int(item.get("object_count") or item.get("metrics", {}).get("object_count") or 0)
    object_prompt = item.get("object_prompt") or "stuffed animal"
    controller_prompt = item.get("controller_prompt") or "controller"
    if object_count == 1:
        return object_prompt
    if obj_idx == 0:
        return controller_prompt
    if obj_idx == 1:
        return object_prompt
    return f"obj{obj_idx}"


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
