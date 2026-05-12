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
    parser.add_argument("--teacher-force-json", nargs="*", default=[])
    parser.add_argument("--state-drift-json", default=None)
    parser.add_argument("--state-commit-json", nargs="*", default=[])
    parser.add_argument("--memory-slot-audit-json", default=None)
    parser.add_argument("--component-equivalence-json", nargs="*", default=[])
    parser.add_argument("--current-frame-json", nargs="*", default=[])
    parser.add_argument("--decoder-diff-json", default=None)
    parser.add_argument("--precision-json", nargs="*", default=[])
    parser.add_argument("--batch-order-json", default=None)
    parser.add_argument("--storage-alias-json", default=None)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    correctness = [compact_correctness_payload(item) for item in load_jsons(args.correctness_json)]
    profiles = load_jsons(args.profile_json)
    iou_ref_summary = load_optional_json(args.iou_ref_summary)
    different_types_summary = load_optional_json(args.different_types_summary)
    candidate_delta = load_optional_json(args.candidate_delta)
    first_bad_frame = load_optional_json(args.first_bad_frame)
    teacher_force = [compact_diagnostic_payload(item) for item in load_jsons(args.teacher_force_json)]
    state_drift = compact_diagnostic_payload(load_optional_json(args.state_drift_json))
    state_commit = [compact_diagnostic_payload(item) for item in load_jsons(args.state_commit_json)]
    memory_slot_audit = compact_diagnostic_payload(load_optional_json(args.memory_slot_audit_json))
    component_equivalence = [compact_diagnostic_payload(item) for item in load_jsons(args.component_equivalence_json)]
    current_frame = [compact_diagnostic_payload(item) for item in load_jsons(args.current_frame_json)]
    decoder_diff = compact_diagnostic_payload(load_optional_json(args.decoder_diff_json))
    precision = [compact_diagnostic_payload(item) for item in load_jsons(args.precision_json)]
    batch_order = compact_diagnostic_payload(load_optional_json(args.batch_order_json))
    storage_alias = compact_diagnostic_payload(load_optional_json(args.storage_alias_json))
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
    full_usable = is_strict_full_closed_loop_pass(full, full_contract)
    full_failure_stage = full_failure_stage_for(full, full_contract)
    precision_decision = summarize_full_precision_decision(full_candidates)
    full_bf16_pass = precision_decision["all_bf16_strict_pass"]
    full_fp32_pass = precision_decision["all_fp32_strict_pass"]
    full_blocker = full_blocker_for(full, full_contract, first_bad_frame)
    if precision_decision["recommended_precision_mode"] and not full_usable:
        full_failure_stage = precision_decision["failure_stage"] or full_failure_stage
        if precision_decision["exact_blocker"]:
            full_blocker = precision_decision["exact_blocker"]
    elif full_fp32_pass and not full_bf16_pass:
        full_failure_stage = "precision"
        full_blocker = (
            f"{full_blocker}; bf16 closed-loop strict fails, while diagnostic all-fp32 eager strict passes; "
            "next patch must implement selective mixed memory/decoder path and compiled correctness"
        )
    recommended_compile_mode = (sam31_best or {}).get("compile_mode") or (best_profile or {}).get("compile_mode")
    if full_usable:
        recommended_compile_mode = (
            "reduce-overhead"
            if precision_decision["reduce_overhead_pass"]
            else "max-autotune-no-cudagraphs"
            if precision_decision["max_autotune_no_cudagraphs_pass"]
            else (full or {}).get("compile_mode")
        )
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
        "teacher_force": teacher_force,
        "state_drift": state_drift,
        "state_commit": state_commit,
        "memory_slot_audit": memory_slot_audit,
        "component_equivalence": component_equivalence,
        "current_frame": current_frame,
        "decoder_diff": decoder_diff,
        "precision": precision,
        "batch_order": batch_order,
        "storage_alias": storage_alias,
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
            "hf_batched_multisession_reason": (
                "contract pass + strict compiled closed-loop correctness pass" if full_usable else full_blocker
            ),
            "hf_batched_multisession_blockers": None if full_usable else full_blocker,
            "full_batched_bf16_strict_pass": full_bf16_pass,
            "full_batched_memory_attention_fp32_strict_pass": precision_decision[
                "memory_attention_fp32_strict_pass"
            ],
            "full_batched_decoder_fp32_strict_pass": precision_decision["decoder_fp32_strict_pass"],
            "full_batched_memory_path_fp32_strict_pass": precision_decision["memory_path_fp32_strict_pass"],
            "full_batched_all_fp32_strict_pass": full_fp32_pass,
            "recommended_precision_mode": precision_decision["recommended_precision_mode"],
            "recommended_precision_mode_reason": precision_decision["recommended_precision_mode_reason"],
            "full_batched_compile_max_autotune_no_cudagraphs_pass": precision_decision[
                "max_autotune_no_cudagraphs_pass"
            ],
            "full_batched_compile_default_pass": precision_decision["default_compile_pass"],
            "full_batched_compile_reduce_overhead_pass": precision_decision["reduce_overhead_pass"],
            "full_batched_vs_sam31_not_worse": (
                (candidate_delta or {})
                .get("per_object", {})
                .get("stuffed animal", {})
                .get("not_worse")
            ),
            "full_batched_vs_sam31_delta": (
                (candidate_delta or {})
                .get("per_object", {})
                .get("stuffed animal", {})
                .get("delta")
            ),
            "recurrent_memory_slot_order_pass": (
                None if memory_slot_audit is None else bool(memory_slot_audit.get("pass"))
            ),
            "current_frame_inferred_issue": first_current_frame_issue(current_frame),
            "batch_order_dependent": None if batch_order is None else bool(batch_order.get("order_dependent")),
            "storage_alias_found": None if storage_alias is None else bool(storage_alias.get("alias_found")),
            "recurrent_drift_first_tensor": (
                ((state_drift or {}).get("summary") or {}).get("first_tensor_drift")
            ),
            "faster_than_77_92_ms_baseline": bool(
                best_profile and (best_profile.get("stage_wall_p50_ms") or 1e9) < 77.92
            ),
            "recommended_backend": "hf_batched_multisession" if full_usable else "hf_batch_vision_seq_session",
            "recommended_compile_mode": recommended_compile_mode,
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
    for item in (metrics.get("per_camera_object") or {}).values():
        if isinstance(item, dict):
            item.pop("iou_values_on_evaluated", None)
    return payload


def compact_diagnostic_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    payload.pop("rows", None)
    payload.pop("strategy_results", None)
    variants = payload.pop("variants", None)
    if isinstance(variants, list):
        payload["variants_summary"] = [
            {
                "variant": item.get("variant"),
                "status": item.get("status"),
                "iou_vs_hf_public": item.get("iou_vs_hf_public"),
                "raw_iou_vs_hf_public": item.get("raw_iou_vs_hf_public"),
                "summary_diffs": item.get("summary_diffs"),
            }
            for item in variants
            if isinstance(item, dict)
        ]
    records = payload.get("records")
    if isinstance(records, list) and len(records) > 20:
        payload["records"] = records[:20]
        payload["records_truncated"] = True
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

    def score(item: dict[str, Any]) -> tuple[int, int, int, int, int, int, int, str]:
        metrics = item.get("metrics") or {}
        contract = item.get("backend_contract") or metrics.get("backend_contract") or {}
        strict = bool(item.get("strict_full_batched"))
        closed_loop_reference = item.get("reference_source") in {"hf-public", "hf-public-seq"}
        contract_pass = bool(contract.get("contract_pass"))
        strict_pass = bool(metrics.get("strict_correctness_pass"))
        correctness_pass = bool(metrics.get("correctness_pass"))
        compile_rank = {
            "reduce-overhead": 4,
            "max-autotune-no-cudagraphs": 3,
            "default": 2,
            "none": 1,
        }.get(str(item.get("compile_mode") or "none"), 0)
        precision_rank = {
            "memory_path_fp32": 4,
            "memory_attention_fp32": 3,
            "decoder_fp32": 2,
            "all_fp32": 1,
            "all_bf16": 0,
        }.get(str(item.get("precision_mode") or "all_bf16"), 0)
        path = str(item.get("_path") or "")
        return (
            int(closed_loop_reference),
            int(strict),
            int(contract_pass),
            int(strict_pass),
            int(correctness_pass),
            compile_rank,
            precision_rank,
            path,
        )

    return max(candidates, key=score)


def is_strict_full_closed_loop_pass(full: dict[str, Any] | None, contract: dict[str, Any]) -> bool:
    if not full:
        return False
    metrics = full.get("metrics") or {}
    return bool(
        full.get("backend") == "hf_batched_multisession"
        and full.get("strict_full_batched") is True
        and full.get("reference_source") in {"hf-public", "hf-public-seq"}
        and contract.get("contract_pass") is True
        and metrics.get("strict_correctness_pass") is True
        and full.get("dtype") not in {"float32", "fp32", "torch.float32"}
        and full.get("compile_mode") in {"max-autotune-no-cudagraphs", "reduce-overhead"}
    )


def full_strict_pass_for_dtype(candidates: list[dict[str, Any]], dtype_names: set[str]) -> bool:
    normalized = {name.lower() for name in dtype_names}
    for item in candidates:
        metrics = item.get("metrics") or {}
        contract = item.get("backend_contract") or metrics.get("backend_contract") or {}
        if (
            item.get("backend") == "hf_batched_multisession"
            and item.get("reference_source") in {"hf-public", "hf-public-seq"}
            and contract.get("contract_pass") is True
            and metrics.get("strict_correctness_pass") is True
            and str(item.get("dtype") or "").lower() in normalized
        ):
            return True
    return False


def full_strict_pass_for_precision(
    candidates: list[dict[str, Any]],
    precision_modes: set[str],
    compile_modes: set[str] | None = None,
) -> bool:
    normalized_precision = {name.lower() for name in precision_modes}
    normalized_compile = {name.lower() for name in compile_modes} if compile_modes else None
    for item in candidates:
        metrics = item.get("metrics") or {}
        contract = item.get("backend_contract") or metrics.get("backend_contract") or {}
        precision = str(item.get("precision_mode") or "all_bf16").lower()
        compile_mode = str(item.get("compile_mode") or "none").lower()
        if normalized_compile is not None and compile_mode not in normalized_compile:
            continue
        if (
            item.get("backend") == "hf_batched_multisession"
            and item.get("reference_source") in {"hf-public", "hf-public-seq"}
            and contract.get("contract_pass") is True
            and metrics.get("strict_correctness_pass") is True
            and precision in normalized_precision
        ):
            return True
    return False


def summarize_full_precision_decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eager = [
        item
        for item in candidates
        if item.get("backend") == "hf_batched_multisession"
        and item.get("reference_source") in {"hf-public", "hf-public-seq"}
        and item.get("compile_mode") == "none"
        and ((item.get("backend_contract") or item.get("metrics", {}).get("backend_contract") or {}).get("contract_pass"))
        is True
        and (item.get("metrics") or {}).get("strict_correctness_pass") is True
    ]
    non_all_fp32 = [item for item in eager if item.get("precision_mode") != "all_fp32"]

    def rank(item: dict[str, Any]) -> tuple[float, float, int]:
        metrics = item.get("metrics") or {}
        iou = metrics.get("global_iou_on_evaluated") or metrics.get("global_mask_iou") or {}
        priority = {
            "memory_path_fp32": 4,
            "memory_attention_fp32": 3,
            "decoder_fp32": 2,
            "all_fp32": 1,
        }.get(item.get("precision_mode"), 0)
        return (float(iou.get("avg") or 0.0), float(iou.get("p50") or 0.0), priority)

    best = max(non_all_fp32 or eager, key=rank) if eager else None
    recommended_precision = best.get("precision_mode") if best else None
    max_compile_pass = (
        full_strict_pass_for_precision(
            candidates,
            {recommended_precision},
            {"max-autotune-no-cudagraphs"},
        )
        if recommended_precision
        else False
    )
    reduce_compile_pass = (
        full_strict_pass_for_precision(candidates, {recommended_precision}, {"reduce-overhead"})
        if recommended_precision
        else False
    )
    failure_stage = None
    exact_blocker = None
    if recommended_precision and not (max_compile_pass or reduce_compile_pass):
        failure_stage = "compile_correctness"
        exact_blocker = (
            f"{recommended_precision} eager strict correctness passes, but compiled strict correctness "
            "fails for max-autotune-no-cudagraphs and reduce-overhead; keep ring_buffer and debug "
            "compiled numeric/state lifetime path before profiling."
        )
    elif not recommended_precision and full_strict_pass_for_precision(candidates, {"all_bf16"}, {"none"}) is False:
        failure_stage = "precision"
        exact_blocker = "all_bf16 closed-loop strict fails and no selective mixed precision eager strict pass was found."
    return {
        "all_bf16_strict_pass": full_strict_pass_for_precision(candidates, {"all_bf16"}, {"none"}),
        "memory_attention_fp32_strict_pass": full_strict_pass_for_precision(
            candidates, {"memory_attention_fp32"}, {"none"}
        ),
        "decoder_fp32_strict_pass": full_strict_pass_for_precision(candidates, {"decoder_fp32"}, {"none"}),
        "memory_path_fp32_strict_pass": full_strict_pass_for_precision(candidates, {"memory_path_fp32"}, {"none"}),
        "all_fp32_strict_pass": full_strict_pass_for_precision(candidates, {"all_fp32"}, {"none"}),
        "recommended_precision_mode": recommended_precision,
        "recommended_precision_mode_reason": (
            f"{recommended_precision} is the best non-all-fp32 eager strict pass by global IoU"
            if recommended_precision
            else None
        ),
        "default_compile_pass": (
            full_strict_pass_for_precision(candidates, {recommended_precision}, {"default"})
            if recommended_precision
            else False
        ),
        "max_autotune_no_cudagraphs_pass": max_compile_pass,
        "reduce_overhead_pass": reduce_compile_pass,
        "failure_stage": failure_stage,
        "exact_blocker": exact_blocker,
    }


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
            "## Full batched recurrent drift localization",
            "",
            *render_recurrent_drift_section(payload),
            "",
            "## Current-frame divergence probes",
            "",
            *render_current_frame_probe_section(payload),
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
    if full.get("reference_source") not in {"hf-public", "hf-public-seq"}:
        return "closed_loop_reference_missing"
    if metrics.get("strict_correctness_pass") is not True:
        return "correctness"
    if full.get("dtype") in {"float32", "fp32", "torch.float32"}:
        return "precision"
    if full.get("compile_mode") == "none":
        return "compile"
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
    if metrics.get("strict_correctness_pass") is True and full.get("dtype") in {"float32", "fp32", "torch.float32"}:
        return "strict closed-loop pass only in diagnostic all-fp32; bf16/mixed compiled runtime still required"
    if metrics.get("strict_correctness_pass") is True and full.get("compile_mode") == "none":
        return "strict closed-loop eager pass only; compiled reduce-overhead correctness still required"
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


def render_recurrent_drift_section(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    component_rows = []
    for item in payload.get("component_equivalence") or []:
        component_rows.append(
            [
                item.get("component"),
                item.get("groups"),
                item.get("groups_passed"),
                item.get("all_components_pass"),
                _round(item.get("max_abs_diff")),
                _round(item.get("p95_abs_diff")),
                item.get("_path"),
            ]
        )
    if component_rows:
        lines.extend(
            [
                "### Component equivalence fixtures",
                "",
                markdown_table(
                    ["component", "groups", "passed", "pass", "max_abs_diff", "p95_abs_diff", "path"],
                    component_rows,
                ),
                "",
            ]
        )

    teacher_rows = []
    for item in payload.get("teacher_force") or []:
        bad = item.get("first_bad_frame") or {}
        teacher_rows.append(
            [
                item.get("teacher_force_mode"),
                item.get("status"),
                bad.get("frame_idx"),
                bad.get("camera"),
                _round(bad.get("iou")),
                _round(item.get("global_iou_avg")),
                _round(item.get("global_iou_p50")),
                item.get("drift_source_hypothesis"),
            ]
        )
    if teacher_rows:
        lines.extend(
            [
                "### Teacher forcing",
                "",
                markdown_table(
                    ["mode", "status", "first_bad_frame", "camera", "iou", "avg", "p50", "hypothesis"],
                    teacher_rows,
                ),
                "",
            ]
        )

    drift = payload.get("state_drift") or {}
    summary = drift.get("summary") or {}
    if summary:
        lines.extend(
            [
                "### State drift curve",
                "",
                markdown_table(
                    ["field", "value"],
                    [
                        ["mask_iou_avg", _round((summary.get("mask_iou") or {}).get("avg"))],
                        ["mask_iou_min", _round((summary.get("mask_iou") or {}).get("min"))],
                        ["first_iou_lt_0_90", summary.get("first_iou_lt_0_90")],
                        ["first_iou_lt_0_98", summary.get("first_iou_lt_0_98")],
                        ["first_tensor_drift", summary.get("first_tensor_drift")],
                        [
                            "maskmem_features_first_p95_gt_1",
                            (summary.get("maskmem_features") or {}).get("first_frame_p95_gt_1"),
                        ],
                        [
                            "object_pointer_first_p95_gt_1e_1",
                            (summary.get("object_pointer") or {}).get("first_frame_p95_gt_1e_1"),
                        ],
                        [
                            "maskmem_pos_enc_p95",
                            _round(((summary.get("maskmem_pos_enc") or {}).get("p95_abs_diff") or {}).get("p95")),
                        ],
                    ],
                ),
                "",
            ]
        )

    commit_rows = []
    for item in payload.get("state_commit") or []:
        bad = item.get("first_bad_frame") or {}
        commit_rows.append(
            [
                item.get("state_commit_mode"),
                bad.get("frame_idx"),
                bad.get("camera"),
                _round(bad.get("iou")),
                _round(item.get("global_iou_avg")),
                _round(item.get("global_iou_p50")),
                item.get("interpretation"),
            ]
        )
    if commit_rows:
        lines.extend(
            [
                "### State commit ablation",
                "",
                markdown_table(
                    ["mode", "first_bad_frame", "camera", "iou", "avg", "p50", "interpretation"],
                    commit_rows,
                ),
                "",
            ]
        )

    audit = payload.get("memory_slot_audit")
    if audit:
        lines.extend(
            [
                "### Memory slot audit",
                "",
                markdown_table(
                    ["field", "value"],
                    [
                        ["pass", audit.get("pass")],
                        ["first_mismatch", audit.get("first_mismatch")],
                        ["path", audit.get("_path")],
                    ],
                ),
            ]
        )
    if not lines:
        return ["No recurrent drift localization reports provided."]
    return lines


def render_current_frame_probe_section(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    current_rows = []
    for item in payload.get("current_frame") or []:
        normal = item.get("normal_variant") or {}
        current_rows.append(
            [
                item.get("frame_idx"),
                item.get("camera"),
                item.get("replace"),
                item.get("precision_mode"),
                _round(normal.get("raw_iou_vs_hf_public")),
                item.get("inferred_issue"),
                item.get("_path"),
            ]
        )
    if current_rows:
        lines.extend(
            [
                "### Current-frame isolation",
                "",
                markdown_table(
                    ["frame", "camera", "replace", "precision", "raw_iou", "inferred_issue", "path"],
                    current_rows,
                ),
                "",
            ]
        )

    decoder = payload.get("decoder_diff")
    if decoder:
        lines.extend(
            [
                "### Decoder diff probe",
                "",
                markdown_table(
                    ["field", "value"],
                    [
                        ["raw_iou", _round(decoder.get("raw_iou_vs_hf_public"))],
                        ["first_divergent_tensor", decoder.get("first_divergent_tensor")],
                        ["threshold_flip_count", decoder.get("logit_diff_causes_threshold_flip_count")],
                        ["reference_area", decoder.get("reference_area")],
                        ["candidate_area", decoder.get("candidate_area")],
                        ["bbox_center_distance", _round(decoder.get("bbox_center_distance"))],
                    ],
                ),
                "",
            ]
        )

    precision_rows = []
    for item in payload.get("precision") or []:
        precision_rows.append(
            [
                item.get("precision_mode"),
                item.get("implemented"),
                item.get("effective_dtype"),
                _round(item.get("raw_iou_vs_hf_public")),
                item.get("conclusion"),
                item.get("reason"),
            ]
        )
    if precision_rows:
        lines.extend(
            [
                "### Precision ladder",
                "",
                markdown_table(["mode", "implemented", "dtype", "raw_iou", "conclusion", "reason"], precision_rows),
                "",
            ]
        )

    batch_order = payload.get("batch_order")
    if batch_order:
        lines.extend(
            [
                "### Batch order ablation",
                "",
                markdown_table(
                    ["field", "value"],
                    [
                        ["order_dependent", batch_order.get("order_dependent")],
                        ["diagonal_slicing_bug", batch_order.get("diagonal_slicing_bug")],
                        ["path", batch_order.get("_path")],
                    ],
                ),
                "",
            ]
        )

    storage = payload.get("storage_alias")
    if storage:
        lines.extend(
            [
                "### Storage alias audit",
                "",
                markdown_table(
                    ["field", "value"],
                    [
                        ["alias_found", storage.get("alias_found")],
                        ["alias_record_count", storage.get("alias_record_count")],
                        ["path", storage.get("_path")],
                    ],
                ),
            ]
        )
    if not lines:
        return ["No current-frame divergence probes provided."]
    return lines


def first_current_frame_issue(items: list[dict[str, Any] | None]) -> str | None:
    for item in items:
        if item and item.get("inferred_issue"):
            return item.get("inferred_issue")
    return None


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
