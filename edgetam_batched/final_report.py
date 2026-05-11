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
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    correctness = load_jsons(args.correctness_json)
    profiles = load_jsons(args.profile_json)
    repo = {
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
    }
    best_profile = choose_best_profile(profiles)
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
        "best_profile": best_profile,
        "decision": {
            "hf_batched_multisession_usable": bool(
                full and full.get("metrics", {}).get("correctness_pass") is True
            ),
            "faster_than_77_92_ms_baseline": bool(
                best_profile and (best_profile.get("stage_wall_p50_ms") or 1e9) < 77.92
            ),
            "recommended_backend": (best_profile or {}).get("backend") or "hf_batch_vision_seq_session",
            "fallback_backend": "hf_batch_vision_seq_session",
        },
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    print(args.output_md)
    return 0


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
            "## Decision",
            "",
            markdown_table(["field", "value"], payload["decision"].items()),
        ]
    )


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
