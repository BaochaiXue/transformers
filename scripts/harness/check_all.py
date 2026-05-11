"""Deterministic harness checks for the EdgeTAM batched runtime fork."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-generated", action="store_true", help="Do not regenerate docs/generated reports.")
    parser.add_argument("--with-synthetic-profile", action="store_true", help="Run the HF synthetic profile smoke.")
    args = parser.parse_args()

    commands: list[list[str]] = [
        [sys.executable, "-m", "py_compile", *sorted(str(path) for path in (ROOT / "edgetam_batched").glob("*.py"))],
        [
            sys.executable,
            "-m",
            "unittest",
            "-v",
            "tests.test_state_map",
            "tests.test_camera_order",
            "tests.test_leakage_test",
            "tests.test_profile_stats",
            "tests.test_ring_buffer",
            "tests.test_compile_config",
            "tests.test_batched_runtime_shapes",
            "tests.test_component_adapter",
            "tests.test_compare_multisession",
            "tests.test_profile_multisession",
        ],
    ]

    if not args.skip_generated:
        commands.extend(
            [
                [
                    sys.executable,
                    "-m",
                    "edgetam_batched.source_inspection",
                    "--repo-root",
                    str(ROOT),
                    "--output-md",
                    "docs/generated/edgetam_batched_source_inspection.md",
                    "--output-json",
                    "docs/generated/edgetam_batched_source_inspection.json",
                ],
                [
                    sys.executable,
                    "-m",
                    "edgetam_batched.state_map",
                    "--camera-count",
                    "3",
                    "--object-count",
                    "2",
                    "--object-prompt",
                    "stuffed animal",
                    "--controller-prompt",
                    "towel",
                    "--dtype",
                    "bfloat16",
                    "--frames-for-mutation-check",
                    "3",
                    "--output-md",
                    "docs/generated/edgetam_batched_session_state_map.md",
                    "--output-json",
                    "docs/generated/edgetam_batched_session_state_map.json",
                ],
                [
                    sys.executable,
                    "-m",
                    "edgetam_batched.profile_multisession",
                    "--backend",
                    "hf_batch_vision_seq_session",
                    "--frames",
                    "100",
                    "--warmup",
                    "20",
                    "--object-count",
                    "2",
                    "--dtype",
                    "bfloat16",
                    "--compile-mode",
                    "none",
                    "--graph-output-policy",
                    "ring_buffer",
                    "--output-json",
                    "docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none_scaffold.json",
                    "--output-md",
                    "docs/generated/edgetam_batched_profile_hf_batch_vision_seq_session_none_scaffold.md",
                ],
            ]
        )

    if args.with_synthetic_profile:
        commands.append(
            [
                sys.executable,
                "-m",
                "edgetam_batched.profile_multisession",
                "--backend",
                "hf_ref_seq_public",
                "--frames",
                "10",
                "--warmup",
                "3",
                "--object-count",
                "2",
                "--dtype",
                "bfloat16",
                "--compile-mode",
                "none",
                "--graph-output-policy",
                "ring_buffer",
                "--synthetic-hf-public",
                "--output-json",
                "docs/generated/edgetam_batched_profile_hf_ref_seq_public_synthetic.json",
                "--output-md",
                "docs/generated/edgetam_batched_profile_hf_ref_seq_public_synthetic.md",
            ]
        )

    for command in commands:
        print("+", " ".join(command), flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
