"""Closed-loop BatchTam TRT profile entrypoint.

This module is intentionally gated on closed-loop TRT correctness. The current
toolchain records a precise blocker if the TRT-backed scheduler has not been
enabled yet, instead of profiling a PyTorch fallback.
"""

from __future__ import annotations

import argparse

from edgetam_batched.report_utils import markdown_table, write_json, write_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="hf_batched_multisession_trt_components")
    parser.add_argument("--component-runtime", default="trt")
    parser.add_argument("--trt-engine-dir", required=True)
    parser.add_argument("--trt-scope", default="memory_path_all")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--debug", action="store_true")
    _args, _unknown = parser.parse_known_args()
    payload = {
        "backend": _args.backend,
        "component_runtime": _args.component_runtime,
        "trt_engine_dir": _args.trt_engine_dir,
        "trt_scope": _args.trt_scope,
        "profile_pass": False,
        "failure_stage": "closed_loop_trt_runtime",
        "exact_blocker": (
            "TRT component scheduler integration is not enabled until ONNX export, "
            "TRT build, component validation, and closed-loop strict correctness pass."
        ),
    }
    write_json(_args.output_json, payload)
    write_markdown(
        _args.output_md,
        "\n".join(
            [
                "# BatchTam Closed-loop TRT Profile",
                "",
                markdown_table(
                    ["field", "value"],
                    [
                        ["backend", payload["backend"]],
                        ["component_runtime", payload["component_runtime"]],
                        ["trt_scope", payload["trt_scope"]],
                        ["profile_pass", payload["profile_pass"]],
                        ["failure_stage", payload["failure_stage"]],
                        ["exact_blocker", payload["exact_blocker"]],
                    ],
                ),
            ]
        ),
    )
    if _args.debug:
        print(payload)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
