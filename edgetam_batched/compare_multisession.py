"""Correctness harness entrypoint for batched multi-session EdgeTAM backends."""

from __future__ import annotations

import argparse

from .config import BACKENDS
from .report_utils import write_json, write_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--object-count", type=int, default=2)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="towel")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--compile-mode", default="none")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = {
        "backend": args.backend,
        "frames": args.frames,
        "correctness_pass": False,
        "status": "not_run",
        "reason": "real RGB replay and HF reference stepping are not implemented in this milestone",
        "scope": {
            "object_count": args.object_count,
            "object_prompt": args.object_prompt,
            "controller_prompt": args.controller_prompt,
            "dtype": args.dtype,
            "compile_mode": args.compile_mode,
            "rgb_replay": args.rgb_replay,
        },
    }
    write_json(args.output_json, payload)
    write_markdown(
        args.output_md,
        "\n".join(
            [
                "# EdgeTAM Batched Correctness",
                "",
                f"- backend: `{args.backend}`",
                "- correctness_pass: `false`",
                "- status: `not_run`",
                "",
                "Real 100-frame RGB replay correctness must be implemented before this backend can be claimed usable.",
            ]
        ),
    )
    if args.debug:
        print(payload)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
