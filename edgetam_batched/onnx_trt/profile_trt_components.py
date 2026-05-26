"""Profile built BatchTam TensorRT engines with trtexec."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgetam_batched.report_utils import write_json, write_markdown

from .build_trt_engine import render
from .trt_engine_utils import load_io_spec, run_command, trtexec_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True)
    parser.add_argument("--engine-path", required=True)
    parser.add_argument("--io-spec", required=True)
    parser.add_argument("--export-profile", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    io_spec = load_io_spec(args.io_spec)
    if trtexec_path() is None:
        payload = {
            "component": args.component,
            "engine_path": args.engine_path,
            "trt_profile_pass": False,
            "failure_stage": "trt_profile",
            "exact_blocker": "trtexec not found in PATH",
        }
    elif not Path(args.engine_path).exists():
        payload = {
            "component": args.component,
            "engine_path": args.engine_path,
            "trt_profile_pass": False,
            "failure_stage": "trt_profile",
            "exact_blocker": "engine file missing",
        }
    else:
        cmd = [
            trtexec_path() or "trtexec",
            f"--loadEngine={args.engine_path}",
            "--dumpProfile",
            f"--exportProfile={args.export_profile}",
        ]
        for shape_arg in [arg.replace("--minShapes", "--shapes") for arg in []]:
            cmd.append(shape_arg)
        result = run_command(cmd)
        payload = {
            "component": args.component,
            "engine_path": args.engine_path,
            "profile_path": args.export_profile,
            "io_spec": io_spec.to_json(),
            "command": result["command"],
            "returncode": result["returncode"],
            "stdout_tail": (result.get("stdout") or "")[-8000:],
            "trt_profile_pass": result["returncode"] == 0,
            "failure_stage": None if result["returncode"] == 0 else "trt_profile",
            "exact_blocker": None if result["returncode"] == 0 else result.get("error") or "trtexec profile failed",
        }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render({**payload, "trt_build_pass": payload.get("trt_profile_pass")}))
    if args.debug:
        print(payload)
    return 0 if payload.get("trt_profile_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
