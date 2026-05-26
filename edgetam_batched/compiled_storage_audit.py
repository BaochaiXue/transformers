"""Audit storage lifetime for compiled full-batched EdgeTAM outputs."""

from __future__ import annotations

import argparse

from .config import COMPILE_SCOPES
from .precision_policy import PRECISION_POLICY_NAMES
from .report_utils import write_json, write_markdown
from .storage_alias_audit import render_storage_alias_audit, run_storage_alias_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--backend", default="hf_batched_multisession")
    parser.add_argument("--reference-source", choices=("hf-public", "hf-public-seq"), default="hf-public-seq")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="hand")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--precision-mode", choices=PRECISION_POLICY_NAMES, default="memory_path_fp32")
    parser.add_argument("--compile-mode", required=True)
    parser.add_argument("--compile-scope", choices=COMPILE_SCOPES, default=None)
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.reference_source not in {"hf-public", "hf-public-seq"}:
        raise ValueError("compiled_storage_audit currently supports only HF public reference")
    payload = run_storage_alias_audit(
        rgb_replay=args.rgb_replay,
        backend=args.backend,
        object_count=args.object_count,
        object_prompt=args.object_prompt,
        controller_prompt=args.controller_prompt,
        frames=args.frames,
        dtype=args.dtype,
        device=args.device,
        compile_mode=args.compile_mode,
        precision_mode=args.precision_mode,
        compile_scope=args.compile_scope,
        graph_output_policy=args.graph_output_policy,
    )
    payload["raw_graph_output_stored"] = bool(payload.get("view_found"))
    payload["session_alias"] = bool(payload.get("alias_found"))
    payload["ring_buffer_pass"] = not payload["raw_graph_output_stored"] and not payload["session_alias"]
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_storage_alias_audit(payload))
    if args.debug:
        print(payload)
    return 0 if payload["ring_buffer_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
