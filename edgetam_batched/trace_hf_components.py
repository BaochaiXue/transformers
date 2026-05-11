"""Trace real HF EdgeTAM component calls during public session tracking."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .component_adapter import EdgeTamComponentAdapter
from .component_trace import session_digest, trace_components
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames


def render_trace_report(payload: dict) -> str:
    rows = [[name, count] for name, count in payload["component_call_counts"].items()]
    return "\n".join(
        [
            "# HF EdgeTAM Component Trace",
            "",
            f"- rgb_replay: `{payload['rgb_replay']}`",
            f"- frames: `{payload['frames']}`",
            f"- object_count: `{payload['object_count']}`",
            "",
            "## Component Calls",
            "",
            markdown_table(["component", "calls"], rows),
            "",
            "## Full Batching Blockers",
            "",
            "- `EdgeTamVideoModel.forward` loops over objects and calls `_run_single_frame_inference` per object.",
            "- `EdgeTamVideoModel._batch_encode_memories` is currently `NotImplemented` in the modular source.",
            "- Memory selection/object pointer gathering is driven by nested `inference_session.output_dict_per_obj` Python dictionaries.",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-replay", required=True)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default="hand")
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    frames = load_replay_frames(args.rgb_replay, args.frames)
    runtime = HfEdgeTamReferenceRuntime(
        ReferenceRuntimeConfig(
            model_id=args.model_id,
            dtype=args.dtype,
            device=args.device,
            object_prompt=args.object_prompt,
            controller_prompt=args.controller_prompt,
            object_count=args.object_count,
        )
    )
    runtime.load()
    runtime.init_sessions(frames[0])
    adapter = EdgeTamComponentAdapter(runtime.model)
    component_map = {name: module for name, (_, module) in adapter.components.items()}

    before_digest = [session_digest(session) for session in runtime.sessions]
    with trace_components(component_map) as recorder:
        for frame in frames:
            runtime.step_public(frame)
    after_digest = [session_digest(session) for session in runtime.sessions]

    records = recorder.to_json()["records"]
    counts = Counter(record["component"] for record in records)
    payload = {
        "rgb_replay": args.rgb_replay,
        "frames": len(frames),
        "object_count": args.object_count,
        "component_call_counts": dict(sorted(counts.items())),
        "records": records,
        "before_session_digest": before_digest,
        "after_session_digest": after_digest,
        "full_batching_blockers": [
            "EdgeTamVideoModel._batch_encode_memories is NotImplemented",
            "memory/object-pointer selection is Python-session-dict driven",
            "current backend uses public session step per camera for decoder/state update",
        ],
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_trace_report(payload))
    if args.debug:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
