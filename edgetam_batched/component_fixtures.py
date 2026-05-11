"""Capture executable EdgeTAM component fixtures from the HF public runtime."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .component_adapter import EdgeTamComponentAdapter
from .component_trace import session_digest, summarize_nested
from .reference_runtime import HfEdgeTamReferenceRuntime, ReferenceRuntimeConfig, autocast_context
from .report_utils import markdown_table, write_json, write_markdown
from .rgb_replay import load_replay_frames


class ComponentFixtureRecorder:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.current: dict[str, Any] | None = None
        self.pending: list[dict[str, Any]] = []
        self.records: list[dict[str, Any]] = []

    def attach(self, module: Any, component_name: str):
        def hook(_module, args, kwargs, output):
            if self.current is None:
                return
            self.pending.append(
                {
                    "component": component_name,
                    "camera_idx": self.current["camera_idx"],
                    "frame_idx": self.current["frame_idx"],
                    "args": to_cpu_nested(args),
                    "kwargs": to_cpu_nested(kwargs or {}),
                    "output": to_cpu_nested(output),
                    "session_digest_before": self.current["session_digest_before"],
                }
            )

        try:
            return module.register_forward_hook(hook, with_kwargs=True)
        except TypeError:
            return module.register_forward_hook(lambda m, a, o: hook(m, a, {}, o))

    def begin(self, *, frame_idx: int, camera_idx: int, session_digest_before: dict[str, Any]) -> None:
        self.current = {
            "frame_idx": int(frame_idx),
            "camera_idx": int(camera_idx),
            "session_digest_before": session_digest_before,
        }
        self.pending = []

    def end(self, *, session_digest_after: dict[str, Any]) -> None:
        counters: Counter[str] = Counter()
        for item in self.pending:
            component = item["component"]
            counters[component] += 1
            suffix = "" if counters[component] == 1 else f"_call{counters[component]}"
            frame_dir = self.output_dir / f"frame_{int(item['frame_idx']):03d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            path = frame_dir / f"cam{int(item['camera_idx'])}_{component}{suffix}.pt"
            item["session_digest_after"] = session_digest_after
            save_fixture(path, item)
            self.records.append(
                {
                    "path": str(path),
                    "component": component,
                    "camera_idx": item["camera_idx"],
                    "frame_idx": item["frame_idx"],
                    "args_summary": summarize_nested(item["args"]),
                    "kwargs_summary": summarize_nested(item["kwargs"]),
                    "output_summary": summarize_nested(item["output"]),
                }
            )
        self.current = None
        self.pending = []


def save_fixture(path: Path, payload: dict[str, Any]) -> None:
    import torch

    torch.save(payload, path)


def to_cpu_nested(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(to_cpu_nested(item) for item in value)
    if isinstance(value, list):
        return [to_cpu_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: to_cpu_nested(item) for key, item in value.items()}
    if hasattr(value, "to_tuple"):
        return tuple(to_cpu_nested(item) for item in value.to_tuple())
    return value


def render_fixture_report(payload: dict[str, Any]) -> str:
    rows = [[name, count] for name, count in sorted(payload["component_counts"].items())]
    return "\n".join(
        [
            "# EdgeTAM Executable Component Fixtures",
            "",
            f"- rgb_replay: `{payload['rgb_replay']}`",
            f"- output_dir: `{payload['output_dir']}`",
            f"- frames: `{payload['frames']}`",
            f"- object_count: `{payload['object_count']}`",
            f"- fixture_count: `{payload['fixture_count']}`",
            "",
            markdown_table(["component", "fixtures"], rows),
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
    parser.add_argument("--output-dir", required=True)
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
    recorder = ComponentFixtureRecorder(args.output_dir)
    handles = [
        recorder.attach(module, name)
        for name, (_, module) in adapter.components.items()
        if module is not None and name in {"memory_attention", "mask_decoder", "memory_encoder"}
    ]

    try:
        with runtime.torch.inference_mode(), autocast_context(runtime.torch, runtime.config.device, runtime.dtype):
            for frame in frames:
                pixel_values, _ = runtime.preprocess_frames(frame)
                for cam_idx, session in enumerate(runtime.sessions):
                    recorder.begin(
                        frame_idx=int(frame.frame_idx),
                        camera_idx=cam_idx,
                        session_digest_before=session_digest(session),
                    )
                    runtime.model(
                        inference_session=session,
                        frame_idx=int(frame.frame_idx),
                        frame=pixel_values[cam_idx],
                    )
                    if str(runtime.config.device).startswith("cuda"):
                        runtime.torch.cuda.synchronize()
                    recorder.end(session_digest_after=session_digest(session))
    finally:
        for handle in handles:
            handle.remove()

    counts = Counter(item["component"] for item in recorder.records)
    payload = {
        "rgb_replay": args.rgb_replay,
        "output_dir": str(args.output_dir),
        "frames": len(frames),
        "object_count": args.object_count,
        "fixture_count": len(recorder.records),
        "component_counts": dict(counts),
        "records": recorder.records,
    }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render_fixture_report(payload))
    if args.debug:
        print({key: payload[key] for key in ("fixture_count", "component_counts")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
