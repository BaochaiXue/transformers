"""Golden-trace utilities for HF EdgeTAM component calls."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class ComponentCallRecord:
    component: str
    frame_idx: int | None
    camera_idx: int | None
    input_args: Any
    input_kwargs: Any
    output: Any
    runtime_ms: float


@dataclass
class ComponentTraceRecorder:
    records: list[ComponentCallRecord] = field(default_factory=list)

    def attach(self, module: Any, component_name: str):
        def pre_hook(_module, args, kwargs=None):
            _module.__edgetam_trace_start = time.perf_counter()

        def post_hook(_module, args, kwargs, output):
            start = getattr(_module, "__edgetam_trace_start", time.perf_counter())
            self.records.append(
                ComponentCallRecord(
                    component=component_name,
                    frame_idx=None,
                    camera_idx=None,
                    input_args=summarize_nested(args),
                    input_kwargs=summarize_nested(kwargs or {}),
                    output=summarize_nested(output),
                    runtime_ms=(time.perf_counter() - start) * 1000.0,
                )
            )

        try:
            return module.register_forward_hook(post_hook, with_kwargs=True), module.register_forward_pre_hook(
                pre_hook, with_kwargs=True
            )
        except TypeError:
            return module.register_forward_hook(lambda m, a, o: post_hook(m, a, {}, o)), module.register_forward_pre_hook(
                lambda m, a: pre_hook(m, a, {})
            )

    def to_json(self) -> dict[str, Any]:
        return {"records": [record.__dict__ for record in self.records], "record_count": len(self.records)}


@contextmanager
def trace_components(component_map: dict[str, Any]) -> Iterator[ComponentTraceRecorder]:
    recorder = ComponentTraceRecorder()
    handles = []
    for name, module in component_map.items():
        if module is not None:
            handles.extend(recorder.attach(module, name))
    try:
        yield recorder
    finally:
        for handle in handles:
            handle.remove()


def summarize_nested(value: Any, *, max_depth: int = 8) -> Any:
    return _summarize(value, path="root", depth=0, max_depth=max_depth)


def session_digest(session: Any) -> dict[str, Any]:
    output_counts = {}
    for obj_idx, outputs in getattr(session, "output_dict_per_obj", {}).items():
        output_counts[str(obj_idx)] = {
            "cond": len(outputs.get("cond_frame_outputs", {})),
            "non_cond": len(outputs.get("non_cond_frame_outputs", {})),
        }
    return {
        "num_frames": getattr(session, "num_frames", None),
        "obj_ids": list(getattr(session, "obj_ids", [])),
        "obj_with_new_inputs": sorted(list(getattr(session, "obj_with_new_inputs", []))),
        "output_counts": output_counts,
        "frames_tracked": {
            str(obj_idx): sorted(list(frames.keys()))
            for obj_idx, frames in getattr(session, "frames_tracked_per_obj", {}).items()
        },
    }


def _summarize(value: Any, *, path: str, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        return {"type": type(value).__name__, "path": path, "truncated": True}
    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, tuple):
        return {"type": "tuple", "len": len(value), "items": [_summarize(v, path=f"{path}[{i}]", depth=depth + 1, max_depth=max_depth) for i, v in enumerate(value)]}
    if isinstance(value, list):
        return {"type": "list", "len": len(value), "items": [_summarize(v, path=f"{path}[{i}]", depth=depth + 1, max_depth=max_depth) for i, v in enumerate(value)]}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "len": len(value),
            "items": {
                str(k): _summarize(v, path=f"{path}.{k}", depth=depth + 1, max_depth=max_depth)
                for k, v in value.items()
            },
        }
    if hasattr(value, "__class__"):
        return {"type": value.__class__.__name__}
    return {"type": type(value).__name__}
