"""Summarize whether traced EdgeTAM components have a proven batch equivalence path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .component_adapter import EdgeTamComponentAdapter
from .report_utils import markdown_table, write_json, write_markdown


COMPONENT_REQUIREMENTS = {
    "vision_encoder": "batch path exists through model.get_image_features(pixel_values[B,...])",
    "memory_attention": "requires tensorized memory/object-pointer inputs across camera sessions",
    "mask_decoder": "requires tensorized prompted/empty-tracking decoder inputs across camera sessions",
    "memory_encoder": "EdgeTamVideoModel._batch_encode_memories is NotImplemented in modular source",
}


def analyze_component(trace: dict[str, Any], component: str) -> dict[str, Any]:
    records = [r for r in trace.get("records", []) if r.get("component") == component]
    if component == "vision_encoder":
        passed = bool(records)
        blockers = [] if passed else ["no vision_encoder calls found in trace"]
    elif component == "memory_encoder":
        passed = False
        blockers = ["EdgeTamVideoModel._batch_encode_memories is NotImplemented; full state update cannot be batched yet"]
    else:
        passed = False
        blockers = [COMPONENT_REQUIREMENTS[component], "raw tensor equivalence is not proven by shape-only trace"]
    return {
        "component": component,
        "records": len(records),
        "equivalence_pass": passed,
        "requirement": COMPONENT_REQUIREMENTS.get(component, ""),
        "blockers": blockers,
    }


def render(payload: dict[str, Any]) -> str:
    rows = [
        [row["component"], row["records"], row["equivalence_pass"], "; ".join(row["blockers"])]
        for row in payload["components"]
    ]
    return "\n".join(
        [
            "# EdgeTAM Component Batch Equivalence",
            "",
            f"- trace: `{payload.get('trace_json')}`",
            f"- fixtures_dir: `{payload.get('fixtures_dir')}`",
            f"- stack_strategy: `{payload.get('stack_strategy')}`",
            f"- all_components_pass: `{payload['all_components_pass']}`",
            f"- groups: `{payload.get('groups')}`",
            f"- groups_passed: `{payload.get('groups_passed')}`",
            f"- max_abs_diff: `{payload.get('max_abs_diff')}`",
            f"- p95_abs_diff: `{payload.get('p95_abs_diff')}`",
            "",
            markdown_table(["component", "trace_records", "pass", "blockers"], rows),
        ]
    )


def run_fixture_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import EdgeTamVideoModel

    dtype = _torch_dtype(torch, args.dtype)
    model = EdgeTamVideoModel.from_pretrained(args.model_id).to(device=args.device, dtype=dtype).eval()
    adapter = EdgeTamComponentAdapter(model)
    component = getattr(adapter, args.component)
    if component is None:
        return {
            "fixtures_dir": args.fixtures_dir,
            "component": args.component,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "strategy_results": [],
            "all_components_pass": False,
            "components": [
                {
                    "component": args.component,
                    "records": 0,
                    "equivalence_pass": False,
                    "requirement": COMPONENT_REQUIREMENTS.get(args.component, ""),
                    "blockers": [f"component {args.component} not found on model"],
                }
            ],
        }

    groups = load_fixture_groups(args.fixtures_dir, args.component, args.batch_size, args.frames)
    strategy = default_strategy(args.component) if args.stack_strategy == "auto" else args.stack_strategy
    rows = []
    blockers = []
    with torch.inference_mode():
        for group in groups:
            try:
                result = run_one_fixture_group(
                    component=component,
                    fixtures=group,
                    component_name=args.component,
                    strategy=strategy,
                    device=args.device,
                    dtype=dtype,
                )
                rows.append(result)
            except Exception as exc:  # noqa: BLE001 - report exact fixture blocker
                blockers.append(
                    {
                        "frame_idx": group[0].get("frame_idx") if group else None,
                        "reason": repr(exc),
                    }
                )
    pass_rows = [row for row in rows if row["pass"]]
    all_pass = bool(groups) and len(pass_rows) == len(groups) and not blockers
    diff_values = [row["max_abs_diff"] for row in rows if row.get("max_abs_diff") is not None]
    component_summary = {
        "component": args.component,
        "records": len(groups) * args.batch_size,
        "equivalence_pass": all_pass,
        "requirement": COMPONENT_REQUIREMENTS.get(args.component, ""),
        "blockers": [item["reason"] for item in blockers[:5]]
        or ([] if all_pass else ["one or more fixture groups failed equivalence"]),
    }
    return {
        "fixtures_dir": args.fixtures_dir,
        "component": args.component,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "stack_strategy": strategy,
        "groups": len(groups),
        "groups_passed": len(pass_rows),
        "all_components_pass": all_pass,
        "max_abs_diff": max(diff_values) if diff_values else None,
        "mean_abs_diff": sum(row["mean_abs_diff"] for row in rows) / len(rows) if rows else None,
        "p95_abs_diff": max(row["p95_abs_diff"] for row in rows) if rows else None,
        "strategy_results": rows[:20],
        "blockers": blockers[:20],
        "components": [component_summary],
    }


def load_fixture_groups(fixtures_dir: str | Path, component: str, batch_size: int, frames: int) -> list[list[dict[str, Any]]]:
    import torch

    root = Path(fixtures_dir)
    groups = []
    for frame_dir in sorted(root.glob("frame_*"))[:frames]:
        group = []
        for cam_idx in range(batch_size):
            matches = sorted(frame_dir.glob(f"cam{cam_idx}_{component}*.pt"))
            if not matches:
                group = []
                break
            group.append(torch.load(matches[0], map_location="cpu", weights_only=False))
        if len(group) == batch_size:
            groups.append(group)
    return groups


def run_one_fixture_group(
    *,
    component: Any,
    fixtures: list[dict[str, Any]],
    component_name: str,
    strategy: str,
    device: str,
    dtype: Any,
) -> dict[str, Any]:
    import torch

    args_list = [move_nested(fixture["args"], device=device, dtype=dtype) for fixture in fixtures]
    kwargs_list = [move_nested(fixture["kwargs"], device=device, dtype=dtype) for fixture in fixtures]
    expected_outputs = [move_nested(fixture["output"], device=device, dtype=dtype) for fixture in fixtures]
    args_batched, kwargs_batched = stack_component_inputs(args_list, kwargs_list, component_name, strategy)
    actual_output = component(*args_batched, **kwargs_batched)
    split_outputs = split_component_output(actual_output, component_name, len(fixtures), strategy)
    stats = compare_output_list(expected_outputs, split_outputs)
    return {
        "frame_idx": fixtures[0]["frame_idx"],
        "strategy": strategy,
        "pass": stats["pass"],
        "max_abs_diff": stats["max_abs_diff"],
        "mean_abs_diff": stats["mean_abs_diff"],
        "p95_abs_diff": stats["p95_abs_diff"],
    }


def default_strategy(component: str) -> str:
    if component == "memory_attention":
        return "seq_first_dim1"
    return "batch_dim0"


def stack_component_inputs(args_list: list[Any], kwargs_list: list[Any], component: str, strategy: str) -> tuple[Any, Any]:
    if strategy == "seq_first_dim1":
        return stack_seq_first(args_list), stack_seq_first(kwargs_list)
    if strategy == "batch_dim0":
        return stack_batch_dim0(args_list), stack_batch_dim0(kwargs_list)
    raise ValueError(f"unsupported stack strategy: {strategy}")


def stack_seq_first(values: list[Any]) -> Any:
    import torch

    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim >= 3 and first.shape[1] == 1 and all(value.shape == first.shape for value in values):
            return torch.cat(values, dim=1)
        if first.ndim >= 1 and first.shape[0] == 1 and all(value.shape == first.shape for value in values):
            return torch.cat(values, dim=0)
        raise ValueError(f"cannot seq-first batch tensor shape {tuple(first.shape)}")
    return stack_metadata_or_nested(values, stack_seq_first)


def stack_batch_dim0(values: list[Any]) -> Any:
    import torch

    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim >= 1 and first.shape[0] == 1 and all(value.shape == first.shape for value in values):
            return torch.cat(values, dim=0)
        raise ValueError(f"cannot dim0 batch tensor shape {tuple(first.shape)}")
    return stack_metadata_or_nested(values, stack_batch_dim0)


def stack_metadata_or_nested(values: list[Any], stack_fn) -> Any:
    first = values[0]
    if first is None or isinstance(first, (bool, int, float, str)):
        if all(value == first for value in values):
            return first
        raise ValueError(f"metadata mismatch: {values}")
    if isinstance(first, tuple):
        return tuple(stack_fn([value[idx] for value in values]) for idx in range(len(first)))
    if isinstance(first, list):
        return [stack_fn([value[idx] for value in values]) for idx in range(len(first))]
    if isinstance(first, dict):
        return {key: stack_fn([value[key] for value in values]) for key in first}
    raise ValueError(f"unsupported nested value type {type(first).__name__}")


def split_component_output(output: Any, component: str, batch_size: int, strategy: str) -> list[Any]:
    if component == "memory_attention":
        return split_nested_output(output, batch_size, dim=1)
    return split_nested_output(output, batch_size, dim=0)


def split_nested_output(value: Any, batch_size: int, dim: int) -> list[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        if value.shape[dim] != batch_size:
            raise ValueError(f"output tensor shape {tuple(value.shape)} does not have batch {batch_size} at dim {dim}")
        return [value.select(dim, idx).unsqueeze(dim).contiguous() for idx in range(batch_size)]
    if isinstance(value, tuple):
        parts = [split_nested_output(item, batch_size, dim) for item in value]
        return [tuple(part[idx] for part in parts) for idx in range(batch_size)]
    if isinstance(value, list):
        parts = [split_nested_output(item, batch_size, dim) for item in value]
        return [[part[idx] for part in parts] for idx in range(batch_size)]
    if isinstance(value, dict):
        split_by_key = {key: split_nested_output(item, batch_size, dim) for key, item in value.items()}
        return [{key: split_by_key[key][idx] for key in split_by_key} for idx in range(batch_size)]
    raise ValueError(f"unsupported output type {type(value).__name__}")


def move_nested(value: Any, *, device: str, dtype: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        target_dtype = dtype if torch.is_floating_point(value) else value.dtype
        return value.to(device=device, dtype=target_dtype)
    if isinstance(value, tuple):
        return tuple(move_nested(item, device=device, dtype=dtype) for item in value)
    if isinstance(value, list):
        return [move_nested(item, device=device, dtype=dtype) for item in value]
    if isinstance(value, dict):
        return {key: move_nested(item, device=device, dtype=dtype) for key, item in value.items()}
    return value


def compare_output_list(expected: list[Any], actual: list[Any]) -> dict[str, Any]:
    diffs = []
    for exp, act in zip(expected, actual, strict=True):
        diffs.extend(tensor_abs_diffs(exp, act))
    if not diffs:
        return {"pass": False, "max_abs_diff": None, "mean_abs_diff": None, "p95_abs_diff": None}
    import torch

    values = torch.cat([diff.flatten().float().cpu() for diff in diffs if diff.numel()])
    if values.numel() == 0:
        return {"pass": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "p95_abs_diff": 0.0}
    max_abs = float(values.max().item())
    mean_abs = float(values.mean().item())
    p95 = float(values.quantile(0.95).item())
    return {
        "pass": max_abs <= 5e-1 and p95 <= 1.25e-1 and mean_abs <= 4e-2,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "p95_abs_diff": p95,
    }


def tensor_abs_diffs(expected: Any, actual: Any) -> list[Any]:
    import torch

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape:
            raise ValueError(f"output shape mismatch {tuple(expected.shape)} vs {tuple(actual.shape)}")
        return [(expected.float() - actual.float()).abs()]
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        return [diff for e, a in zip(expected, actual, strict=True) for diff in tensor_abs_diffs(e, a)]
    if isinstance(expected, list) and isinstance(actual, list):
        return [diff for e, a in zip(expected, actual, strict=True) for diff in tensor_abs_diffs(e, a)]
    if isinstance(expected, dict) and isinstance(actual, dict):
        return [diff for key in expected for diff in tensor_abs_diffs(expected[key], actual[key])]
    if expected is None and actual is None:
        return []
    raise ValueError(f"output type mismatch {type(expected).__name__} vs {type(actual).__name__}")


def _torch_dtype(torch_module, name: str):
    lowered = name.lower().replace("torch.", "")
    if lowered in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch_module.float16
    if lowered in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", default=None)
    parser.add_argument("--component", default=None)
    parser.add_argument("--fixtures-dir", default=None)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--stack-strategy", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.fixtures_dir:
        if not args.component:
            raise ValueError("--fixtures-dir requires --component")
        payload = run_fixture_equivalence(args)
    else:
        if not args.trace_json:
            raise ValueError("either --trace-json or --fixtures-dir is required")
        trace = json.loads(Path(args.trace_json).read_text())
        components = [args.component] if args.component else list(COMPONENT_REQUIREMENTS)
        rows = [analyze_component(trace, component) for component in components]
        payload = {
            "trace_json": args.trace_json,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "components": rows,
            "all_components_pass": all(row["equivalence_pass"] for row in rows),
        }
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload["all_components_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
