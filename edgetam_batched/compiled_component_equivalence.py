"""Check compiled component output equivalence against eager component fixtures."""

from __future__ import annotations

import argparse
from typing import Any

from .component_adapter import EdgeTamComponentAdapter
from .component_batch_equivalence import (
    compare_output_list,
    default_strategy,
    load_fixture_groups,
    move_nested,
    split_component_output,
    stack_component_inputs,
)
from .config import COMPILE_MODES
from .precision_policy import PRECISION_POLICY_NAMES, policy_torch_dtype, resolve_precision_policy
from .report_utils import markdown_table, write_json, write_markdown
from .stats import summarize


COMPONENT_POLICY_DTYPE_FIELD = {
    "memory_attention": "memory_attention_dtype",
    "mask_decoder": "mask_decoder_dtype",
    "memory_encoder": "memory_encoder_dtype",
    "vision_encoder": "vision_dtype",
}


def run_compiled_component_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import EdgeTamVideoModel

    base_dtype = _torch_dtype(torch, args.dtype)
    policy = resolve_precision_policy(args.precision_mode)
    component_dtype = policy_torch_dtype(torch, getattr(policy, COMPONENT_POLICY_DTYPE_FIELD.get(args.component, "vision_dtype")))
    model = EdgeTamVideoModel.from_pretrained(args.model_id).to(device=args.device, dtype=base_dtype).eval()
    _apply_component_precision(model, args.component, component_dtype)
    adapter = EdgeTamComponentAdapter(model)
    component = getattr(adapter, args.component, None)
    if component is None:
        return _failure_payload(args, f"component {args.component} not found")

    try:
        compiled_component = torch.compile(component, mode=args.compile_mode) if args.compile_mode != "none" else component
        compile_error = None
    except Exception as exc:  # noqa: BLE001
        compiled_component = component
        compile_error = repr(exc)

    groups = load_fixture_groups(args.fixtures_dir, args.component, args.batch_size, args.frames)
    strategy = default_strategy(args.component) if args.stack_strategy == "auto" else args.stack_strategy
    rows = []
    blockers = []
    with torch.inference_mode():
        for group in groups:
            try:
                rows.append(
                    run_one_group(
                        eager_component=component,
                        compiled_component=compiled_component,
                        fixtures=group,
                        component_name=args.component,
                        strategy=strategy,
                        device=args.device,
                        dtype=component_dtype,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                blockers.append(
                    {
                        "frame_idx": group[0].get("frame_idx") if group else None,
                        "reason": repr(exc),
                    }
                )
    eager_pass = [row for row in rows if row.get("eager_pass")]
    compiled_pass = [row for row in rows if row.get("compiled_pass")]
    eager_compiled_pass = [row for row in rows if row.get("eager_compiled_pass")]
    pass_all = bool(groups) and len(compiled_pass) == len(groups) and len(eager_compiled_pass) == len(groups)
    if compile_error:
        pass_all = False
        blockers.insert(0, {"frame_idx": None, "reason": f"compile failed: {compile_error}"})
    return {
        "fixtures_dir": args.fixtures_dir,
        "component": args.component,
        "frames_requested": args.frames,
        "groups": len(groups),
        "groups_eager_passed": len(eager_pass),
        "groups_compiled_passed": len(compiled_pass),
        "groups_eager_compiled_passed": len(eager_compiled_pass),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "component_dtype": str(component_dtype),
        "precision_mode": args.precision_mode,
        "compile_mode": args.compile_mode,
        "graph_output_policy": args.graph_output_policy,
        "stack_strategy": strategy,
        "compiled_component_equivalence_pass": pass_all,
        "all_components_pass": pass_all,
        "compile_error": compile_error,
        "blockers": blockers[:20],
        "rows": rows[:20],
        "compiled_max_abs_diff": summarize(row["compiled_max_abs_diff"] for row in rows if row.get("compiled_max_abs_diff") is not None),
        "compiled_p95_abs_diff": summarize(row["compiled_p95_abs_diff"] for row in rows if row.get("compiled_p95_abs_diff") is not None),
        "eager_compiled_max_abs_diff": summarize(
            row["eager_compiled_max_abs_diff"] for row in rows if row.get("eager_compiled_max_abs_diff") is not None
        ),
    }


def run_one_group(
    *,
    eager_component: Any,
    compiled_component: Any,
    fixtures: list[dict[str, Any]],
    component_name: str,
    strategy: str,
    device: str,
    dtype: Any,
) -> dict[str, Any]:
    args_list = [move_nested(fixture["args"], device=device, dtype=dtype) for fixture in fixtures]
    kwargs_list = [move_nested(fixture["kwargs"], device=device, dtype=dtype) for fixture in fixtures]
    expected_outputs = [move_nested(fixture["output"], device=device, dtype=dtype) for fixture in fixtures]
    args_batched, kwargs_batched = stack_component_inputs(args_list, kwargs_list, component_name, strategy)
    eager_output = eager_component(*args_batched, **kwargs_batched)
    compiled_output = compiled_component(*args_batched, **kwargs_batched)
    eager_split = split_component_output(eager_output, component_name, len(fixtures), strategy)
    compiled_split = split_component_output(compiled_output, component_name, len(fixtures), strategy)
    eager_stats = compare_output_list(expected_outputs, eager_split)
    compiled_stats = compare_output_list(expected_outputs, compiled_split)
    eager_compiled_stats = compare_output_list(eager_split, compiled_split)
    return {
        "frame_idx": fixtures[0]["frame_idx"],
        "eager_pass": eager_stats["pass"],
        "compiled_pass": compiled_stats["pass"],
        "eager_compiled_pass": eager_compiled_stats["pass"],
        "eager_max_abs_diff": eager_stats["max_abs_diff"],
        "compiled_max_abs_diff": compiled_stats["max_abs_diff"],
        "eager_compiled_max_abs_diff": eager_compiled_stats["max_abs_diff"],
        "eager_p95_abs_diff": eager_stats["p95_abs_diff"],
        "compiled_p95_abs_diff": compiled_stats["p95_abs_diff"],
        "eager_compiled_p95_abs_diff": eager_compiled_stats["p95_abs_diff"],
    }


def _apply_component_precision(model: Any, component: str, dtype: Any) -> None:
    targets = {
        "memory_attention": ["memory_attention"],
        "mask_decoder": ["mask_decoder", "prompt_encoder", "object_pointer_proj", "shared_image_embedding"],
        "memory_encoder": ["memory_encoder", "spatial_perceiver", "mask_downsample"],
        "vision_encoder": ["vision_encoder"],
    }.get(component, [component])
    for attr in targets:
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "to"):
            module.to(dtype=dtype)


def _failure_payload(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "fixtures_dir": args.fixtures_dir,
        "component": args.component,
        "compiled_component_equivalence_pass": False,
        "all_components_pass": False,
        "blockers": [{"frame_idx": None, "reason": reason}],
    }


def render(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload.get("rows") or []:
        rows.append(
            [
                row.get("frame_idx"),
                row.get("eager_pass"),
                row.get("compiled_pass"),
                row.get("eager_compiled_pass"),
                _fmt(row.get("compiled_max_abs_diff")),
                _fmt(row.get("compiled_p95_abs_diff")),
                _fmt(row.get("eager_compiled_max_abs_diff")),
            ]
        )
    blockers = payload.get("blockers") or []
    return "\n".join(
        [
            "# Compiled EdgeTAM Component Equivalence",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["component", payload.get("component")],
                    ["precision_mode", payload.get("precision_mode")],
                    ["component_dtype", payload.get("component_dtype")],
                    ["compile_mode", payload.get("compile_mode")],
                    ["groups", payload.get("groups")],
                    ["compiled_component_equivalence_pass", payload.get("compiled_component_equivalence_pass")],
                    ["compile_error", payload.get("compile_error")],
                ],
            ),
            "",
            markdown_table(
                ["frame", "eager", "compiled", "eager_vs_compiled", "compiled_max", "compiled_p95", "eager_compiled_max"],
                rows,
            ),
            "",
            "## Blockers",
            "",
            "\n".join(f"- frame {item.get('frame_idx')}: {item.get('reason')}" for item in blockers) or "- none",
        ]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _torch_dtype(torch_module: Any, name: str):
    lowered = name.lower().replace("torch.", "")
    if lowered in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if lowered in {"fp32", "float32"}:
        return torch_module.float32
    if lowered in {"fp16", "float16"}:
        return torch_module.float16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--component", choices=sorted(COMPONENT_POLICY_DTYPE_FIELD), required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--precision-mode", choices=PRECISION_POLICY_NAMES, default="memory_path_fp32")
    parser.add_argument("--compile-mode", choices=COMPILE_MODES, required=True)
    parser.add_argument("--graph-output-policy", default="ring_buffer")
    parser.add_argument("--stack-strategy", default="auto")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    payload = run_compiled_component_equivalence(args)
    write_json(args.output_json, payload)
    write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload.get("compiled_component_equivalence_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
