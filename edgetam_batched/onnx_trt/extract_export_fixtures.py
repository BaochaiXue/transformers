"""Extract fixed-shape BatchTam ONNX/TRT export fixtures from component traces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edgetam_batched.component_batch_equivalence import (
    default_strategy,
    load_fixture_groups,
    stack_batch_dim0,
    stack_component_inputs,
    stack_seq_first,
)
from edgetam_batched.precision_policy import policy_torch_dtype, resolve_precision_policy
from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from . import COMPONENTS
from .trt_engine_utils import ComponentIOSpec, TensorSpec, flatten_tensors, write_io_spec


def extract_component(args: argparse.Namespace, component: str) -> dict[str, Any]:
    import torch

    target_dtype = component_torch_dtype(torch, component, args.precision_mode)
    component_dir = Path(args.out_dir) / component
    component_dir.mkdir(parents=True, exist_ok=True)
    groups = load_fixture_groups(args.fixtures_dir, component, args.batch_size, args.frames)
    if not groups:
        return {
            "component": component,
            "export_fixture_pass": False,
            "failure_stage": "fixture_extract",
            "exact_blocker": f"no {component} fixture groups found under {args.fixtures_dir}",
            "fixture_dir": str(component_dir),
        }

    group = groups[0]
    strategy = default_strategy(component)
    args_list = [cast_floating_nested(fixture["args"], target_dtype) for fixture in group]
    kwargs_list = [cast_floating_nested(fixture["kwargs"], target_dtype) for fixture in group]
    args_batched, kwargs_batched = stack_component_inputs(args_list, kwargs_list, component, strategy)
    expected_outputs = [cast_floating_nested(fixture["output"], target_dtype) for fixture in group]
    output_batched = stack_seq_first(expected_outputs) if component == "memory_attention" else stack_batch_dim0(expected_outputs)

    input_items = flatten_tensors((args_batched, kwargs_batched), prefix="input")
    output_items = flatten_tensors(output_batched, prefix="output")
    input_specs = [
        TensorSpec.from_tensor(name, tensor, path=path)
        for name, path, tensor in input_items
        if isinstance(tensor, torch.Tensor)
    ]
    output_specs = [
        TensorSpec.from_tensor(name, tensor, path=path)
        for name, path, tensor in output_items
        if isinstance(tensor, torch.Tensor)
    ]
    io_spec = ComponentIOSpec(
        component=component,
        batch_size=args.batch_size,
        object_count=args.object_count,
        precision_mode=args.precision_mode,
        inputs=input_specs,
        outputs=output_specs,
        trt_scope="memory_path" if component in {"memory_attention", "mask_decoder", "memory_encoder"} else "vision",
    )

    torch.save(
        {
            "component": component,
            "strategy": strategy,
            "args": args_batched,
            "kwargs": kwargs_batched,
            "flat_input_names": [item.name for item in input_specs],
            "flat_input_paths": [item.path for item in input_specs],
            "flat_inputs": [tensor.clone() for _name, _path, tensor in input_items],
        },
        component_dir / "sample_inputs.pt",
    )
    torch.save(
        {
            "component": component,
            "strategy": strategy,
            "output": output_batched,
            "flat_output_names": [item.name for item in output_specs],
            "flat_output_paths": [item.path for item in output_specs],
            "flat_outputs": [tensor.clone() for _name, _path, tensor in output_items],
        },
        component_dir / "sample_outputs_eager.pt",
    )
    write_io_spec(component_dir / "io_spec.json", io_spec)
    return {
        "component": component,
        "export_fixture_pass": True,
        "fixture_dir": str(component_dir),
        "strategy": strategy,
        "frame_idx": group[0].get("frame_idx"),
        "input_count": len(input_specs),
        "output_count": len(output_specs),
        "inputs": [spec.__dict__ for spec in input_specs],
        "outputs": [spec.__dict__ for spec in output_specs],
    }


def component_torch_dtype(torch_module: Any, component: str, precision_mode: str):
    policy = resolve_precision_policy(precision_mode)
    field = {
        "vision_encoder": "vision_dtype",
        "memory_attention": "memory_attention_dtype",
        "mask_decoder": "mask_decoder_dtype",
        "memory_encoder": "memory_encoder_dtype",
    }[component]
    return policy_torch_dtype(torch_module, getattr(policy, field))


def cast_floating_nested(value: Any, dtype: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if torch.is_floating_point(value) else value
    if isinstance(value, tuple):
        return tuple(cast_floating_nested(item, dtype) for item in value)
    if isinstance(value, list):
        return [cast_floating_nested(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: cast_floating_nested(item, dtype) for key, item in value.items()}
    return value


def render(payload: dict[str, Any]) -> str:
    rows = [
        [
            item.get("component"),
            item.get("export_fixture_pass"),
            item.get("input_count"),
            item.get("output_count"),
            item.get("failure_stage"),
            item.get("exact_blocker"),
        ]
        for item in payload["components"].values()
    ]
    return "\n".join(
        [
            "# BatchTam Export Fixtures",
            "",
            f"- fixtures_dir: `{payload['fixtures_dir']}`",
            f"- out_dir: `{payload['out_dir']}`",
            f"- batch_size: `{payload['batch_size']}`",
            f"- object_count: `{payload['object_count']}`",
            f"- precision_mode: `{payload['precision_mode']}`",
            f"- all_components_pass: `{payload['all_components_pass']}`",
            "",
            markdown_table(["component", "pass", "inputs", "outputs", "failure_stage", "blocker"], rows),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--precision-mode", default="memory_path_fp32")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    components = {component: extract_component(args, component) for component in args.components}
    payload = {
        "name": "BatchTam ONNX/TRT",
        "fixtures_dir": args.fixtures_dir,
        "out_dir": args.out_dir,
        "batch_size": args.batch_size,
        "object_count": args.object_count,
        "precision_mode": args.precision_mode,
        "components": components,
        "all_components_pass": all(item.get("export_fixture_pass") for item in components.values()),
    }
    if args.output_json:
        write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload["all_components_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
