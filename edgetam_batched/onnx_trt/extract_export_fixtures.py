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
from .memory_attention_shape_key import MemoryAttentionShapeKey
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


def extract_memory_attention_shape_buckets(args: argparse.Namespace) -> dict[str, Any]:
    import json
    import torch

    sequence = json.loads(Path(args.shape_sequence_json).read_text(encoding="utf-8"))
    out_root = Path(args.out_dir) / "memory_attention"
    out_root.mkdir(parents=True, exist_ok=True)
    components: dict[str, Any] = {}
    for shape_key, bucket in sequence.get("shape_buckets", {}).items():
        bucket_dir = out_root / bucket["bucket_dir_name"]
        bucket_dir.mkdir(parents=True, exist_ok=True)
        fixtures = [torch.load(path, map_location="cpu", weights_only=False) for path in bucket["representative_fixture_paths"]]
        if len(fixtures) != args.batch_size:
            components[shape_key] = {
                "component": "memory_attention",
                "shape_key": shape_key,
                "export_fixture_pass": False,
                "failure_stage": "fixture_extract",
                "exact_blocker": f"expected {args.batch_size} representative fixtures, got {len(fixtures)}",
            }
            continue
        target_dtype = component_torch_dtype(torch, "memory_attention", args.precision_mode)
        args_list = [cast_floating_nested(fixture["args"], target_dtype) for fixture in fixtures]
        kwargs_list = [cast_floating_nested(fixture["kwargs"], target_dtype) for fixture in fixtures]
        args_batched, kwargs_batched = stack_component_inputs(
            args_list,
            kwargs_list,
            "memory_attention",
            default_strategy("memory_attention"),
        )
        expected_outputs = [cast_floating_nested(fixture["output"], target_dtype) for fixture in fixtures]
        output_batched = stack_seq_first(expected_outputs)
        input_items = flatten_tensors((args_batched, kwargs_batched), prefix="input")
        output_items = flatten_tensors(output_batched, prefix="output")
        input_specs = [TensorSpec.from_tensor(name, tensor, path=path) for name, path, tensor in input_items]
        output_specs = [TensorSpec.from_tensor(name, tensor, path=path) for name, path, tensor in output_items]
        key = MemoryAttentionShapeKey.from_inputs(
            current_vision_features=kwargs_batched["current_vision_features"],
            current_vision_position_embeddings=kwargs_batched["current_vision_position_embeddings"],
            memory=kwargs_batched["memory"],
            memory_posision_embeddings=kwargs_batched["memory_posision_embeddings"],
            num_object_pointer_tokens=kwargs_batched["num_object_pointer_tokens"],
            num_spatial_memory_tokens=kwargs_batched["num_spatial_memory_tokens"],
        )
        io_spec = ComponentIOSpec(
            component="memory_attention",
            batch_size=args.batch_size,
            object_count=args.object_count,
            precision_mode=args.precision_mode,
            inputs=input_specs,
            outputs=output_specs,
            trt_scope="memory_path",
            shape_key=key.slug,
            num_object_pointer_tokens=key.num_object_pointer_tokens,
            num_spatial_memory_tokens=key.num_spatial_memory_tokens,
        )
        torch.save(
            {
                "component": "memory_attention",
                "strategy": default_strategy("memory_attention"),
                "shape_key": key.slug,
                "args": args_batched,
                "kwargs": kwargs_batched,
                "flat_input_names": [item.name for item in input_specs],
                "flat_input_paths": [item.path for item in input_specs],
                "flat_inputs": [tensor.clone() for _name, _path, tensor in input_items],
            },
            bucket_dir / "sample_inputs.pt",
        )
        torch.save(
            {
                "component": "memory_attention",
                "strategy": default_strategy("memory_attention"),
                "shape_key": key.slug,
                "output": output_batched,
                "flat_output_names": [item.name for item in output_specs],
                "flat_output_paths": [item.path for item in output_specs],
                "flat_outputs": [tensor.clone() for _name, _path, tensor in output_items],
            },
            bucket_dir / "sample_outputs_eager.pt",
        )
        write_io_spec(bucket_dir / "io_spec.json", io_spec)
        components[shape_key] = {
            "component": "memory_attention",
            "shape_key": key.slug,
            "bucket_dir_name": bucket["bucket_dir_name"],
            "fixture_dir": str(bucket_dir),
            "export_fixture_pass": True,
            "input_count": len(input_specs),
            "output_count": len(output_specs),
            "inputs": [spec.__dict__ for spec in input_specs],
            "outputs": [spec.__dict__ for spec in output_specs],
        }
    return {
        "name": "BatchTam memory_attention bucket export fixtures",
        "out_dir": args.out_dir,
        "shape_sequence_json": args.shape_sequence_json,
        "batch_size": args.batch_size,
        "object_count": args.object_count,
        "precision_mode": args.precision_mode,
        "components": components,
        "all_components_pass": all(item.get("export_fixture_pass") for item in components.values()),
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
            item.get("bucket_dir_name") or "",
            item.get("shape_key") or "",
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
            f"- fixtures_dir: `{payload.get('fixtures_dir', '')}`",
            f"- out_dir: `{payload['out_dir']}`",
            f"- shape_sequence_json: `{payload.get('shape_sequence_json', '')}`",
            f"- batch_size: `{payload['batch_size']}`",
            f"- object_count: `{payload['object_count']}`",
            f"- precision_mode: `{payload['precision_mode']}`",
            f"- all_components_pass: `{payload['all_components_pass']}`",
            "",
            markdown_table(
                ["component", "bucket", "shape_key", "pass", "inputs", "outputs", "failure_stage", "blocker"],
                rows,
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--component", choices=COMPONENTS, default=None)
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=list(COMPONENTS))
    parser.add_argument("--shape-sequence-json", default=None)
    parser.add_argument("--one-fixture-per-shape-key", action="store_true")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--precision-mode", default="memory_path_fp32")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.one_fixture_per_shape_key:
        if args.component not in {None, "memory_attention"}:
            raise ValueError("--one-fixture-per-shape-key currently supports memory_attention only")
        if not args.shape_sequence_json:
            raise ValueError("--shape-sequence-json is required for bucket fixture extraction")
        payload = extract_memory_attention_shape_buckets(args)
    else:
        selected = [args.component] if args.component else args.components
        components = {component: extract_component(args, component) for component in selected}
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
