"""Export tensor-only BatchTam component fixtures to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from edgetam_batched.component_adapter import EdgeTamComponentAdapter
from edgetam_batched.precision_policy import policy_torch_dtype, resolve_precision_policy
from edgetam_batched.report_utils import markdown_table, write_json, write_markdown

from . import COMPONENTS
from .component_wrappers import FlatTensorOnnxWrapper, VisionEncoderOnnxWrapper
from .trt_engine_utils import load_io_spec


COMPONENT_DTYPE_FIELD = {
    "vision_encoder": "vision_dtype",
    "memory_attention": "memory_attention_dtype",
    "mask_decoder": "mask_decoder_dtype",
    "memory_encoder": "memory_encoder_dtype",
}


def load_component(model: Any, component: str):
    if component == "vision_encoder":
        return model
    adapter = EdgeTamComponentAdapter(model)
    return getattr(adapter, component)


def apply_component_precision(model: Any, torch_module: Any, component: str, precision_mode: str) -> str:
    policy = resolve_precision_policy(precision_mode)
    dtype = policy_torch_dtype(torch_module, getattr(policy, COMPONENT_DTYPE_FIELD[component]))
    targets = {
        "vision_encoder": ["vision_encoder"],
        "memory_attention": ["memory_attention"],
        "mask_decoder": ["mask_decoder", "prompt_encoder", "object_pointer_proj", "shared_image_embedding"],
        "memory_encoder": ["memory_encoder", "spatial_perceiver", "mask_downsample"],
    }[component]
    if precision_mode == "all_fp32":
        model.to(dtype=torch_module.float32)
    for attr in targets:
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "to"):
            module.to(dtype=dtype)
    return str(dtype)


def export_component(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import EdgeTamVideoModel

    fixture_dir = Path(args.fixtures_dir) / args.component
    sample_inputs_path = fixture_dir / "sample_inputs.pt"
    io_spec_path = fixture_dir / "io_spec.json"
    if not sample_inputs_path.exists() or not io_spec_path.exists():
        return failure(args, "onnx_export", f"missing export fixture under {fixture_dir}")

    sample_inputs = torch.load(sample_inputs_path, map_location=args.device, weights_only=False)
    io_spec = load_io_spec(io_spec_path)
    base_dtype = _torch_dtype(torch, args.dtype)
    model = EdgeTamVideoModel.from_pretrained(args.model_id).to(device=args.device, dtype=base_dtype).eval()
    component_dtype = apply_component_precision(model, torch, args.component, args.precision_mode)
    component = load_component(model, args.component)
    if component is None:
        return failure(args, "onnx_export", f"component {args.component} not found")

    if args.component == "vision_encoder":
        wrapper = VisionEncoderOnnxWrapper(model).eval()
    else:
        wrapper = FlatTensorOnnxWrapper(component, sample_inputs["args"], sample_inputs["kwargs"]).eval()
    wrapper.to(device=args.device)
    flat_inputs = [tensor.to(args.device).contiguous() for tensor in sample_inputs["flat_inputs"]]
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / args.onnx_name if args.onnx_name else output_dir / f"{args.component}_b3.onnx"
    with torch.inference_mode():
        dry_output = wrapper(*flat_inputs)
    output_count = len(dry_output) if isinstance(dry_output, tuple) else 1
    output_names = [spec.name for spec in io_spec.outputs[:output_count]] or [f"output_{idx}" for idx in range(output_count)]
    export_error = None
    try:
        torch.onnx.export(
            wrapper,
            tuple(flat_inputs),
            str(onnx_path),
            input_names=[spec.name for spec in io_spec.inputs],
            output_names=output_names,
            opset_version=args.opset,
            dynamo=bool(args.dynamo),
            export_params=True,
            do_constant_folding=True,
        )
    except Exception as exc:  # noqa: BLE001
        if not args.allow_legacy_retry or not args.dynamo:
            return failure(args, "onnx_export", repr(exc), component_dtype=component_dtype)
        export_error = repr(exc)
        try:
            torch.onnx.export(
                wrapper,
                tuple(flat_inputs),
                str(onnx_path),
                input_names=[spec.name for spec in io_spec.inputs],
                output_names=output_names,
                opset_version=args.opset,
                dynamo=False,
                export_params=True,
                do_constant_folding=True,
            )
            exporter_mode = "legacy"
        except Exception as legacy_exc:  # noqa: BLE001
            return failure(
                args,
                "onnx_export",
                f"dynamo error: {export_error}; legacy error: {legacy_exc!r}",
                component_dtype=component_dtype,
            )
    else:
        exporter_mode = "dynamo" if args.dynamo else "legacy"

    checker_error = None
    node_count = None
    op_types: dict[str, int] = {}
    try:
        import onnx

        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        node_count = len(model_proto.graph.node)
        for node in model_proto.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    except Exception as exc:  # noqa: BLE001
        checker_error = repr(exc)

    return {
        "component": args.component,
        "onnx_export_pass": checker_error is None,
        "onnx_path": str(onnx_path),
        "component_dtype": component_dtype,
        "exporter_mode": exporter_mode,
        "dynamo_export_error_before_legacy_retry": export_error,
        "checker_error": checker_error,
        "node_count": node_count,
        "op_types": op_types,
        "failure_stage": None if checker_error is None else "onnx_check",
        "exact_blocker": checker_error,
    }


def failure(args: argparse.Namespace, stage: str, blocker: str, **extra) -> dict[str, Any]:
    return {
        "component": args.component,
        "onnx_export_pass": False,
        "onnx_path": str(Path(args.out_dir) / (args.onnx_name or f"{args.component}_b3.onnx")),
        "failure_stage": stage,
        "exact_blocker": blocker,
        **extra,
    }


def render(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# BatchTam ONNX Export: {payload.get('component')}",
            "",
            markdown_table(
                ["field", "value"],
                [
                    ["component", payload.get("component")],
                    ["onnx_export_pass", payload.get("onnx_export_pass")],
                    ["onnx_path", payload.get("onnx_path")],
                    ["component_dtype", payload.get("component_dtype")],
                    ["exporter_mode", payload.get("exporter_mode")],
                    ["node_count", payload.get("node_count")],
                    ["failure_stage", payload.get("failure_stage")],
                    ["exact_blocker", payload.get("exact_blocker")],
                ],
            ),
        ]
    )


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
    parser.add_argument("--component", choices=COMPONENTS, required=True)
    parser.add_argument("--fixtures-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--onnx-name", default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--allow-legacy-retry", action="store_true")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--object-count", type=int, default=1)
    parser.add_argument("--precision-mode", default="memory_path_fp32")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    payload = export_component(args)
    if args.output_json:
        write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, render(payload))
    if args.debug:
        print(payload)
    return 0 if payload.get("onnx_export_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
