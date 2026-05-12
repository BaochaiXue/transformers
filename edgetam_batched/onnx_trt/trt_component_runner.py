"""TensorRT component runner for BatchTam validation and hot-path integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .trt_engine_utils import ComponentIOSpec, load_io_spec


class TrtComponentRunner:
    """Run a TensorRT engine using torch CUDA tensor bindings.

    This runner intentionally does not provide a silent PyTorch fallback. If
    TensorRT cannot load or execute the engine, the caller receives an error and
    the backend must be marked unusable.
    """

    def __init__(self, engine_path: str | Path, io_spec: str | Path | ComponentIOSpec, name: str | None = None):
        import tensorrt as trt
        import torch

        self.torch = torch
        self.trt = trt
        self.engine_path = Path(engine_path)
        self.io_spec = load_io_spec(io_spec) if not isinstance(io_spec, ComponentIOSpec) else io_spec
        self.name = name or self.io_spec.component
        if not self.engine_path.exists():
            raise FileNotFoundError(str(self.engine_path))
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create execution context: {self.engine_path}")
        self.input_names = [spec.name for spec in self.io_spec.inputs]
        self.output_names = [spec.name for spec in self.io_spec.outputs]

    def __call__(self, *inputs: Any):
        torch = self.torch
        if len(inputs) != len(self.input_names):
            raise ValueError(f"{self.name}: expected {len(self.input_names)} inputs, got {len(inputs)}")
        if not all(isinstance(tensor, torch.Tensor) and tensor.is_cuda for tensor in inputs):
            raise TypeError("TrtComponentRunner requires CUDA torch.Tensor inputs")
        for name, tensor in zip(self.input_names, inputs, strict=True):
            tensor = tensor.contiguous()
            self.context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        outputs = []
        for spec in self.io_spec.outputs:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(spec.name))
            dtype = _torch_dtype(torch, spec.dtype)
            output = torch.empty(shape, device=inputs[0].device, dtype=dtype)
            self.context.set_tensor_address(spec.name, int(output.data_ptr()))
            outputs.append(output)
        stream = torch.cuda.current_stream(device=inputs[0].device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"{self.name}: TensorRT execute_async_v3 failed")
        for output in outputs:
            output.record_stream(stream)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _torch_dtype(torch_module: Any, dtype_name: str):
    lowered = dtype_name.lower().replace("torch.", "")
    if lowered in {"float32", "fp32"}:
        return torch_module.float32
    if lowered in {"float16", "fp16"}:
        return torch_module.float16
    if lowered in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if lowered == "int32":
        return torch_module.int32
    if lowered == "int64":
        return torch_module.int64
    if lowered == "bool":
        return torch_module.bool
    raise ValueError(f"unsupported TensorRT output dtype for torch allocation: {dtype_name}")
