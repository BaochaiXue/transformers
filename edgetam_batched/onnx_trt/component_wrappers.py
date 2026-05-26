"""Tensor-only ONNX wrappers for BatchTam components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .trt_engine_utils import flatten_tensors


@dataclass(frozen=True)
class TensorRef:
    index: int


def make_tensor_skeleton(value: Any, *, prefix: str = "input") -> tuple[Any, list[str], list[torch.Tensor]]:
    flat = flatten_tensors(value, prefix=prefix)
    path_to_index = {path: idx for idx, (_name, path, _tensor) in enumerate(flat)}
    tensors = [tensor for _name, _path, tensor in flat]
    names = [name for name, _path, _tensor in flat]
    return _replace_tensors(value, path_to_index=path_to_index), names, tensors


def _replace_tensors(value: Any, *, path_to_index: dict[str, int], path: str = "root") -> Any:
    if isinstance(value, torch.Tensor):
        return TensorRef(path_to_index[path])
    if isinstance(value, tuple):
        return tuple(_replace_tensors(item, path_to_index=path_to_index, path=f"{path}[{idx}]") for idx, item in enumerate(value))
    if isinstance(value, list):
        return [_replace_tensors(item, path_to_index=path_to_index, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _replace_tensors(value[key], path_to_index=path_to_index, path=f"{path}.{key}") for key in value}
    return value


def rebuild_from_skeleton(skeleton: Any, tensors: tuple[torch.Tensor, ...]) -> Any:
    if isinstance(skeleton, TensorRef):
        return tensors[skeleton.index]
    if isinstance(skeleton, tuple):
        return tuple(rebuild_from_skeleton(item, tensors) for item in skeleton)
    if isinstance(skeleton, list):
        return [rebuild_from_skeleton(item, tensors) for item in skeleton]
    if isinstance(skeleton, dict):
        return {key: rebuild_from_skeleton(value, tensors) for key, value in skeleton.items()}
    return skeleton


class FlatTensorOnnxWrapper(torch.nn.Module):
    """Call a component using tensor-only flat ONNX inputs.

    The scheduler owns Python metadata. The wrapper freezes non-tensor metadata
    and exposes only tensors to ONNX/TensorRT.
    """

    def __init__(self, component: Any, args: Any, kwargs: Any):
        super().__init__()
        self.component = component
        args_skeleton, arg_names, _arg_tensors = make_tensor_skeleton(args, prefix="arg")
        kwargs_skeleton, kw_names, _kw_tensors = make_tensor_skeleton(kwargs, prefix="kw")
        self.args_skeleton = args_skeleton
        self.kwargs_skeleton = kwargs_skeleton
        self.input_names = arg_names + kw_names

    def forward(self, *flat_inputs: torch.Tensor):
        args_count = _count_tensor_refs(self.args_skeleton)
        args = rebuild_from_skeleton(self.args_skeleton, tuple(flat_inputs[:args_count]))
        kwargs = rebuild_from_skeleton(self.kwargs_skeleton, tuple(flat_inputs[args_count:]))
        output = self.component(*args, **kwargs)
        outputs = [tensor for _name, _path, tensor in flatten_tensors(output, prefix="out")]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class VisionEncoderOnnxWrapper(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(self, frames_b3: torch.Tensor):
        outputs = self.model.get_image_features(frames_b3, return_dict=True)
        tensors = []
        tensors.extend(list(outputs.fpn_hidden_states))
        tensors.extend(list(outputs.fpn_position_encoding))
        return tuple(tensors)


class MemoryAttentionOnnxWrapper(FlatTensorOnnxWrapper):
    pass


class MaskDecoderOnnxWrapper(FlatTensorOnnxWrapper):
    pass


class MemoryEncoderOnnxWrapper(FlatTensorOnnxWrapper):
    pass


def _count_tensor_refs(value: Any) -> int:
    if isinstance(value, TensorRef):
        return 1
    if isinstance(value, tuple):
        return sum(_count_tensor_refs(item) for item in value)
    if isinstance(value, list):
        return sum(_count_tensor_refs(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_tensor_refs(item) for item in value.values())
    return 0
