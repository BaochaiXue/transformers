"""Precision helpers used by the full batched EdgeTAM runtime."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any


def cast_nested_floating(value: Any, dtype: Any) -> Any:
    """Recursively cast floating tensors while preserving nested structure."""

    if _is_tensor(value):
        return value.to(dtype=dtype) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: cast_nested_floating(item, dtype) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cast_nested_floating(item, dtype) for item in value)
    if isinstance(value, list):
        return [cast_nested_floating(item, dtype) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        updates = {field.name: cast_nested_floating(getattr(value, field.name), dtype) for field in fields(value)}
        return replace(value, **updates)
    return value


def detach_clone_to_dtype(value: Any, dtype: Any, *, contiguous: bool = True) -> Any:
    """Detach, clone, and cast floating tensors before publishing to session state."""

    if _is_tensor(value):
        out = value.detach().clone()
        if out.is_floating_point():
            out = out.to(dtype=dtype)
        return out.contiguous() if contiguous else out
    if isinstance(value, dict):
        return {key: detach_clone_to_dtype(item, dtype, contiguous=contiguous) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_clone_to_dtype(item, dtype, contiguous=contiguous) for item in value)
    if isinstance(value, list):
        return [detach_clone_to_dtype(item, dtype, contiguous=contiguous) for item in value]
    return value


class PrecisionComponentRunner:
    def __init__(self, module_or_fn, dtype, *, disable_autocast: bool, name: str):
        self.module_or_fn = module_or_fn
        self.dtype = dtype
        self.disable_autocast = bool(disable_autocast)
        self.name = name

    def __call__(self, *args, **kwargs):
        import torch

        args = cast_nested_floating(args, self.dtype)
        kwargs = cast_nested_floating(kwargs, self.dtype)
        device_type = _first_tensor_device_type(args, kwargs)
        if device_type == "cuda":
            if self.disable_autocast:
                ctx = torch.autocast("cuda", enabled=False)
            else:
                ctx = torch.autocast("cuda", dtype=self.dtype)
        else:
            ctx = _nullcontext()
        with ctx:
            return self.module_or_fn(*args, **kwargs)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _first_tensor_device_type(*values: Any) -> str | None:
    for value in values:
        if _is_tensor(value):
            return value.device.type
        if isinstance(value, dict):
            found = _first_tensor_device_type(*value.values())
            if found:
                return found
        elif isinstance(value, (tuple, list)):
            found = _first_tensor_device_type(*value)
            if found:
                return found
    return None


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "is_floating_point") and hasattr(value, "to") and hasattr(value, "detach")
