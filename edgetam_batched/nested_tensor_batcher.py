"""Recursive tensor stacking/splitting helpers for component batch probes."""

from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class UnsupportedBatchField(ValueError):
    def __init__(self, path: str, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


def can_batch_nested(values: list[Any]) -> bool:
    try:
        stack_nested(values)
    except UnsupportedBatchField:
        return False
    return True


def stack_nested(values: list[Any], batch_dim: int = 0, *, path: str = "root") -> Any:
    if not values:
        raise UnsupportedBatchField(path, "empty value list")
    first = values[0]
    if _is_tensor(first):
        _assert_all_tensors(values, path)
        return torch.stack(values, dim=batch_dim)
    if first is None or isinstance(first, (bool, int, float, str)):
        if all(value == first for value in values):
            return first
        raise UnsupportedBatchField(path, "metadata values differ across batch items")
    if isinstance(first, tuple):
        if not all(isinstance(value, tuple) and len(value) == len(first) for value in values):
            raise UnsupportedBatchField(path, "tuple structures differ")
        return tuple(
            stack_nested([value[idx] for value in values], batch_dim=batch_dim, path=f"{path}[{idx}]")
            for idx in range(len(first))
        )
    if isinstance(first, list):
        if not all(isinstance(value, list) and len(value) == len(first) for value in values):
            raise UnsupportedBatchField(path, "list structures differ")
        return [
            stack_nested([value[idx] for value in values], batch_dim=batch_dim, path=f"{path}[{idx}]")
            for idx in range(len(first))
        ]
    if isinstance(first, dict):
        keys = set(first)
        if not all(isinstance(value, dict) and set(value) == keys for value in values):
            raise UnsupportedBatchField(path, "dict keys differ")
        return {
            key: stack_nested([value[key] for value in values], batch_dim=batch_dim, path=f"{path}.{key}")
            for key in sorted(keys)
        }
    if is_dataclass(first):
        raise UnsupportedBatchField(path, "dataclass stacking requires an explicit adapter")
    raise UnsupportedBatchField(path, f"unsupported type {type(first).__name__}")


def split_nested(value: Any, batch_size: int, batch_dim: int = 0, *, path: str = "root") -> list[Any]:
    if _is_tensor(value):
        if value.shape[batch_dim] != batch_size:
            raise UnsupportedBatchField(path, f"tensor batch dim has size {value.shape[batch_dim]}, expected {batch_size}")
        return [value.select(batch_dim, idx).contiguous() for idx in range(batch_size)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return [value for _ in range(batch_size)]
    if isinstance(value, tuple):
        parts = [split_nested(item, batch_size, batch_dim=batch_dim, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
        return [tuple(part[idx] for part in parts) for idx in range(batch_size)]
    if isinstance(value, list):
        parts = [split_nested(item, batch_size, batch_dim=batch_dim, path=f"{path}[{idx}]") for idx, item in enumerate(value)]
        return [[part[idx] for part in parts] for idx in range(batch_size)]
    if isinstance(value, dict):
        split_by_key = {
            key: split_nested(item, batch_size, batch_dim=batch_dim, path=f"{path}.{key}")
            for key, item in value.items()
        }
        return [{key: split_by_key[key][idx] for key in split_by_key} for idx in range(batch_size)]
    raise UnsupportedBatchField(path, f"unsupported type {type(value).__name__}")


def compare_nested(a: Any, b: Any, atol: float = 1e-3, rtol: float = 1e-3, *, path: str = "root") -> dict[str, Any]:
    if _is_tensor(a) and _is_tensor(b):
        if a.shape != b.shape:
            return {"pass": False, "path": path, "reason": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
        diff = (a.float() - b.float()).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        return {
            "pass": bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)),
            "path": path,
            "max_abs": max_abs,
        }
    if type(a) is not type(b):
        return {"pass": False, "path": path, "reason": f"type mismatch {type(a).__name__} vs {type(b).__name__}"}
    if isinstance(a, (bool, int, float, str, type(None))):
        return {"pass": a == b, "path": path}
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return {"pass": False, "path": path, "reason": "length mismatch"}
        children = [compare_nested(x, y, atol=atol, rtol=rtol, path=f"{path}[{idx}]") for idx, (x, y) in enumerate(zip(a, b))]
        return {"pass": all(child.get("pass") for child in children), "path": path, "children": children}
    if isinstance(a, dict):
        if set(a) != set(b):
            return {"pass": False, "path": path, "reason": "key mismatch"}
        children = [compare_nested(a[key], b[key], atol=atol, rtol=rtol, path=f"{path}.{key}") for key in sorted(a)]
        return {"pass": all(child.get("pass") for child in children), "path": path, "children": children}
    return {"pass": False, "path": path, "reason": f"unsupported type {type(a).__name__}"}


def _is_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _assert_all_tensors(values: list[Any], path: str) -> None:
    first = values[0]
    for idx, value in enumerate(values):
        if not _is_tensor(value):
            raise UnsupportedBatchField(f"{path}[{idx}]", "mixed tensor/non-tensor values")
        if value.shape != first.shape:
            raise UnsupportedBatchField(f"{path}[{idx}]", f"shape {tuple(value.shape)} != {tuple(first.shape)}")
        if value.dtype != first.dtype:
            raise UnsupportedBatchField(f"{path}[{idx}]", f"dtype {value.dtype} != {first.dtype}")
        if value.device != first.device:
            raise UnsupportedBatchField(f"{path}[{idx}]", f"device {value.device} != {first.device}")
