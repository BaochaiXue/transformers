"""Compile-mode validation and lightweight wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .config import COMPILE_NONE, CompileConfig


@dataclass
class CompiledModuleRecord:
    name: str
    mode: str
    compiled: bool
    reason: str = ""


class CompiledComponentWrapper:
    def __init__(self, module: Any, name: str, config: CompileConfig):
        config.validate()
        self.name = name
        self.config = config
        self.original = module
        self.module = module
        self.record = CompiledModuleRecord(name=name, mode=config.mode, compiled=False)
        if config.mode != COMPILE_NONE:
            if torch is None:
                self.record.reason = "torch unavailable"
            else:
                try:
                    self.module = torch.compile(module, mode=config.mode)
                    self.record.compiled = True
                except Exception as exc:  # pragma: no cover - depends on torch build.
                    self.record.reason = repr(exc)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)
