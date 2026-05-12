"""Registry for BatchTam TensorRT component engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .trt_component_runner import TrtComponentRunner
from .trt_engine_utils import ComponentIOSpec, load_io_spec
from .trt_runtime_config import TrtRuntimeConfig, components_for_scope


class TrtComponentRegistry:
    """Load the TensorRT engines required by a BatchTam runtime scope."""

    ENGINE_NAMES = {
        "memory_attention": "memory_attention_b3.engine",
        "mask_decoder": "mask_decoder_b3.engine",
        "memory_encoder": "memory_encoder_b3.engine",
    }
    IO_SPEC_DIRS = {
        "memory_attention": "memory_attention",
        "mask_decoder": "mask_decoder",
        "memory_encoder": "memory_encoder",
    }

    def __init__(self, config: TrtRuntimeConfig):
        self.config = config
        self.required_components = components_for_scope(config.trt_scope)
        self.runners: dict[str, TrtComponentRunner] = {}
        self.io_specs: dict[str, ComponentIOSpec] = {}
        if config.require_all_scope_engines:
            self._assert_required_files()
        for component in self.required_components:
            self.io_specs[component] = load_io_spec(self._io_spec_path(component))
            self.runners[component] = TrtComponentRunner(
                self._engine_path(component),
                self.io_specs[component],
                name=component,
            )

    def uses(self, component: str) -> bool:
        return component in self.required_components

    def runner(self, component: str) -> TrtComponentRunner:
        if component not in self.runners:
            if self.config.allow_torch_fallback:
                raise KeyError(component)
            raise RuntimeError(f"TRT scope {self.config.trt_scope} does not include {component}")
        return self.runners[component]

    def io_spec(self, component: str) -> ComponentIOSpec:
        return self.io_specs[component]

    def contract_fields(self) -> dict[str, Any]:
        return {
            "component_runtime": "trt",
            "trt_scope": self.config.trt_scope,
            "trt_memory_attention": self.uses("memory_attention"),
            "trt_mask_decoder": self.uses("mask_decoder"),
            "trt_memory_encoder": self.uses("memory_encoder"),
            "torch_fallback_used": False,
        }

    def _assert_required_files(self) -> None:
        missing: list[str] = []
        for component in self.required_components:
            engine = self._engine_path(component)
            spec = self._io_spec_path(component)
            if not engine.exists():
                missing.append(str(engine))
            if not spec.exists():
                missing.append(str(spec))
        if missing:
            raise FileNotFoundError("missing TensorRT component artifact(s): " + ", ".join(missing))

    def _engine_path(self, component: str) -> Path:
        return self.config.engine_dir / self.ENGINE_NAMES[component]

    def _io_spec_path(self, component: str) -> Path:
        artifact_root = self.config.engine_dir.parent
        return artifact_root / "export_fixtures" / self.IO_SPEC_DIRS[component] / "io_spec.json"
