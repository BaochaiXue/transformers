"""Registry for BatchTam TensorRT component engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .trt_component_runner import TrtComponentRunner
from .trt_engine_utils import ComponentIOSpec, load_io_spec
from .memory_attention_shape_key import MemoryAttentionShapeKey
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
        self.memory_attention_buckets: MemoryAttentionBucketRegistry | None = None
        if config.require_all_scope_engines:
            self._assert_required_files()
        for component in self.required_components:
            if component == "memory_attention" and config.memory_attention_bucket_dir is not None:
                self.memory_attention_buckets = MemoryAttentionBucketRegistry(
                    config.memory_attention_bucket_dir,
                    self._memory_attention_bucket_fixture_root(config.memory_attention_bucket_dir),
                )
                continue
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

    def memory_attention_runner_for(
        self,
        *,
        current_vision_features: Any,
        current_vision_position_embeddings: Any,
        memory: Any,
        memory_posision_embeddings: Any,
        num_object_pointer_tokens: int,
        num_spatial_memory_tokens: int,
    ) -> tuple[TrtComponentRunner, ComponentIOSpec, MemoryAttentionShapeKey]:
        if self.memory_attention_buckets is None:
            return self.runner("memory_attention"), self.io_spec("memory_attention"), MemoryAttentionShapeKey.from_inputs(
                current_vision_features=current_vision_features,
                current_vision_position_embeddings=current_vision_position_embeddings,
                memory=memory,
                memory_posision_embeddings=memory_posision_embeddings,
                num_object_pointer_tokens=num_object_pointer_tokens,
                num_spatial_memory_tokens=num_spatial_memory_tokens,
            )
        return self.memory_attention_buckets.get_runner_for_inputs(
            current_vision_features=current_vision_features,
            current_vision_position_embeddings=current_vision_position_embeddings,
            memory=memory,
            memory_posision_embeddings=memory_posision_embeddings,
            num_object_pointer_tokens=num_object_pointer_tokens,
            num_spatial_memory_tokens=num_spatial_memory_tokens,
        )

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
            "memory_attention_shape_strategy": "bucketed_static_engines"
            if self.memory_attention_buckets is not None
            else "single_static_engine",
        }

    def _assert_required_files(self) -> None:
        missing: list[str] = []
        for component in self.required_components:
            if component == "memory_attention" and self.config.memory_attention_bucket_dir is not None:
                bucket_dir = self.config.memory_attention_bucket_dir
                if not bucket_dir.exists() or not any(bucket_dir.glob("*.engine")):
                    missing.append(str(bucket_dir))
                continue
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

    def _memory_attention_bucket_fixture_root(self, bucket_dir: Path) -> Path:
        artifact_root = bucket_dir.parent.parent
        return artifact_root / "export_fixtures" / "memory_attention"


class MemoryAttentionBucketRegistry:
    def __init__(self, bucket_engine_dir: str | Path, fixture_root: str | Path):
        self.bucket_engine_dir = Path(bucket_engine_dir)
        self.fixture_root = Path(fixture_root)
        self.runners: dict[MemoryAttentionShapeKey, TrtComponentRunner] = {}
        self.io_specs: dict[MemoryAttentionShapeKey, ComponentIOSpec] = {}
        for engine_path in sorted(self.bucket_engine_dir.glob("*.engine")):
            fixture_dir = self.fixture_root / engine_path.stem
            io_spec = load_io_spec(fixture_dir / "io_spec.json")
            key = MemoryAttentionShapeKey.from_io_spec(io_spec)
            self.io_specs[key] = io_spec
            self.runners[key] = TrtComponentRunner(engine_path, io_spec, name=f"memory_attention:{engine_path.stem}")
        if not self.runners:
            raise FileNotFoundError(f"no memory_attention bucket engines found in {self.bucket_engine_dir}")

    def get_runner_for_inputs(
        self,
        *,
        current_vision_features: Any,
        current_vision_position_embeddings: Any,
        memory: Any,
        memory_posision_embeddings: Any,
        num_object_pointer_tokens: int,
        num_spatial_memory_tokens: int,
    ) -> tuple[TrtComponentRunner, ComponentIOSpec, MemoryAttentionShapeKey]:
        key = MemoryAttentionShapeKey.from_inputs(
            current_vision_features=current_vision_features,
            current_vision_position_embeddings=current_vision_position_embeddings,
            memory=memory,
            memory_posision_embeddings=memory_posision_embeddings,
            num_object_pointer_tokens=num_object_pointer_tokens,
            num_spatial_memory_tokens=num_spatial_memory_tokens,
        )
        if key not in self.runners:
            available = ", ".join(item.slug for item in self.runners)
            raise RuntimeError(f"No memory_attention TRT engine for shape_key={key.slug}. Available={available}")
        return self.runners[key], self.io_specs[key], key
