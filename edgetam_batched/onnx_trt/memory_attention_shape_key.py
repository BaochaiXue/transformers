"""Shape keys for BatchTam memory-attention TensorRT buckets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryAttentionShapeKey:
    batch_size: int
    num_object_pointer_tokens: int
    num_spatial_memory_tokens: int
    current_seq_len: int
    memory_seq_len: int
    hidden_dim: int
    pos_dim: int | None
    input_shape_signature: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_inputs(
        cls,
        *,
        current_vision_features: Any,
        current_vision_position_embeddings: Any,
        memory: Any,
        memory_posision_embeddings: Any,
        num_object_pointer_tokens: int,
        num_spatial_memory_tokens: int,
    ) -> "MemoryAttentionShapeKey":
        signature = (
            ("current_vision_features", _shape(current_vision_features)),
            ("current_vision_position_embeddings", _shape(current_vision_position_embeddings)),
            ("memory", _shape(memory)),
            ("memory_posision_embeddings", _shape(memory_posision_embeddings)),
        )
        current_shape = signature[0][1]
        memory_shape = signature[2][1]
        memory_pos_shape = signature[3][1]
        return cls(
            batch_size=int(current_shape[1]),
            num_object_pointer_tokens=int(num_object_pointer_tokens),
            num_spatial_memory_tokens=int(num_spatial_memory_tokens),
            current_seq_len=int(current_shape[0]),
            memory_seq_len=int(memory_shape[0]),
            hidden_dim=int(current_shape[-1]),
            pos_dim=int(memory_pos_shape[-1]) if memory_pos_shape else None,
            input_shape_signature=signature,
        )

    @classmethod
    def from_io_spec(cls, io_spec: Any) -> "MemoryAttentionShapeKey":
        inputs = {item.name: item for item in io_spec.inputs}

        def find(suffix: str):
            for name, spec in inputs.items():
                if name.endswith(suffix):
                    return spec
            raise KeyError(f"memory_attention io_spec missing input ending with {suffix}")

        current = find("current_vision_features")
        current_pos = find("current_vision_position_embeddings")
        memory = find("memory")
        memory_pos = find("memory_posision_embeddings")
        return cls(
            batch_size=int(current.shape[1]),
            num_object_pointer_tokens=int(getattr(io_spec, "num_object_pointer_tokens", 0) or 0),
            num_spatial_memory_tokens=int(getattr(io_spec, "num_spatial_memory_tokens", 0) or 0),
            current_seq_len=int(current.shape[0]),
            memory_seq_len=int(memory.shape[0]),
            hidden_dim=int(current.shape[-1]),
            pos_dim=int(memory_pos.shape[-1]) if memory_pos.shape else None,
            input_shape_signature=(
                ("current_vision_features", tuple(current.shape)),
                ("current_vision_position_embeddings", tuple(current_pos.shape)),
                ("memory", tuple(memory.shape)),
                ("memory_posision_embeddings", tuple(memory_pos.shape)),
            ),
        )

    @property
    def slug(self) -> str:
        return (
            f"objptr{self.num_object_pointer_tokens}_spmem{self.num_spatial_memory_tokens}"
            f"_cur{self.current_seq_len}_mem{self.memory_seq_len}"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_object_pointer_tokens": self.num_object_pointer_tokens,
            "num_spatial_memory_tokens": self.num_spatial_memory_tokens,
            "current_seq_len": self.current_seq_len,
            "memory_seq_len": self.memory_seq_len,
            "hidden_dim": self.hidden_dim,
            "pos_dim": self.pos_dim,
            "input_shape_signature": [
                [name, list(shape)] for name, shape in self.input_shape_signature
            ],
            "slug": self.slug,
        }


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)
