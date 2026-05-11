"""Precision policies for full batched EdgeTAM recurrent runtime."""

from __future__ import annotations

from dataclasses import dataclass


DTYPE_BF16 = "bf16"
DTYPE_FP32 = "fp32"


@dataclass(frozen=True)
class PrecisionPolicy:
    name: str
    vision_dtype: str = DTYPE_BF16
    memory_attention_dtype: str = DTYPE_BF16
    mask_decoder_dtype: str = DTYPE_BF16
    memory_encoder_dtype: str = DTYPE_BF16
    store_maskmem_features_dtype: str = DTYPE_BF16
    store_object_pointer_dtype: str = DTYPE_BF16
    store_mask_logits_dtype: str = DTYPE_BF16
    postprocess_dtype: str = DTYPE_BF16
    disable_autocast_for_memory_attention: bool = False
    disable_autocast_for_mask_decoder: bool = False
    disable_autocast_for_memory_encoder: bool = False

    @property
    def uses_fp32(self) -> bool:
        return any(
            getattr(self, field) == DTYPE_FP32
            for field in (
                "vision_dtype",
                "memory_attention_dtype",
                "mask_decoder_dtype",
                "memory_encoder_dtype",
                "store_maskmem_features_dtype",
                "store_object_pointer_dtype",
                "store_mask_logits_dtype",
                "postprocess_dtype",
            )
        )


PRECISION_POLICY_NAMES = (
    "all_bf16",
    "object_pointer_fp32",
    "maskmem_features_fp32",
    "mask_logits_fp32",
    "decoder_fp32",
    "mask_decoder_fp32",
    "memory_attention_fp32",
    "memory_encoder_fp32",
    "memory_path_fp32",
    "all_fp32",
)


def resolve_precision_policy(name: str | None) -> PrecisionPolicy:
    normalized = (name or "all_bf16").strip().lower().replace("-", "_")
    if normalized == "all_bf16":
        return PrecisionPolicy(name="all_bf16")
    if normalized == "object_pointer_fp32":
        return PrecisionPolicy(name=normalized, store_object_pointer_dtype=DTYPE_FP32)
    if normalized == "maskmem_features_fp32":
        return PrecisionPolicy(name=normalized, store_maskmem_features_dtype=DTYPE_FP32)
    if normalized == "mask_logits_fp32":
        return PrecisionPolicy(name=normalized, store_mask_logits_dtype=DTYPE_FP32, postprocess_dtype=DTYPE_FP32)
    if normalized in {"decoder_fp32", "mask_decoder_fp32"}:
        return PrecisionPolicy(
            name="decoder_fp32" if normalized == "mask_decoder_fp32" else normalized,
            mask_decoder_dtype=DTYPE_FP32,
            store_object_pointer_dtype=DTYPE_FP32,
            store_mask_logits_dtype=DTYPE_FP32,
            postprocess_dtype=DTYPE_FP32,
            disable_autocast_for_mask_decoder=True,
        )
    if normalized == "memory_attention_fp32":
        return PrecisionPolicy(
            name=normalized,
            memory_attention_dtype=DTYPE_FP32,
            disable_autocast_for_memory_attention=True,
        )
    if normalized == "memory_encoder_fp32":
        return PrecisionPolicy(
            name=normalized,
            memory_encoder_dtype=DTYPE_FP32,
            store_maskmem_features_dtype=DTYPE_FP32,
            disable_autocast_for_memory_encoder=True,
        )
    if normalized == "memory_path_fp32":
        return PrecisionPolicy(
            name=normalized,
            memory_attention_dtype=DTYPE_FP32,
            mask_decoder_dtype=DTYPE_FP32,
            memory_encoder_dtype=DTYPE_FP32,
            store_maskmem_features_dtype=DTYPE_FP32,
            store_object_pointer_dtype=DTYPE_FP32,
            store_mask_logits_dtype=DTYPE_FP32,
            postprocess_dtype=DTYPE_FP32,
            disable_autocast_for_memory_attention=True,
            disable_autocast_for_mask_decoder=True,
            disable_autocast_for_memory_encoder=True,
        )
    if normalized == "all_fp32":
        return PrecisionPolicy(
            name=normalized,
            vision_dtype=DTYPE_FP32,
            memory_attention_dtype=DTYPE_FP32,
            mask_decoder_dtype=DTYPE_FP32,
            memory_encoder_dtype=DTYPE_FP32,
            store_maskmem_features_dtype=DTYPE_FP32,
            store_object_pointer_dtype=DTYPE_FP32,
            store_mask_logits_dtype=DTYPE_FP32,
            postprocess_dtype=DTYPE_FP32,
            disable_autocast_for_memory_attention=True,
            disable_autocast_for_mask_decoder=True,
            disable_autocast_for_memory_encoder=True,
        )
    raise ValueError(f"unsupported precision policy: {name}")


def policy_torch_dtype(torch_module, dtype_name: str):
    normalized = str(dtype_name).strip().lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"unsupported policy dtype: {dtype_name}")


def reference_dtype_for_precision_mode(base_dtype: str, precision_mode: str | None) -> str:
    policy = resolve_precision_policy(precision_mode)
    if policy.name == "all_fp32":
        return "float32"
    return base_dtype
