"""Component access wrapper for HF EdgeTAM models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .report_utils import markdown_table, write_json, write_markdown


@dataclass
class ComponentAccess:
    name: str
    found: bool
    path: str | None
    python_class: str | None
    can_batch: bool
    notes: str = ""


class EdgeTamComponentAdapter:
    def __init__(self, model: Any):
        self.model = model
        self.components = {
            "vision_encoder": self._find_component("vision_encoder"),
            "memory_attention": self._find_component("memory_attention"),
            "mask_decoder": self._find_component("mask_decoder"),
            "memory_encoder": self._find_component("memory_encoder"),
            "prompt_encoder": self._find_component("prompt_encoder"),
        }

    @property
    def vision_encoder(self):
        return self.components["vision_encoder"][1]

    @property
    def memory_attention(self):
        return self.components["memory_attention"][1]

    @property
    def mask_decoder(self):
        return self.components["mask_decoder"][1]

    @property
    def memory_encoder(self):
        return self.components["memory_encoder"][1]

    @property
    def prompt_encoder(self):
        return self.components["prompt_encoder"][1]

    def get_image_features_batched(self, pixel_values_b3):
        return self.model.get_image_features(pixel_values_b3, return_dict=True)

    def report(self) -> list[ComponentAccess]:
        records = []
        for name, (path, module) in self.components.items():
            records.append(
                ComponentAccess(
                    name=name,
                    found=module is not None,
                    path=path,
                    python_class=module.__class__.__name__ if module is not None else None,
                    can_batch=name == "vision_encoder" and hasattr(self.model, "get_image_features"),
                    notes=(
                        "batch path uses model.get_image_features"
                        if name == "vision_encoder" and hasattr(self.model, "get_image_features")
                        else "requires explicit state tensorization"
                    ),
                )
            )
        return records

    def _find_component(self, token: str) -> tuple[str | None, Any | None]:
        if hasattr(self.model, "get_batched_component_handles"):
            handles = self.model.get_batched_component_handles()
            if token in handles:
                return token, handles[token]
        if hasattr(self.model, token):
            return token, getattr(self.model, token)
        for name, module in self.model.named_modules():
            if name.endswith(token) or token in name:
                return name, module
        return None, None


def render_component_report(records: list[ComponentAccess]) -> str:
    return "\n".join(
        [
            "# EdgeTAM Component Access",
            "",
            markdown_table(
                ["component", "found", "path", "class", "can_batch", "notes"],
                ([r.name, r.found, r.path, r.python_class, r.can_batch, r.notes] for r in records),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="yonigozlan/EdgeTAM-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    import torch
    from transformers import EdgeTamVideoModel

    dtype = getattr(torch, "bfloat16" if args.dtype in {"bf16", "bfloat16"} else args.dtype)
    model = EdgeTamVideoModel.from_pretrained(args.model_id).to(device=args.device, dtype=dtype).eval()
    records = EdgeTamComponentAdapter(model).report()
    write_json(args.output_json, {"components": [asdict(record) for record in records]})
    write_markdown(args.output_md, render_component_report(records))
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
