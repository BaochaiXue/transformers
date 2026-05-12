"""BatchTam ONNX/TensorRT component runtime tools.

BatchTam keeps the EdgeTAM video scheduler and recurrent session state in
Python, while compiling tensor-only batch=3 components to ONNX/TensorRT.
"""

from __future__ import annotations

BATCHTAM_NAME = "BatchTam ONNX/TRT"

COMPONENTS = ("vision_encoder", "memory_attention", "mask_decoder", "memory_encoder")
MEMORY_PATH_COMPONENTS = ("memory_attention", "mask_decoder", "memory_encoder")
