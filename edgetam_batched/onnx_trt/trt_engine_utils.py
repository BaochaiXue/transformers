"""Utilities for BatchTam ONNX/TensorRT fixed-shape component engines."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - exercised in integration environments
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DTYPE_TO_NUMPY = {
    "float32": "float32",
    "fp32": "float32",
    "bfloat16": "float32",
    "bf16": "float32",
    "float16": "float16",
    "fp16": "float16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
}


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str
    path: str = ""

    @classmethod
    def from_tensor(cls, name: str, tensor: Any, *, path: str = "") -> "TensorSpec":
        return cls(name=name, shape=[int(dim) for dim in tensor.shape], dtype=normalize_dtype(tensor.dtype), path=path)

    @property
    def trtexec_shape(self) -> str:
        return f"{self.name}:{'x'.join(str(dim) for dim in self.shape)}"


@dataclass(frozen=True)
class ComponentIOSpec:
    component: str
    batch_size: int
    object_count: int
    precision_mode: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    trt_scope: str = "component"
    fixed_shape: bool = True

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["name"] = f"{self.component}_b{self.batch_size}"
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ComponentIOSpec":
        return cls(
            component=payload["component"],
            batch_size=int(payload["batch_size"]),
            object_count=int(payload.get("object_count", 1)),
            precision_mode=payload.get("precision_mode", "memory_path_fp32"),
            inputs=[TensorSpec(**item) for item in payload.get("inputs", [])],
            outputs=[TensorSpec(**item) for item in payload.get("outputs", [])],
            trt_scope=payload.get("trt_scope", "component"),
            fixed_shape=bool(payload.get("fixed_shape", True)),
        )


def normalize_dtype(dtype: Any) -> str:
    text = str(dtype).replace("torch.", "").lower()
    if text in {"bfloat16", "bf16"}:
        return "bfloat16"
    if text in {"float16", "half", "fp16"}:
        return "float16"
    if text in {"float32", "fp32", "float"}:
        return "float32"
    if text in {"int64", "long"}:
        return "int64"
    if text in {"int32", "int"}:
        return "int32"
    if text in {"bool", "torch.bool"}:
        return "bool"
    return text


def sanitize_name(path: str, *, prefix: str) -> str:
    cleaned = path.replace("root", prefix)
    for old, new in ((".", "_"), ("[", "_"), ("]", ""), ("-", "_"), ("/", "_")):
        cleaned = cleaned.replace(old, new)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or prefix


def flatten_tensors(value: Any, *, prefix: str = "input", path: str = "root") -> list[tuple[str, str, Any]]:
    if _is_tensor(value):
        return [(sanitize_name(path, prefix=prefix), path, value)]
    if isinstance(value, tuple):
        out: list[tuple[str, str, Any]] = []
        for idx, item in enumerate(value):
            out.extend(flatten_tensors(item, prefix=prefix, path=f"{path}[{idx}]"))
        return out
    if isinstance(value, list):
        out = []
        for idx, item in enumerate(value):
            out.extend(flatten_tensors(item, prefix=prefix, path=f"{path}[{idx}]"))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(flatten_tensors(value[key], prefix=prefix, path=f"{path}.{key}"))
        return out
    return []


def fixed_shape_profile_args(io_spec: ComponentIOSpec) -> list[str]:
    if not io_spec.inputs:
        return []
    shapes = ",".join(item.trtexec_shape for item in io_spec.inputs)
    return [f"--minShapes={shapes}", f"--optShapes={shapes}", f"--maxShapes={shapes}"]


def trtexec_path() -> str | None:
    return shutil.which("trtexec")


def build_trtexec_command(
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    io_spec: ComponentIOSpec,
    builder_optimization_level: int = 5,
    timing_cache: str | Path | None = None,
    no_tf32: bool = False,
    export_layer_info: str | Path | None = None,
    export_profile: str | Path | None = None,
    skip_inference: bool = True,
    include_shape_profiles: bool = True,
) -> list[str]:
    exe = trtexec_path() or "trtexec"
    cmd = [
        exe,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--builderOptimizationLevel={int(builder_optimization_level)}",
        "--profilingVerbosity=detailed",
    ]
    if include_shape_profiles:
        cmd.extend(fixed_shape_profile_args(io_spec))
    if timing_cache:
        cmd.append(f"--timingCacheFile={timing_cache}")
    if no_tf32:
        cmd.append("--noTF32")
    if export_layer_info:
        cmd.extend(["--dumpLayerInfo", f"--exportLayerInfo={export_layer_info}"])
    if export_profile:
        cmd.extend(["--dumpProfile", f"--exportProfile={export_profile}"])
    if skip_inference:
        cmd.append("--skipInference")
    return cmd


def run_command(cmd: list[str], *, cwd: str | Path | None = None, timeout: int | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "error": repr(exc), "command": cmd}
    except subprocess.TimeoutExpired as exc:
        return {"returncode": 124, "stdout": exc.stdout or "", "error": repr(exc), "command": cmd}
    return {"returncode": completed.returncode, "stdout": completed.stdout, "command": cmd}


def load_io_spec(path: str | Path) -> ComponentIOSpec:
    return ComponentIOSpec.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def write_io_spec(path: str | Path, spec: ComponentIOSpec) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def component_status_table(component_payloads: Iterable[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            item.get("component"),
            item.get("onnx_export_pass"),
            item.get("onnx_validation_pass"),
            item.get("trt_build_pass"),
            item.get("trt_validation_pass"),
            item.get("failure_stage"),
            item.get("exact_blocker"),
        ]
        for item in component_payloads
    ]


def trt_components_usable(report: dict[str, Any], *, scope: str = "memory_path_all") -> bool:
    decision = report.get("decision") or {}
    if decision.get("trt_components_usable") is not None:
        return bool(decision["trt_components_usable"])
    components = report.get("components") or {}
    required = ("memory_attention", "mask_decoder", "memory_encoder")
    if scope == "full_components":
        required = ("vision_encoder",) + required
    return all(bool((components.get(name) or {}).get("trt_validation_pass")) for name in required)


def _is_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)
