import tempfile
import unittest
from pathlib import Path
from unittest import mock

from edgetam_batched.onnx_trt.trt_component_registry import TrtComponentRegistry
from edgetam_batched.onnx_trt.trt_engine_utils import ComponentIOSpec, TensorSpec, write_io_spec
from edgetam_batched.onnx_trt.trt_runtime_config import TrtRuntimeConfig, components_for_scope


def _write_component(root: Path, component: str) -> None:
    (root / "engines").mkdir(parents=True, exist_ok=True)
    (root / "engines" / f"{component}_b3.engine").write_bytes(b"engine")
    spec_dir = root / "export_fixtures" / component
    spec_dir.mkdir(parents=True, exist_ok=True)
    write_io_spec(
        spec_dir / "io_spec.json",
        ComponentIOSpec(
            component=component,
            batch_size=3,
            object_count=1,
            precision_mode="memory_path_fp32",
            inputs=[TensorSpec("x", [3, 4], "float32")],
            outputs=[TensorSpec("y", [3, 4], "float32")],
        ),
    )


class TrtComponentRegistryTests(unittest.TestCase):
    def test_components_for_memory_path_all(self):
        self.assertEqual(
            components_for_scope("memory_path_all"),
            ("memory_attention", "mask_decoder", "memory_encoder"),
        )

    def test_missing_engine_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_component(root, "memory_attention")
            with self.assertRaises(FileNotFoundError):
                TrtComponentRegistry(TrtRuntimeConfig(root / "engines", "memory_path_all"))

    @mock.patch("edgetam_batched.onnx_trt.trt_component_registry.TrtComponentRunner")
    def test_registry_loads_required_scope_without_fallback(self, runner_cls):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for component in ("memory_attention", "mask_decoder", "memory_encoder"):
                _write_component(root, component)

            registry = TrtComponentRegistry(TrtRuntimeConfig(root / "engines", "memory_path_all"))

        self.assertTrue(registry.uses("memory_attention"))
        self.assertTrue(registry.uses("mask_decoder"))
        self.assertTrue(registry.uses("memory_encoder"))
        self.assertEqual(runner_cls.call_count, 3)
        self.assertFalse(registry.contract_fields()["torch_fallback_used"])


if __name__ == "__main__":
    unittest.main()
