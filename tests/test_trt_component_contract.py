import tempfile
import unittest
from pathlib import Path

from edgetam_batched.onnx_trt.build_trt_engine import build
from edgetam_batched.onnx_trt.trt_engine_utils import ComponentIOSpec, TensorSpec, write_io_spec


class TrtComponentContractTests(unittest.TestCase):
    def test_build_reports_missing_onnx_as_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            io_spec = Path(tmp) / "io_spec.json"
            write_io_spec(
                io_spec,
                ComponentIOSpec(
                    component="memory_encoder",
                    batch_size=3,
                    object_count=1,
                    precision_mode="memory_path_fp32",
                    inputs=[TensorSpec("x", [3, 4], "float32")],
                    outputs=[TensorSpec("y", [3, 4], "float32")],
                ),
            )

            class Args:
                component = "memory_encoder"
                onnx_path = str(Path(tmp) / "missing.onnx")
                engine_path = str(Path(tmp) / "out.engine")
                builder_optimization_level = 5
                precision_policy = "memory_path_fp32"
                timing_cache = None
                no_tf32_for_memory_path = True
                export_layer_info = None
                export_profile = None
                skip_inference = True
                timeout_s = None

            Args.io_spec = str(io_spec)

            payload = build(Args())

        self.assertFalse(payload["trt_build_pass"])
        self.assertEqual(payload["failure_stage"], "trt_build")
        self.assertIn("missing ONNX", payload["exact_blocker"])


if __name__ == "__main__":
    unittest.main()
