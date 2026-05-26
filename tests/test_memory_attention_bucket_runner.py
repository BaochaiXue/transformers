import tempfile
import unittest
from pathlib import Path
from unittest import mock

from edgetam_batched.onnx_trt.trt_component_registry import TrtComponentRegistry
from edgetam_batched.onnx_trt.trt_engine_utils import ComponentIOSpec, TensorSpec, write_io_spec
from edgetam_batched.onnx_trt.trt_runtime_config import TrtRuntimeConfig


class MemoryAttentionBucketRunnerTests(unittest.TestCase):
    @mock.patch("edgetam_batched.onnx_trt.trt_component_registry.TrtComponentRunner")
    def test_component_registry_accepts_memory_attention_bucket_dir(self, runner_cls):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "engines"
            bucket_dir = engine_dir / "memory_attention_buckets"
            fixture_root = root / "export_fixtures" / "memory_attention"
            bucket = "shape_000_objptr4_spmem1"
            bucket_dir.mkdir(parents=True)
            (bucket_dir / f"{bucket}.engine").write_bytes(b"engine")
            (engine_dir / "mask_decoder_b3.engine").write_bytes(b"engine")
            spec_dir = root / "export_fixtures" / "mask_decoder"
            spec_dir.mkdir(parents=True)
            write_io_spec(
                spec_dir / "io_spec.json",
                ComponentIOSpec(
                    component="mask_decoder",
                    batch_size=3,
                    object_count=1,
                    precision_mode="memory_path_fp32",
                    inputs=[TensorSpec("x", [3, 4], "float32")],
                    outputs=[TensorSpec("y", [3, 4], "float32")],
                ),
            )
            (fixture_root / bucket).mkdir(parents=True)
            write_io_spec(
                fixture_root / bucket / "io_spec.json",
                ComponentIOSpec(
                    component="memory_attention",
                    batch_size=3,
                    object_count=1,
                    precision_mode="memory_path_fp32",
                    inputs=[
                        TensorSpec("input_1_current_vision_features", [4096, 3, 256], "float32"),
                        TensorSpec("input_1_current_vision_position_embeddings", [4096, 3, 256], "float32"),
                        TensorSpec("input_1_memory", [516, 3, 64], "float32"),
                        TensorSpec("input_1_memory_posision_embeddings", [516, 3, 64], "float32"),
                    ],
                    outputs=[TensorSpec("output", [1, 3, 4096, 256], "float32")],
                    num_object_pointer_tokens=4,
                    num_spatial_memory_tokens=1,
                ),
            )

            registry = TrtComponentRegistry(
                TrtRuntimeConfig(
                    engine_dir=engine_dir,
                    trt_scope="memory_attention_mask_decoder",
                    memory_attention_bucket_dir=bucket_dir,
                )
            )

        self.assertTrue(registry.uses("memory_attention"))
        self.assertIsNotNone(registry.memory_attention_buckets)
        self.assertEqual(runner_cls.call_count, 2)


if __name__ == "__main__":
    unittest.main()
