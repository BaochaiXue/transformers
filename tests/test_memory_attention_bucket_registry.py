import tempfile
import unittest
from pathlib import Path
from unittest import mock

from edgetam_batched.onnx_trt.memory_attention_shape_key import MemoryAttentionShapeKey
from edgetam_batched.onnx_trt.trt_component_registry import MemoryAttentionBucketRegistry
from edgetam_batched.onnx_trt.trt_engine_utils import ComponentIOSpec, TensorSpec, write_io_spec


class MemoryAttentionBucketRegistryTests(unittest.TestCase):
    @mock.patch("edgetam_batched.onnx_trt.trt_component_registry.TrtComponentRunner")
    def test_bucket_registry_loads_by_shape_key(self, _runner_cls):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine_dir = root / "engines"
            fixture_root = root / "fixtures"
            bucket = "shape_000_objptr4_spmem1"
            engine_dir.mkdir()
            (engine_dir / f"{bucket}.engine").write_bytes(b"engine")
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

            registry = MemoryAttentionBucketRegistry(engine_dir, fixture_root)

        key = next(iter(registry.runners))
        self.assertIsInstance(key, MemoryAttentionShapeKey)
        self.assertEqual(key.num_object_pointer_tokens, 4)
        self.assertEqual(key.memory_seq_len, 516)


if __name__ == "__main__":
    unittest.main()
