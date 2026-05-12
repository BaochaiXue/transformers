import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from edgetam_batched.onnx_trt.memory_attention_shape_key import MemoryAttentionShapeKey


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class MemoryAttentionShapeKeyTests(unittest.TestCase):
    def test_shape_key_includes_counts_and_input_shapes(self):
        key = MemoryAttentionShapeKey.from_inputs(
            current_vision_features=torch.zeros(4096, 3, 256),
            current_vision_position_embeddings=torch.zeros(4096, 3, 256),
            memory=torch.zeros(1032, 3, 64),
            memory_posision_embeddings=torch.zeros(1032, 3, 64),
            num_object_pointer_tokens=8,
            num_spatial_memory_tokens=2,
        )

        self.assertEqual(key.batch_size, 3)
        self.assertEqual(key.num_object_pointer_tokens, 8)
        self.assertEqual(key.num_spatial_memory_tokens, 2)
        self.assertEqual(key.memory_seq_len, 1032)
        self.assertIn("objptr8_spmem2", key.slug)


if __name__ == "__main__":
    unittest.main()
