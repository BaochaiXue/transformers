import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - base env lacks torch
    torch = None

from edgetam_batched.onnx_trt.trt_engine_utils import ComponentIOSpec, TensorSpec
from edgetam_batched.onnx_trt.trt_io_adapter import TrtIoAdapter


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TrtIoAdapterTests(unittest.TestCase):
    def test_adapter_casts_and_contiguizes_inputs(self):
        spec = ComponentIOSpec(
            component="memory_encoder",
            batch_size=3,
            object_count=1,
            precision_mode="memory_path_fp32",
            inputs=[
                TensorSpec("pix", [3, 4], "float32"),
                TensorSpec("mask", [3, 4], "float32"),
            ],
            outputs=[],
        )
        adapter = TrtIoAdapter(torch)
        pix = torch.ones(4, 3, dtype=torch.bfloat16).t()
        mask = torch.ones(3, 4, dtype=torch.float32)

        out_pix, out_mask = adapter.memory_encoder_inputs(io_spec=spec, pix_feat=pix, mask_for_mem=mask)

        self.assertEqual(out_pix.dtype, torch.float32)
        self.assertTrue(out_pix.is_contiguous())
        self.assertEqual(tuple(out_mask.shape), (3, 4))

    def test_adapter_rejects_shape_mismatch(self):
        spec = ComponentIOSpec(
            component="memory_attention",
            batch_size=3,
            object_count=1,
            precision_mode="memory_path_fp32",
            inputs=[TensorSpec("x", [3, 4], "float32")],
            outputs=[],
        )
        adapter = TrtIoAdapter(torch)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            adapter._coerce_inputs("memory_attention", spec, (torch.ones(3, 5),))


if __name__ == "__main__":
    unittest.main()
