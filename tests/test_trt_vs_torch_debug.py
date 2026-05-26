import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - base env lacks torch
    torch = None

from edgetam_batched.onnx_trt.trt_vs_torch_debug import summarize_tensor_diff


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TrtVsTorchDebugTests(unittest.TestCase):
    def test_summarize_tensor_diff_reports_p95_and_max(self):
        payload = summarize_tensor_diff(torch.zeros(4), torch.tensor([0.0, 1.0, 2.0, 3.0]))
        self.assertTrue(payload["comparable"])
        self.assertEqual(payload["max_abs_diff"], 3.0)
        self.assertIn("p95_abs_diff", payload)

    def test_summarize_tensor_diff_reports_shape_mismatch(self):
        payload = summarize_tensor_diff(torch.zeros(4), torch.zeros(1, 4))
        self.assertFalse(payload["comparable"])
        self.assertEqual(payload["reason"], "shape_mismatch")


if __name__ == "__main__":
    unittest.main()
