import unittest

import torch

from edgetam_batched.dtype_trace import collect_nested_dtypes, dtype_gates, summarize_dtype_rows


class DtypeTraceTests(unittest.TestCase):
    def test_collect_and_summarize_nested_dtypes(self):
        rows = collect_nested_dtypes({"x": torch.zeros(1, dtype=torch.float32), "y": [torch.zeros(1, dtype=torch.bfloat16)]})
        summary = summarize_dtype_rows(rows)
        self.assertEqual(summary["x"], ["torch.float32"])
        self.assertEqual(summary["y[0]"], ["torch.bfloat16"])

    def test_memory_path_gate_requires_fp32_persistent_state(self):
        summary = {
            "maskmem_features": ["torch.float32"],
            "object_pointer": ["torch.float32"],
            "pred_masks": ["torch.float32"],
            "object_score_logits": ["torch.float32"],
        }
        gate = dtype_gates("memory_path_fp32", summary)
        self.assertTrue(gate["pass"])
        summary["object_pointer"] = ["torch.bfloat16"]
        gate = dtype_gates("memory_path_fp32", summary)
        self.assertFalse(gate["pass"])


if __name__ == "__main__":
    unittest.main()
