from __future__ import annotations

import unittest

from edgetam_batched.component_batch_equivalence import analyze_component


class ComponentBatchEquivalenceTests(unittest.TestCase):
    def test_vision_passes_when_trace_records_exist(self):
        trace = {"records": [{"component": "vision_encoder"}]}
        result = analyze_component(trace, "vision_encoder")

        self.assertTrue(result["equivalence_pass"])

    def test_memory_encoder_is_explicit_blocker(self):
        trace = {"records": [{"component": "memory_encoder"}]}
        result = analyze_component(trace, "memory_encoder")

        self.assertFalse(result["equivalence_pass"])
        self.assertIn("NotImplemented", result["blockers"][0])

    def test_memory_attention_requires_tensorized_state(self):
        trace = {"records": [{"component": "memory_attention"}]}
        result = analyze_component(trace, "memory_attention")

        self.assertFalse(result["equivalence_pass"])
        self.assertTrue(result["blockers"])


if __name__ == "__main__":
    unittest.main()
