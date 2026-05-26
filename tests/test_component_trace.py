from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from edgetam_batched.component_trace import session_digest, summarize_nested, trace_components


class ComponentTraceTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_trace_components_records_tensor_shapes(self):
        module = torch.nn.Linear(2, 2)

        with trace_components({"linear": module}) as recorder:
            _ = module(torch.zeros((1, 2)))

        payload = recorder.to_json()
        self.assertEqual(payload["record_count"], 1)
        self.assertEqual(payload["records"][0]["component"], "linear")
        self.assertEqual(payload["records"][0]["output"]["shape"], [1, 2])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_summarize_nested_handles_dicts(self):
        summary = summarize_nested({"x": torch.zeros((2, 3)), "flag": True})
        self.assertEqual(summary["type"], "dict")
        self.assertEqual(summary["items"]["x"]["shape"], [2, 3])

    def test_session_digest_counts_output_frames(self):
        session = type("Session", (), {})()
        session.output_dict_per_obj = {
            0: {"cond_frame_outputs": {0: "a"}, "non_cond_frame_outputs": {1: "b", 2: "c"}}
        }
        session.obj_ids = [1]
        session.obj_with_new_inputs = {0}
        session.frames_tracked_per_obj = {0: {0: True, 1: True}}

        digest = session_digest(session)

        self.assertEqual(digest["output_counts"]["0"]["cond"], 1)
        self.assertEqual(digest["output_counts"]["0"]["non_cond"], 2)
        self.assertEqual(digest["frames_tracked"]["0"], [0, 1])


if __name__ == "__main__":
    unittest.main()
