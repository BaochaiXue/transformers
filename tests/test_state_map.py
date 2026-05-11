from __future__ import annotations

import unittest

from edgetam_batched.state_map import build_state_map, create_synthetic_sessions


class StateMapTests(unittest.TestCase):
    def test_synthetic_session_fields_are_classified(self):
        before = create_synthetic_sessions(camera_count=3, object_count=2, frame_count=0)
        after = create_synthetic_sessions(camera_count=3, object_count=2, frame_count=3)
        payload = build_state_map(before, after)
        self.assertEqual(payload["summary"]["session_count"], 3)
        self.assertGreater(payload["summary"]["tensor_field_count"], 0)
        self.assertGreater(payload["summary"]["stackable_tensor_field_count"], 0)
        self.assertTrue(payload["required_field_presence"]["obj_ids"])
        self.assertTrue(payload["required_field_presence"]["vision_feature_cache"])
        self.assertTrue(payload["required_field_presence"]["conditioning_outputs"])

    def test_missing_session_field_marks_not_stackable(self):
        sessions = create_synthetic_sessions(camera_count=3, object_count=2, frame_count=0)
        del sessions[2].previous_masks
        payload = build_state_map(sessions)
        previous = [field for field in payload["fields"] if field["path"].endswith("previous_masks")]
        self.assertEqual(len(previous), 1)
        self.assertFalse(previous[0]["can_stack"])
        self.assertIn("missing_in_some_sessions", previous[0]["notes"])


if __name__ == "__main__":
    unittest.main()
