from __future__ import annotations

import unittest

from edgetam_batched.drift_teacher_force import infer_teacher_force_source


class DriftTeacherForceTests(unittest.TestCase):
    def test_before_step_pass_points_to_closed_loop_accumulation(self):
        self.assertEqual(
            infer_teacher_force_source("before_step_state", None),
            "closed_loop_accumulation_or_state_writeback",
        )

    def test_none_failure_is_baseline_closed_loop_drift(self):
        self.assertEqual(
            infer_teacher_force_source("none", {"frame_idx": 3}),
            "baseline_full_batched_closed_loop_drift",
        )

    def test_after_memory_encoder_pass_points_to_memory_writeback(self):
        self.assertEqual(
            infer_teacher_force_source("after_memory_encoder", None),
            "memory_encoder_or_maskmem_writeback",
        )


if __name__ == "__main__":
    unittest.main()
