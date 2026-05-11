from __future__ import annotations

import unittest

from edgetam_batched.state_commit_ablation import COMMIT_TO_FORCE


class StateCommitAblationTests(unittest.TestCase):
    def test_commit_modes_map_to_teacher_force_modes(self):
        self.assertEqual(COMMIT_TO_FORCE["batched_scatter"], "none")
        self.assertEqual(COMMIT_TO_FORCE["reference_store_output"], "after_mask_decoder")
        self.assertEqual(COMMIT_TO_FORCE["hybrid_reference_dict_batched_tensors"], "after_memory_encoder")


if __name__ == "__main__":
    unittest.main()
