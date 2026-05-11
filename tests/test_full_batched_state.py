from __future__ import annotations

import unittest

from edgetam_batched.backend_contract import FullBatchedContractError
from edgetam_batched.full_state_scatter import scatter_full_state
from edgetam_batched.full_state_stack import stack_full_state


class FullBatchedStateTests(unittest.TestCase):
    def test_stack_full_state_is_incomplete_until_session_state_is_tensorized(self):
        class Session:
            def get_obj_num(self):
                return 1

        state = stack_full_state([Session(), Session(), Session()], frame_idx=5)

        self.assertEqual(state.batch_size, 3)
        self.assertEqual(state.object_count, 1)
        self.assertFalse(state.stack_complete)
        self.assertTrue(state.unsupported_fields)

    def test_scatter_rejects_incomplete_state(self):
        class Session:
            def get_obj_num(self):
                return 1

        state = stack_full_state([Session(), Session(), Session()], frame_idx=5)

        with self.assertRaises(FullBatchedContractError):
            scatter_full_state(state, [Session(), Session(), Session()])


if __name__ == "__main__":
    unittest.main()
