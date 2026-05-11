from __future__ import annotations

import unittest

from edgetam_batched.backend_contract import FullBatchedContractError
from edgetam_batched.full_state_scatter import scatter_full_state
from edgetam_batched.full_state_stack import stack_full_state


class FullStateScatterTests(unittest.TestCase):
    def test_scatter_rejects_incomplete_tensorized_state(self):
        class Session:
            def get_obj_num(self):
                return 1

        sessions = [Session(), Session(), Session()]
        state = stack_full_state(sessions, frame_idx=3)

        with self.assertRaises(FullBatchedContractError):
            scatter_full_state(state, sessions)


if __name__ == "__main__":
    unittest.main()
