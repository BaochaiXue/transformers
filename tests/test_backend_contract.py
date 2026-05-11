from __future__ import annotations

import unittest

from edgetam_batched.backend_contract import (
    BackendContractResult,
    FullBatchedContractError,
    assert_hf_batched_multisession_contract,
    contract_for_current_runtime,
)
from edgetam_batched.config import BACKEND_BATCHED_MULTISESSION


class BackendContractTests(unittest.TestCase):
    def test_full_contract_passes_only_when_all_required_parts_are_true(self):
        result = BackendContractResult(
            backend=BACKEND_BATCHED_MULTISESSION,
            batch_vision=True,
            batch_memory_attention=True,
            batch_mask_decoder=True,
            batch_memory_encoder=True,
            batched_state_scatter=True,
        )

        assert_hf_batched_multisession_contract(result)
        self.assertTrue(result.contract_pass)

    def test_public_session_step_fails_full_contract(self):
        result = BackendContractResult(
            backend=BACKEND_BATCHED_MULTISESSION,
            batch_vision=True,
            batch_memory_attention=True,
            batch_mask_decoder=True,
            batch_memory_encoder=True,
            batched_state_scatter=True,
            used_public_session_step_in_hot_path=True,
        )

        with self.assertRaises(FullBatchedContractError):
            result.assert_full_batched()

    def test_current_runtime_contract_marks_full_backend_incomplete(self):
        result = contract_for_current_runtime(
            backend=BACKEND_BATCHED_MULTISESSION,
            batch_vision=True,
            partial_fallback_used=True,
        )

        self.assertFalse(result.contract_pass)
        self.assertIn("batch_memory_attention", result.missing_requirements())
        self.assertIn("used_public_session_step_in_hot_path", result.missing_requirements())
        self.assertTrue(result.blockers)


if __name__ == "__main__":
    unittest.main()
