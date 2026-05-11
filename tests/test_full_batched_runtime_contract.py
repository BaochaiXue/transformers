from __future__ import annotations

import unittest

from edgetam_batched.backend_contract import FullBatchedContractError
from edgetam_batched.batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from edgetam_batched.config import BACKEND_BATCHED_MULTISESSION


class DummyModel:
    def named_modules(self):
        return []


class FullBatchedRuntimeContractTests(unittest.TestCase):
    def test_strict_full_batched_runtime_rejects_current_partial_implementation(self):
        with self.assertRaises(FullBatchedContractError):
            BatchedEdgeTamMultiSessionRuntime(
                DummyModel(),
                processor=None,
                backend=BACKEND_BATCHED_MULTISESSION,
                strict_full_batched=True,
                disallow_partial_backend_success=True,
            )


if __name__ == "__main__":
    unittest.main()
