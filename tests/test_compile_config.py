from __future__ import annotations

import unittest

from edgetam_batched.config import (
    COMPILE_MAX_AUTOTUNE_NO_CUDAGRAPHS,
    COMPILE_NONE,
    COMPILE_REDUCE_OVERHEAD,
    GRAPH_OUTPUT_CLONE,
    GRAPH_OUTPUT_RING_BUFFER,
    CompileConfig,
)


class CompileConfigTests(unittest.TestCase):
    def test_reduce_overhead_requires_ring_buffer(self):
        with self.assertRaises(ValueError):
            CompileConfig(mode=COMPILE_REDUCE_OVERHEAD, graph_output_policy=GRAPH_OUTPUT_CLONE).validate()

    def test_reduce_overhead_ring_buffer_valid(self):
        CompileConfig(mode=COMPILE_REDUCE_OVERHEAD, graph_output_policy=GRAPH_OUTPUT_RING_BUFFER).validate()

    def test_no_compile_valid(self):
        CompileConfig(mode=COMPILE_NONE, graph_output_policy=GRAPH_OUTPUT_CLONE).validate()

    def test_no_cudagraphs_allows_clone(self):
        CompileConfig(mode=COMPILE_MAX_AUTOTUNE_NO_CUDAGRAPHS, graph_output_policy=GRAPH_OUTPUT_CLONE).validate()


if __name__ == "__main__":
    unittest.main()
