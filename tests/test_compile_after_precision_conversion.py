import unittest

from edgetam_batched.batched_multisession_runtime import BatchedEdgeTamMultiSessionRuntime
from edgetam_batched.config import (
    COMPILE_SCOPE_MASK_DECODER,
    COMPILE_SCOPE_MEMORY_ATTENTION,
    COMPILE_SCOPE_MEMORY_ENCODER,
    COMPILE_SCOPE_MEMORY_PATH_ALL,
    COMPILE_SCOPE_NONE,
    COMPILE_SCOPE_VISION_ENCODER,
)


class CompileAfterPrecisionConversionTests(unittest.TestCase):
    def _runtime_with_scope(self, scope):
        runtime = object.__new__(BatchedEdgeTamMultiSessionRuntime)
        runtime.compile_scope = scope
        return runtime

    def test_memory_path_scope_excludes_vision(self):
        runtime = self._runtime_with_scope(COMPILE_SCOPE_MEMORY_PATH_ALL)
        self.assertTrue(runtime._scope_compiles(COMPILE_SCOPE_MEMORY_ATTENTION))
        self.assertTrue(runtime._scope_compiles(COMPILE_SCOPE_MASK_DECODER))
        self.assertTrue(runtime._scope_compiles(COMPILE_SCOPE_MEMORY_ENCODER))
        self.assertFalse(runtime._scope_compiles(COMPILE_SCOPE_VISION_ENCODER))

    def test_none_scope_compiles_nothing(self):
        runtime = self._runtime_with_scope(COMPILE_SCOPE_NONE)
        self.assertFalse(runtime._scope_compiles(COMPILE_SCOPE_MEMORY_ATTENTION))
        self.assertFalse(runtime._scope_compiles(COMPILE_SCOPE_VISION_ENCODER))


if __name__ == "__main__":
    unittest.main()
