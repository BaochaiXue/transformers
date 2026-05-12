import inspect
import unittest

from edgetam_batched.onnx_trt.trt_component_runner import TrtComponentRunner


class TrtRunnerNoCpuCopyTests(unittest.TestCase):
    def test_runner_hot_path_uses_async_v3_and_no_cpu_copy(self):
        source = inspect.getsource(TrtComponentRunner.__call__)
        self.assertIn("set_tensor_address", source)
        self.assertIn("execute_async_v3", source)
        self.assertIn("record_stream", source)
        self.assertNotIn(".cpu()", source)
        self.assertNotIn(".numpy()", source)
        self.assertNotIn("synchronize", source)


if __name__ == "__main__":
    unittest.main()
