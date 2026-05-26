import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from edgetam_batched.onnx_trt.component_wrappers import make_tensor_skeleton, rebuild_from_skeleton
from edgetam_batched.onnx_trt.trt_engine_utils import TensorSpec, flatten_tensors, sanitize_name


class OnnxTrtShapeTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_flatten_tensors_records_paths_and_shapes(self):
        value = {"x": torch.zeros(3, 4), "meta": "fixed", "y": [torch.ones(1)]}

        flat = flatten_tensors(value, prefix="input")

        self.assertEqual([path for _name, path, _tensor in flat], ["root.x", "root.y[0]"])
        spec = TensorSpec.from_tensor(flat[0][0], flat[0][2], path=flat[0][1])
        self.assertEqual(spec.shape, [3, 4])
        self.assertEqual(spec.trtexec_shape, f"{spec.name}:3x4")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_tensor_skeleton_round_trip(self):
        value = (torch.zeros(1, 2), {"a": torch.ones(3), "flag": True})

        skeleton, names, tensors = make_tensor_skeleton(value)
        rebuilt = rebuild_from_skeleton(skeleton, tuple(tensors))

        self.assertEqual(len(names), 2)
        self.assertTrue(torch.equal(rebuilt[0], value[0]))
        self.assertTrue(torch.equal(rebuilt[1]["a"], value[1]["a"]))
        self.assertTrue(rebuilt[1]["flag"])

    def test_sanitize_name_is_onnx_safe_enough(self):
        self.assertEqual(sanitize_name("root.kw.foo[0]", prefix="input"), "input_kw_foo_0")


if __name__ == "__main__":
    unittest.main()
