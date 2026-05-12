import unittest

from edgetam_batched.onnx_trt.trt_engine_utils import (
    ComponentIOSpec,
    TensorSpec,
    build_trtexec_command,
    fixed_shape_profile_args,
    trt_components_usable,
)


class TrtEngineUtilsTests(unittest.TestCase):
    def test_fixed_shape_profile_args_use_min_opt_max_same_shape(self):
        spec = ComponentIOSpec(
            component="memory_attention",
            batch_size=3,
            object_count=1,
            precision_mode="memory_path_fp32",
            inputs=[TensorSpec("x", [10, 3, 256], "float32")],
            outputs=[],
        )

        args = fixed_shape_profile_args(spec)

        self.assertEqual(args, ["--minShapes=x:10x3x256", "--optShapes=x:10x3x256", "--maxShapes=x:10x3x256"])

    def test_build_trtexec_command_includes_precision_and_profile(self):
        spec = ComponentIOSpec(
            component="mask_decoder",
            batch_size=3,
            object_count=1,
            precision_mode="memory_path_fp32",
            inputs=[TensorSpec("image_embed", [3, 256, 64, 64], "float32")],
            outputs=[TensorSpec("mask", [3, 1, 256, 256], "float32")],
        )

        cmd = build_trtexec_command(
            onnx_path="m.onnx",
            engine_path="m.engine",
            io_spec=spec,
            no_tf32=True,
            export_layer_info="layers.json",
            export_profile="profile.json",
        )

        self.assertIn("--onnx=m.onnx", cmd)
        self.assertIn("--saveEngine=m.engine", cmd)
        self.assertIn("--noTF32", cmd)
        self.assertIn("--dumpLayerInfo", cmd)
        self.assertIn("--exportProfile=profile.json", cmd)

    def test_report_gate_requires_memory_path_components(self):
        report = {
            "components": {
                "memory_attention": {"trt_validation_pass": True},
                "mask_decoder": {"trt_validation_pass": True},
                "memory_encoder": {"trt_validation_pass": False},
            }
        }

        self.assertFalse(trt_components_usable(report, scope="memory_path_all"))
        report["components"]["memory_encoder"]["trt_validation_pass"] = True
        self.assertTrue(trt_components_usable(report, scope="memory_path_all"))


if __name__ == "__main__":
    unittest.main()
