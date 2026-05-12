import unittest

from edgetam_batched.onnx_trt.report_trt import merge_component_reports


class TrtReportTests(unittest.TestCase):
    def test_merge_component_reports_marks_unpassed_stage(self):
        reports = [
            {
                "component": "memory_attention",
                "onnx_export_pass": True,
                "onnx_validation_pass": True,
                "trt_build_pass": True,
                "trt_validation_pass": True,
            },
            {
                "component": "mask_decoder",
                "onnx_export_pass": True,
                "onnx_validation_pass": False,
            },
        ]

        merged = merge_component_reports(reports)

        self.assertTrue(merged["memory_attention"]["trt_validation_pass"])
        self.assertEqual(merged["mask_decoder"]["failure_stage"], "onnx_validation")
        self.assertEqual(merged["memory_encoder"]["failure_stage"], "onnx_export")


if __name__ == "__main__":
    unittest.main()
