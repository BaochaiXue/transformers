from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from edgetam_batched.component_fixtures import ComponentFixtureRecorder, to_cpu_nested


class ComponentFixtureTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_to_cpu_nested_clones_tensors(self):
        value = {"x": torch.ones((2, 2)), "meta": "ok"}

        copied = to_cpu_nested(value)

        self.assertEqual(copied["x"].device.type, "cpu")
        self.assertTrue(torch.equal(copied["x"], value["x"]))
        self.assertEqual(copied["meta"], "ok")

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_recorder_writes_component_fixture(self):
        module = torch.nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ComponentFixtureRecorder(tmp)
            handle = recorder.attach(module, "linear")
            try:
                recorder.begin(frame_idx=0, camera_idx=1, session_digest_before={"before": True})
                _ = module(torch.zeros((1, 2)))
                recorder.end(session_digest_after={"after": True})
            finally:
                handle.remove()

            path = Path(tmp) / "frame_000" / "cam1_linear.pt"
            payload = torch.load(path, weights_only=False)

        self.assertEqual(payload["component"], "linear")
        self.assertEqual(payload["camera_idx"], 1)
        self.assertEqual(payload["session_digest_after"], {"after": True})


if __name__ == "__main__":
    unittest.main()
