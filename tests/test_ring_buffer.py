from __future__ import annotations

import unittest

import torch

from edgetam_batched.ring_buffer import TensorRingBuffer, TensorSpec


class RingBufferTests(unittest.TestCase):
    def test_ring_slots_rotate_and_copy(self):
        ring = TensorRingBuffer([TensorSpec((2,), torch.float32, "cpu")], ring_size=3)
        out0 = torch.tensor([1.0, 2.0])
        slot0 = ring.copy_from([out0])[0]
        self.assertTrue(torch.equal(slot0, out0))
        self.assertIsNot(slot0, out0)
        out1 = torch.tensor([3.0, 4.0])
        slot1 = ring.copy_from([out1])[0]
        self.assertTrue(torch.equal(slot1, out1))
        self.assertIsNot(slot0, slot1)

    def test_ring_size_minimum(self):
        with self.assertRaises(ValueError):
            TensorRingBuffer([TensorSpec((1,), torch.float32, "cpu")], ring_size=2)


if __name__ == "__main__":
    unittest.main()
