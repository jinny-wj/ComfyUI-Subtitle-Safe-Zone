"""Real tensor tests; optional on development machines without PyTorch."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this interpreter")
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "safezone_runtime_test", ROOT/"__init__.py", submodule_search_locations=[str(ROOT)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.node_class = module.NODE_CLASS_MAPPINGS["SubtitleSafeZonePlanner"]

    def setUp(self):
        self.images = torch.zeros((8, 120, 200, 3), dtype=torch.float32)
        self.masks = torch.zeros((8, 120, 200), dtype=torch.float32)
        self.masks[3, 60:, :] = 1
        self.node = self.node_class()

    def test_actual_outputs_and_input_preservation(self):
        images_before, masks_before = self.images.clone(), self.masks.clone()
        out = self.node.plan(self.images, protect_mask=self.masks, max_overlap=0)
        preview, region, x, y, w, h, met, report_text = out
        report = json.loads(report_text)
        self.assertEqual(len(out), 8)
        self.assertEqual(tuple(preview.shape), (3, 120, 200, 3))
        self.assertEqual(tuple(region.shape), (1, 120, 200))
        self.assertEqual(preview.dtype, torch.float32)
        self.assertEqual(region.dtype, torch.float32)
        self.assertTrue(torch.isfinite(preview).all())
        self.assertTrue((preview >= 0).all() and (preview <= 1).all())
        self.assertEqual(region.sum().item(), w*h)
        self.assertTrue(met)
        self.assertLessEqual(y+h, 60)
        self.assertEqual(report["plugin_version"], "0.1.1")
        self.assertEqual(report["preview_frame_indices"], [0, 3, 7])
        self.assertTrue(torch.equal(self.images, images_before))
        self.assertTrue(torch.equal(self.masks, masks_before))

    def test_single_mask_and_2d_mask_are_equivalent(self):
        out1 = self.node.plan(self.images, protect_mask=self.masks[3:4])
        out2 = self.node.plan(self.images, protect_mask=self.masks[3])
        self.assertEqual(out1[2:], out2[2:])
        self.assertTrue(torch.equal(out1[0], out2[0]))

    def test_single_frame_and_no_mask(self):
        result = self.node.plan(self.images[:1])
        self.assertEqual(result[0].shape[0], 1)
        self.assertFalse(result[6])
        self.assertTrue(json.loads(result[7])["warnings"])

    def test_all_protected_is_not_reported_as_clear(self):
        result = self.node.plan(self.images, protect_mask=torch.ones_like(self.masks))
        self.assertFalse(result[6])
        self.assertEqual(json.loads(result[7])["max_mask_overlap"], 1)

    def test_invalid_mask_dimensions_and_empty_images(self):
        for mask in (torch.zeros((2, 120, 200)), torch.zeros((1, 60, 100))):
            with self.assertRaises(ValueError):
                self.node.plan(self.images, protect_mask=mask)
        with self.assertRaises(ValueError):
            self.node.plan(self.images[:0])

    def test_cpu_half_and_noncontiguous_inputs(self):
        # Full result equivalence verifies adapters accept normal upstream layouts.
        noncontiguous = self.images.transpose(1, 2).contiguous().transpose(1, 2)
        a = self.node.plan(self.images)
        b = self.node.plan(noncontiguous.half())
        self.assertEqual(a[2:], b[2:])
        self.assertTrue(torch.equal(a[0], b[0]))

    def test_mps_inputs_when_available(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable to this process")
        a = self.node.plan(self.images, protect_mask=self.masks)
        gpu_images, gpu_masks = self.images.to("mps"), self.masks.to("mps")
        b = self.node.plan(gpu_images, protect_mask=gpu_masks)
        self.assertEqual(a[2:], b[2:])
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(gpu_images.cpu(), self.images))
        self.assertEqual(b[0].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
