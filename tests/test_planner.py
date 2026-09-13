import importlib.util
import pathlib
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("planner", ROOT / "planner.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


class PlannerTests(unittest.TestCase):
    def run_plan(self, masks=None, frames=None, **kwargs):
        if frames is None:
            frames = np.zeros((3, 60, 100, 3), dtype=np.float32)
        return planner.plan_region(lambda i: frames[i], len(frames), 100, 60, 50, 10,
                                   mask_at=None if masks is None else lambda i: masks[i],
                                   **kwargs)

    def test_quiet_frame_prefers_bottom_center(self):
        result = self.run_plan()
        self.assertEqual(result["box"], dict(x=25, y=50, width=50, height=10))
        self.assertFalse(result["all_frames_mask_constraint_met"])
        self.assertIsNone(result["max_mask_overlap"])

    def test_subject_enters_bottom_only_in_middle(self):
        masks = np.zeros((3, 60, 100), dtype=np.float32)
        masks[1, 30:, :] = 1
        result = self.run_plan(masks, max_overlap=0)
        self.assertTrue(result["all_frames_mask_constraint_met"])
        self.assertLessEqual(result["box"]["y"]+10, 30)
        self.assertEqual(result["max_mask_overlap"], 0)

    def test_no_space_is_flagged(self):
        result = self.run_plan(np.ones((3, 60, 100), dtype=np.float32))
        self.assertFalse(result["sampled_mask_constraint_met"])
        self.assertEqual(result["max_mask_overlap"], 1)
        self.assertTrue(result["warnings"])

    def test_sampling_cannot_claim_all_frames_checked(self):
        result = self.run_plan(np.zeros((3, 60, 100)), sample_step=2)
        self.assertTrue(result["sampled_mask_constraint_met"])
        self.assertFalse(result["all_frames_mask_constraint_met"])
        self.assertEqual(result["sampled_frames"], [0, 2])

    def test_end_frame_is_always_sampled(self):
        frames = np.zeros((4, 60, 100, 3))
        result = self.run_plan(frames=frames, sample_step=2)
        self.assertEqual(result["sampled_frames"], [0, 2, 3])

    def test_interval_excludes_unrelated_subject(self):
        masks = np.zeros((3, 60, 100), dtype=np.float32)
        masks[0] = 1
        result = self.run_plan(masks, start=1, end=3)
        self.assertTrue(result["all_frames_mask_constraint_met"])
        self.assertEqual(result["sampled_frames"], [1, 2])

    def test_invalid_ranges_and_geometry(self):
        for arguments in ({"start": 3}, {"end": 4}, {"sample_step": 0},
                          {"margin": 30}, {"max_overlap": float("nan")}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.run_plan(**arguments)

    def test_bad_mask_and_pixels(self):
        with self.assertRaises(ValueError):
            self.run_plan(np.zeros((3, 20, 20)))
        frames = np.zeros((3, 60, 100, 3))
        frames[1, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.run_plan(frames=frames)

    def test_scoring_matches_direct_means(self):
        rng = np.random.default_rng(123)
        mask = rng.random((60, 100), dtype=np.float32)
        boxes = planner.candidate_boxes(100, 60, 50, 10, 2)
        expected = [mask[y:y+h, x:x+w].mean() for x, y, w, h in boxes]
        np.testing.assert_allclose(planner.rectangle_means(mask, boxes), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
