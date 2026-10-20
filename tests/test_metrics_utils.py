import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "her2pred"))
from metrics_utils import binary_metrics


class PrAucTest(unittest.TestCase):
    def test_trapezoidal_area_is_independent_of_threshold(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.4, 0.35, 0.8]
        for threshold in (0.3, 0.5):
            with self.subTest(threshold=threshold):
                metrics = binary_metrics(labels, scores, threshold)
                self.assertAlmostEqual(metrics["pr_auc"], 19 / 24)

    def test_tied_scores(self):
        metrics = binary_metrics([0, 1, 0, 1], [0.5] * 4)
        self.assertAlmostEqual(metrics["pr_auc"], 0.75)

    def test_single_class_remains_undefined(self):
        metrics = binary_metrics([1, 1], [0.2, 0.8])
        self.assertTrue(math.isnan(metrics["pr_auc"]))


if __name__ == "__main__":
    unittest.main()
