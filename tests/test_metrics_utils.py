import unittest

from metrics_utils import compute_regression_metrics


class MetricsUtilsTests(unittest.TestCase):
    def test_percentage_metrics_are_computed(self):
        y_true = [100, 200, 300]
        y_pred = [90, 220, 330]

        metrics = compute_regression_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["r2"], 0.93, places=2)
        self.assertAlmostEqual(metrics["mae"], 20.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 21.602469, places=6)
        self.assertAlmostEqual(metrics["normalized_percentage"], 10.0, places=6)
        self.assertAlmostEqual(metrics["mae_pct_of_mean"], 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
