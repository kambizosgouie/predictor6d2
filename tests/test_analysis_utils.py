import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from analysis_utils import (
    correlation_table,
    distance_correlation,
    partial_dependence_curve,
    permutation_importance_table,
    standardized_coefficients,
    variance_inflation_factors,
)


def sample_frame(rows=120):
    rng = np.random.default_rng(0)
    a = rng.normal(size=rows)
    b = rng.normal(size=rows)
    c = a + 0.01 * rng.normal(size=rows)
    return pd.DataFrame({"a": a, "b": b, "c": c, "y": 3 * a + 0.1 * b})


class AnalysisUtilsTests(unittest.TestCase):
    def test_distance_correlation_detects_nonlinear_dependence(self):
        x = np.linspace(-3, 3, 200)
        self.assertGreater(distance_correlation(x, x ** 2), 0.4)
        self.assertAlmostEqual(distance_correlation(x, x), 1.0, places=6)
        self.assertEqual(distance_correlation(x, np.ones_like(x)), 0.0)

    def test_correlation_table_has_all_methods(self):
        frame = sample_frame()
        table = correlation_table(frame, ["a", "b", "c"], "y")
        self.assertEqual(
            list(table.columns),
            ["Pearson", "Spearman", "Kendall", "Distance corr.", "Mutual info"],
        )
        self.assertGreater(table.loc["a", "Pearson"], 0.9)

    def test_vif_flags_collinear_features(self):
        vif = variance_inflation_factors(sample_frame(), ["a", "b", "c"])
        self.assertGreater(vif["a"], 100)
        self.assertLess(vif["b"], 2)

    def test_permutation_importance_ranks_informative_feature_first(self):
        frame = sample_frame()
        features = ["a", "b"]
        model = LinearRegression().fit(frame[features], frame["y"])
        table = permutation_importance_table(
            [("Linear", model)], frame[features], frame["y"], n_repeats=3
        )
        self.assertEqual(table["Linear"].idxmax(), "a")

    def test_standardized_coefficients_and_partial_dependence(self):
        frame = sample_frame()
        features = ["a", "b"]
        model = LinearRegression().fit(frame[features], frame["y"])
        coefficients = standardized_coefficients(model, features, frame[features], frame["y"])
        self.assertGreater(abs(coefficients["a"]), abs(coefficients["b"]))
        grid, curve = partial_dependence_curve(model, frame[features], "a")
        self.assertGreater(curve[-1], curve[0])
        self.assertIsNone(
            standardized_coefficients(object(), features, frame[features], frame["y"])
        )


if __name__ == "__main__":
    unittest.main()
