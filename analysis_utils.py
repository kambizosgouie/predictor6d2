from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance


def _double_centered(values: np.ndarray) -> np.ndarray:
    distances = np.abs(values[:, None] - values[None, :])
    return (
        distances
        - distances.mean(axis=0, keepdims=True)
        - distances.mean(axis=1, keepdims=True)
        + distances.mean()
    )


def distance_correlation(x, y) -> float:
    """Székely distance correlation; detects any dependence, range 0..1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = _double_centered(x)
    b = _double_centered(y)
    dcov2 = (a * b).mean()
    denominator = np.sqrt((a * a).mean() * (b * b).mean())
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / denominator))


def correlation_table(frame: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """Association of every feature with the target using five methods."""
    rows = {}
    for method in ("pearson", "spearman", "kendall"):
        rows[method.capitalize()] = (
            frame[features + [target]].corr(method=method).loc[target, features]
        )
    rows["Distance corr."] = pd.Series(
        {f: distance_correlation(frame[f], frame[target]) for f in features}
    )
    mutual = mutual_info_regression(
        frame[features].values, frame[target].values, random_state=42
    )
    rows["Mutual info"] = pd.Series(mutual, index=features)
    return pd.DataFrame(rows)


def variance_inflation_factors(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    """VIF per feature; values above ~5-10 signal multicollinearity."""
    values = frame[features].values.astype(float)
    result = {}
    for index, name in enumerate(features):
        others = np.delete(values, index, axis=1)
        if others.shape[1] == 0:
            result[name] = 1.0
            continue
        design = np.column_stack([np.ones(len(others)), others])
        target = values[:, index]
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        total = ((target - target.mean()) ** 2).sum()
        r2 = 1.0 - (residual ** 2).sum() / total if total > 0 else 0.0
        result[name] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(result, name="VIF")


def permutation_importance_table(models, x_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    """Mean drop in R² when a feature is shuffled, for each fitted model."""
    columns = {}
    for model_name, model in models:
        result = permutation_importance(
            model, x_test, y_test, scoring="r2",
            n_repeats=n_repeats, random_state=42,
        )
        columns[model_name] = pd.Series(result.importances_mean, index=x_test.columns)
    return pd.DataFrame(columns)


def standardized_coefficients(model, features, x_train, y_train) -> pd.Series | None:
    """Coefficients rescaled to per-standard-deviation effect on the target."""
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return None
    coefficients = np.ravel(coefficients)
    if len(coefficients) != len(features):
        return None
    y_std = float(np.std(y_train))
    if y_std == 0:
        return None
    return pd.Series(coefficients * x_train.std().values / y_std, index=features)


def partial_dependence_curve(model, frame: pd.DataFrame, feature: str, points: int = 25):
    """Average prediction while sweeping one feature over its observed range."""
    grid = np.unique(np.quantile(frame[feature], np.linspace(0, 1, points)))
    averages = []
    for value in grid:
        shifted = frame.copy()
        shifted[feature] = value
        averages.append(float(np.mean(model.predict(shifted))))
    return grid, np.array(averages)
