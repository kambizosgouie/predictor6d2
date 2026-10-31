import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    abs_errors = np.abs(y_true - y_pred)
    non_zero = y_true != 0
    perc_errors = np.full(y_true.shape, np.nan, dtype=float)
    perc_errors[non_zero] = np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]) * 100.0
    mean_true = float(np.mean(y_true))

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.nanmean(perc_errors)),
        "aad": float(np.nanmean(perc_errors)),
        "mae_pct_of_mean": float(mean_absolute_error(y_true, y_pred) / mean_true * 100.0) if mean_true != 0 else float("nan"),
    }
