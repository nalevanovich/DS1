# metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmsle(y_true_log, y_pred_log):
    """RMSLE on log-transformed target."""
    return np.sqrt(mean_squared_error(y_true_log, y_pred_log))


def evaluate(name, y_true_log, y_pred_log):
    rmsl = rmsle(y_true_log, y_pred_log)
    mae  = mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))
    r2   = r2_score(y_true_log, y_pred_log)
    print(f"  [{name}]  RMSLE={rmsl:.4f}  MAE={mae:.1f}s  R²={r2:.4f}")
    return {"model": name, "RMSLE": rmsl, "MAE_sec": mae, "R2": r2}