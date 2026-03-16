"""
metrics.py
──────────
Вычисление метрик качества прогнозирования.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score)


def evaluate(y_true: np.ndarray,
             y_pred: np.ndarray,
             name: str = "Model") -> dict:
    """MAE, RMSE, MAPE, R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE: защита от нулей
    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
    r2   = r2_score(y_true, y_pred)

    print(f"  [{name:<15}]  MAE={mae:8.1f}  RMSE={rmse:8.1f}  "
          f"MAPE={mape:6.2f}%  R²={r2:.4f}")
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE%": mape, "R2": r2}


def summary_table(metrics_list: list) -> pd.DataFrame:
    """Красивый DataFrame из списка dict-метрик."""
    return pd.DataFrame(metrics_list).set_index("Model").round(3)
