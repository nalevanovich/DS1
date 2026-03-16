"""
models/arima_model.py
─────────────────────
ARIMA для агрегированных продаж.
Для продаж используем order=(7,1,1) — учитываем недельный лаг.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def fit_predict(y_train: np.ndarray,
                test_size: int,
                order: tuple = (7, 1, 1)) -> np.ndarray:
    """
    Обучает ARIMA и возвращает прогноз на test_size шагов.

    Parameters
    ----------
    y_train   : обучающий ряд (суммарные дневные продажи)
    test_size : горизонт прогноза
    order     : (p, d, q). Для продаж рекомендуется (7,1,1)

    Returns
    -------
    np.ndarray прогнозов (могут быть отрицательными → clip в 0)
    """
    model  = ARIMA(y_train, order=order)
    fitted = model.fit()
    pred   = fitted.forecast(steps=test_size)
    pred   = np.clip(pred, 0, None)   # продажи не могут быть < 0

    print(f"✓ ARIMA{order} — прогноз готов")
    return pred
