"""
models/prophet_model.py
────────────────────────
Prophet для продаж Walmart M5.
Включаем регрессор has_event и snap для учёта праздников и SNAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet


def fit_predict(ts_agg: pd.DataFrame,
                test_size: int,
                regressors: list = None,
                plot_components: bool = True,
                save: bool = False) -> np.ndarray:
    """
    Обучает Prophet на агрегированном ряде.

    Parameters
    ----------
    ts_agg      : датафрейм с колонками sales + опциональные регрессоры
    test_size   : горизонт прогноза
    regressors  : список доп. регрессоров (напр. ['has_event', 'snap'])

    Returns
    -------
    np.ndarray прогнозов
    """
    df_tr = ts_agg.iloc[:-test_size].copy()
    df_te = ts_agg.iloc[-test_size:].copy()

    # Prophet требует колонки ds и y
    df_prophet = pd.DataFrame({
        "ds": df_tr.index,
        "y":  df_tr["sales"].values,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.5, # было 0.1 — делаем тренд гибче
    )

    # Добавляем доп. регрессоры
    if regressors:
        for reg in regressors:
            if reg in ts_agg.columns:
                df_prophet[reg] = df_tr[reg].values
                model.add_regressor(reg)

    model.fit(df_prophet)

    # Future dataframe
    future = pd.DataFrame({"ds": df_te.index})
    if regressors:
        for reg in regressors:
            if reg in ts_agg.columns:
                future[reg] = df_te[reg].values

    forecast = model.predict(future)
    pred = np.clip(forecast["yhat"].values, 0, None)

    if plot_components:
        full_future = model.make_future_dataframe(periods=test_size, freq="D")
        if regressors:
            for reg in regressors:
                if reg in ts_agg.columns:
                    full_future[reg] = pd.concat([
                        df_tr[reg], df_te[reg]
                    ]).values[-len(full_future):]
        full_forecast = model.predict(full_future)
        fig = model.plot_components(full_forecast)
        fig.suptitle("Prophet — компоненты (продажи)", y=1.01)
        plt.tight_layout()
        if save:
            fig.savefig("prophet_components.png", dpi=120)
        plt.show()

    print("✓ Prophet — прогноз готов")
    return pred
