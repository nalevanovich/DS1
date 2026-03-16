"""
anomaly.py
──────────
Обнаружение аномалий в продажах Walmart M5:
  1. Z-score на дневных продажах
  2. IQR
  3. Isolation Forest (с признаками: sales, day_of_week, has_event)
  4. STL-остаток (residuals после декомпозиции)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


def detect(ts_agg: pd.DataFrame,
           zscore_threshold: float = 3.0,
           iqr_multiplier: float   = 3.0,
           contamination: float    = 0.02,
           plot: bool = True,
           save: bool = False) -> pd.DataFrame:
    """
    Запускает 4 метода обнаружения аномалий.

    Parameters
    ----------
    ts_agg : агрегированный ряд (выход preprocessing.make_agg_series)

    Returns
    -------
    pd.DataFrame с флагами аномалий по каждому методу
    """
    df = ts_agg[["sales"]].copy()
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month

    if "has_event" in ts_agg.columns:
        df["has_event"] = ts_agg["has_event"]
    else:
        df["has_event"] = 0

    # ── 1. Z-score ───────────────────────────────────────────
    mu, sigma = df["sales"].mean(), df["sales"].std()
    df["zscore"] = (df["sales"] - mu) / sigma
    df["anomaly_zscore"] = df["zscore"].abs() > zscore_threshold

    # ── 2. IQR ───────────────────────────────────────────────
    Q1 = df["sales"].quantile(0.25)
    Q3 = df["sales"].quantile(0.75)
    iqr = Q3 - Q1
    df["anomaly_iqr"] = (
        (df["sales"] < Q1 - iqr_multiplier * iqr) |
        (df["sales"] > Q3 + iqr_multiplier * iqr)
    )

    # ── 3. Isolation Forest ──────────────────────────────────
    feat_cols = ["sales", "day_of_week", "month", "has_event"]
    scaler = StandardScaler()
    iso_X = scaler.fit_transform(df[feat_cols])
    iso = IsolationForest(contamination=contamination,
                          n_estimators=200, random_state=42)
    df["anomaly_isoforest"] = iso.fit_predict(iso_X) == -1

    # ── 4. STL-остатки ───────────────────────────────────────
    try:
        stl = STL(df["sales"], period=7, robust=True)
        stl_res = stl.fit()
        residuals = stl_res.resid
        res_std = residuals.std()
        df["stl_residual"] = residuals.values
        df["anomaly_stl"] = residuals.abs() > 3 * res_std
    except Exception:
        df["stl_residual"] = np.nan
        df["anomaly_stl"] = False

    # ── Сводка ───────────────────────────────────────────────
    print(f"✓ Аномалии Z-score  (|z|>{zscore_threshold}):  {df['anomaly_zscore'].sum()}")
    print(f"✓ Аномалии IQR      ({iqr_multiplier}×IQR):    {df['anomaly_iqr'].sum()}")
    print(f"✓ Аномалии IsoForest:                           {df['anomaly_isoforest'].sum()}")
    print(f"✓ Аномалии STL-остаток:                         {df['anomaly_stl'].sum()}")

    if plot:
        _plot(df, save)

    return df

def _plot(df: pd.DataFrame, save: bool = False) -> None:
    methods = {
        "Z-score":          "anomaly_zscore",
        "IQR":              "anomaly_iqr",
        "Isolation Forest": "anomaly_isoforest",
        "STL-остаток":      "anomaly_stl",
    }
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    for ax, (name, col) in zip(axes, methods.items()):
        ax.plot(df.index, df["sales"],
                color=PALETTE[0], linewidth=0.8, alpha=0.7)
        idx = df[df[col]].index
        ax.scatter(idx, df.loc[idx, "sales"],
                   color=PALETTE[1], s=30, zorder=5,
                   label=f"Аномалия ({len(idx)})")
        ax.set_title(f"{name}")
        ax.set_ylabel("Продажи")
        ax.legend(loc="upper left", fontsize=9)

    plt.suptitle("Обнаружение аномалий в продажах Walmart M5",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("anomaly_detection.png", dpi=120)
    plt.show()
