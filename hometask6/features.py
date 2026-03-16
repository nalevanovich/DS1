"""
features.py
───────────
Инженерия признаков для агрегированного ряда продаж Walmart M5:
  - лаги продаж (7, 14, 28, 35, 42 дней)
  - скользящие средние и стандартное отклонение
  - скользящие min/max
  - тренд (порядковый номер дня)
  - временные признаки (день недели, месяц, квартал, год)
  - флаги: выходной, SNAP, событие
  - ценовые признаки (средняя цена, rolling avg цены)
"""

import numpy as np
import pandas as pd


def build(ts_agg: pd.DataFrame,
          lags: list = [7, 14, 28, 35, 42],
          ma_windows: list = [7, 14, 28]) -> tuple:
    """
    Создаёт признаки из агрегированного дневного ряда.

    Parameters
    ----------
    ts_agg     : выход preprocessing.make_agg_series()
    lags       : лаги в днях (кратно 7 — важно для продаж)
    ma_windows : окна скользящих средних

    Returns
    -------
    (df_features, feature_cols)
    """
    df = ts_agg.copy()
    sales = df["sales"]

    # ── Лаги ─────────────────────────────────────────────────
    for lag in lags:
        df[f"lag_{lag}"] = sales.shift(lag)

    # ── Скользящие статистики ─────────────────────────────────
    for w in ma_windows:
        df[f"ma_{w}"]    = sales.shift(1).rolling(w).mean()
        df[f"std_{w}"]   = sales.shift(1).rolling(w).std()
        df[f"min_{w}"]   = sales.shift(1).rolling(w).min()
        df[f"max_{w}"]   = sales.shift(1).rolling(w).max()

    # Недельный тренд (разность lag_7)
    df["weekly_diff"]    = sales.shift(7) - sales.shift(14)

    # ── Тренд ────────────────────────────────────────────────
    df["trend"] = np.arange(len(df))

    # ── Временные признаки ───────────────────────────────────
    df["day_of_week"] = df.index.dayofweek        # 0=пн..6=вс
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = df.index.month
    df["quarter"]     = df.index.quarter
    df["year"]        = df.index.year
    df["day_of_year"] = df.index.dayofyear
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    # Синус/косинус для цикличности
    df["sin_wday"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_wday"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Ценовые признаки ─────────────────────────────────────
    if "sell_price" in df.columns:
        df["price_ma_7"]   = df["sell_price"].shift(1).rolling(7).mean()
        df["price_change"] = df["sell_price"].pct_change(7)

    # ── Событийные признаки ──────────────────────────────────
    if "has_event" in df.columns:
        df["event_lag1"] = df["has_event"].shift(1)
        df["event_lead1"] = df["has_event"].shift(-1)

    # ── Признак аномалии ─────────────────────────────────────
    if "is_anomaly" in df.columns:
        df["is_anomaly"] = df["is_anomaly"].astype(int)
        
    # Убираем строки с NaN (из-за лагов)
    df = df.dropna()

    # Список признаков
    exclude = {"sales", "sell_price"}
    feature_cols = [c for c in df.columns if c not in exclude]

    print(f"✓ Признаков: {len(feature_cols)}  |  Строк: {len(df)}")
    return df, feature_cols


def train_test_split(df_feat: pd.DataFrame,
                     feature_cols: list,
                     test_size: int = 28) -> tuple:
    """
    Временное разбиение train/test.

    Parameters
    ----------
    test_size : последние N дней → test (default 28 = 4 недели)

    Returns
    -------
    X_train, X_test, y_train, y_test, dates_train, dates_test
    """
    X = df_feat[feature_cols].values
    y = df_feat["sales"].values
    dates = df_feat.index

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    d_train, d_test = dates[:-test_size], dates[-test_size:]

    print(f"✓ Train: {d_train[0].date()} — {d_train[-1].date()}  ({len(y_train)} obs)")
    print(f"✓ Test : {d_test[0].date()}  — {d_test[-1].date()}  ({len(y_test)} obs)")
    return X_train, X_test, y_train, y_test, d_train, d_test
