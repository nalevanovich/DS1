"""
preprocessing.py
────────────────
Предобработка данных Walmart M5:
  - заполнение пропусков в продажах (fillna 0)
  - заполнение пропусков в ценах (forward fill per item)
  - флаги событий (has_event, has_event2, event_type_enc)
  - удаление ненужных строковых колонок
"""

import numpy as np
import pandas as pd


def process(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Полная предобработка long-format датафрейма M5.

    Parameters
    ----------
    ts : long-format DF из data_loader.load()

    Returns
    -------
    Чистый pd.DataFrame, готовый для make_agg_series()
    """
    df = ts.copy()

    # ── 1. Пропуски в продажах ───────────────────────────────
    before = df["sales"].isnull().sum()
    df["sales"] = df["sales"].fillna(0)
    print(f"✓ Пропуски в sales: {before} → заполнены нулями")

    # ── 2. Пропуски в цене — forward fill per item ───────────
    price_na = df["sell_price"].isnull().sum()
    df["sell_price"] = (
        df.sort_values("date")
        .groupby("item_id")["sell_price"]
        .transform(lambda x: x.ffill().bfill())
    )
    print(f"✓ Пропуски в sell_price: {price_na} → {df['sell_price'].isnull().sum()}")

    # ── 3. Флаг события ──────────────────────────────────────
    df["has_event"] = df["event_name_1"].notna().astype(np.int8)

    # ── 4. Убираем ненужные строковые колонки ────────────────
    drop_cols = ["event_name_1", "event_type_1",
                 "event_name_2", "event_type_2",
                 "snap_CA", "snap_TX", "snap_WI", "d"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"✓ Итого строк: {len(df):,}  |  колонок: {df.shape[1]}")
    return df


def make_agg_series(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует продажи по магазину в дневной ряд.
    Используется для всех моделей (ARIMA, Prophet, RF, LSTM...).
    """
    ts_agg = (
        ts.groupby("date")
        .agg(sales=("sales", "sum"),
             sell_price=("sell_price", "mean"),
             has_event=("has_event", "max"))
        .sort_index()
    )
    ts_agg.index.name = "date"
    print(f"✓ Агрегированный ряд: {len(ts_agg)} дней")
    return ts_agg
