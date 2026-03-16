#data_loader.py

import pandas as pd
from pathlib import Path


# ── Описание колонок ─────────────────────────────────────────

COLUMN_DESCRIPTIONS = {
    # sales_train_validation.csv
    "id":           "Уникальный ID товара (item_dept_cat_store_state_validation)",
    "item_id":      "ID товара (напр. HOBBIES_1_001)",
    "dept_id":      "ID отдела (напр. HOBBIES_1)",
    "cat_id":       "Категория: HOBBIES / HOUSEHOLD / FOODS",
    "store_id":     "ID магазина (напр. CA_1)",
    "state_id":     "Штат: CA / TX / WI",
    "d_1..d_1913":  "Ежедневные продажи в штуках (d_1 = 2011-01-29)",
    # calendar.csv
    "date":         "Дата",
    "wm_yr_wk":     "Неделя в формате Walmart",
    "weekday":      "День недели (строка)",
    "wday":         "День недели (1=воскресенье)",
    "month":        "Месяц",
    "year":         "Год",
    "event_name_1": "Название праздника/события 1",
    "event_type_1": "Тип события 1 (Cultural/National/Religious/Sporting)",
    "event_name_2": "Название события 2 (если есть)",
    "event_type_2": "Тип события 2",
    "snap_CA/TX/WI":"Флаг SNAP (программа продовольственных талонов) по штату",
    # sell_prices.csv
    "sell_price":   "Цена товара в данном магазине на данной неделе",
}


def describe_columns() -> None:
    """Печатает описание всех колонок датасета M5."""
    print(f"\n{'Колонка':<22} {'Описание'}")
    print("─" * 75)
    for col, desc in COLUMN_DESCRIPTIONS.items():
        print(f"  {col:<20} {desc}")


def load(data_dir: str = "data",
         store_id: str = "CA_1",
         aggregate: bool = True) -> dict:
    """
    Загружает файлы M5 и возвращает словарь датафреймов.

    Parameters
    ----------
    data_dir  : папка с CSV-файлами от Kaggle
    store_id  : фильтр по магазину (None = все магазины)
    aggregate : если True — суммирует продажи по всем товарам магазина
                в один ряд

    Returns
    -------
    dict с ключами: 'sales', 'calendar', 'prices', 'ts'
      sales    — исходные данные по товарам (wide format)
      calendar — календарь событий и SNAP
      prices   — цены товаров
      ts       — готовый временной ряд (long format, merged)
    """
    data_dir = Path(data_dir)

    print("Загрузка файлов M5...")
    sales    = pd.read_csv(data_dir / "sales_train_validation.csv")
    calendar = pd.read_csv(data_dir / "calendar.csv")
    prices   = pd.read_csv(data_dir / "sell_prices.csv")

    # Фильтр по магазину
    if store_id:
        sales  = sales[sales["store_id"] == store_id].reset_index(drop=True)
        prices = prices[prices["store_id"] == store_id].reset_index(drop=True)
        print(f"  Магазин: {store_id}  |  товаров: {len(sales)}")

    print(f"  Дней: {sales.shape[1] - 6}  |  Календарь: {len(calendar)} строк")

    # Преобразуем в long format
    day_cols  = [c for c in sales.columns if c.startswith("d_")]
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    sales_long = sales.melt(
        id_vars=meta_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    # Присоединяем календарь
    ts = sales_long.merge(
        calendar[["d", "date", "wm_yr_wk", "wday", "month", "year",
                  "event_name_1", "event_type_1",
                  "event_name_2", "event_type_2",
                  "snap_CA", "snap_TX", "snap_WI"]],
        on="d", how="left"
    )
    ts["date"] = pd.to_datetime(ts["date"])

    # Присоединяем цены
    ts = ts.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    if aggregate:
        # Агрегируем: суммарные продажи по магазину на каждый день
        ts_agg = (
            ts.groupby("date")
            .agg(sales=("sales", "sum"),
                 avg_price=("sell_price", "mean"),
                 event=("event_name_1", lambda x: x.notna().any().astype(int)))
            .reset_index()
            .set_index("date")
            .sort_index()
        )
        print(f"  ✓ Агрегированный ряд: {len(ts_agg)} дней")
        return {"sales": sales, "calendar": calendar,
                "prices": prices, "ts": ts, "ts_agg": ts_agg}

    ts = ts.sort_values(["item_id", "date"]).reset_index(drop=True)
    print(f"  ✓ Long format: {len(ts):,} строк")
    return {"sales": sales, "calendar": calendar, "prices": prices, "ts": ts}