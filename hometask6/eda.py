"""
eda.py
──────
Разведочный анализ данных Walmart M5:
  - общая статистика продаж
  - иерархическая структура
  - временной ряд агрегированных продаж
  - сезонность (дни недели, месяцы)
  - влияние событий и SNAP
  - распределение нулевых продаж
  - тесты на стационарность (ADF, KPSS)
  - декомпозиция и ACF/PACF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


def plot_hierarchy(data: dict, save: bool = False) -> None:
    """Иерархическая структура датасета: категории и отделы."""
    sales = data["sales"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, col, title in zip(
        axes,
        ["cat_id", "dept_id"],
        ["По категориям", "По отделам"]
    ):
        counts = sales[col].value_counts()
        counts.plot(kind="bar", ax=ax, color=PALETTE[:len(counts)], edgecolor="none")
        ax.set_title(f"Кол-во товаров: {title}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Иерархическая структура Walmart M5", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("eda_hierarchy.png", dpi=120)
    plt.show()


def plot_aggregate_sales(ts_agg: pd.DataFrame, save: bool = False) -> None:
    """Агрегированный ряд продаж магазина: дневной + недельный."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    # ── 1. Недельные продажи (читаемо) ───────────────────────
    weekly = ts_agg["sales"].resample("W").sum()
    ma_12w = weekly.rolling(12).mean()
    axes[0].bar(weekly.index, weekly.values,
                color=PALETTE[0], alpha=0.5, width=5, label="Недельные продажи")
    axes[0].plot(weekly.index, ma_12w,
                 color=PALETTE[1], linewidth=1.5, label="MA-12 недель")
    axes[0].set_title("Суммарные продажи магазина (по неделям)")
    axes[0].set_ylabel("Продажи, шт.")
    axes[0].legend()

    # ── 2. Дневные продажи с MA ───────────────────────────────
    ma30 = ts_agg["sales"].rolling(30).mean()
    axes[1].plot(ts_agg.index, ts_agg["sales"],
                 color=PALETTE[0], linewidth=0.6, alpha=0.5, label="Дневные продажи")
    axes[1].plot(ts_agg.index, ma30,
                 color=PALETTE[1], linewidth=1.5, label="MA-30 дней")
    # События
    event_days = ts_agg[ts_agg["event"] == 1].index
    axes[1].scatter(event_days, ts_agg.loc[event_days, "sales"],
                    color=PALETTE[2], s=15, zorder=5, label="Событие", alpha=0.7)
    axes[1].set_title("Суммарные продажи магазина (дневные)")
    axes[1].set_ylabel("Продажи, шт.")
    axes[1].legend()

    # ── 3. Средняя цена ───────────────────────────────────────
    axes[2].plot(ts_agg.index, ts_agg["avg_price"],
                 color=PALETTE[3], linewidth=0.8)
    axes[2].set_title("Средняя цена товара")
    axes[2].set_ylabel("Цена, $")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        plt.savefig("eda_aggregate_sales.png", dpi=120)
    plt.show()


def plot_seasonality(ts_agg: pd.DataFrame, save: bool = False) -> None:
    """Сезонность: суммарные продажи по дням недели и по месяцам."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # По дням недели (0=Пн .. 6=Вс)
    day_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    by_wday = ts_agg["sales"].groupby(ts_agg.index.dayofweek).mean()
    axes[0].bar(by_wday.index, by_wday.values,
                color=PALETTE[:len(by_wday)], edgecolor="none")
    axes[0].set_xticks(by_wday.index)
    axes[0].set_xticklabels(day_names)
    axes[0].set_title("Средние продажи по дням недели")
    axes[0].set_ylabel("Продажи, шт.")

    # По месяцам
    by_month = ts_agg["sales"].groupby(ts_agg.index.month).mean()
    month_names = ["Янв","Фев","Мар","Апр","Май","Июн",
                   "Июл","Авг","Сен","Окт","Ноя","Дек"]
    axes[1].bar(by_month.index, by_month.values,
                color=PALETTE[2], edgecolor="none", alpha=0.85)
    axes[1].set_xticks(by_month.index)
    axes[1].set_xticklabels(month_names, rotation=30)
    axes[1].set_title("Средние продажи по месяцам")
    axes[1].set_ylabel("Продажи, шт.")

    plt.suptitle("Сезонность продаж", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("eda_seasonality.png", dpi=120)
    plt.show()


def plot_event_impact(ts: pd.DataFrame, save: bool = False) -> None:
    """Влияние событий и SNAP на продажи."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # SNAP vs no SNAP (берём штат CA как пример)
    snap_col = "snap_CA"
    if snap_col in ts.columns:
        snap_sales = ts.groupby(snap_col)["sales"].mean()
        axes[0].bar(["Обычный день", "SNAP день"],
                    snap_sales.values,
                    color=[PALETTE[0], PALETTE[2]], edgecolor="none")
        axes[0].set_title("Продажи: обычный vs SNAP день (CA)")
        axes[0].set_ylabel("Средние продажи, шт.")

    # События по типу
    ts_ev = ts[ts["event_type_1"].notna()].copy()
    if len(ts_ev) > 0:
        by_event = ts_ev.groupby("event_type_1")["sales"].mean().sort_values()
        no_event  = ts[ts["event_type_1"].isna()]["sales"].mean()
        all_vals  = pd.concat([by_event,
                               pd.Series({"Нет события": no_event})])
        colors = [PALETTE[1] if k != "Нет события" else PALETTE[4]
                  for k in all_vals.index]
        axes[1].barh(all_vals.index, all_vals.values,
                     color=colors, edgecolor="none")
        axes[1].set_title("Средние продажи по типу события")
        axes[1].set_xlabel("Продажи, шт.")

    plt.suptitle("Влияние событий и SNAP", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("eda_events.png", dpi=120)
    plt.show()


def plot_zero_sales(ts: pd.DataFrame, save: bool = False) -> None:
    """Доля нулевых продаж по товарам."""
    zero_pct = (
        ts.groupby("item_id")["sales"]
        .apply(lambda x: (x == 0).mean() * 100)
    )
    median_val = zero_pct.median()
    over50 = (zero_pct > 50).sum()
    over90 = (zero_pct > 90).sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(zero_pct, bins=40, color=PALETTE[0], edgecolor="none", alpha=0.85)

    ax.axvline(median_val, color=PALETTE[1], linestyle="--",
               linewidth=1.5, label=f"Медиана: {median_val:.1f}%")

    # Аннотации зон
    ax.axvspan(0,  50, alpha=0.05, color=PALETTE[2],
               label=f"< 50% нулей: {(zero_pct <= 50).sum()} товаров (продаются часто)")
    ax.axvspan(50, 100, alpha=0.05, color=PALETTE[1],
               label=f"> 50% нулей: {over50} товаров (продаются редко)")

    ax.annotate(f"{over90} товаров\nне продаются\n>90% времени",
                xy=(93, ax.get_ylim()[1] * 0.5),
                fontsize=9, color=PALETTE[1],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_title("Доля нулевых продаж по товарам\n"
                 "(0% = товар продаётся каждый день, 100% = никогда не продаётся)")
    ax.set_xlabel("% дней без продаж")
    ax.set_ylabel("Кол-во товаров")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig("eda_zeros.png", dpi=120)
    plt.show()

    print(f"  Медиана нулевых продаж: {median_val:.1f}%")
    print(f"  Товаров с >50% нулей:   {over50} из {len(zero_pct)}")
    print(f"  Товаров с >90% нулей:   {over90} — редкие товары")
    print(f"  Вывод: агрегация по магазину устраняет проблему нулей")


def stationarity_tests(series: pd.Series, name: str = "series") -> dict:
    """ADF и KPSS тесты для временного ряда."""
    s = series.dropna()
    adf = adfuller(s)
    kpss_res = kpss(s, regression="c", nlags="auto")

    print(f"\n── ADF тест [{name}] ─────────────────────")
    print(f"  Statistic : {adf[0]:.4f}  |  p-value: {adf[1]:.6f}")
    print(f"  Вывод     : {'Стационарный ✓' if adf[1] < 0.05 else 'НЕ стационарный ✗'}")

    print(f"\n── KPSS тест [{name}] ────────────────────")
    print(f"  Statistic : {kpss_res[0]:.4f}  |  p-value: {kpss_res[1]:.4f}")
    print(f"  Вывод     : {'Стационарный ✓' if kpss_res[1] > 0.05 else 'НЕ стационарный ✗'}")

    return {"adf": adf, "kpss": kpss_res}


def plot_decomposition(series: pd.Series,
                       period: int = 7,
                       save: bool = False):
    """Декомпозиция временного ряда — кастомный стиль."""
    from statsmodels.tsa.seasonal import STL
    stl = STL(series.dropna(), period=period, robust=True)
    result = stl.fit()

    # Недельная агрегация для читаемости тренда и остатка
    trend_w    = result.trend.resample("W").mean()
    seasonal_w = result.seasonal.resample("W").mean()
    resid_w    = result.resid.resample("W").mean()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f"STL-декомпозиция продаж (период={period} дней)",
                 fontsize=13, fontweight="bold", y=1.01)

    # 1. Исходный ряд (недельный для читаемости)
    sales_w = series.resample("W").sum()
    axes[0].plot(sales_w.index, sales_w.values,
                 color=PALETTE[0], linewidth=1)
    axes[0].set_title("Исходный ряд (недельные продажи)")
    axes[0].set_ylabel("Продажи, шт.")

    # 2. Тренд
    axes[1].plot(trend_w.index, trend_w.values,
                 color=PALETTE[4], linewidth=1.5)
    axes[1].set_title("Тренд — общее направление роста/падения")
    axes[1].set_ylabel("Продажи, шт.")

    # 3. Сезонность — показываем только 3 месяца чтобы был виден паттерн
    seasonal_zoom = result.seasonal.iloc[:90]
    axes[2].fill_between(seasonal_zoom.index, seasonal_zoom.values,
                         color=PALETTE[2], alpha=0.6)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_title("Сезонность — недельный паттерн (первые 90 дней)")
    axes[2].set_ylabel("Отклонение")

    # 4. Остаток (шум)
    axes[3].bar(resid_w.index, resid_w.values,
                color=PALETTE[1], alpha=0.6, width=5)
    axes[3].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[3].set_title("Остаток — необъяснённый шум (праздники, аномалии)")
    axes[3].set_ylabel("Отклонение")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig("eda_decomposition.png", dpi=120, bbox_inches="tight")
    plt.show()

    # Возвращаем result для дашборда
    return result


def plot_acf_pacf(series: pd.Series, lags: int = 40, save: bool = False) -> None:
    """ACF и PACF для ряда продаж."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(),  lags=lags, ax=axes[0], title="ACF продаж")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title="PACF продаж")
    plt.tight_layout()
    if save:
        plt.savefig("eda_acf_pacf.png", dpi=120)
    plt.show()


def run_full_eda(data: dict, ts_agg: pd.DataFrame, save: bool = False) -> None:
    """Запускает полный EDA одной командой."""
    print("=" * 55)
    print("EDA — WALMART M5")
    print("=" * 55)

    ts = data["ts"]

    print(f"\nОбщая статистика продаж:")
    print(ts["sales"].describe().round(2))

    plot_hierarchy(data, save)
    plot_aggregate_sales(ts_agg, save)
    plot_seasonality(ts_agg, save)
    plot_event_impact(ts, save)
    plot_zero_sales(ts, save)
    stationarity_tests(ts_agg["sales"], name="агрегированные продажи")
    decomp = plot_decomposition(ts_agg["sales"], period=7, save=save)
    plot_acf_pacf(ts_agg["sales"], save=save)
    return decomp