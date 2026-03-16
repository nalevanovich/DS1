"""
visualization.py
─────────────────
Графики результатов моделирования Walmart M5:
  - прогнозы vs факт
  - ошибки (residuals)
  - тепловая карта метрик
  - столбчатые диаграммы метрик
  - важность признаков (RF, XGBoost)
  - итоговый дашборд
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2", "#E238A7"]


def plot_forecasts(predictions: dict,
                   y_test: np.ndarray,
                   dates_test,
                   title: str = "Прогнозы продаж vs Факт",
                   save: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_test, y_test, label="Факт", color="black",
            linewidth=2, zorder=5)

    for i, (name, val) in enumerate(predictions.items()):
        if isinstance(val, tuple):
            pred_vals, pred_dates = val[0], val[1]
        else:
            pred_vals, pred_dates = val, dates_test
        ax.plot(pred_dates, pred_vals, label=name,
                linewidth=1.4, color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel("Продажи, шт.")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save:
        plt.savefig("all_forecasts.png", dpi=120)
    plt.show()


def plot_residuals(predictions: dict,
                   y_test: np.ndarray,
                   save: bool = False) -> None:
    n = len(predictions)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten()

    for i, (name, val) in enumerate(predictions.items()):
        y_true_i = val[2] if isinstance(val, tuple) else y_test
        y_pred_i = val[0] if isinstance(val, tuple) else val
        errors = y_true_i - y_pred_i
        axes[i].plot(errors, color=PALETTE[i % len(PALETTE)],
                     linewidth=0.9, alpha=0.8)
        axes[i].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[i].set_title(f"{name} — Residuals")
        axes[i].set_ylabel("Error (шт.)")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Ошибки прогнозирования (Residuals)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("forecast_errors.png", dpi=120)
    plt.show()


def plot_metrics_heatmap(metrics_df: pd.DataFrame, save: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    data = metrics_df.drop(columns="R2", errors="ignore").astype(float)
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Метрики качества моделей (ниже = лучше)")
    plt.tight_layout()
    if save:
        plt.savefig("metrics_heatmap.png", dpi=120)
    plt.show()


def plot_metrics_bars(metrics_df: pd.DataFrame, save: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE%"]):
        vals = metrics_df[metric].sort_values()
        bars = ax.barh(vals.index, vals.values,
                       color=PALETTE[:len(vals)])
        ax.set_title(metric)
        ax.set_xlabel("Value (ниже = лучше)")
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_width() * 1.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}", va="center", fontsize=9)
    plt.suptitle("Сравнение моделей", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("metrics_bars.png", dpi=120)
    plt.show()


def plot_feature_importance_comparison(rf_model,
                                       xgb_model,
                                       feature_cols: list,
                                       top_n: int = 15,
                                       save: bool = False) -> None:
    rf_imp  = pd.Series(rf_model.feature_importances_,  index=feature_cols)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    combined = ((rf_imp + xgb_imp) / 2).nlargest(top_n).index

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    rf_imp[combined].sort_values().plot(kind="barh", ax=axes[0], color=PALETTE[0])
    axes[0].set_title(f"Random Forest — топ-{top_n}")

    xgb_imp[combined].sort_values().plot(kind="barh", ax=axes[1], color=PALETTE[3])
    axes[1].set_title(f"XGBoost — топ-{top_n}")

    plt.suptitle("Важность признаков: RF vs XGBoost",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("feature_importance_comparison.png", dpi=120)
    plt.show()


def plot_dashboard(predictions: dict,
                   y_test: np.ndarray,
                   dates_test,
                   metrics_df: pd.DataFrame,
                   df_anom: pd.DataFrame,
                   xgb_model,
                   feature_cols: list,
                   save: bool = False) -> None:
    """Итоговый дашборд."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.5, wspace=0.5)

    # Продажи + аномалии (только STL) — занимает 3 колонки
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.plot(df_anom.index, df_anom["sales"],
             color=PALETTE[0], linewidth=0.8)
    anom_idx = df_anom[df_anom["anomaly_stl"]].index
    ax1.scatter(anom_idx, df_anom.loc[anom_idx, "sales"],
                color=PALETTE[1], s=20, zorder=5,
                label=f"Аномалия STL ({len(anom_idx)})")
    ax1.set_title("Суммарные продажи + Аномалии (STL)")
    ax1.legend(fontsize=8)

    # R² — занимает 2 колонки
    ax2 = fig.add_subplot(gs[0, 3:])
    r2 = metrics_df["R2"].sort_values()
    colors = [PALETTE[2] if v > 0.9 else PALETTE[3] if v > 0.7 else PALETTE[1]
              for v in r2.values]
    ax2.barh(r2.index, r2.values, color=colors)
    ax2.set_title("R² по моделям")
    ax2.set_xlim(0, 1.05)
    ax2.axvline(0.9, color="gray", linestyle="--", linewidth=0.8)

    # Недельный паттерн — последние 5 недель с подписями дней
    ax3 = fig.add_subplot(gs[1, :3])
    last_5w = df_anom["sales"].iloc[-35:]  # последние 35 дней (5 недель)
    day_names = {0:"Пн", 1:"Вт", 2:"Ср", 3:"Чт", 4:"Пт", 5:"Сб", 6:"Вс"}
    bar_colors = [PALETTE[2] if d >= 5 else PALETTE[0]
                  for d in last_5w.index.dayofweek]
    ax3.bar(range(len(last_5w)), last_5w.values,
            color=bar_colors, edgecolor="none", alpha=0.85)
    ax3.set_xticks(range(len(last_5w)))
    ax3.set_xticklabels(
        [day_names[d] for d in last_5w.index.dayofweek],
        fontsize=7, rotation=0
    )
    # Вертикальные линии между неделями
    for i in range(7, len(last_5w), 7):
        ax3.axvline(i - 0.5, color="gray", linewidth=0.8, linestyle="--")
    ax3.set_title("Недельный паттерн продаж (последние 5 недель)  |  зелёный = выходные")
    ax3.set_ylabel("Продажи, шт.")

    # Прогноз лучшей модели + MAPE — занимает 2 колонки
    ax4 = fig.add_subplot(gs[1, 3:])
    best_model = metrics_df["MAPE%"].idxmin()
    best_mape  = metrics_df.loc[best_model, "MAPE%"]
    pred_best  = predictions[best_model]
    pred_vals  = pred_best[0] if isinstance(pred_best, tuple) else pred_best
    pred_dates = pred_best[1] if isinstance(pred_best, tuple) else dates_test
    y_true_b   = pred_best[2] if isinstance(pred_best, tuple) else y_test
    ax4.plot(pred_dates, y_true_b,  color="black",    linewidth=1.5, label="Факт")
    ax4.plot(pred_dates, pred_vals, color=PALETTE[0], linewidth=1.2,
             label=f"{best_model} (лучшая)")
    ax4.set_title(f"Лучшая модель vs Факт\nMAPE = {best_mape:.2f}%")
    ax4.legend(fontsize=8)
    ax4.tick_params(axis="x", rotation=30)

    # MAPE — занимает 3 колонки
    ax5 = fig.add_subplot(gs[2, :3])
    mape = metrics_df["MAPE%"].sort_values()
    bar_colors = [PALETTE[1] if m == "Prophet" else PALETTE[0]
                  for m in mape.index]
    bars = ax5.bar(mape.index, mape.values, color=bar_colors)
    ax5.set_title("MAPE% по моделям (ниже = лучше)  |  красный = аутсайдер")
    for bar, v in zip(bars, mape.values):
        ax5.text(bar.get_x() + bar.get_width() / 2,
                 v + 0.1, f"{v:.1f}%", ha="center", fontsize=8)

    # XGBoost top-10 — занимает 2 колонки
    ax6 = fig.add_subplot(gs[2, 3:])
    imp = pd.Series(xgb_model.feature_importances_, index=feature_cols)
    top10 = imp.nlargest(10).sort_values()
    ax6.barh(top10.index, top10.values, color=PALETTE[3])
    ax6.set_title("XGBoost — топ-10 признаков")

    fig.suptitle("Walmart M5 — Итоговый дашборд",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("final_dashboard.png", bbox_inches="tight", dpi=120)
    plt.show()