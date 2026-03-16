"""
models/adaboost_model.py
─────────────────────────
AdaBoost Regressor для продаж Walmart M5.
Использует Decision Tree как базовый estimator.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


def fit_predict(X_train: np.ndarray,
                X_test: np.ndarray,
                y_train: np.ndarray,
                feature_cols: list,
                n_estimators: int = 300,
                learning_rate: float = 0.05,
                max_depth: int = 4,
                plot_importance: bool = True,
                top_n: int = 20,
                save: bool = False):
    """
    Обучает AdaBoostRegressor и возвращает прогноз.

    Parameters
    ----------
    X_train, X_test : матрицы признаков
    y_train         : целевой ряд (продажи)
    feature_cols    : названия признаков
    n_estimators    : число слабых learner-ов
    learning_rate   : вклад каждого дерева (shrinkage)
    max_depth       : глубина базового дерева

    Returns
    -------
    (predictions, fitted_model)
    """
    base = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

    model = AdaBoostRegressor(
        estimator=base,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        loss="linear",          # linear / square / exponential
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)

    if plot_importance:
        # AdaBoost накапливает feature_importances_ из базовых деревьев
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top = imp.nlargest(top_n).sort_values()

        fig, ax = plt.subplots(figsize=(8, 7))
        top.plot(kind="barh", ax=ax, color=PALETTE[5])
        ax.set_title(f"AdaBoost — топ-{top_n} важных признаков")
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        if save:
            plt.savefig("adaboost_feature_importance.png", dpi=120)
        plt.show()

    # Кривая ошибки по числу estimators
    staged_errors = list(model.staged_predict(X_test))
    staged_mae = [
        np.mean(np.abs(y_train[:len(p)] - p))   # приближение на train
        for p in model.staged_predict(X_train)
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(staged_mae, color=PALETTE[5], linewidth=1)
    ax.set_title("AdaBoost — MAE по числу деревьев (train)")
    ax.set_xlabel("Число деревьев")
    ax.set_ylabel("MAE")
    plt.tight_layout()
    if save:
        plt.savefig("adaboost_staged_error.png", dpi=120)
    plt.show()

    print("✓ AdaBoost — прогноз готов")
    return pred, model
