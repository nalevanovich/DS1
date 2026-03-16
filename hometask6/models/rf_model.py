"""
models/rf_model.py
───────────────────
Random Forest Regressor для продаж Walmart M5.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


def fit_predict(X_train, X_test, y_train,
                feature_cols: list,
                n_estimators: int = 300,
                max_depth: int = 12,
                plot_importance: bool = True,
                top_n: int = 20,
                save: bool = False):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, None)

    if plot_importance:
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top = imp.nlargest(top_n).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7))
        top.plot(kind="barh", ax=ax, color=PALETTE[0])
        ax.set_title(f"Random Forest — топ-{top_n} признаков")
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        if save:
            plt.savefig("rf_feature_importance.png", dpi=120)
        plt.show()

    print("✓ RandomForest — прогноз готов")
    return pred, model
