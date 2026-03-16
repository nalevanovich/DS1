"""
models/xgb_model.py
────────────────────
XGBoost Regressor для продаж Walmart M5.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]


def fit_predict(X_train, X_test, y_train, y_test,
                feature_cols: list,
                n_estimators: int = 500,
                learning_rate: float = 0.05,
                max_depth: int = 6,
                plot_importance: bool = True,
                top_n: int = 20,
                save: bool = False):
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    pred = np.clip(model.predict(X_test), 0, None)

    if plot_importance:
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        top = imp.nlargest(top_n).sort_values()
        fig, ax = plt.subplots(figsize=(8, 7))
        top.plot(kind="barh", ax=ax, color=PALETTE[3])
        ax.set_title(f"XGBoost — топ-{top_n} признаков")
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        if save:
            plt.savefig("xgb_feature_importance.png", dpi=120)
        plt.show()

    print("✓ XGBoost — прогноз готов")
    return pred, model
