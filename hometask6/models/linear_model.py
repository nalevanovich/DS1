"""
models/linear_model.py
───────────────────────
Linear Regression baseline для продаж M5.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def fit_predict(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_tr, y_train)
    pred = np.clip(model.predict(X_te), 0, None)
    print("✓ LinearRegression — прогноз готов")
    return pred, model
