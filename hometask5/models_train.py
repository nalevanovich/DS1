# models_train.py

import numpy as np
from sklearn.model_selection import KFold
from config import N_CV_FOLDS, RANDOM_STATE


def cross_validate_model(name, model, X, y):
    kf  = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"    fold {fold}/{N_CV_FOLDS}", end="\r")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        oof[val_idx] = model.predict(X_val)

    return oof