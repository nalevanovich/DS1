# models_stacking.py

import numpy as np
from sklearn.linear_model import Ridge
from config import META_MODEL_ALPHA

SKIP_MODELS = {"Dummy"}

def build_stacking_meta(oof_dict: dict, y_train, test_preds_dict: dict):
    """
    oof_dict        — {model_name: oof_predictions} на train
    test_preds_dict — {model_name: predictions} на test
    Мета-модель: Ridge регрессия поверх OOF
    """
    # Убираем Dummy
    oof_filtered  = {k: v for k, v in oof_dict.items()  if k not in SKIP_MODELS}
    test_filtered = {k: v for k, v in test_preds_dict.items() if k not in SKIP_MODELS}

    X_meta_train = np.column_stack(list(oof_filtered.values()))
    X_meta_test  = np.column_stack(list(test_filtered.values()))

    meta_model = Ridge(alpha=META_MODEL_ALPHA)
    meta_model.fit(X_meta_train, y_train)

    y_pred_stack = meta_model.predict(X_meta_test)
    return y_pred_stack, meta_model