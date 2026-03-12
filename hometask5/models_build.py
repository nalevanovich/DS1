# models_build.py

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from config import LGBM_PARAMS, XGB_PARAMS, CB_PARAMS, RF_PARAMS


def build_models():
    return {
        "Dummy":          DummyRegressor(strategy="mean"),
        "RandomForest":   RandomForestRegressor(**RF_PARAMS),
        "LightGBM":       lgb.LGBMRegressor(**LGBM_PARAMS),
        "XGBoost":        xgb.XGBRegressor(**XGB_PARAMS),
        "CatBoost":       cb.CatBoostRegressor(**CB_PARAMS),
    }