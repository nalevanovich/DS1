from pathlib import Path

# ── Данные ───────────────────────────────────────────────────────────────────
DATA_PATH   = Path("train.csv")
SAMPLE_FRAC = 0.5
TARGET      = "trip_time"

# ── Валидация ─────────────────────────────────────────────────────────────────
N_CV_FOLDS  = 3
TEST_SIZE   = 0.2
RANDOM_STATE = 42

# ── Признаки ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "hour", "dow", "month", "is_weekend", "is_rush",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "call_type_enc", "day_type_enc", "taxi_freq",
    "start_lon", "start_lat", "end_lon", "end_lat",
    "direct_dist_km", "total_dist_km", "straightness",
    "mean_lon", "mean_lat",
    "log_direct_dist", "log_total_dist", "bbox_area",
]

# ── Модели ────────────────────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":   300,
    "max_depth":      20,
    "min_samples_leaf": 10,
    "max_features":   0.5,
    "n_jobs":         -1,
    "random_state":   RANDOM_STATE,
}

LGBM_PARAMS = {
    "n_estimators":     1000,
    "learning_rate":    0.03,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
    "verbose":          -1,
}

XGB_PARAMS = {
    "n_estimators":       1000,
    "learning_rate":      0.03,
    "max_depth":          8,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   5,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "n_jobs":             -1,
    "random_state":       RANDOM_STATE,
    "verbosity":          0,
    "eval_metric":        "rmse",
}

CB_PARAMS = {
    "iterations":          1000,
    "learning_rate":       0.03,
    "depth":               8,
    "l2_leaf_reg":         3,
    "random_seed":         RANDOM_STATE,
    "verbose":             0,
}

# ── Стекинг ───────────────────────────────────────────────────────────────────
META_MODEL_ALPHA = 1.0   # Ridge alpha