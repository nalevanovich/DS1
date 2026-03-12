# data_preprocessor.py

import json
import pandas as pd
import numpy as np
from config import TARGET, FEATURE_COLS


def parse_polyline(poly_str: str):
    """Return list of [lon, lat] coordinates."""
    try:
        coords = json.loads(poly_str)
        return coords if isinstance(coords, list) else []
    except Exception:
        return []


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trip_time from POLYLINE length.
    Each GPS point is recorded every 15 seconds.
    """
    print("[2/6] Computing target (trip_time) …")
    df = df.copy()
    df["coords"]    = df["POLYLINE"].apply(parse_polyline)
    df["n_points"]  = df["coords"].apply(len)
    df[TARGET]      = (df["n_points"] - 1) * 15   # seconds
    # Remove empty / missing trips
    df = df[df["n_points"] > 1].reset_index(drop=True)
    # Remove extreme outliers (>2h trip)
    q_high = df[TARGET].quantile(0.9995)
    df = df[df[TARGET] <= q_high].reset_index(drop=True)
    print(f"      Rows after filtering: {len(df):,} | "
          f"mean trip_time = {df[TARGET].mean():.0f}s")
    return df


def prepare_xy(df: pd.DataFrame):
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    X = df[FEATURE_COLS].astype(float)
    y = np.log1p(df[TARGET].astype(float))   # log-transform target
    return X, y