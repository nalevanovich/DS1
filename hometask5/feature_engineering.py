# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2*R*np.arcsin(np.sqrt(a))


def extract_coords_features(row):
    coords = row["coords"]
    n = len(coords)
    if n < 2:
        return pd.Series({
            "start_lon": np.nan, "start_lat": np.nan,
            "end_lon":   np.nan, "end_lat":   np.nan,
            "direct_dist_km": np.nan,
            "total_dist_km":  np.nan,
            "straightness":   np.nan,
            "mean_lon": np.nan, "mean_lat": np.nan,
        })
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    # direct distance
    direct = haversine(lons[0], lats[0], lons[-1], lats[-1])

    # cumulative distance along the route
    seg_dists = [
        haversine(lons[i], lats[i], lons[i+1], lats[i+1])
        for i in range(n-1)
    ]
    total = sum(seg_dists)

    straightness = direct / total if total > 0 else 1.0

    return pd.Series({
        "start_lon":      lons[0],
        "start_lat":      lats[0],
        "end_lon":        lons[-1],
        "end_lat":        lats[-1],
        "direct_dist_km": direct,
        "total_dist_km":  total,
        "straightness":   straightness,
        "mean_lon":       np.mean(lons),
        "mean_lat":       np.mean(lats),
    })


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Engineering features …")
    df = df.copy()

    # ── Temporal features ────────────────────────────────────────────────────
    ts = pd.to_datetime(df["TIMESTAMP"], unit="s", utc=True)
    df["hour"]        = ts.dt.hour
    df["dow"]         = ts.dt.dayofweek          # 0=Mon … 6=Sun
    df["month"]       = ts.dt.month
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["is_rush"]     = df["hour"].isin([7,8,9,17,18,19]).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["dow"]  / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["dow"]  / 7)

    # ── CALL_TYPE → numeric ──────────────────────────────────────────────────
    df["call_type_enc"] = LabelEncoder().fit_transform(
        df["CALL_TYPE"].fillna("UNKNOWN")
    )

    # ── DAY_TYPE → numeric ───────────────────────────────────────────────────
    df["day_type_enc"] = LabelEncoder().fit_transform(
        df["DAY_TYPE"].fillna("A")
    )

    # ── TAXI_ID frequency encoding ───────────────────────────────────────────
    freq = df["TAXI_ID"].value_counts() / len(df)
    df["taxi_freq"] = df["TAXI_ID"].map(freq).fillna(0)

    # ── Coordinate / route features ──────────────────────────────────────────
    coord_feats = df.apply(extract_coords_features, axis=1)
    df = pd.concat([df, coord_feats], axis=1)

    # ── Log-distance (robust to 0) ───────────────────────────────────────────
    df["log_direct_dist"] = np.log1p(df["direct_dist_km"].clip(0))
    df["log_total_dist"]  = np.log1p(df["total_dist_km"].clip(0))

    # ── Speed proxy (dist per point = dist per 15s) ──────────────────────────
    df["speed_proxy"] = df["total_dist_km"] / (df["n_points"] * 15 / 3600 + 1e-9)

    # ── Bounding box ─────────────────────────────────────────────────────────
    df["bbox_area"] = (
        (df["end_lon"] - df["start_lon"]).abs() *
        (df["end_lat"] - df["start_lat"]).abs()
    )

    print(f"      Feature count: {df.shape[1]}")
    return df