"""
Head-to-head comparison: Our RF model vs NWS model chain vs baselines.

Compares forecast accuracy at 3h, 6h, and 12h horizons using:
1. Persistence (current wind = future wind)
2. Climatology (average for that hour+month)
3. NWS "best match" model (Open-Meteo's blend of best available models)
4. GFS 0.13° (raw GFS, the backbone of NWS marine forecasts)
5. Our multi-station RF model

All evaluated on the SAME test period with the SAME actuals from TPLM2.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def load_comparison_dataset() -> pd.DataFrame:
    """Load and merge TPLM2 actuals, model forecasts, and multi-station features."""
    # Load unified multi-station data
    unified = pd.read_parquet(DATA_DIR / "processed" / "unified_hourly.parquet")

    # Load model forecasts
    forecasts = pd.read_csv(DATA_DIR / "raw" / "model_forecasts.csv", index_col="time", parse_dates=True)

    # Merge on time index (inner join — only keep hours where we have both)
    merged = unified.join(forecasts, how="inner")

    logger.info(f"Merged dataset: {len(merged)} hours, {merged.shape[1]} columns")
    return merged


def build_rf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for our multi-station RF model."""
    features = pd.DataFrame(index=df.index)

    # Current TPLM2 conditions
    features["TPLM2_WSPD_now"] = df["TPLM2_WSPD"]
    features["TPLM2_GST_now"] = df["TPLM2_GST"]
    features["TPLM2_PRES_now"] = df["TPLM2_PRES"]
    features["TPLM2_ATMP_now"] = df["TPLM2_ATMP"]
    features["TPLM2_WDIR_sin"] = df.get("TPLM2_WDIR_sin")
    features["TPLM2_WDIR_cos"] = df.get("TPLM2_WDIR_cos")

    # Time features
    for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        features[col] = df[col]

    # Lagged TPLM2
    for lag in [1, 2, 3, 6, 12, 24]:
        features[f"TPLM2_WSPD_lag{lag}"] = df["TPLM2_WSPD"].shift(lag)
        features[f"TPLM2_GST_lag{lag}"] = df["TPLM2_GST"].shift(lag)
        features[f"TPLM2_PRES_lag{lag}"] = df["TPLM2_PRES"].shift(lag)

    # Pressure tendency
    for hours in [1, 3, 6, 12]:
        features[f"TPLM2_PRES_diff{hours}"] = df["TPLM2_PRES"].diff(hours)

    # Wind acceleration (is wind picking up or dying?)
    for hours in [1, 3, 6]:
        features[f"TPLM2_WSPD_diff{hours}"] = df["TPLM2_WSPD"].diff(hours)

    # Surrounding stations — current + lagged wind and pressure
    key_stations = ["APAM2", "COVM2", "CAMM2", "SLIM2", "WASD2", "44009", "BLTM2"]
    for station in key_stations:
        wspd_col = f"{station}_WSPD"
        pres_col = f"{station}_PRES"
        if wspd_col in df.columns:
            features[f"{station}_WSPD_now"] = df[wspd_col]
            for lag in [1, 3, 6, 12]:
                features[f"{station}_WSPD_lag{lag}"] = df[wspd_col].shift(lag)
        if pres_col in df.columns:
            features[f"{station}_PRES_now"] = df[pres_col]
            for hours in [3, 6, 12]:
                features[f"{station}_PRES_diff{hours}"] = df[pres_col].diff(hours)

    # Spatial pressure gradient (difference between stations)
    if "44009_PRES" in df.columns:
        features["PRES_gradient_ocean"] = df["TPLM2_PRES"] - df["44009_PRES"]
    if "WASD2_PRES" in df.columns:
        features["PRES_gradient_west"] = df["TPLM2_PRES"] - df["WASD2_PRES"]
    if "SLIM2_PRES" in df.columns:
        features["PRES_gradient_south"] = df["TPLM2_PRES"] - df["SLIM2_PRES"]

    return features


def evaluate_all(test_start: str = "2025-06-01", horizons: list[int] = [3, 6, 12]):
    """
    Run full comparison across all models and horizons.

    Train on data before test_start, test on data from test_start onward.
    """
    df = load_comparison_dataset()

    # Build RF features once
    rf_features = build_rf_features(df)

    # Climatology lookup: mean wind for each (month, hour) pair
    train_mask = df.index < test_start
    clim_lookup = df.loc[train_mask].groupby(
        [df.loc[train_mask].index.month, df.loc[train_mask].index.hour]
    )["TPLM2_WSPD"].mean()

    results = []

    for horizon in horizons:
        print(f"\n{'='*60}")
        print(f"  {horizon}-HOUR FORECAST HORIZON")
        print(f"{'='*60}")

        # Target: actual TPLM2 wind speed N hours in the future
        actual_future = df["TPLM2_WSPD"].shift(-horizon)

        # --- Test set: where we have actuals AND model forecasts ---
        test_mask = (df.index >= test_start) & actual_future.notna()

        # 1. Persistence: current wind persists
        persist_pred = df.loc[test_mask, "TPLM2_WSPD"]
        actual = actual_future[test_mask]

        valid = actual.notna() & persist_pred.notna()
        persist_mae = mean_absolute_error(actual[valid], persist_pred[valid])
        persist_rmse = np.sqrt(mean_squared_error(actual[valid], persist_pred[valid]))

        # 2. Climatology
        clim_pred = pd.Series(
            [clim_lookup.get((t.month, t.hour), actual.mean()) for t in actual.index],
            index=actual.index,
        )
        clim_mae = mean_absolute_error(actual[valid], clim_pred[valid])

        # 3. NWS Best Match model
        # The model forecast at time T is the model's prediction for time T
        # made from the most recent model run. To compare at N-hour lead time,
        # we compare the model's forecast for time T+N with the actual at T+N.
        # Since Open-Meteo returns ~0-6h lead time forecasts, for horizons > 6h
        # this is optimistic for the NWS model (it's effectively a shorter lead time).
        nws_best_col = "best_match_wind_speed_10m"
        nws_gfs_col = "ncep_gfs013_wind_speed_10m"

        # For fair comparison: the NWS forecast at time T+N is compared to actual at T+N
        # We shift by -horizon to align "what was forecast for time T+N" with "actual at T+N"
        # But actually, the historical forecast API already gives the forecast FOR each hour
        # So: model forecast for hour X vs actual observation at hour X
        # The lead time is baked into the model data (roughly 1-6h for best_match)

        # For a TRUE N-hour comparison, we'd need the forecast made at time T for time T+N.
        # Since we can't get exact lead times from Open-Meteo, we compare:
        # - NWS model: forecast value at the target time (best available ~1-6h lead)
        # - Our model: prediction from time T for time T+N (exact N-hour lead)
        # This gives NWS an advantage at longer horizons — making our comparison conservative.

        # NWS "best match" accuracy on the test set
        nws_best_future = df[nws_best_col].shift(-horizon) if horizon <= 6 else df[nws_best_col]
        # For <=6h, shift to match the same future target
        # For >6h, use the model value at target time (shorter effective lead — gives NWS advantage)

        # Actually simpler: compare model forecast for time X with actual at time X
        # This tests "how good is the model at predicting what actually happens?"
        nws_actual_pairs = test_mask & df[nws_best_col].notna() & df["TPLM2_WSPD"].notna()

        # NWS model value at time T+horizon vs actual at time T+horizon
        nws_best_at_target = df[nws_best_col].shift(-horizon)
        actual_at_target = actual_future  # = TPLM2_WSPD shifted by -horizon

        valid_nws = test_mask & nws_best_at_target.notna() & actual_at_target.notna()

        nws_best_mae = mean_absolute_error(
            actual_at_target[valid_nws], nws_best_at_target[valid_nws]
        )

        # GFS raw
        nws_gfs_at_target = df[nws_gfs_col].shift(-horizon)
        valid_gfs = test_mask & nws_gfs_at_target.notna() & actual_at_target.notna()
        nws_gfs_mae = mean_absolute_error(
            actual_at_target[valid_gfs], nws_gfs_at_target[valid_gfs]
        )

        # 4. Our RF model
        # Require key features to be present
        key_feat = ["TPLM2_WSPD_now", "TPLM2_PRES_now"]
        feat_valid = rf_features[key_feat].notna().all(axis=1)
        train_rf = train_mask & feat_valid & actual_future.notna()
        test_rf = test_mask & feat_valid & actual_future.notna()

        X_train = rf_features[train_rf].fillna(-999)
        y_train = actual_future[train_rf]
        X_test = rf_features[test_rf].fillna(-999)
        y_test = actual_future[test_rf]

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

        # Convert all to knots for reporting
        kt = 1.944

        print(f"\n  Test period: {test_start} onward ({valid.sum():,} hours)")
        print(f"\n  {'Model':<25s} {'MAE (kt)':>10s} {'vs Persist':>12s} {'vs NWS Best':>12s}")
        print(f"  {'-'*60}")

        models_results = [
            ("Climatology", clim_mae),
            ("Persistence", persist_mae),
            ("NWS Best Match", nws_best_mae),
            ("GFS 0.13°", nws_gfs_mae),
            ("Our RF (multi-station)", rf_mae),
        ]

        for name, mae in models_results:
            vs_persist = (persist_mae - mae) / persist_mae * 100
            vs_nws = (nws_best_mae - mae) / nws_best_mae * 100
            print(
                f"  {name:<25s} {mae*kt:8.2f}   {vs_persist:+10.1f}%   {vs_nws:+10.1f}%"
            )
            results.append({
                "horizon": horizon,
                "model": name,
                "mae_kt": mae * kt,
                "mae_ms": mae,
                "vs_persist_pct": vs_persist,
                "vs_nws_best_pct": vs_nws,
            })

        # Feature importance for this horizon
        importance = pd.Series(rf.feature_importances_, index=X_test.columns).sort_values(
            ascending=False
        )
        print(f"\n  Top 10 RF features ({horizon}h horizon):")
        for feat, imp in importance.head(10).items():
            print(f"    {feat:35s}: {imp:.4f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = evaluate_all(test_start="2025-06-01")
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results.to_string(index=False))
