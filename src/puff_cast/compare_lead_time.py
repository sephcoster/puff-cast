"""
Apples-to-apples comparison at exact lead times.

For each valid hour in the test period:
1. HRRR 12h-ahead forecast (initialized 12h before valid time)
2. GFS 12h-ahead forecast (initialized 12h before valid time)
3. ECMWF (best available, ~6-12h lead from Open-Meteo)
4. Our RF model (station observations from 12h before)
5. Persistence (wind at T-12)
6. Climatology

All compared against TPLM2 actuals at the valid time.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KT = 1.944  # m/s to knots


def load_all_data():
    """Load and merge all data sources."""
    # TPLM2 observations + multi-station features
    unified = pd.read_parquet(DATA_DIR / "processed" / "unified_hourly.parquet")

    # Lead-time specific model forecasts
    hrrr = pd.read_csv(DATA_DIR / "raw" / "hrrr_lead_time.csv", parse_dates=["init_time", "valid_time"])
    gfs = pd.read_csv(DATA_DIR / "raw" / "gfs_lead_time.csv", parse_dates=["init_time", "valid_time"])

    # ECMWF from Open-Meteo
    ecmwf_path = DATA_DIR / "raw" / "ecmwf_openmeteo.csv"
    ecmwf = pd.read_csv(ecmwf_path, index_col="time", parse_dates=True) if ecmwf_path.exists() else None

    # Open-Meteo best match (for reference)
    om_path = DATA_DIR / "raw" / "model_forecasts.csv"
    openmeteo = pd.read_csv(om_path, index_col="time", parse_dates=True) if om_path.exists() else None

    return unified, hrrr, gfs, ecmwf, openmeteo


def pivot_model_forecasts(model_df: pd.DataFrame, model_name: str) -> dict[int, pd.Series]:
    """
    Pivot model forecasts into a dict of {lead_hours: Series indexed by valid_time}.
    When multiple init times produce a forecast for the same valid time at the same lead,
    keep the most recent initialization (closest to real-time usage).
    """
    result = {}
    for lead in model_df["lead_hours"].unique():
        subset = model_df[model_df["lead_hours"] == lead].copy()
        subset = subset.sort_values("init_time")
        # Keep last (most recent init) for each valid_time
        subset = subset.drop_duplicates(subset="valid_time", keep="last")
        series = subset.set_index("valid_time")["wspd_ms"]
        series.name = f"{model_name}_{lead}h"
        result[int(lead)] = series
    return result


def build_rf_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from station observations (same as compare.py)."""
    cols = {}
    cols["TPLM2_WSPD_now"] = df["TPLM2_WSPD"]
    cols["TPLM2_GST_now"] = df["TPLM2_GST"]
    cols["TPLM2_PRES_now"] = df["TPLM2_PRES"]
    cols["TPLM2_ATMP_now"] = df["TPLM2_ATMP"]
    cols["TPLM2_WDIR_sin"] = df.get("TPLM2_WDIR_sin")
    cols["TPLM2_WDIR_cos"] = df.get("TPLM2_WDIR_cos")
    for c in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        cols[c] = df[c]
    for lag in [1, 2, 3, 6, 12, 24]:
        cols[f"TPLM2_WSPD_lag{lag}"] = df["TPLM2_WSPD"].shift(lag)
        cols[f"TPLM2_GST_lag{lag}"] = df["TPLM2_GST"].shift(lag)
        cols[f"TPLM2_PRES_lag{lag}"] = df["TPLM2_PRES"].shift(lag)
    for h in [1, 3, 6, 12]:
        cols[f"TPLM2_PRES_diff{h}"] = df["TPLM2_PRES"].diff(h)
        cols[f"TPLM2_WSPD_diff{h}"] = df["TPLM2_WSPD"].diff(h)
    for station in ["APAM2", "COVM2", "CAMM2", "SLIM2", "WASD2", "44009", "BLTM2"]:
        for var in ["WSPD", "PRES"]:
            col = f"{station}_{var}"
            if col in df.columns:
                cols[f"{station}_{var}_now"] = df[col]
                for lag in [1, 3, 6, 12]:
                    cols[f"{station}_{var}_lag{lag}"] = df[col].shift(lag)
    if "44009_PRES" in df.columns:
        cols["PRES_gradient_ocean"] = df["TPLM2_PRES"] - df["44009_PRES"]
    if "WASD2_PRES" in df.columns:
        cols["PRES_gradient_west"] = df["TPLM2_PRES"] - df["WASD2_PRES"]
    return pd.concat(cols, axis=1)


def build_mos_features(df: pd.DataFrame, ecmwf: pd.DataFrame | None, openmeteo: pd.DataFrame | None) -> pd.DataFrame:
    """Build MOS feature matrix: station obs + NWS model output available at forecast time."""
    station_features = build_rf_features(df)

    mos_cols = {}

    # Add Open-Meteo model values at current time (available at forecast time T)
    if openmeteo is not None:
        for col in openmeteo.columns:
            if col in df.columns:
                continue
            mos_cols[f"OM_{col}"] = openmeteo[col].reindex(df.index)

    # Add ECMWF values at current time
    if ecmwf is not None:
        for col in ecmwf.columns:
            mos_cols[f"ECMWF_{col}"] = ecmwf[col].reindex(df.index)

    if mos_cols:
        extra = pd.concat(mos_cols, axis=1).reindex(df.index)
        return pd.concat([station_features, extra], axis=1)
    return station_features


def run_comparison(
    test_start: str = "2025-10-01",
    horizons: list[int] = [3, 6, 12],
):
    """Run the full apples-to-apples comparison."""
    unified, hrrr_df, gfs_df, ecmwf, openmeteo = load_all_data()

    # Pivot model forecasts by lead time
    hrrr_by_lead = pivot_model_forecasts(hrrr_df, "hrrr")
    gfs_by_lead = pivot_model_forecasts(gfs_df, "gfs")

    # Build features
    train_mask = unified.index < test_start
    test_mask = unified.index >= test_start

    station_features = build_rf_features(unified)
    mos_features = build_mos_features(unified, ecmwf, openmeteo)

    # Climatology
    clim_lookup = unified.loc[train_mask].groupby(
        [unified.loc[train_mask].index.month, unified.loc[train_mask].index.hour]
    )["TPLM2_WSPD"].mean()

    print("=" * 75)
    print("APPLES-TO-APPLES COMPARISON AT EXACT LEAD TIMES")
    print("=" * 75)
    print(f"Test period: {test_start} onward")
    print()

    all_results = []

    for horizon in horizons:
        target = unified["TPLM2_WSPD"].shift(-horizon)

        # Find hours where we have TPLM2 actuals AND model forecasts
        hrrr_series = hrrr_by_lead.get(horizon)
        gfs_series = gfs_by_lead.get(horizon)

        # Build valid mask: actuals + key station features
        key_valid = station_features["TPLM2_WSPD_now"].notna() & target.notna() & test_mask

        # Train RF (station-only)
        train_valid = station_features["TPLM2_WSPD_now"].notna() & target.notna() & train_mask
        X_train = station_features[train_valid].fillna(-999)
        y_train = target[train_valid]

        rf = RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=8, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Train MOS model (station + NWS model features)
        mos_train_valid = mos_features.notna().all(axis=1) & target.notna() & train_mask
        if mos_train_valid.sum() > 500:
            X_mos_train = mos_features[mos_train_valid].fillna(-999)
            y_mos_train = target[mos_train_valid]
            mos_rf = RandomForestRegressor(
                n_estimators=300, max_depth=18, min_samples_leaf=8, random_state=42, n_jobs=-1
            )
            mos_rf.fit(X_mos_train, y_mos_train)
            has_mos = True
        else:
            has_mos = False

        # Evaluate on hours where ALL models have predictions
        valid_hours = unified.index[key_valid]
        if hrrr_series is not None:
            valid_hours = valid_hours.intersection(hrrr_series.index)
        if gfs_series is not None:
            valid_hours = valid_hours.intersection(gfs_series.index)

        actual = target.reindex(valid_hours).dropna()
        valid_hours = actual.index

        if len(valid_hours) < 50:
            print(f"\n--- {horizon}h Horizon: INSUFFICIENT DATA ({len(valid_hours)} hours) ---")
            continue

        print(f"\n{'─' * 75}")
        print(f"  {horizon}-HOUR AHEAD FORECAST  ({len(valid_hours):,} test hours)")
        print(f"{'─' * 75}")

        # Collect predictions
        models = {}

        # Persistence
        models["Persistence"] = unified["TPLM2_WSPD"].reindex(valid_hours)

        # Climatology
        models["Climatology"] = pd.Series(
            [clim_lookup.get((t.month, t.hour), actual.mean()) for t in valid_hours],
            index=valid_hours,
        )

        # HRRR at exact lead time
        if hrrr_series is not None:
            models[f"HRRR (exact {horizon}h lead)"] = hrrr_series.reindex(valid_hours)

        # GFS at exact lead time
        if gfs_series is not None:
            models[f"GFS (exact {horizon}h lead)"] = gfs_series.reindex(valid_hours)

        # ECMWF (best available ~6h lead)
        if ecmwf is not None:
            ecmwf_shifted = ecmwf.iloc[:, 0].shift(-horizon)  # wind_speed_10m at target time
            models["ECMWF IFS (~6h lead)"] = ecmwf_shifted.reindex(valid_hours)

        # Our RF (station observations only)
        X_test = station_features.reindex(valid_hours).fillna(-999)
        models[f"Our RF (stations, {horizon}h)"] = pd.Series(rf.predict(X_test), index=valid_hours)

        # MOS (station + model features)
        if has_mos:
            X_mos_test = mos_features.reindex(valid_hours).fillna(-999)
            models[f"MOS RF (stations+models, {horizon}h)"] = pd.Series(
                mos_rf.predict(X_mos_test), index=valid_hours
            )

        # Compute metrics
        print(f"\n  {'Model':<40s} {'MAE (kt)':>9s} {'RMSE (kt)':>10s} {'Bias (kt)':>10s}")
        print(f"  {'─' * 72}")

        best_mae = float("inf")
        best_name = ""
        for name, pred in models.items():
            valid = actual.notna() & pred.notna()
            if valid.sum() < 50:
                continue
            a = actual[valid]
            p = pred[valid]
            mae = mean_absolute_error(a, p) * KT
            rmse = np.sqrt(mean_squared_error(a, p)) * KT
            bias = (p.mean() - a.mean()) * KT

            marker = ""
            if mae < best_mae:
                best_mae = mae
                best_name = name
            all_results.append({
                "horizon": horizon, "model": name,
                "mae_kt": round(mae, 2), "rmse_kt": round(rmse, 2),
                "bias_kt": round(bias, 2), "n_hours": int(valid.sum()),
            })
            print(f"  {name:<40s} {mae:7.2f}    {rmse:8.2f}    {bias:+8.2f}")

        print(f"\n  >>> WINNER at {horizon}h: {best_name} ({best_mae:.2f} kt MAE)")

        # Feature importance for MOS model
        if has_mos:
            imp = pd.Series(mos_rf.feature_importances_, index=X_mos_test.columns)
            imp = imp.sort_values(ascending=False)
            print(f"\n  MOS top 10 features:")
            for feat, v in imp.head(10).items():
                print(f"    {feat:40s} {v:.4f}")

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_comparison()
    print("\n\n")
    print(results.to_string(index=False))
