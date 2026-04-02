"""
HRRR MOS (Model Output Statistics) correction model.

Learns systematic HRRR errors at Bay stations and corrects them using
local observations. The model predicts HRRR error (actual - HRRR forecast),
which is a much easier problem than forecasting wind from scratch.

Key insight: if HRRR consistently underforecasts when pressure is falling
and wind at WASD2 is high, the MOS model learns that correction.

Designed to be extensible to any Bay location where we have observations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from puff_cast.stations import ALL_STATIONS, STATION_BY_ID

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KT = 1.944


def load_regional_data():
    """Load HRRR regional forecasts, station observations, and merge."""
    # HRRR forecasts at all stations
    hrrr = pd.read_csv(
        DATA_DIR / "raw" / "hrrr_regional.csv",
        parse_dates=["init_time", "valid_time"],
    )

    # Station observations
    unified = pd.read_parquet(DATA_DIR / "processed" / "unified_hourly.parquet")

    return hrrr, unified


def compute_hrrr_errors(hrrr: pd.DataFrame, unified: pd.DataFrame) -> pd.DataFrame:
    """
    For each HRRR forecast, compute the error vs actual observation.

    Returns DataFrame with HRRR forecast, actual, and error for each station/time.
    """
    records = []

    for station in ALL_STATIONS:
        sid = station.id
        obs_col = f"{sid}_WSPD"
        if obs_col not in unified.columns:
            continue

        station_hrrr = hrrr[hrrr["station_id"] == sid].copy()
        if len(station_hrrr) == 0:
            continue

        for _, row in station_hrrr.iterrows():
            vt = row["valid_time"]
            if vt not in unified.index:
                continue

            actual = unified.loc[vt, obs_col]
            if pd.isna(actual):
                continue

            records.append(
                {
                    "station_id": sid,
                    "init_time": row["init_time"],
                    "valid_time": vt,
                    "lead_hours": row["lead_hours"],
                    "hrrr_wspd": row["hrrr_wspd_ms"],
                    "hrrr_wdir": row["hrrr_wdir"],
                    "actual_wspd": actual,
                    "error": actual - row["hrrr_wspd_ms"],  # positive = HRRR underforecast
                    "abs_error": abs(actual - row["hrrr_wspd_ms"]),
                }
            )

    return pd.DataFrame(records)


def analyze_regional_errors(errors: pd.DataFrame):
    """Print analysis of HRRR error patterns across all stations."""
    print("=" * 80)
    print("HRRR ERROR ANALYSIS ACROSS CHESAPEAKE BAY STATIONS")
    print("=" * 80)

    # Per-station error stats
    print(f"\n{'Station':<10s} {'Cat':10s} {'N':>5s} {'MAE(kt)':>8s} {'Bias(kt)':>9s} {'RMSE(kt)':>9s}")
    print("─" * 55)

    for station in ALL_STATIONS:
        sid = station.id
        sub = errors[errors["station_id"] == sid]
        if len(sub) < 10:
            continue
        mae = sub["abs_error"].mean() * KT
        bias = sub["error"].mean() * KT
        rmse = np.sqrt((sub["error"] ** 2).mean()) * KT
        print(f"{sid:<10s} {station.category:10s} {len(sub):5d} {mae:8.2f} {bias:+9.2f} {rmse:9.2f}")

    # Error correlation between stations
    print("\n\nHRRR Error Correlation Between Stations (12h lead):")
    print("(Do errors at one station predict errors at another?)")

    errors_12 = errors[errors["lead_hours"] == 12]
    pivot = errors_12.pivot_table(
        index="valid_time", columns="station_id", values="error", aggfunc="first"
    )

    if pivot.shape[1] > 2:
        corr = pivot.corr()
        # Show correlations with TPLM2
        if "TPLM2" in corr.columns:
            print(f"\n  Error correlation with TPLM2:")
            tplm2_corr = corr["TPLM2"].drop("TPLM2", errors="ignore").sort_values(ascending=False)
            for sid, r in tplm2_corr.items():
                if not pd.isna(r):
                    print(f"    {sid:10s}: r={r:.3f}")

    # Error by lead time
    print("\n\nHRRR Error by Lead Time (all stations combined):")
    for lead in sorted(errors["lead_hours"].unique()):
        sub = errors[errors["lead_hours"] == lead]
        mae = sub["abs_error"].mean() * KT
        bias = sub["error"].mean() * KT
        print(f"  {lead:2d}h lead: MAE={mae:.2f} kt, Bias={bias:+.2f} kt, N={len(sub)}")

    # Error by wind speed regime
    print("\n\nHRRR Error by Wind Speed Regime (12h lead, all stations):")
    sub = errors_12.copy()
    sub["actual_kt"] = sub["actual_wspd"] * KT
    bins = [(0, 8, "Light (0-8kt)"), (8, 15, "Moderate (8-15kt)"),
            (15, 25, "Strong (15-25kt)"), (25, 50, "Gale (25+kt)")]
    for lo, hi, label in bins:
        mask = (sub["actual_kt"] >= lo) & (sub["actual_kt"] < hi)
        if mask.sum() < 5:
            continue
        mae = sub.loc[mask, "abs_error"].mean() * KT
        bias = sub.loc[mask, "error"].mean() * KT
        print(f"  {label:20s}: MAE={mae:.2f} kt, Bias={bias:+.2f} kt, N={mask.sum()}")


def build_mos_correction(
    errors: pd.DataFrame,
    unified: pd.DataFrame,
    target_station: str = "TPLM2",
    lead_hours: int = 12,
    test_start: str = "2025-11-15",
):
    """
    Build a MOS model to correct HRRR forecasts at a target station.

    The model predicts HRRR error using:
    1. HRRR forecast value itself (captures nonlinear bias)
    2. HRRR forecast direction (bias varies by wind direction)
    3. Current observations at surrounding stations (available at forecast time)
    4. Pressure gradients and tendencies
    5. Time of day / season
    6. HRRR errors at OTHER stations (if HRRR is wrong at Solomons, it may be wrong here too)
    """
    # Filter to target station and lead time
    target_errors = errors[
        (errors["station_id"] == target_station) & (errors["lead_hours"] == lead_hours)
    ].set_index("valid_time").sort_index()

    if len(target_errors) < 20:
        print(f"Only {len(target_errors)} records for {target_station} at {lead_hours}h, insufficient")
        return None

    # Build feature matrix
    # Features are available at INIT time (= valid_time - lead_hours)
    features = pd.DataFrame(index=target_errors.index)

    # HRRR forecast itself
    features["hrrr_wspd"] = target_errors["hrrr_wspd"]
    features["hrrr_wdir_sin"] = np.sin(np.deg2rad(target_errors["hrrr_wdir"]))
    features["hrrr_wdir_cos"] = np.cos(np.deg2rad(target_errors["hrrr_wdir"]))
    features["hrrr_wspd_sq"] = target_errors["hrrr_wspd"] ** 2  # nonlinear bias

    # Time features
    features["hour_sin"] = np.sin(2 * np.pi * target_errors.index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * target_errors.index.hour / 24)
    features["month_sin"] = np.sin(2 * np.pi * target_errors.index.month / 12)
    features["month_cos"] = np.cos(2 * np.pi * target_errors.index.month / 12)

    # Station observations at INIT time (lead_hours before valid time)
    init_times = target_errors.index - pd.Timedelta(hours=lead_hours)

    key_stations = ["TPLM2", "APAM2", "COVM2", "CAMM2", "SLIM2", "WASD2", "44009", "BLTM2"]
    for sid in key_stations:
        wspd_col = f"{sid}_WSPD"
        pres_col = f"{sid}_PRES"
        if wspd_col in unified.columns:
            features[f"{sid}_obs_wspd"] = unified[wspd_col].reindex(init_times).values
        if pres_col in unified.columns:
            features[f"{sid}_obs_pres"] = unified[pres_col].reindex(init_times).values
            # Pressure tendency at init time
            pres_diff = unified[pres_col].diff(6)
            features[f"{sid}_pres_diff6"] = pres_diff.reindex(init_times).values

    # Pressure gradients at init time
    if "TPLM2_PRES" in unified.columns and "44009_PRES" in unified.columns:
        grad = unified["TPLM2_PRES"] - unified["44009_PRES"]
        features["pres_grad_ocean"] = grad.reindex(init_times).values
    if "TPLM2_PRES" in unified.columns and "WASD2_PRES" in unified.columns:
        grad = unified["TPLM2_PRES"] - unified["WASD2_PRES"]
        features["pres_grad_west"] = grad.reindex(init_times).values

    # Wind at target station at init time + recent history
    target_wspd = f"{target_station}_WSPD"
    if target_wspd in unified.columns:
        features["target_obs_wspd_init"] = unified[target_wspd].reindex(init_times).values
        wspd_diff = unified[target_wspd].diff(6)
        features["target_wspd_trend"] = wspd_diff.reindex(init_times).values

    # HRRR errors at other stations for same valid time (regional error signal)
    for sid in ["SLIM2", "COVM2", "APAM2", "CAMM2", "44009"]:
        if sid == target_station:
            continue
        other = errors[
            (errors["station_id"] == sid) & (errors["lead_hours"] == lead_hours)
        ].set_index("valid_time")
        if len(other) > 0:
            features[f"{sid}_hrrr_wspd"] = other["hrrr_wspd"].reindex(target_errors.index)

    # Target: HRRR error (what we want to predict and correct)
    target = target_errors["error"]

    # Split
    train_mask = features.index < test_start
    test_mask = features.index >= test_start

    if train_mask.sum() < 20 or test_mask.sum() < 10:
        print(f"Insufficient data: {train_mask.sum()} train, {test_mask.sum()} test")
        return None

    X_train = features[train_mask].fillna(-999)
    y_train = target[train_mask]
    X_test = features[test_mask].fillna(-999)
    y_test = target[test_mask]

    # Use GradientBoosting for better handling of small datasets
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict corrections
    predicted_error = model.predict(X_test)

    # Corrected HRRR = HRRR + predicted_correction
    hrrr_raw = target_errors.loc[test_mask, "hrrr_wspd"]
    actual = target_errors.loc[test_mask, "actual_wspd"]
    corrected = hrrr_raw + predicted_error

    # Also get persistence for comparison
    persist = unified[f"{target_station}_WSPD"].reindex(
        target_errors.index[test_mask] - pd.Timedelta(hours=lead_hours)
    )
    persist.index = target_errors.index[test_mask]

    # Results
    print(f"\n{'=' * 75}")
    print(f"MOS CORRECTION RESULTS: {target_station} at {lead_hours}h lead")
    print(f"{'=' * 75}")
    print(f"Test period: {test_start} onward ({test_mask.sum()} forecasts)")

    results = {
        "Persistence": persist,
        f"HRRR raw ({lead_hours}h lead)": hrrr_raw,
        f"HRRR + MOS correction": corrected,
    }

    print(f"\n  {'Model':<35s} {'MAE(kt)':>8s} {'Bias(kt)':>9s} {'RMSE(kt)':>9s}")
    print(f"  {'─' * 63}")

    for name, pred in results.items():
        valid = actual.notna() & pred.notna()
        if valid.sum() < 5:
            continue
        a, p = actual[valid], pred[valid]
        mae = mean_absolute_error(a, p) * KT
        bias = (p.mean() - a.mean()) * KT
        rmse = np.sqrt(mean_squared_error(a, p)) * KT
        print(f"  {name:<35s} {mae:8.2f} {bias:+9.2f} {rmse:9.2f}")

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
    print(f"\n  MOS correction model — top 10 features:")
    for feat, v in imp.head(10).items():
        print(f"    {feat:35s} {v:.4f}")

    return model, features, target


def run_all_stations_mos(lead_hours: int = 12):
    """Run MOS correction for all stations that have enough data."""
    hrrr, unified = load_regional_data()
    errors = compute_hrrr_errors(hrrr, unified)

    # First, show the regional error analysis
    analyze_regional_errors(errors)

    # Then run MOS for each station with enough data
    print("\n\n" + "=" * 80)
    print("MOS CORRECTION RESULTS BY STATION")
    print("=" * 80)

    stations_with_data = errors[errors["lead_hours"] == lead_hours]["station_id"].value_counts()
    for sid in stations_with_data[stations_with_data >= 30].index:
        build_mos_correction(errors, unified, target_station=sid, lead_hours=lead_hours)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_stations_mos(lead_hours=12)
