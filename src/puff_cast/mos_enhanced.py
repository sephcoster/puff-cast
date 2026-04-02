"""
Enhanced MOS correction model for Chesapeake Bay wind forecasting.

Improvements over basic MOS:
1. Predicts BOTH wind speed and direction (direction matters >5kt)
2. Uses air-water temperature difference (convective potential)
3. Includes pressure gradients across multiple axes
4. Direction-dependent bias correction
5. Extensible to any station in the Bay
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from puff_cast.stations import ALL_STATIONS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KT = 1.944


def angular_difference(a, b):
    """Signed angular difference in degrees, range [-180, 180]."""
    diff = (a - b + 180) % 360 - 180
    return diff


def direction_mae(actual, predicted):
    """Mean absolute error for circular quantities (degrees)."""
    diff = angular_difference(actual, predicted)
    return np.abs(diff).mean()


def build_enhanced_features(
    unified: pd.DataFrame,
    hrrr_row: dict,
    init_time: pd.Timestamp,
    target_station: str,
    hrrr_regional: pd.DataFrame | None = None,
) -> dict:
    """
    Build rich feature set for MOS correction.

    All features must be available at init_time (no future data leakage).
    """
    feat = {}

    # === HRRR forecast features ===
    feat["hrrr_wspd"] = hrrr_row["hrrr_wspd_ms"]
    feat["hrrr_wspd_sq"] = hrrr_row["hrrr_wspd_ms"] ** 2
    hrrr_wdir_rad = np.deg2rad(hrrr_row["hrrr_wdir"])
    feat["hrrr_wdir_sin"] = np.sin(hrrr_wdir_rad)
    feat["hrrr_wdir_cos"] = np.cos(hrrr_wdir_rad)

    # HRRR wind regime (lets model learn different corrections per regime)
    wspd_kt = hrrr_row["hrrr_wspd_ms"] * KT
    feat["hrrr_light"] = 1.0 if wspd_kt < 5 else 0.0
    feat["hrrr_moderate"] = 1.0 if 5 <= wspd_kt < 15 else 0.0
    feat["hrrr_strong"] = 1.0 if wspd_kt >= 15 else 0.0

    # === Time features ===
    vt = hrrr_row["valid_time"] if isinstance(hrrr_row["valid_time"], pd.Timestamp) else pd.Timestamp(hrrr_row["valid_time"])
    feat["hour_sin"] = np.sin(2 * np.pi * vt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * vt.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * vt.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * vt.month / 12)

    # === Station observations at init time ===
    if init_time in unified.index:
        row = unified.loc[init_time]

        # Target station current conditions
        for var in ["WSPD", "GST", "PRES", "ATMP"]:
            col = f"{target_station}_{var}"
            if col in unified.columns:
                feat[f"target_{var}"] = row.get(col, np.nan)

        # Target wind direction
        for comp in ["WDIR_sin", "WDIR_cos"]:
            col = f"{target_station}_{comp}"
            if col in unified.columns:
                feat[f"target_{comp}"] = row.get(col, np.nan)

        # Target wind/pressure trends
        for var in ["WSPD", "PRES"]:
            col = f"{target_station}_{var}"
            if col in unified.columns:
                for lag in [3, 6, 12]:
                    prev_time = init_time - pd.Timedelta(hours=lag)
                    if prev_time in unified.index:
                        prev_val = unified.loc[prev_time, col]
                        feat[f"target_{var}_diff{lag}"] = row.get(col, np.nan) - prev_val if not pd.isna(prev_val) else np.nan

        # Key surrounding stations
        for sid in ["APAM2", "COVM2", "CAMM2", "SLIM2", "WASD2", "44009", "BLTM2"]:
            if sid == target_station:
                continue
            wspd_col = f"{sid}_WSPD"
            pres_col = f"{sid}_PRES"
            if wspd_col in unified.columns:
                feat[f"{sid}_wspd"] = row.get(wspd_col, np.nan)
            if pres_col in unified.columns:
                feat[f"{sid}_pres"] = row.get(pres_col, np.nan)
                # Pressure tendency
                prev_time = init_time - pd.Timedelta(hours=6)
                if prev_time in unified.index:
                    prev_pres = unified.loc[prev_time, pres_col]
                    if not pd.isna(prev_pres):
                        feat[f"{sid}_pres_diff6"] = row.get(pres_col, np.nan) - prev_pres

        # === Air-water temperature difference (convective potential) ===
        for sid in ["APAM2", "CAMM2", "SLIM2", "44009"]:
            td_col = f"{sid}_TEMP_DIFF"
            if td_col in unified.columns:
                feat[f"{sid}_temp_diff"] = row.get(td_col, np.nan)

        # === Pressure gradients ===
        target_pres = row.get(f"{target_station}_PRES", np.nan)
        if not pd.isna(target_pres):
            for sid, label in [("44009", "ocean"), ("WASD2", "west"), ("SLIM2", "south"), ("BLTM2", "north")]:
                other_pres_col = f"{sid}_PRES"
                if other_pres_col in unified.columns:
                    other_pres = row.get(other_pres_col, np.nan)
                    if not pd.isna(other_pres):
                        feat[f"pres_grad_{label}"] = target_pres - other_pres

        # === ASOS airport observations at init time ===
        for sid in ["KBWI", "KDCA", "KNAK", "KNHK", "KESN", "KSBY", "KDOV"]:
            # Wind speed and direction
            wspd_col = f"{sid}_wspd_ms"
            if wspd_col in unified.columns:
                feat[f"{sid}_wspd"] = row.get(wspd_col, np.nan)
            drct_col = f"{sid}_drct"
            if drct_col in unified.columns:
                drct_val = row.get(drct_col, np.nan)
                if not pd.isna(drct_val):
                    feat[f"{sid}_wdir_sin"] = np.sin(np.deg2rad(drct_val))
                    feat[f"{sid}_wdir_cos"] = np.cos(np.deg2rad(drct_val))
            # Pressure
            pres_col = f"{sid}_mslp"
            if pres_col in unified.columns:
                feat[f"{sid}_mslp"] = row.get(pres_col, np.nan)
                # Pressure tendency
                prev_time = init_time - pd.Timedelta(hours=6)
                if prev_time in unified.index:
                    prev_pres = unified.loc[prev_time, pres_col]
                    if not pd.isna(prev_pres):
                        feat[f"{sid}_mslp_diff6"] = row.get(pres_col, np.nan) - prev_pres
            # Visibility and ceiling (weather approaching)
            for vcol, vname in [("vsby_km", "vsby"), ("ceil_m", "ceil"), ("cloud", "cloud")]:
                full_col = f"{sid}_{vcol}"
                if full_col in unified.columns:
                    feat[f"{sid}_{vname}"] = row.get(full_col, np.nan)

    # === HRRR at other stations (regional model state) ===
    if hrrr_regional is not None:
        vt_val = hrrr_row["valid_time"]
        lead = hrrr_row["lead_hours"]
        regional_same_time = hrrr_regional[
            (hrrr_regional["valid_time"] == vt_val) & (hrrr_regional["lead_hours"] == lead)
        ]
        for _, rrow in regional_same_time.iterrows():
            if rrow["station_id"] != target_station:
                feat[f"{rrow['station_id']}_hrrr_wspd"] = rrow["hrrr_wspd_ms"]

    return feat


def run_enhanced_mos(
    target_station: str = "TPLM2",
    lead_hours: int = 12,
    test_start: str = "2025-10-01",
    regional_file: str = "hrrr_regional.csv",
):
    """Run enhanced MOS for a target station with direction correction."""
    # Load data
    unified = pd.read_parquet(DATA_DIR / "processed" / "unified_hourly.parquet")
    hrrr = pd.read_csv(DATA_DIR / "raw" / regional_file, parse_dates=["init_time", "valid_time"])

    # Filter to target station and lead time
    station_hrrr = hrrr[
        (hrrr["station_id"] == target_station) & (hrrr["lead_hours"] == lead_hours)
    ].copy().sort_values("valid_time").drop_duplicates("valid_time", keep="last")

    print(f"Building features for {target_station} at {lead_hours}h lead...")
    print(f"  HRRR forecasts: {len(station_hrrr)}")

    # Build features and targets
    feature_rows = []
    speed_targets = []
    dir_targets = []
    hrrr_speeds = []
    hrrr_dirs = []
    valid_times = []

    obs_wspd_col = f"{target_station}_WSPD"
    obs_wdir_col = f"{target_station}_WDIR"
    has_wdir = obs_wdir_col in unified.columns

    for _, row in station_hrrr.iterrows():
        vt = row["valid_time"]
        init_time = row["init_time"]

        if vt not in unified.index:
            continue

        actual_wspd = unified.loc[vt, obs_wspd_col] if obs_wspd_col in unified.columns else np.nan
        actual_wdir = unified.loc[vt, obs_wdir_col] if has_wdir else np.nan

        if pd.isna(actual_wspd):
            continue

        feat = build_enhanced_features(unified, row, init_time, target_station, hrrr)
        feature_rows.append(feat)
        speed_targets.append(actual_wspd - row["hrrr_wspd_ms"])  # error to predict
        hrrr_speeds.append(row["hrrr_wspd_ms"])
        hrrr_dirs.append(row["hrrr_wdir"])
        valid_times.append(vt)

        if not pd.isna(actual_wdir):
            dir_targets.append(angular_difference(actual_wdir, row["hrrr_wdir"]))
        else:
            dir_targets.append(np.nan)

    X = pd.DataFrame(feature_rows, index=valid_times)
    y_speed = pd.Series(speed_targets, index=valid_times, name="speed_error")
    y_dir = pd.Series(dir_targets, index=valid_times, name="dir_error")
    hrrr_spd = pd.Series(hrrr_speeds, index=valid_times)
    hrrr_dir = pd.Series(hrrr_dirs, index=valid_times)

    print(f"  Valid samples: {len(X)}")

    # Split
    train_mask = X.index < test_start
    test_mask = X.index >= test_start

    if train_mask.sum() < 30 or test_mask.sum() < 10:
        print(f"  Insufficient data: {train_mask.sum()} train, {test_mask.sum()} test")
        return None

    # === Speed correction model ===
    speed_model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=5, learning_rate=0.05, random_state=42
    )
    speed_model.fit(X[train_mask].fillna(-999), y_speed[train_mask])
    speed_correction = speed_model.predict(X[test_mask].fillna(-999))
    corrected_speed = hrrr_spd[test_mask] + speed_correction

    # === Direction correction model (only where actual dir available and >5kt) ===
    dir_model = None
    if has_wdir:
        dir_valid = y_dir.notna() & (hrrr_spd * KT >= 5)
        dir_train = train_mask & dir_valid
        dir_test = test_mask & dir_valid

        if dir_train.sum() >= 20 and dir_test.sum() >= 5:
            dir_model = GradientBoostingRegressor(
                n_estimators=150, max_depth=4, min_samples_leaf=5, learning_rate=0.05, random_state=42
            )
            dir_model.fit(X[dir_train].fillna(-999), y_dir[dir_train])

    # === Evaluate ===
    actual_wspd = unified[obs_wspd_col].reindex(X.index[test_mask])
    actual_wdir = unified[obs_wdir_col].reindex(X.index[test_mask]) if has_wdir else None

    # Persistence
    persist_times = X.index[test_mask] - pd.Timedelta(hours=lead_hours)
    persist_wspd = unified[obs_wspd_col].reindex(persist_times)
    persist_wspd.index = X.index[test_mask]
    persist_wdir = unified[obs_wdir_col].reindex(persist_times) if has_wdir else None
    if persist_wdir is not None:
        persist_wdir.index = X.index[test_mask]

    print(f"\n{'=' * 75}")
    print(f"ENHANCED MOS: {target_station} at {lead_hours}h lead")
    print(f"Test: {test_start} onward ({test_mask.sum()} forecasts)")
    print(f"{'=' * 75}")

    # Speed results
    print(f"\n  WIND SPEED:")
    print(f"  {'Model':<40s} {'MAE(kt)':>8s} {'Bias(kt)':>9s} {'RMSE(kt)':>9s}")
    print(f"  {'─' * 68}")

    valid = actual_wspd.notna()
    for name, pred in [
        ("Persistence", persist_wspd),
        (f"HRRR raw ({lead_hours}h lead)", hrrr_spd[test_mask]),
        ("HRRR + MOS correction", corrected_speed),
    ]:
        if isinstance(pred, np.ndarray):
            pred = pd.Series(pred, index=X.index[test_mask])
        v = valid & pred.notna()
        if v.sum() < 5:
            continue
        a, p = actual_wspd[v], pred[v]
        mae = mean_absolute_error(a, p) * KT
        bias = (p.mean() - a.mean()) * KT
        rmse = np.sqrt(mean_squared_error(a, p)) * KT
        print(f"  {name:<40s} {mae:8.2f} {bias:+9.2f} {rmse:9.2f}")

    # Direction results (only >5kt)
    if actual_wdir is not None and dir_model is not None:
        print(f"\n  WIND DIRECTION (>5kt only):")
        print(f"  {'Model':<40s} {'MAE(°)':>8s} {'Bias(°)':>9s}")
        print(f"  {'─' * 60}")

        dir_test_mask = test_mask & (hrrr_spd * KT >= 5)
        dir_valid_times = X.index[dir_test_mask]
        actual_dir_v = unified[obs_wdir_col].reindex(dir_valid_times).dropna()
        dir_valid_times = actual_dir_v.index

        if len(dir_valid_times) >= 5:
            hrrr_dir_v = hrrr_dir.reindex(dir_valid_times)
            persist_dir_v = persist_wdir.reindex(dir_valid_times) if persist_wdir is not None else None

            # Persistence direction
            if persist_dir_v is not None:
                v = actual_dir_v.notna() & persist_dir_v.notna()
                if v.sum() >= 5:
                    diff = angular_difference(actual_dir_v[v].values, persist_dir_v[v].values)
                    print(f"  {'Persistence':<40s} {np.abs(diff).mean():8.1f} {diff.mean():+9.1f}")

            # HRRR raw direction
            v = actual_dir_v.notna() & hrrr_dir_v.notna()
            if v.sum() >= 5:
                diff = angular_difference(actual_dir_v[v].values, hrrr_dir_v[v].values)
                print(f"  {f'HRRR raw ({lead_hours}h lead)':<40s} {np.abs(diff).mean():8.1f} {diff.mean():+9.1f}")

            # MOS corrected direction
            dir_correction = dir_model.predict(X.reindex(dir_valid_times).fillna(-999))
            corrected_dir = (hrrr_dir_v + dir_correction) % 360
            v = actual_dir_v.notna() & corrected_dir.notna()
            if v.sum() >= 5:
                diff = angular_difference(actual_dir_v[v].values, corrected_dir[v].values)
                print(f"  {'HRRR + MOS correction':<40s} {np.abs(diff).mean():8.1f} {diff.mean():+9.1f}")

    # Feature importance
    imp = pd.Series(speed_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n  Speed correction — top 15 features:")
    for feat, v in imp.head(15).items():
        print(f"    {feat:40s} {v:.4f}")

    if dir_model is not None:
        imp_dir = pd.Series(dir_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"\n  Direction correction — top 10 features:")
        for feat, v in imp_dir.head(10).items():
            print(f"    {feat:40s} {v:.4f}")

    return speed_model, dir_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for station in ["TPLM2", "APAM2", "SLIM2", "CAMM2"]:
        run_enhanced_mos(
            target_station=station,
            lead_hours=12,
            test_start="2025-10-01",
            regional_file="hrrr_regional_full.csv",
        )
        print("\n")
