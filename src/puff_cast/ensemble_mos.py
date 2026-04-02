"""
Ensemble MOS: combines multiple NWP model outputs with station observations.

Instead of correcting a single model, this takes HRRR, GFS, and ECMWF forecasts
at the target valid time, plus station/ASOS observations at init time, and learns
an optimal blend. The model can learn:
- When to trust HRRR vs GFS (HRRR better in convective; GFS may be better in synoptic)
- How much to adjust based on current station conditions
- Direction-dependent and regime-dependent corrections

This is the "kitchen sink" approach — throw everything we have at a GBR and see
what sticks. The feature importance tells us what actually matters.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KT = 1.944


def angular_difference(a, b):
    """Signed angular difference in degrees, range [-180, 180]."""
    return (a - b + 180) % 360 - 180


def load_ensemble_data(target_station: str = "TPLM2"):
    """Load all model forecasts + observations aligned by valid_time and lead."""
    unified = pd.read_parquet(DATA_DIR / "processed" / "unified_hourly.parquet")

    # HRRR regional — prefer enhanced (4 inits + extra vars) if available
    enhanced_path = DATA_DIR / "raw" / "hrrr_enhanced.csv"
    fallback_path = DATA_DIR / "raw" / "hrrr_regional_full.csv"
    hrrr_path = enhanced_path if enhanced_path.exists() else fallback_path
    hrrr_reg = pd.read_csv(
        hrrr_path,
        parse_dates=["init_time", "valid_time"],
    )
    logger.info(f"Loaded HRRR from {hrrr_path.name}: {len(hrrr_reg)} records")

    # GFS lead-time forecasts (TPLM2 only)
    gfs = pd.read_csv(
        DATA_DIR / "raw" / "gfs_lead_time.csv",
        parse_dates=["init_time", "valid_time"],
    )

    # ECMWF from Open-Meteo (hourly, not lead-specific — best available ~6h)
    ecmwf_path = DATA_DIR / "raw" / "ecmwf_openmeteo.csv"
    ecmwf = pd.read_csv(ecmwf_path, index_col="time", parse_dates=True) if ecmwf_path.exists() else None

    return unified, hrrr_reg, gfs, ecmwf


def build_ensemble_features(
    unified: pd.DataFrame,
    hrrr_reg: pd.DataFrame,
    gfs: pd.DataFrame,
    ecmwf: pd.DataFrame | None,
    target_station: str,
    lead_hours: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix for ensemble MOS.

    Returns (features, actual_wspd, actual_wdir) aligned by valid_time.
    """
    # Get ALL HRRR forecasts at target station for this lead time
    # (multiple init times can produce forecasts for the same valid time)
    target_hrrr_all = hrrr_reg[
        (hrrr_reg["station_id"] == target_station) & (hrrr_reg["lead_hours"] == lead_hours)
    ].copy().sort_values(["valid_time", "init_time"])

    # For test-time lookup: keep latest init per valid_time (most recent forecast)
    target_hrrr = target_hrrr_all.drop_duplicates("valid_time", keep="last").set_index("valid_time")

    # Get GFS forecasts (only at TPLM2 currently)
    gfs_lead = gfs[gfs["lead_hours"] == lead_hours].copy()
    gfs_lead = gfs_lead.sort_values("init_time").drop_duplicates("valid_time", keep="last")
    gfs_lead = gfs_lead.set_index("valid_time")

    obs_col = f"{target_station}_WSPD"
    obs_wdir_col = f"{target_station}_WDIR"

    # Pre-compute HRRR trend: for each valid time, difference between
    # latest and earliest init forecasts (model "changing its mind")
    hrrr_trend = {}
    for vt_g, group in target_hrrr_all.groupby("valid_time"):
        if len(group) >= 2:
            sg = group.sort_values("init_time")
            hrrr_trend[vt_g] = sg.iloc[-1]["hrrr_wspd_ms"] - sg.iloc[0]["hrrr_wspd_ms"]

    # Build one sample per (init_time, valid_time) pair — each init gives different
    # station obs context, so these are genuinely different training samples
    sample_rows = []
    for _, hrrr_row in target_hrrr_all.iterrows():
        vt = hrrr_row["valid_time"]
        init_time = hrrr_row["init_time"]

        if vt not in unified.index:
            continue
        actual = unified.loc[vt, obs_col] if obs_col in unified.columns else np.nan
        if pd.isna(actual):
            continue
        sample_rows.append((vt, init_time, hrrr_row))

    if len(sample_rows) == 0:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    feat_rows = []
    valid_times_list = []
    actuals_list = []

    for vt, init_time, hrrr_row_data in sample_rows:
        feat = {}

        # === NWP model forecasts for this valid time ===

        # HRRR at target station (from this specific init)
        feat["hrrr_wspd"] = hrrr_row_data["hrrr_wspd_ms"]
        feat["hrrr_wspd_sq"] = hrrr_row_data["hrrr_wspd_ms"] ** 2
        hdir = np.deg2rad(hrrr_row_data["hrrr_wdir"])
        feat["hrrr_wdir_sin"] = np.sin(hdir)
        feat["hrrr_wdir_cos"] = np.cos(hdir)

        # Enhanced HRRR variables (from hrrr_enhanced.csv)
        for var in ["hrrr_gust_ms", "hrrr_cape_jkg", "hrrr_pbl_m", "hrrr_sp_pa", "hrrr_fricv_ms"]:
            if var in hrrr_row_data.index:
                feat[var] = hrrr_row_data[var]

        # Gust factor (gust / mean wind — indicates turbulence)
        if "hrrr_gust_ms" in hrrr_row_data.index and hrrr_row_data["hrrr_wspd_ms"] > 0.5:
            gust_val = hrrr_row_data.get("hrrr_gust_ms", np.nan)
            if not pd.isna(gust_val):
                feat["hrrr_gust_factor"] = gust_val / hrrr_row_data["hrrr_wspd_ms"]

        # HRRR trend: is the model changing its mind between successive inits?
        if vt in hrrr_trend:
            feat["hrrr_trend"] = hrrr_trend[vt]

        # HRRR at surrounding stations (regional context)
        regional = hrrr_reg[
            (hrrr_reg["valid_time"] == vt) & (hrrr_reg["lead_hours"] == lead_hours)
        ]
        for _, rrow in regional.iterrows():
            sid = rrow["station_id"]
            if sid != target_station:
                feat[f"{sid}_hrrr_wspd"] = rrow["hrrr_wspd_ms"]
                # Regional CAPE and gusts where available
                if "hrrr_cape_jkg" in rrow.index:
                    feat[f"{sid}_hrrr_cape"] = rrow["hrrr_cape_jkg"]
                if "hrrr_gust_ms" in rrow.index:
                    feat[f"{sid}_hrrr_gust"] = rrow["hrrr_gust_ms"]

        # GFS at target
        if vt in gfs_lead.index:
            feat["gfs_wspd"] = gfs_lead.loc[vt, "wspd_ms"]
            gdir = np.deg2rad(gfs_lead.loc[vt, "wdir"])
            feat["gfs_wdir_sin"] = np.sin(gdir)
            feat["gfs_wdir_cos"] = np.cos(gdir)
            # HRRR-GFS spread (disagreement signal)
            if "hrrr_wspd" in feat:
                feat["hrrr_gfs_spread"] = feat["hrrr_wspd"] - feat["gfs_wspd"]

        # ECMWF (shifted to approximate the valid time)
        if ecmwf is not None and vt in ecmwf.index:
            if "ecmwf_wind_speed_10m" in ecmwf.columns:
                ecmwf_wspd = ecmwf.loc[vt, "ecmwf_wind_speed_10m"]
                feat["ecmwf_wspd"] = ecmwf_wspd
                if "hrrr_wspd" in feat:
                    feat["hrrr_ecmwf_spread"] = feat["hrrr_wspd"] - ecmwf_wspd
            if "ecmwf_wind_direction_10m" in ecmwf.columns:
                edir = np.deg2rad(ecmwf.loc[vt, "ecmwf_wind_direction_10m"])
                feat["ecmwf_wdir_sin"] = np.sin(edir)
                feat["ecmwf_wdir_cos"] = np.cos(edir)

        # GFS-ECMWF spread
        if "gfs_wspd" in feat and "ecmwf_wspd" in feat:
            feat["gfs_ecmwf_spread"] = feat["gfs_wspd"] - feat["ecmwf_wspd"]

        # Model consensus and spread (disagreement signal)
        model_wspds = [v for k, v in feat.items()
                       if k in ("hrrr_wspd", "gfs_wspd", "ecmwf_wspd") and not pd.isna(v)]
        if len(model_wspds) >= 2:
            feat["model_consensus"] = np.mean(model_wspds)
            feat["model_spread_std"] = np.std(model_wspds)

        # === Wind regime indicators ===
        if "hrrr_wspd" in feat:
            wspd_kt = feat["hrrr_wspd"] * KT
            feat["regime_light"] = 1.0 if wspd_kt < 5 else 0.0
            feat["regime_moderate"] = 1.0 if 5 <= wspd_kt < 15 else 0.0
            feat["regime_strong"] = 1.0 if wspd_kt >= 15 else 0.0

        # === Time features ===
        feat["hour_sin"] = np.sin(2 * np.pi * vt.hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * vt.hour / 24)
        feat["month_sin"] = np.sin(2 * np.pi * vt.month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * vt.month / 12)

        # === Station observations at init time ===
        if init_time in unified.index:
            row = unified.loc[init_time]

            # Target station
            for var in ["WSPD", "GST", "PRES", "ATMP"]:
                col = f"{target_station}_{var}"
                if col in unified.columns:
                    feat[f"target_{var}"] = row.get(col, np.nan)

            for comp in ["WDIR_sin", "WDIR_cos"]:
                col = f"{target_station}_{comp}"
                if col in unified.columns:
                    feat[f"target_{comp}"] = row.get(col, np.nan)

            # Target trends
            for var in ["WSPD", "PRES"]:
                col = f"{target_station}_{var}"
                if col in unified.columns:
                    for lag in [3, 6]:
                        prev = init_time - pd.Timedelta(hours=lag)
                        if prev in unified.index:
                            prev_val = unified.loc[prev, col]
                            if not pd.isna(prev_val):
                                feat[f"target_{var}_diff{lag}"] = row.get(col, np.nan) - prev_val

            # Surrounding buoy stations
            for sid in ["APAM2", "COVM2", "CAMM2", "SLIM2", "WASD2", "44009", "BLTM2"]:
                if sid == target_station:
                    continue
                for var_col, var_name in [(f"{sid}_WSPD", f"{sid}_wspd"), (f"{sid}_PRES", f"{sid}_pres")]:
                    if var_col in unified.columns:
                        feat[var_name] = row.get(var_col, np.nan)

            # Pressure gradients
            target_pres = row.get(f"{target_station}_PRES", np.nan)
            if not pd.isna(target_pres):
                for sid, label in [("44009", "ocean"), ("WASD2", "west"), ("SLIM2", "south")]:
                    pcol = f"{sid}_PRES"
                    if pcol in unified.columns:
                        op = row.get(pcol, np.nan)
                        if not pd.isna(op):
                            feat[f"pres_grad_{label}"] = target_pres - op

            # Temperature differences (convective potential)
            for sid in ["APAM2", "CAMM2", "SLIM2", "44009"]:
                td_col = f"{sid}_TEMP_DIFF"
                if td_col in unified.columns:
                    feat[f"{sid}_temp_diff"] = row.get(td_col, np.nan)

            # ASOS airport stations
            for sid in ["KBWI", "KDCA", "KNAK", "KNHK", "KESN", "KSBY", "KDOV"]:
                wspd_col = f"{sid}_wspd_ms"
                if wspd_col in unified.columns:
                    feat[f"{sid}_wspd"] = row.get(wspd_col, np.nan)
                mslp_col = f"{sid}_mslp"
                if mslp_col in unified.columns:
                    feat[f"{sid}_mslp"] = row.get(mslp_col, np.nan)
                for vcol, vname in [("vsby_km", "vsby"), ("ceil_m", "ceil"), ("cloud", "cloud")]:
                    full_col = f"{sid}_{vcol}"
                    if full_col in unified.columns:
                        feat[f"{sid}_{vname}"] = row.get(full_col, np.nan)

            # === Tidal / CO-OPS features at init time ===
            # Water level and tide state at Annapolis
            for col_suffix in ["water_level_m", "water_level_diff1", "water_level_diff3"]:
                col = f"COOPS_8575512_{col_suffix}"
                if col in unified.columns:
                    feat[col_suffix] = row.get(col, np.nan)

            # Tidal current speed and direction at Bay Bridge
            for col_suffix in ["current_speed_ms", "current_dir"]:
                col = f"tidal_{col_suffix}"
                if col in unified.columns:
                    val = row.get(col, np.nan)
                    feat[col_suffix] = val
                    if col_suffix == "current_dir" and not pd.isna(val):
                        feat["current_dir_sin"] = np.sin(np.deg2rad(val))
                        feat["current_dir_cos"] = np.cos(np.deg2rad(val))

            # Wind-current interaction: are wind and current aligned or opposing?
            if "hrrr_wspd" in feat and not pd.isna(feat.get("current_speed_ms", np.nan)):
                # Current opposing HRRR wind direction increases effective roughness
                hrrr_wdir_deg = np.rad2deg(np.arctan2(
                    feat.get("hrrr_wdir_sin", 0), feat.get("hrrr_wdir_cos", 1)
                )) % 360
                curr_dir = feat.get("current_dir", np.nan)
                if not pd.isna(curr_dir):
                    wind_current_angle = abs(angular_difference(hrrr_wdir_deg, curr_dir))
                    feat["wind_current_angle"] = wind_current_angle
                    feat["wind_current_opposing"] = 1.0 if wind_current_angle > 90 else 0.0
                    # Wind-over-current: stronger effect when both are strong
                    feat["wind_current_product"] = feat["hrrr_wspd"] * feat["current_speed_ms"]

            # CO-OPS water temperature (for air-water temp diff at Annapolis)
            wtmp_col = "COOPS_8575512_coops_wtmp_c"
            if wtmp_col in unified.columns:
                wtmp = row.get(wtmp_col, np.nan)
                feat["coops_wtmp"] = wtmp
                # Air-water temp diff using ASOS air temp
                atmp_col = "KNAK_atmp_c"
                if atmp_col in unified.columns and not pd.isna(wtmp):
                    atmp = row.get(atmp_col, np.nan)
                    if not pd.isna(atmp):
                        feat["annapolis_air_water_diff"] = atmp - wtmp

        feat_rows.append(feat)
        valid_times_list.append(vt)
        actuals_list.append(unified.loc[vt, obs_col])

    # Use (vt, init_time_idx) as index to avoid duplicate index issues
    sample_index = pd.RangeIndex(len(feat_rows))
    X = pd.DataFrame(feat_rows, index=sample_index)

    # Store valid_time as a column for train/test splitting
    X["_valid_time"] = valid_times_list
    actual_wspd = pd.Series(actuals_list, index=sample_index, name="actual_wspd")

    # Get actual wind direction for direction model
    actual_wdir_vals = []
    for vt in valid_times_list:
        if obs_wdir_col in unified.columns:
            actual_wdir_vals.append(unified.loc[vt, obs_wdir_col])
        else:
            actual_wdir_vals.append(np.nan)
    actual_wdir = pd.Series(actual_wdir_vals, index=sample_index)

    return X, actual_wspd, actual_wdir


def run_ensemble_mos(
    target_station: str = "TPLM2",
    lead_hours: int = 12,
    test_start: str = "2025-10-01",
):
    """Run ensemble MOS combining HRRR + GFS + ECMWF + station obs."""
    unified, hrrr_reg, gfs, ecmwf = load_ensemble_data(target_station)

    print(f"Building ensemble features for {target_station} at {lead_hours}h lead...")
    X, actual_wspd, actual_wdir = build_ensemble_features(
        unified, hrrr_reg, gfs, ecmwf, target_station, lead_hours
    )

    if len(X) == 0:
        print("  No valid samples!")
        return None

    print(f"  Samples: {len(X)}, Features: {len(X.columns)}")

    # Split
    train_mask = X.index < test_start
    test_mask = X.index >= test_start

    if train_mask.sum() < 30 or test_mask.sum() < 10:
        print(f"  Insufficient data: {train_mask.sum()} train, {test_mask.sum()} test")
        return None

    # === Speed model (predicts actual, not error) ===
    speed_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=5,
        learning_rate=0.05, random_state=42,
    )
    speed_model.fit(X[train_mask].fillna(-999), actual_wspd[train_mask])
    ensemble_speed = pd.Series(
        speed_model.predict(X[test_mask].fillna(-999)),
        index=X.index[test_mask],
    )

    # === Direction model (predicts angular error from HRRR) ===
    has_wdir = actual_wdir.notna().sum() > 0
    dir_model = None
    if has_wdir and "hrrr_wdir_sin" in X.columns:
        # Reconstruct HRRR direction from sin/cos
        hrrr_wdir = np.rad2deg(np.arctan2(X["hrrr_wdir_sin"], X["hrrr_wdir_cos"])) % 360
        dir_error = angular_difference(actual_wdir, hrrr_wdir)
        dir_valid = dir_error.notna() & (X.get("hrrr_wspd", pd.Series(dtype=float)) * KT >= 5).fillna(False)
        dir_train = train_mask & dir_valid
        dir_test = test_mask & dir_valid

        if dir_train.sum() >= 20 and dir_test.sum() >= 5:
            dir_model = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, min_samples_leaf=5,
                learning_rate=0.05, random_state=42,
            )
            dir_model.fit(X[dir_train].fillna(-999), dir_error[dir_train])

    # === Evaluate ===
    test_actual = actual_wspd[test_mask]
    test_idx = X.index[test_mask]

    # Baselines
    persist_times = test_idx - pd.Timedelta(hours=lead_hours)
    persist = unified[f"{target_station}_WSPD"].reindex(persist_times)
    persist.index = test_idx

    hrrr_raw = X.loc[test_mask, "hrrr_wspd"] if "hrrr_wspd" in X.columns else None
    gfs_raw = X.loc[test_mask, "gfs_wspd"] if "gfs_wspd" in X.columns else None
    ecmwf_raw = X.loc[test_mask, "ecmwf_wspd"] if "ecmwf_wspd" in X.columns else None

    print(f"\n{'=' * 75}")
    print(f"ENSEMBLE MOS: {target_station} at {lead_hours}h lead")
    print(f"Test: {test_start} onward ({test_mask.sum()} forecasts)")
    print(f"{'=' * 75}")

    print(f"\n  WIND SPEED:")
    print(f"  {'Model':<40s} {'MAE(kt)':>8s} {'Bias(kt)':>9s} {'RMSE(kt)':>9s}")
    print(f"  {'─' * 68}")

    models = [("Persistence", persist)]
    if hrrr_raw is not None:
        models.append((f"HRRR raw ({lead_hours}h)", hrrr_raw))
    if gfs_raw is not None:
        models.append((f"GFS raw ({lead_hours}h)", gfs_raw))
    if ecmwf_raw is not None:
        models.append(("ECMWF IFS (best avail)", ecmwf_raw))
    models.append((">>> Ensemble MOS", ensemble_speed))

    for name, pred in models:
        if pred is None:
            continue
        v = test_actual.notna() & pred.notna()
        if v.sum() < 5:
            continue
        a, p = test_actual[v], pred[v]
        mae = mean_absolute_error(a, p) * KT
        bias = (p.mean() - a.mean()) * KT
        rmse = np.sqrt(mean_squared_error(a, p)) * KT
        print(f"  {name:<40s} {mae:8.2f} {bias:+9.2f} {rmse:9.2f}")

    # Direction
    if dir_model is not None and has_wdir:
        print(f"\n  WIND DIRECTION (>5kt only):")
        print(f"  {'Model':<40s} {'MAE(°)':>8s} {'Bias(°)':>9s}")
        print(f"  {'─' * 60}")

        hrrr_wdir_test = np.rad2deg(np.arctan2(
            X.loc[test_mask, "hrrr_wdir_sin"], X.loc[test_mask, "hrrr_wdir_cos"]
        )) % 360
        actual_dir_test = actual_wdir.reindex(test_idx)
        dir_valid_test = actual_dir_test.notna() & (X.loc[test_mask, "hrrr_wspd"] * KT >= 5)
        dir_idx = test_idx[dir_valid_test]

        if len(dir_idx) >= 5:
            ad = actual_dir_test.reindex(dir_idx)
            hd = hrrr_wdir_test.reindex(dir_idx)

            # HRRR raw direction
            diff = angular_difference(ad.values, hd.values)
            print(f"  {f'HRRR raw ({lead_hours}h)':<40s} {np.abs(diff).mean():8.1f} {diff.mean():+9.1f}")

            # GFS direction
            if "gfs_wdir_sin" in X.columns:
                gfs_wdir = np.rad2deg(np.arctan2(
                    X.loc[dir_idx, "gfs_wdir_sin"], X.loc[dir_idx, "gfs_wdir_cos"]
                )) % 360
                diff_g = angular_difference(ad.values, gfs_wdir.values)
                print(f"  {f'GFS raw ({lead_hours}h)':<40s} {np.abs(diff_g).mean():8.1f} {diff_g.mean():+9.1f}")

            # Ensemble corrected direction
            dir_correction = dir_model.predict(X.reindex(dir_idx).fillna(-999))
            corrected_dir = (hd + dir_correction) % 360
            diff_e = angular_difference(ad.values, corrected_dir.values)
            print(f"  {'>>> Ensemble MOS':<40s} {np.abs(diff_e).mean():8.1f} {diff_e.mean():+9.1f}")

    # Feature importance
    imp = pd.Series(speed_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n  Ensemble speed — top 15 features:")
    for feat, v in imp.head(15).items():
        print(f"    {feat:40s} {v:.4f}")

    if dir_model is not None:
        imp_dir = pd.Series(dir_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"\n  Ensemble direction — top 10 features:")
        for feat, v in imp_dir.head(10).items():
            print(f"    {feat:40s} {v:.4f}")

    return speed_model, dir_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    stations = ["TPLM2", "APAM2", "SLIM2", "CAMM2"]
    leads = [3, 6, 12]

    for lead in leads:
        for station in stations:
            run_ensemble_mos(target_station=station, lead_hours=lead, test_start="2025-10-01")
            print()
