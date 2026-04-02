"""
Stacking ensemble with regime-specific models and HRRR trend features.

Architecture:
  Level 1: Separate MOS correction models for each NWP
    - HRRR MOS: corrects HRRR using station obs + ASOS + tidal
    - GFS MOS: corrects GFS using same features
    - ECMWF MOS: corrects ECMWF
    - Station-only model: predicts from obs alone (no NWP input)

  Level 2: Blender that combines L1 outputs + regime features
    - Learns when to trust each model
    - Regime-aware: different weights for light/moderate/strong wind

  Additionally:
    - HRRR trend features: difference between successive HRRR runs for same valid time
    - Regime-specific sub-models for light (<5kt) vs moderate (5-15kt) vs strong (>15kt)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
KT = 1.944


def angular_difference(a, b):
    return (a - b + 180) % 360 - 180


def load_all_data(target_station):
    """Load all data sources."""
    unified = pd.read_parquet(DATA_DIR / "processed" / "unified_hourly.parquet")
    hrrr_reg = pd.read_csv(
        DATA_DIR / "raw" / "hrrr_regional_full.csv",
        parse_dates=["init_time", "valid_time"],
    )
    gfs = pd.read_csv(
        DATA_DIR / "raw" / "gfs_lead_time.csv",
        parse_dates=["init_time", "valid_time"],
    )
    ecmwf_path = DATA_DIR / "raw" / "ecmwf_openmeteo.csv"
    ecmwf = pd.read_csv(ecmwf_path, index_col="time", parse_dates=True) if ecmwf_path.exists() else None
    return unified, hrrr_reg, gfs, ecmwf


def build_station_features(unified, init_time, target_station):
    """Build observation-only features at init time."""
    feat = {}
    if init_time not in unified.index:
        return feat

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

    # Temp differences
    for sid in ["APAM2", "CAMM2", "SLIM2", "44009"]:
        td_col = f"{sid}_TEMP_DIFF"
        if td_col in unified.columns:
            feat[f"{sid}_temp_diff"] = row.get(td_col, np.nan)

    # ASOS airports
    for sid in ["KBWI", "KDCA", "KNAK", "KNHK", "KESN", "KSBY", "KDOV"]:
        for col_suffix, feat_name in [
            ("wspd_ms", "wspd"), ("mslp", "mslp"),
            ("vsby_km", "vsby"), ("ceil_m", "ceil"), ("cloud", "cloud"),
        ]:
            col = f"{sid}_{col_suffix}"
            if col in unified.columns:
                feat[f"{sid}_{feat_name}"] = row.get(col, np.nan)

    # Tidal / CO-OPS
    for col_suffix in ["water_level_m", "water_level_diff1", "water_level_diff3"]:
        col = f"COOPS_8575512_{col_suffix}"
        if col in unified.columns:
            feat[col_suffix] = row.get(col, np.nan)
    for col_suffix in ["current_speed_ms", "current_dir"]:
        col = f"tidal_{col_suffix}"
        if col in unified.columns:
            feat[col_suffix] = row.get(col, np.nan)

    # Time features
    feat["hour_sin"] = np.sin(2 * np.pi * init_time.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * init_time.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * init_time.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * init_time.month / 12)

    return feat


def build_all_samples(unified, hrrr_reg, gfs, ecmwf, target_station, lead_hours):
    """Build aligned samples with all model forecasts and station features."""
    # Target HRRR
    target_hrrr = hrrr_reg[
        (hrrr_reg["station_id"] == target_station) & (hrrr_reg["lead_hours"] == lead_hours)
    ].copy().sort_values("valid_time").drop_duplicates("valid_time", keep="last")
    target_hrrr = target_hrrr.set_index("valid_time")

    # GFS
    gfs_lead = gfs[gfs["lead_hours"] == lead_hours].copy()
    gfs_lead = gfs_lead.sort_values("init_time").drop_duplicates("valid_time", keep="last")
    gfs_lead = gfs_lead.set_index("valid_time")

    # Valid times
    obs_col = f"{target_station}_WSPD"
    valid_times = target_hrrr.index.intersection(unified.index)
    valid_times = valid_times[unified[obs_col].reindex(valid_times).notna()]

    # Regional HRRR for surrounding stations
    regional_by_vt = {}
    for _, rrow in hrrr_reg[hrrr_reg["lead_hours"] == lead_hours].iterrows():
        vt = rrow["valid_time"]
        if vt not in regional_by_vt:
            regional_by_vt[vt] = {}
        regional_by_vt[vt][rrow["station_id"]] = rrow["hrrr_wspd_ms"]

    # === HRRR trend: difference between this init and previous init for same valid time ===
    # For 12h lead from 00Z init, the "previous" forecast for the same valid time
    # would be the 12Z init's 24h forecast (if we had it) — but we can approximate
    # by looking at the previous day's forecast
    hrrr_all_for_station = hrrr_reg[
        (hrrr_reg["station_id"] == target_station)
    ].copy().sort_values(["valid_time", "init_time"])

    # For each valid time, get multiple forecasts from different inits
    hrrr_trend = {}
    for vt, group in hrrr_all_for_station.groupby("valid_time"):
        if len(group) >= 2:
            # Latest minus earliest init for same valid time
            sorted_g = group.sort_values("init_time")
            hrrr_trend[vt] = sorted_g.iloc[-1]["hrrr_wspd_ms"] - sorted_g.iloc[0]["hrrr_wspd_ms"]

    rows = []
    for vt in valid_times:
        init_time = vt - pd.Timedelta(hours=lead_hours)

        # Station features
        station_feat = build_station_features(unified, init_time, target_station)

        # HRRR features
        hrrr_feat = {}
        if vt in target_hrrr.index:
            hrrr_wspd = target_hrrr.loc[vt, "hrrr_wspd_ms"]
            hrrr_wdir = target_hrrr.loc[vt, "hrrr_wdir"]
            hrrr_feat["hrrr_wspd"] = hrrr_wspd
            hrrr_feat["hrrr_wspd_sq"] = hrrr_wspd ** 2
            hrrr_feat["hrrr_wdir_sin"] = np.sin(np.deg2rad(hrrr_wdir))
            hrrr_feat["hrrr_wdir_cos"] = np.cos(np.deg2rad(hrrr_wdir))

            # Regional HRRR
            if vt in regional_by_vt:
                for sid, wspd in regional_by_vt[vt].items():
                    if sid != target_station:
                        hrrr_feat[f"{sid}_hrrr_wspd"] = wspd

            # HRRR trend
            if vt in hrrr_trend:
                hrrr_feat["hrrr_trend"] = hrrr_trend[vt]

            # Regime
            wspd_kt = hrrr_wspd * KT
            hrrr_feat["regime_light"] = 1.0 if wspd_kt < 5 else 0.0
            hrrr_feat["regime_moderate"] = 1.0 if 5 <= wspd_kt < 15 else 0.0
            hrrr_feat["regime_strong"] = 1.0 if wspd_kt >= 15 else 0.0

        # GFS features
        gfs_feat = {}
        if vt in gfs_lead.index:
            gfs_feat["gfs_wspd"] = gfs_lead.loc[vt, "wspd_ms"]
            gfs_feat["gfs_wdir_sin"] = np.sin(np.deg2rad(gfs_lead.loc[vt, "wdir"]))
            gfs_feat["gfs_wdir_cos"] = np.cos(np.deg2rad(gfs_lead.loc[vt, "wdir"]))

        # ECMWF features
        ecmwf_feat = {}
        if ecmwf is not None and vt in ecmwf.index:
            if "ecmwf_wind_speed_10m" in ecmwf.columns:
                ecmwf_feat["ecmwf_wspd"] = ecmwf.loc[vt, "ecmwf_wind_speed_10m"]
            if "ecmwf_wind_direction_10m" in ecmwf.columns:
                ecmwf_feat["ecmwf_wdir_sin"] = np.sin(np.deg2rad(ecmwf.loc[vt, "ecmwf_wind_direction_10m"]))
                ecmwf_feat["ecmwf_wdir_cos"] = np.cos(np.deg2rad(ecmwf.loc[vt, "ecmwf_wind_direction_10m"]))

        # Model spread features
        spreads = {}
        if "hrrr_wspd" in hrrr_feat and "gfs_wspd" in gfs_feat:
            spreads["hrrr_gfs_spread"] = hrrr_feat["hrrr_wspd"] - gfs_feat["gfs_wspd"]
        if "hrrr_wspd" in hrrr_feat and "ecmwf_wspd" in ecmwf_feat:
            spreads["hrrr_ecmwf_spread"] = hrrr_feat["hrrr_wspd"] - ecmwf_feat["ecmwf_wspd"]
        if "gfs_wspd" in gfs_feat and "ecmwf_wspd" in ecmwf_feat:
            spreads["gfs_ecmwf_spread"] = gfs_feat["gfs_wspd"] - ecmwf_feat["ecmwf_wspd"]
        # Model consensus (average of available models)
        model_wspds = [v for k, v in {**hrrr_feat, **gfs_feat, **ecmwf_feat}.items()
                       if k in ("hrrr_wspd", "gfs_wspd", "ecmwf_wspd") and not pd.isna(v)]
        if len(model_wspds) >= 2:
            spreads["model_consensus"] = np.mean(model_wspds)
            spreads["model_spread_std"] = np.std(model_wspds)

        # Combine all features
        all_feat = {**station_feat, **hrrr_feat, **gfs_feat, **ecmwf_feat, **spreads}
        all_feat["_vt"] = vt
        all_feat["_actual_wspd"] = unified.loc[vt, obs_col]
        rows.append(all_feat)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    valid_times_out = df["_vt"]
    actual_wspd = df["_actual_wspd"]
    df = df.drop(columns=["_vt", "_actual_wspd"])

    actual_wspd.index = valid_times_out.values
    df.index = valid_times_out.values

    return df, actual_wspd, pd.Series(dtype=float)


def train_l1_model(X_train, y_train, prefix_filter=None):
    """Train a Level 1 model, optionally filtering features by prefix."""
    if prefix_filter:
        # Only use features matching the prefix + station features
        cols = [c for c in X_train.columns if any(c.startswith(p) for p in prefix_filter)]
        if len(cols) == 0:
            return None, []
        X_sub = X_train[cols]
    else:
        X_sub = X_train
        cols = list(X_train.columns)

    model = GradientBoostingRegressor(
        n_estimators=500, max_depth=4, min_samples_leaf=8,
        learning_rate=0.02, subsample=0.8, random_state=42,
    )
    model.fit(X_sub.fillna(-999), y_train)
    return model, cols


def run_stacking_ensemble(
    target_station: str = "TPLM2",
    lead_hours: int = 12,
    test_start: str = "2025-10-01",
):
    """Run full stacking ensemble with regime models."""
    unified, hrrr_reg, gfs, ecmwf = load_all_data(target_station)

    print(f"Building samples for {target_station} at {lead_hours}h lead...")
    X, actual_wspd, _ = build_all_samples(
        unified, hrrr_reg, gfs, ecmwf, target_station, lead_hours
    )

    if len(X) < 50:
        print(f"  Insufficient data: {len(X)} samples")
        return None

    train_mask = X.index < test_start
    test_mask = X.index >= test_start

    if train_mask.sum() < 30 or test_mask.sum() < 10:
        print(f"  Insufficient split: {train_mask.sum()} train, {test_mask.sum()} test")
        return None

    X_train, y_train = X[train_mask], actual_wspd[train_mask]
    X_test, y_test = X[test_mask], actual_wspd[test_mask]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")

    # === Station feature columns (shared by all L1 models) ===
    station_cols = [c for c in X.columns if not c.startswith(("hrrr_", "gfs_", "ecmwf_", "regime_", "model_"))
                    and c not in ("hrrr_trend", "hrrr_gfs_spread", "hrrr_ecmwf_spread", "gfs_ecmwf_spread")]
    # Add back surrounding station HRRR (they're station-context, not model output)
    regional_hrrr_cols = [c for c in X.columns if c.endswith("_hrrr_wspd")]

    # === Level 1: Separate correction models ===
    print("\n  Level 1: Training specialist models...")

    # HRRR MOS: HRRR output + station obs
    hrrr_cols = [c for c in X.columns if c.startswith("hrrr_")] + regional_hrrr_cols + station_cols
    hrrr_cols = list(set(c for c in hrrr_cols if c in X.columns))
    hrrr_model, hrrr_feat_cols = train_l1_model(X_train[hrrr_cols], y_train)
    print(f"    HRRR MOS: {len(hrrr_feat_cols)} features")

    # GFS MOS: GFS output + station obs
    gfs_cols = [c for c in X.columns if c.startswith("gfs_")] + station_cols
    gfs_cols = list(set(c for c in gfs_cols if c in X.columns))
    gfs_model, gfs_feat_cols = train_l1_model(X_train[gfs_cols], y_train)
    print(f"    GFS MOS: {len(gfs_feat_cols)} features")

    # ECMWF MOS: ECMWF output + station obs
    ecmwf_cols = [c for c in X.columns if c.startswith("ecmwf_")] + station_cols
    ecmwf_cols = list(set(c for c in ecmwf_cols if c in X.columns))
    ecmwf_model, ecmwf_feat_cols = train_l1_model(X_train[ecmwf_cols], y_train)
    print(f"    ECMWF MOS: {len(ecmwf_feat_cols)} features")

    # Station-only model (no NWP)
    station_model, station_feat_cols = train_l1_model(X_train[station_cols], y_train)
    print(f"    Station-only: {len(station_feat_cols)} features")

    # === Generate L1 predictions (using CV to avoid leakage) ===
    print("  Generating L1 predictions via CV...")
    l1_train_preds = pd.DataFrame(index=X_train.index)
    l1_test_preds = pd.DataFrame(index=X_test.index)

    # For test set, use models trained on full training data
    l1_test_preds["l1_hrrr"] = hrrr_model.predict(X_test[hrrr_feat_cols].fillna(-999))
    l1_test_preds["l1_gfs"] = gfs_model.predict(X_test[gfs_feat_cols].fillna(-999)) if gfs_model else np.nan
    l1_test_preds["l1_ecmwf"] = ecmwf_model.predict(X_test[ecmwf_feat_cols].fillna(-999)) if ecmwf_model else np.nan
    l1_test_preds["l1_station"] = station_model.predict(X_test[station_feat_cols].fillna(-999))

    # For training set, use K-fold CV predictions to avoid leakage
    tscv = TimeSeriesSplit(n_splits=4)
    for col_name, model_class_cols in [
        ("l1_hrrr", hrrr_feat_cols),
        ("l1_gfs", gfs_feat_cols),
        ("l1_ecmwf", ecmwf_feat_cols),
        ("l1_station", station_feat_cols),
    ]:
        l1_train_preds[col_name] = np.nan
        for tr_idx, val_idx in tscv.split(X_train):
            m = GradientBoostingRegressor(
                n_estimators=500, max_depth=4, min_samples_leaf=8,
                learning_rate=0.02, subsample=0.8, random_state=42,
            )
            m.fit(X_train.iloc[tr_idx][model_class_cols].fillna(-999), y_train.iloc[tr_idx])
            l1_train_preds.iloc[val_idx, l1_train_preds.columns.get_loc(col_name)] = \
                m.predict(X_train.iloc[val_idx][model_class_cols].fillna(-999))

    # === Level 2: Blender ===
    print("  Level 2: Training blender...")

    # Blender features: L1 predictions + regime indicators + model spread
    meta_cols = ["regime_light", "regime_moderate", "regime_strong",
                 "model_consensus", "model_spread_std", "hrrr_trend",
                 "hrrr_gfs_spread", "hrrr_ecmwf_spread"]
    meta_cols = [c for c in meta_cols if c in X.columns]

    blend_train = pd.concat([l1_train_preds, X_train[meta_cols]], axis=1)
    blend_test = pd.concat([l1_test_preds, X_test[meta_cols]], axis=1)

    # Drop rows where CV didn't produce predictions (first fold)
    valid_blend = blend_train["l1_hrrr"].notna()
    blend_train = blend_train[valid_blend]
    y_blend_train = y_train[valid_blend]

    # Tune blender
    best_blend_mae = float("inf")
    best_blend_params = None
    for params in [
        {"n_estimators": 100, "max_depth": 3, "min_samples_leaf": 5, "learning_rate": 0.05},
        {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 8, "learning_rate": 0.03},
        {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 10, "learning_rate": 0.02},
        {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 8, "learning_rate": 0.03},
    ]:
        fold_maes = []
        for tr_idx, val_idx in TimeSeriesSplit(n_splits=3).split(blend_train):
            bm = GradientBoostingRegressor(random_state=42, subsample=0.8, **params)
            bm.fit(blend_train.iloc[tr_idx].fillna(-999), y_blend_train.iloc[tr_idx])
            bp = bm.predict(blend_train.iloc[val_idx].fillna(-999))
            fold_maes.append(mean_absolute_error(y_blend_train.iloc[val_idx], bp) * KT)
        cv_mae = np.mean(fold_maes)
        if cv_mae < best_blend_mae:
            best_blend_mae = cv_mae
            best_blend_params = params

    print(f"    Blender CV MAE: {best_blend_mae:.3f} kt, params: {best_blend_params}")

    blender = GradientBoostingRegressor(random_state=42, subsample=0.8, **best_blend_params)
    blender.fit(blend_train.fillna(-999), y_blend_train)
    stacked_pred = pd.Series(blender.predict(blend_test.fillna(-999)), index=X_test.index)

    # === Regime-specific models ===
    print("  Training regime-specific models...")
    regime_pred = pd.Series(np.nan, index=X_test.index)

    for regime_name, regime_col in [("light", "regime_light"), ("moderate", "regime_moderate"), ("strong", "regime_strong")]:
        if regime_col not in X.columns:
            continue
        r_train = X_train[regime_col] == 1.0
        r_test = X_test[regime_col] == 1.0

        if r_train.sum() < 20 or r_test.sum() < 3:
            continue

        rm = GradientBoostingRegressor(
            n_estimators=400, max_depth=4, min_samples_leaf=max(5, int(r_train.sum() * 0.05)),
            learning_rate=0.02, subsample=0.8, random_state=42,
        )
        rm.fit(X_train[r_train].fillna(-999), y_train[r_train])
        regime_pred[r_test] = rm.predict(X_test[r_test].fillna(-999))
        train_mae = mean_absolute_error(y_train[r_train], rm.predict(X_train[r_train].fillna(-999))) * KT
        print(f"    {regime_name}: {int(r_train.sum())} train, {int(r_test.sum())} test, train MAE={train_mae:.2f} kt")

    # Fill any gaps in regime predictions with stacked prediction
    regime_pred = regime_pred.fillna(stacked_pred)

    # === Also run the flat ensemble for comparison ===
    flat_model = GradientBoostingRegressor(
        n_estimators=800, max_depth=5, min_samples_leaf=8,
        learning_rate=0.02, subsample=0.8, random_state=42,
    )
    flat_model.fit(X_train.fillna(-999), y_train)
    flat_pred = pd.Series(flat_model.predict(X_test.fillna(-999)), index=X_test.index)

    # === Evaluate ===
    test_idx = X_test.index
    persist_times = test_idx - pd.Timedelta(hours=lead_hours)
    persist = unified[f"{target_station}_WSPD"].reindex(persist_times)
    persist.index = test_idx

    hrrr_raw = X_test["hrrr_wspd"] if "hrrr_wspd" in X_test.columns else None
    gfs_raw = X_test["gfs_wspd"] if "gfs_wspd" in X_test.columns else None
    ecmwf_raw = X_test["ecmwf_wspd"] if "ecmwf_wspd" in X_test.columns else None

    print(f"\n{'=' * 75}")
    print(f"STACKING ENSEMBLE: {target_station} at {lead_hours}h lead")
    print(f"Test: {test_start} onward ({test_mask.sum()} forecasts)")
    print(f"{'=' * 75}")

    print(f"\n  WIND SPEED:")
    print(f"  {'Model':<45s} {'MAE(kt)':>8s} {'Bias(kt)':>9s} {'RMSE(kt)':>9s}")
    print(f"  {'─' * 73}")

    models = [
        ("Persistence", persist),
    ]
    if hrrr_raw is not None:
        models.append((f"HRRR raw ({lead_hours}h)", hrrr_raw))
    if gfs_raw is not None:
        models.append((f"GFS raw ({lead_hours}h)", gfs_raw))
    if ecmwf_raw is not None:
        models.append(("ECMWF IFS (best avail)", ecmwf_raw))
    models += [
        ("L1: HRRR MOS alone", pd.Series(l1_test_preds["l1_hrrr"].values, index=test_idx)),
        ("L1: Station-only", pd.Series(l1_test_preds["l1_station"].values, index=test_idx)),
        ("Flat ensemble (prev best)", flat_pred),
        (">>> Stacked ensemble", stacked_pred),
        (">>> Regime-specific", regime_pred),
    ]

    for name, pred in models:
        if pred is None:
            continue
        v = y_test.notna() & pred.notna()
        if v.sum() < 5:
            continue
        a, p = y_test[v], pred[v]
        mae = mean_absolute_error(a, p) * KT
        bias = (p.mean() - a.mean()) * KT
        rmse = np.sqrt(mean_squared_error(a, p)) * KT
        print(f"  {name:<45s} {mae:8.2f} {bias:+9.2f} {rmse:9.2f}")

    # Blender feature importance
    imp = pd.Series(blender.feature_importances_, index=blend_test.columns).sort_values(ascending=False)
    print(f"\n  Blender weights:")
    for feat, v in imp.items():
        if v > 0.01:
            print(f"    {feat:30s} {v:.4f}")

    # Flat model top features
    imp_flat = pd.Series(flat_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n  Flat ensemble — top 15 features:")
    for feat, v in imp_flat.head(15).items():
        print(f"    {feat:40s} {v:.4f}")

    return blender, stacked_pred


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    stations = ["TPLM2", "APAM2", "SLIM2", "CAMM2"]
    leads = [3, 6, 12]

    for lead in leads:
        for station in stations:
            run_stacking_ensemble(target_station=station, lead_hours=lead, test_start="2025-10-01")
            print("\n")
