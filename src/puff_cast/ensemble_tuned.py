"""
Tuned ensemble MOS with time-series cross-validation and hyperparameter search.

Improvements over ensemble_mos.py:
1. Time-series CV (expanding window) instead of single train/test split
2. Hyperparameter grid search via CV
3. More estimators and regularization options
4. Reports CV variance to detect overfitting
5. Learning curves to show if more data would help
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


def load_ensemble_data(target_station: str = "TPLM2"):
    unified = pd.read_parquet(DATA_DIR / "processed" / "unified_hourly.parquet")
    # Prefer enhanced HRRR (4 inits + extra vars) if available
    enhanced_path = DATA_DIR / "raw" / "hrrr_enhanced.csv"
    fallback_path = DATA_DIR / "raw" / "hrrr_regional_full.csv"
    hrrr_path = enhanced_path if enhanced_path.exists() else fallback_path
    hrrr_reg = pd.read_csv(
        hrrr_path,
        parse_dates=["init_time", "valid_time"],
    )
    gfs = pd.read_csv(
        DATA_DIR / "raw" / "gfs_lead_time.csv",
        parse_dates=["init_time", "valid_time"],
    )
    ecmwf_path = DATA_DIR / "raw" / "ecmwf_openmeteo.csv"
    ecmwf = pd.read_csv(ecmwf_path, index_col="time", parse_dates=True) if ecmwf_path.exists() else None
    return unified, hrrr_reg, gfs, ecmwf


def build_ensemble_features(unified, hrrr_reg, gfs, ecmwf, target_station, lead_hours):
    """Same feature builder as ensemble_mos.py — reused here."""
    from puff_cast.ensemble_mos import build_ensemble_features as _build
    return _build(unified, hrrr_reg, gfs, ecmwf, target_station, lead_hours)


def run_tuned_ensemble(
    target_station: str = "TPLM2",
    lead_hours: int = 12,
    test_start: str = "2025-10-01",
):
    """Run ensemble with hyperparameter tuning via time-series CV."""
    unified, hrrr_reg, gfs, ecmwf = load_ensemble_data(target_station)

    print(f"Building features for {target_station} at {lead_hours}h lead...")
    X, actual_wspd, actual_wdir = build_ensemble_features(
        unified, hrrr_reg, gfs, ecmwf, target_station, lead_hours
    )

    if len(X) == 0:
        print("  No valid samples!")
        return None

    # Split by valid_time (may be stored as _valid_time column or as index)
    if "_valid_time" in X.columns:
        vt_series = pd.to_datetime(X["_valid_time"])
        train_mask = vt_series < test_start
        test_mask = vt_series >= test_start
        X = X.drop(columns=["_valid_time"])
    else:
        train_mask = X.index < test_start
        test_mask = X.index >= test_start

    X_train = X[train_mask].fillna(-999)
    y_train = actual_wspd[train_mask]
    X_test = X[test_mask].fillna(-999)
    y_test = actual_wspd[test_mask]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")

    # === Hyperparameter grid ===
    param_grid = [
        {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 5, "learning_rate": 0.05, "subsample": 1.0},
        {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 5, "learning_rate": 0.05, "subsample": 1.0},
        {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 5, "learning_rate": 0.03, "subsample": 1.0},
        {"n_estimators": 500, "max_depth": 6, "min_samples_leaf": 8, "learning_rate": 0.03, "subsample": 0.8},
        {"n_estimators": 800, "max_depth": 4, "min_samples_leaf": 5, "learning_rate": 0.02, "subsample": 0.8},
        {"n_estimators": 800, "max_depth": 5, "min_samples_leaf": 8, "learning_rate": 0.02, "subsample": 0.8},
        {"n_estimators": 1000, "max_depth": 4, "min_samples_leaf": 10, "learning_rate": 0.01, "subsample": 0.8},
        {"n_estimators": 1000, "max_depth": 5, "min_samples_leaf": 5, "learning_rate": 0.02, "subsample": 0.7},
        {"n_estimators": 1500, "max_depth": 4, "min_samples_leaf": 8, "learning_rate": 0.01, "subsample": 0.8},
    ]

    # === Time-series cross-validation ===
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\n  Time-series CV ({n_splits} folds):")
    print(f"  {'Config':<55s} {'CV MAE(kt)':>10s} {'± Std':>8s}")
    print(f"  {'─' * 75}")

    best_cv_mae = float("inf")
    best_params = None
    best_cv_std = None

    for params in param_grid:
        fold_maes = []
        for train_idx, val_idx in tscv.split(X_train):
            model = GradientBoostingRegressor(random_state=42, **params)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_train.iloc[val_idx])
            mae = mean_absolute_error(y_train.iloc[val_idx], pred) * KT
            fold_maes.append(mae)

        cv_mae = np.mean(fold_maes)
        cv_std = np.std(fold_maes)

        desc = f"n={params['n_estimators']} d={params['max_depth']} lr={params['learning_rate']} ss={params['subsample']}"
        print(f"  {desc:<55s} {cv_mae:10.3f} {cv_std:7.3f}")

        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_params = params
            best_cv_std = cv_std

    print(f"\n  >>> Best CV: {best_cv_mae:.3f} ± {best_cv_std:.3f} kt")
    print(f"  >>> Params: {best_params}")

    # === Train final model with best params on all training data ===
    final_model = GradientBoostingRegressor(random_state=42, **best_params)
    final_model.fit(X_train, y_train)
    ensemble_pred = pd.Series(final_model.predict(X_test), index=X_test.index)

    # Also train baseline (our previous default params)
    baseline_params = {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 5, "learning_rate": 0.05, "subsample": 1.0}
    baseline_model = GradientBoostingRegressor(random_state=42, **baseline_params)
    baseline_model.fit(X_train, y_train)
    baseline_pred = pd.Series(baseline_model.predict(X_test), index=X_test.index)

    # === Learning curve: does more training data help? ===
    print(f"\n  Learning curve (best params, test MAE as training set grows):")
    fractions = [0.25, 0.50, 0.75, 1.0]
    for frac in fractions:
        n = int(len(X_train) * frac)
        if n < 30:
            continue
        m = GradientBoostingRegressor(random_state=42, **best_params)
        m.fit(X_train.iloc[:n], y_train.iloc[:n])
        train_pred = m.predict(X_train.iloc[:n])
        train_mae = mean_absolute_error(y_train.iloc[:n], train_pred) * KT
        test_pred = m.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_pred) * KT
        gap = test_mae - train_mae
        print(f"    {frac*100:5.0f}% ({n:4d} samples): train={train_mae:.3f}  test={test_mae:.3f}  gap={gap:.3f} kt")

    # === Direction model with tuning ===
    dir_model = None
    obs_wdir_col = f"{target_station}_WDIR"
    has_wdir = obs_wdir_col in unified.columns and actual_wdir.notna().sum() > 0

    if has_wdir and "hrrr_wdir_sin" in X.columns:
        hrrr_wdir = np.rad2deg(np.arctan2(X["hrrr_wdir_sin"], X["hrrr_wdir_cos"])) % 360
        dir_error = angular_difference(actual_wdir, hrrr_wdir)
        dir_valid = dir_error.notna() & (X.get("hrrr_wspd", pd.Series(dtype=float)) * KT >= 5).fillna(False)
        dir_train = train_mask & dir_valid
        dir_test = test_mask & dir_valid

        if dir_train.sum() >= 20 and dir_test.sum() >= 5:
            # Use more conservative params for direction (smaller target, more noise)
            dir_params = [
                {"n_estimators": 200, "max_depth": 3, "min_samples_leaf": 8, "learning_rate": 0.03, "subsample": 0.8},
                {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 8, "learning_rate": 0.03, "subsample": 0.8},
                {"n_estimators": 500, "max_depth": 3, "min_samples_leaf": 10, "learning_rate": 0.02, "subsample": 0.8},
                {"n_estimators": 500, "max_depth": 4, "min_samples_leaf": 10, "learning_rate": 0.02, "subsample": 0.7},
            ]

            X_dir_train = X[dir_train].fillna(-999)
            y_dir_train = dir_error[dir_train]

            best_dir_mae = float("inf")
            best_dir_params = dir_params[0]

            for dp in dir_params:
                fold_maes = []
                tscv_dir = TimeSeriesSplit(n_splits=min(3, max(2, len(X_dir_train) // 40)))
                for ti, vi in tscv_dir.split(X_dir_train):
                    dm = GradientBoostingRegressor(random_state=42, **dp)
                    dm.fit(X_dir_train.iloc[ti], y_dir_train.iloc[ti])
                    dp_pred = dm.predict(X_dir_train.iloc[vi])
                    fold_maes.append(np.abs(dp_pred - y_dir_train.iloc[vi].values).mean())
                cv_dir = np.mean(fold_maes)
                if cv_dir < best_dir_mae:
                    best_dir_mae = cv_dir
                    best_dir_params = dp

            dir_model = GradientBoostingRegressor(random_state=42, **best_dir_params)
            dir_model.fit(X_dir_train, y_dir_train)
            print(f"\n  Direction model params: {best_dir_params}")

    # === Final evaluation ===
    # For baselines, use HRRR/GFS/ECMWF values from the test features directly
    hrrr_raw = X_test["hrrr_wspd"] if "hrrr_wspd" in X_test.columns else None
    gfs_raw = X_test["gfs_wspd"] if "gfs_wspd" in X_test.columns else None
    ecmwf_raw = X_test["ecmwf_wspd"] if "ecmwf_wspd" in X_test.columns else None

    print(f"\n{'=' * 75}")
    print(f"TUNED ENSEMBLE: {target_station} at {lead_hours}h lead")
    print(f"Test: {test_start} onward ({test_mask.sum()} forecasts)")
    print(f"{'=' * 75}")

    print(f"\n  WIND SPEED:")
    print(f"  {'Model':<40s} {'MAE(kt)':>8s} {'Bias(kt)':>9s} {'RMSE(kt)':>9s}")
    print(f"  {'─' * 68}")

    models = []
    if hrrr_raw is not None:
        models.append((f"HRRR raw ({lead_hours}h)", hrrr_raw))
    if gfs_raw is not None:
        models.append((f"GFS raw ({lead_hours}h)", gfs_raw))
    if ecmwf_raw is not None:
        models.append(("ECMWF IFS (best avail)", ecmwf_raw))
    models.append(("Ensemble (prev params)", baseline_pred))
    models.append((">>> Ensemble (tuned)", ensemble_pred))

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
        print(f"  {name:<40s} {mae:8.2f} {bias:+9.2f} {rmse:9.2f}")

    # Direction
    if dir_model is not None:
        print(f"\n  WIND DIRECTION (>5kt only):")
        print(f"  {'Model':<40s} {'MAE(°)':>8s} {'Bias(°)':>9s}")
        print(f"  {'─' * 60}")

        hrrr_wdir_test = np.rad2deg(np.arctan2(
            X_test["hrrr_wdir_sin"], X_test["hrrr_wdir_cos"]
        )) % 360
        actual_dir_test = actual_wdir[test_mask]
        dir_valid_test = actual_dir_test.notna() & (X_test["hrrr_wspd"] * KT >= 5)

        if dir_valid_test.sum() >= 5:
            ad = actual_dir_test[dir_valid_test].values
            hd = hrrr_wdir_test[dir_valid_test].values

            diff = angular_difference(ad, hd)
            print(f"  {f'HRRR raw ({lead_hours}h)':<40s} {np.abs(diff).mean():8.1f} {diff.mean():+9.1f}")

            dir_correction = dir_model.predict(X_test[dir_valid_test].fillna(-999))
            corrected_dir = (hd + dir_correction) % 360
            diff_e = angular_difference(ad, corrected_dir)
            print(f"  {'>>> Ensemble (tuned)':<40s} {np.abs(diff_e).mean():8.1f} {diff_e.mean():+9.1f}")

    # Feature importance
    imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\n  Top 15 features:")
    for feat, v in imp.head(15).items():
        print(f"    {feat:40s} {v:.4f}")

    return final_model, dir_model, best_params


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    stations = ["TPLM2", "APAM2", "SLIM2", "CAMM2"]
    leads = [3, 6, 12]

    for lead in leads:
        for station in stations:
            run_tuned_ensemble(target_station=station, lead_hours=lead, test_start="2025-10-01")
            print("\n")
