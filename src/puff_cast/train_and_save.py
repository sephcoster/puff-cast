"""
Train final models and save to disk for operational use.

Saves one speed model and one direction-correction model per
(station, lead_hours) pair, plus the feature column list.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from puff_cast.ensemble_mos import build_ensemble_features, load_ensemble_data

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"
KT = 1.944

SPEED_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 5,
    "min_samples_leaf": 5,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "random_state": 42,
}

# Direction models are noisier — use more conservative params
DIR_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "random_state": 42,
}

STATIONS = ["TPLM2", "APAM2", "SLIM2", "CAMM2"]
LEAD_HOURS = [1, 3, 6, 12, 18, 24]


def angular_difference(a: pd.Series, b: np.ndarray) -> pd.Series:
    """Signed angular difference in degrees, range [-180, 180]."""
    return (a - b + 180) % 360 - 180


def train_and_save_all():
    """Train speed + direction models for all station/lead combos."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for station in STATIONS:
        unified, hrrr_reg, gfs, ecmwf = load_ensemble_data(station)

        for lead in LEAD_HOURS:
            print(f"Training {station} {lead}h...")
            X, actual_wspd, actual_wdir = build_ensemble_features(
                unified, hrrr_reg, gfs, ecmwf, station, lead
            )

            if len(X) < 50:
                print(f"  Skipping — only {len(X)} samples")
                continue

            # Drop internal column
            if "_valid_time" in X.columns:
                X = X.drop(columns=["_valid_time"])

            X_filled = X.fillna(-999)

            # === Speed model ===
            speed_model = GradientBoostingRegressor(**SPEED_PARAMS)
            speed_model.fit(X_filled, actual_wspd)

            speed_path = MODEL_DIR / f"{station}_{lead}h.pkl"
            with open(speed_path, "wb") as f:
                pickle.dump(speed_model, f)

            # === Direction model (predicts angular error from HRRR direction) ===
            dir_trained = False
            dir_n = 0
            if (
                "hrrr_wdir_sin" in X.columns
                and "hrrr_wdir_cos" in X.columns
                and actual_wdir.notna().any()
            ):
                hrrr_wdir = (
                    np.rad2deg(
                        np.arctan2(X["hrrr_wdir_sin"], X["hrrr_wdir_cos"])
                    )
                    % 360
                )
                dir_error = angular_difference(actual_wdir, hrrr_wdir.values)
                # Only train on samples where wind is >5 kt (direction is noise-dominated below that)
                has_wind = (X["hrrr_wspd"] * KT >= 5).fillna(False) if "hrrr_wspd" in X.columns else pd.Series([False] * len(X), index=X.index)
                dir_mask = dir_error.notna() & has_wind
                if dir_mask.sum() >= 50:
                    dir_model = GradientBoostingRegressor(**DIR_PARAMS)
                    dir_model.fit(X_filled[dir_mask], dir_error[dir_mask])
                    dir_path = MODEL_DIR / f"{station}_{lead}h_dir.pkl"
                    with open(dir_path, "wb") as f:
                        pickle.dump(dir_model, f)
                    dir_trained = True
                    dir_n = int(dir_mask.sum())

            # Save metadata
            meta = {
                "station": station,
                "lead_hours": lead,
                "features": list(X.columns),
                "n_train": len(X),
                "n_dir_train": dir_n,
                "has_dir_model": dir_trained,
                "train_date_range": "all available data",
                "speed_params": {k: v for k, v in SPEED_PARAMS.items() if k != "random_state"},
                "dir_params": {k: v for k, v in DIR_PARAMS.items() if k != "random_state"} if dir_trained else None,
            }
            meta_path = MODEL_DIR / f"{station}_{lead}h.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            extras = f" + dir ({dir_n} samples)" if dir_trained else ""
            print(f"  Saved {speed_path.name} ({len(X)} samples, {len(X.columns)} features){extras}")

    print(f"\nAll models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    train_and_save_all()
