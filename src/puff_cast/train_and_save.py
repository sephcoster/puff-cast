"""
Train final models and save to disk for operational use.

Saves one model per (station, lead_hours) pair, plus the feature column list
so the forecast script knows what to build.
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

# Best hyperparams from CV tuning (per station, but using a good general set)
BEST_PARAMS = {
    "default": {
        "n_estimators": 1000,
        "max_depth": 5,
        "min_samples_leaf": 5,
        "learning_rate": 0.02,
        "subsample": 0.7,
        "random_state": 42,
    },
}

STATIONS = ["TPLM2", "APAM2", "SLIM2", "CAMM2"]
LEAD_HOURS = [1, 3, 6, 12]


def train_and_save_all():
    """Train models for all station/lead combos and save."""
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

            # Train on ALL data (no holdout — we've validated via CV)
            params = BEST_PARAMS.get(station, BEST_PARAMS["default"])
            model = GradientBoostingRegressor(**params)
            model.fit(X.fillna(-999), actual_wspd)

            # Save model
            model_path = MODEL_DIR / f"{station}_{lead}h.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save feature columns
            meta = {
                "station": station,
                "lead_hours": lead,
                "features": list(X.columns),
                "n_train": len(X),
                "train_date_range": f"all available data",
                "params": params,
            }
            meta_path = MODEL_DIR / f"{station}_{lead}h.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  Saved {model_path.name} ({len(X)} samples, {len(X.columns)} features)")

    print(f"\nAll models saved to {MODEL_DIR}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    train_and_save_all()
