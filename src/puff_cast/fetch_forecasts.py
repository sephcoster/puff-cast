"""
Fetch archived NWS model forecasts from Open-Meteo for Thomas Point coordinates.

We pull from two sources:
1. "best_match" — Open-Meteo's blend of best available models (≈ what NWS uses)
2. "ncep_gfs013" — raw GFS at 0.13° resolution (the backbone of NWS marine forecasts)

The historical-forecast-api returns the most recent model run's prediction for each hour,
which represents roughly a 1-6 hour lead time for GFS (runs every 6h) and shorter for
higher-frequency models.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

THOMAS_POINT = {"latitude": 38.899, "longitude": -76.436}

API_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"

HOURLY_VARS = "wind_speed_10m,wind_direction_10m,wind_gusts_10m,pressure_msl,temperature_2m"


def fetch_model_forecasts(
    model: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch archived model forecast data for Thomas Point from Open-Meteo.

    Pulls in 3-month chunks to stay within API limits.
    """
    all_frames = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current < end:
        chunk_end = min(current + pd.DateOffset(months=3), end)

        params = {
            **THOMAS_POINT,
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": (chunk_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "hourly": HOURLY_VARS,
            "wind_speed_unit": "ms",
            "timezone": "UTC",
            "models": model,
        }

        logger.info(f"  Fetching {model} {params['start_date']} to {params['end_date']}...")
        resp = requests.get(API_BASE, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            logger.warning(f"  No data for {model} {params['start_date']}: {data.get('reason', '')}")
            current = chunk_end
            continue

        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        all_frames.append(df)

        current = chunk_end
        time.sleep(0.5)  # be polite to the API

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Prefix columns with model name
    combined.columns = [f"{model}_{col}" for col in combined.columns]

    return combined


def fetch_all_forecasts(start_date: str = "2024-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """Fetch forecast data from multiple models and merge."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    models = ["best_match", "ncep_gfs013"]
    frames = []

    for model in models:
        print(f"--- Fetching {model} ---")
        df = fetch_model_forecasts(model, start_date, end_date)
        if len(df) > 0:
            frames.append(df)
            print(f"  {model}: {len(df)} hours")

    if not frames:
        raise RuntimeError("No forecast data retrieved")

    merged = pd.concat(frames, axis=1)
    merged.index.name = "time"

    outpath = RAW_DIR / "model_forecasts.csv"
    merged.to_csv(outpath)
    print(f"\nSaved {len(merged)} hours of forecast data to {outpath}")

    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_all_forecasts()
