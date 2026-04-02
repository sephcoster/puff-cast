"""
Fetch tidal current and water level data from NOAA CO-OPS API.

Tidal currents at Thomas Point affect surface roughness and local wind profiles.
When wind opposes strong current, chop steepens and effective drag changes,
modifying the near-surface wind profile that buoys measure.

Data source: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
"""

import logging
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

COOPS_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# Stations near Thomas Point / Annapolis
COOPS_STATIONS = {
    "8575512": {
        "name": "Annapolis (Naval Academy)",
        "products": ["water_level", "wind", "air_pressure", "water_temperature"],
    },
    "cb1102": {
        "name": "Bay Bridge Current Meter",
        "products": ["currents"],
    },
}

SESSION = requests.Session()


def fetch_coops_product(
    station: str,
    product: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch one product from CO-OPS API in monthly chunks (31-day limit for 6-min data).

    Parameters
    ----------
    station : str
        CO-OPS station ID (e.g., "8575512" or "cb1102")
    product : str
        Data product (water_level, wind, currents, air_pressure, water_temperature)
    start, end : str
        Date strings like "2025-01-01"
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    all_chunks = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + pd.Timedelta(days=30), end_dt)

        params = {
            "station": station,
            "begin_date": chunk_start.strftime("%Y%m%d"),
            "end_date": chunk_end.strftime("%Y%m%d"),
            "product": product,
            "units": "metric",
            "time_zone": "gmt",
            "format": "csv",
            "application": "puff_cast",
        }

        # Water level needs datum
        if product == "water_level":
            params["datum"] = "MLLW"

        try:
            resp = SESSION.get(COOPS_URL, params=params, timeout=30)
            resp.raise_for_status()

            # Check for API error messages
            if "Error" in resp.text[:100] or len(resp.text) < 50:
                logger.warning(f"  {station}/{product} {chunk_start.date()}: {resp.text[:100]}")
            else:
                df = pd.read_csv(StringIO(resp.text))
                if len(df) > 0:
                    all_chunks.append(df)

        except Exception as e:
            logger.warning(f"  {station}/{product} {chunk_start.date()}: {e}")

        chunk_start = chunk_end + pd.Timedelta(days=1)
        time.sleep(1)  # Be polite

    if not all_chunks:
        return pd.DataFrame()

    combined = pd.concat(all_chunks, ignore_index=True)
    return combined


def process_water_level(df: pd.DataFrame) -> pd.DataFrame:
    """Process water level data to hourly."""
    if len(df) == 0:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["Date Time"], utc=True)
    df = df.set_index("time").sort_index()

    hourly = pd.DataFrame()
    if " Water Level" in df.columns:
        hourly["water_level_m"] = df[" Water Level"].resample("1h").mean()
        # Rate of water level change (tide state)
        hourly["water_level_diff1"] = hourly["water_level_m"].diff(1)
        hourly["water_level_diff3"] = hourly["water_level_m"].diff(3)
    return hourly


def process_currents(df: pd.DataFrame) -> pd.DataFrame:
    """Process tidal current data to hourly."""
    if len(df) == 0:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["Date Time"], utc=True)
    df = df.set_index("time").sort_index()

    hourly = pd.DataFrame()
    if " Speed" in df.columns:
        hourly["current_speed_ms"] = df[" Speed"].resample("1h").mean()
    if " Direction" in df.columns:
        # Circular mean for current direction
        import numpy as np
        drct = df[" Direction"].dropna()
        rad = np.deg2rad(drct)
        sin_mean = np.sin(rad).resample("1h").mean()
        cos_mean = np.cos(rad).resample("1h").mean()
        hourly["current_dir"] = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
    return hourly


def process_wind(df: pd.DataFrame) -> pd.DataFrame:
    """Process CO-OPS wind data to hourly."""
    if len(df) == 0:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["Date Time"], utc=True)
    df = df.set_index("time").sort_index()

    hourly = pd.DataFrame()
    if " Speed" in df.columns:
        hourly["coops_wspd_ms"] = df[" Speed"].resample("1h").mean()
    if " Gust" in df.columns:
        hourly["coops_gust_ms"] = df[" Gust"].resample("1h").max()
    if " Direction" in df.columns:
        hourly["coops_wdir"] = df[" Direction"].resample("1h").last()
    return hourly


def process_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Process air pressure to hourly."""
    if len(df) == 0:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["Date Time"], utc=True)
    df = df.set_index("time").sort_index()

    hourly = pd.DataFrame()
    if " Pressure" in df.columns:
        hourly["coops_pres_mb"] = df[" Pressure"].resample("1h").mean()
    return hourly


def process_water_temp(df: pd.DataFrame) -> pd.DataFrame:
    """Process water temperature to hourly."""
    if len(df) == 0:
        return pd.DataFrame()

    df["time"] = pd.to_datetime(df["Date Time"], utc=True)
    df = df.set_index("time").sort_index()

    hourly = pd.DataFrame()
    if " Water Temperature" in df.columns:
        hourly["coops_wtmp_c"] = df[" Water Temperature"].resample("1h").mean()
    return hourly


PROCESSORS = {
    "water_level": process_water_level,
    "currents": process_currents,
    "wind": process_wind,
    "air_pressure": process_pressure,
    "water_temperature": process_water_temp,
}


def fetch_all_coops(
    start: str = "2025-01-01",
    end: str = "2026-04-01",
) -> pd.DataFrame:
    """Fetch all CO-OPS data and combine into hourly DataFrame."""
    all_hourly = []

    for station_id, info in COOPS_STATIONS.items():
        name = info["name"]
        for product in info["products"]:
            logger.info(f"Fetching {station_id} ({name}) / {product}...")
            raw = fetch_coops_product(station_id, product, start, end)

            if len(raw) == 0:
                logger.warning(f"  No data for {station_id}/{product}")
                continue

            processor = PROCESSORS.get(product)
            if processor:
                hourly = processor(raw)
                if len(hourly) > 0:
                    # Prefix columns with station context
                    prefix = f"COOPS_{station_id}_" if station_id != "cb1102" else "tidal_"
                    hourly.columns = [f"{prefix}{c}" for c in hourly.columns]
                    all_hourly.append(hourly)
                    logger.info(f"  {station_id}/{product}: {len(raw)} raw -> {len(hourly)} hourly")

    if not all_hourly:
        return pd.DataFrame()

    combined = pd.concat(all_hourly, axis=1).sort_index()
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Fetching NOAA CO-OPS tidal/water data...")
    df = fetch_all_coops(start="2025-01-01", end="2026-04-01")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RAW_DIR / "coops_hourly.csv"
    df.to_csv(outpath)

    print(f"\nSaved {len(df)} hours x {len(df.columns)} columns to {outpath}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns: {list(df.columns)}")

    coverage = df.notna().mean() * 100
    print(f"\nData coverage (% non-null):")
    for col, pct in coverage.items():
        print(f"  {col:40s} {pct:5.1f}%")
