"""
Download historical standard meteorological data from NDBC for all stations.

Two download strategies:
1. Historical yearly files (complete years): direct .txt.gz from NDBC
2. Recent 45-day realtime data: from NDBC realtime2 endpoint

Data is saved as one CSV per station in data/raw/.
"""

import gzip
import io
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from puff_cast.stations import ALL_STATIONS, Station

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# NDBC URLs
HISTORICAL_URL = "https://www.ndbc.noaa.gov/view_text_file.php?filename={station}h{year}.txt.gz&dir=data/historical/stdmet/"
REALTIME_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station}.txt"

# Column names for standard meteorological data (post-2007 format)
STDMET_COLS = [
    "YY", "MM", "DD", "hh", "mm",
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "PTDY", "TIDE",
]

# Columns we actually keep
KEEP_COLS = ["WDIR", "WSPD", "GST", "PRES", "ATMP", "WTMP", "DEWP", "PTDY"]

# Missing value sentinels used by NDBC
MISSING_VALUES = {
    "WDIR": [999],
    "WSPD": [99.0],
    "GST": [99.0],
    "WVHT": [99.0],
    "DPD": [99.0],
    "APD": [99.0],
    "MWD": [999],
    "PRES": [9999.0],
    "ATMP": [999.0, 99.0],
    "WTMP": [999.0, 99.0],
    "DEWP": [999.0, 99.0],
    "VIS": [99.0],
    "PTDY": [99.0],
    "TIDE": [99.0],
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "puff-cast/0.1 (marine-wind-research)"})


def fetch_year(station_id: str, year: int) -> pd.DataFrame | None:
    """Fetch one year of historical stdmet data for a station."""
    url = HISTORICAL_URL.format(station=station_id.lower(), year=year)
    resp = SESSION.get(url, timeout=30)

    if resp.status_code == 404 or "no data" in resp.text.lower()[:200]:
        return None
    resp.raise_for_status()

    # Parse the text data — skip comment lines starting with #
    lines = resp.text.strip().split("\n")
    header_lines = sum(1 for line in lines if line.startswith("#"))

    df = pd.read_csv(
        io.StringIO(resp.text),
        skiprows=header_lines,
        sep=r"\s+",
        header=None,
        na_values=["MM"],
    )

    # Handle varying column counts across years
    if df.shape[1] < len(STDMET_COLS):
        # Older files may lack PTDY and TIDE columns
        col_names = STDMET_COLS[: df.shape[1]]
    else:
        col_names = STDMET_COLS[: df.shape[1]]
    df.columns = col_names

    # Build datetime index
    # Handle 2-digit vs 4-digit year
    year_col = df["YY"]
    if (year_col < 100).all():
        year_col = year_col + 1900
        year_col = year_col.where(year_col >= 1950, year_col + 100)

    # Minutes column may not exist in very old data
    minutes = df["mm"] if "mm" in df.columns else 0

    df.index = pd.to_datetime(
        {
            "year": year_col,
            "month": df["MM"],
            "day": df["DD"],
            "hour": df["hh"],
            "minute": minutes,
        }
    )
    df.index.name = "time"

    # Keep only the columns we care about
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")

    # Replace NDBC missing value sentinels with NaN
    for col in df.columns:
        if col in MISSING_VALUES:
            for mv in MISSING_VALUES[col]:
                df[col] = df[col].replace(mv, pd.NA)

    return df


def fetch_realtime(station_id: str) -> pd.DataFrame | None:
    """Fetch recent ~45 days of realtime stdmet data."""
    url = REALTIME_URL.format(station=station_id.upper())
    resp = SESSION.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    header_lines = sum(1 for line in lines if line.startswith("#"))

    df = pd.read_csv(
        io.StringIO(resp.text),
        skiprows=header_lines,
        sep=r"\s+",
        header=None,
        na_values=["MM"],
    )

    col_names = STDMET_COLS[: df.shape[1]]
    df.columns = col_names

    df.index = pd.to_datetime(
        {
            "year": df["YY"],
            "month": df["MM"],
            "day": df["DD"],
            "hour": df["hh"],
            "minute": df["mm"],
        }
    )
    df.index.name = "time"

    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")

    for col in df.columns:
        if col in MISSING_VALUES:
            for mv in MISSING_VALUES[col]:
                df[col] = df[col].replace(mv, pd.NA)

    return df


def fetch_station(station: Station, start_year: int = 2020, end_year: int | None = None) -> Path:
    """
    Fetch all available data for a station from start_year to present.
    Saves a single CSV to data/raw/{station_id}.csv.
    Returns the path to the saved file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RAW_DIR / f"{station.id}.csv"

    if end_year is None:
        end_year = datetime.now().year

    frames = []

    # Historical yearly files
    for year in range(start_year, end_year + 1):
        try:
            df = fetch_year(station.id, year)
            if df is not None and len(df) > 0:
                frames.append(df)
                logger.info(f"  {station.id} {year}: {len(df)} records")
        except Exception as e:
            logger.warning(f"  {station.id} {year}: failed ({e})")

    # Also grab realtime for most recent data
    try:
        rt = fetch_realtime(station.id)
        if rt is not None and len(rt) > 0:
            frames.append(rt)
            logger.info(f"  {station.id} realtime: {len(rt)} records")
    except Exception as e:
        logger.warning(f"  {station.id} realtime: failed ({e})")

    if not frames:
        logger.warning(f"  {station.id}: NO DATA FOUND")
        return outpath

    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    combined.to_csv(outpath)
    logger.info(f"  {station.id}: saved {len(combined)} records to {outpath}")
    return outpath


def fetch_all(start_year: int = 2020, end_year: int | None = None) -> dict[str, Path]:
    """Fetch data for all stations. Returns dict of station_id -> file path."""
    results = {}
    for station in tqdm(ALL_STATIONS, desc="Fetching stations"):
        print(f"\n--- {station.id}: {station.name} ---")
        results[station.id] = fetch_station(station, start_year, end_year)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Default: fetch 2020-present (5+ years of data)
    fetch_all(start_year=2020)
