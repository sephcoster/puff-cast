"""
Fetch ASOS airport weather observations from Iowa Environmental Mesonet.

Airports near Chesapeake Bay provide higher-frequency observations and
land-side weather signals that buoys miss: visibility, ceiling, cloud cover,
precipitation type, and inland wind patterns that precede Bay conditions.

Data source: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Airports near Thomas Point / Chesapeake Bay
ASOS_STATIONS = {
    "KBWI": "Baltimore-Washington Intl",
    "KDCA": "Reagan National (DC)",
    "KESN": "Easton, MD",
    "KNHK": "Patuxent River NAS",
    "KSBY": "Salisbury, MD",
    "KDOV": "Dover AFB, DE",
    "KNAK": "Annapolis (Naval Academy)",
}

# Variables to request from IEM
# drct=direction, sknt=speed(kt), gust=gust(kt), mslp=sea-level pressure,
# tmpf=temp(F), dwpf=dewpoint(F), relh=RH%, vsby=visibility(mi),
# skyc1-3=cloud cover codes, skyl1-3=ceiling heights, p01i=1hr precip(in)
IEM_VARS = [
    "drct", "sknt", "gust", "mslp", "tmpf", "dwpf", "relh",
    "vsby", "skyc1", "skyl1", "skyc2", "skyl2", "p01i",
]

SESSION = requests.Session()


def fetch_asos_station(
    station: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Fetch ASOS data for one station from IEM.

    Parameters
    ----------
    station : str
        ICAO station ID (e.g., "KBWI")
    start, end : str
        Date strings like "2025-01-01"

    Returns
    -------
    DataFrame with datetime index and weather columns
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    params = {
        "station": station,
        "data": ",".join(IEM_VARS),
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "elev": "no",
        "missing": "empty",
        "trace": "0.0001",
        "report_type": "3",  # METAR + specials
        "year1": start_dt.year,
        "month1": start_dt.month,
        "day1": start_dt.day,
        "year2": end_dt.year,
        "month2": end_dt.month,
        "day2": end_dt.day,
    }

    resp = SESSION.get(IEM_URL, params=params, timeout=120)
    resp.raise_for_status()

    # IEM returns CSV with header
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text), low_memory=False)

    if len(df) == 0:
        logger.warning(f"No data returned for {station}")
        return pd.DataFrame()

    # Parse timestamp
    df["time"] = pd.to_datetime(df["valid"], utc=True)
    df = df.set_index("time").sort_index()

    # Convert types
    numeric_cols = ["drct", "sknt", "gust", "mslp", "tmpf", "dwpf", "relh", "vsby", "skyl1", "skyl2", "p01i"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert units to metric (matching NDBC conventions)
    # sknt is already in knots; convert to m/s for consistency with buoy data
    KT_TO_MS = 1 / 1.944
    if "sknt" in df.columns:
        df["wspd_ms"] = df["sknt"] * KT_TO_MS
    if "gust" in df.columns:
        df["gust_ms"] = df["gust"] * KT_TO_MS
    # Temperature F -> C
    if "tmpf" in df.columns:
        df["atmp_c"] = (df["tmpf"] - 32) * 5 / 9
    if "dwpf" in df.columns:
        df["dewp_c"] = (df["dwpf"] - 32) * 5 / 9
    # Visibility miles -> km
    if "vsby" in df.columns:
        df["vsby_km"] = df["vsby"] * 1.60934
    # Ceiling feet -> meters
    for col in ["skyl1", "skyl2"]:
        if col in df.columns:
            df[f"{col}_m"] = df[col] * 0.3048

    # Cloud cover codes to numeric (for ML)
    sky_map = {"CLR": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4}
    for col in ["skyc1", "skyc2"]:
        if col in df.columns:
            df[f"{col}_num"] = df[col].map(sky_map)

    return df


def resample_asos_hourly(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """Resample ASOS data to hourly to match buoy data cadence."""
    if len(df) == 0:
        return pd.DataFrame()

    # Select columns for resampling
    hourly_cols = {}

    # Wind: use last observation in the hour (closest to METAR top-of-hour)
    for col in ["wspd_ms", "gust_ms", "drct"]:
        if col in df.columns:
            hourly_cols[f"{station}_{col}"] = df[col].resample("1h").last()

    # Pressure, temp: mean
    for col in ["mslp", "atmp_c", "dewp_c", "relh"]:
        if col in df.columns:
            hourly_cols[f"{station}_{col}"] = df[col].resample("1h").mean()

    # Visibility: minimum (worst conditions in the hour)
    if "vsby_km" in df.columns:
        hourly_cols[f"{station}_vsby_km"] = df["vsby_km"].resample("1h").min()

    # Ceiling: minimum (lowest ceiling in the hour)
    if "skyl1_m" in df.columns:
        hourly_cols[f"{station}_ceil_m"] = df["skyl1_m"].resample("1h").min()

    # Cloud cover: maximum (worst coverage in the hour)
    if "skyc1_num" in df.columns:
        hourly_cols[f"{station}_cloud"] = df["skyc1_num"].resample("1h").max()

    # Precip: sum
    if "p01i" in df.columns:
        hourly_cols[f"{station}_precip_in"] = df["p01i"].resample("1h").sum()

    result = pd.DataFrame(hourly_cols)
    return result


def fetch_all_asos(
    start: str = "2025-01-01",
    end: str = "2026-04-01",
    stations: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Fetch ASOS data for all airport stations and return combined hourly DataFrame.

    Parameters
    ----------
    start, end : str
        Date range
    stations : dict, optional
        Override default ASOS_STATIONS

    Returns
    -------
    DataFrame indexed by UTC hour with columns like KBWI_wspd_ms, KDCA_mslp, etc.
    """
    if stations is None:
        stations = ASOS_STATIONS

    all_hourly = []

    for sid, name in tqdm(stations.items(), desc="ASOS stations"):
        logger.info(f"Fetching {sid} ({name})...")
        for attempt in range(4):
            try:
                raw = fetch_asos_station(sid, start, end)
                if len(raw) == 0:
                    logger.warning(f"No data for {sid}")
                    break

                hourly = resample_asos_hourly(raw, sid)
                all_hourly.append(hourly)
                logger.info(f"  {sid}: {len(raw)} raw obs -> {len(hourly)} hourly")

                # Be polite to IEM servers
                time.sleep(5)
                break

            except Exception as e:
                wait = 10 * (attempt + 1)
                logger.warning(f"  {sid} attempt {attempt+1} failed: {e}")
                if attempt < 3:
                    logger.info(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Failed to fetch {sid} after 4 attempts")

    if not all_hourly:
        return pd.DataFrame()

    combined = pd.concat(all_hourly, axis=1)
    combined = combined.sort_index()

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("Fetching ASOS airport data for Chesapeake Bay region...")
    df = fetch_all_asos(start="2025-01-01", end="2026-04-01")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RAW_DIR / "asos_hourly.csv"
    df.to_csv(outpath)

    print(f"\nSaved {len(df)} hours x {len(df.columns)} columns to {outpath}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns: {list(df.columns)}")

    # Quick data quality check
    print(f"\nData coverage (% non-null):")
    coverage = df.notna().mean() * 100
    for col, pct in coverage.items():
        print(f"  {col:30s} {pct:5.1f}%")
