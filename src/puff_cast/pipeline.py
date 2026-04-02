"""
Data cleaning and alignment pipeline.

Takes raw per-station CSVs and produces a unified hourly DataFrame
with all stations aligned on the same time index.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from puff_cast.stations import ALL_STATIONS, Station

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Columns we keep from each station
KEEP_COLS = ["WDIR", "WSPD", "GST", "PRES", "ATMP", "WTMP", "DEWP", "PTDY"]


def load_raw(station: Station) -> pd.DataFrame:
    """Load raw CSV for a station, parse datetime index."""
    path = RAW_DIR / f"{station.id}.csv"
    df = pd.read_csv(path, index_col="time", parse_dates=True)
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to hourly frequency. Many stations report at 6-min intervals;
    we take the mean for continuous variables and circular mean for wind direction.
    """
    # Separate wind direction (needs circular averaging) from other columns
    other_cols = [c for c in df.columns if c != "WDIR"]

    parts = []

    # Regular mean for non-directional columns
    if other_cols:
        parts.append(df[other_cols].resample("1h").mean())

    # Circular mean for wind direction
    if "WDIR" in df.columns:
        wdir_rad = np.deg2rad(df["WDIR"].dropna())
        sin_mean = np.sin(wdir_rad).resample("1h").mean()
        cos_mean = np.cos(wdir_rad).resample("1h").mean()
        wdir_mean = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360
        wdir_df = wdir_mean.to_frame(name="WDIR")
        parts.append(wdir_df)

    result = pd.concat(parts, axis=1)
    # Reorder to match KEEP_COLS
    cols = [c for c in KEEP_COLS if c in result.columns]
    return result[cols]


def add_derived_features(df: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Add derived features useful for wind prediction."""
    out = df.copy()

    # Wind direction as sin/cos components (handles circular nature)
    if "WDIR" in out.columns:
        wdir_rad = np.deg2rad(out["WDIR"])
        out[f"WDIR_sin"] = np.sin(wdir_rad)
        out[f"WDIR_cos"] = np.cos(wdir_rad)

    # Pressure tendency (3-hour change) if not already available
    if "PRES" in out.columns and ("PTDY" not in out.columns or out["PTDY"].isna().all()):
        out["PTDY"] = out["PRES"].diff(periods=3)

    # Air-water temperature difference (drives convective activity)
    if "ATMP" in out.columns and "WTMP" in out.columns:
        out["TEMP_DIFF"] = out["ATMP"] - out["WTMP"]

    return out


def build_unified_dataset(start: str = "2020-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Build a single DataFrame with all stations on the same hourly time index.

    Columns are prefixed with station ID: e.g., TPLM2_WSPD, APAM2_PRES.
    Target columns (TPLM2) are also available without prefix for convenience.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if end is None:
        end = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Build the common time index
    time_index = pd.date_range(start=start, end=end, freq="1h")

    all_frames = {}

    for station in ALL_STATIONS:
        path = RAW_DIR / f"{station.id}.csv"
        if not path.exists():
            logger.warning(f"No data file for {station.id}, skipping")
            continue

        logger.info(f"Processing {station.id}...")
        df = load_raw(station)
        df = resample_to_hourly(df)
        df = add_derived_features(df, station.id)

        # Reindex to common time axis
        df = df.reindex(time_index)

        # Prefix columns with station ID
        df.columns = [f"{station.id}_{col}" for col in df.columns]
        all_frames[station.id] = df

        avail = (1 - df.isna().mean()).mean() * 100
        logger.info(f"  {station.id}: {avail:.0f}% data availability")

    # Merge all stations
    unified = pd.concat(all_frames.values(), axis=1)
    unified.index.name = "time"

    # Merge ASOS airport data if available
    asos_path = RAW_DIR / "asos_hourly.csv"
    if asos_path.exists():
        logger.info("Merging ASOS airport data...")
        asos = pd.read_csv(asos_path, index_col=0, parse_dates=True)
        # Strip timezone if present (match buoy data which is tz-naive UTC)
        if asos.index.tz is not None:
            asos.index = asos.index.tz_localize(None)
        asos = asos.reindex(time_index)
        unified = pd.concat([unified, asos], axis=1)
        asos_cols = len(asos.columns)
        asos_avail = (1 - asos.isna().mean()).mean() * 100
        logger.info(f"  ASOS: {asos_cols} columns, {asos_avail:.0f}% availability")

    # Merge CO-OPS tidal/water data if available
    coops_path = RAW_DIR / "coops_hourly.csv"
    if coops_path.exists():
        logger.info("Merging CO-OPS tidal/water data...")
        coops = pd.read_csv(coops_path, index_col=0, parse_dates=True)
        if coops.index.tz is not None:
            coops.index = coops.index.tz_localize(None)
        coops = coops.reindex(time_index)
        unified = pd.concat([unified, coops], axis=1)
        coops_cols = len(coops.columns)
        coops_avail = (1 - coops.isna().mean()).mean() * 100
        logger.info(f"  CO-OPS: {coops_cols} columns, {coops_avail:.0f}% availability")

    # Add time features
    unified["hour"] = unified.index.hour
    unified["month"] = unified.index.month
    unified["day_of_year"] = unified.index.dayofyear
    unified["hour_sin"] = np.sin(2 * np.pi * unified["hour"] / 24)
    unified["hour_cos"] = np.cos(2 * np.pi * unified["hour"] / 24)
    unified["month_sin"] = np.sin(2 * np.pi * unified["month"] / 12)
    unified["month_cos"] = np.cos(2 * np.pi * unified["month"] / 12)

    # Save
    outpath = PROCESSED_DIR / "unified_hourly.parquet"
    unified.to_parquet(outpath)
    logger.info(f"Saved unified dataset: {unified.shape} to {outpath}")

    return unified


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_unified_dataset()
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumn count by station:")
    from collections import Counter
    prefixes = Counter(c.split("_")[0] for c in df.columns if "_" in c)
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count} columns")
