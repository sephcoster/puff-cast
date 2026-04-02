"""
Fetch archived model forecasts at EXACT lead times from AWS via Herbie.

For each model initialization, extracts the N-hour forecast at Thomas Point.
This gives us apples-to-apples comparison: what did each model predict
for Thomas Point N hours before the valid time?

Models:
- HRRR: 3km, runs hourly, forecasts 0-18h (some runs to 48h)
- GFS:  0.25°, runs 4x/day (00/06/12/18Z), forecasts 0-384h
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# Thomas Point Lighthouse
TARGET_LAT = 38.899
TARGET_LON = -76.436
TARGET_LON_POS = 360 + TARGET_LON  # GRIB uses 0-360


def extract_wind_at_point(ds, lat=TARGET_LAT, lon_pos=TARGET_LON_POS):
    """Extract 10m wind speed and direction at nearest grid point.

    Handles both regular grids (GFS — 1D lat/lon) and Lambert Conformal (HRRR — 2D lat/lon).
    """
    lat_arr = ds.latitude.values
    lon_arr = ds.longitude.values

    if lat_arr.ndim == 1:
        # Regular grid (GFS) — use xarray sel
        u = float(ds["u10"].sel(latitude=lat, longitude=lon_pos, method="nearest").values)
        v = float(ds["v10"].sel(latitude=lat, longitude=lon_pos, method="nearest").values)
    else:
        # 2D grid (HRRR Lambert Conformal) — manual nearest neighbor
        dist = np.sqrt((lat_arr - lat) ** 2 + (lon_arr - lon_pos) ** 2)
        iy, ix = np.unravel_index(dist.argmin(), dist.shape)
        u = float(ds["u10"].values[iy, ix])
        v = float(ds["v10"].values[iy, ix])

    wspd = np.sqrt(u**2 + v**2)
    wdir = (270 - np.degrees(np.arctan2(v, u))) % 360

    return wspd, wdir


def fetch_hrrr_forecasts(
    start_date: str,
    end_date: str,
    lead_hours: list[int] = [3, 6, 12],
    init_hours: list[int] = [0, 6, 12, 18],
) -> pd.DataFrame:
    """
    Fetch HRRR forecasts at specific lead times.

    For each init time, downloads the forecast at each lead hour and extracts
    wind at Thomas Point. Only uses init_hours where all lead_hours are available
    (HRRR hourly runs only go to 18h; 00/06/12/18Z runs go to 48h).
    """
    from herbie import Herbie

    records = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    dates = pd.date_range(current, end, freq="1D")

    for date in tqdm(dates, desc="HRRR"):
        for init_h in init_hours:
            init_time = date + pd.Timedelta(hours=init_h)

            for fxx in lead_hours:
                valid_time = init_time + pd.Timedelta(hours=fxx)

                try:
                    H = Herbie(
                        init_time.strftime("%Y-%m-%d %H:%M"),
                        model="hrrr",
                        product="sfc",
                        fxx=fxx,
                        verbose=False,
                    )
                    ds = H.xarray(":(?:U|V)GRD:10 m above ground", verbose=False)
                    wspd, wdir = extract_wind_at_point(ds)

                    records.append(
                        {
                            "init_time": init_time,
                            "valid_time": valid_time,
                            "lead_hours": fxx,
                            "model": "hrrr",
                            "wspd_ms": wspd,
                            "wdir": wdir,
                        }
                    )
                except Exception as e:
                    logger.debug(f"HRRR {init_time} f{fxx:02d}: {e}")

    return pd.DataFrame(records)


def fetch_gfs_forecasts(
    start_date: str,
    end_date: str,
    lead_hours: list[int] = [3, 6, 12],
    init_hours: list[int] = [0, 6, 12, 18],
) -> pd.DataFrame:
    """Fetch GFS forecasts at specific lead times."""
    from herbie import Herbie

    records = []
    dates = pd.date_range(start_date, end_date, freq="1D")

    for date in tqdm(dates, desc="GFS"):
        for init_h in init_hours:
            init_time = date + pd.Timedelta(hours=init_h)

            for fxx in lead_hours:
                valid_time = init_time + pd.Timedelta(hours=fxx)

                try:
                    H = Herbie(
                        init_time.strftime("%Y-%m-%d %H:%M"),
                        model="gfs",
                        product="pgrb2.0p25",
                        fxx=fxx,
                        verbose=False,
                    )
                    ds = H.xarray(":(?:U|V)GRD:10 m above ground", verbose=False)
                    wspd, wdir = extract_wind_at_point(ds)

                    records.append(
                        {
                            "init_time": init_time,
                            "valid_time": valid_time,
                            "lead_hours": fxx,
                            "model": "gfs",
                            "wspd_ms": wspd,
                            "wdir": wdir,
                        }
                    )
                except Exception as e:
                    logger.debug(f"GFS {init_time} f{fxx:03d}: {e}")

    return pd.DataFrame(records)


def fetch_all_lead_time_forecasts(
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    lead_hours: list[int] = [3, 6, 12],
) -> pd.DataFrame:
    """Fetch forecasts from all models at specified lead times."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching HRRR forecasts ({start_date} to {end_date})...")
    hrrr = fetch_hrrr_forecasts(start_date, end_date, lead_hours)
    print(f"  HRRR: {len(hrrr)} records")

    print(f"Fetching GFS forecasts ({start_date} to {end_date})...")
    gfs = fetch_gfs_forecasts(start_date, end_date, lead_hours)
    print(f"  GFS: {len(gfs)} records")

    combined = pd.concat([hrrr, gfs], ignore_index=True)

    outpath = RAW_DIR / "lead_time_forecasts.csv"
    combined.to_csv(outpath, index=False)
    print(f"\nSaved {len(combined)} forecast records to {outpath}")

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_all_lead_time_forecasts()
