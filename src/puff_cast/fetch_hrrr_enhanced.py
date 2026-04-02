"""
Enhanced HRRR fetch: more init times + additional atmospheric variables.

Improvements over fetch_hrrr_regional.py:
1. Four init times per day (00/06/12/18Z) instead of two (00/12Z)
2. Additional variables: gusts, CAPE, boundary layer height, surface pressure, friction velocity
3. More lead times to capture HRRR trend (successive forecasts for same valid time)

Each GRIB download is ~2-4MB via byte-range requests. With 4 inits × 4 leads × 365 days
= ~5800 downloads for a full year — feasible but takes a few hours.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from puff_cast.stations import ALL_STATIONS

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def extract_wind_all_stations(ds) -> dict[str, tuple[float, float]]:
    """Extract 10m wind at all station locations from one GRIB dataset."""
    lat_arr = ds.latitude.values
    lon_arr = ds.longitude.values

    results = {}
    for station in ALL_STATIONS:
        lon_pos = 360 + station.lon if station.lon < 0 else station.lon

        if lat_arr.ndim == 1:
            u = float(ds["u10"].sel(latitude=station.lat, longitude=lon_pos, method="nearest").values)
            v = float(ds["v10"].sel(latitude=station.lat, longitude=lon_pos, method="nearest").values)
        else:
            dist = np.sqrt((lat_arr - station.lat) ** 2 + (lon_arr - lon_pos) ** 2)
            iy, ix = np.unravel_index(dist.argmin(), dist.shape)
            u = float(ds["u10"].values[iy, ix])
            v = float(ds["v10"].values[iy, ix])

        wspd = np.sqrt(u**2 + v**2)
        wdir = (270 - np.degrees(np.arctan2(v, u))) % 360
        results[station.id] = (wspd, wdir)

    return results


def extract_scalar_all_stations(ds, var_name: str) -> dict[str, float]:
    """Extract a scalar field at all station locations."""
    lat_arr = ds.latitude.values
    lon_arr = ds.longitude.values

    # Find the data variable (Herbie names can vary)
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        return {}
    dvar = data_vars[0]  # Usually just one variable per search

    results = {}
    for station in ALL_STATIONS:
        lon_pos = 360 + station.lon if station.lon < 0 else station.lon

        try:
            if lat_arr.ndim == 1:
                val = float(ds[dvar].sel(latitude=station.lat, longitude=lon_pos, method="nearest").values)
            else:
                dist = np.sqrt((lat_arr - station.lat) ** 2 + (lon_arr - lon_pos) ** 2)
                iy, ix = np.unravel_index(dist.argmin(), dist.shape)
                val = float(ds[dvar].values[iy, ix])
            results[station.id] = val
        except Exception:
            results[station.id] = np.nan

    return results


# GRIB search strings for additional variables
EXTRA_VARIABLES = {
    "gust_ms": ":GUST:surface",
    "cape_jkg": ":CAPE:surface",
    "pbl_m": ":HPBL:surface",
    "sp_pa": ":PRES:surface",
    "fricv_ms": ":FRICV:surface",
}


def fetch_hrrr_enhanced(
    start_date: str,
    end_date: str,
    lead_hours: list[int] = [3, 6, 9, 12],
    init_hours: list[int] = [0, 6, 12, 18],
) -> pd.DataFrame:
    """
    Fetch HRRR forecasts with additional variables at all station locations.

    Returns DataFrame with columns:
        init_time, valid_time, lead_hours, station_id,
        hrrr_wspd_ms, hrrr_wdir, hrrr_gust_ms, hrrr_cape_jkg,
        hrrr_pbl_m, hrrr_sp_pa, hrrr_fricv_ms
    """
    from herbie import Herbie

    # Resume support: load existing partial results
    outpath = RAW_DIR / "hrrr_enhanced.csv"
    existing_keys = set()
    records = []
    if outpath.exists():
        existing = pd.read_csv(outpath, parse_dates=["init_time", "valid_time"])
        records = existing.to_dict("records")
        # Track what we already have: (init_time, lead_hours, station_id)
        for _, row in existing.iterrows():
            existing_keys.add((str(row["init_time"]), int(row["lead_hours"])))
        logger.info(f"Resuming: {len(existing_keys)} init/lead combos already fetched")

    dates = pd.date_range(start_date, end_date, freq="1D")
    total = len(dates) * len(init_hours) * len(lead_hours)
    pbar = tqdm(total=total, desc="HRRR enhanced")
    new_count = 0

    for date in dates:
        for init_h in init_hours:
            init_time = date + pd.Timedelta(hours=init_h)

            for fxx in lead_hours:
                valid_time = init_time + pd.Timedelta(hours=fxx)
                pbar.update(1)

                # Skip if already fetched
                key = (str(init_time), fxx)
                if key in existing_keys:
                    continue

                try:
                    H = Herbie(
                        init_time.strftime("%Y-%m-%d %H:%M"),
                        model="hrrr",
                        product="sfc",
                        fxx=fxx,
                        verbose=False,
                    )

                    # Wind (u/v components)
                    ds_wind = H.xarray(":(?:U|V)GRD:10 m above ground", verbose=False)
                    station_winds = extract_wind_all_stations(ds_wind)

                    # Extract additional variables
                    extra_data = {}
                    for var_name, search_str in EXTRA_VARIABLES.items():
                        try:
                            ds_var = H.xarray(search_str, verbose=False)
                            extra_data[var_name] = extract_scalar_all_stations(ds_var, var_name)
                        except Exception as e:
                            logger.debug(f"  {var_name} not available for {init_time} f{fxx:02d}: {e}")
                            extra_data[var_name] = {}

                    for station_id, (wspd, wdir) in station_winds.items():
                        rec = {
                            "init_time": init_time,
                            "valid_time": valid_time,
                            "lead_hours": fxx,
                            "station_id": station_id,
                            "hrrr_wspd_ms": wspd,
                            "hrrr_wdir": wdir,
                        }
                        # Add extra variables
                        for var_name in EXTRA_VARIABLES:
                            rec[f"hrrr_{var_name}"] = extra_data.get(var_name, {}).get(station_id, np.nan)

                        records.append(rec)

                    new_count += 1

                    # Save checkpoint every 50 successful downloads
                    if new_count % 50 == 0:
                        df_save = pd.DataFrame(records)
                        df_save.to_csv(outpath, index=False)
                        logger.info(f"  Checkpoint: {len(records)} total records saved")

                except Exception as e:
                    logger.debug(f"HRRR {init_time} f{fxx:02d}: {e}")

    pbar.close()
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Fetching enhanced HRRR regional forecasts (4 inits/day + extra vars)...")
    df = fetch_hrrr_enhanced(
        start_date="2025-01-01",
        end_date="2026-03-31",
        lead_hours=[3, 6, 9, 12],
        init_hours=[0, 6, 12, 18],
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RAW_DIR / "hrrr_enhanced.csv"
    df.to_csv(outpath, index=False)
    n_downloads = len(df) // 13 if len(df) > 0 else 0
    print(f"Saved {len(df)} records ({n_downloads} GRIB downloads) to {outpath}")
