"""
Fetch HRRR forecasts at ALL station locations from a single GRIB download.

For each HRRR initialization, downloads the wind field once and extracts
forecasts at all 13 station locations. Much more efficient than per-station
fetching, and gives us the regional error structure.
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


def extract_all_stations(ds) -> dict[str, tuple[float, float]]:
    """Extract 10m wind at all station locations from one GRIB dataset."""
    lat_arr = ds.latitude.values
    lon_arr = ds.longitude.values

    results = {}
    for station in ALL_STATIONS:
        lon_pos = 360 + station.lon if station.lon < 0 else station.lon

        if lat_arr.ndim == 1:
            # Regular grid
            u = float(ds["u10"].sel(latitude=station.lat, longitude=lon_pos, method="nearest").values)
            v = float(ds["v10"].sel(latitude=station.lat, longitude=lon_pos, method="nearest").values)
        else:
            # 2D grid (Lambert Conformal)
            dist = np.sqrt((lat_arr - station.lat) ** 2 + (lon_arr - lon_pos) ** 2)
            iy, ix = np.unravel_index(dist.argmin(), dist.shape)
            u = float(ds["u10"].values[iy, ix])
            v = float(ds["v10"].values[iy, ix])

        wspd = np.sqrt(u**2 + v**2)
        wdir = (270 - np.degrees(np.arctan2(v, u))) % 360
        results[station.id] = (wspd, wdir)

    return results


def fetch_hrrr_regional(
    start_date: str,
    end_date: str,
    lead_hours: list[int] = [3, 6, 12],
    init_hours: list[int] = [0, 12],
) -> pd.DataFrame:
    """
    Fetch HRRR forecasts at all station locations.

    Returns a DataFrame with columns:
        init_time, valid_time, lead_hours, station_id, wspd_ms, wdir
    """
    from herbie import Herbie

    records = []
    dates = pd.date_range(start_date, end_date, freq="1D")

    for date in tqdm(dates, desc="HRRR regional"):
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

                    # Extract all stations from this one download
                    station_winds = extract_all_stations(ds)

                    for station_id, (wspd, wdir) in station_winds.items():
                        records.append(
                            {
                                "init_time": init_time,
                                "valid_time": valid_time,
                                "lead_hours": fxx,
                                "station_id": station_id,
                                "hrrr_wspd_ms": wspd,
                                "hrrr_wdir": wdir,
                            }
                        )
                except Exception as e:
                    logger.debug(f"HRRR {init_time} f{fxx:02d}: {e}")

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Fetching HRRR regional forecasts...")
    df = fetch_hrrr_regional(
        start_date="2025-10-01",
        end_date="2025-12-31",
        lead_hours=[3, 6, 12],
        init_hours=[0, 12],
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outpath = RAW_DIR / "hrrr_regional.csv"
    df.to_csv(outpath, index=False)
    print(f"Saved {len(df)} records ({len(df) // 13} GRIB downloads) to {outpath}")
