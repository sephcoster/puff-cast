"""
Research findings: Getting archived weather model forecasts at specific lead times
for Thomas Point (38.899N, 76.436W) in Chesapeake Bay.

WINNER: AWS S3 GRIB archives with byte-range downloads (idx files)
RUNNER-UP: Open-Meteo Historical Forecast API (easy but NO lead-time control)
"""

import struct
import urllib.request
import json
from datetime import datetime, timedelta


# =============================================================================
# APPROACH 1 (BEST): AWS S3 byte-range GRIB extraction
# =============================================================================
#
# Pros:
#   - Exact control over initialization time AND forecast hour (lead time)
#   - HRRR: hourly inits, f00-f18 (all runs), f00-f48 (00/06/12/18Z runs)
#   - GFS: 4x/day inits (00/06/12/18Z), f000-f384
#   - Both archives go back to at least 2021
#   - FREE, no API key, no rate limits
#   - Byte-range download: ~2MB per variable instead of ~142MB full file
#
# Cons:
#   - Need GRIB2 parsing (eccodes/cfgrib/wgrib2) to extract point values
#   - Need to compute wind speed from U/V components
#   - More complex code than an API call
#
# File patterns:
#   HRRR: s3://noaa-hrrr-bdp-pds/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfsfcfFF.grib2
#   GFS:  s3://noaa-gfs-bdp-pds/gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
#
# Example: To get the 12-hour-ahead forecast valid at 2025-03-01 12:00Z:
#   - HRRR init 00Z: hrrr.20250301/conus/hrrr.t00z.wrfsfcf12.grib2
#   - GFS  init 00Z: gfs.20250301/00/atmos/gfs.t00z.pgrb2.0p25.f012


def get_hrrr_idx_url(date_str, init_hour, fhour):
    """Get the .idx URL for a HRRR forecast file."""
    return (f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{date_str}/conus/hrrr.t{init_hour:02d}z.wrfsfcf{fhour:02d}.grib2.idx")

def get_hrrr_grib_url(date_str, init_hour, fhour):
    """Get the GRIB2 URL for a HRRR forecast file."""
    return (f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
            f"hrrr.{date_str}/conus/hrrr.t{init_hour:02d}z.wrfsfcf{fhour:02d}.grib2")

def get_gfs_idx_url(date_str, init_hour, fhour):
    """Get the .idx URL for a GFS forecast file."""
    return (f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
            f"gfs.{date_str}/{init_hour:02d}/atmos/"
            f"gfs.t{init_hour:02d}z.pgrb2.0p25.f{fhour:03d}.idx")

def get_gfs_grib_url(date_str, init_hour, fhour):
    """Get the GRIB2 URL for a GFS forecast file."""
    return (f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
            f"gfs.{date_str}/{init_hour:02d}/atmos/"
            f"gfs.t{init_hour:02d}z.pgrb2.0p25.f{fhour:03d}")


def parse_idx(idx_url, variable, level):
    """
    Parse a GRIB2 .idx file to find byte ranges for a specific variable/level.
    Returns (start_byte, end_byte) or None.
    
    idx format: line_num:byte_offset:d=YYYYMMDDHH:VAR:level:forecast_info:
    """
    req = urllib.request.Request(idx_url)
    with urllib.request.urlopen(req) as resp:
        lines = resp.read().decode().strip().split('\n')
    
    for i, line in enumerate(lines):
        parts = line.split(':')
        if len(parts) >= 5 and parts[3] == variable and parts[4] == level:
            start_byte = int(parts[1])
            # End byte is start of next record minus 1
            if i + 1 < len(lines):
                next_parts = lines[i + 1].split(':')
                end_byte = int(next_parts[1]) - 1
            else:
                end_byte = ''  # Last record: download to end of file
            return start_byte, end_byte
    return None


def download_grib_subset(grib_url, start_byte, end_byte):
    """Download a byte range from a GRIB2 file."""
    range_str = f"{start_byte}-{end_byte}" if end_byte else f"{start_byte}-"
    req = urllib.request.Request(grib_url, headers={'Range': f'bytes={range_str}'})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def demo_byte_range_download():
    """
    Demo: download 10m UGRD and VGRD from HRRR 00Z init, 12h forecast, 2025-03-01.
    This downloads ~2MB per component instead of the full 142MB file.
    """
    date_str = "20250301"
    init_hour = 0
    fhour = 12  # 12-hour lead time
    
    idx_url = get_hrrr_idx_url(date_str, init_hour, fhour)
    grib_url = get_hrrr_grib_url(date_str, init_hour, fhour)
    
    print(f"IDX URL: {idx_url}")
    print(f"GRIB URL: {grib_url}")
    
    # Find byte ranges for 10m wind components
    ugrd_range = parse_idx(idx_url, "UGRD", "10 m above ground")
    vgrd_range = parse_idx(idx_url, "VGRD", "10 m above ground")
    
    if ugrd_range:
        print(f"UGRD 10m byte range: {ugrd_range[0]}-{ugrd_range[1]}")
        data = download_grib_subset(grib_url, ugrd_range[0], ugrd_range[1])
        print(f"Downloaded UGRD: {len(data)} bytes ({len(data)/1024:.0f} KB)")
    
    if vgrd_range:
        print(f"VGRD 10m byte range: {vgrd_range[0]}-{vgrd_range[1]}")
        data = download_grib_subset(grib_url, vgrd_range[0], vgrd_range[1])
        print(f"Downloaded VGRD: {len(data)} bytes ({len(data)/1024:.0f} KB)")
    
    print("\nTo extract the point value at 38.899N, 76.436W, use one of:")
    print("  - pip install eccodes cfgrib xarray  (then xarray.open_dataset)")
    print("  - pip install herbie-data  (wraps all of this; see below)")
    print("  - wgrib2 (command line): wgrib2 file.grib2 -lon -76.436 38.899")


# =============================================================================
# APPROACH 1b: Using the Herbie library (recommended wrapper for Approach 1)
# =============================================================================
#
# pip install herbie-data
# Herbie automates idx parsing, byte-range downloads, and point extraction.

HERBIE_EXAMPLE = """
from herbie import Herbie
import pandas as pd
from datetime import datetime, timedelta

lat, lon = 38.899, -76.436

def get_forecast_at_lead_time(valid_time, lead_hours, model='hrrr'):
    '''Get the wind forecast that was MADE lead_hours before valid_time.'''
    init_time = valid_time - timedelta(hours=lead_hours)
    
    if model == 'hrrr':
        H = Herbie(init_time, model='hrrr', product='sfc', fxx=lead_hours)
        ds = H.xarray('(?:UGRD|VGRD):10 m above ground')
        # Extract point value
        u = ds['u10'].sel(latitude=lat, longitude=360+lon, method='nearest').values
        v = ds['v10'].sel(latitude=lat, longitude=360+lon, method='nearest').values
        
    elif model == 'gfs':
        H = Herbie(init_time, model='gfs', product='pgrb2.0p25', fxx=lead_hours)
        ds = H.xarray('(?:UGRD|VGRD):10 m above ground')
        u = ds['u10'].sel(latitude=lat, longitude=360+lon, method='nearest').values
        v = ds['v10'].sel(latitude=lat, longitude=360+lon, method='nearest').values
    
    wind_speed_ms = (u**2 + v**2)**0.5
    wind_speed_knots = wind_speed_ms * 1.94384
    return wind_speed_knots

# Example: Get 12h-ahead forecasts for March 1, 2025 12:00Z
valid = datetime(2025, 3, 1, 12, 0)
hrrr_12h = get_forecast_at_lead_time(valid, lead_hours=12, model='hrrr')
gfs_12h  = get_forecast_at_lead_time(valid, lead_hours=12, model='gfs')

# Loop over a time range to build a comparison dataset
results = []
for hour_offset in range(0, 24*30):  # 30 days
    valid = datetime(2025, 3, 1, 0, 0) + timedelta(hours=hour_offset)
    try:
        hrrr_val = get_forecast_at_lead_time(valid, 12, 'hrrr')
        gfs_val = get_forecast_at_lead_time(valid, 12, 'gfs')
        results.append({'valid_time': valid, 'hrrr_12h': hrrr_val, 'gfs_12h': gfs_val})
    except Exception as e:
        print(f"  Skip {valid}: {e}")

df = pd.DataFrame(results)
"""


# =============================================================================
# APPROACH 2 (EASY BUT LIMITED): Open-Meteo Historical Forecast API
# =============================================================================
#
# Pros:
#   - Dead simple JSON API, no GRIB parsing
#   - Multiple models: ecmwf_ifs025, gfs_global, icon_global, gem_global, etc.
#   - Data back to at least 2024 (and likely further)
#   - Free (with rate limits)
#
# CRITICAL LIMITATION:
#   - Returns the "best" forecast for each valid time (shortest lead time)
#   - NO way to request a specific lead time or initialization run
#   - previous_day parameter does NOT work (returns identical values)
#   - gfs_seamless and ncep_hrrr_conus return IDENTICAL values (seamless = blended)
#   - You CANNOT determine what lead time the returned value represents
#
# This is fine for "what did model X predict for this hour" but NOT for
# "what did model X predict 12 hours in advance for this hour"
#
# Working model names that give DISTINCT values:
#   ecmwf_ifs025  - ECMWF IFS 0.25 degree
#   gfs_global    - GFS raw (not seamless blend)  
#   icon_global   - DWD ICON global
#   gem_global    - Canadian GEM global
#   
# DO NOT USE (return blended/identical values):
#   gfs_seamless, ncep_hrrr_conus  - these are IDENTICAL
#   icon_seamless, dwd_icon        - identical to icon_global

def open_meteo_historical(start_date, end_date, model='ecmwf_ifs025'):
    """
    Get historical forecast data from Open-Meteo.
    WARNING: This returns the best/shortest-lead-time forecast, not a specific lead time.
    """
    url = (f"https://historical-forecast-api.open-meteo.com/v1/forecast?"
           f"latitude=38.899&longitude=-76.436"
           f"&start_date={start_date}&end_date={end_date}"
           f"&hourly=wind_speed_10m,wind_direction_10m,wind_gusts_10m"
           f"&models={model}"
           f"&wind_speed_unit=kn")
    
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# =============================================================================
# APPROACH 3: NOMADS OPeNDAP (NOT recommended)
# =============================================================================
#
# - https://nomads.ncep.noaa.gov/ only keeps ~10 days of GFS data
# - Historical data returns 403 errors
# - Not suitable for building a multi-month archive
#
# The NCEI (formerly NCDC) archive at https://www.ncei.noaa.gov/thredds/
# has historical GFS but access is slow and unreliable.


# =============================================================================
# SUMMARY & RECOMMENDATION
# =============================================================================
SUMMARY = """
RECOMMENDATION: Use AWS S3 GRIB archives with the Herbie library.

Approach comparison:
┌─────────────────────┬──────────────┬──────────────┬─────────────┐
│ Criteria            │ S3+Herbie    │ Open-Meteo   │ NOMADS      │
├─────────────────────┼──────────────┼──────────────┼─────────────┤
│ Lead time control   │ YES (exact)  │ NO           │ YES         │
│ Model selection     │ HRRR, GFS    │ 4+ models    │ GFS/HRRR    │
│ Historical depth    │ 2021+        │ 2024+        │ ~10 days    │
│ Ease of use         │ Medium       │ Very easy    │ Complex     │
│ Data per request    │ ~2MB/var     │ <1KB JSON    │ Variable    │
│ Rate limits         │ None         │ Yes (free)   │ Moderate    │
│ ECMWF available     │ NO           │ YES          │ NO          │
└─────────────────────┴──────────────┴──────────────┴─────────────┘

For your specific need (12h-ahead forecast comparison):
  - HRRR 12h ahead: S3 archive (every hour, f12 always available)
  - GFS  12h ahead: S3 archive (4x/day, f012)
  - ECMWF 12h ahead: Open-Meteo only (AWS doesn't have ECMWF)
    BUT: Open-Meteo doesn't guarantee which lead time you're getting
  
  Best hybrid approach:
  1. Use Herbie/S3 for HRRR and GFS (exact 12h lead time control)
  2. Use Open-Meteo for ECMWF/ICON/GEM (approximate, shortest lead time)
  3. Accept that the Open-Meteo models give their "best" forecast,
     not necessarily the 12h-ahead one (usually ~6h for ECMWF, ~3h for ICON)

Required packages: pip install herbie-data cfgrib xarray pandas
"""


if __name__ == '__main__':
    print("=" * 70)
    print("DEMO: Byte-range download from HRRR S3 archive")
    print("=" * 70)
    demo_byte_range_download()
    
    print("\n" + "=" * 70)
    print("DEMO: Open-Meteo Historical Forecast API")
    print("=" * 70)
    data = open_meteo_historical('2025-03-01', '2025-03-01', 'ecmwf_ifs025')
    times = data['hourly']['time'][:6]
    winds = data['hourly']['wind_speed_10m'][:6]
    print(f"ECMWF IFS 2025-03-01 (first 6 hours, knots):")
    for t, w in zip(times, winds):
        print(f"  {t}: {w} kn")
    
    print(SUMMARY)
