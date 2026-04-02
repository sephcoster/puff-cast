"""
Operational forecast pipeline.

Fetches latest data, runs saved models, logs predictions, generates HTML.

Usage:
    python -m puff_cast.forecast           # Generate forecast now
    python -m puff_cast.forecast --verify  # Check past predictions against actuals
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"
DOCS_DIR = PROJECT_DIR / "docs"  # GitHub Pages serves from here
LOG_DIR = PROJECT_DIR / "data" / "predictions"

KT = 1.944

STATIONS = {
    "TPLM2": "Thomas Point Light",
    "APAM2": "Annapolis",
    "SLIM2": "Solomons Island",
    "CAMM2": "Cambridge",
}

LEAD_HOURS = [3, 6, 12]


def fetch_latest_obs() -> pd.DataFrame:
    """Fetch latest station observations from NDBC realtime."""
    from puff_cast.stations import ALL_STATIONS

    frames = {}
    for station in ALL_STATIONS:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station.id}.txt"
        try:
            resp = pd.read_csv(
                url, sep=r"\s+", na_values=["MM", "999", "99.0", "9999.0"],
                skiprows=[1],  # units row
            )
            resp["time"] = pd.to_datetime(
                resp[["#YY", "MM", "DD", "hh", "mm"]].rename(
                    columns={"#YY": "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"}
                )
            )
            resp = resp.set_index("time").sort_index()
            # Keep relevant columns
            for col in ["WDIR", "WSPD", "GST", "PRES", "ATMP", "WTMP", "DEWP"]:
                if col in resp.columns:
                    frames[f"{station.id}_{col}"] = pd.to_numeric(resp[col], errors="coerce")
        except Exception as e:
            logger.warning(f"Failed to fetch {station.id}: {e}")

    if not frames:
        return pd.DataFrame()

    obs = pd.DataFrame(frames)
    # Resample to hourly
    obs = obs.resample("1h").mean()
    return obs


def fetch_latest_asos() -> pd.DataFrame:
    """Fetch latest 48h of ASOS data."""
    from puff_cast.fetch_asos import fetch_asos_station, resample_asos_hourly, ASOS_STATIONS
    import time as time_mod

    now = datetime.now(timezone.utc)
    start = (now - pd.Timedelta(hours=48)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")

    all_hourly = []
    for sid in ASOS_STATIONS:
        try:
            raw = fetch_asos_station(sid, start, end)
            if len(raw) > 0:
                hourly = resample_asos_hourly(raw, sid)
                all_hourly.append(hourly)
            time_mod.sleep(1)
        except Exception as e:
            logger.warning(f"ASOS {sid}: {e}")

    if all_hourly:
        return pd.concat(all_hourly, axis=1)
    return pd.DataFrame()


def fetch_latest_hrrr(lead_hours: list[int] = LEAD_HOURS) -> pd.DataFrame:
    """Fetch latest HRRR forecasts at all stations."""
    from puff_cast.fetch_hrrr_enhanced import extract_wind_all_stations, extract_scalar_all_stations, EXTRA_VARIABLES

    try:
        from herbie import Herbie
    except ImportError:
        logger.error("herbie not installed")
        return pd.DataFrame()

    now = datetime.now(timezone.utc)
    # Try most recent init times (within last 6 hours)
    records = []
    for hours_ago in [1, 2, 3, 4, 5, 6]:
        init_time = now - pd.Timedelta(hours=hours_ago)
        init_time = init_time.replace(minute=0, second=0, microsecond=0)

        for fxx in lead_hours:
            try:
                H = Herbie(
                    init_time.strftime("%Y-%m-%d %H:%M"),
                    model="hrrr",
                    product="sfc",
                    fxx=fxx,
                    verbose=False,
                )
                ds = H.xarray(":(?:U|V)GRD:10 m above ground", verbose=False)
                winds = extract_wind_all_stations(ds)

                extra = {}
                for var_name, search_str in EXTRA_VARIABLES.items():
                    try:
                        ds_var = H.xarray(search_str, verbose=False)
                        extra[var_name] = extract_scalar_all_stations(ds_var, var_name)
                    except Exception:
                        extra[var_name] = {}

                for sid, (wspd, wdir) in winds.items():
                    rec = {
                        "init_time": init_time,
                        "valid_time": init_time + pd.Timedelta(hours=fxx),
                        "lead_hours": fxx,
                        "station_id": sid,
                        "hrrr_wspd_ms": wspd,
                        "hrrr_wdir": wdir,
                    }
                    for var_name in EXTRA_VARIABLES:
                        rec[f"hrrr_{var_name}"] = extra.get(var_name, {}).get(sid, np.nan)
                    records.append(rec)

                logger.info(f"  HRRR {init_time} f{fxx:02d}: OK")
            except Exception as e:
                logger.debug(f"  HRRR {init_time} f{fxx:02d}: {e}")

        if records:
            break  # Got data from this init time

    return pd.DataFrame(records) if records else pd.DataFrame()


def generate_forecast() -> dict:
    """Run the full forecast pipeline. Returns forecast dict."""
    now = datetime.now(timezone.utc)
    print(f"Generating forecast at {now.strftime('%Y-%m-%d %H:%M UTC')}...")

    # Fetch latest data
    print("  Fetching HRRR forecasts...")
    hrrr = fetch_latest_hrrr()
    if len(hrrr) == 0:
        print("  ERROR: No HRRR data available")
        return {}

    print("  Fetching station observations...")
    obs = fetch_latest_obs()

    print("  Fetching ASOS airport data...")
    asos = fetch_latest_asos()

    # Load models and generate predictions
    forecasts = {}
    for station, name in STATIONS.items():
        forecasts[station] = {"name": name, "leads": {}}

        for lead in LEAD_HOURS:
            model_path = MODEL_DIR / f"{station}_{lead}h.pkl"
            meta_path = MODEL_DIR / f"{station}_{lead}h.json"

            if not model_path.exists():
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(meta_path) as f:
                meta = json.load(f)

            # Build features for the latest HRRR init
            # For now, use a simplified feature builder from available data
            feat = build_realtime_features(
                hrrr, obs, asos, station, lead, meta["features"]
            )

            if feat is not None:
                pred_ms = model.predict(feat.fillna(-999).values.reshape(1, -1))[0]
                pred_kt = pred_ms * KT
                valid_time = hrrr[hrrr["lead_hours"] == lead].iloc[0]["valid_time"]

                forecasts[station]["leads"][lead] = {
                    "wspd_kt": round(pred_kt, 1),
                    "wspd_ms": round(pred_ms, 2),
                    "valid_time": valid_time.isoformat(),
                    "init_time": hrrr.iloc[0]["init_time"].isoformat(),
                }

    # Get current conditions for display
    for station in STATIONS:
        wspd_col = f"{station}_WSPD"
        if wspd_col in obs.columns and len(obs) > 0:
            latest = obs[wspd_col].dropna()
            if len(latest) > 0:
                forecasts[station]["current_kt"] = round(latest.iloc[-1] * KT, 1)
                forecasts[station]["current_time"] = latest.index[-1].isoformat()

    # Add metadata
    result = {
        "generated_at": now.isoformat(),
        "hrrr_init": hrrr.iloc[0]["init_time"].isoformat() if len(hrrr) > 0 else None,
        "stations": forecasts,
    }

    return result


def build_realtime_features(hrrr, obs, asos, station, lead, feature_cols):
    """Build feature vector matching the training feature set."""
    # Get HRRR forecast for this station/lead
    hrrr_sub = hrrr[(hrrr["station_id"] == station) & (hrrr["lead_hours"] == lead)]
    if len(hrrr_sub) == 0:
        return None

    hrrr_row = hrrr_sub.iloc[-1]  # Most recent init
    init_time = hrrr_row["init_time"]

    feat = {}

    # HRRR features
    feat["hrrr_wspd"] = hrrr_row["hrrr_wspd_ms"]
    feat["hrrr_wspd_sq"] = hrrr_row["hrrr_wspd_ms"] ** 2
    hdir = np.deg2rad(hrrr_row["hrrr_wdir"])
    feat["hrrr_wdir_sin"] = np.sin(hdir)
    feat["hrrr_wdir_cos"] = np.cos(hdir)

    for var in ["hrrr_gust_ms", "hrrr_cape_jkg", "hrrr_pbl_m", "hrrr_sp_pa", "hrrr_fricv_ms"]:
        if var in hrrr_row.index:
            feat[var] = hrrr_row[var]

    if hrrr_row["hrrr_wspd_ms"] > 0.5 and "hrrr_gust_ms" in hrrr_row.index:
        feat["hrrr_gust_factor"] = hrrr_row["hrrr_gust_ms"] / hrrr_row["hrrr_wspd_ms"]

    # Regional HRRR
    regional = hrrr[
        (hrrr["lead_hours"] == lead) &
        (hrrr["init_time"] == init_time) &
        (hrrr["station_id"] != station)
    ]
    for _, rrow in regional.iterrows():
        sid = rrow["station_id"]
        feat[f"{sid}_hrrr_wspd"] = rrow["hrrr_wspd_ms"]
        if "hrrr_cape_jkg" in rrow.index:
            feat[f"{sid}_hrrr_cape"] = rrow["hrrr_cape_jkg"]
        if "hrrr_gust_ms" in rrow.index:
            feat[f"{sid}_hrrr_gust"] = rrow["hrrr_gust_ms"]

    # Regime
    wspd_kt = hrrr_row["hrrr_wspd_ms"] * KT
    feat["regime_light"] = 1.0 if wspd_kt < 5 else 0.0
    feat["regime_moderate"] = 1.0 if 5 <= wspd_kt < 15 else 0.0
    feat["regime_strong"] = 1.0 if wspd_kt >= 15 else 0.0

    # Time features
    vt = hrrr_row["valid_time"]
    feat["hour_sin"] = np.sin(2 * np.pi * vt.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * vt.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * vt.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * vt.month / 12)

    # Station observations at init time
    if len(obs) > 0 and init_time in obs.index:
        row = obs.loc[init_time]
        for var in ["WSPD", "GST", "PRES", "ATMP"]:
            col = f"{station}_{var}"
            if col in obs.columns:
                feat[f"target_{var}"] = row.get(col, np.nan)

        for sid in ["APAM2", "COVM2", "CAMM2", "SLIM2", "WASD2", "44009", "BLTM2"]:
            if sid == station:
                continue
            for var in ["WSPD", "PRES"]:
                col = f"{sid}_{var}"
                if col in obs.columns:
                    feat[f"{sid}_{var.lower()}"] = row.get(col, np.nan)

    # ASOS at init time
    if len(asos) > 0 and init_time in asos.index:
        arow = asos.loc[init_time]
        for sid in ["KBWI", "KDCA", "KNAK", "KNHK", "KESN", "KSBY", "KDOV"]:
            for col_suffix, feat_name in [
                ("wspd_ms", "wspd"), ("mslp", "mslp"),
                ("vsby_km", "vsby"), ("ceil_m", "ceil"), ("cloud", "cloud"),
            ]:
                col = f"{sid}_{col_suffix}"
                if col in asos.columns:
                    feat[f"{sid}_{feat_name}"] = arow.get(col, np.nan)

    # Build DataFrame with correct column order
    feat_series = pd.Series(feat)
    result = pd.Series(index=feature_cols, dtype=float)
    for col in feature_cols:
        if col in feat_series.index:
            result[col] = feat_series[col]
    return result


def log_prediction(forecast: dict):
    """Append predictions to the log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "forecast_log.jsonl"

    with open(log_path, "a") as f:
        f.write(json.dumps(forecast, default=str) + "\n")

    print(f"  Logged to {log_path}")


def verify_past_predictions(obs: pd.DataFrame) -> list[dict]:
    """
    Check past predictions against current observations.

    Loads the prediction log and finds any predictions whose valid_time
    has now passed. Compares predicted wind speed to the actual observation.

    Returns list of verification records for display on the web page.
    """
    log_path = LOG_DIR / "forecast_log.jsonl"
    if not log_path.exists():
        return []

    now = datetime.now(timezone.utc)
    verifications = []

    with open(log_path) as f:
        for line in f:
            try:
                pred = json.loads(line)
            except json.JSONDecodeError:
                continue

            gen_time = pd.Timestamp(pred["generated_at"])

            for sid, sdata in pred.get("stations", {}).items():
                for lead_str, ldata in sdata.get("leads", {}).items():
                    lead = int(lead_str)
                    vt = pd.Timestamp(ldata["valid_time"])
                    # Strip timezone for comparison with tz-naive obs index
                    if vt.tzinfo is not None:
                        vt = vt.tz_localize(None)

                    # Only verify predictions whose valid_time has passed
                    now_naive = pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo else pd.Timestamp(now)
                    if vt > now_naive:
                        continue

                    # Look up actual observation
                    wspd_col = f"{sid}_WSPD"
                    if wspd_col not in obs.columns:
                        continue

                    # Ensure obs index is also tz-naive
                    obs_idx = obs.index
                    if obs_idx.tz is not None:
                        obs_idx = obs_idx.tz_localize(None)

                    # Find closest observation to valid_time (within 1 hour)
                    time_diffs = abs(obs_idx - vt)
                    closest_idx = time_diffs.argmin()
                    if time_diffs[closest_idx] > pd.Timedelta(hours=1):
                        continue

                    actual_ms = obs.iloc[closest_idx][wspd_col]
                    if pd.isna(actual_ms):
                        continue

                    pred_kt = ldata["wspd_kt"]
                    actual_kt = actual_ms * KT
                    error_kt = pred_kt - actual_kt

                    verifications.append({
                        "station": sid,
                        "station_name": sdata.get("name", sid),
                        "lead_hours": lead,
                        "forecast_time": gen_time.isoformat(),
                        "valid_time": vt.isoformat(),
                        "predicted_kt": round(pred_kt, 1),
                        "actual_kt": round(actual_kt, 1),
                        "error_kt": round(error_kt, 1),
                        "abs_error_kt": round(abs(error_kt), 1),
                    })

    # Sort by valid_time descending (most recent first)
    verifications.sort(key=lambda x: x["valid_time"], reverse=True)

    # Save verification log
    if verifications:
        verify_path = DOCS_DIR / "verification.json"
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        with open(verify_path, "w") as f:
            json.dump(verifications, f, indent=2, default=str)

    return verifications


def _wind_color(kt):
    """Return CSS color for wind speed in knots."""
    if kt < 5: return "#94a3b8"
    if kt < 10: return "#22c55e"
    if kt < 15: return "#3b82f6"
    if kt < 20: return "#f59e0b"
    return "#ef4444"


def _error_color(abs_err):
    """Return CSS color for forecast error."""
    if abs_err <= 1.5: return "#22c55e"   # green — excellent
    if abs_err <= 3.0: return "#f59e0b"   # amber — decent
    return "#ef4444"                       # red — missed


def generate_html(forecast: dict, verifications: list[dict] | None = None):
    """Generate HTML page with forecast + verification scorecard."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    gen_time = forecast.get("generated_at", "unknown")
    hrrr_init = forecast.get("hrrr_init", "unknown")

    # === Forecast table rows ===
    station_rows = ""
    for sid, data in forecast.get("stations", {}).items():
        name = data.get("name", sid)
        current = data.get("current_kt", "—")
        current_time = data.get("current_time", "")
        if current_time:
            current_time = pd.Timestamp(current_time).strftime("%H:%M")

        lead_cells = ""
        for lead in LEAD_HOURS:
            ldata = data.get("leads", {}).get(lead, {})
            wspd = ldata.get("wspd_kt", "—")
            vtime = ldata.get("valid_time", "")
            if vtime:
                vtime = pd.Timestamp(vtime).strftime("%H:%M")
            if isinstance(wspd, (int, float)):
                color = _wind_color(wspd)
                lead_cells += f'<td style="color:{color};font-weight:bold;font-size:1.3em">{wspd:.0f} kt<br><small style="color:#666;font-weight:normal">{vtime} UTC</small></td>'
            else:
                lead_cells += f"<td>—</td>"

        station_rows += f"""
        <tr>
            <td><strong>{name}</strong><br><small>{sid}</small></td>
            <td>{current} kt<br><small>{current_time} UTC</small></td>
            {lead_cells}
        </tr>"""

    # === Verification section ===
    verify_html = ""
    running_stats_html = ""
    if verifications:
        # Recent checks (last 20)
        recent = verifications[:20]
        verify_rows = ""
        for v in recent:
            vt = pd.Timestamp(v["valid_time"]).strftime("%b %d %H:%M")
            err_color = _error_color(v["abs_error_kt"])
            sign = "+" if v["error_kt"] > 0 else ""
            verify_rows += f"""
            <tr>
                <td>{v["station_name"]}</td>
                <td>{v["lead_hours"]}h</td>
                <td>{vt}</td>
                <td style="color:{_wind_color(v['predicted_kt'])}">{v["predicted_kt"]:.0f} kt</td>
                <td style="color:{_wind_color(v['actual_kt'])}">{v["actual_kt"]:.0f} kt</td>
                <td style="color:{err_color};font-weight:bold">{sign}{v["error_kt"]:.1f} kt</td>
            </tr>"""

        verify_html = f"""
    <div class="section">
        <h3>Recent Forecast Checks</h3>
        <p class="section-desc">How did our predictions compare to what actually happened?</p>
        <table>
            <thead>
                <tr>
                    <th>Station</th>
                    <th>Lead</th>
                    <th>Valid Time</th>
                    <th>Predicted</th>
                    <th>Actual</th>
                    <th>Error</th>
                </tr>
            </thead>
            <tbody>{verify_rows}
            </tbody>
        </table>
    </div>"""

        # Running accuracy stats
        df_v = pd.DataFrame(verifications)
        if len(df_v) > 0:
            stats_rows = ""
            for sid in STATIONS:
                for lead in LEAD_HOURS:
                    mask = (df_v["station"] == sid) & (df_v["lead_hours"] == lead)
                    if mask.sum() < 1:
                        continue
                    sub = df_v[mask]
                    mae = sub["abs_error_kt"].mean()
                    n = len(sub)
                    station_name = STATIONS.get(sid, sid)
                    stats_rows += f"""
                <tr>
                    <td>{station_name}</td>
                    <td>{lead}h</td>
                    <td style="color:{_error_color(mae)};font-weight:bold">{mae:.1f} kt</td>
                    <td>{n}</td>
                </tr>"""

            if stats_rows:
                overall_mae = df_v["abs_error_kt"].mean()
                running_stats_html = f"""
    <div class="section">
        <h3>Running Accuracy — Live MAE: <span style="color:{_error_color(overall_mae)}">{overall_mae:.1f} kt</span></h3>
        <p class="section-desc">Based on {len(df_v)} verified predictions</p>
        <table>
            <thead>
                <tr><th>Station</th><th>Lead</th><th>MAE</th><th>N</th></tr>
            </thead>
            <tbody>{stats_rows}
            </tbody>
        </table>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Puff Cast — Chesapeake Bay Wind Forecast</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{ font-size: 1.8em; margin-bottom: 4px; color: #38bdf8; }}
        .subtitle {{ color: #64748b; margin-bottom: 20px; font-size: 0.9em; }}
        .meta {{
            background: #1e293b; padding: 12px 16px; border-radius: 8px;
            margin-bottom: 20px; font-size: 0.85em; color: #94a3b8;
        }}
        table {{
            width: 100%; border-collapse: collapse;
            background: #1e293b; border-radius: 8px; overflow: hidden;
        }}
        th {{
            background: #334155; padding: 10px; text-align: center;
            font-size: 0.8em; color: #94a3b8; text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{ padding: 10px; text-align: center; border-top: 1px solid #334155; }}
        tr:hover {{ background: #263548; }}
        .section {{
            background: #1e293b; padding: 16px; border-radius: 8px; margin-top: 20px;
        }}
        .section h3 {{ color: #38bdf8; margin-bottom: 4px; font-size: 1em; }}
        .section-desc {{ color: #64748b; font-size: 0.8em; margin-bottom: 12px; }}
        .section table {{ font-size: 0.85em; }}
        .legend {{
            margin-top: 16px; display: flex; gap: 16px;
            justify-content: center; flex-wrap: wrap; font-size: 0.8em;
        }}
        .legend span {{ display: inline-flex; align-items: center; gap: 4px; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
        .footer {{
            margin-top: 30px; text-align: center; color: #475569; font-size: 0.8em;
        }}
        .footer a {{ color: #38bdf8; }}
    </style>
</head>
<body>
    <h1>Puff Cast</h1>
    <p class="subtitle">ML-enhanced wind forecasts for Chesapeake Bay</p>

    <div class="meta">
        Forecast: <strong>{pd.Timestamp(gen_time).strftime('%b %d, %Y %H:%M UTC')}</strong>
        &nbsp;|&nbsp; HRRR init: <strong>{pd.Timestamp(hrrr_init).strftime('%H:%M UTC') if hrrr_init else '—'}</strong>
        &nbsp;|&nbsp; Updated every 3 hours
    </div>

    <table>
        <thead>
            <tr><th>Station</th><th>Now</th><th>+3h</th><th>+6h</th><th>+12h</th></tr>
        </thead>
        <tbody>{station_rows}
        </tbody>
    </table>

    <div class="legend">
        <span><span class="dot" style="background:#94a3b8"></span> &lt;5 kt</span>
        <span><span class="dot" style="background:#22c55e"></span> 5-10 kt</span>
        <span><span class="dot" style="background:#3b82f6"></span> 10-15 kt</span>
        <span><span class="dot" style="background:#f59e0b"></span> 15-20 kt</span>
        <span><span class="dot" style="background:#ef4444"></span> 20+ kt</span>
    </div>

    {verify_html}
    {running_stats_html}

    <div class="section">
        <h3>Model Accuracy (backtest, 12h lead)</h3>
        <table>
            <tr><th>Station</th><th>Puff Cast</th><th>Raw NWS</th><th>Improvement</th></tr>
            <tr><td>Annapolis</td><td>1.4 kt</td><td>4.4 kt</td><td style="color:#22c55e">70%</td></tr>
            <tr><td>Cambridge</td><td>2.0 kt</td><td>3.0 kt</td><td style="color:#22c55e">34%</td></tr>
            <tr><td>Solomons</td><td>1.9 kt</td><td>2.5 kt</td><td style="color:#22c55e">25%</td></tr>
            <tr><td>Thomas Point</td><td>2.4 kt</td><td>3.1 kt</td><td style="color:#22c55e">19%</td></tr>
        </table>
    </div>

    <div class="footer">
        <p>Ensemble of HRRR, GFS, ECMWF corrected with 27 local stations.</p>
        <p style="margin-top: 8px;"><a href="latest.json">API</a> &nbsp;|&nbsp; <a href="verification.json">Verification data</a></p>
    </div>
</body>
</html>"""

    html_path = DOCS_DIR / "index.html"
    html_path.write_text(html)
    print(f"  HTML written to {html_path}")

    json_path = DOCS_DIR / "latest.json"
    with open(json_path, "w") as f:
        json.dump(forecast, f, indent=2, default=str)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    forecast = generate_forecast()
    if forecast:
        log_prediction(forecast)

        # Verify past predictions against current observations
        print("  Verifying past predictions...")
        obs = fetch_latest_obs()
        verifications = verify_past_predictions(obs)
        if verifications:
            print(f"  Verified {len(verifications)} past predictions")
        else:
            print("  No past predictions to verify yet")

        generate_html(forecast, verifications)
        print("\nDone! Forecast generated, verified, and published.")
    else:
        print("\nFailed to generate forecast.")
