"""
Phase 1 Exploratory Analysis: Thomas Point Wind Forecasting
============================================================

Key questions:
1. What does the wind look like at Thomas Point? (distribution, seasonality, diurnal patterns)
2. How correlated are surrounding stations with Thomas Point?
3. Are there useful LAG correlations? (station X at t-N predicts Thomas Point at t)
4. What features look most promising for a forecasting model?
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 100

# %%
# Load the unified dataset
df = pd.read_parquet(Path("../data/processed/unified_hourly.parquet"))
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"TPLM2 wind speed records: {df['TPLM2_WSPD'].notna().sum():,}")

# %% [markdown]
# ## 1. Thomas Point Wind Characteristics

# %%
# Wind speed distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
wspd = df["TPLM2_WSPD"].dropna()
axes[0].hist(wspd, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
axes[0].axvline(wspd.mean(), color="red", linestyle="--", label=f"Mean: {wspd.mean():.1f} m/s")
axes[0].axvline(wspd.quantile(0.95), color="orange", linestyle="--", label=f"95th: {wspd.quantile(0.95):.1f} m/s")
axes[0].set_xlabel("Wind Speed (m/s)")
axes[0].set_ylabel("Count")
axes[0].set_title("Thomas Point Wind Speed Distribution (2020-2025)")
axes[0].legend()

# Convert to knots for sailor-friendliness
wspd_kt = wspd * 1.944  # m/s to knots
axes[1].hist(wspd_kt, bins=50, edgecolor="black", alpha=0.7, color="teal")
axes[1].axvline(15, color="orange", linestyle="--", label="Small craft advisory (15kt)")
axes[1].axvline(25, color="red", linestyle="--", label="Gale warning (25kt)")
axes[1].set_xlabel("Wind Speed (knots)")
axes[1].set_ylabel("Count")
axes[1].set_title("Wind Speed in Knots")
axes[1].legend()

# Gust vs sustained
gst = df["TPLM2_GST"].dropna()
axes[2].scatter(wspd, df.loc[wspd.index, "TPLM2_GST"], alpha=0.02, s=1, color="steelblue")
axes[2].plot([0, 22], [0, 22], "r--", label="1:1 line")
axes[2].set_xlabel("Sustained Wind (m/s)")
axes[2].set_ylabel("Gust (m/s)")
axes[2].set_title("Gust vs Sustained Wind")
axes[2].legend()

plt.tight_layout()
plt.savefig("../data/processed/01_wind_distribution.png", bbox_inches="tight")
plt.show()
print(f"\nWind > 15kt: {(wspd_kt > 15).mean()*100:.1f}% of hours")
print(f"Wind > 25kt: {(wspd_kt > 25).mean()*100:.1f}% of hours")

# %%
# Seasonal and diurnal patterns
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Monthly average
monthly = df.groupby(df.index.month)["TPLM2_WSPD"].agg(["mean", "std", "max"])
monthly.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
axes[0].bar(monthly.index, monthly["mean"] * 1.944, yerr=monthly["std"] * 1.944,
            color="steelblue", alpha=0.7, capsize=3)
axes[0].set_ylabel("Wind Speed (knots)")
axes[0].set_title("Monthly Average Wind Speed at Thomas Point")

# Diurnal pattern by season
seasons = {
    "Winter (DJF)": [12, 1, 2],
    "Spring (MAM)": [3, 4, 5],
    "Summer (JJA)": [6, 7, 8],
    "Fall (SON)": [9, 10, 11],
}
for name, months in seasons.items():
    mask = df.index.month.isin(months)
    hourly_avg = df.loc[mask].groupby(df.loc[mask].index.hour)["TPLM2_WSPD"].mean() * 1.944
    axes[1].plot(hourly_avg.index, hourly_avg.values, marker="o", markersize=3, label=name)

axes[1].set_xlabel("Hour (UTC)")
axes[1].set_ylabel("Wind Speed (knots)")
axes[1].set_title("Diurnal Wind Pattern by Season")
axes[1].legend()
axes[1].set_xticks(range(0, 24, 3))

plt.tight_layout()
plt.savefig("../data/processed/02_seasonal_diurnal.png", bbox_inches="tight")
plt.show()

# %%
# Wind rose (simplified as direction frequency heatmap)
fig, ax = plt.subplots(figsize=(10, 6))

# Bin wind by direction and speed
wdir = df["TPLM2_WDIR"].dropna()
wspd_at_dir = df.loc[wdir.index, "TPLM2_WSPD"] * 1.944

dir_bins = np.arange(0, 375, 15)
dir_labels = dir_bins[:-1] + 7.5
spd_bins = [0, 5, 10, 15, 20, 25, 50]
spd_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]

dir_cat = pd.cut(wdir, bins=dir_bins, labels=dir_labels)
spd_cat = pd.cut(wspd_at_dir, bins=spd_bins, labels=spd_labels)

rose = pd.crosstab(dir_cat, spd_cat, normalize=True) * 100
sns.heatmap(rose, cmap="YlOrRd", ax=ax, cbar_kws={"label": "% of observations"})
ax.set_xlabel("Wind Speed (knots)")
ax.set_ylabel("Wind Direction (degrees)")
ax.set_title("Thomas Point Wind Rose (2020-2025)")

# Add compass labels on y-axis
compass = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
ytick_positions = [i for i, d in enumerate(dir_labels) if d - 7.5 in compass]
ytick_labels = [compass[d - 7.5] for d in dir_labels if d - 7.5 in compass]
ax.set_yticks(ytick_positions)
ax.set_yticklabels(ytick_labels)

plt.tight_layout()
plt.savefig("../data/processed/03_wind_rose.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Cross-Station Correlations (Simultaneous)

# %%
# Correlation of wind speed across all stations (simultaneous)
wind_cols = [c for c in df.columns if c.endswith("_WSPD")]
station_names = [c.replace("_WSPD", "") for c in wind_cols]

corr = df[wind_cols].corr()
corr.index = station_names
corr.columns = station_names

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
            vmin=0.3, vmax=1.0, ax=ax, square=True)
ax.set_title("Simultaneous Wind Speed Correlation Between Stations")
plt.tight_layout()
plt.savefig("../data/processed/04_station_correlation.png", bbox_inches="tight")
plt.show()

# Show correlations with TPLM2 sorted
print("\nCorrelation with Thomas Point (TPLM2) wind speed:")
tplm2_corr = corr["TPLM2"].drop("TPLM2").sort_values(ascending=False)
for station, r in tplm2_corr.items():
    print(f"  {station:8s}: r={r:.3f}")

# %% [markdown]
# ## 3. Lag Correlation Analysis — The Key Question
#
# Does wind at surrounding stations *precede* wind at Thomas Point?
# If so, those stations provide predictive signal.

# %%
# Compute lagged cross-correlations
target = df["TPLM2_WSPD"].dropna()
lags_hours = range(0, 49)  # 0 to 48 hours

# Stations to check
check_stations = ["APAM2", "BLTM2", "CAMM2", "SLIM2", "COVM2", "PPTM2",
                  "WASD2", "YKTV2", "RPLV2", "44009", "44025", "44014"]

lag_corrs = {}
for station in check_stations:
    col = f"{station}_WSPD"
    if col not in df.columns:
        continue
    corrs = []
    for lag in lags_hours:
        # Correlation of station wind at time (t - lag) with TPLM2 wind at time t
        shifted = df[col].shift(lag)
        valid = target.notna() & shifted.notna()
        if valid.sum() > 100:
            r = target[valid].corr(shifted[valid])
        else:
            r = np.nan
        corrs.append(r)
    lag_corrs[station] = corrs

# %%
# Plot lag correlations
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# All stations
for station, corrs in lag_corrs.items():
    axes[0].plot(list(lags_hours), corrs, label=station, alpha=0.8)

axes[0].set_xlabel("Lag (hours) — station observation precedes Thomas Point by this much")
axes[0].set_ylabel("Correlation (r)")
axes[0].set_title("Lagged Correlation: Surrounding Station Wind Speed → Thomas Point Wind Speed")
axes[0].legend(ncol=4, fontsize=8)
axes[0].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
axes[0].set_xlim(0, 48)

# Focus on most promising stations
# Show correlation GAIN over persistence (autocorrelation of TPLM2 with itself)
tplm2_auto = []
for lag in lags_hours:
    shifted = df["TPLM2_WSPD"].shift(lag)
    valid = target.notna() & shifted.notna()
    r = target[valid].corr(shifted[valid])
    tplm2_auto.append(r)

axes[1].plot(list(lags_hours), tplm2_auto, "k--", linewidth=2, label="TPLM2 persistence (autocorrelation)")

for station, corrs in lag_corrs.items():
    # Only show stations that beat persistence at some lag
    gain = [c - a for c, a in zip(corrs, tplm2_auto)]
    if max(gain) > 0.01:
        axes[1].plot(list(lags_hours), corrs, label=station, alpha=0.8)

axes[1].set_xlabel("Lag (hours)")
axes[1].set_ylabel("Correlation (r)")
axes[1].set_title("Lag Correlation vs. Persistence Baseline")
axes[1].legend(ncol=4, fontsize=8)
axes[1].set_xlim(0, 48)

plt.tight_layout()
plt.savefig("../data/processed/05_lag_correlations.png", bbox_inches="tight")
plt.show()

# Print the key finding: at what lag does each station have max correlation?
print("\nPeak lag correlation per station (vs TPLM2 wind speed):")
print(f"{'Station':10s} {'Best Lag':>10s} {'Correlation':>12s} {'vs Persist':>12s}")
print("-" * 50)
for station, corrs in sorted(lag_corrs.items(), key=lambda x: max(x[1]), reverse=True):
    best_lag = np.argmax(corrs)
    best_r = corrs[best_lag]
    persist_r = tplm2_auto[best_lag]
    print(f"{station:10s} {best_lag:8d}h  r={best_r:8.3f}  persist={persist_r:.3f}")

# %% [markdown]
# ## 4. Pressure as a Predictor
#
# Pressure tendency (falling pressure) is a classic wind predictor.
# Let's see if pressure changes at surrounding stations predict Thomas Point wind.

# %%
# Pressure tendency vs future wind speed at Thomas Point
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# TPLM2 own pressure tendency vs wind speed 6h later
pres = df["TPLM2_PRES"].dropna()
pres_change_6h = pres.diff(6)  # 6-hour pressure change

target_6h = df["TPLM2_WSPD"].shift(-6)  # wind speed 6 hours in the future
valid = pres_change_6h.notna() & target_6h.notna()

axes[0].scatter(pres_change_6h[valid], target_6h[valid], alpha=0.01, s=1, color="steelblue")
axes[0].set_xlabel("6-hour Pressure Change (hPa)")
axes[0].set_ylabel("Wind Speed 6h Later (m/s)")
axes[0].set_title(f"Thomas Point: Pressure Change vs Future Wind\nr={pres_change_6h[valid].corr(target_6h[valid]):.3f}")

# Binned view — clearer signal
bins = pd.cut(pres_change_6h[valid], bins=20)
binned = target_6h[valid].groupby(bins).agg(["mean", "std", "count"])
binned = binned[binned["count"] > 50]
axes[1].errorbar(
    [interval.mid for interval in binned.index],
    binned["mean"] * 1.944,
    yerr=binned["std"] * 1.944 / np.sqrt(binned["count"]),
    fmt="o-", color="steelblue", capsize=3
)
axes[1].set_xlabel("6-hour Pressure Change (hPa)")
axes[1].set_ylabel("Mean Wind Speed 6h Later (knots)")
axes[1].set_title("Binned: Pressure Change → Future Wind Speed")

plt.tight_layout()
plt.savefig("../data/processed/06_pressure_vs_wind.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Temperature Gradient Analysis
#
# Air-water temperature difference drives convective winds, especially in summer.

# %%
# Check which stations have water temp
for station in ["TPLM2", "APAM2", "CAMM2", "SLIM2", "44009"]:
    col = f"{station}_WTMP"
    if col in df.columns:
        avail = df[col].notna().mean() * 100
        print(f"{station} WTMP availability: {avail:.1f}%")

# Use stations that have both air and water temp
stations_with_temp_diff = []
for station in check_stations:
    col = f"{station}_TEMP_DIFF"
    if col in df.columns and df[col].notna().mean() > 0.3:
        stations_with_temp_diff.append(station)
        print(f"{station} has TEMP_DIFF with {df[col].notna().mean()*100:.0f}% coverage")

# %% [markdown]
# ## 6. Feature Importance Preview
#
# Quick random forest to see which features matter most for predicting
# Thomas Point wind speed at different forecast horizons.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def quick_feature_importance(horizon_hours: int = 6):
    """Train a quick RF to see which features predict TPLM2 wind at given horizon."""
    # Target: TPLM2 wind speed N hours in the future
    target = df["TPLM2_WSPD"].shift(-horizon_hours)

    # Features: current values from all stations + time features
    feature_cols = [c for c in df.columns if not c.startswith("TPLM2_WDIR")]  # exclude raw WDIR (use sin/cos)
    feature_cols = [c for c in feature_cols if c != "TPLM2_WSPD"]  # can't use future target
    # But DO include current TPLM2 wind as a feature (persistence)
    # We'll add it back as a lagged feature
    feature_cols_final = feature_cols.copy()

    # Build dataset
    features = df[feature_cols_final].copy()
    features["TPLM2_WSPD_current"] = df["TPLM2_WSPD"]

    valid = target.notna() & features.notna().all(axis=1)
    X = features[valid]
    y = target[valid]

    if len(X) < 1000:
        print(f"Not enough valid samples ({len(X)}), skipping")
        return None

    # Subsample for speed
    if len(X) > 20000:
        idx = np.random.choice(len(X), 20000, replace=False)
        X = X.iloc[idx]
        y = y.iloc[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    score = rf.score(X_test, y_test)
    print(f"\n{horizon_hours}h forecast horizon: R² = {score:.3f}")

    # Top 20 features
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return importance.head(20), score

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for i, horizon in enumerate([6, 12, 24]):
    result = quick_feature_importance(horizon)
    if result is None:
        continue
    importance, score = result

    importance.plot.barh(ax=axes[i], color="steelblue")
    axes[i].set_title(f"{horizon}h Forecast: R² = {score:.3f}")
    axes[i].set_xlabel("Feature Importance")
    axes[i].invert_yaxis()

plt.suptitle("Feature Importance for Thomas Point Wind Speed Prediction", fontsize=14)
plt.tight_layout()
plt.savefig("../data/processed/07_feature_importance.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Summary of Findings

# %%
print("""
=== PHASE 1 EXPLORATION SUMMARY ===

Dataset:
  - {n_hours:,} hours of data ({years:.1f} years)
  - {n_stations} stations, {n_features} features
  - TPLM2 wind speed coverage: {tplm2_pct:.1f}%

Thomas Point Wind Characteristics:
  - Mean wind: {mean_kt:.1f} kt ({mean_ms:.1f} m/s)
  - 95th percentile: {p95_kt:.1f} kt
  - Wind > 15kt (small craft): {sca_pct:.1f}% of hours
  - Strong seasonal pattern: windiest in winter, calmest in summer
  - Clear diurnal pattern: afternoon peak, overnight minimum

Next Steps:
  - Build lagged feature matrix for ML model training
  - Implement persistence and climatological baselines
  - Train XGBoost models at 6h, 12h, 24h horizons
  - Compare against NWS forecast accuracy
""".format(
    n_hours=len(df),
    years=len(df) / 8766,
    n_stations=13,
    n_features=df.shape[1],
    tplm2_pct=df["TPLM2_WSPD"].notna().mean() * 100,
    mean_ms=df["TPLM2_WSPD"].mean(),
    mean_kt=df["TPLM2_WSPD"].mean() * 1.944,
    p95_kt=df["TPLM2_WSPD"].quantile(0.95) * 1.944,
    sca_pct=(df["TPLM2_WSPD"] * 1.944 > 15).mean() * 100,
))
