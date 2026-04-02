---
name: puff-cast project overview
description: Localized marine wind forecasting for Chesapeake Bay - project goals, data sources, and initial findings
type: project
---

Puff Cast is a project to improve marine wind forecasting for the Annapolis, MD area (Chesapeake Bay). The user's motivation: NWS marine forecasts for the area have been unreliable — wind appearing when calm was forecast, calm when gale warnings were issued.

**Approach:** Use NOAA buoy/station data from Thomas Point Lighthouse (TPLM2) and 12 surrounding stations to build ML models that predict wind at Thomas Point using surrounding observations from the prior 1-48 hours.

**Why:** NWS forecasts come from coarse-grid models. The Bay is a narrow body with channeling and thermal effects that coarse models handle poorly. Local station data has signal that models miss.

**How to apply:** The focus is on the ML/prediction side, not app presentation. Key stations are TPLM2 (target), APAM2, COVM2, SLIM2, WASD2, 44009 (ocean), BLTM2. Data is from NDBC direct download, 2020-present.

**Apples-to-apples results (2026-03-31, Oct-Dec 2025 test, exact lead times):**
- HRRR (3km) is the clear winner at all horizons: 3.96/2.62/3.20 kt MAE at 3/6/12h
- Our station-only RF: 3.86/4.26/4.64 kt MAE — competitive at 3h but falls behind at 6-12h
- GFS beats our RF at 3h and 12h but has large negative bias (-2 to -3 kt underforecast)
- ECMWF underforecasts significantly at Thomas Point (-2.7 to -4.2 kt bias)
- Key finding: HRRR at 12h (3.20 kt) dramatically outperforms station-only RF (4.64 kt)
- MOS approach did NOT improve over RF alone — likely needs HRRR output as input feature, not just ECMWF
- Sample size is small (29-31 hours) due to only using 00Z/12Z inits — need more data
- Next step: add HRRR output as MOS feature (the winning model's output + station corrections)
- Note: WASD2 (DC) and pressure gradients are key RF features — weather systems from west
