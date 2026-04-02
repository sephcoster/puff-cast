"""
Station registry for Thomas Point Lighthouse and surrounding NDBC/C-MAN stations.

Stations are selected based on:
- Proximity to TPLM2 (Thomas Point)
- Data reliability and availability
- Geographic spread (up-bay, down-bay, ocean-side, inland)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    id: str
    name: str
    lat: float
    lon: float
    distance_nm: float  # approximate distance from TPLM2
    direction: str  # compass direction from TPLM2
    category: str  # "target", "inner_bay", "mid_bay", "outer_bay", "ocean", "river"


# Thomas Point Lighthouse — our prediction target
TARGET = Station(
    id="TPLM2",
    name="Thomas Point Lighthouse",
    lat=38.899,
    lon=-76.436,
    distance_nm=0,
    direction="",
    category="target",
)

# Surrounding stations selected for geographic coverage and data quality.
# Inner bay stations provide short-lag predictors; outer/ocean stations
# provide longer-lead signals of approaching weather systems.
SURROUNDING = [
    # === Inner Bay / Close Proximity (< 20 nm) ===
    Station(
        id="APAM2",
        name="Annapolis, MD",
        lat=38.963,
        lon=-76.481,
        distance_nm=6,
        direction="NNW",
        category="inner_bay",
    ),
    Station(
        id="BLTM2",
        name="Baltimore, MD (Fort McHenry)",
        lat=39.263,
        lon=-76.580,
        distance_nm=24,
        direction="N",
        category="inner_bay",
    ),
    Station(
        id="CAMM2",
        name="Cambridge, MD",
        lat=38.573,
        lon=-76.068,
        distance_nm=26,
        direction="SE",
        category="inner_bay",
    ),
    # === Mid Bay (20-50 nm) ===
    Station(
        id="SLIM2",
        name="Solomons Island, MD",
        lat=38.317,
        lon=-76.451,
        distance_nm=34,
        direction="S",
        category="mid_bay",
    ),
    Station(
        id="COVM2",
        name="Cove Point, MD",
        lat=38.386,
        lon=-76.382,
        distance_nm=29,
        direction="S",
        category="mid_bay",
    ),
    Station(
        id="PPTM2",
        name="Piney Point, MD",
        lat=38.133,
        lon=-76.533,
        distance_nm=46,
        direction="S",
        category="mid_bay",
    ),
    Station(
        id="WASD2",
        name="Washington, DC",
        lat=38.873,
        lon=-77.022,
        distance_nm=27,
        direction="W",
        category="river",
    ),
    # === Outer Bay / Bay Mouth (50-100 nm) ===
    Station(
        id="YKTV2",
        name="Yorktown, VA (USCG)",
        lat=37.227,
        lon=-76.479,
        distance_nm=100,
        direction="S",
        category="outer_bay",
    ),
    Station(
        id="RPLV2",
        name="Rappahannock Light, VA",
        lat=37.538,
        lon=-76.015,
        distance_nm=84,
        direction="SSE",
        category="outer_bay",
    ),
    # === Ocean Side — catch approaching weather systems ===
    Station(
        id="44009",
        name="Delaware Bay 26nm SE of Cape May",
        lat=38.464,
        lon=-74.703,
        distance_nm=86,
        direction="E",
        category="ocean",
    ),
    Station(
        id="44025",
        name="Long Island 33nm S of Islip, NY",
        lat=40.251,
        lon=-73.164,
        distance_nm=190,
        direction="NE",
        category="ocean",
    ),
    Station(
        id="44014",
        name="Virginia Beach 64nm E of VA Beach",
        lat=36.611,
        lon=-74.836,
        distance_nm=150,
        direction="SSE",
        category="ocean",
    ),
]

ALL_STATIONS = [TARGET] + SURROUNDING

# Quick lookup by ID
STATION_BY_ID = {s.id: s for s in ALL_STATIONS}

# Standard meteorological parameters we care about
WIND_PARAMS = ["WDIR", "WSPD", "GST"]
PRESSURE_PARAMS = ["PRES", "PTDY"]
TEMP_PARAMS = ["ATMP", "WTMP", "DEWP"]
ALL_PARAMS = WIND_PARAMS + PRESSURE_PARAMS + TEMP_PARAMS
