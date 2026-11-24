# -*- coding: utf-8 -*-
"""Step 2 - add census tract information to each pharmacy and set aside far ones.

For one city, Step 2 finds the census tract each Step 1 pharmacy sits in, copies
that tract's American Community Survey (ACS) numbers onto the pharmacy's row,
works out the tract's poverty rate and vehicles per person, and sets aside
pharmacies that are too far from the city to belong in the analysis. Step 0
calls attach_census_info once per city.

A census tract is a small area (usually 1,200 to 8,000 people) for which the
Census Bureau publishes statistics. Tracts are identified by FIPS codes: a
2-digit state code, a 3-digit county code and a 6-digit tract code, which
together make the 11-digit GEOID (e.g. 55 + 079 + 000101 = 55079000101).

WHAT STEP 2 READS
-----------------
- step01_pharmacy/pharmacy_data.csv (from Step 1): one row per pharmacy, with
  ''pharmacy_id'', ''analysis_run_tag'', NPI, name, address, and the latitude and
  longitude of Google Places' match.
- Three web services:
  - OpenStreetMap's Nominatim service, through OSMnx (once per city): the city's
    boundary, whose centre the distance filter measures from;
  - the Census Data API (once per city; needs CENSUS_API_KEY): ACS 5-year
    estimates for every tract in the state;
  - the Census Geocoder (once per pharmacy): which tract contains this point?
    This is the slow part of Step 2; a progress bar shows how far it has got.

WHAT STEP 2 WRITES (in <CityState>/step02_census/)
--------------------------------------------------
- tract_data.csv: one row per kept pharmacy: its Step 1 identity columns, its
  tract codes, the tract's ACS values and the computed fields below. Step 3 reads
  this file; these are the pharmacies that go on to the travel-time matrices.
- far_pharmacies.csv: the same columns, for pharmacies set aside by the distance
  filter. It is written even when no pharmacy is far (it then has no rows).
- step02_metadata.json: the settings used (ACS year, maximum distance), how many
  pharmacies were read, kept, set aside as far, and dropped for each reason (the
  counters in attach_census_info), and the file paths. It is written last.

A pharmacy dropped for a reason other than distance (no usable coordinates,
geocoder failure, no matching tract) is in neither CSV file; only the counts in
the metadata show it. Every Step 1 pharmacy ends up in exactly one group:
    input_pharmacy_count = retained_pharmacy_count + far_pharmacy_count
                           + missing_coordinates + invalid_coordinates
                           + geocoder_failures + no_tract_match

ACS VARIABLES (5-year estimates; the E at the end of each code means "estimate")
--------------------------------------------------------------------------------
    B01003_001E  total population
    B09001_001E  population under 18
    B08201_003E  households with 1 vehicle available
    B08201_004E  households with 2 vehicles available
    B08201_005E  households with 3 vehicles available
    B08201_006E  households with 4 or more vehicles available
    C17002_001E  ratio of income to poverty level, total: everyone whose poverty
                 status is determined (see the Poverty_Rate note below)
    C17002_002E  ratio under 0.50: income below half the poverty line
    C17002_003E  ratio 0.50 to 0.99: below the poverty line, but above half of it
    NAME         the tract's name (tract number, county and state)

B08201_002E (households with no vehicle) is not downloaded, because those
households add no vehicles to the total. Margins of error (the "M" codes) are
not downloaded either. ACS_YEAR in Step 0 picks the survey: 2022 means the
2018-2022 5-year estimates (five years of answers pooled together; these are the
only ACS estimates published for areas as small as tracts).

COMPUTED FIELDS (for the tract a pharmacy sits in)
--------------------------------------------------
    Poverty_Rate        = 100 * (C17002_002E + C17002_003E) / B01003_001E
    vehicles            = 1*n1 + 2*n2 + 3*n3 + 4*n4
                          (n1..n4 = B08201_003E..B08201_006E; "4 or more" counts as 4)
    Vehicles_per_capita = vehicles / B01003_001E
    pop_over_18         = B01003_001E - B09001_001E

Poverty_Rate and Vehicles_per_capita are NaN when the tract's population is 0
or missing. The vehicle total, the B08201 counts and B09001_001E are used here
but not saved; the C17002 counts and B01003_001E are saved.

WORKED EXAMPLE (one tract)
--------------------------
    B01003_001E = 3000 people, B09001_001E = 700 of them under 18
    households with 1, 2, 3, 4+ vehicles: n1..n4 = 600, 400, 100, 50
    C17002_002E = 250, C17002_003E = 350 (C17002_001E = 2900)

    vehicles            = 600 + 2*400 + 3*100 + 4*50 = 1900
    Vehicles_per_capita = 1900 / 3000                = 0.633
    pop_over_18         = 3000 - 700                 = 2300
    Poverty_Rate        = 100 * (250 + 350) / 3000   = 20.0 (%)

Step 4 uses B01003_001E as the tract population P_i and Vehicles_per_capita as
the car share C_i (capped at 1 there). No travel-time or persistent-homology
calculation uses Poverty_Rate or pop_over_18; they are kept for reference.

HOW THE PIECES FIT TOGETHER (attach_census_info, at the bottom)
---------------------------------------------------------------
1. Check the needed packages, the Census API key, and that no Step 2 output
   exists yet.
2. Read and check the Step 1 file ...................... validate_step1_input
   and look up the state's FIPS code (e.g. "WI" -> "55").
3. Find the city's centre for the distance filter (OSMnx geocode_to_gdf).
4. Download ACS values for every tract in the state .... prepare_acs_dataframe
5. For each pharmacy: find its tract with the Census Geocoder
   ...................................................... extract_census_geoids
   then that tract's ACS row (by county code + tract code), and compute
   Poverty_Rate, Vehicles_per_capita and pop_over_18 .... safe_poverty_rate
6. Check the matched rows (one run tag, unique pharmacy IDs).
7. Distance filter, measured from the city centre ...... haversine
       distance_mi > max_distance_miles  ->  far_pharmacies.csv
8. Save tract_data.csv and far_pharmacies.csv, then step02_metadata.json.

THE DISTANCE FILTER (far pharmacies)
------------------------------------
Step 1 finds pharmacies by the city name in their NPI address, then places each
one at Google Places' first match. A pharmacy can therefore sit well outside the
city: postal city names often cover suburbs and unincorporated areas, and
Google's first match can occasionally be the wrong place. Step 3 downloads the
street network for a box around all the pharmacies, so one distant pharmacy
would make that download much larger and pull places outside the city into the
analysis. Step 2 therefore measures each pharmacy's straight-line distance from
the city's centre (haversine) and sets aside every pharmacy more than
max_distance_miles away (Step 0's MAX_DISTANCE_MILES, 20 miles by default).

The city's centre is the centroid (geometric centre) of the city boundary that
OSMnx's geocode_to_gdf gets from OpenStreetMap's Nominatim service for
"<city>, <state>, USA". It is the centre of the city's shape, which need not be
downtown. If that lookup fails (no network, Nominatim not responding, place not
found), Step 2 prints a WARNING and carries on WITHOUT the filter: every matched
pharmacy is kept and far_pharmacies.csv has no rows. Limitation: the metadata
does not record that the filter was skipped (it still lists max_distance_miles),
so the printed WARNING is the only sign.

NOTES AND KNOWN ISSUES
----------------------
- Known issue (e): if the Census Geocoder request fails for a pharmacy (for
  example during a brief network outage), that pharmacy is counted in
  ''geocoder_failures'' and silently dropped. Step 0 then treats Step 2 as
  complete, so rerunning with the same ANALYSIS_RUN_TAG does not retry it.
  Check the count in step02_metadata.json.
- Known issue (f): Vehicles_per_capita divides household vehicles ("4 or more"
  counted as 4) by the total population of all ages (B01003_001E).
  ''pop_over_18'' (people aged 18 and over, i.e. the voting-age population) is
  computed but not used for it. Hickok et al. divide vehicles by the voting-age
  population, which gives a higher car share (1900 / 2300 = 0.826 instead of
  0.633 in the example above).
- Counting "4 or more vehicles" as exactly 4 slightly undercounts vehicles in
  tracts where some households have 5 or more.
- Vehicles_per_capita is not capped here and can be above 1 (more vehicles than
  people); Step 4 caps C_i at 1.
- Poverty_Rate divides by total population (B01003_001E), not by C17002_001E
  (the people whose poverty status is determined). That group leaves out people
  living in institutions (such as prisons and nursing homes), college dormitories
  and military barracks, and unrelated children under 15, so the rate comes out
  somewhat lower than with the official denominator (20.0% instead of
  100 * 600 / 2900 = 20.7% in the example), and noticeably lower in tracts with
  large dormitories or prisons.
- ACS values are downloaded for one state only. A pharmacy whose point falls in
  another state (for a city on a state line, the far side of the line) is
  counted in ''no_tract_match'' and dropped; it is not written to
  far_pharmacies.csv.
- The geocoder returns current tract boundaries. If its tract is not in the
  chosen ACS year's data (for example after tract boundaries were redrawn), the
  pharmacy is also counted in ''no_tract_match''.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_common import (
    ANALYSIS_TAG_COLUMN,
    utc_now_iso,
    validate_analysis_tag,
    validate_pharmacy_ids,
    write_json,
)

# The web-service packages are imported inside try/except so this file can still
# be imported without them, for example to test the plain-Python helpers
# (haversine, extract_census_geoids) offline. Each import error is saved, and
# attach_census_info stops with a clear message at the start if a package it
# needs is missing.

# OSMnx: gets the city's boundary from OpenStreetMap (for the filter's centre).
try:
    import osmnx as ox
    _OSMNX_IMPORT_ERROR = None
except Exception as exc:
    ox = None
    _OSMNX_IMPORT_ERROR = exc

# census: downloads ACS tables from the Census Data API.
# us: US state abbreviations and their FIPS codes (e.g. "WI" -> "55").
try:
    from census import Census
    from us import states
    _CENSUS_IMPORT_ERROR = None
except Exception as exc:
    Census = None
    states = None
    _CENSUS_IMPORT_ERROR = exc

# cg_compat.py (in the pipeline folder): asks the Census Geocoder which tract
# contains a point. It uses the censusgeocode package when that works, and
# otherwise calls the Census Geocoder website directly.
try:
    import cg_compat as cg
    _CG_IMPORT_ERROR = None
except Exception as exc:
    cg = None
    _CG_IMPORT_ERROR = exc

# tqdm draws the progress bar for the geocoder loop. Without it the loop runs the
# same way, just with no progress bar.
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_kwargs):
        """Stand-in for tqdm when it is not installed: return the rows unchanged."""
        return iterable

# Step 1 columns copied onto every Step 2 row (only those present in the Step 1
# file), so each row can be traced back to its Step 1 pharmacy. ''status'' and
# ''candidates'' are Step 1's Google Places result: the business status (always
# OPERATIONAL after Step 1's filter) and how many matches Google returned.
IDENTITY_FIELDS = [
    "pharmacy_id",
    ANALYSIS_TAG_COLUMN,
    "NPI",
    "google_place_id",
    "OrganizationName",
    "AddressLocation",
    "latitude",
    "longitude",
    "status",
    "candidates",
]

# Mean radius of the Earth in miles (6,371 km), used by haversine.
EARTH_RADIUS_MI = 3958.8


# =============================================================================
# Calculations: poverty rate and straight-line distance
# =============================================================================

def safe_poverty_rate(under50: Any, under100: Any, total_pop: Any) -> float:
    """Return the share of people below the poverty line, as a percentage (0 to 100).

        Poverty_Rate = 100 * (under50 + under100) / total_pop

    attach_census_info passes:
    - under50:   C17002_002E, people whose income is under 0.50 times the poverty
                 line (below half of it);
    - under100:  C17002_003E, people at 0.50 to 0.99 times the poverty line.
                 Despite the name, this is only that band, not everyone under
                 1.00; adding the two bands gives everyone below the poverty line;
    - total_pop: B01003_001E, the tract's total population.

    Example: 250 + 350 = 600 people below the poverty line in a tract of 3000
    people gives 100 * 600 / 3000 = 20.0.

    Missing values: a missing band counts as 0 people. If total_pop is missing or
    0, the rate can't be calculated and NaN is returned (not an error).

    Note: the denominator is the total population, not C17002_001E (the people
    whose poverty status is determined), so the rate is somewhat lower than the
    official definition gives; see the notes at the top of this file.
    """
    if pd.isna(total_pop) or float(total_pop) <= 0:
        return float("nan")
    value50 = 0.0 if pd.isna(under50) else float(under50)
    value100 = 0.0 if pd.isna(under100) else float(under100)
    return 100.0 * (value50 + value100) / float(total_pop)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line ("as the crow flies") distance in miles between two points.

    Points are given as latitude and longitude in degrees. This is used only for
    the far-pharmacy filter; travel times come from Step 3's street networks.

    The haversine formula gives the distance along the Earth's surface, treating
    the Earth as a sphere of radius R = 3958.8 miles (angles in radians):

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        distance = 2 * R * asin(sqrt(a))

    where dlat = lat2 - lat1 and dlon = lon2 - lon1. Treating the Earth as a
    sphere is off by less than about 0.5%, which doesn't matter for a cut-off.

    Examples, from (43.00, -88.00):
        to (43.30, -88.00): 20.73 miles (just over the 20-mile default)
        to (43.10, -87.80): 12.24 miles
    One degree of latitude is about 69.1 miles; one degree of longitude is
    shorter away from the equator (about 50.5 miles at 43 degrees north).
    """
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_MI * 2 * math.asin(math.sqrt(a))


# =============================================================================
# Reading tract codes from the geocoder and the ACS download
# =============================================================================

def _first_present_case_insensitive(mapping: dict, *keys):
    """Return the value of the first of ''keys'' found in ''mapping'', ignoring case.

    Keys are tried in the order given, and a key whose value is None or "" counts
    as not found. Returns None if none is found, or if ''mapping'' is not a dict.
    This lets extract_census_geoids accept "GEOID", "geoid" or "GEO_ID" alike.

    Example: mapping = {"Geoid": "", "GEO_ID": "1400000US55079000101"} and
    keys = ("GEOID", "GEO_ID") returns "1400000US55079000101" (the empty Geoid is
    skipped).
    """
    if not isinstance(mapping, dict):
        return None
    # Lower-case every key once, so the search below ignores capital letters.
    lookup = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        value = lookup.get(str(key).lower())
        if value not in (None, ""):
            return value
    return None


def _normalize_code(value: Any, width: int) -> str | None:
    """Return a FIPS code as text, padded with leading zeros to ''width'' digits.

    The same code can arrive as a number (79), as text ("79" or "079") or as a
    decimal (79.0), and a number loses its leading zeros. Padding them back makes
    the codes from the geocoder and from the ACS compare equal:
        _normalize_code(79, 3)       -> "079"     (county)
        _normalize_code("101.0", 6)  -> "000101"  (tract)
        _normalize_code(None, 3)     -> None
    Text that is not a number keeps its characters before any "." and is padded
    the same way. Missing or empty values give None.
    """
    if value in (None, "") or pd.isna(value):
        return None
    try:
        # 79, "079" and 79.0 all become "79" here; zfill adds the zeros back.
        text = str(int(float(value)))
    except Exception:
        # Not a number (e.g. "12A"): keep the text before any decimal point.
        text = str(value).strip().split(".")[0]
    return text.zfill(width) if text else None


def extract_census_geoids(tract_info: dict) -> tuple[str | None, str | None, str | None, str | None]:
    """Read the state, county and tract codes and the GEOID from one geocoder tract.

    ''tract_info'' is one tract record from the Census Geocoder. It normally looks
    like {"GEOID": "55079000101", "STATE": "55", "COUNTY": "079",
    "TRACT": "000101", ...}, but key names differ between Census products and
    versions, so several spellings are accepted (ignoring case):
        GEOID:   "GEOID", "GEO_ID", "GEOID20"
        state:   "STATE", "STATEFP"
        county:  "COUNTY", "COUNTYFP"
        tract:   "TRACT", "TRACTCE"

    Returns (state_fips, county_fips, tract_id, geoid) as text, for example
    ("55", "079", "000101", "55079000101"); any part that can't be found is None.

    Cleaning the GEOID: a "1400000US" prefix (as in data.census.gov's GEO_ID,
    e.g. "1400000US55079000101") is removed, only digits are kept, and if 11 or
    more remain, the last 11 are used (2 state + 3 county + 6 tract digits).
    The two sources then fill each other's gaps: a code missing from the record
    is taken from its place in an 11-digit GEOID, and a missing GEOID is built
    from the three codes.
    """
    geoid = _first_present_case_insensitive(tract_info, "GEOID", "GEO_ID", "GEOID20")
    geoid = str(geoid).strip() if geoid not in (None, "") else None
    if geoid and geoid.startswith("1400000US"):
        geoid = geoid.replace("1400000US", "", 1)
    if geoid:
        geoid = "".join(character for character in geoid if character.isdigit())
        if len(geoid) >= 11:
            geoid = geoid[-11:]
    state_fips = _normalize_code(
        _first_present_case_insensitive(tract_info, "STATE", "STATEFP"), 2
    )
    county_fips = _normalize_code(
        _first_present_case_insensitive(tract_info, "COUNTY", "COUNTYFP"), 3
    )
    tract_id = _normalize_code(
        _first_present_case_insensitive(tract_info, "TRACT", "TRACTCE"), 6
    )
    # Fill any missing code from its place in the GEOID: SS CCC TTTTTT.
    if geoid and len(geoid) == 11:
        state_fips = state_fips or geoid[:2]
        county_fips = county_fips or geoid[2:5]
        tract_id = tract_id or geoid[5:]
    # No GEOID in the record: build it from the three codes.
    if not geoid and state_fips and county_fips and tract_id:
        geoid = f"{state_fips}{county_fips}{tract_id}"
    return state_fips, county_fips, tract_id, geoid


def prepare_acs_dataframe(records: list[dict]) -> pd.DataFrame:
    """Turn the ACS download into a table with padded state, county and tract codes.

    ''records'' is what the census package returns: one dict per tract, holding
    the requested ACS variables plus the tract's "state", "county" and "tract"
    codes. The codes are padded to 2, 3 and 6 digits (see _normalize_code) so
    they match the codes extract_census_geoids reads from the geocoder; for
    example county 79 and "079" both become "079". Returns one row per tract.
    """
    frame = pd.DataFrame(records)
    for column, width in (("state", 2), ("county", 3), ("tract", 6)):
        if column in frame.columns:
            frame[column] = frame[column].apply(lambda value: _normalize_code(value, width))
    return frame


# =============================================================================
# Checks on the Step 1 input
# =============================================================================

def validate_step1_input(df: pd.DataFrame, expected_tag: str, context: str) -> None:
    """Check the Step 1 table before any web request is made.

    Stops with an error if:
    - the ''analysis_run_tag'' column is missing, has blanks, mixes tags, or is
      not this run's tag;
    - any ''pharmacy_id'' is missing or appears twice;
    - any of the columns NPI, OrganizationName, AddressLocation, latitude or
      longitude is missing.
    Only the columns' presence is checked here; blank or bad coordinates are
    handled pharmacy by pharmacy in attach_census_info. ''context'' (usually the
    file path) goes into the error messages so you can tell which file failed.
    """
    validate_analysis_tag(df, expected_tag, context)
    validate_pharmacy_ids(df, context)
    required = {
        "NPI",
        "OrganizationName",
        "AddressLocation",
        "latitude",
        "longitude",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}.")


def _identity_record(row: pd.Series) -> dict[str, Any]:
    """Copy one Step 1 row's identity columns (IDENTITY_FIELDS) into a new dict.

    Columns the Step 1 file doesn't have are left out rather than filled with
    blanks. attach_census_info then adds the tract fields to this dict.
    """
    return {field: row.get(field) for field in IDENTITY_FIELDS if field in row.index}


# =============================================================================
# Step 2 for one city: runs the pieces above in order
# =============================================================================

def attach_census_info(
    input_csv_path: str | os.PathLike[str],
    output_folder: str | os.PathLike[str],
    *,
    city: str,
    state: str,
    analysis_run_tag: str,
    census_api_key: str,
    max_distance_miles: float = 20.0,
    # Default tentatively updated to a more recent survey: 2024 instead of 2022.
    # Step 0's ACS_YEAR and the command-line default below are still 2022.
    acs_year: int = 2024,
) -> dict[str, str]:
    """Run Step 2 for one city: attach tract and ACS data, set aside far pharmacies.

    Step 0 calls this once per city. The numbered comments below match the
    numbered list at the top of this file.

    Inputs:
    - input_csv_path: Step 1's pharmacy_data.csv for this city.
    - output_folder: this city's step02_census folder (created if needed).
    - city, state: e.g. "Milwaukee" and "WI" (a two-letter state abbreviation).
    - analysis_run_tag: this run's tag; every Step 1 row must carry it.
    - census_api_key: key for the Census Data API (Step 0's CENSUS_API_KEY).
    - max_distance_miles: the distance filter's limit, in miles. Step 0 passes
      MAX_DISTANCE_MILES; the 20.0 default only applies when it is left out.
    - acs_year: the last year of the ACS 5-year survey (2022 = 2018-2022).
      Step 0 passes ACS_YEAR; the 2024 default only applies when it is left out.

    Returns the paths of the three files written, as
    {"tract_output": ..., "far_output": ..., "metadata_output": ...}.

    Stops with an error, writing no output files, if a package or the API key is
    missing, if any Step 2 file already exists in the folder, if the Step 1 file
    fails its checks (columns, run tag, duplicate IDs), if the state abbreviation
    is not recognized, or if the ACS download fails. Problems with a single
    pharmacy do not stop it: the pharmacy is counted and skipped (see the
    counters below; known issue (e)). If no pharmacy is kept, tract_data.csv is
    written empty, and Step 0's check of the Step 2 output then stops the run.
    """
    # 1. Stop before any web request if a package or the Census API key is
    #    missing (see the imports at the top of this file).
    if _CENSUS_IMPORT_ERROR is not None:
        raise ImportError(f"Step 2 requires census and us packages: {_CENSUS_IMPORT_ERROR}")
    if _CG_IMPORT_ERROR is not None:
        raise ImportError(f"Step 2 requires cg_compat.py: {_CG_IMPORT_ERROR}")
    if ox is None:
        raise ImportError(f"Step 2 requires OSMnx: {_OSMNX_IMPORT_ERROR}")
    if not census_api_key:
        raise ValueError("Step 2 requires CENSUS_API_KEY.")

    input_path = Path(input_csv_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    tract_output = output_dir / "tract_data.csv"
    far_output = output_dir / "far_pharmacies.csv"
    metadata_output = output_dir / "step02_metadata.json"
    # Step 0 skips a city whose Step 2 is already complete, so any of these files
    # existing here means an earlier run stopped partway. Stop rather than mix old
    # and new files (nothing is overwritten).
    if any(path.exists() for path in (tract_output, far_output, metadata_output)):
        raise FileExistsError(
            f"Step 2 output already exists in {output_dir}; the master should validate and skip it."
        )

    # 2. Read and check the Step 1 file. IDs are read as text so pandas does not
    #    turn them into numbers (an NPI column with a blank cell would otherwise
    #    come back as decimals such as 1234567890.0).
    dtype = {"pharmacy_id": "string", "NPI": "string", "google_place_id": "string"}
    source = pd.read_csv(input_path, dtype=dtype)
    validate_step1_input(source, analysis_run_tag, str(input_path))
    #    The state's FIPS code (e.g. "WI" -> "55") chooses the ACS download and is
    #    compared with each geocoder result. An abbreviation that the us package
    #    does not list as a state stops the run.
    state_abbr = str(state).strip().upper()
    state_map = {item.abbr: item for item in states.STATES}
    if state_abbr not in state_map:
        raise ValueError(f"Unsupported state abbreviation: {state_abbr!r}.")
    expected_state_fips = state_map[state_abbr].fips

    # 3. The city's centre, for the distance filter in stage 7: the centroid of
    #    the city boundary that OSMnx gets from OpenStreetMap's Nominatim service.
    #    If the lookup fails, Step 2 carries on without the distance filter.
    place_name = f"{city}, {state_abbr}, USA"
    try:
        boundary = ox.geocode_to_gdf(place_name)
        # Row 0 holds the boundary OSMnx chose for this place name. The centroid
        # is computed in degrees, which is close enough for a cut-off in miles.
        centroid = boundary.loc[0, "geometry"].centroid
        # A point stores (x, y) = (longitude, latitude).
        center_lat, center_lon = float(centroid.y), float(centroid.x)
    except Exception as exc:
        print(f"WARNING: could not geocode city centroid for {place_name}: {exc}")
        # None switches the distance filter off in stage 7.
        center_lat = center_lon = None

    # 4. Download the ACS 5-year values for every tract in the state in one
    #    request ("*" = all counties, all tracts). An error here (e.g. a bad API
    #    key or no network) stops Step 2 before any output file is written.
    census_client = Census(census_api_key)
    acs_records = census_client.acs5.state_county_tract(
        fields=(
            "NAME",  # the tract's name (tract number, county and state)
            "C17002_001E",  # ratio of income to poverty level: total
            "C17002_002E",  # ... under 0.50 (below half the poverty line)
            "C17002_003E",  # ... 0.50 to 0.99 (below the line, above half of it)
            "B01003_001E",  # total population
            "B08201_003E",  # households with 1 vehicle available
            "B08201_004E",  # households with 2 vehicles available
            "B08201_005E",  # households with 3 vehicles available
            "B08201_006E",  # households with 4 or more vehicles available
            "B09001_001E",  # population under 18
        ),
        state_fips=expected_state_fips,
        county_fips="*",
        tract="*",
        year=int(acs_year),
    )
    state_df = prepare_acs_dataframe(acs_records)

    # One counter per outcome; every Step 1 row adds 1 to exactly one of them, and
    # all five are saved in step02_metadata.json.
    #   missing_coordinates:      latitude or longitude blank or not a number
    #   invalid_coordinates:      latitude outside -90..90 or longitude outside
    #                             -180..180
    #   geocoder_failures:        the Census Geocoder request failed (known issue (e))
    #   no_tract_match:           no tract for the point, a tract in another
    #                             state, or no ACS row for the tract
    #   successful_tract_matches: tract found and its ACS values attached
    counters = {
        "missing_coordinates": 0,
        "invalid_coordinates": 0,
        "geocoder_failures": 0,
        "no_tract_match": 0,
        "successful_tract_matches": 0,
    }
    records: list[dict[str, Any]] = []

    # 5. Find each pharmacy's tract and attach that tract's ACS values.
    iterator = tqdm(source.iterrows(), total=len(source), desc="Step 2 census matching", unit="row")
    for _, row in iterator:
        # Step 1 already removes pharmacies without coordinates, so these two
        # checks are a safety net and their counters are normally 0.
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
        except Exception:
            counters["missing_coordinates"] += 1
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            counters["invalid_coordinates"] += 1
            continue
        # Ask the Census Geocoder which tract contains the point (one web request;
        # x = longitude, y = latitude). Known issue (e): any failure here is
        # counted and the pharmacy is skipped, with no retry.
        try:
            result = cg.coordinates(x=lon, y=lat)
            tracts = result.get("Census Tracts", []) if isinstance(result, dict) else []
        except Exception:
            counters["geocoder_failures"] += 1
            continue
        # No tract contains the point (e.g. a point outside the United States).
        if not tracts:
            counters["no_tract_match"] += 1
            continue

        # A point lies in only one tract, so the first record is used.
        geocoder_state, county_fips, tract_id, geoid = extract_census_geoids(tracts[0])
        # The ACS table holds only this state's tracts, so a point in another
        # state (e.g. across a state line) can't be matched and is dropped. If the
        # record has no state code, this check is skipped.
        if geocoder_state and geocoder_state != expected_state_fips:
            counters["no_tract_match"] += 1
            continue
        if not county_fips or not tract_id:
            counters["no_tract_match"] += 1
            continue
        # Find the tract's ACS row by county code plus tract code. Both are needed
        # because tract codes repeat between counties; the state is already fixed
        # by the download.
        tract_data = state_df[
            (state_df["county"] == county_fips) & (state_df["tract"] == tract_id)
        ]
        # The geocoder's tract is not in this ACS year's data.
        if tract_data.empty:
            counters["no_tract_match"] += 1
            continue

        # Poverty bands (C17002) and total population. Poverty_Rate is computed
        # from these when the row is built below (safe_poverty_rate).
        under50 = tract_data["C17002_002E"].iloc[0]
        # The 0.50 to 0.99 band only, not everyone under 1.00.
        under100 = tract_data["C17002_003E"].iloc[0]
        total_pop = tract_data["B01003_001E"].iloc[0]
        # Households with 1, 2, 3, and 4 or more vehicles (B08201_003E..006E).
        vehicle_values = [
            tract_data["B08201_003E"].iloc[0],
            tract_data["B08201_004E"].iloc[0],
            tract_data["B08201_005E"].iloc[0],
            tract_data["B08201_006E"].iloc[0],
        ]
        # A missing count is treated as 0 households.
        vehicle_values = [0.0 if pd.isna(value) else float(value) for value in vehicle_values]
        # vehicles = 1*n1 + 2*n2 + 3*n3 + 4*n4, e.g. 600 + 2*400 + 3*100 + 4*50 = 1900.
        # "4 or more" counts as 4, so vehicles are slightly undercounted where some
        # households have 5 or more.
        total_vehicles = (
            vehicle_values[0]
            + 2 * vehicle_values[1]
            + 3 * vehicle_values[2]
            + 4 * vehicle_values[3]
        )
        # pop_over_18 = total population - population under 18, i.e. people aged
        # 18 and over (e.g. 3000 - 700 = 2300). A missing under-18 count counts as
        # 0; a missing total gives NaN. Known issue (f): it is saved for reference
        # but not used for Vehicles_per_capita.
        under18 = tract_data["B09001_001E"].iloc[0]
        pop_over_18 = (
            float(total_pop) - (0.0 if pd.isna(under18) else float(under18))
            if not pd.isna(total_pop)
            else float("nan")
        )
        # Vehicles_per_capita = vehicles / total population of all ages, e.g.
        # 1900 / 3000 = 0.633; NaN if the population is 0 or missing. It is not
        # capped and can be above 1; Step 4 uses it as the car share C_i and caps
        # it at 1. Known issue (f): Hickok et al. divide by voting-age population.
        vehicles_per_capita = (
            total_vehicles / float(total_pop)
            if not pd.isna(total_pop) and float(total_pop) > 0
            else float("nan")
        )
        # The output row: the Step 1 identity columns, then the tract fields.
        # latitude and longitude are replaced by the checked numeric values.
        record = _identity_record(row)
        record.update(
            {
                "latitude": lat,
                "longitude": lon,
                "state": state_abbr,
                "state_fips": expected_state_fips,
                "county_fips": county_fips,
                "TRACT": tract_id,
                # The 11-digit GEOID, built from the codes if the geocoder gave none.
                "GEOID": geoid or f"{expected_state_fips}{county_fips}{tract_id}",
                "NAME": tract_data["NAME"].iloc[0],
                "C17002_001E": tract_data["C17002_001E"].iloc[0],
                "C17002_002E": under50,
                "C17002_003E": under100,
                "B01003_001E": total_pop,
                "pop_over_18": pop_over_18,
                "Vehicles_per_capita": vehicles_per_capita,
                "Poverty_Rate": safe_poverty_rate(under50, under100, total_pop),
            }
        )
        records.append(record)
        counters["successful_tract_matches"] += 1

    # 6. Check the matched rows: one run tag and unique pharmacy IDs, as in Step 1.
    matched = pd.DataFrame(records)
    if not matched.empty:
        validate_analysis_tag(matched, analysis_run_tag, "Step 2 matched rows")
        validate_pharmacy_ids(matched, "Step 2 matched rows")

    # 7. Distance filter: straight-line miles from the city centre (haversine).
    #    More than max_distance_miles away -> far_pharmacies.csv; a pharmacy
    #    exactly at the limit is kept. Skipped when stage 3 found no centre (then
    #    every matched pharmacy is kept) or when no pharmacy was matched.
    if center_lat is not None and not matched.empty:
        matched["distance_mi"] = matched.apply(
            lambda row: haversine(
                center_lat, center_lon, float(row["latitude"]), float(row["longitude"])
            ),
            axis=1,
        )
        far_mask = matched["distance_mi"] > float(max_distance_miles)
        far = matched.loc[far_mask].copy()
        retained = matched.loc[~far_mask].copy()
    else:
        # No filter: an empty far table with the same columns, and keep everyone.
        far = pd.DataFrame(columns=matched.columns)
        retained = matched.copy()

    # 8. Save the two tables, check them, then write the metadata file last.
    #    The distance column was only needed for the filter, so it is dropped from
    #    both tables. This keeps the two files' columns the same whether or not the
    #    filter ran (so the files do not record how far each pharmacy was).
    retained = retained.drop(columns=["distance_mi"], errors="ignore").reset_index(drop=True)
    far = far.drop(columns=["distance_mi"], errors="ignore").reset_index(drop=True)
    far.to_csv(far_output, index=False)
    retained.to_csv(tract_output, index=False)

    # Check both tables again (as in stage 6) before the metadata file is written.
    # Step 0 counts Step 2 as complete only when all three files exist, so a run
    # that stops here leaves a partial output, which Step 0 reports next time.
    if not retained.empty:
        validate_analysis_tag(retained, analysis_run_tag, str(tract_output))
        validate_pharmacy_ids(retained, str(tract_output))
    if not far.empty:
        validate_analysis_tag(far, analysis_run_tag, str(far_output))
        validate_pharmacy_ids(far, str(far_output))

    # Settings, counts (including the five counters) and file paths for this city.
    # input_pharmacy_count = retained + far + the four drop counters.
    metadata = {
        "pipeline_step": 2,
        "city": city,
        "state": state_abbr,
        ANALYSIS_TAG_COLUMN: analysis_run_tag,
        "created_utc": utc_now_iso(),
        "source_step01_csv": str(input_path.resolve()),
        "acs_year": int(acs_year),
        "max_distance_miles": float(max_distance_miles),
        "input_pharmacy_count": len(source),
        "retained_pharmacy_count": len(retained),
        "far_pharmacy_count": len(far),
        **counters,
        "tract_output": str(tract_output.resolve()),
        "far_output": str(far_output.resolve()),
    }
    write_json(metadata_output, metadata)
    print("\n=== Step 2 summary ===")
    print(f"Retained pharmacies: {len(retained)}")
    print(f"Far pharmacies: {len(far)}")
    print(f"Output: {tract_output}")
    return {
        "tract_output": str(tract_output),
        "far_output": str(far_output),
        "metadata_output": str(metadata_output),
    }


# Older name for attach_census_info, kept so that any code still using it keeps
# working. The old version worked through several folders at once; this one
# handles one city per call. Nothing in the current pipeline calls it (Step 0
# calls attach_census_info).
def tie_census_info(*args, **kwargs):
    """Same as attach_census_info (older name)."""
    return attach_census_info(*args, **kwargs)


# Run Step 2 for one city from the command line, without Step 0. For example:
#   python step02_attach_censustract_info.py pharmacy_data.csv step02_census
#       Milwaukee WI test_261011
# The Census API key is read from the CENSUS_API_KEY environment variable unless
# --census-api-key is given. Note: --acs-year defaults to 2022 here, while the
# function's own default is 2024.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Attach census tract information to pharmacies.")
    parser.add_argument("input_csv")
    parser.add_argument("output_folder")
    parser.add_argument("city")
    parser.add_argument("state")
    parser.add_argument("analysis_run_tag")
    parser.add_argument("--census-api-key", default=os.getenv("CENSUS_API_KEY", ""))
    parser.add_argument("--max-distance-miles", type=float, default=20.0)
    parser.add_argument("--acs-year", type=int, default=2022)
    args = parser.parse_args()
    attach_census_info(
        args.input_csv,
        args.output_folder,
        city=args.city,
        state=args.state,
        analysis_run_tag=args.analysis_run_tag,
        census_api_key=args.census_api_key,
        max_distance_miles=args.max_distance_miles,
        acs_year=args.acs_year,
    )

