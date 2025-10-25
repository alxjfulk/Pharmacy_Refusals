from __future__ import annotations# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 08:26:25 2025

@author: bigfo
"""

"""
cg_compat.py
A replacement for the core bits of the `censusgeocode` package:
- Geocode single-line addresses and (lon, lat) to get tract GEOIDs
- Fetch ACS5 variables for a tract

Design goals:
- No fragile transitive deps (works with requests>=2.32, urllib3>=2)
- Clear, explicit timeouts and retries
- Small, readable, well-documented functions that you can extend

Author: (Alex Fulk)
"""



import time
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- Configuration ----------

CENSUS_GEOCODER_BASE = "https://geocoding.geo.census.gov/geocoder"
DEFAULT_BENCHMARK = "Public_AR_Current"
DEFAULT_VINTAGE = "Current_Current"
DEFAULT_TIMEOUT = 30  # seconds

# You can bump the ACS vintage centrally here; override per-call if needed.
DEFAULT_ACS_VINTAGE = 2025
DEFAULT_ACS_DATASET = "acs5"

# ---------- Logging ----------
logger = logging.getLogger("cg_compat")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ---------- HTTP session with retries ----------

def _build_session() -> requests.Session:
    """
    Build a requests.Session with sane retry defaults for idempotent GETs.
    """
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {
            "User-Agent": "cg-compat/1.0 (+https://your-org.example)",
            "Accept": "application/json",
        }
    )
    return s

_SESSION = _build_session()

# ---------- Helpers ----------

def _first_or_none(seq):
    return seq[0] if seq else None

def _ensure_ok(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except Exception as e:
        # Surface more useful context in logs
        logger.error("HTTP %s for %s", resp.status_code, resp.url)
        raise

def _split_tract_geoid(tract_geoid: str) -> Tuple[str, str, str]:
    """
    Split an 11-digit GEOID into (state2, county3, tract6)
    """
    if len(tract_geoid) != 11 or not tract_geoid.isdigit():
        raise ValueError(f"Expected 11-digit tract GEOID, got: {tract_geoid!r}")
    return tract_geoid[:2], tract_geoid[2:5], tract_geoid[5:]

# ---------- Geocoder (address → geographies) ----------

def geocode_onelineaddress(
    address: str,
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    vintage: str = DEFAULT_VINTAGE,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict:
    """
    Geocode a single-line address and return the raw Census Geocoder JSON.
    This mirrors the 'onelineaddress' + returntype=geographies behavior.

    Returns the whole JSON so you can extract any geography you need.
    """
    params = {
        "address": address,
        "benchmark": benchmark,
        "vintage": vintage,
        "format": "json",
    }
    url = f"{CENSUS_GEOCODER_BASE}/geographies/onelineaddress"
    resp = _SESSION.get(url, params=params, timeout=timeout)
    _ensure_ok(resp)
    data = resp.json()
    return data


def geocode_address(
    *,
    street: str,
    city: str,
    state: str,
    zipcode: Optional[str] = None,
    benchmark: str = DEFAULT_BENCHMARK,
    vintage: str = DEFAULT_VINTAGE,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict:
    """
    Geocode a structured address (street, city, state[, zipcode]).
    Mirrors 'address' endpoint with returntype=geographies.
    """
    params = {
        "street": street,
        "city": city,
        "state": state,
        "benchmark": benchmark,
        "vintage": vintage,
        "format": "json",
    }
    if zipcode:
        params["zip"] = zipcode
    url = f"{CENSUS_GEOCODER_BASE}/geographies/address"
    resp = _SESSION.get(url, params=params, timeout=timeout)
    _ensure_ok(resp)
    return resp.json()


def geocode_coordinates(
    *,
    lon: float,
    lat: float,
    benchmark: str = DEFAULT_BENCHMARK,
    vintage: str = DEFAULT_VINTAGE,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict:
    """
    Reverse-geocode coordinates (lon, lat) to geographies.
    Mirrors the 'coordinates' endpoint with returntype=geographies.
    """
    params = {
        "x": lon,
        "y": lat,
        "benchmark": benchmark,
        "vintage": vintage,
        "format": "json",
    }
    url = f"{CENSUS_GEOCODER_BASE}/geographies/coordinates"
    resp = _SESSION.get(url, params=params, timeout=timeout)
    _ensure_ok(resp)
    return resp.json()

# ---------- Extractors (normalize geocoder output) ----------

def extract_geoid_from_geographies(geo_json: Dict, level: str = "Census Tracts") -> str:
    """
    Given a geocoder JSON (from any of the three functions above),
    extract the first GEOID for the specified geography level.
    Default level is 'Census Tracts'. Other common values:
      'States', 'Counties', 'Census Blocks', 'Census Block Groups'
    """
    # onelineaddress/address responses use result.addressMatches[0].geographies
    matches = (
        geo_json.get("result", {})
        .get("addressMatches", [])
    )
    if matches:
        geos = matches[0].get("geographies", {})
        candidates = geos.get(level) or []
        first = _first_or_none(candidates)
        if first and "GEOID" in first:
            return first["GEOID"]

    # coordinates responses use result.geographies directly
    geos = geo_json.get("result", {}).get("geographies", {})
    candidates = geos.get(level) or []
    first = _first_or_none(candidates)
    if first and "GEOID" in first:
        return first["GEOID"]

    raise ValueError(f"No {level!r} GEOID found in geocoder response.")

# ---------- ACS pulls (tract-level) ----------

@lru_cache(maxsize=4096)
def acs_tract(
    tract_geoid: str,
    variables: Tuple[str, ...],
    *,
    year: int = DEFAULT_ACS_VINTAGE,
    dataset: str = DEFAULT_ACS_DATASET,
    api_key: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, str]:
    """
    Fetch selected ACS variables for a single tract GEOID.
    Caches by (tract_geoid, variables, year, dataset, api_key) to avoid repeated calls.

    Returns a dict: {var1: value1, var2: value2, ..., "state": "..", "county": "..", "tract": ".."}
    Values are strings as returned by the API; cast as needed upstream.
    """
    if not variables:
        raise ValueError("Provide at least one ACS variable name (e.g., 'B01003_001E').")

    state, county, tract = _split_tract_geoid(tract_geoid)
    base = f"https://api.census.gov/data/{year}/acs/{dataset}"
    params = {
        "get": ",".join(variables),
        "for": f"tract:{tract}",
        "in": f"state:{state} county:{county}",
    }
    if api_key:
        params["key"] = api_key

    resp = _SESSION.get(base, params=params, timeout=timeout)
    _ensure_ok(resp)
    rows = resp.json()
    if not rows or len(rows) < 2:
        raise ValueError(f"No ACS rows returned for tract {tract_geoid}")

    header, values = rows[0], rows[1]
    return dict(zip(header, values))

# ---------- Convenience wrappers you can drop into pipelines ----------

def address_to_tract_geoid(address: str) -> str:
    """
    One-liner to get an 11-digit tract GEOID from a single-line address.
    """
    data = geocode_onelineaddress(address)
    geoid = extract_geoid_from_geographies(data, level="Census Tracts")
    logger.info("Address → tract GEOID: %s", geoid)
    return geoid


def coords_to_tract_geoid(lon: float, lat: float) -> str:
    """
    One-liner to get an 11-digit tract GEOID from (lon, lat).
    """
    data = geocode_coordinates(lon=lon, lat=lat)
    geoid = extract_geoid_from_geographies(data, level="Census Tracts")
    logger.info("Coordinates → tract GEOID: %s", geoid)
    return geoid


def tract_demographics(
    tract_geoid: str,
    variables: List[str],
    *,
    year: int = DEFAULT_ACS_VINTAGE,
    dataset: str = DEFAULT_ACS_DATASET,
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """
    Friendly facade over acs_tract(), accepting a list and returning a dict.
    """
    # Convert vars to tuple for cache key stability
    result = acs_tract(tract_geoid, tuple(variables), year=year, dataset=dataset, api_key=api_key)
    logger.info("Fetched %d ACS vars for tract %s (year=%s, dataset=%s)",
                len(variables), tract_geoid, year, dataset)
    return result

def coordinates(*, x: float, y: float, **kwargs):
    """
    Compatibility shim for censusgeocode.coordinates(x=..., y=...).
    """
    geo_json = geocode_coordinates(lon=x, lat=y, **kwargs)
    return geo_json.get("result", {}).get("geographies", {})