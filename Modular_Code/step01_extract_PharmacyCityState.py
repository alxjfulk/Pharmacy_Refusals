# -*- coding: utf-8 -*-
"""Step 1 - find the pharmacies in one city and record where they are.

Step 1 builds the list of pharmacies that every later step works with. It asks the
national NPI Registry for every pharmacy organization in the city, checks each one
with Google Places to confirm it is open and to get its coordinates, removes
duplicates, and gives each remaining pharmacy an ID for this analysis run.

WHERE THE DATA COME FROM
------------------------
- NPI Registry (npiregistry.cms.hhs.gov; public, no key needed). Every US health
  care provider has a National Provider Identifier (NPI). NPI-1 numbers belong to
  individual people (for example, a pharmacist); NPI-2 numbers belong to
  organizations (for example, a pharmacy store). Step 1 always asks for NPI-2
  records whose taxonomy (type of provider) description is "Pharmacy".
- Google Places API, Text Search (needs a Google Places API key). For each NPI
  record Step 1 searches for "<organization name> <address>" and takes the first
  result's business status (OPERATIONAL, CLOSED_TEMPORARILY or CLOSED_PERMANENTLY)
  and its latitude and longitude.

WHAT STEP 1 READS (settings passed in by Step 0; there are no input files)
--------------------------------------------------------------------------
- City name and two-letter state abbreviation, e.g. ("Milwaukee", "WI").
- ''ANALYSIS_RUN_TAG'': the name of this run. It becomes part of every pharmacy ID.
- ''GOOGLE_PLACES_API_KEY''.
- ''NPI_ADDRESS_PURPOSE'' (Step 0 uses "PRIMARY"): which of a provider's addresses
  the registry compares with the city and state.
- ''GOOGLE_REQUEST_DELAY_SECONDS'' (Step 0 uses 1.0): pause after each Google request.

WHAT STEP 1 WRITES
------------------
Everything goes in the city's Step 1 folder for this run:
    pipeline_results/analysis_runs/<tag>/<CityState>/step01_pharmacy/

- pharmacy_data.csv: the final list; one row = one open pharmacy. Step 2 reads this.
  Columns, in order: ''pharmacy_id'', ''analysis_run_tag'', ''NPI'',
  ''OrganizationName'', ''AddressLocation'' (practice address), ''google_place_id'',
  ''status'' (always OPERATIONAL here), ''candidates'' (how many places Google
  returned), ''latitude'' and ''longitude'' (decimal degrees, from Google), then
  ''addresses'' (all of the NPI record's addresses, as text) and ''basic.status''
  (the NPI record's own status field, when the registry sends it).
- extraction_metadata.json: the settings used, the number of records at each stage
  (see the worked example below) and the paths of the files. Step 0 reads it
  (''processing_status'', ''analysis_run_tag'', ''final_pharmacy_count'') to decide
  whether Step 1 is complete for the city.
- WARNING_NPI_CEILING.txt: written only when the NPI download reached its
  1,200-record ceiling, so the list is probably incomplete (see Known issues).
- mailing_only.csv: one row = one NPI record that has a mailing address but no
  practice-location address, so it can't be put on a map. Removed. All the fields
  the registry sent are kept.
- duplicates_address.csv: one row = one NPI record removed because another record
  has exactly the same practice address. Only the removed records are listed; the
  record that was kept goes on to the Google check.
- inactive_pharmacies.csv: one row = one record that Google did not report as
  OPERATIONAL (closed, not found, or the request failed); ''status'' says which.
  Removed.
- missing_coordinates.csv: one row = one OPERATIONAL record for which Google gave
  no latitude or longitude. Removed. Usually empty.
- duplicates_latlon.csv: one row = one record in a group of records with exactly
  the same Google coordinates. Every member of each group is listed, including the
  one that was kept (the first of the group).

HOW THE PIECES FIT TOGETHER (extract_pharmacy_data, at the bottom)
------------------------------------------------------------------
1. Check the settings, and stop if Step 1 output already exists.
2. Download the NPI records, 200 per request ........ u3_request (retries: make_retry)
3. Check every record is NPI-2; if the download reached the 1,200-record
   ceiling, write WARNING_NPI_CEILING.txt ............ write_npi_ceiling_warning
4. Turn the records into a table with one practice address
   per record ........................................ _get_practice_address
5. Set aside records with only a mailing address ..... _has_mailing_only
6. Remove records with exactly the same practice address, keeping the most
   recently updated NPI record (duplicates_address.csv).
7. Ask Google Places about every record and keep the
   OPERATIONAL ones .......... filter_inactive_pharmacies, check_google_places_status
8. Drop records without coordinates, remove records with identical
   coordinates, and number the rest .................. finalize_active_pharmacies
9. Save pharmacy_data.csv and extraction_metadata.json.

Nothing is written to the folder before stage 3, that is, before the whole NPI
download has succeeded.

WORKED EXAMPLES
---------------
Paging (at most 200 records per request; the registry accepts skip values up to 1,000):
    A city with 437 records:   skip=0 -> 200, skip=200 -> 200, skip=400 -> 37.
                               37 < 200, so that was the last page: 437 records.
    A city with 1,500 records: skip=0, 200, 400, 600, 800 and 1000 -> 200 each,
                               1,200 records. The skip=1000 page is full and no
                               higher skip is allowed, so the last 300 records (by
                               organization name) are never downloaded, and
                               WARNING_NPI_CEILING.txt is written.

Records at each stage (made-up numbers; the real ones are in
extraction_metadata.json under the names in brackets):
      437  NPI records downloaded           [total_npi_records_retrieved]
    -   2  mailing-only records             [mailing_only_count]
    -  11  exact-address duplicates         [address_duplicates_removed]
    = 424  records sent to the Google check
    -  30  not OPERATIONAL                  [inactive_records_returned_by_google]
    = 394  OPERATIONAL                      [active_records_returned_by_google]
    -   0  without coordinates              [records_removed_missing_coordinates]
    -   3  identical-coordinate duplicates  [exact_coordinate_duplicates_removed]
    = 391  pharmacies in pharmacy_data.csv  [final_pharmacy_count]

Pharmacy IDs:
    ''pharmacy_id'' = <CityState>_<tag>_P<running number, 4 digits>
    e.g. MilwaukeeWI_test_261011_P0001, MilwaukeeWI_test_261011_P0002, ...

PHARMACY IDS
------------
IDs are numbered 1, 2, 3, ... in the order the rows reach the end of Step 1. When
the NPI records include a last-updated date (''basic.last_updated''), that order
runs from the most recently updated record to the oldest, because of the sort in
stage 6; it is not alphabetical. An ID is unique only within this analysis run: the
same pharmacy can get a different number in another run, and nothing links IDs
between runs (no lookup table, no matching by coordinates). To compare runs, match
on ''NPI'' or ''google_place_id''. ''analysis_run_tag'' is the only column that
records which run a row belongs to.

RERUNNING
---------
Step 0 skips a city whose Step 1 output is complete and stops if it is partial. A
failed NPI download (network problem, registry error, no records) leaves the
folder empty, so the city can be run again under the same tag. A failure after
that point (for example, a city where no pharmacy passes the Google check) leaves
some of the review files behind; Step 0 then treats the folder as partial, and you
need a new ''ANALYSIS_RUN_TAG'' or must empty that folder by hand.

KNOWN ISSUES AND LIMITATIONS
----------------------------
- Known issue (NPI ceiling): the registry returns at most 1,200 records per query
  (200 per page, skip up to 1,000), sorted by organization name. In a very large
  city the list is cut off alphabetically: pharmacies whose names come late in the
  alphabet are missing. Step 1 writes WARNING_NPI_CEILING.txt, sets
  ''npi_retrieval_ceiling_reached'' in the metadata, and carries on with the
  records it has. This is treated as a documented limitation.
- Known issue (Google first result): only Google's first text-search result is
  used. If Google ranks a different place first, the status and coordinates
  belong to that place. The number of results is saved in ''candidates'' (and the
  number of searches with more than one result in
  ''ambiguous_google_candidate_count''), but the other results are not examined.
  Every row whose status is not OPERATIONAL is excluded, including
  CLOSED_TEMPORARILY and rows whose request failed (REQUEST_DENIED, ERROR), so a
  temporary Google problem removes a pharmacy without stopping the run. Those rows
  are kept in inactive_pharmacies.csv and counted in ''google_status_breakdown''.
- Duplicates are found by exact matches only: identical address text (stage 6)
  and identical coordinates (stage 8). See finalize_active_pharmacies for why.
- Records with neither a practice address nor a mailing address are dropped in
  stage 5 without being saved or counted. They should be rare.
"""

from __future__ import annotations

import json
import os
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import urllib3

from pipeline_common_260801 import (
    ANALYSIS_TAG_COLUMN,
    city_state_key,
    normalize_analysis_run_tag,
    utc_now_iso,
    validate_analysis_tag,
    validate_pharmacy_ids,
    write_json,
)


# =============================================================================
# NPI Registry download settings
# =============================================================================

# The registry sends at most 200 records per request (one "page") ...
NPI_PAGE_LIMIT = 200
# ... and accepts a skip (the number of records to pass over) of at most 1,000, so
# the last page that can be requested holds records 1,001 to 1,200.
NPI_MAX_SKIP = 1000
# The most records one query can return: 1,000 + 200 = 1,200 (known issue; see the
# top of this file).
NPI_MAX_RETRIEVABLE = NPI_MAX_SKIP + NPI_PAGE_LIMIT
NPI_ENUMERATION_TYPE = "NPI-2"  # organizations; Step 1 never asks for NPI-1 (people)
# Allowed values for npi_address_purpose (Step 0's NPI_ADDRESS_PURPOSE).
VALID_NPI_ADDRESS_PURPOSES = {"LOCATION", "MAILING", "PRIMARY", "SECONDARY"}


# =============================================================================
# Web requests: retries, and differences between urllib3 versions
# =============================================================================

def make_retry(total: int = 10, backoff_factor: float = 1.5):
    """Build the retry rule used for every web request in Step 1 (NPI and Google).

    A request that fails for a temporary reason is tried again, up to ''total''
    more times (default 10, so up to 11 attempts in all). It is retried after:
    - a connection failure or timeout (no answer from the server), or
    - an answer with HTTP status 429 (too many requests) or 500, 502, 503, 504
      (server errors).
    Both GET (used for NPI) and POST (used for Google) may be retried.

    Backoff (waiting longer after each subsequent failure). Before retry n,
    urllib3 waits
        0 seconds for n = 1, then backoff_factor * 2 ** (n - 1) seconds, at most 120
    With backoff_factor = 1.5 the waits are 0, 3, 6, 12, 24, 48, 96, 120, 120 and
    120 seconds: about 9 minutes in all if every attempt fails. For 429 and 503,
    if the server says how long to wait (a Retry-After header), urllib3 waits that
    long instead.

    After the last retry, the final error answer is returned rather than raised
    (raise_on_status=False), so the caller can read it: the NPI download then
    stops with the HTTP status, and the Google check records REQUEST_DENIED or
    ERROR. A connection that still fails after the last retry raises an error.

    urllib3 1.26 and later (including 2.x) call the list of methods
    ''allowed_methods''; older 1.x versions call it ''method_whitelist''. The
    try/except uses whichever name the installed version accepts.
    """
    retry_kwargs = dict(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        # "Too many requests" and server errors: worth trying again.
        status_forcelist=(429, 500, 502, 503, 504),
        # After the last retry, hand back the error answer instead of raising.
        raise_on_status=False,
    )
    # POST must be listed here: urllib3 does not retry POST requests by default.
    methods = frozenset(["GET", "POST"])
    try:
        return urllib3.util.Retry(allowed_methods=methods, **retry_kwargs)
    except TypeError:
        # Reached only with urllib3 older than 1.26, which uses the old name.
        return urllib3.util.Retry(method_whitelist=methods, **retry_kwargs)


def u3_request(method: str, url: str, **kwargs):
    """Send one web request with Step 1's timeouts and retries, on any urllib3 version.

    Timeouts: 10 seconds to connect and 60 seconds to wait for data, unless the
    caller passes ''timeout''. Retries: make_retry() unless the caller passes
    ''retries''. Other keyword arguments (''fields'', ''headers'', ''body'') go
    straight to urllib3. Returns the urllib3 response without checking its HTTP
    status; the callers do that.

    urllib3 2.x has a ready-made urllib3.request() function. In urllib3 1.x,
    urllib3.request is a module rather than a function, so a PoolManager (urllib3's
    connection handler) is created instead. Both send the same request.
    """
    timeout = kwargs.pop("timeout", urllib3.Timeout(connect=10, read=60))
    retries = kwargs.pop("retries", make_retry())
    # urllib3 2.x: use the urllib3.request() function.
    if (
        hasattr(urllib3, "request")
        and callable(getattr(urllib3, "request"))
        and not isinstance(urllib3.request, types.ModuleType)
    ):
        return urllib3.request(method, url, timeout=timeout, retries=retries, **kwargs)
    # urllib3 1.x: no such function, so go through a PoolManager.
    return urllib3.PoolManager().request(
        method, url, timeout=timeout, retries=retries, **kwargs
    )


def u3_json(response):
    """Read the JSON body of a urllib3 response as Python data (a dict for these APIs).

    urllib3 2.x responses have a .json() method; 1.x responses do not, so the raw
    bytes are decoded as UTF-8 text and parsed. Raises an error if the body is not
    valid JSON.
    """
    return (
        response.json()
        if hasattr(response, "json")
        else json.loads(response.data.decode("utf-8"))
    )


# =============================================================================
# Warning file for the NPI ceiling
# =============================================================================

def _write_text_marker(output_dir: Path, filename: str, heading: str, message: str) -> str:
    """Write a short text file (a heading line, then a message) and return its path.

    Creates the folder if needed. A plain text file is used so that the warning
    stands out when someone looks through the output folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(f"{heading}\n{message}\n", encoding="utf-8")
    return str(path)


def write_npi_ceiling_warning(output_dir: Path, city: str, state: str, message: str) -> str:
    """Write WARNING_NPI_CEILING.txt: the NPI download hit its 1,200-record ceiling.

    extract_pharmacy_data calls this when the last page it may request
    (skip = 1,000) was full, which means the registry may hold more records that
    it would not send. The heading names the city and state; ''message'' goes
    below it. Returns the file path, which is saved in the metadata as
    ''npi_ceiling_warning_file''.
    """
    return _write_text_marker(
        output_dir,
        "WARNING_NPI_CEILING.txt",
        f"NPI RETRIEVAL CEILING WARNING: {city}, {state}",
        message,
    )


# =============================================================================
# Google Places check: is each pharmacy open, and where is it?
# =============================================================================

def check_google_places_status(name: str, address: str, api_key: str):
    """Ask Google Places about one pharmacy and return what its first result says.

    Sends a Text Search request (Places API, places:searchText) for
    "<name> <address>", e.g. "ABC PHARMACY LLC 1234 W MAIN ST, MILWAUKEE, WI 53202",
    much as if it were typed into Google Maps. The name is the organization's
    legal name from the NPI record, which can differ from the name on the store.

    Returns ''(status, latitude, longitude, candidate_count, place_id)'':
    - status: the first result's businessStatus (OPERATIONAL, CLOSED_TEMPORARILY
      or CLOSED_PERMANENTLY), or UNKNOWN if the first result has no businessStatus.
      When there is no usable result, one of Step 1's own codes instead:
        ZERO_RESULTS    Google found nothing.
        REQUEST_DENIED  Google answered with an error message (bad key, quota,
                        billing, ...), which is printed. This label is used for
                        every error Google reports, not only refusals.
        ERROR           the request itself failed (network failure after all
                        retries, or an answer that could not be read); printed.
    - latitude, longitude: the first result's location in decimal degrees, or None.
    - candidate_count: how many places Google returned (0 if none).
    - place_id: Google's ID for the first result, or None.

    Known issue: only the first result is used, even when Google returns several
    (candidate_count > 1).

    Every failure becomes a status rather than an error, so one bad request does
    not stop the run.
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "X-Goog-Api-Key": api_key,
        # Ask only for the three fields Step 1 uses. This API requires a field list,
        # and the fields requested also set the price of each request.
        "X-Goog-FieldMask": "places.businessStatus,places.location,places.id",
        "Content-Type": "application/json",
    }
    payload = {"textQuery": f"{name} {address}"}
    try:
        response = u3_request(
            "POST", url, headers=headers, body=json.dumps(payload).encode("utf-8")
        )
        data = u3_json(response)
        # Google reports problems (bad key, quota, ...) as {"error": {"message": ...}}.
        if isinstance(data, dict) and "error" in data:
            message = data["error"].get("message", "Unknown error")
            print(f"Google Places API error: {message}")
            return "REQUEST_DENIED", None, None, 0, None
        places = data.get("places", []) if isinstance(data, dict) else []
        if not places:
            return "ZERO_RESULTS", None, None, 0, None
        # Known issue: only the first (best-ranked) result is used.
        first = places[0]
        location = first.get("location") or {}
        return (
            first.get("businessStatus", "UNKNOWN"),
            location.get("latitude"),
            location.get("longitude"),
            len(places),
            first.get("id"),
        )
    except Exception as exc:
        # A network failure after all retries, or an answer that is not JSON.
        print(f"Error querying Google Places for {name!r} at {address!r}: {exc}")
        return "ERROR", None, None, 0, None


def filter_inactive_pharmacies(
    df: pd.DataFrame,
    api_key: str,
    output_dir: Path,
    request_delay_seconds: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Check every record with Google Places and keep the OPERATIONAL ones.

    Why: a registered NPI does not prove the store is still open, so Google is used
    to confirm that it is operating and to get its coordinates.

    Goes through ''df'' one row at a time (one row = one NPI record, with
    ''OrganizationName'' and ''AddressLocation''), calls check_google_places_status,
    and adds five columns: ''latitude'', ''longitude'', ''status'', ''candidates''
    and ''google_place_id''. Rows whose status is OPERATIONAL are returned. All
    others are saved to inactive_pharmacies.csv in ''output_dir'': closed (including
    CLOSED_TEMPORARILY), not found, failed requests, and MISSING_DATA rows (no name
    or no address, so nothing was sent to Google).

    Pause between requests: after each Google request the code waits
    ''request_delay_seconds'' (Step 0 passes GOOGLE_REQUEST_DELAY_SECONDS = 1.0; the
    0.5 default applies only when Step 1 runs on its own). Spacing out the requests
    makes "too many requests" errors less likely. It also sets the minimum time this
    stage takes: with 400 records and a 1.0-second delay, at least 400 seconds
    (about 7 minutes), plus the time Google takes to answer.

    Returns ''(active, summary)''. ''active'' has the columns of ''df'' plus the
    five new ones. ''summary'' holds the numbers saved in extraction_metadata.json:
    API calls made, searches with more than one candidate, active and inactive
    counts, the count for each status, elapsed seconds, and the path of
    inactive_pharmacies.csv.

    Known issue: only Google's first result is used, and every non-OPERATIONAL row
    is excluded, including rows whose request failed (see the top of this file).
    """
    active_rows: list[dict[str, Any]] = []
    inactive_rows: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()  # how many rows got each status
    ambiguous_count = 0  # searches where Google returned more than one place
    total_calls = 0
    started = time.time()

    for _, row in df.iterrows():
        name = row.get("OrganizationName")
        address = row.get("AddressLocation")
        if pd.isna(name) or pd.isna(address):
            # Nothing to search for, so no request is sent.
            status, lat, lng, count, place_id = "MISSING_DATA", None, None, 0, None
        else:
            status, lat, lng, count, place_id = check_google_places_status(
                str(name), str(address), api_key
            )
            total_calls += 1
            if count > 1:
                ambiguous_count += 1
            # Pause before the next request (see the docstring); max() turns a
            # negative setting into no pause.
            time.sleep(max(0.0, float(request_delay_seconds)))

        status_counter[status] += 1
        # The NPI columns plus the five Google columns.
        record = row.to_dict()
        record.update(
            {
                "latitude": lat,
                "longitude": lng,
                "status": status,
                "candidates": count,
                "google_place_id": place_id,
            }
        )
        print(
            f"Checked {name!r} @ {address!r} -> status={status}, candidates={count}, "
            f"place_id={place_id}, lat={lat}, lon={lng}"
        )
        # Only OPERATIONAL is kept; every other status goes to the inactive list.
        (active_rows if status == "OPERATIONAL" else inactive_rows).append(record)

    # Same column order in both tables: the NPI columns, then the Google columns.
    extra = ["latitude", "longitude", "status", "candidates", "google_place_id"]
    ordered = list(df.columns) + [col for col in extra if col not in df.columns]
    active = pd.DataFrame(active_rows, columns=ordered)
    inactive = pd.DataFrame(inactive_rows, columns=ordered)
    # Written even when empty, so the file is always there to review.
    inactive_path = output_dir / "inactive_pharmacies.csv"
    inactive.to_csv(inactive_path, index=False)

    elapsed = time.time() - started
    print("\n--- Google Places summary ---")
    print(f"API calls: {total_calls}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Ambiguous candidate sets: {ambiguous_count}")
    print(f"Status counts: {dict(status_counter)}")

    return active, {
        "google_places_api_call_count": total_calls,
        "ambiguous_google_candidate_count": ambiguous_count,
        "active_records_returned_by_google": len(active),
        "inactive_records_returned_by_google": len(inactive),
        "google_status_breakdown": dict(status_counter),
        "google_elapsed_seconds": elapsed,
        "inactive_output_path": str(inactive_path),
    }


# =============================================================================
# Final clean-up and pharmacy IDs
# =============================================================================

def finalize_active_pharmacies(
    df_active: pd.DataFrame,
    city: str,
    state: str,
    analysis_run_tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Drop rows without coordinates, remove coordinate duplicates, and assign IDs.

    Input: the OPERATIONAL rows from filter_inactive_pharmacies. Returns four things:
    - final: one row per pharmacy, with ''pharmacy_id'' and ''analysis_run_tag''
      added as the first two columns;
    - missing: rows whose latitude or longitude is missing or not a number (the
      caller saves them as missing_coordinates.csv);
    - duplicates: every row whose (latitude, longitude) pair is shared with another
      row, including the row that was kept (saved as duplicates_latlon.csv);
    - summary: counts and the first and last ID, for extraction_metadata.json.

    Why identical coordinates are removed: two NPI records can describe the same
    store, for example with the address written two ways ("1234 W MAIN ST" and
    "1234 W MAIN ST STE 1") that the exact-address check in extract_pharmacy_data
    does not catch. Google then returns the same place, with the same coordinates,
    for both records, and keeping both would count one store twice. In each group
    the first row is kept, which is the most recently updated NPI record because of
    the sort in extract_pharmacy_data.

    Exact equality is intentional. Coordinates are not rounded and no distance
    tolerance is used, because in dense areas separate pharmacies can be very close
    together. Limitation: two different pharmacies that Google places at exactly
    the same point (for example, in one building) would be merged into one; check
    duplicates_latlon.csv.

    IDs: ''pharmacy_id'' = <CityState>_<tag>_P<number>, numbered from 1 in the
    current row order and padded to four digits, e.g. MilwaukeeWI_test_261011_P0001,
    MilwaukeeWI_test_261011_P0002, ... Before returning, the IDs and tag are checked
    (one tag on every row; no missing or repeated IDs).

    Raises ValueError if the latitude or longitude column is missing, or if no
    pharmacy is left (the tag check then finds no rows).
    """
    required = {"latitude", "longitude"}
    missing_columns = sorted(required - set(df_active.columns))
    if missing_columns:
        raise ValueError(f"Active pharmacy data are missing columns: {missing_columns}.")

    working = df_active.copy()
    # Text or blank values become NaN, so they count as missing just below.
    working["latitude"] = pd.to_numeric(working["latitude"], errors="coerce")
    working["longitude"] = pd.to_numeric(working["longitude"], errors="coerce")
    # A row counts as missing coordinates if either value is missing.
    missing_mask = working[["latitude", "longitude"]].isna().any(axis=1)
    missing_rows = working.loc[missing_mask].copy()
    working = working.loc[~missing_mask].copy()

    # keep=False marks every member of each group of identical coordinates, so the
    # review file shows the kept row next to the ones that were removed.
    duplicate_mask = working.duplicated(
        subset=["latitude", "longitude"], keep=False
    )
    duplicate_rows = working.loc[duplicate_mask].copy()
    before = len(working)
    # Keep the first row of each group (the most recently updated NPI record).
    working = (
        working.drop_duplicates(subset=["latitude", "longitude"], keep="first")
        .reset_index(drop=True)
    )

    safe_tag = normalize_analysis_run_tag(analysis_run_tag)
    prefix = city_state_key(city, state)  # e.g. "MilwaukeeWI"
    # Insert the tag column first and then the ID in front of it, so the ID ends up
    # as the first column.
    working.insert(0, ANALYSIS_TAG_COLUMN, safe_tag)
    working.insert(
        0,
        "pharmacy_id",
        # {number:04d} pads the running number to four digits: 1 -> "0001".
        [f"{prefix}_{safe_tag}_P{number:04d}" for number in range(1, len(working) + 1)],
    )
    # Final checks: the same tag on every row; IDs present and unique.
    validate_analysis_tag(working, safe_tag, "Step 1 final pharmacy data")
    ids = validate_pharmacy_ids(working, "Step 1 final pharmacy data")
    return working, missing_rows, duplicate_rows, {
        "records_removed_missing_coordinates": len(missing_rows),
        "exact_coordinate_duplicates_removed": before - len(working),
        "final_pharmacy_count": len(working),
        "first_pharmacy_id": ids[0] if ids else None,
        "last_pharmacy_id": ids[-1] if ids else None,
    }


# =============================================================================
# Reading the addresses in an NPI record
# =============================================================================

def _get_practice_address(addresses: Any):
    """Return the practice-location address of one NPI record as one line of text.

    ''addresses'' is the record's list of addresses from the registry. Each has an
    ''address_purpose'': LOCATION (where the pharmacy actually is) or MAILING (where
    its mail goes). The first LOCATION address is written as
        "<street line 1>, <city>, <state> <5-digit ZIP>"
        e.g. "1234 W MAIN ST, MILWAUKEE, WI 53202"
    A nine-digit ZIP such as 532021234 is cut to its first five digits, 53202.
    This text is what Google is searched with and what the exact-address duplicate
    check compares. Only street line 1 is used, so a suite or unit number on line 2
    is left out.

    Returns NaN (missing) if there is no LOCATION address or ''addresses'' is not a
    list.
    """
    if isinstance(addresses, list):
        for address in addresses:
            if address.get("address_purpose") == "LOCATION":
                street = str(address.get("address_1", "")).strip()
                city = str(address.get("city", "")).strip()
                state = str(address.get("state", "")).strip()
                zip5 = str(address.get("postal_code", ""))[:5]
                return f"{street}, {city}, {state} {zip5}"
    return np.nan


def _has_mailing_only(addresses: Any) -> bool:
    """Return True if an NPI record has a mailing address but no practice address.

    Such a record can't be put on a map, so extract_pharmacy_data saves it to
    mailing_only.csv and removes it. Returns False when ''addresses'' is not a list.
    """
    if not isinstance(addresses, list):
        return False
    purposes = {address.get("address_purpose") for address in addresses}
    return "LOCATION" not in purposes and "MAILING" in purposes


# =============================================================================
# Step 1 for one city: runs the pieces above in order
# =============================================================================

def extract_pharmacy_data(
    city: str,
    state: str,
    output_folder: str | os.PathLike[str],
    api_key: str,
    *,
    analysis_run_tag: str,
    npi_address_purpose: str = "PRIMARY",
    google_request_delay_seconds: float = 0.5,
) -> str:
    """Run Step 1 for one city and return the path of pharmacy_data.csv.

    Step 0 calls this once per city when the city's Step 1 folder is empty. The
    numbered comments below match the numbered list at the top of this file.

    Arguments:
    - city, state: e.g. "Milwaukee", "WI" (the state is changed to capitals).
    - output_folder: the city's step01_pharmacy folder; created if needed.
    - api_key: the Google Places API key; must not be blank.
    - analysis_run_tag: this run's tag, tidied by normalize_analysis_run_tag
      (characters other than letters, digits, ".", "_" and "-" become "-").
    - npi_address_purpose: which address the registry compares with the city and
      state: LOCATION, MAILING, PRIMARY or SECONDARY. Step 0 uses PRIMARY (the
      main practice location). The address saved is always the record's practice
      location (see _get_practice_address), so with another setting a saved
      address may lie outside the city.
    - google_request_delay_seconds: pause after each Google request. Step 0 passes
      GOOGLE_REQUEST_DELAY_SECONDS (1.0); the 0.5 default applies only when Step 1
      runs on its own.

    Stops with an error, before writing anything, if: Step 1 output already exists,
    the API key is blank, npi_address_purpose is not one of the four values, an NPI
    request fails, the registry returns no records, or a record is not NPI-2. An
    error after that point can leave review files behind (see RERUNNING at the top
    of this file).

    Existing output is never overwritten. Step 0 checks the folder first: it skips
    a complete Step 1 and stops on a partial one, asking for a new
    ''ANALYSIS_RUN_TAG''. The check at the start of this function is a second
    safeguard for when Step 1 is run on its own.
    """
    # 1. Check the settings, and stop if Step 1 output already exists.
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pharmacy_data.csv"
    metadata_file = output_dir / "extraction_metadata.json"
    if output_file.exists() or metadata_file.exists():
        raise FileExistsError(
            f"Step 1 output already exists in {output_dir}. The main should validate "
            "and skip complete output rather than overwrite it."
        )

    city = str(city).strip()
    state = str(state).strip().upper()
    tag = normalize_analysis_run_tag(analysis_run_tag)
    if not api_key or not str(api_key).strip():
        raise ValueError("Step 1 requires a nonblank Google Places API key.")

    address_purpose = str(npi_address_purpose).strip().upper()
    if address_purpose not in VALID_NPI_ADDRESS_PURPOSES:
        raise ValueError(f"Invalid npi_address_purpose={address_purpose!r}.")

    print(f"Analysis run tag: {tag}")
    # 2. Download the NPI records, one page (up to 200 records) per request.
    all_results: list[dict[str, Any]] = []
    skip = 0  # records to pass over: 0, 200, 400, ..., up to 1,000
    ceiling_reached = False  # set to True if the last page allowed is full
    base_url = "https://npiregistry.cms.hhs.gov/api/"

    while skip <= NPI_MAX_SKIP:
        # The search sent to the registry. "Pharmacy" is matched against the
        # provider's taxonomy (type of provider) description. The registry sorts
        # the results by organization name, so the pages come in name order.
        params = {
            "taxonomy_description": "Pharmacy",
            "enumeration_type": NPI_ENUMERATION_TYPE,
            "address_purpose": address_purpose,
            "city": city,
            "state": state,
            "limit": NPI_PAGE_LIMIT,
            "skip": skip,
            "version": "2.1",
            # Active NPIs only; saved in the metadata as npi_api_status_filter.
            "status": "active",
        }
        # The full request address is printed so it can be pasted into a browser to
        # see exactly what the registry sent back.
        print(f"NPI request URL: {base_url}?{urlencode(params)}")
        started = time.time()  # for the elapsed time in the error message below
        try:
            # Up to 10 retries, waiting longer each time (see make_retry).
            response = u3_request(
                "GET",
                base_url,
                fields=params,
                timeout=urllib3.Timeout(connect=10, read=60),
                retries=make_retry(total=10, backoff_factor=1.5),
            )
            # A final error answer arrives here as a response, not an exception
            # (raise_on_status=False in make_retry), so check the HTTP status.
            status_code = getattr(response, "status", None)
            if status_code is None or not 200 <= int(status_code) < 300:
                # Include the first 500 characters of the answer, to help diagnose it.
                preview = getattr(response, "data", b"")[:500]
                if isinstance(preview, bytes):
                    preview = preview.decode("utf-8", errors="replace")
                raise RuntimeError(f"NPI Registry returned HTTP {status_code}: {preview}")
            data = u3_json(response)
            # The registry reports a bad search in an "Errors" list even when the
            # HTTP status is 200 (OK). Both spellings are checked.
            errors = data.get("Errors") or data.get("errors")
            if errors:
                raise RuntimeError(f"NPI Registry API error: {errors}")
            page = data.get("results", [])
            if not isinstance(page, list):
                raise RuntimeError("NPI response field 'results' was not a list.")
        except Exception as exc:
            # Any failure stops Step 1 for this city. Nothing has been written yet,
            # so the city can simply be run again.
            raise RuntimeError(
                f"NPI request failed for {city}, {state}, skip={skip}, "
                f"after {time.time() - started:.2f}s: {exc}"
            ) from exc

        print(
            f"NPI page skip={skip}: {len(page)} records; cumulative={len(all_results) + len(page)}"
        )
        # Stop when the page is empty (no more records), when it is short (it was
        # the last page), or when the last page allowed (skip = 1,000) is full.
        if not page:
            break
        all_results.extend(page)
        if len(page) < NPI_PAGE_LIMIT:
            break
        if skip == NPI_MAX_SKIP:
            # Known issue: the 1,200-record ceiling. More records may exist but the
            # registry will not send them. (A city with exactly 1,200 records also
            # ends up here, although nothing is missing.)
            ceiling_reached = True
            break
        skip += NPI_PAGE_LIMIT

    total_npi = len(all_results)
    # No records at all: for example, a misspelled city name.
    if total_npi == 0:
        raise RuntimeError(f"No NPI records were returned for {city}, {state}.")

    # 3. Check that every record is an organization (NPI-2), then write the ceiling
    #    warning if it is needed.
    # The registry should send only NPI-2 records, since that is what was asked
    # for; this check makes sure individual providers (NPI-1) never slip in. It is
    # done before any file is written, so a failure leaves the Step 1 folder empty.
    not_npi2 = [
        record.get("number")
        for record in all_results
        if record.get("enumeration_type") != NPI_ENUMERATION_TYPE
    ]
    if not_npi2:
        raise RuntimeError(
            f"The NPI query requested {NPI_ENUMERATION_TYPE} only, but {len(not_npi2)} "
            f"returned record(s) are not {NPI_ENUMERATION_TYPE}. First NPIs: {not_npi2[:5]}"
        )

    ceiling_warning = None  # path of WARNING_NPI_CEILING.txt, if it is written
    if ceiling_reached:
        message = (
            f"The query reached the {NPI_MAX_RETRIEVABLE}-record retrieval ceiling. "
            "Step 1 continued with the records that were successfully retrieved."
        )
        ceiling_warning = write_npi_ceiling_warning(output_dir, city, state, message)
        print(f"WARNING: {message}")

    # 4. Turn the records into a table with one practice address per record.
    # json_normalize makes a column for each nested field: for example, the
    # organization_name inside "basic" becomes the column "basic.organization_name".
    # The "addresses" list stays a list, in a single column.
    df = pd.json_normalize(all_results)
    if "number" not in df.columns or "basic.organization_name" not in df.columns:
        raise KeyError("NPI response lacks required organization fields.")

    # Kept as text: an NPI is an identifier, not a quantity.
    df["NPI"] = df["number"].astype("string")
    df["OrganizationName"] = df["basic.organization_name"]  # the legal name
    if "addresses" not in df.columns:
        raise KeyError("NPI response lacks required 'addresses' field.")
    df["AddressLocation"] = df["addresses"].apply(_get_practice_address)

    # 5. Set aside the records that have only a mailing address (mailing_only.csv).
    mailing_mask = df["addresses"].apply(_has_mailing_only)
    mailing_only = df.loc[mailing_mask].copy()
    mailing_only_path = output_dir / "mailing_only.csv"
    mailing_only.to_csv(mailing_only_path, index=False)
    # dropna also removes records with neither a practice nor a mailing address.
    # These are rare and are not saved or counted anywhere.
    df = df.loc[~mailing_mask].dropna(subset=["AddressLocation"]).copy()

    # Keep the columns later steps need, plus two NPI fields that are useful for
    # review: the record's status and the date it was last updated.
    keep = ["NPI", "OrganizationName", "AddressLocation", "addresses"]
    if "basic.status" in df.columns:
        keep.append("basic.status")
    if "basic.last_updated" in df.columns:
        keep.append("basic.last_updated")
    df = df[keep].copy()

    # 6. Remove records with exactly the same practice address (duplicates_address.csv).
    # Records that share an address are treated as one store (for example, a
    # pharmacy that registered a new NPI after a change of owner). Sorting newest
    # first means drop_duplicates keeps the most recently updated record at each
    # address. This sort also sets the order of the pharmacy IDs: when the dates
    # are present, the most recently updated record that survives becomes P0001.
    before_address_dedup = df.copy()  # kept so the removed rows can be found below
    if "basic.last_updated" in df.columns:
        # Dates that can't be read become NaT (missing) and sort last.
        df["_last_updated"] = pd.to_datetime(df["basic.last_updated"], errors="coerce")
        df = df.sort_values("_last_updated", ascending=False)
    # Exact text only: "1234 W MAIN ST" and "1234 W MAIN STREET" count as different
    # addresses. Such pairs are caught in stage 8 if Google gives both records the
    # same coordinates.
    df = df.drop_duplicates(subset=["AddressLocation"], keep="first")
    # The removed rows are the ones whose row label is no longer in df.
    address_duplicates = before_address_dedup.loc[
        ~before_address_dedup.index.isin(df.index)
    ].copy()
    address_duplicates_path = output_dir / "duplicates_address.csv"
    address_duplicates.to_csv(address_duplicates_path, index=False)
    # The dates were needed only for choosing which duplicate to keep.
    df.drop(columns=["_last_updated", "basic.last_updated"], errors="ignore", inplace=True)

    # 7. Ask Google Places about every record and keep the OPERATIONAL ones
    #    (the others go to inactive_pharmacies.csv).
    active, google_summary = filter_inactive_pharmacies(
        df,
        api_key=str(api_key).strip(),
        output_dir=output_dir,
        request_delay_seconds=google_request_delay_seconds,
    )
    # 8. Drop records without coordinates, remove records with identical
    #    coordinates, and assign the pharmacy IDs.
    final_df, missing_coordinates, coordinate_duplicates, final_summary = (
        finalize_active_pharmacies(active, city, state, tag)
    )
    missing_path = output_dir / "missing_coordinates.csv"
    duplicates_path = output_dir / "duplicates_latlon.csv"
    missing_coordinates.to_csv(missing_path, index=False)
    coordinate_duplicates.to_csv(duplicates_path, index=False)

    # 9. Save pharmacy_data.csv with the main columns first, in this order, and any
    #    other columns (addresses, basic.status) after them.
    preferred = [
        "pharmacy_id",
        ANALYSIS_TAG_COLUMN,
        "NPI",
        "OrganizationName",
        "AddressLocation",
        "google_place_id",
        "status",
        "candidates",
        "latitude",
        "longitude",
    ]
    final_df = final_df[
        [column for column in preferred if column in final_df.columns]
        + [column for column in final_df.columns if column not in preferred]
    ]
    final_df.to_csv(output_file, index=False)

    # Then the metadata file, written last on purpose: Step 0 counts Step 1 as
    # complete only when this file exists and agrees with pharmacy_data.csv.
    metadata = {
        "pipeline_step": 1,
        "processing_status": "complete",
        "city": city,
        "state": state,
        ANALYSIS_TAG_COLUMN: tag,
        "created_utc": utc_now_iso(),
        "npi_enumeration_type": NPI_ENUMERATION_TYPE,
        "npi_address_purpose": address_purpose,
        "npi_api_status_filter": "active",
        "total_npi_records_retrieved": total_npi,
        "npi_retrieval_ceiling_reached": ceiling_reached,
        "npi_ceiling_warning_file": ceiling_warning,
        # Always equal to the total: any record that is not NPI-2 stops Step 1 above.
        "npi2_organization_count": total_npi,
        "mailing_only_count": len(mailing_only),
        "address_duplicates_removed": len(address_duplicates),
        # Counts from stages 7 and 8.
        **google_summary,
        **final_summary,
        "final_output": str(output_file.resolve()),
        "missing_coordinate_output": str(missing_path.resolve()),
        "duplicate_coordinate_output": str(duplicates_path.resolve()),
    }
    write_json(metadata_file, metadata)

    print("\n=== Step 1 summary ===")
    print(f"Analysis run tag: {tag}")
    print(f"Final pharmacies: {len(final_df)}")
    print(f"Output: {output_file}")
    print(f"Metadata: {metadata_file}")
    return str(output_file)


# Run Step 1 for one city from the command line, without Step 0. The Google key is
# read from the GOOGLE_PLACES_API_KEY environment variable unless --google-api-key
# is given. The Google request delay can't be set here, so the 0.5-second default
# is used (Step 0 uses 1.0).
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract pharmacy locations for one city/state.")
    parser.add_argument("city")
    parser.add_argument("state")
    parser.add_argument("output_folder")
    parser.add_argument("analysis_run_tag")
    parser.add_argument("--google-api-key", default=os.getenv("GOOGLE_PLACES_API_KEY", ""))
    parser.add_argument("--npi-address-purpose", default="PRIMARY")
    args = parser.parse_args()
    extract_pharmacy_data(
        args.city,
        args.state,
        args.output_folder,
        args.google_api_key,
        analysis_run_tag=args.analysis_run_tag,
        npi_address_purpose=args.npi_address_purpose,
    )

