# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 11:55:26 2025

Updated: 2026-07-11
- Added comprehensive console statistics at each major step
- Count of ambiguous/multiple Google candidates
- Address-level and lat/lon dedup stats
- Added retry/backoff handling for transient NPI Registry/API connection resets
- Added NPI page-level request/response diagnostics
- Enforced the NPI API maximum skip value of 1000 (1200-record ceiling)
- Continue with the first 1200 records when the NPI ceiling is reached
- Added explicit server-side enumeration_type and address_purpose filtering
- Added encoded request-URL logging to verify filters are sent before retrieval
"""

import requests            # for HTTP calls to Google Places API
import pandas as pd        # for DataFrame operations
import time                # for rate‐limit sleeps and timing
import os                  # for file/directory operations
import types
import urllib3
# import urllib.request      # for downloading JSON pages
import json                # for parsing JSON
import numpy as np         # for NaN handling
from collections import Counter  # for tallying status codes
from urllib.parse import urlencode  # for transparent NPI request logging


# NPI Registry API paging constraints.
NPI_PAGE_LIMIT = 200
NPI_MAX_SKIP = 1000
NPI_MAX_RETRIEVABLE = NPI_MAX_SKIP + NPI_PAGE_LIMIT

# Official API values. Passing None omits the filter and restores the older
# behavior in which city/state can match mailing or practice-location addresses.
VALID_NPI_ENUMERATION_TYPES = {"NPI-1", "NPI-2"}
VALID_NPI_ADDRESS_PURPOSES = {"LOCATION", "MAILING", "PRIMARY", "SECONDARY"}


def make_retry(total: int = 5, backoff_factor: float = 1.5):
    """
    Build a urllib3 Retry object that works across urllib3 versions.

    The NPI Registry and Google Places calls can occasionally fail because the
    remote host resets the HTTPS connection. These are usually transient network
    failures, not data errors. Retry/backoff makes Step 1 less likely to abort
    the whole pipeline because of a single dropped connection.
    """
    retry_kwargs = dict(
        total=total,
        connect=total,
        read=total,
        status=total,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    try:
        return urllib3.util.Retry(
            allowed_methods=frozenset(["GET", "POST"]),
            **retry_kwargs,
        )
    except TypeError:
        # urllib3 < 2.0 used method_whitelist instead of allowed_methods.
        return urllib3.util.Retry(
            method_whitelist=frozenset(["GET", "POST"]),
            **retry_kwargs,
        )

def write_skip_marker(output_folder: str, city: str, state: str, reason: str) -> str:
    """Write a small marker file documenting that a city/state run was skipped."""
    os.makedirs(output_folder, exist_ok=True)
    city = (city or "").strip()
    state = (state or "").strip()
    marker_path = os.path.join(output_folder, f"SKIPPED_{city}{state}.txt")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"SKIPPED CITY RUN: {city}, {state}\n")
        f.write(f"REASON: {reason}\n")
    return marker_path


def write_npi_ceiling_warning(output_folder: str, city: str, state: str, message: str) -> str:
    """Document that the run continued with a ceiling-limited NPI result."""
    os.makedirs(output_folder, exist_ok=True)
    city = (city or "").strip()
    state = (state or "").strip()
    marker_path = os.path.join(output_folder, f"WARNING_NPI_CEILING_{city}{state}.txt")
    with open(marker_path, "w", encoding="utf-8") as f:
        f.write(f"NPI RETRIEVAL CEILING WARNING: {city}, {state}\n")
        f.write(f"MESSAGE: {message}\n")
    return marker_path


def u3_request(method: str, url: str, **kwargs):
    """
    A wrapper that calls urllib3.request(...) on v2+,
    or falls back to PoolManager().request(...) on v1.x.
    - Pass `timeout=urllib3.Timeout(connect=5, read=30)` if you want a custom timeout.
    - Pass `retries=False` to disable retries, or a urllib3.util.Retry instance to enable.
    - Pass `fields={...}` for query parameters on GET (urllib3 encodes these into the URL).
    Returns: urllib3.response.HTTPResponse
    """
    # Sensible defaults if caller does not specify them.
    timeout = kwargs.pop("timeout", urllib3.Timeout(connect=10, read=60))
    retries = kwargs.pop("retries", make_retry())

    # urllib3>=2: top-level urllib3.request is a callable function
    if hasattr(urllib3, "request") and callable(getattr(urllib3, "request")) \
       and not isinstance(urllib3.request, types.ModuleType):
        return urllib3.request(method, url, timeout=timeout, retries=retries, **kwargs)

    # urllib3 1.26.x: use a PoolManager instance
    http = urllib3.PoolManager()
    return http.request(method, url, timeout=timeout, retries=retries, **kwargs)

def u3_json(resp):
    """Decode JSON from an urllib3 HTTPResponse across versions."""
    # urllib3 2.x has HTTPResponse.json(); older versions do not
    return resp.json() if hasattr(resp, "json") else json.loads(resp.data.decode("utf-8"))

def check_google_places_status(name, address, api_key):
    """
    New implementation using Places API (New) Text Search.
    Returns:
        status (str): businessStatus ('OPERATIONAL', 'CLOSED_TEMPORARILY', etc.) or API-level status string
        lat (float|None)
        lng (float|None)
        candidate_count (int)
    """
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "X-Goog-Api-Key": api_key,
        # Ask only for what we use:
        "X-Goog-FieldMask": "places.businessStatus,places.location,places.id",
        "Content-Type": "application/json",
    }
    payload = {"textQuery": f"{name} {address}"}

    try:
        # use existing urllib3 wrapper for consistency
        resp = u3_request(
            "POST",
            url,
            headers=headers,
            body=json.dumps(payload).encode("utf-8"),
            # you can pass retries=Retry(...) here if you like
        )
        data = u3_json(resp)

        # New API errors return an "error" object; select it for debugging
        if isinstance(data, dict) and "error" in data:
            err = data["error"]
            msg = err.get("message", "Unknown error")
            print(f"Google Places API error: {msg}")
            return "REQUEST_DENIED", None, None, 0

        places = data.get("places", []) if isinstance(data, dict) else []
        count = len(places)
        if count > 0:
            first = places[0]
            status = first.get("businessStatus", "UNKNOWN")
            loc = (first.get("location") or {})
            lat = loc.get("latitude")
            lng = loc.get("longitude")
            return status, lat, lng, count

        # No results (the new API doesn’t use the legacy status codes)
        return "ZERO_RESULTS", None, None, 0

    except Exception as e:
        print(f"Error querying Google Places for '{name}' at '{address}': {e}")
        return "ERROR", None, None, 0


def filter_inactive_pharmacies(df, api_key, output_folder, city, state):
    """
    For each row in df:
      - Query Google Places for business_status + geometry
      - Tally status codes and ambiguous counts
      - Separate active/inactive rows
      - Report stats on API calls
    Returns:
        df_active (DataFrame): only the OPERATIONAL rows + lat/lng columns
    """
    active_rows = []
    inactive_rows = []
    status_counter = Counter()      # tally for each status string
    ambiguous_count = 0             # count of cases with >1 Google candidate
    total_api_calls = 0             # number of calls made
    start_time = time.time()        # track elapsed time

    for _, row in df.iterrows():
        name = row["OrganizationName"]
        address = row["AddressLocation"]

        # Skip rows missing essential data
        if pd.isna(name) or pd.isna(address):
            rec = row.to_dict()
            rec.update({"latitude": None, "longitude": None, "status": "MISSING_DATA", "candidates": 0})
            inactive_rows.append(rec)
            status_counter["MISSING_DATA"] += 1
            continue

        # Call Google Places
        status, lat, lng, cand_count = check_google_places_status(name, address, api_key)
        total_api_calls += 1
        status_counter[status] += 1

        # Count ambiguous candidate situations
        if cand_count > 1:
            ambiguous_count += 1

        # Log the lookup
        print(f"Checked '{name}' @ '{address}' → Status: {status}, "
              f"Candidates: {cand_count}, Lat: {lat}, Lng: {lng}")

        # Build the record dict
        rec = row.to_dict()
        rec.update({"latitude": lat, "longitude": lng, "status": status, "candidates": cand_count})

        # Distribute into active vs. inactive
        if status == "OPERATIONAL":
            active_rows.append(rec)
        else:
            inactive_rows.append(rec)

        # Sleep to respect rate limits (200 ms by default)
        time.sleep(0.2)

    # Build DataFrames for export
    df_active = pd.DataFrame(active_rows)
    df_inactive = pd.DataFrame(inactive_rows)

    # Save inactive pharmacies
    inactive_file = os.path.join(output_folder, f"Data{city}{state}_inactive_pharmacies.csv")
    df_inactive.to_csv(inactive_file, index=False)
    print(f"\n--- Google Places Summary ---")
    print(f"Total API calls: {total_api_calls}")
    elapsed = time.time() - start_time
    avg_elapsed = elapsed / total_api_calls if total_api_calls else float("nan")
    print(f"Elapsed time: {elapsed:.2f}s (avg {avg_elapsed:.3f}s per call)")
    print("Status breakdown:")
    for st, ct in status_counter.items():
        pct = ct / total_api_calls * 100 if total_api_calls else 0
        print(f"  {st}: {ct} ({pct:.1f}%)")
    ambiguous_pct = ambiguous_count / total_api_calls * 100 if total_api_calls else 0
    print(f"Ambiguous (>1 candidate): {ambiguous_count} "
          f"({ambiguous_pct:.1f}%)")
    print(f"Inactive saved to: {inactive_file}\n")

    return df_active


def extract_pharmacy_data(
    city,
    state,
    output_folder,
    api_key,
    skip_on_zero_npi=True,
    npi_enumeration_type="NPI-2",
    npi_address_purpose="PRIMARY",
):
    """
    1. Fetch NPI Registry entries for pharmacies using server-side NPI-2
       and address-purpose filters.
    2. Log each page request, HTTP response, elapsed time, and record count.
    3. Stop at the API's documented skip ceiling and continue with the first
       1200 records when a city-level result is saturated.
    4. Normalize JSON and defensively confirm NPI-2 organizations.
    5. Extract only 'LOCATION' addresses.
    6. Separate out and save any records that have only 'MAILING' addresses.
    7. Proceed with address dedup, Google filtering, and lat/lon dedup.

    Parameters
    ----------
    npi_enumeration_type : {"NPI-2"}
        Sent to the NPI Registry with every page request. This pipeline requires
        organization records, so NPI-2 is the supported value.
    npi_address_purpose : {"LOCATION", "MAILING", "PRIMARY", "SECONDARY"}
        Sent to the NPI Registry with every page request so city/state filtering
        occurs before records are returned. PRIMARY is the default because this
        script extracts the primary practice address from the response's
        ``addresses`` list.
    """
    os.makedirs(output_folder, exist_ok=True)

    # --- STEP 1: Download NPI pages safely and within API limits ---
    # Normalize city/state in case the caller passed trailing spaces or mixed case.
    city = (city or "").strip()
    state = (state or "").strip().upper()

    # Normalize and validate the server-side filters before the first request.
    npi_enumeration_type = str(npi_enumeration_type).strip().upper()
    if npi_enumeration_type not in VALID_NPI_ENUMERATION_TYPES:
        allowed = ", ".join(sorted(VALID_NPI_ENUMERATION_TYPES))
        raise ValueError(
            "Invalid npi_enumeration_type="
            f"{npi_enumeration_type!r}. Use one of {allowed}."
        )
    if npi_enumeration_type != "NPI-2":
        raise ValueError(
            "This pharmacy extraction pipeline requires "
            "npi_enumeration_type='NPI-2' because it uses organization fields."
        )

    npi_address_purpose = str(npi_address_purpose).strip().upper()
    if npi_address_purpose not in VALID_NPI_ADDRESS_PURPOSES:
        allowed = ", ".join(sorted(VALID_NPI_ADDRESS_PURPOSES))
        raise ValueError(
            "Invalid npi_address_purpose="
            f"{npi_address_purpose!r}. Use one of {allowed}."
        )

    all_results = []
    skip = 0
    npi_ceiling_reached = False
    base_url = "https://npiregistry.cms.hhs.gov/api/"

    while skip <= NPI_MAX_SKIP:
        # ``fields`` lets urllib3 URL-encode city/state and other parameters.
        # Both filters are part of the query sent to the Registry. They are not
        # merely applied after download. This dictionary is passed as ``fields``
        # to urllib3, which URL-encodes it onto the GET request.
        params = {
            "taxonomy_description": "Pharmacy",
            "enumeration_type": npi_enumeration_type,
            "address_purpose": npi_address_purpose,
            "city": city,
            "state": state,
            "limit": NPI_PAGE_LIMIT,
            "skip": skip,
            "version": "2.1",
        }

        request_url = f"{base_url}?{urlencode(params)}"
        print(
            f"Requesting NPI page: city={city}, state={state}, "
            f"enumeration_type={npi_enumeration_type}, "
            f"address_purpose={npi_address_purpose}, "
            f"skip={skip}, limit={NPI_PAGE_LIMIT}"
        )
        print(f"NPI request URL: {request_url}")
        request_start = time.time()

        try:
            resp = u3_request(
                "GET",
                base_url,
                fields=params,
                timeout=urllib3.Timeout(connect=10, read=60),
                retries=make_retry(total=5, backoff_factor=1.5),
            )
            elapsed = time.time() - request_start
            http_status = getattr(resp, "status", None)

            print(
                f"NPI response received: HTTP {http_status}, "
                f"skip={skip}, elapsed={elapsed:.2f}s"
            )

            # Do not interpret an HTTP error page as an empty results page.
            if http_status is None or not 200 <= int(http_status) < 300:
                preview = getattr(resp, "data", b"")[:500]
                if isinstance(preview, bytes):
                    preview = preview.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"NPI Registry returned HTTP {http_status}. "
                    f"Response preview: {preview}"
                )

            data = u3_json(resp)

            if not isinstance(data, dict):
                raise RuntimeError(
                    f"NPI Registry returned {type(data).__name__}, expected a JSON object."
                )

            api_errors = data.get("Errors") or data.get("errors")
            if api_errors:
                raise RuntimeError(f"NPI Registry API error response: {api_errors}")

            page = data.get("results", [])
            if not isinstance(page, list):
                raise RuntimeError(
                    "NPI Registry response field 'results' was not a list."
                )

        except Exception as e:
            elapsed = time.time() - request_start
            msg = (
                f"NPI Registry API request failed for {city}, {state} "
                f"at skip={skip} after {elapsed:.2f}s: {e}"
            )
            if skip_on_zero_npi:
                marker = write_skip_marker(output_folder, city, state, msg)
                print(f"SKIPPING {city}, {state}: {msg}")
                print(f"Skip marker written to: {marker}\n")
                return None
            raise

        print(
            f"NPI page complete: skip={skip}, records={len(page)}, "
            f"cumulative={len(all_results) + len(page)}"
        )

        if not page:
            # A successful empty page means pagination is complete.
            break

        all_results.extend(page)

        # A short page is the final page; avoid making an unnecessary request.
        if len(page) < NPI_PAGE_LIMIT:
            break

        # skip=1000 is the last legal page. A full page here means the search
        # has reached the 1200-record retrieval ceiling and may be incomplete.
        if skip == NPI_MAX_SKIP:
            npi_ceiling_reached = True
            break

        skip += NPI_PAGE_LIMIT

    npi_ceiling_warning_file = None
    if npi_ceiling_reached:
        msg = (
            f"NPI query reached the API retrieval ceiling of "
            f"{NPI_MAX_RETRIEVABLE} records (final legal page skip={NPI_MAX_SKIP} "
            f"returned {NPI_PAGE_LIMIT} records). The true number of matching "
            "records may be greater than 1200. Step 1 will continue using the "
            "1200 records that were successfully retrieved."
        )
        npi_ceiling_warning_file = write_npi_ceiling_warning(
            output_folder, city, state, msg
        )
        print(f"WARNING for {city}, {state}: {msg}")
        print(f"Warning marker written to: {npi_ceiling_warning_file}\n")

    # Report fetch stats
    total_npi = len(all_results)
    print(f"\n--- NPI Fetch: {total_npi} total records ---")

    # If NPI returns nothing, we can skip this city to avoid downstream KeyErrors.
    if total_npi == 0:
        msg = "No NPI records returned from NPI Registry API (results=0)."
        if skip_on_zero_npi:
            marker = write_skip_marker(output_folder, city, state, msg)
            print(f"SKIPPING {city}, {state}: {msg}")
            print(f"Skip marker written to: {marker}\n")
            return None
        else:
            print(f"WARNING: {msg} Continuing because skip_on_zero_npi=False.")


    # STEP 2: Normalize + filter to organizations
    df = pd.json_normalize(all_results)
    if "enumeration_type" in df.columns:
        df = df[df["enumeration_type"] == npi_enumeration_type].copy()
    else:
        # Unexpected schema (can happen if the API changes). Do not crash; skip city.
        msg = "NPI response did not include expected column 'enumeration_type'."
        if skip_on_zero_npi:
            marker = write_skip_marker(output_folder, city, state, msg)
            print(f"SKIPPING {city}, {state}: {msg}")
            print(f"Skip marker written to: {marker}\n")
            return None
        else:
            raise KeyError("enumeration_type")
    # === Add the NPI column ===
    # the raw JSON field "number" is the NPI identifier
    df["NPI"] = df["number"]
    org_count = len(df)
    print(f"NPI-2 organizations: {org_count} ({org_count/total_npi*100:.1f}%)")

    # If we have NPI results but none are organizations, skip the city (optional).
    if org_count == 0:
        msg = (
            "NPI results returned, but none matched "
            f"enumeration_type=={npi_enumeration_type!r}."
        )
        if skip_on_zero_npi:
            marker = write_skip_marker(output_folder, city, state, msg)
            print(f"SKIPPING {city}, {state}: {msg}")
            print(f"Skip marker written to: {marker}\n")
            return None
        else:
            print(f"WARNING: {msg} Continuing because skip_on_zero_npi=False.")


    # Helper to pick the 'LOCATION' address
    def get_practice_address(addr_list):
        if isinstance(addr_list, list):
            for addr in addr_list:
                if addr.get("address_purpose") == "LOCATION":
                    # format location address
                    street = addr.get("address_1","").strip()
                    city_  = addr.get("city","").strip()
                    state_ = addr.get("state","").strip()
                    zip5   = addr.get("postal_code","")[:5]
                    return f"{street}, {city_}, {state_} {zip5}"
        return np.nan

    # STEP 3: Build OrganizationName + AddressLocation
    df["OrganizationName"] = df["basic.organization_name"]
    df["AddressLocation"] = df["addresses"].apply(get_practice_address)

    # Identify records with only 'MAILING' (no 'LOCATION')
    def has_mailing_only(addr_list):
        if isinstance(addr_list, list):
            purposes = {addr.get("address_purpose") for addr in addr_list}
            return ("LOCATION" not in purposes) and ("MAILING" in purposes)
        return False

    mailing_mask = df["addresses"].apply(has_mailing_only)
    mailing_only_df = df[mailing_mask].copy()
    mailing_count = len(mailing_only_df)
    print(f"Mailing-only addresses: {mailing_count} "
          f"({mailing_count/org_count*100:.1f}%)")

    # Save mailing-only records to CSV
    mailing_file = os.path.join(output_folder, f"Data{city}{state}_mailing_only.csv")
    mailing_only_df.to_csv(mailing_file, index=False)
    print(f"Mailing-only records saved to: {mailing_file}")

    # Drop mailing-only records before further processing
    df = df[~mailing_mask].copy()
    post_mailing = len(df)
    print(f"Records remaining after removing mailing-only: {post_mailing}")

    # STEP 4: Drop any remaining missing addresses entirely
    missing_addr = df["AddressLocation"].isna().sum()
    if missing_addr:
        print(f"Records with no LOCATION address: {missing_addr} "
              f"({missing_addr/post_mailing*100:.1f}%)")
    df = df.dropna(subset=["AddressLocation"])

    # Keep only necessary columns + last_updated if present
    cols = ["NPI", "OrganizationName", "AddressLocation", "basic.status", "addresses"]
    if "basic.last_updated" in df.columns:
        cols.append("basic.last_updated")
    df = df[cols].copy()
    before_address_dedup = df.copy()

    # STEP 5: Address deduplication
    pre_addr = len(df)
    if "basic.last_updated" in df.columns:
        df["last_updated_parsed"] = pd.to_datetime(df["basic.last_updated"], errors="coerce")
        df = df.sort_values("last_updated_parsed", ascending=False)
    df = df.drop_duplicates(subset=["AddressLocation"], keep="first")
    post_addr = len(df)
    removed_addr = pre_addr - post_addr
    print(f"Address dedup: removed {removed_addr} "
          f"({removed_addr/pre_addr*100:.1f}%), remaining {post_addr}")
    addr_dups = before_address_dedup.loc[~before_address_dedup.index.isin(df.index)]
    addr_dup_file = os.path.join(output_folder, f"Data{city}{state}_duplicates_address.csv")
    addr_dups.to_csv(addr_dup_file, index=False)
    print(f"Address duplicates saved to: {addr_dup_file}")
    df.drop(columns=["last_updated_parsed","basic.last_updated"], errors="ignore", inplace=True)

    # STEP 6: Google Places filtering + stats 
    df_active = filter_inactive_pharmacies(df, api_key, output_folder, city, state)

    # STEP 7: Latitude/Longitude deduplication 
    pre_ll = len(df_active)
    ll_dups = df_active[df_active.duplicated(subset=["latitude", "longitude"], keep=False)]
    df_final = df_active.drop_duplicates(subset=["latitude", "longitude"], keep="first")
    removed_ll = pre_ll - len(df_final)
    print(
        f"Lat/Lon dedup: removed {removed_ll} "
        f"({(f'{removed_ll/pre_ll*100:.1f}%' if pre_ll else 'n/a')}), "
        f"remaining {len(df_final)}"
    )
    latlon_dup_file = os.path.join(output_folder, f"Data{city}{state}_duplicates_latlon.csv")
    ll_dups.to_csv(latlon_dup_file, index=False)
    print(f"Lat/Lon duplicates saved to: {latlon_dup_file}")

    # STEP 8: Save final active set
    # Use the same naming pattern the rest of the pipeline expects:
    # Pharmacy_Data_<City><State>.csv
    output_file = os.path.join(output_folder, f"Pharmacy_Data_{city}{state}.csv")
    df_final.to_csv(output_file, index=False)
    print(f"Final active pharmacies saved to: {output_file}\n")




        # === Final Extraction Summary ===
    print("=== Pharmacy Extraction Overall Summary ===")
    print(f"Total NPI records fetched:      {total_npi}")
    print(
        "NPI retrieval ceiling reached:  "
        + ("YES — continued with first 1200 records" if npi_ceiling_reached else "NO")
    )
    if npi_ceiling_warning_file:
        print(f"NPI ceiling warning file:       {npi_ceiling_warning_file}")
    print(f"Organization (NPI-2) records:   {org_count}")
    print(f"Mailing-only records removed:   {mailing_count}")
    print(f"Address-level duplicates removed: {removed_addr}")
    print(f"Remaining after address dedup:   {post_addr}")
    print(f"Records after Google filter:     {pre_ll}")
    print(f"Lat/Lon duplicates removed:      {removed_ll}")
    print(f"Final active pharmacies:         {len(df_final)}")
    print("============================================\n")

    return output_file


if __name__ == "__main__":
    # Example invocation; replace with real paths & key
    extract_pharmacy_data(
        "Milwaukee",
        "WI",
        r"C:\path\to\test_pharmacy",
        api_key="YOUR_GOOGLE_API_KEY",
        npi_enumeration_type="NPI-2",
        # Valid values: "LOCATION", "MAILING", "PRIMARY", or "SECONDARY".
        npi_address_purpose="PRIMARY",
    )