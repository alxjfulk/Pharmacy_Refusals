# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 11:55:26 2025

Updated: [Date]
- Added comprehensive console statistics at each major step
- Count of ambiguous/multiple Google candidates
- Address‐level and lat/lon dedup stats
"""

import requests            # for HTTP calls to Google Places API
import pandas as pd        # for DataFrame operations
import time                # for rate‐limit sleeps and timing
import os                  # for file/directory operations
import urllib.request      # for downloading JSON pages
import json                # for parsing JSON
import numpy as np         # for NaN handling
from collections import Counter  # for tallying status codes

def check_google_places_status(name, address, api_key):
    """
    Query Google Places Find Place API for business status and geometry.
    Returns:
        status (str): business_status or API-level status (e.g., ZERO_RESULTS)
        lat (float|None): latitude of the first candidate
        lng (float|None): longitude of the first candidate
        candidate_count (int): number of candidates returned
    """
    base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": f"{name} {address}",
        "inputtype": "textquery",
        "fields": "business_status,geometry",
        "key": api_key
    }
    try:
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

        # How many candidates did we get?
        candidates = data.get("candidates", [])
        count = len(candidates)

        if data.get("status") == "OK" and count > 0:
            cand = candidates[0]
            status = cand.get("business_status", "UNKNOWN")
            loc = cand.get("geometry", {}).get("location", {})
            lat = loc.get("lat")
            lng = loc.get("lng")
            return status, lat, lng, count

        # No candidates or non-OK status
        return data.get("status", "NOT_FOUND"), None, None, count

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
    print(f"Elapsed time: {elapsed:.2f}s (avg {elapsed/total_api_calls:.3f}s per call)")
    print("Status breakdown:")
    for st, ct in status_counter.items():
        pct = ct / total_api_calls * 100 if total_api_calls else 0
        print(f"  {st}: {ct} ({pct:.1f}%)")
    print(f"Ambiguous (>1 candidate): {ambiguous_count} "
          f"({ambiguous_count/total_api_calls*100:.1f}%)")
    print(f"Inactive saved to: {inactive_file}\n")

    return df_active


def extract_pharmacy_data(city, state, output_folder, api_key):
    """
    1. Fetch NPI registry entries for pharmacies.
    2. Normalize JSON + filter to NPI-2 organizations.
    3. Extract only 'LOCATION' addresses.
    4. Separate out and save any records that have only 'MAILING' addresses.
    5. Proceed with address dedup, Google filtering, lat/lon dedup, as before.
    """
    os.makedirs(output_folder, exist_ok=True)

    # STEP 1: Download all NPI pages
    all_results = []
    skip = 0
    while True:
        url = (
            "https://npiregistry.cms.hhs.gov/api/?taxonomy_description=Pharmacy"
            f"&city={city}&state={state}&limit=200&skip={skip}&version=2.1"
        )
        tmpfile = f"{city}{state}_NPI_{skip}.json"
        urllib.request.urlretrieve(url, tmpfile)
        with open(tmpfile) as f:
            page = json.load(f).get("results", [])
        os.remove(tmpfile)
        if not page:
            break
        all_results.extend(page)
        skip += 200

    # Report fetch stats
    total_npi = len(all_results)
    print(f"\n--- NPI Fetch: {total_npi} total records ---")

    # STEP 2: Normalize + filter to organizations
    df = pd.json_normalize(all_results)
    df = df[df["enumeration_type"] == "NPI-2"].copy()
    # === Add the NPI column ===
    # the raw JSON field "number" is the NPI identifier
    df["NPI"] = df["number"]
    org_count = len(df)
    print(f"NPI-2 organizations: {org_count} ({org_count/total_npi*100:.1f}%)")

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

    # STEP 5: Address deduplication (unchanged logic)
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

    # STEP 6: Google Places filtering + stats (unchanged)
    df_active = filter_inactive_pharmacies(df, api_key, output_folder, city, state)

    # STEP 7: Latitude/Longitude deduplication (unchanged)
    pre_ll = len(df_active)
    ll_dups = df_active[df_active.duplicated(subset=["latitude","longitude"], keep=False)]
    removed_ll = len(ll_dups)
    print(f"Lat/Lon dedup: removed {removed_ll} "
          f"({removed_ll/pre_ll*100:.1f}%), remaining {pre_ll-removed_ll}")
    latlon_dup_file = os.path.join(output_folder, f"Data{city}{state}_duplicates_latlon.csv")
    ll_dups.to_csv(latlon_dup_file, index=False)
    print(f"Lat/Lon duplicates saved to: {latlon_dup_file}")
    df_final = df_active.drop_duplicates(subset=["latitude","longitude"], keep="first")

    # STEP 8: Save final active set
    output_file = os.path.join(output_folder, f"Data{city}{state}.csv")
    df_final.to_csv(output_file, index=False)
    print(f"Final active pharmacies saved to: {output_folder}\n")



        # === Final Extraction Summary ===
    print("=== Pharmacy Extraction Overall Summary ===")
    print(f"Total NPI records fetched:      {total_npi}")
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
        api_key="YOUR_GOOGLE_API_KEY"
    )
