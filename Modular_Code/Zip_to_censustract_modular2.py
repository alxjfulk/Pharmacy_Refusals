# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 01:20:43 2025

@author: bigfo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 11:22:35 2025

@author: bigfo

Updated to:
  • Identify state by parsing the two-letter abbreviation from AddressLocation
  • Filter out “far” pharmacies (>75 miles from city center) based on geodesic distance
  • Save those far-away records to a separate CSV
  • Include counts of far-away removals in per-file and overall summaries
  • Automatically load ACS tables per state as before
"""

import os
import glob
import math
import pandas as pd
import csv
from us import states
from tqdm import tqdm
import censusgeocode as cg
from census import Census

# Earth radius in miles for haversine calculation
EARTH_RADIUS_MI = 3958.8

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points (in decimal degrees)
    using the haversine formula, returning miles.
    """
    # convert decimal degrees to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    # differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    # haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_MI * c

def tie_census_info(input_folder, output_folder, max_distance_miles=75):
    """
    Reads primary Data<City><State>.csv files from input_folder, each containing
    OrganizationName, AddressLocation, latitude, longitude, etc.
    For each:
      - Parse two-letter state from AddressLocation
      - Reverse-geocode to a tract in that state
      - Compute poverty rate via ACS (cached per state)
      - Compute distance from city center (mean of in-state coords)
      - Remove far-away pharmacies (> max_distance_miles), saving them separately
    Writes <City><State>_tract_data.csv and <City><State>_far_pharmacies.csv.
    Prints per-file stats (missing coords, failures, far removals, etc.)
    and a final overall summary.
    """
    # ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # (4) discover primary CSVs (no underscores in basename)
    all_files = glob.glob(os.path.join(input_folder, "Data*.csv"))
    primary_files = [
        f for f in all_files
        if "_" not in os.path.splitext(os.path.basename(f))[0]
    ]
    print(f"Found {len(primary_files)} primary Data*.csv in '{input_folder}':")
    for f in primary_files:
        print(f"  • {os.path.basename(f)}")
    print()

    # overall summary counters (9)
    files_processed = 0
    total_tracts_written = 0
    total_rows_skipped = 0
    total_far_removed = 0

    # map for state abbreviation → us.state object
    state_map = {s.abbr: s for s in states.STATES}
    # cache for ACS tables by state abbreviation
    acs_cache = {}
    # Census API client
    C = Census("985901667535f61f5ea97bfbf8e4fdfcd8c743c4")

    for path in primary_files:
        filename = os.path.basename(path)
        city_state = os.path.splitext(filename)[0].replace("Data", "")
        df = pd.read_csv(path)
        n_rows = len(df)
        print(f"Processing '{filename}' ({n_rows} rows)…")

        # per-file stats (3,5)
        missing_coords = invalid_coords = geocode_failures = no_tract_match = geocoded_success = 0
        tract_records = []

        # 1st pass: geocode + ACS lookup
        for i in tqdm(range(n_rows), desc="  geocoding rows", unit="row"):
            # 3a) extract lat/lon
            try:
                lat = float(df.loc[i, 'latitude'])
                lng = float(df.loc[i, 'longitude'])
            except Exception:
                missing_coords += 1
                continue

            # 3b) validate ranges
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                invalid_coords += 1
                continue

            # 5) reverse-geocode to obtain tract info
            try:
                result = cg.coordinates(x=lng, y=lat)
            except Exception:
                geocode_failures += 1
                continue

            tracts = result.get('Census Tracts', [])
            if not tracts:
                no_tract_match += 1
                continue
            tract_info = tracts[0]
            tract_id = tract_info['TRACT']

            # parse state abbreviation from AddressLocation ("..., ST ZIP")
            addr = df.loc[i, 'AddressLocation']
            parts = addr.split(',')
            if len(parts) < 3:
                no_tract_match += 1
                continue
            state_zip = parts[-1].strip()            # e.g. "WI 53202"
            state_abbr = state_zip.split()[0].upper()  # e.g. "WI"
            if state_abbr not in state_map:
                no_tract_match += 1
                continue

            # load ACS for this state if needed
            if state_abbr not in acs_cache:
                st_obj = state_map[state_abbr]
                acs = C.acs5.state_county_tract(
                    fields=('NAME','C17002_001E','C17002_002E','C17002_003E','B01003_001E'),
                    state_fips=st_obj.fips, county_fips="*", tract="*", year=2022
                )
                acs_cache[state_abbr] = pd.DataFrame(acs)

            # lookup tract
            state_df = acs_cache[state_abbr]
            tract_data = state_df[state_df['tract'] == tract_id]
            if tract_data.empty:
                no_tract_match += 1
                continue

            # success
            geocoded_success += 1
            under50 = tract_data['C17002_002E'].iloc[0]
            under100 = tract_data['C17002_003E'].iloc[0]
            total_pop = tract_data['B01003_001E'].iloc[0]
            pov_rate = ((under50 + under100) / total_pop * 100) if total_pop else None

            # preserve pharmacy fields and add geocode+ACS info
            tract_records.append({
                'OrganizationName': df.loc[i, 'OrganizationName'],
                'AddressLocation': addr,
                'latitude': lat,
                'longitude': lng,
                'state': state_abbr,
                'NAME': tract_data['NAME'].iloc[0],
                'C17002_001E': tract_data['C17002_001E'].iloc[0],
                'C17002_002E': under50,
                'C17002_003E': under100,
                'B01003_001E': total_pop,
                'Poverty_Rate': pov_rate,
                'county': result['Counties'][0]['NAME'],
                'ID': tract_id
            })

        # assemble into DataFrame
        df_pharm = pd.DataFrame(tract_records)

        # pick the first in-state pharmacy as a rough city center 
        base_state = city_state[-2:]
        in_state = df_pharm[df_pharm['state'] == base_state]
        if not in_state.empty:
            # take the very first pharmacy’s coords
            center_lat = in_state.iloc[0]['latitude']
            center_lng = in_state.iloc[0]['longitude']
        else:
            center_lat = center_lng = None

        # identify far-away pharmacies
        far_removed = []
        if center_lat is not None:
            # compute distance for every row
            df_pharm['distance_mi'] = df_pharm.apply(
                lambda r: haversine(center_lat, center_lng, r['latitude'], r['longitude']), axis=1
            )
            # select those beyond threshold *and* in a different state
            mask_far = (df_pharm['distance_mi'] > max_distance_miles) & \
                       (df_pharm['state'] != base_state)
            far_removed = df_pharm[mask_far]
            df_pharm = df_pharm[~mask_far]

            # save far-away records
            far_file = os.path.join(output_folder, f"{city_state}_far_pharmacies.csv")
            far_removed.to_csv(far_file, index=False, quoting=csv.QUOTE_ALL)
            print(f"  Far pharmacies (> {max_distance_miles} mi): {len(far_removed)} saved to '{os.path.basename(far_file)}'")
        else:
            print("  ⚠ Could not compute city center; skipping far-away filtering")

        # write cleaned tract_data
        if not df_pharm.empty:
            out_name = f"{city_state}_tract_data.csv"
            out_path = os.path.join(output_folder, out_name)
            df_pharm.drop(columns=['distance_mi'], errors='ignore')\
                   .to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
            print(f"  ✔ Wrote {len(df_pharm)} tracts → '{out_name}'")
        else:
            print(f"  ⚠ No valid tracts remain for '{filename}' after filtering")

        # per-file summary
        print(f"  --- Geocoding & filtering stats for '{filename}' ---")
        print(f"    Missing coords:       {missing_coords}")
        print(f"    Invalid coords:       {invalid_coords}")
        print(f"    Geocode failures:     {geocode_failures}")
        print(f"    No tract match:       {no_tract_match}")
        print(f"    Successful geocodes:  {geocoded_success}")
        print(f"    Far-away removed:     {len(far_removed)}\n")

        # accumulate overall counters
        files_processed     += 1
        total_tracts_written  += len(df_pharm)
        total_rows_skipped    += (missing_coords + invalid_coords +
                                  geocode_failures + no_tract_match +
                                  len(far_removed))
        total_far_removed     += len(far_removed)

    # (9) final overall summary
    print("=== Census Attachment Overall Summary ===")
    print(f"  Primary files found:        {len(primary_files)}")
    print(f"  Files processed:            {files_processed}")
    print(f"  Total tracts written:       {total_tracts_written}")
    print(f"  Total rows skipped:         {total_rows_skipped}")
    print(f"  Total far-away removed:     {total_far_removed}\n")
