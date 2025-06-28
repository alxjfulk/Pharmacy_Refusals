# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 00:12:34 2025

@author: bigfo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 01:20:43 2025

@author: bigfo

Updated to:
  • Identify state by parsing the two-letter abbreviation from AddressLocation
  • Compute a stable city center via OSMnx geocoding of the municipal boundary
  • Filter out “far” pharmacies (> max_distance_miles from that center), regardless of state
  • Save those far-away records to a separate CSV
  • Include counts of far-away removals in per-file and overall summaries
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
import osmnx as ox

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

def tie_census_info(input_folder, output_folder, max_distance_miles):
    """
    Reads primary Data<City><State>.csv files from input_folder, each containing
    OrganizationName, AddressLocation, latitude, longitude, etc.
    For each:
      - Parse two-letter state from AddressLocation
      - Reverse-geocode to a tract in that state
      - Compute poverty rate via ACS (cached per state)
      - Compute distance from a stable city center (geocoded municipal centroid)
      - Remove far-away pharmacies (> max_distance_miles), saving them separately
    Writes <City><State>_tract_data.csv and <City><State>_far_pharmacies.csv.
    Prints per-file stats (missing coords, failures, far removals, etc.)
    and a final overall summary.
    """
    os.makedirs(output_folder, exist_ok=True)

    # discover primary CSVs (no underscores in basename)
    all_files = glob.glob(os.path.join(input_folder, "Pharmacy_Data_*.csv"))
    primary_files = [f for f in all_files]
    print(f"Found {len(primary_files)} primary Pharmacy_Data_*.csv in '{input_folder}':")
    for f in primary_files:
        print(f"  • {os.path.basename(f)}")
    print()

    # overall summary counters
    files_processed     = 0
    total_tracts_written  = 0
    total_rows_skipped    = 0
    total_far_removed     = 0

    # map for state abbreviation → us.state object
    state_map = {s.abbr: s for s in states.STATES}
    # cache for ACS tables by state abbreviation
    acs_cache = {}
    # Census API client (replace with your key)
    C = Census("985901667535f61f5ea97bfbf8e4fdfcd8c743c4")

    for path in primary_files:
        filename    = os.path.basename(path)
        city_state  = os.path.splitext(filename)[0].replace("Pharmacy_Data_", "")
        df          = pd.read_csv(path)
        n_rows      = len(df)
        print(f"Processing '{filename}' ({n_rows} rows)…")

        # Derive city & state for geocoding municipal boundary
        if len(city_state) > 2 and city_state[-2:].isalpha():
            city_name  = city_state[:-2]
            state_abbr = city_state[-2:]
        else:
            parts = city_state.split('_')
            city_name  = parts[0]
            state_abbr = parts[1] if len(parts) > 1 else ''
        place_name = f"{city_name}, {state_abbr}, USA"

        # Geocode the city polygon and take its centroid
        try:
            gdf = ox.geocode_to_gdf(place_name)
            city_poly   = gdf.loc[0, 'geometry']
            centroid_pt = city_poly.centroid
            center_lat  = centroid_pt.y
            center_lng  = centroid_pt.x
            print(f"--- {city_state}: using centroid from OSMnx for {place_name}")
        except Exception as e:
            print(f"⚠ Could not geocode {place_name}: {e}") 
            center_lat = center_lng = None

        # initialize per-file stats
        missing_coords = invalid_coords = geocode_failures = no_tract_match = geocoded_success = 0
        tract_records = []

        # 1) Geocode each pharmacy record to census tract + pull ACS
        for i in tqdm(range(n_rows), desc="  geocoding rows", unit="row"):
            # 1a) Extract lat/lon
            try:
                lat = float(df.loc[i, 'latitude'])
                lng = float(df.loc[i, 'longitude'])
            except Exception:
                missing_coords += 1
                continue

            # 1b) Validate ranges
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                invalid_coords += 1
                continue

            # 2) Reverse-geocode to tract in that state
            try:
                addr = df.loc[i, 'AddressLocation']
                parts = addr.split(',')
                state_zip = parts[-1].strip()            # e.g. "GA 30303"
                state_abbr = state_zip.split()[0].upper()
                if state_abbr not in state_map:
                    no_tract_match += 1
                    continue
                result = cg.coordinates(x=lng, y=lat)
                tracts = result['Census Tracts']
            except Exception:
                geocode_failures += 1
                continue

            if not tracts:
                no_tract_match += 1
                continue
            tract_info = tracts[0]
            tract_id   = tract_info['TRACT']

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

            under50  = tract_data['C17002_002E'].iloc[0]
            under100 = tract_data['C17002_003E'].iloc[0]
            total_pop = tract_data['B01003_001E'].iloc[0]
            pov_rate  = 100 * (under50 + under100) / total_pop

            geocoded_success += 1

            # preserve pharmacy fields and add ACS info
            tract_records.append({
                'OrganizationName': df.loc[i, 'OrganizationName'],
                'AddressLocation':  addr,
                'latitude':         lat,
                'longitude':        lng,
                'state':            state_abbr,
                'TRACT':            tract_id,
                'NAME':             tract_data['NAME'].iloc[0],
                'C17002_001E':      tract_data['C17002_001E'].iloc[0],
                'C17002_002E':      under50,
                'C17002_003E':      under100,
                'B01003_001E':      total_pop,
                'Poverty_Rate':     pov_rate
            })

        # Assemble into DataFrame
        df_pharm = pd.DataFrame(tract_records)

        # 2) Filter out far-away pharmacies using the city centroid
        far_removed = []
        if center_lat is not None:
            df_pharm['distance_mi'] = df_pharm.apply(
                lambda r: haversine(center_lat, center_lng, r['latitude'], r['longitude']),
                axis=1
            )
            # select those beyond threshold (regardless of state)
            mask_far     = df_pharm['distance_mi'] > max_distance_miles
            far_removed  = df_pharm[mask_far].copy()
            df_pharm     = df_pharm[~mask_far]

            # save far-away records
            far_file = os.path.join(output_folder, f"{city_state}_far_pharmacies.csv")
            far_removed.to_csv(far_file, index=False, quoting=csv.QUOTE_ALL)
            print(f"  Far pharmacies (> {max_distance_miles} mi): {len(far_removed)} saved to '{os.path.basename(far_file)}'")
        else:
            print("  ⚠ Could not compute city center; skipping far-away filtering")

        # 3) Drop the helper column & write the remaining tracts
        df_pharm.drop(columns=['distance_mi'], errors='ignore')\
               .to_csv(os.path.join(output_folder, f"{city_state}_tract_data.csv"),
                       index=False, quoting=csv.QUOTE_ALL)
        print(f"  ✔ Wrote {len(df_pharm)} tracts → '{city_state}_tract_data.csv'")

        # Per-file summary
        print(f"  --- Geocoding & filtering stats for '{filename}' ---")
        print(f"    Missing coords:       {missing_coords}")
        print(f"    Invalid coords:       {invalid_coords}")
        print(f"    Geocode failures:     {geocode_failures}")
        print(f"    No tract match:       {no_tract_match}")
        print(f"    Successful geocodes:  {geocoded_success}")
        print(f"    Far-away removed:     {len(far_removed)}\n")

        # Accumulate overall counters
        files_processed      += 1
        total_tracts_written += len(df_pharm)
        total_rows_skipped   += (missing_coords + invalid_coords +
                                 geocode_failures + no_tract_match +
                                 len(far_removed))
        total_far_removed    += len(far_removed)

    # Final overall summary
    print("=== Census Attachment Overall Summary ===")
    print(f"  Primary files found:        {len(primary_files)}")
    print(f"  Files processed:            {files_processed}")
    print(f"  Total tracts written:       {total_tracts_written}")
    print(f"  Total rows skipped:         {total_rows_skipped}")
    print(f"  Total far-away removed:     {total_far_removed}\n")


# If you want to run this as a script:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Attach census tracts to pharmacies and filter by distance")
    parser.add_argument("input_folder", help="Folder containing Data<City><State>.csv files")
    parser.add_argument("output_folder", help="Folder to write outputs")
    parser.add_argument("--max-distance", type=float, default=75,
                        help="Maximum distance (miles) from city center")
    args = parser.parse_args()
    tie_census_info(args.input_folder, args.output_folder, args.max_distance)
