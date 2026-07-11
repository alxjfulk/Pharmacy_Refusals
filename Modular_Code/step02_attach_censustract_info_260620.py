# -*- coding: utf-8 -*-
"""
Attach census tract information to pharmacy records and filter out pharmacies
that are far from the target city center.

Key update:
- Census tract ACS matching now uses county FIPS + tract code, not tract code
  alone. Tract codes repeat across counties, so county-level matching is needed
  for cities or pharmacy search results that span multiple counties.
"""

import os
import glob
import math
import pandas as pd
import csv
from us import states
from tqdm import tqdm
#import censusgeocode as cg
import cg_compat as cg
from census import Census
import osmnx as ox


def safe_poverty_rate(under50, under100, total_pop):
    """
    Compute 100 * (under50 + under100) / total_pop safely.
    - If total_pop is missing or <= 0, return NaN (poverty rate undefined).
    - If under50/under100 are missing, treat them as 0.
    """
    if pd.isna(total_pop) or total_pop <= 0:
        return float("nan")

    u50 = 0 if pd.isna(under50) else under50
    u100 = 0 if pd.isna(under100) else under100

    return 100.0 * (u50 + u100) / float(total_pop)


# Earth radius in miles for haversine calculation
EARTH_RADIUS_MI = 3958.8


def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance between two points (in decimal degrees)
    using the haversine formula, returning miles.
    """
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_MI * c


def _first_present_case_insensitive(d: dict, *keys):
    """Return the first available value in d, matching keys case-insensitively."""
    if not isinstance(d, dict):
        return None
    lookup = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        val = lookup.get(str(key).lower())
        if val not in (None, ""):
            return val
    return None


def _normalize_code(value, width: int) -> str | None:
    """Normalize a FIPS/tract component to a zero-padded string."""
    if value in (None, "") or pd.isna(value):
        return None
    try:
        # Some APIs return codes as numbers; int(float()) handles values like 31.0.
        value = str(int(float(value)))
    except Exception:
        value = str(value).strip()
    value = value.split(".")[0].strip()
    if not value:
        return None
    return value.zfill(width)


def _extract_census_geoids(tract_info: dict):
    """
    Extract state FIPS, county FIPS, tract code, and full GEOID from a
    censusgeocode/cg_compat Census Tracts result.

    Expected components:
      state_fips  = 2 digits
      county_fips = 3 digits
      tract_id    = 6 digits
      geoid       = 11 digits = state + county + tract

    The helper is defensive because different censusgeocode wrappers may use
    slightly different key names.
    """
    geoid = _first_present_case_insensitive(tract_info, "GEOID", "geoid", "GEO_ID", "GEOID20")
    geoid = str(geoid).strip() if geoid not in (None, "") else None
    if geoid and geoid.startswith("1400000US"):
        geoid = geoid.replace("1400000US", "", 1)
    if geoid:
        geoid = "".join(ch for ch in geoid if ch.isdigit())
        if len(geoid) >= 11:
            geoid = geoid[-11:]

    state_fips = _normalize_code(_first_present_case_insensitive(tract_info, "STATE", "state", "STATEFP"), 2)
    county_fips = _normalize_code(_first_present_case_insensitive(tract_info, "COUNTY", "county", "COUNTYFP"), 3)
    tract_id = _normalize_code(_first_present_case_insensitive(tract_info, "TRACT", "tract", "TRACTCE"), 6)

    # If a full GEOID is available, use it to fill any missing components.
    if geoid and len(geoid) == 11:
        state_fips = state_fips or geoid[:2]
        county_fips = county_fips or geoid[2:5]
        tract_id = tract_id or geoid[5:]

    if (not geoid) and state_fips and county_fips and tract_id:
        geoid = f"{state_fips}{county_fips}{tract_id}"

    return state_fips, county_fips, tract_id, geoid


def _prepare_acs_df(acs_records) -> pd.DataFrame:
    """Convert ACS records to a DataFrame and normalize state/county/tract IDs."""
    acs_df = pd.DataFrame(acs_records)
    for col, width in [("state", 2), ("county", 3), ("tract", 6)]:
        if col in acs_df.columns:
            acs_df[col] = acs_df[col].apply(lambda x: _normalize_code(x, width))
    return acs_df


def tie_census_info(input_folder, output_folder, max_distance_miles):
    """
    Reads Pharmacy_Data_<City><State>.csv files from input_folder.

    For each pharmacy record:
      - Parse the two-letter state abbreviation from AddressLocation.
      - Reverse-geocode latitude/longitude to a census tract.
      - Match ACS data using county FIPS + tract code.
      - Compute poverty rate and vehicles-per-capita proxy.
      - Remove pharmacies farther than max_distance_miles from the city centroid.

    Writes:
      - <City><State>_tract_data.csv
      - <City><State>_far_pharmacies.csv
    """
    os.makedirs(output_folder, exist_ok=True)

    all_files = glob.glob(os.path.join(input_folder, "Pharmacy_Data_*.csv"))
    primary_files = [f for f in all_files]
    print(f"Found {len(primary_files)} primary Pharmacy_Data_*.csv in '{input_folder}':")
    for f in primary_files:
        print(f"  • {os.path.basename(f)}")
    print()

    files_processed = 0
    total_tracts_written = 0
    total_rows_skipped = 0
    total_far_removed = 0

    state_map = {s.abbr: s for s in states.STATES}
    acs_cache = {}

    # Existing behavior retained: the Census key is supplied directly here.
    # For publication/reproducibility, consider moving this to CENSUS_API_KEY.
    C = Census("985901667535f61f5ea97bfbf8e4fdfcd8c743c4")

    for path in primary_files:
        filename = os.path.basename(path)
        city_state = os.path.splitext(filename)[0].replace("Pharmacy_Data_", "")
        df = pd.read_csv(path)
        n_rows = len(df)
        print(f"Processing '{filename}' ({n_rows} rows)…")

        # Derive city & state for geocoding municipal boundary.
        if len(city_state) > 2 and city_state[-2:].isalpha():
            city_name = city_state[:-2]
            state_abbr_from_filename = city_state[-2:]
        else:
            parts = city_state.split('_')
            city_name = parts[0]
            state_abbr_from_filename = parts[1] if len(parts) > 1 else ''
        place_name = f"{city_name}, {state_abbr_from_filename}, USA"

        try:
            gdf = ox.geocode_to_gdf(place_name)
            city_poly = gdf.loc[0, 'geometry']
            centroid_pt = city_poly.centroid
            center_lat = centroid_pt.y
            center_lng = centroid_pt.x
            print(f"--- {city_state}: using centroid from OSMnx for {place_name}")
        except Exception as e:
            print(f"⚠ Could not geocode {place_name}: {e}")
            center_lat = center_lng = None

        missing_coords = invalid_coords = geocode_failures = no_tract_match = geocoded_success = 0
        tract_records = []

        for i in tqdm(range(n_rows), desc="  geocoding rows", unit="row"):
            try:
                lat = float(df.loc[i, 'latitude'])
                lng = float(df.loc[i, 'longitude'])
            except Exception:
                missing_coords += 1
                continue

            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                invalid_coords += 1
                continue

            try:
                addr = df.loc[i, 'AddressLocation']
                parts = str(addr).split(',')
                state_zip = parts[-1].strip()  # e.g., "GA 30303"
                state_abbr = state_zip.split()[0].upper()
                if state_abbr not in state_map:
                    no_tract_match += 1
                    continue

                result = cg.coordinates(x=lng, y=lat)
                tracts = result.get('Census Tracts', []) if isinstance(result, dict) else []
            except Exception:
                geocode_failures += 1
                continue

            if not tracts:
                no_tract_match += 1
                continue

            tract_info = tracts[0]
            geocoder_state_fips, county_fips, tract_id, geoid = _extract_census_geoids(tract_info)
            expected_state_fips = state_map[state_abbr].fips

            # If the geocoder supplies a state FIPS and it conflicts with the parsed
            # address state, skip the record. This prevents an address-state/county-state
            # mismatch from joining to the wrong ACS table.
            if geocoder_state_fips and geocoder_state_fips != expected_state_fips:
                no_tract_match += 1
                continue

            if not county_fips or not tract_id:
                no_tract_match += 1
                continue

            if state_abbr not in acs_cache:
                st_obj = state_map[state_abbr]
                acs = C.acs5.state_county_tract(
                    fields=(
                        'NAME',
                        'C17002_001E',
                        'C17002_002E',
                        'C17002_003E',
                        'B01003_001E',
                        'B08201_003E',
                        'B08201_004E',
                        'B08201_005E',
                        'B08201_006E',
                        'B09001_001E'
                    ),
                    state_fips=st_obj.fips,
                    county_fips="*",
                    tract="*",
                    year=2022
                )
                acs_cache[state_abbr] = _prepare_acs_df(acs)

            state_df = acs_cache[state_abbr]
            tract_data = state_df[
                (state_df['county'] == county_fips) &
                (state_df['tract'] == tract_id)
            ]
            if tract_data.empty:
                no_tract_match += 1
                continue

            under50 = tract_data['C17002_002E'].iloc[0]
            under100 = tract_data['C17002_003E'].iloc[0]
            total_pop = tract_data['B01003_001E'].iloc[0]
            pov_rate = safe_poverty_rate(under50, under100, total_pop)

            hh_veh_1 = tract_data['B08201_003E'].iloc[0]
            hh_veh_2 = tract_data['B08201_004E'].iloc[0]
            hh_veh_3 = tract_data['B08201_005E'].iloc[0]
            hh_veh_4p = tract_data['B08201_006E'].iloc[0]

            total_vehicles = (
                1 * hh_veh_1 +
                2 * hh_veh_2 +
                3 * hh_veh_3 +
                4 * hh_veh_4p
            )

            pop_over_18 = tract_data['B01003_001E'].iloc[0] - tract_data['B09001_001E'].iloc[0]

            if (total_pop is not None) and (total_pop > 0):
                vehicles_per_capita = total_vehicles / float(total_pop)
            else:
                vehicles_per_capita = float("nan")

            geocoded_success += 1

            if not geoid:
                geoid = f"{expected_state_fips}{county_fips}{tract_id}"

            tract_records.append({
                'OrganizationName': df.loc[i, 'OrganizationName'],
                'AddressLocation': addr,
                'latitude': lat,
                'longitude': lng,
                'state': state_abbr,
                'state_fips': expected_state_fips,
                'county_fips': county_fips,
                'TRACT': tract_id,
                'GEOID': geoid,
                'NAME': tract_data['NAME'].iloc[0],
                'C17002_001E': tract_data['C17002_001E'].iloc[0],
                'C17002_002E': under50,
                'C17002_003E': under100,
                'B01003_001E': total_pop,
                'pop_over_18': pop_over_18,
                'Vehicles_per_capita': vehicles_per_capita,
                'Poverty_Rate': pov_rate
            })

        df_pharm = pd.DataFrame(tract_records)
        far_removed = pd.DataFrame()

        if df_pharm.empty:
            print("  ⚠ No pharmacy records were successfully matched to ACS tracts.")
        elif center_lat is not None:
            df_pharm['distance_mi'] = df_pharm.apply(
                lambda r: haversine(center_lat, center_lng, r['latitude'], r['longitude']),
                axis=1
            )
            mask_far = df_pharm['distance_mi'] > max_distance_miles
            far_removed = df_pharm[mask_far].copy()
            df_pharm = df_pharm[~mask_far].copy()

            far_file = os.path.join(output_folder, f"{city_state}_far_pharmacies.csv")
            far_removed.to_csv(far_file, index=False, quoting=csv.QUOTE_ALL)
            print(f"  Far pharmacies (> {max_distance_miles} mi): {len(far_removed)} saved to '{os.path.basename(far_file)}'")
        else:
            print("  ⚠ Could not compute city center; skipping far-away filtering")

        df_pharm.drop(columns=['distance_mi'], errors='ignore') \
               .to_csv(os.path.join(output_folder, f"{city_state}_tract_data.csv"),
                       index=False, quoting=csv.QUOTE_ALL)
        print(f"  ✔ Wrote {len(df_pharm)} tracts → '{city_state}_tract_data.csv'")

        print(f"  --- Geocoding & filtering stats for '{filename}' ---")
        print(f"    Missing coords:       {missing_coords}")
        print(f"    Invalid coords:       {invalid_coords}")
        print(f"    Geocode failures:     {geocode_failures}")
        print(f"    No tract match:       {no_tract_match}")
        print(f"    Successful geocodes:  {geocoded_success}")
        print(f"    Far-away removed:     {len(far_removed)}\n")

        files_processed += 1
        total_tracts_written += len(df_pharm)
        total_rows_skipped += (
            missing_coords + invalid_coords + geocode_failures +
            no_tract_match + len(far_removed)
        )
        total_far_removed += len(far_removed)

    print("=== Census Attachment Overall Summary ===")
    print(f"  Primary files found:        {len(primary_files)}")
    print(f"  Files processed:            {files_processed}")
    print(f"  Total tracts written:       {total_tracts_written}")
    print(f"  Total rows skipped:         {total_rows_skipped}")
    print(f"  Total far-away removed:     {total_far_removed}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Attach census tracts to pharmacies and filter by distance")
    parser.add_argument("input_folder", help="Folder containing Pharmacy_Data_<City><State>.csv files")
    parser.add_argument("output_folder", help="Folder to write outputs")
    parser.add_argument("--max-distance", type=float, default=75,
                        help="Maximum distance (miles) from city center")
    args = parser.parse_args()
    tie_census_info(args.input_folder, args.output_folder, args.max_distance)
