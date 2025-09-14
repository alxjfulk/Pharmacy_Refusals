# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 11:33:08 2025

@author: bigfo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 21:36:17 2025

Now integrates Giotto-TDA persistence analysis with detailed step prints.
"""
import os
os.chdir(r'C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\modular_code')

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import glob
from PharmacyCityState_modular3_250913 import extract_pharmacy_data
from Zip_to_censustract_modular3 import tie_census_info
from raw_walk_and_drive_times_test_2 import compute_matrices_dijkstra
from compute_t_matrices_1 import batch_process_all
import giotto_persistence_modular_1

def main():
    city_state_list = [
        ("Atlanta",     "GA")
        # ("Pittsburgh",  "PA"),
        # ("Albuquerque", "NM"),
        # ("Austin",      "TX"),
    ]

    api_key = "AIzaSyAJibJPithWPQwSUbvmemdxGS97Qak9kZs"
    pharmacy_output = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_pharmacy"
    tract_output     = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_tract"
    raw_travel_matrix_output = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_travel_matrices"
    wdm_output       = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_WDM"
    tda_output       = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_TDA\test2"

    # Ensure output directories exist
    for folder in [pharmacy_output, tract_output, raw_travel_matrix_output, wdm_output, tda_output]:
        os.makedirs(folder, exist_ok=True)

    for city, state in city_state_list:
        print(f"\n=== Processing {city}, {state} ===")

        pharm_file = os.path.join(
            pharmacy_output,
            f"Pharmacy_Data_{city}{state}.csv"
        )
        if os.path.exists(pharm_file):
            print(f"→ [SKIP] Pharmacy data already exists: {pharm_file}")
        else:
            print("Step 1: Extracting pharmacy data...")
            extract_pharmacy_data(city, state, pharmacy_output, api_key)
            
        # — Step 2: Census tract attachment —
        tract_file = os.path.join(
            tract_output,
            f"{city}{state}_tract_data.csv"
        )
        if os.path.exists(tract_file):
            print(f"→ [SKIP] Tract data already exists: {tract_file}")
        else:
            print("Step 2: Attaching census tract information...")
            tie_census_info(pharmacy_output, tract_output, max_distance_miles=20)#depending on the size of the cities of interest and the maximum difference between two pharmacies in the city, the max miles may need to be adjusted

        # — Step 3: Raw travel-time matrices —
        today = date.today().isoformat()
        drive_path = os.path.join(
            raw_travel_matrix_output,
            f"drive_time_{city.replace(' ', '_')}_{state.replace(' ', '_')}_{today}.npy"
        )
        walk_path = os.path.join(
            raw_travel_matrix_output,
            f"walk_time_{city.replace(' ', '_')}_{state.replace(' ', '_')}_{today}.npy"
        )
        
        drive_exists = os.path.exists(drive_path)
        walk_exists  = os.path.exists(walk_path)
        
        # if both are there, skip altogether
        if drive_exists and walk_exists:
            print(f"→ [SKIP] Both travel-time matrices exist:\n    {drive_path}\n    {walk_path}")
        else:
            # decide which to compute
            do_drive = not drive_exists
            do_walk  = not walk_exists
        
            tasks = []
            if do_drive: tasks.append("drive")
            if do_walk:  tasks.append("walk")
            print(f"Step 3: Computing raw {' & '.join(tasks)} matrix{'s' if len(tasks)>1 else ''}...")
        
            tract_csv = os.path.join(tract_output, f"{city}{state}_tract_data.csv")
            compute_matrices_dijkstra(
                input_csv_path=tract_csv,
                output_dir=raw_travel_matrix_output,
                city=city,
                state=state,
                do_drive=do_drive,
                do_walk=do_walk
            )

    # Step 4: Compute weighted d̃/d matrices for all processed cities
    print("\nStep 4: Computing weighted distance matrices (d̃ and d)...")
    batch_process_all(
        input_folder=tract_output,
        travel_folder=raw_travel_matrix_output,
        output_folder=wdm_output,
        travel_types=['drive', 'walk']
    )

    # --- Step 5: Persistence analysis via ph_computations_orig ---
    print("\n=== Step 5: Persistence analysis ===")
    # Override parameters if needed:
    giotto_persistence_modular_1.matrix_folder     = wdm_output
    giotto_persistence_modular_1.output_folder     = tda_output
    giotto_persistence_modular_1.time_unit_divisor = 60
    giotto_persistence_modular_1.max_homology_dim  = 1

    # Run the entire loop
    # giotto_persistence_modular_1.process_all_matrices()
    

#     batch_generate_birth_death_gifs(
#     matrix_folder=r"path\to\test_WDM",
#     pharmacy_folder=r"path\to\test_pharmacy",
#     streets_shp=None,       # or "path/to/streets.shp"
#     tracts_shp=None,        # or "path/to/tracts.shp"
#     n_frames=100,
#     dt=1.0,
#     t_max=None,
#     tol=None,
#     output_gif_folder=r"path\to\gifs"
# )
    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
