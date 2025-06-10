# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 09:31:59 2025

@author: bigfo
"""

# -*- coding: utf-8 -*-
"""
master_process_flow1.py

Now includes Step 3: raw walk/drive matrix computation via travel_time_helpers.py
"""

import os                                     # file/directory ops
# Optionally print and change working directory if needed
# print(os.getcwd())
os.chdir(r'C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\modular_code')

from PharmacyCityState_modular3 import extract_pharmacy_data
from Zip_to_censustract_modular2 import tie_census_info
from raw_walk_and_drive_times1 import compute_matrices
from compute_t_matrices_1 import batch_process_all
# from ph_computations import compute_accessibility_metrics
import multiprocessing as mp       # to parallelize shortest-path calculations

def main():
    # --- Parameters (customize these paths & keys) ---
    city = "Milwaukee"
    state = "WI"
    api_key = "AIzaSyAJibJPithWPQwSUbvmemdxGS97Qak9kZs"

    pharmacy_output = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_pharmacy"
    tract_output     = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_tract"
    raw_travel_matrix_output = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_travel_matrices"
    wdm_output       = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_WDM"
    tda_output       = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_TDA"

    # Step 1: Extract and filter pharmacy data
    print("Step 1: Extracting pharmacy data...")
    # extract_pharmacy_data(city, state, pharmacy_output, api_key)

    # Step 2: Tie census tract info to pharmacies
    print("Step 2: Attaching census tract information...")
    # tie_census_info(pharmacy_output, tract_output)

    # Step 3: Compute raw walk & drive travel-time matrices
    print("Step 3: Computing raw walk & drive matrices...")
    tract_csv = os.path.join(tract_output, f"{city}{state}_tract_data.csv")
    compute_matrices(
    input_csv_path=tract_csv,
    output_dir=raw_travel_matrix_output,
    city=city,
    state=state,
    do_drive=True,
    do_walk=True)
    
    # Step 4: Compute weighted d̃/d matrices from raw matrices
    print("Step 4: Computing weighted distance matrices (d̃  and d)...")
    # batch_process_all(
    #     input_folder=tract_output,
    #     travel_folder=raw_travel_matrix_output,
    #     output_folder=wdm_output,
    #     travel_types=['drive', 'walk']
    # )

    # Step 5: Compute accessibility metrics & persistence diagrams
    print("Step 5: Computing accessibility & persistence metrics...")
    # compute_accessibility_metrics(
    #     matrix_folder=wdm_output,
    #     tract_folder=tract_output,
    #     output_folder=tda_output,
    #     matrix_type="d",
    #     threshold=30
    # )

    print("All steps completed successfully.")

if __name__ == "__main__":
    main()
