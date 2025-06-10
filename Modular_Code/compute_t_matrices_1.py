# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 22:53:17 2025

@author: bigfo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 09:58:33 2025

@author: bigfo
"""

import pandas as pd       # pandas for reading/writing CSVs and handling tabular data
import numpy as np        # numpy for numerical arrays and matrix operations
import os                 # os for file path manipulation and directory operations
import glob               # glob for finding files matching a pattern
import csv                # csv for controlling CSV quoting behavior in to_csv()

def compute_dtilde_matrix(travel_matrix, populations):
    """
    Compute the d̃ (“dtilde”) matrix from a raw travel-time matrix and population vector.
    
    travel_matrix: 2D numpy array where entry [i,j] is travel time from tract i to tract j
    populations:    1D numpy array of population values for each tract
    """
    # Reshape populations into column (n×1) and row (1×n) arrays for broadcasting
    pop_i = populations.reshape(-1, 1)  # shape: (n, 1)
    pop_j = populations.reshape(1, -1)  # shape: (1, n)
    
    # Temporarily ignore divide-by-zero or invalid operations (e.g., 0/0)
    with np.errstate(divide='ignore', invalid='ignore'):
        # Compute dtilde_{i,j} = (pop_i * travel[i,j] + pop_j * travel[j,i]) / (pop_i + pop_j)
        dtilde = (pop_i * travel_matrix + pop_j * travel_matrix.T) / (pop_i + pop_j)
        # Wherever division yielded NaN (e.g., both pop_i and pop_j zero), replace with 0
        dtilde[np.isnan(dtilde)] = 0

    return dtilde  # Return the computed dtilde matrix as a numpy array

def compute_d_matrix(dtilde_matrix, populations):
    """
    Compute the final d matrix from dtilde and population array, handling zero-population edge cases.
    
    dtilde_matrix: 2D numpy array (output from compute_dtilde_matrix)
    populations:   1D numpy array of population values
    """
    n = len(populations)  # number of tracts
    # Initialize an n×n zero array for the final d matrix
    d_matrix = np.zeros((n, n))
    
    # Loop over each pair of tracts (i, j)
    for i in range(n):
        for j in range(n):
            popi, popj = populations[i], populations[j]
            # If both tracts have zero population, set distance to 0
            if popi == 0 and popj == 0:
                d_matrix[i, j] = 0
            # If tract i has zero population, use the reverse entry from dtilde[j,i]
            elif popi == 0:
                d_matrix[i, j] = dtilde_matrix[j, i]
            # If tract j has zero population, use the direct entry from dtilde[i,j]
            elif popj == 0:
                d_matrix[i, j] = dtilde_matrix[i, j]
            else:
                # Otherwise, compute weighted average:
                # d_{i,j} = (popi * dtilde[i,j] + popj * dtilde[j,i]) / (popi + popj)
                d_matrix[i, j] = (
                    (popi * dtilde_matrix[i, j] + popj * dtilde_matrix[j, i])
                    / (popi + popj)
                )
    return d_matrix  # Return the computed d matrix

def process_files(travel_type, input_csv_path, travel_matrix_path, output_folder):
    """
    Read a tract_data CSV and its corresponding raw travel-time .npy file, compute
    dtilde and d matrices, then save each to CSV and NPY in the output folder.
    
    travel_type:         either 'drive' or 'walk' (string)
    input_csv_path:      full path to "<base_name>_tract_data.csv"
    travel_matrix_path:  full path to "<base_name>_{travel_type}.npy"
    output_folder:       directory where resulting files should be saved
    """
    # Load the tract-level data CSV into a pandas DataFrame
    df = pd.read_csv(input_csv_path)
    # Extract the population column "B01003_001E" (total population); coerce errors to NaN, fill with 0
    populations = pd.to_numeric(df['B01003_001E'], errors='coerce').fillna(0).values.astype(float)
    # Extract the tract IDs as a pandas Series, then as a numpy-friendly index
    ids = df['ID'].reset_index(drop=True)

    # Load the precomputed travel-time matrix from .npy file
    travel_matrix = np.load(travel_matrix_path)
    # Verify dimensions match the number of population entries; else, raise an error
    if travel_matrix.shape[0] != len(populations):
        raise ValueError("Mismatch: travel matrix size does not match number of population entries.")

    # Compute the intermediate dtilde matrix
    dtilde = compute_dtilde_matrix(travel_matrix, populations)
    # Compute the final d matrix from dtilde
    dmat = compute_d_matrix(dtilde, populations)

    # Derive base_name from the CSV filename (e.g., "CityX_tract_data.csv" → "CityX")
    base_name = os.path.basename(input_csv_path).replace('_tract_data.csv', '')

    # Loop over both matrices (dtilde then d) and label them
    for matrix, label in zip([dtilde, dmat], ['dtilde', 'd']):
        # Convert numpy matrix into a pandas DataFrame with tract IDs as row/column labels
        df_matrix = pd.DataFrame(matrix, index=ids, columns=ids)
        # Build output paths for CSV and NPY using the format:
        #   "<base_name>_<label>_<travel_type>.csv" and ".npy"
        csv_path = os.path.join(output_folder, f"{base_name}_{label}_{travel_type}.csv")
        npy_path = os.path.join(output_folder, f"{base_name}_{label}_{travel_type}.npy")
        # Save the labeled matrix DataFrame to CSV (quoting all fields for safety)
        df_matrix.to_csv(csv_path, quoting=csv.QUOTE_ALL)
        # Save the raw numpy matrix to .npy for faster reloading later
        np.save(npy_path, matrix)
        # Print a confirmation to stdout
        print(f"Saved {label} matrix for {base_name} ({travel_type}) to CSV and NPY.")

def batch_process_all(input_folder, travel_folder, output_folder, travel_types=['drive', 'walk']):
    """
    Iterate over all "*_tract_data.csv" files in input_folder. For each one, check
    if corresponding travel-time matrices exist in travel_folder for each travel_type.
    If found, call process_file() to generate and save dtilde/d matrices.
    
    input_folder:    directory containing "<base_name>_tract_data.csv" files
    travel_folder:   directory containing "<base_name>_{drive,walk}.npy" files
    output_folder:   directory to save computed matrices (will be created if needed)
    travel_types:    list of travel types to look for (default ['drive','walk'])
    """
    # Ensure the output directory exists; create it if necessary
    os.makedirs(output_folder, exist_ok=True)
    # Ensure travel_types is a list, not a string
    if isinstance(travel_types, str):
        travel_types = [travel_types]
    # Find all CSVs in input_folder matching "*_tract_data.csv"
    csv_files = glob.glob(os.path.join(input_folder, '*_tract_data.csv'))

    # Loop over each tract_data CSV file
    for csv_file in csv_files:
        # Derive base_name (e.g., "CityX" from "CityX_tract_data.csv")
        base_name = os.path.basename(csv_file).replace('_tract_data.csv', '')
        # For each specified travel_type (e.g., 'drive' and 'walk')
        for ttype in travel_types:
            # Build the expected travel-time .npy filename
            travel_file = os.path.join(travel_folder, f"{base_name}_{ttype}.npy")
            # If the file exists, process it
            if os.path.exists(travel_file):
                process_files(ttype, csv_file, travel_file, output_folder)
            else:
                # Otherwise, print a warning and skip
                print(f"Warning: {travel_file} not found — skipping.")

def load_labeled_matrix(npy_path, tract_csv_path):
    """
    Load a .npy matrix and label it with tract IDs from a matching CSV file.
    
    npy_path:        Path to the .npy file (e.g., "CityX_dtilde_drive.npy")
    tract_csv_path:  Path to the corresponding "_tract_data.csv" (e.g., "CityX_tract_data.csv")
    
    Returns:
        pd.DataFrame: The loaded matrix with row/column indices set to tract IDs
    """
    # Load the raw numpy matrix from .npy
    matrix = np.load(npy_path)
    # Read the tract_data CSV into a pandas DataFrame
    df = pd.read_csv(tract_csv_path)
    # Extract the "ID" column as a list of tract identifiers
    ids = df['ID'].reset_index(drop=True)

    # Verify the matrix size matches the number of IDs; otherwise, error out
    if matrix.shape[0] != len(ids):
        raise ValueError("Matrix size and number of IDs do not match.")

    # Wrap the numpy array in a pandas DataFrame, labeling rows and columns with tract IDs
    return pd.DataFrame(matrix, index=ids, columns=ids)

if __name__ == "__main__":
    # If this file is run as a script, these hard-coded paths will be used.
    # Users can modify these to point to their local directories.

    input_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_tract"
    # Folder containing "<base_name>_tract_data.csv" files

    travel_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\travel_matrices"
    # Folder containing raw travel-time ".npy" files named like "<base_name>_drive.npy" or "_walk.npy"

    output_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\weight_travel_time_Results"
    # Folder where computed dtilde/d matrices will be saved (CSV & NPY)

    # Run the batch processing over all tract_data CSVs and travel .npy files
    batch_process_all(input_folder, travel_folder, output_folder)
