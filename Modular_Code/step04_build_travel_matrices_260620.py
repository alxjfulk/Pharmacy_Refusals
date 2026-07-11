# -*- coding: utf-8 -*-
"""
compute_t_matrices_251230.py

Purpose
-------
Convert raw (directed) travel-time matrices into the weighted-distance matrices
used by downstream TDA steps (d̃ / "dtilde" and final d).

Key update (2025-12-30)
----------------------
The original Hickok polling-sites pipeline defines d̃ to first adjust travel
times for mode-access (car vs non-car), then defines d as the population-weighted
symmetrization of d̃.


- For single-mode matrices (walk-only or drive-only):
    d̃_{i,j} = t_{i,j} + t_{j,i}   (round-trip time; diagonal = 0)

- For multimodal (car access vs not), WITHOUT public transport:
    walk_rt  = t_walk[i,j]  + t_walk[j,i]
    drive_rt = t_drive[i,j] + t_drive[j,i]
    driver_min     = min(walk_rt, drive_rt)
    non_driver_min = walk_rt
    d̃_{i,j} = C_i * driver_min + (1 - C_i) * non_driver_min

  where C_i is a proxy for the proportion of the population with access to a car
  in tract i (we use the "vehicles_per_capita" style variable from the tract CSV,
  clipped to [0,1]).

Then, as in the polling notebook, we compute:

    d_{i,j} = ( P_i * d̃_{i,j} + P_j * d̃_{j,i} ) / (P_i + P_j)

with zero-pop edge-case handling.

Outputs are now date-tagged (or suffix-tagged) to prevent overwriting when
multiple travel matrices exist for the same city.
"""

import os                 # file paths and directory operations
import glob               # file discovery (pattern matching)
import csv                # control CSV quoting behavior
import re                 # robust filename tag parsing

import pandas as pd       # tabular data
import numpy as np        # numerical arrays / matrix operations


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _parse_tag_from_matrix_path(matrix_path: str, base_name: str, travel_type: str) -> str:
    """
    Extract a run-tag / suffix from a travel-matrix filename.

    Expected pattern (from Step 3):
        <base_name>_<travel_type>_<TAG>.npy

    Examples:
        "Milwaukee_drive_2025-12-30.npy"  -> "2025-12-30"
        "Milwaukee_walk_2025-12-26.npy"   -> "2025-12-26"

    If no tag is present, returns "notag".
    """
    fname = os.path.basename(matrix_path)
    # Build a defensive regex that captures anything between "..._<travel_type>_" and ".npy"
    # NOTE: base_name may contain spaces/odd chars, but os.path.basename(csv_file) is used
    # for base_name elsewhere and should match the travel-matrix prefix in practice.
    pattern = rf"^{re.escape(base_name)}_{re.escape(travel_type)}_(.+)\.npy$"
    m = re.match(pattern, fname)
    if m:
        return m.group(1)
    return "notag"


def _get_car_prop_vector(df: pd.DataFrame) -> np.ndarray:
    """
    Pull (or derive) a car-access proxy vector C_i from the tract-level CSV.

    We look for common column names used in this project; if none are found, we
    default to 0 (i.e., everyone treated as non-driver / walk-only).

    Expected range:
        0.0 to 1.0

    If values look like percentages (0–100), we convert to proportions.
    If values exceed 1 (e.g., vehicles per capita > 1), we clip to 1.
    """
    # NOTE: We match case-insensitively because upstream code has used
    #       'Vehicles_per_capita' (capital V) in the output CSV.
    candidate_cols = [
        "vehicles_per_capita",
        "vehicle_per_capita",
        "vehicles_pc",
        "car_prop",
        "car_proportion",
        "cars_prop",
        "cars_proportion",
        "car_access",
        # Defensive: common typo in notes
        "vehicles_per_captia",
    ]

    # Case-insensitive match to avoid column-name issues
    col_lookup = {c.lower(): c for c in df.columns}
    col = None
    for c in candidate_cols:
        if c.lower() in col_lookup:
            col = col_lookup[c.lower()]
            break

    if col is None:
        # No car-access proxy available; treat everyone as non-driver (walk-only)
        print("Warning: no car-access column found in tract CSV; defaulting car_prop=0 for all tracts.")
        return np.zeros(len(df), dtype=float)

    # Coerce to numeric, fill missing with 0
    car_prop = pd.to_numeric(df[col], errors="coerce").fillna(0).values.astype(float)

    # Heuristic: if values look like percents (e.g., 35, 82), convert to [0,1]
    finite_vals = car_prop[np.isfinite(car_prop)]
    if finite_vals.size > 0:
        vmax = np.nanmax(finite_vals)
        # If vmax is clearly above 1, but still in a plausible percent range, scale down.
        if 1.01 < vmax <= 100.0:
            car_prop = car_prop / 100.0

    # Clip to [0,1] because vehicles_per_capita can exceed 1 and we're using it as a proxy proportion
    car_prop = np.clip(car_prop, 0.0, 1.0)

    return car_prop


def _compute_roundtrip_matrix(travel_matrix: np.ndarray) -> np.ndarray:
    """
    Compute a simple round-trip travel-time matrix:
        rt[i,j] = t[i,j] + t[j,i]
    with diagonal forced to 0.
    """
    rt = travel_matrix + travel_matrix.T
    # Force diagonal = 0 (no "travel" from a tract to itself in this context)
    np.fill_diagonal(rt, 0.0)
    return rt


def _get_matrix_ids(df: pd.DataFrame) -> pd.Series:
    """
    Return stable row/column labels for CSV matrix outputs.

    Step 2 now writes full census tract GEOIDs after matching ACS data by
    county FIPS + tract code. GEOID is preferred because 6-digit TRACT codes can
    repeat across counties. Older tract files without GEOID still work because
    the function falls back to TRACT and then ID.
    """
    for col in ["GEOID", "geoid", "TRACT", "ID"]:
        if col in df.columns:
            return df[col].reset_index(drop=True)
    raise ValueError("Could not find GEOID, TRACT, or ID column in the tract CSV for labeling.")

# -----------------------------------------------------------------------------
# Drive-time adjustments
# -----------------------------------------------------------------------------

def _apply_drive_parking_penalty(
    drive_matrix: np.ndarray,
    penalty_minutes: float = 5.0,
) -> np.ndarray:
    """
    Add a static *parking* penalty to every finite, off-diagonal entry in a drive-time matrix.

    Rationale: in many real-world "drive" trips, you spend extra time parking, walking from the
    parked car to the destination, and (sometimes) exiting a lot/garage. This is a simple proxy
    for that overhead.

    Notes on units:
    - This function tries to infer whether the matrix is in seconds or minutes.
      If the median positive finite entry is > 300, we assume *seconds* and convert minutes->seconds.
      Otherwise we assume the matrix is already in *minutes*.
    - Diagonal entries remain 0.
    """
    if penalty_minutes is None or penalty_minutes <= 0:
        return drive_matrix

    drive = np.array(drive_matrix, dtype=float, copy=True)

    # Infer units from typical magnitudes (robust to inf / zeros).
    finite = np.isfinite(drive)
    positive = (drive > 0) & finite
    med = np.nan
    if np.any(positive):
        med = float(np.nanmedian(drive[positive]))

    penalty = float(penalty_minutes)
    if np.isfinite(med) and med > 300.0:
        # Likely seconds
        penalty *= 60.0

    # Add to all finite off-diagonal entries
    n = drive.shape[0]
    offdiag = finite.copy()
    np.fill_diagonal(offdiag, False)
    drive[offdiag] = drive[offdiag] + penalty

    # Preserve diagonal convention
    np.fill_diagonal(drive, 0.0)
    return drive



# -----------------------------------------------------------------------------
# Core computations
# -----------------------------------------------------------------------------

def compute_dtilde_single_mode(travel_matrix: np.ndarray) -> np.ndarray:
    """
    Compute d̃ for a SINGLE mode (walk-only or drive-only) to mirror the polling notebook.

    d̃_{i,j} = t_{i,j} + t_{j,i}   (round-trip)
    """
    return _compute_roundtrip_matrix(travel_matrix)


def compute_dtilde_multimodal_no_transit(
    walk_matrix: np.ndarray,
    drive_matrix: np.ndarray,
    car_props: np.ndarray,
) -> np.ndarray:
    """
    Compute d̃ for a TWO-mode setting (walk + drive) with car-access weighting,
    mirroring the polling notebook but WITHOUT public transport.

    For i != j:
        walk_rt  = walk[i,j]  + walk[j,i]
        drive_rt = drive[i,j] + drive[j,i]
        driver_min     = min(walk_rt, drive_rt)
        non_driver_min = walk_rt
        d̃[i,j] = C_i * driver_min + (1 - C_i) * non_driver_min

    Diagonal forced to 0.
    """
    n = walk_matrix.shape[0]
    if walk_matrix.shape != drive_matrix.shape:
        raise ValueError("walk_matrix and drive_matrix shapes do not match.")
    if len(car_props) != n:
        raise ValueError("car_props length does not match matrix dimensions.")

    # Precompute round-trip matrices (mirrors the polling notebook's 'walk = t[i,j] + t[j,i]' pattern)
    walk_rt = _compute_roundtrip_matrix(walk_matrix)
    drive_rt = _compute_roundtrip_matrix(drive_matrix)

    # Initialize output
    dtilde = np.zeros((n, n), dtype=float)

    # Compute row-by-row so that C_i weights *origin i* (as in the polling notebook)
    for i in range(n):
        Ci = float(car_props[i])
        # For each destination j, take mins between walk and drive
        driver_min = np.minimum(walk_rt[i, :], drive_rt[i, :])
        non_driver_min = walk_rt[i, :]

        # Weighted mixture by car access at origin i
        dtilde[i, :] = Ci * driver_min + (1.0 - Ci) * non_driver_min

    # Ensure diagonal = 0
    np.fill_diagonal(dtilde, 0.0)

    return dtilde


def compute_d_matrix(dtilde_matrix: np.ndarray, populations: np.ndarray) -> np.ndarray:
    """
    Compute the final d matrix from d̃ and population array, handling zero-pop edge cases.

    Mirrors the polling notebook logic:
        if popi == 0 and popj == 0: d[i,j] = 0
        elif popi == 0:            d[i,j] = d̃[j,i]
        elif popj == 0:            d[i,j] = d̃[i,j]
        else:
            d[i,j] = ( P_i*d̃[i,j] + P_j*d̃[j,i] ) / (P_i + P_j)
    """
    n = len(populations)
    d_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            popi = float(populations[i])
            popj = float(populations[j])

            if popi == 0 and popj == 0:
                d_matrix[i, j] = 0.0
            elif popi == 0:
                d_matrix[i, j] = float(dtilde_matrix[j, i])
            elif popj == 0:
                d_matrix[i, j] = float(dtilde_matrix[i, j])
            else:
                d_matrix[i, j] = (popi * dtilde_matrix[i, j] + popj * dtilde_matrix[j, i]) / (popi + popj)

    return d_matrix


# -----------------------------------------------------------------------------
# I/O + batch processing
# -----------------------------------------------------------------------------

def _save_matrix_pair(
    base_name: str,
    tag: str,
    mode_label: str,
    dtilde: np.ndarray,
    dmat: np.ndarray,
    ids: pd.Series,
    output_folder: str,
) -> None:
    """
    Save d̃ and d matrices to CSV and NPY with defensive quoting and date-tagged filenames.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Label matrices with tract IDs for CSV outputs
    df_dtilde = pd.DataFrame(dtilde, index=ids, columns=ids)
    df_d = pd.DataFrame(dmat, index=ids, columns=ids)

    # Filenames include the tag to avoid overwrites
    dtilde_csv = os.path.join(output_folder, f"{base_name}_dtilde_{mode_label}_{tag}.csv")
    dtilde_npy = os.path.join(output_folder, f"{base_name}_dtilde_{mode_label}_{tag}.npy")
    d_csv = os.path.join(output_folder, f"{base_name}_d_{mode_label}_{tag}.csv")
    d_npy = os.path.join(output_folder, f"{base_name}_d_{mode_label}_{tag}.npy")

    # Save CSVs with QUOTE_ALL to prevent weird parsing issues downstream
    df_dtilde.to_csv(dtilde_csv, quoting=csv.QUOTE_ALL)
    df_d.to_csv(d_csv, quoting=csv.QUOTE_ALL)

    # Save raw NPYs for fast re-load
    np.save(dtilde_npy, dtilde)
    np.save(d_npy, dmat)

    print(f"Saved dtilde/d ({mode_label}, tag={tag}) for {base_name} -> {output_folder}")


def process_files(travel_type: str, input_csv_path: str, travel_matrix_path: str, output_folder: str, drive_parking_penalty_minutes: float = 5.0) -> None:
    """
    Backwards-compatible wrapper (older pipeline versions called process_files() per travel_type).

    This now computes SINGLE-MODE d̃/d (round-trip d̃, then population-weighted d),
    and saves outputs with the parsed TAG in the filename.

    travel_type:        'drive' or 'walk'
    input_csv_path:     "<base_name>_tract_data.csv"
    travel_matrix_path: "<base_name>_{drive|walk}_<TAG>.npy"
    output_folder:      directory for outputs
    """
    if travel_type not in {"drive", "walk"}:
        raise ValueError("travel_type must be 'drive' or 'walk'.")

    if travel_type == "drive":
        process_city_tag(
            input_csv_path=input_csv_path,
            drive_matrix_path=travel_matrix_path,
            walk_matrix_path=None,
            drive_parking_penalty_minutes=drive_parking_penalty_minutes,
            output_folder=output_folder,
        )
    else:
        process_city_tag(
            input_csv_path=input_csv_path,
            drive_matrix_path=None,
            walk_matrix_path=travel_matrix_path,
            output_folder=output_folder,
        )

def process_city_tag(
    input_csv_path: str,
    drive_matrix_path: str | None,
    walk_matrix_path: str | None,
    output_folder: str,
    drive_parking_penalty_minutes: float = 5.0,
) -> None:
    """
    For a single city + tag, compute and save:

    - walk-only   d̃/d (if walk_matrix_path provided)
    - drive-only  d̃/d (if drive_matrix_path provided)
    - multimodal  d̃/d (if BOTH drive and walk matrices provided)

    Multimodal uses car-access weighting and NO public transport.
    Drive-time matrices can optionally include a static parking penalty (minutes) added to off-diagonal entries.
    """
    df = pd.read_csv(input_csv_path)

    # Population used for final d (same as before)
    populations = pd.to_numeric(df["B01003_001E"], errors="coerce").fillna(0).values.astype(float)

    # Row/column labels for CSV output. Prefer full GEOID when available because
    # 6-digit TRACT codes are not unique across counties.
    ids = _get_matrix_ids(df)

    # Car-access proxy used ONLY for multimodal d̃
    car_props = _get_car_prop_vector(df)

    base_name = os.path.basename(input_csv_path).replace("_tract_data.csv", "")

    # ----------------
    # Walk-only
    # ----------------
    if walk_matrix_path is not None:
        walk = np.load(walk_matrix_path)
        if walk.shape[0] != len(populations):
            raise ValueError(f"Mismatch: walk matrix size ({walk.shape}) does not match number of tracts ({len(populations)}).")
        tag = _parse_tag_from_matrix_path(walk_matrix_path, base_name, "walk")

        dtilde_walk = compute_dtilde_single_mode(walk)
        d_walk = compute_d_matrix(dtilde_walk, populations)

        _save_matrix_pair(base_name, tag, "walk", dtilde_walk, d_walk, ids, output_folder)

    # ----------------
    # Drive-only
    # ----------------
    if drive_matrix_path is not None:
        drive = _apply_drive_parking_penalty(np.load(drive_matrix_path), drive_parking_penalty_minutes)
        if drive.shape[0] != len(populations):
            raise ValueError(f"Mismatch: drive matrix size ({drive.shape}) does not match number of tracts ({len(populations)}).")
        tag = _parse_tag_from_matrix_path(drive_matrix_path, base_name, "drive")

        dtilde_drive = compute_dtilde_single_mode(drive)
        d_drive = compute_d_matrix(dtilde_drive, populations)

        _save_matrix_pair(base_name, tag, "drive", dtilde_drive, d_drive, ids, output_folder)

    # ----------------
    # Multimodal (walk + drive)
    # ----------------
    if (walk_matrix_path is not None) and (drive_matrix_path is not None):
        walk = np.load(walk_matrix_path)
        drive = _apply_drive_parking_penalty(np.load(drive_matrix_path), drive_parking_penalty_minutes)

        if walk.shape != drive.shape:
            raise ValueError(f"Mismatch: walk matrix shape {walk.shape} != drive matrix shape {drive.shape}.")
        if walk.shape[0] != len(populations):
            raise ValueError(f"Mismatch: matrices size ({walk.shape}) does not match number of tracts ({len(populations)}).")

        # Prefer a tag that is consistent between the paired matrices.
        base_name = os.path.basename(input_csv_path).replace("_tract_data.csv", "")
        walk_tag = _parse_tag_from_matrix_path(walk_matrix_path, base_name, "walk")
        drive_tag = _parse_tag_from_matrix_path(drive_matrix_path, base_name, "drive")
        tag = walk_tag if walk_tag == drive_tag else f"walk-{walk_tag}__drive-{drive_tag}"

        dtilde_multi = compute_dtilde_multimodal_no_transit(walk, drive, car_props)
        d_multi = compute_d_matrix(dtilde_multi, populations)

        _save_matrix_pair(base_name, tag, "multi", dtilde_multi, d_multi, ids, output_folder)


def batch_process_all(
    input_folder: str,
    travel_folder: str,
    output_folder: str,
    travel_types: list[str] = ["drive", "walk"],
    drive_parking_penalty_minutes: float = 5.0,
) -> None:
    """
    Iterate over all "*_tract_data.csv" files in input_folder and compute
    d̃/d outputs for each available travel-matrix tag.

    Unlike the older version, this function:
      - pairs drive+walk matrices by TAG (when possible),
      - saves outputs with TAG in filenames to prevent overwriting,
      - also computes multimodal (walk+drive) outputs when both are available.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Find all tract_data CSVs
    csv_files = glob.glob(os.path.join(input_folder, "*_tract_data.csv"))

    for csv_file in csv_files:
        base_name = os.path.basename(csv_file).replace("_tract_data.csv", "")

        # Discover all drive/walk matrices for this base_name
        drive_files = glob.glob(os.path.join(travel_folder, f"{base_name}_drive_*.npy"))
        walk_files = glob.glob(os.path.join(travel_folder, f"{base_name}_walk_*.npy"))

        # Build tag -> path maps for pairing
        drive_map = { _parse_tag_from_matrix_path(p, base_name, "drive"): p for p in drive_files }
        walk_map  = { _parse_tag_from_matrix_path(p, base_name, "walk"):  p for p in walk_files  }

        all_tags = sorted(set(drive_map.keys()) | set(walk_map.keys()))

        if not all_tags:
            print(f"Warning: no travel matrices found for {base_name} in {travel_folder}. Skipping.")
            continue

        for tag in all_tags:
            drive_path = drive_map.get(tag)
            walk_path = walk_map.get(tag)

            # If only one mode exists for this tag, we still compute the single-mode output(s).
            # If both exist, we also compute multimodal.
            process_city_tag(
                input_csv_path=csv_file,
                drive_matrix_path=drive_path,
                walk_matrix_path=walk_path,
                drive_parking_penalty_minutes=drive_parking_penalty_minutes,
                output_folder=output_folder,
            )


# -----------------------------------------------------------------------------
# Convenience loader (not used currently, holdover from previous version)
# -----------------------------------------------------------------------------

def load_labeled_matrix(npy_path: str, tract_csv_path: str) -> pd.DataFrame:
    """
    Load a .npy matrix and label it with tract IDs from a matching CSV file.
    """
    matrix = np.load(npy_path)
    df = pd.read_csv(tract_csv_path)

    # Prefer GEOID when present; fall back to TRACT or ID for older files.
    ids = _get_matrix_ids(df)

    if matrix.shape[0] != len(ids):
        raise ValueError("Matrix size and number of IDs do not match.")

    return pd.DataFrame(matrix, index=ids, columns=ids)


if __name__ == "__main__":
    # If run as a script, these hard-coded paths are used.
    # Modify to point to your local directories.

    input_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_tract"
    travel_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\travel_matrices"
    output_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\weight_travel_time_Results"

    batch_process_all(input_folder, travel_folder, output_folder)