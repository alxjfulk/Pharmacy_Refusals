# -*- coding: utf-8 -*-
"""Step 4 - turn the Step 3 travel times into the six distance matrices.

WHAT STEP 4 READS (from Step 3; times in seconds)
-------------------------------------------------
Every matrix has one row and one column per pharmacy, in the order of the
manifest's matrix_index column (row k and column k are the same pharmacy).

- walk/matrix.npy:  walk[i, j]  = one-way walking time from pharmacy i to pharmacy j
  (shortest street path divided by walking speed). Walking is the same both ways.
- drive/matrix.npy: drive[i, j] = one-way driving time from i to j at the speed
  limits recorded in OpenStreetMap (no traffic). It can differ by direction
  because of one-way streets, and it is NaN if j can't be reached by car from i.
- manifest.csv: one row per pharmacy, including two values for the tract the
  pharmacy sits in: its population, P_i (B01003_001E), and its vehicles per
  person, used as the share of people with car access, C_i (Vehicles_per_capita).

WHAT STEP 4 WRITES (still in seconds; Step 5 converts to minutes)
-----------------------------------------------------------------
Two matrices for each travel mode (walk, drive, multimodal):

- dtilde: the round-trip time from pharmacy i to pharmacy j and back.
- d: the two directions of dtilde combined into one value, weighted by the tract
  populations. d is symmetric (d[i, j] == d[j, i]), as persistent homology calculations require.
  This is the form Steps 5-7 analyze by default (ANALYSIS_MATRIX_FORMS = ["d"]).

HOW THE PIECES FIT TOGETHER (build_all_derived_matrices, at the bottom)
-----------------------------------------------------------------------
1. Read the manifest and both Step 3 matrices, and check they line up.
2. Get P_i and C_i for every pharmacy ............ get_car_prop_vector
3. Add the driving overhead (parking, walking to and from the car) to every
   one-way drive ................................. apply_drive_parking_penalty
       drive[i, j] + overhead
4. Round trips for each single mode .............. compute_dtilde_single_mode
       walk_dtilde[i, j]  = walk[i, j]  + walk[j, i]
       drive_dtilde[i, j] = drive[i, j] + drive[j, i]   (overhead counted twice)
5. Multimodal round trip ......................... compute_dtilde_multimodal_no_transit
       multimodal_dtilde[i, j] = C_i * min(walk_dtilde[i, j], drive_dtilde[i, j])
                               + (1 - C_i) * walk_dtilde[i, j]
6. Population weighting, for each mode ........... compute_d_matrix
       d[i, j] = (P_i * dtilde[i, j] + P_j * dtilde[j, i]) / (P_i + P_j)
7. Save all six matrices, each with a metadata file, then a summary metadata file.

The diagonal (a pharmacy to itself) is always 0.

WORKED EXAMPLE (two pharmacies, i and j)
----------------------------------------
    walk[i, j] = walk[j, i] = 900 s     drive[i, j] = 300 s, drive[j, i] = 360 s
    overhead = 2.5 min = 150 s per leg  C_i = 0.6, C_j = 0.3
    P_i = 3000, P_j = 1000

    walk_dtilde[i, j]  = 900 + 900                  = 1800 s (30 min)
    drive_dtilde[i, j] = (300 + 150) + (360 + 150)  =  960 s (16 min)
    multimodal_dtilde[i, j] = 0.6 * 960 + 0.4 * 1800 = 1296 s
    multimodal_dtilde[j, i] = 0.3 * 960 + 0.7 * 1800 = 1548 s
    multimodal_d[i, j] = (3000 * 1296 + 1000 * 1548) / 4000 = 1359 s (22.65 min)
    walk_d[i, j] = 1800 s and drive_d[i, j] = 960 s (unchanged; see below)

WHY d EQUALS dtilde FOR WALK AND DRIVE
--------------------------------------
A round trip is the same in both directions, so walk_dtilde and drive_dtilde are
already symmetric, and a weighted average of two equal numbers is that number.
Population weighting only changes the multimodal matrix, whose two directions
differ because C_i belongs to the starting pharmacy. (Pairs where both tracts
have population 0 are the exception; see compute_d_matrix.)
--------------------------------------
Step 4 always writes all six matrices. ANALYSIS_TRAVEL_MODES and
ANALYSIS_MATRIX_FORMS in Step 0 choose which ones Steps 5-7 analyze. If a dtilde
matrix is analyzed, Step 5 makes it symmetric by keeping the smaller of the two
directions for each pair.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common import (
    ALL_DERIVED_MATRIX_TYPES,
    ANALYSIS_TAG_COLUMN,
    file_sha256,
    ordered_id_checksum,
    utc_now_iso,
    validate_analysis_tag,
    validate_pharmacy_ids,
    write_json,
)


# =============================================================================
# Inputs for the formulas: car share (C_i)
# =============================================================================

def get_car_prop_vector(manifest: pd.DataFrame) -> np.ndarray:
    """Return C_i, the share of people with car access, for every pharmacy.

    C_i comes from the tract the pharmacy sits in: Step 2's Vehicles_per_capita,
    which is household vehicles divided by total population. Step 4 treats it as
    the share of trips made by someone who can drive (see
    compute_dtilde_multimodal_no_transit). The result is in manifest order, with
    every value between 0 and 1.

    Rules, in order:
    - The column is found by name, ignoring capital letters. If none is found, 
      C_i = 0 for every pharmacy (everyone walks) and a warning is printed.
    - Missing or non-numeric values become 0 (walking only). This includes tracts
      with population 0, where Step 2 can't divide by the population; treating
      them as walk-only is a documented limitation.
    - Finally every value is capped to the range 0 to 1, so a tract with more
      vehicles than people counts as full car access (C_i = 1).
    """
    # Accepted column names, most likely first.
    candidates = [
        "Vehicles_per_capita",
        "vehicles_per_capita",
        "vehicle_per_capita",
        "vehicles_pc",
        "car_prop",
        "car_proportion",
        "cars_prop",
        "cars_proportion",
        "car_access",
        "vehicles_per_captia",
    ]
    # Map lower-case column names to the real names so the search ignores case.
    lookup = {column.lower(): column for column in manifest.columns}
    column = next((lookup[name.lower()] for name in candidates if name.lower() in lookup), None)
    if column is None:
        print("WARNING: no vehicle-access column found; using car_prop=0 for all rows.")
        return np.zeros(len(manifest), dtype=float)
    # Text or blank cells become NaN, then 0 (walking only).
    values = pd.to_numeric(manifest[column], errors="coerce").fillna(0).to_numpy(dtype=float)
    # A share can't be below 0 or above 1.
    return np.clip(values, 0.0, 1.0)


# =============================================================================
# Building blocks for the round-trip (dtilde) matrices
# =============================================================================

def compute_roundtrip_matrix(travel_matrix: np.ndarray) -> np.ndarray:
    """Add each trip to its return trip: roundtrip[i, j] = t[i, j] + t[j, i].

    Example (seconds): t[i, j] = 300 and t[j, i] = 420 give 720 for both
    roundtrip[i, j] and roundtrip[j, i], so the result is always symmetric.

    If either leg is missing (NaN) or impossible (infinite), the round trip is
    NaN or infinite too, since the round trip can't be completed. The diagonal
    is set to 0.
    """
    matrix = np.asarray(travel_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Travel matrix must be square; received {matrix.shape}.")
    # matrix.T swaps rows and columns, so matrix.T[i, j] is the return trip t[j, i].
    roundtrip = matrix + matrix.T
    np.fill_diagonal(roundtrip, 0.0)
    return roundtrip


def apply_drive_parking_penalty(
    drive_matrix: np.ndarray,
    penalty_minutes: float = 2.5,
    *,
    input_time_unit: str = "seconds",
) -> np.ndarray:
    """Add a fixed overhead to every one-way drive, for parking and getting to the car.

        drive[i, j] + overhead, for every pair i != j that has a drive time

    Missing drive times (NaN) stay missing and the diagonal stays 0. A round trip
    has two legs, so the round-trip overhead is twice the penalty: Step 0's
    DRIVE_PARKING_PENALTY_MINUTES = 2.5 adds 5 minutes to every drive_dtilde entry.

    The penalty is given in minutes but the matrix is in seconds (Step 3's unit),
    so it is converted first: 2.5 minutes becomes 150 seconds. Step 0 passes the
    unit explicitly.

    The 2.5 default only applies when Step 4 is run on its own; Step 0 always
    passes DRIVE_PARKING_PENALTY_MINUTES. Returns a new matrix; the input is not
    changed.
    """
    drive = np.asarray(drive_matrix, dtype=float).copy()
    # No penalty requested: return the copy unchanged (apart from a zero diagonal).
    if penalty_minutes is None or float(penalty_minutes) <= 0:
        np.fill_diagonal(drive, 0.0)
        return drive
    # Converting from input unit to seconds.
    unit = str(input_time_unit).strip().lower()
    factors = {"seconds": 60.0, "minutes": 1.0, "hours": 1.0 / 60.0}
    if unit not in factors:
        raise ValueError("input_time_unit must be seconds, minutes, or hours.")
    penalty = float(penalty_minutes) * factors[unit]  # e.g. 2.5 min * 60 = 150 s
    # Add it only to actual drive times: not to missing (NaN) pairs or the diagonal.
    finite = np.isfinite(drive)
    np.fill_diagonal(finite, False)
    drive[finite] += penalty
    np.fill_diagonal(drive, 0.0)
    return drive


def compute_dtilde_single_mode(travel_matrix: np.ndarray) -> np.ndarray:
    """Round-trip matrix (dtilde) for one travel mode: walk or drive.

    The same calculation as compute_roundtrip_matrix; the separate name keeps
    build_all_derived_matrices readable next to the multimodal version below.
    """
    return compute_roundtrip_matrix(travel_matrix)


def compute_dtilde_multimodal_no_transit(
    walk_matrix: np.ndarray,
    drive_matrix: np.ndarray,
    car_props: np.ndarray,
) -> np.ndarray:
    """Round-trip matrix (dtilde) that mixes walking and driving by car access.

    For a round trip that starts at pharmacy i:
    - a share C_i of people have a car and take whichever round trip is faster,
      walking or driving: min(walk_rt[i, j], drive_rt[i, j]);
    - the other 1 - C_i walk: walk_rt[i, j].
    The expected round-trip time averages the two groups by their shares:

        multimodal_dtilde[i, j] = C_i * min(walk_rt[i, j], drive_rt[i, j])
                                + (1 - C_i) * walk_rt[i, j]

    Example: C_i = 0.6, walking round trip 1800 s, driving round trip 960 s:
        0.6 * 960 + 0.4 * 1800 = 576 + 720 = 1296 s (21.6 minutes).

    This is Hickok et al.'s dtilde without public transit (hence 'no_transit'):
    in the paper, people without a car take the faster of transit and walking, and
    people with a car the fastest of car, transit and walking.

    C_i belongs to the starting pharmacy's tract, so multimodal_dtilde[i, j] and
    multimodal_dtilde[j, i] usually differ; compute_d_matrix combines them.

    The drive matrix passed in should already include the overhead. If a pair has
    no drive time (NaN), np.minimum returns NaN, so the multimodal value is NaN even
    when the pair can be walked. Step 5 then treats that pair as unreachable.
    """
    walk = np.asarray(walk_matrix, dtype=float)
    drive = np.asarray(drive_matrix, dtype=float)
    if walk.shape != drive.shape:
        raise ValueError("walk_matrix and drive_matrix shapes do not match.")
    if len(car_props) != walk.shape[0]:
        raise ValueError("car_props length does not match matrix dimension.")
    walk_rt = compute_roundtrip_matrix(walk)
    drive_rt = compute_roundtrip_matrix(drive)
    output = np.zeros_like(walk_rt, dtype=float)
    # One row at a time: row i holds every round trip that starts at pharmacy i,
    # so the whole row uses the same car share C_i.
    for row_index in range(walk.shape[0]):
        # People with a car: the faster of the two round trips, pair by pair.
        driver_min = np.minimum(walk_rt[row_index, :], drive_rt[row_index, :])
        # People without a car: the walking round trip.
        non_driver = walk_rt[row_index, :]
        # Weighted average of the two groups.
        output[row_index, :] = (
            car_props[row_index] * driver_min
            + (1.0 - car_props[row_index]) * non_driver
        )
    np.fill_diagonal(output, 0.0)
    return output


# =============================================================================
# Population weighting: from dtilde to d
# =============================================================================

def compute_d_matrix(dtilde_matrix: np.ndarray, populations: np.ndarray) -> np.ndarray:
    """Combine the two directions of a dtilde matrix into one symmetric distance, d.

        d[i, j] = (P_i * dtilde[i, j] + P_j * dtilde[j, i]) / (P_i + P_j)

    where P_i is the population of pharmacy i's tract (Hickok et al., eq. 3.1).
    It is the average round-trip time for a person picked at random from the two
    tracts: people from i's tract start at i, people from j's tract start at j, so
    the larger tract gets more weight. Swapping i and j gives the same value, so
    d[i, j] == d[j, i].

    Example: P_i = 3000, P_j = 1000, dtilde[i, j] = 1296 s, dtilde[j, i] = 1548 s:
        (3000 * 1296 + 1000 * 1548) / 4000 = 1359 s (22.65 minutes).

    If dtilde[i, j] == dtilde[j, i], as for walk and drive, d is that same value.

    Zero populations:
    - One tract has population 0: only the other tract's direction counts.
      For example P_i = 0 gives d[i, j] = dtilde[j, i].
    - Both tracts have population 0: d[i, j] is set to 0, which makes the two
      pharmacies look like they are in the same place. This should be changed before 
      a final run.
    """
    dtilde = np.asarray(dtilde_matrix, dtype=float)
    populations = np.asarray(populations, dtype=float)
    if dtilde.shape != (len(populations), len(populations)):
        raise ValueError("Population vector length does not match dtilde dimensions.")
    n = len(populations)
    output = np.zeros((n, n), dtype=float)
    # Every ordered pair (i, j). The result is symmetric, so each pair's value is
    # calculated twice, once as (i, j) and once as (j, i).
    for i in range(n):
        for j in range(n):
            pop_i, pop_j = float(populations[i]), float(populations[j])
            if pop_i == 0 and pop_j == 0:
                # Both tracts empty: no weights to use (known issue, see docstring).
                output[i, j] = 0.0
            elif pop_i == 0:
                # Only j's tract has people, so only the trip starting at j counts.
                output[i, j] = float(dtilde[j, i])
            elif pop_j == 0:
                # Only i's tract has people, so only the trip starting at i counts.
                output[i, j] = float(dtilde[i, j])
            else:
                # Population-weighted average of the two directions (eq. 3.1).
                output[i, j] = (
                    pop_i * dtilde[i, j] + pop_j * dtilde[j, i]
                ) / (pop_i + pop_j)
    np.fill_diagonal(output, 0.0)
    return output


# =============================================================================
# Checks on the inputs
# =============================================================================

def validate_manifest(
    manifest: pd.DataFrame,
    expected_tag: str,
    context: str,
) -> list[str]:
    """Check the Step 3 manifest before any matrix is built from it.

    The manifest fixes the order of rows and columns in every matrix: row k
    (and column k) is the pharmacy whose matrix_index is k. This checks that the
    needed columns exist (including the population column, B01003_001E), that
    every row has this run's tag, that pharmacy IDs are unique, and that
    matrix_index runs 0, 1, 2, ... in order. Returns the pharmacy IDs in matrix
    order.
    """
    required = {"matrix_index", "pharmacy_id", "B01003_001E"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}.")
    validate_analysis_tag(manifest, expected_tag, context)
    ids = validate_pharmacy_ids(manifest, context)
    indices = pd.to_numeric(manifest["matrix_index"], errors="coerce")
    if indices.isna().any() or not np.array_equal(indices.to_numpy(), np.arange(len(manifest))):
        raise ValueError(f"{context} matrix_index is not sequential from zero.")
    return ids


def validate_matrix_shape(matrix: np.ndarray, manifest: pd.DataFrame, label: str) -> None:
    """Check that a matrix has exactly one row and one column per manifest pharmacy."""
    expected = (len(manifest), len(manifest))
    if matrix.ndim != 2 or matrix.shape != expected:
        raise ValueError(f"{label} shape {matrix.shape} does not match manifest {expected}.")


# =============================================================================
# Saving the results
# =============================================================================

def _write_matrix_output(
    output_root: Path,
    matrix_type: str,
    matrix: np.ndarray,
    ids: list[str],
    *,
    analysis_run_tag: str,
    source_manifest: Path,
    source_walk: Path,
    source_drive: Path,
    parking_penalty_minutes: float,
) -> dict[str, str]:
    """Save one matrix as three files in output_root/<matrix_type>/.

    - matrix.npy: the numbers themselves; this is what Step 5 reads.
    - matrix.csv: the same numbers with pharmacy IDs as row and column labels,
      for opening in Excel.
    - metadata.json: where the matrix came from (source files, overhead used),
      a checksum of the pharmacy order, and a checksum of matrix.npy that Step 0
      uses to notice if the file is changed later.

    Stops rather than overwrite existing files. Returns the three file paths.
    """
    directory = output_root / matrix_type
    directory.mkdir(parents=True, exist_ok=True)
    npy_path = directory / "matrix.npy"
    csv_path = directory / "matrix.csv"
    metadata_path = directory / "metadata.json"
    if any(path.exists() for path in (npy_path, csv_path, metadata_path)):
        raise FileExistsError(f"Step 4 output already exists for {matrix_type}: {directory}")
    np.save(npy_path, matrix)
    pd.DataFrame(matrix, index=ids, columns=ids).to_csv(csv_path)
    metadata = {
        "pipeline_step": 4,
        ANALYSIS_TAG_COLUMN: analysis_run_tag,
        "matrix_type": matrix_type,
        "matrix_dimension": len(ids),
        "ordered_pharmacy_id_checksum": ordered_id_checksum(ids),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "source_walk_matrix": str(source_walk.resolve()),
        "source_drive_matrix": str(source_drive.resolve()),
        "drive_parking_penalty_minutes": float(parking_penalty_minutes),
        "created_utc": utc_now_iso(),
        "matrix_npy": str(npy_path.resolve()),
        "matrix_csv": str(csv_path.resolve()),
        "matrix_sha256": file_sha256(npy_path),
    }
    write_json(metadata_path, metadata)
    return {
        "matrix_npy": str(npy_path),
        "matrix_csv": str(csv_path),
        "metadata": str(metadata_path),
    }


# =============================================================================
# Step 4 for one city: runs the pieces above in order
# =============================================================================

def build_all_derived_matrices(
    manifest_path: str | os.PathLike[str],
    walk_matrix_path: str | os.PathLike[str],
    drive_matrix_path: str | os.PathLike[str],
    output_folder: str | os.PathLike[str],
    *,
    analysis_run_tag: str,
    drive_parking_penalty_minutes: float = 5.0,
    input_time_unit: str = "seconds",
) -> dict[str, Any]:
    """Build and save all six Step 4 matrices for one city.

    Step 0 calls this once per city. The numbered comments below match the
    numbered list at the top of this file. As in apply_drive_parking_penalty, the
    5.0 default only applies when Step 4 is run on its own.
    """
    manifest_path = Path(manifest_path)
    walk_path = Path(walk_matrix_path)
    drive_path = Path(drive_matrix_path)
    output_root = Path(output_folder)
    step_metadata = output_root / "step04_metadata.json"
    # Step 0 skips a city whose Step 4 is already complete, so anything already in
    # this folder means an earlier run stopped partway. Stop rather than mix old
    # and new files.
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Step 4 output already exists in {output_root}; the master should validate and skip it."
        )

    # 1. Read the manifest and both Step 3 matrices, and check they line up.
    # IDs are read as text so values such as NPIs keep any leading zeros.
    manifest = pd.read_csv(
        manifest_path,
        dtype={"pharmacy_id": "string", "NPI": "string", "google_place_id": "string"},
    )
    ids = validate_manifest(manifest, analysis_run_tag, str(manifest_path))
    walk = np.load(walk_path, allow_pickle=False)
    raw_drive = np.load(drive_path, allow_pickle=False)
    validate_matrix_shape(walk, manifest, "walk")
    validate_matrix_shape(raw_drive, manifest, "drive")

    # 2. P_i: total population of each pharmacy's tract (missing values become 0).
    #    C_i: share of people with car access (see get_car_prop_vector).
    populations = pd.to_numeric(manifest["B01003_001E"], errors="coerce").fillna(0).to_numpy(float)
    car_props = get_car_prop_vector(manifest)

    # 3. Add the driving overhead to every one-way drive time.
    drive = apply_drive_parking_penalty(
        raw_drive,
        drive_parking_penalty_minutes,
        input_time_unit=input_time_unit,
    )

    # 4 and 6. Walking and driving: round trips (dtilde), then population weighting (d).
    #          For these two modes d comes out equal to dtilde (see the top of this file).
    dtilde_walk = compute_dtilde_single_mode(walk)
    d_walk = compute_d_matrix(dtilde_walk, populations)
    dtilde_drive = compute_dtilde_single_mode(drive)
    d_drive = compute_d_matrix(dtilde_drive, populations)
    # 5 and 6. Multimodal: car-share mix of the walking and driving round trips, then
    #          population weighting. This is where weighting changes the values.
    #          It is given the drive times that already include the overhead.
    dtilde_multimodal = compute_dtilde_multimodal_no_transit(walk, drive, car_props)
    d_multimodal = compute_d_matrix(dtilde_multimodal, populations)

    matrices = {
        "walk_d": d_walk,
        "walk_dtilde": dtilde_walk,
        "drive_d": d_drive,
        "drive_dtilde": dtilde_drive,
        "multimodal_d": d_multimodal,
        "multimodal_dtilde": dtilde_multimodal,
    }
    # Safety check: these six names must match the list the rest of the pipeline
    # expects (ALL_DERIVED_MATRIX_TYPES in pipeline_common).
    if set(matrices) != set(ALL_DERIVED_MATRIX_TYPES):
        raise RuntimeError("Internal Step 4 matrix-type definition is incomplete.")

    # 7. Save each matrix (with its own metadata), then the summary metadata file.
    #    Step 0 counts Step 4 as complete only when all of these files exist.
    output_root.mkdir(parents=True, exist_ok=True)
    written = {}
    for matrix_type in ALL_DERIVED_MATRIX_TYPES:
        written[matrix_type] = _write_matrix_output(
            output_root,
            matrix_type,
            matrices[matrix_type],
            ids,
            analysis_run_tag=analysis_run_tag,
            source_manifest=manifest_path,
            source_walk=walk_path,
            source_drive=drive_path,
            parking_penalty_minutes=drive_parking_penalty_minutes,
        )

    metadata = {
        "pipeline_step": 4,
        ANALYSIS_TAG_COLUMN: analysis_run_tag,
        "created_utc": utc_now_iso(),
        "source_manifest": str(manifest_path.resolve()),
        "source_walk_matrix": str(walk_path.resolve()),
        "source_drive_matrix": str(drive_path.resolve()),
        "matrix_dimension": len(ids),
        "ordered_pharmacy_id_checksum": ordered_id_checksum(ids),
        "drive_parking_penalty_minutes": float(drive_parking_penalty_minutes),
        "input_time_unit": input_time_unit,
        "matrix_types": list(ALL_DERIVED_MATRIX_TYPES),
        "outputs": written,
    }
    write_json(step_metadata, metadata)
    print("\n=== Step 4 summary ===")
    print(f"Generated matrix types: {', '.join(ALL_DERIVED_MATRIX_TYPES)}")
    return {"metadata": str(step_metadata), "outputs": written}


# Older name for build_all_derived_matrices, kept so that any code still using it
# keeps working. Nothing in the current pipeline calls it.
def batch_process_all(*args, **kwargs):
    """Same as build_all_derived_matrices (older name)."""
    return build_all_derived_matrices(*args, **kwargs)


# Run Step 4 for one city from the command line, without Step 0.
# Note: --parking-penalty-minutes defaults to 2.5 per leg (5 minutes per round
# trip)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build all derived d and dtilde matrices.")
    parser.add_argument("manifest")
    parser.add_argument("walk_matrix")
    parser.add_argument("drive_matrix")
    parser.add_argument("output_folder")
    parser.add_argument("analysis_run_tag")
    parser.add_argument("--parking-penalty-minutes", type=float, default=5.0)
    args = parser.parse_args()
    build_all_derived_matrices(
        args.manifest,
        args.walk_matrix,
        args.drive_matrix,
        args.output_folder,
        analysis_run_tag=args.analysis_run_tag,
        drive_parking_penalty_minutes=args.parking_penalty_minutes,
    )
