

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 08:56:14 2025

@author: bigfo
"""

# -*- coding: utf-8 -*-
"""
Pharmacy Refusals | Master Process Flow (step00_master_process_flow4_260620.py)
"""

import os
from datetime import date

# ---------------------------------------------------------------------------
# OPTIONAL: set your working directory
# ---------------------------------------------------------------------------
# Keep this master file in the same folder as the step
# files, or set DEFAULT_CODE_DIR to that folder before running.
DEFAULT_CODE_DIR = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\modular_code"

if os.path.isdir(DEFAULT_CODE_DIR):
    os.chdir(DEFAULT_CODE_DIR)

# ---------------------------------------------------------------------------
# Imports from modular pipeline
# ---------------------------------------------------------------------------
from step01_extract_PharmacyCityState_260620 import extract_pharmacy_data
from step02_attach_censustract_info_260620 import tie_census_info
from step03_build_raw_walk_and_drive_time_matrices_260620 import compute_matrices_dijkstra
from step04_build_travel_matrices_260620 import batch_process_all
import step05_run_TDA_260620 as giotto_persistence
import step06_pharmacy_mapping_260620 as pharmacy_mapping

# =============================================================================
# RUN CONTROLS 
# =============================================================================

# Which steps to run:
#   1 = Extract pharmacy data (per city function)
#   2 = Attach census tract info (global/folder-level function - it runs for all files matching a specific pattern)
#   3 = Build raw travel-time matrices (per city)
#   4 = Build weighted distance matrices (global)
#   5 = Persistence / TDA analysis (global)
#   6 = Mapping (static PNG + interactive HTML) (global)
RUN_STEPS = {6}  # examples: {1,2,3,4,5,6} or {1} or {1, 2, 3, 4}  or  {2, 3}  or  {4, 5}


# Edit this list whenever you want to change the subset you run.
CITY_STATE_LIST = [
    ("Las Vegas", "NV"),
    ("Atlanta", "GA"),
    ("Pittsburgh", "PA"),
    ("Albuquerque", "NM"),
    ("Austin", "TX"),
    # ("Forsythe", "GA"), #No NPIs returned
    # ("Dekalb", "GA"), #No NPIs returned
    ("Greenville", "SC")
    # ("Richland", "SC") #No NPIs returned
]

# If True: rerun selected steps even if their outputs exist
FORCE_RERUN = True

# If True: print what would happen, but do not actually run the functions
DRY_RUN = False

# Date tag for travel matrices
TRAVEL_TAG = date.today().isoformat()  # e.g., "2025-12-26"

# Step 2 parameter 
MAX_DISTANCE_MILES = 20

# Step 4 configuration 
TRAVEL_TYPES = ["drive", "walk"]

# Step 5 (TDA) configuration
# Matrices from Step 3 are in seconds; Step 4 derived matrices inherit those units.
TDA_INPUT_TIME_UNIT  = "seconds"   # "seconds" | "minutes" | "hours"
TDA_OUTPUT_TIME_UNIT = "minutes"   # "seconds" | "minutes" | "hours" (presentation/saved outputs)

# Which Step 4 matrix family/families should be passed into the persistence step?
# Step 4 still computes both dtilde and d. This variable only controls Step 5.
#
# Valid options:
#   ["d"]            -> run TDA only on the final population-symmetrized d matrices
#   ["dtilde"]       -> run TDA only on the origin-weighted dtilde matrices
#   ["d", "dtilde"] -> run TDA on both matrix families
#
# Background: following Hickok et al.'s polling-site construction, dtilde is the
# origin-weighted accessibility matrix and d is the population-weighted symmetric
# distance derived from dtilde. Keeping both available is useful for checking how
# the intermediate accessibility surface and the final symmetric distance affect PH.
TDA_MATRIX_METRICS = ["d"]#["d", "dtilde"]

# =============================================================================
# PATHS / OUTPUT FOLDERS (EDIT AS NEEDED)
# =============================================================================
PHARMACY_OUTPUT_DIR = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_pharmacy"
TRACT_OUTPUT_DIR    = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_tract"
RAW_TRAVEL_DIR      = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_travel_matrices"
WDM_OUTPUT_DIR      = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_WDM"
TDA_OUTPUT_DIR      = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_TDA"
MAP_OUTPUT_DIR      = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_graphics"

# =============================================================================
# API KEY (DO NOT HARD-CODE)
# =============================================================================
# Set this in Windows PowerShell (recommended):
#   $env:GOOGLE_PLACES_API_KEY="YOUR_KEY_HERE"
#
# Or set it permanently via System Environment Variables (search in windows "edit the system environment variables" > environment variables).
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()


def main():
    """Master runner for the Pharmacy Refusals pipeline."""

    # Convert to set once so membership checks are fast and consistent
    steps_to_run = set(RUN_STEPS)

    # Helper: should we run this step?
    def step_on(step_num: int) -> bool:
        return step_num in steps_to_run

    # Ensure output directories exist before we start
    for folder in [PHARMACY_OUTPUT_DIR, TRACT_OUTPUT_DIR, RAW_TRAVEL_DIR, WDM_OUTPUT_DIR, TDA_OUTPUT_DIR, MAP_OUTPUT_DIR]:
        os.makedirs(folder, exist_ok=True)

    # ------------------------------------------------------------------------
    # Step 1: Extract pharmacy data (per-city)
    # ------------------------------------------------------------------------
    if step_on(1):
        for city, state in CITY_STATE_LIST:
            print(f"\n=== Step 1 | {city}, {state} ===")

            pharm_file = os.path.join(
                PHARMACY_OUTPUT_DIR,
                f"Pharmacy_Data_{city}{state}.csv"
            )

            # Safety check: Step 1 needs the Google API key
            if not GOOGLE_PLACES_API_KEY:
                print("→ [SKIP] Step 1 cannot run because GOOGLE_PLACES_API_KEY is not set.")
                continue

            if (not FORCE_RERUN) and os.path.exists(pharm_file):
                print(f"→ [SKIP] Step 1 output exists: {pharm_file}")
                continue

            print("Step 1: Extracting pharmacy data...")
            if not DRY_RUN:
                extract_pharmacy_data(city, state, PHARMACY_OUTPUT_DIR, GOOGLE_PLACES_API_KEY)
    else:
        print("\n→ [OFF] Step 1")

    # ------------------------------------------------------------------------
    # Step 2: Attach census tract information + filter pharmacies too far away
    #         (GLOBAL sweep over folders; runs once)
    # ------------------------------------------------------------------------
    if step_on(2):
        print("\n=== Step 2: Attaching census tract information (global) ===")

        # Precondition: need at least one pharmacy CSV to process
        expected_pharm_files = [
            os.path.join(PHARMACY_OUTPUT_DIR, f"Pharmacy_Data_{city}{state}.csv")
            for city, state in CITY_STATE_LIST
        ]
        existing_pharm_files = [p for p in expected_pharm_files if os.path.exists(p)]

        if not existing_pharm_files:
            print("→ [SKIP] Step 2: no pharmacy CSVs found in PHARMACY_OUTPUT_DIR.")
        else:
            # Decide whether Step 2 is necessary when not forcing a rerun.
            # Because tie_census_info operates over folders, we only run it if any expected
            # tract output is missing (or FORCE_RERUN is True).
            expected_tract_files = [
                os.path.join(TRACT_OUTPUT_DIR, f"{city}{state}_tract_data.csv")
                for city, state in CITY_STATE_LIST
            ]
            missing_tract = [t for t in expected_tract_files if not os.path.exists(t)]

            if (not FORCE_RERUN) and (len(missing_tract) == 0):
                print("→ [SKIP] Step 2 outputs exist for all cities in CITY_STATE_LIST.")
            else:
                if missing_tract and (not FORCE_RERUN):
                    print(f"Step 2: {len(missing_tract)} tract file(s) missing; running folder sweep...")
                    for t in missing_tract[:10]:
                        print(f"   - missing: {t}")
                    if len(missing_tract) > 10:
                        print(f"   - ... +{len(missing_tract) - 10} more")
                else:
                    print("Step 2: Running folder sweep...")

                if not DRY_RUN:
                    tie_census_info(
                        PHARMACY_OUTPUT_DIR,
                        TRACT_OUTPUT_DIR,
                        max_distance_miles=MAX_DISTANCE_MILES
                    )
    else:
        print("\n→ [OFF] Step 2")

    # ------------------------------------------------------------------------
    # Step 3: Compute raw walk/drive travel-time matrices (per-city)
    # ------------------------------------------------------------------------
    if step_on(3):
        for city, state in CITY_STATE_LIST:
            print(f"\n=== Step 3 | {city}, {state} ===")

            tract_file = os.path.join(
                TRACT_OUTPUT_DIR,
                f"{city}{state}_tract_data.csv"
            )

            # Precondition: Step 3 expects tract output to exist
            if not os.path.exists(tract_file):
                print(f"→ [SKIP] Step 3 needs tract file but it's missing:\n    {tract_file}")
                continue

            # Date-based tag for versioning.
            # These names match Step 3's actual output and Step 4's expected input:
            #   <City><State>_drive_<TRAVEL_TAG>.npy
            #   <City><State>_walk_<TRAVEL_TAG>.npy
            city_tag = f"{city}{state}"
            drive_path = os.path.join(
                RAW_TRAVEL_DIR,
                f"{city_tag}_drive_{TRAVEL_TAG}.npy"
            )
            walk_path = os.path.join(
                RAW_TRAVEL_DIR,
                f"{city_tag}_walk_{TRAVEL_TAG}.npy"
            )

            drive_exists = os.path.exists(drive_path)
            walk_exists = os.path.exists(walk_path)

            if (not FORCE_RERUN) and drive_exists and walk_exists:
                print(f"→ [SKIP] Step 3 outputs exist:\n    {drive_path}\n    {walk_path}")
                continue

            # Decide which matrices to compute this run
            if FORCE_RERUN:
                do_drive = True
                do_walk = True
            else:
                do_drive = (not drive_exists)
                do_walk = (not walk_exists)

            tasks = []
            if do_drive:
                tasks.append("drive")
            if do_walk:
                tasks.append("walk")

            print(f"Step 3: Computing raw {' & '.join(tasks)} matrix{'s' if len(tasks) != 1 else ''}...")

            if not DRY_RUN:
                compute_matrices_dijkstra(
                    input_csv_path=tract_file,
                    output_dir=RAW_TRAVEL_DIR,
                    city=city,
                    state=state,
                    do_drive=do_drive,
                    do_walk=do_walk,
                    travel_tag=TRAVEL_TAG,
                    output_prefix=city_tag
                )
    else:
        print("\n→ [OFF] Step 3")
    # ------------------------------------------------------------------------
    # Step 4: Weighted distance matrices (global)
    # ------------------------------------------------------------------------
    if step_on(4):
        print("\nStep 4: Computing weighted distance matrices (d̃ and d)...")
        if not DRY_RUN:
            batch_process_all(
                input_folder=TRACT_OUTPUT_DIR,
                travel_folder=RAW_TRAVEL_DIR,
                output_folder=WDM_OUTPUT_DIR,
                travel_types=TRAVEL_TYPES
            )
    else:
        print("\n→ [OFF] Step 4")

    # ------------------------------------------------------------------------
    # Step 5: Persistence / TDA (global)
    # ------------------------------------------------------------------------
    if step_on(5):
        print("\n=== Step 5: Persistence analysis ===")

        # NOTE:
        #   - TDA_INPUT_TIME_UNIT/TDA_OUTPUT_TIME_UNIT control only how birth/death
        #     times are displayed and saved. They do not modify Step 4 matrices.
        #   - TDA_MATRIX_METRICS controls which Step 4 matrix families are analyzed.
        #     Step 4 always creates dtilde and d; this selector only filters inputs
        #     for the persistence calculation.
        if not DRY_RUN:
            giotto_persistence.process_all_matrices(
                matrix_folder_=WDM_OUTPUT_DIR,
                output_folder_=TDA_OUTPUT_DIR,
                input_time_unit_=TDA_INPUT_TIME_UNIT,
                output_time_unit_=TDA_OUTPUT_TIME_UNIT,
                max_homology_dim_=1,
                matrix_metrics_=TDA_MATRIX_METRICS
            )
    else:
        print("\n→ [OFF] Step 5")
    
    # ------------------------------------------------------------------------
    # Step 6: Mapping (global) - create static PNGs + interactive HTML maps + union-of-balls (GIF + slider)
    # ------------------------------------------------------------------------
    if step_on(6):
        print("\n=== Step 6: Mapping (static + interactive) ===")

        # This step scans TRACT_OUTPUT_DIR for "*_tract_data.csv" and writes:
        #   <CityState>_contextily.png
        #   <CityState>_folium.html
        #   <CityState>_union_balls_contextily.gif
        #   <CityState>_union_balls_folium.html
        # to MAP_OUTPUT_DIR.
        if not DRY_RUN:
            pharmacy_mapping.batch_map_folder(
                input_folder=TRACT_OUTPUT_DIR,
                output_folder=MAP_OUTPUT_DIR,
                pattern="*_tract_data.csv",

                # Existing outputs
                make_contextily=True,
                make_folium=True,

                # NEW outputs
                make_unionballs_contextily_gif=True,
                make_unionballs_folium_slider=False,

                # Styling
                point_color="red",
                point_size=30,
                circle_radius=5,

                # Union-of-balls tuning (edit as desired)
                ub_gif_frames=140,
                ub_gif_fps=3,
                ub_gif_eps_quantile=0.70,
                ub_gif_eps_max_m=None,        # set meters to force a fixed max ε
                ub_gif_show_edges=True,      # set True only for small point sets

                ub_html_steps=40,
                ub_html_eps_max_m=6000.0,     # meters (ε max); displayed radius is r=ε/2
                ub_html_zoom_start=11,
            )
        else:
            print("→ [DRY RUN] Step 6 mapping would run here.")
    else:
        print("\n→ [OFF] Step 6")

    print("\nAll selected steps completed successfully.")

if __name__ == "__main__":
    main()















































































































































































# =============================================================================
# OPTIONAL HELPER: Inspect saved .npy matrices (d / dtilde / walk / drive)
# -----------------------------------------------------------------------------
# PURPOSE
#   This block lets you run quick diagnostics on matrix outputs AFTER the pipeline
#   completes, without changing any core logic. It imports your helper module
#   (inspect_matrices.py) and calls its functions to print summary stats.
#
# HOW TO USE
#   1) Save the helper file as: inspect_matrices.py
#      Put it EITHER:
#        - in the same folder as this master_process_flow script, OR
#        - in a folder on your PYTHONPATH.
#   2) Set RUN_MATRIX_INSPECTOR = True when you want it to run.
#   3) Adjust MATRIX_FOLDER (defaults to MATRIX_OUTPUT_DIR if present).
#
# WHAT IT DOES
#   - Prints shape/dtype, NaN/Inf counts, diagonal stats, symmetry check, and
#     percentiles/mean/std for each matrix file found.
#   - It does NOT modify any data files.
#
# NOTES (Spyder)
#   - If you run this file via Spyder's Run button, it will execute at the end
#     of the script only if RUN_MATRIX_INSPECTOR is True.
# =============================================================================

RUN_MATRIX_INSPECTOR = False  # <-- switch to True when you want matrix summaries

if RUN_MATRIX_INSPECTOR:
    import os
    import sys
    import importlib

    # -------------------------------------------------------------------------
    # 1) Ensure Python can find inspect_matrices.py
    #    (Assumes inspect_matrices.py is in the same directory as THIS script.)
    #    If you keep inspect_matrices.py somewhere else, replace script_dir with
    #    that folder path.
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\modular_code")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # -------------------------------------------------------------------------
    # 2) Import (or reload) the helper module so Spyder picks up edits
    # -------------------------------------------------------------------------
    import inspect_matrices as im
    importlib.reload(im)

    # -------------------------------------------------------------------------
    # 3) Choose which folder to scan for .npy matrices
    #    Default: MATRIX_OUTPUT_DIR if defined earlier in this master flow.
    # -------------------------------------------------------------------------
    matrix_folder_dtilde_d_multi = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_WDM"
    matrix_folder_walk_drive = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_travel_matrices"
    # -------------------------------------------------------------------------
    # 4) Run the folder-level scan. These args match the helper’s function
    #    signature in inspect_matrices.py.
    #
    #    patterns: list of glob patterns to include. Common patterns:
    #      "*_d_*.npy", "*_dtilde_*.npy", "*_walk_*.npy", "*_drive_*.npy"
    #
    #    snippet_n: if >0, prints the top-left snippet_n x snippet_n submatrix
    #    save_hists: if True, saves histogram PNGs next to each .npy
    #    hist_bins: number of bins for histograms
    # -------------------------------------------------------------------------
    im.inspect_matrix_folder(
        matrix_folder=matrix_folder_dtilde_d_multi,
        patterns=["*_d_*.npy", "*_dtilde_*.npy", "*_walk_*.npy", "*_drive_*.npy","*_multi_*.npy"],
        snippet_n=5,
        save_hists=False,
        hist_bins=60
    )
    im.inspect_matrix_folder(
        matrix_folder=matrix_folder_walk_drive,
        patterns=["*_d_*.npy", "*_dtilde_*.npy", "*_walk_*.npy", "*_drive_*.npy"],
        snippet_n=0,
        save_hists=False,
        hist_bins=60
    )



# =============================================================================
# OPTIONAL HELPER: Scatter plots comparing travel-time matrices (Step 3 + Step 4)
# -----------------------------------------------------------------------------
# PURPOSE
#   Quickly visualize how different matrix types relate (entrywise OD pairs) for
#   each city:
#     - Step 3 raw matrices: <City><State>_walk_* and <City><State>_drive_*
#     - Step 4 derived matrices: d_* and dtilde_* (and optionally *_multi_*)
#
#   These plots are useful for:
#     - spotting unit issues (seconds vs minutes)
#     - verifying systematic shifts (e.g., parking penalty inflating drive times)
#     - seeing how d differs from dtilde
#
# HOW TO USE
#   - Set RUN_TIME_SCATTER_PLOTS = True
#   - Choose DISPLAY_UNIT (seconds/minutes/hours)
#   - Plots are saved to TIME_SCATTER_OUTPUT_DIR (default: TDA_OUTPUT_DIR/time_scatter)
#
# NOTES
#   - We exclude diagonal entries (i==j).
#   - We only plot OD pairs where both matrices have finite values.
#   - For very large matrices, we randomly subsample to MAX_SCATTER_POINTS points.
# =============================================================================

RUN_TIME_SCATTER_PLOTS = False  # <-- flip to True to generate scatter plots at end of run

# Units for displaying scatter plots (independent of TDA_OUTPUT_TIME_UNIT, but
# you will usually want these to match).
TIME_SCATTER_INPUT_UNIT  = TDA_INPUT_TIME_UNIT
TIME_SCATTER_OUTPUT_UNIT = TDA_OUTPUT_TIME_UNIT  # "seconds" | "minutes" | "hours"

# Maximum number of OD pairs to plot (subsampling keeps plots responsive)
MAX_SCATTER_POINTS = 50000
SCATTER_RANDOM_SEED = 123

# Which comparisons to generate (set any to False if you only want a subset)
PLOT_WALK_VS_DRIVE   = True
PLOT_D_VS_DTILDE     = True
PLOT_WALK_VS_DTILDE  = True
PLOT_DRIVE_VS_DTILDE = True

def _unit_to_seconds(u: str) -> float:
    return {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[u]

def _convert_time_vals(vals, from_unit: str, to_unit: str):
    """Convert numpy array/scalars from one unit to another (e.g., seconds -> minutes)."""
    import numpy as np
    v = np.asarray(vals, dtype=float)
    return v * (_unit_to_seconds(from_unit) / _unit_to_seconds(to_unit))

def _unit_label(u: str) -> str:
    return {"seconds": "sec", "minutes": "min", "hours": "hr"}.get(u, u)

def _latest_file(folder: str, pattern: str):
    """Return most-recently-modified file matching pattern in folder (or None)."""
    import os, glob
    matches = glob.glob(os.path.join(folder, pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))

def _aligned_pairs(A, B, max_points: int, seed: int):
    """Return x,y arrays for scatter where both matrices have finite entries (excluding diagonal)."""
    import numpy as np
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    r = min(A.shape[0], B.shape[0])
    c = min(A.shape[1], B.shape[1])
    A = A[:r, :c]
    B = B[:r, :c]

    mask = np.isfinite(A) & np.isfinite(B)
    if r == c:
        mask &= ~np.eye(r, dtype=bool)

    x = A[mask]
    y = B[mask]

    if x.size == 0:
        return None, None

    if x.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    return x, y

def _save_scatter_png(x, y, out_png, title, xlabel, ylabel):
    """Save a scatter plot with a y=x reference line."""
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.25)

    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    plt.plot([lo, hi], [lo, hi])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def generate_time_scatter_plots_for_city(city: str, state: str, raw_dir: str, wdm_dir: str, out_dir: str):
    """
    Generate scatter plots for one city, saving PNGs to out_dir.

    raw_dir: folder with Step 3 matrices (<City><State>_drive_* / <City><State>_walk_*)
    wdm_dir: folder with Step 4 matrices (d_* / dtilde_*)
    """
    import os
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    # Step 3 filenames are of the form:
    #   <City><State>_drive_<date-or-run-tag>.npy
    #   <City><State>_walk_<date-or-run-tag>.npy
    city_tag = f"{city}{state}"

    walk_fp  = _latest_file(raw_dir, f"{city_tag}_walk_*.npy")
    drive_fp = _latest_file(raw_dir, f"{city_tag}_drive_*.npy")

    # Step 4 naming varies depending on compute_t_matrices version; use robust patterns.
    d_fp      = _latest_file(wdm_dir, f"{city_tag}*_d_*.npy")
    dtilde_fp = _latest_file(wdm_dir, f"{city_tag}*_dtilde_*.npy")

    # Load arrays if present
    walk  = np.load(walk_fp)  if walk_fp  else None
    drive = np.load(drive_fp) if drive_fp else None
    dmat  = np.load(d_fp)     if d_fp     else None
    dtilde = np.load(dtilde_fp) if dtilde_fp else None

    out_unit_lbl = _unit_label(TIME_SCATTER_OUTPUT_UNIT)

    def _compare(A, B, a_name, b_name):
        x_raw, y_raw = _aligned_pairs(A, B, MAX_SCATTER_POINTS, SCATTER_RANDOM_SEED)
        if x_raw is None:
            print(f"   → [SKIP] No finite aligned pairs for {a_name} vs {b_name}")
            return

        x = _convert_time_vals(x_raw, TIME_SCATTER_INPUT_UNIT, TIME_SCATTER_OUTPUT_UNIT)
        y = _convert_time_vals(y_raw, TIME_SCATTER_INPUT_UNIT, TIME_SCATTER_OUTPUT_UNIT)

        out_png = os.path.join(out_dir, f"{city_tag}_scatter_{a_name}_vs_{b_name}_{TIME_SCATTER_OUTPUT_UNIT}.png")
        title = f"{city_tag}: {a_name} vs {b_name} ({out_unit_lbl})"
        xlabel = f"{a_name} time ({out_unit_lbl})"
        ylabel = f"{b_name} time ({out_unit_lbl})"
        _save_scatter_png(x, y, out_png, title, xlabel, ylabel)
        print(f"   ✓ wrote: {out_png}")

    print(f"\n=== Time scatter plots | {city}, {state} ===")
    if PLOT_WALK_VS_DRIVE and (walk is not None) and (drive is not None):
        _compare(walk, drive, "walk", "drive")
    if PLOT_D_VS_DTILDE and (dmat is not None) and (dtilde is not None):
        _compare(dmat, dtilde, "d", "dtilde")
    if PLOT_WALK_VS_DTILDE and (walk is not None) and (dtilde is not None):
        _compare(walk, dtilde, "walk", "dtilde")
    if PLOT_DRIVE_VS_DTILDE and (drive is not None) and (dtilde is not None):
        _compare(drive, dtilde, "drive", "dtilde")

if RUN_TIME_SCATTER_PLOTS:
    import os
    # Default output: a subfolder under TDA_OUTPUT_DIR
    TIME_SCATTER_OUTPUT_DIR = os.path.join(TDA_OUTPUT_DIR, "time_scatter")
    os.makedirs(TIME_SCATTER_OUTPUT_DIR, exist_ok=True)

    print("\n=== OPTIONAL: Generating time scatter plots (Step 3 vs Step 4) ===")
    print(f"Display unit: {TIME_SCATTER_OUTPUT_UNIT} (input assumed: {TIME_SCATTER_INPUT_UNIT})")
    for _city, _state in CITY_STATE_LIST:
        # Only attempt if we have at least the raw matrices or WDM matrices present
        generate_time_scatter_plots_for_city(
            city=_city,
            state=_state,
            raw_dir=RAW_TRAVEL_DIR,
            wdm_dir=WDM_OUTPUT_DIR,
            out_dir=TIME_SCATTER_OUTPUT_DIR
        )


# =============================================================================
# OPTIONAL HELPER: Per-matrix distribution plots (histograms) for Step 3 + Step 4
# -----------------------------------------------------------------------------
# PURPOSE
#   You can turn this on to generate a quick visual of the DISTRIBUTION of values
#   in each matrix (one plot per matrix), per city.
#
#   This is useful for:
#     - confirming whether values look like seconds vs minutes (order of magnitude)
#     - spotting extreme tails, clipping, or inf/NaN patterns
#     - comparing distributions across matrix types (walk, drive, d, dtilde)
#
# WHAT THIS DOES (IMPORTANT: individual matrices only)
#   - For each city, it loads the latest available matrices and saves HISTOGRAM
#     plots for each matrix independently (NO cross-matrix scatter comparisons).
#
# INPUT/OUTPUT UNITS
#   - Matrices are assumed to be stored in TIME_DIST_INPUT_UNIT (typically seconds).
#   - Values are converted for plotting into TIME_DIST_OUTPUT_UNIT, which you can set
#     to "seconds", "minutes", or "hours".
#
# PERFORMANCE NOTE
#   - Matrices can be large; this helper samples up to MAX_DIST_POINTS values
#     from the finite, off-diagonal entries to keep plotting fast.
# =============================================================================

RUN_MATRIX_DISTRIBUTION_PLOTS = False   # <-- flip to True to generate per-matrix histograms

# Units for display (you will usually want these to match the TDA unit settings)
TIME_DIST_INPUT_UNIT  = TDA_INPUT_TIME_UNIT
TIME_DIST_OUTPUT_UNIT = TDA_OUTPUT_TIME_UNIT   # "seconds" | "minutes" | "hours"

# Histogram settings
MAX_DIST_POINTS = 250000        # maximum number of entries to sample from each matrix
DIST_RANDOM_SEED = 123
DIST_BINS = 80                  # number of histogram bins
USE_LOG_X = False               # True = log-scale x-axis (helps when distributions are heavy-tailed)

# Which matrix families to include
INCLUDE_STEP3_RAW = True        # <City><State>_walk_*, <City><State>_drive_*
INCLUDE_STEP4_DERIVED = True    # *_d_*.npy, *_dtilde_*.npy (and optionally *_multi_* if present)

def _unit_to_seconds(u: str) -> float:
    return {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[u]

def _convert_time_vals(vals, from_unit: str, to_unit: str):
    """Convert numpy array/scalars from one unit to another (e.g., seconds -> minutes)."""
    import numpy as np
    v = np.asarray(vals, dtype=float)
    return v * (_unit_to_seconds(from_unit) / _unit_to_seconds(to_unit))

def _unit_label(u: str) -> str:
    return {"seconds": "sec", "minutes": "min", "hours": "hr"}.get(u, u)

def _latest_file(folder: str, pattern: str):
    """Return most-recently-modified file matching pattern in folder (or None)."""
    import os, glob
    matches = glob.glob(os.path.join(folder, pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: os.path.getmtime(p))

def _sample_matrix_values(A, max_points: int, seed: int):
    """
    Return a 1D sample of finite, off-diagonal values from matrix A.
    - Excludes diagonal entries (i==j) for square matrices.
    - Drops NaN/Inf.
    - Randomly subsamples if more than max_points.
    """
    import numpy as np
    A = np.asarray(A, dtype=float)

    mask = np.isfinite(A)
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        mask &= ~np.eye(A.shape[0], dtype=bool)

    vals = A[mask]
    if vals.size == 0:
        return None

    if vals.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(vals.size, size=max_points, replace=False)
        vals = vals[idx]

    return vals

def _plot_hist(vals, out_png, title, xlab, bins=80, log_x=False):
    """
    Save a histogram of vals to out_png.
    (Matplotlib default colors; no explicit styling.)
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.hist(vals, bins=bins)
    if log_x:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def generate_distribution_plots_for_city(city: str, state: str, raw_dir: str, wdm_dir: str, out_dir: str):
    """
    Generate per-matrix histogram plots for one city, saving PNGs to out_dir.

    raw_dir: folder with Step 3 matrices (<City><State>_drive_* / <City><State>_walk_*)
    wdm_dir: folder with Step 4 matrices (d_* / dtilde_*, and potentially *_multi_*)
    """
    import os
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    city_tag = f"{city}{state}"

    unit_lbl = _unit_label(TIME_DIST_OUTPUT_UNIT)

    # Collect candidate matrices (label -> filepath)
    candidates = {}

    if INCLUDE_STEP3_RAW:
        # Raw Step 3 matrices live in raw_dir and are named like:
        #   AlbuquerqueNM_drive_2025-12-30.npy
        #   AlbuquerqueNM_walk_2025-12-30.npy
        raw_walk_fp  = _latest_file(raw_dir, f"{city_tag}_walk_*.npy")
        raw_drive_fp = _latest_file(raw_dir, f"{city_tag}_drive_*.npy")

        if raw_walk_fp:
            candidates["walk_raw"] = raw_walk_fp
        if raw_drive_fp:
            candidates["drive_raw"] = raw_drive_fp

    if INCLUDE_STEP4_DERIVED:
        # Robust patterns: allow for travel tags / date strings in filenames
        d_fp         = _latest_file(wdm_dir, f"{city_tag}*_d_*.npy")
        dtilde_fp    = _latest_file(wdm_dir, f"{city_tag}*_dtilde_*.npy")
        d_multi_fp   = _latest_file(wdm_dir, f"{city_tag}*_d_multi_*.npy")
        dt_multi_fp  = _latest_file(wdm_dir, f"{city_tag}*_dtilde_multi_*.npy")

        if d_fp:
            candidates["d"] = d_fp
        if dtilde_fp:
            candidates["dtilde"] = dtilde_fp
        if d_multi_fp:
            candidates["d_multi"] = d_multi_fp
        if dt_multi_fp:
            candidates["dtilde_multi"] = dt_multi_fp

    if not candidates:
        print(f"\n=== Matrix distributions | {city}, {state} ===")
        print("   → [SKIP] No matching matrices found.")
        return

    print(f"\n=== Matrix distributions | {city}, {state} ===")
    print(f"   Input unit assumed: {TIME_DIST_INPUT_UNIT} | Display unit: {TIME_DIST_OUTPUT_UNIT}")

    for label, fp in candidates.items():
        try:
            A = np.load(fp, allow_pickle=False)
            finite_total = int(np.isfinite(A).sum())
            finite_offdiag = finite_total
            if A.ndim == 2 and A.shape[0] == A.shape[1]:
                finite_offdiag = int((np.isfinite(A) & ~np.eye(A.shape[0], dtype=bool)).sum())

            print(f"   {label}: shape={A.shape} finite_total={finite_total:,} finite_offdiag={finite_offdiag:,}")
        except Exception as e:
            print(f"   → [SKIP] Could not load {label}: {fp} ({e})")
            continue

        vals_raw = _sample_matrix_values(A, MAX_DIST_POINTS, DIST_RANDOM_SEED)
        if vals_raw is None:
            print(f"   → [SKIP] No finite off-diagonal values for {label}")
            continue

        # Convert to display units
        vals = _convert_time_vals(vals_raw, TIME_DIST_INPUT_UNIT, TIME_DIST_OUTPUT_UNIT)

        # ---------------------------------------------------------------------
        # Histogram output naming
        # ---------------------------------------------------------------------
        # IMPORTANT:
        #   Use the source matrix filename stem to preserve travel-type tags.
        #   Example:
        #     AustinTX_d_drive_2025-12-30.npy  ->
        #     AustinTX_d_drive_2025-12-30_hist_minutes.png
        #
        # This also works for any other naming conventions, because we’re not
        # trying to reconstruct the name—just reusing it.
        # ---------------------------------------------------------------------
        base_stem = os.path.splitext(os.path.basename(fp))[0]  # drop ".npy"
        out_png = os.path.join(out_dir, f"{base_stem}_hist_{TIME_DIST_OUTPUT_UNIT}.png")

        # Title/label also reflect the true source name (includes _drive_/_walk_)
        title = f"{base_stem}: distribution"
        xlab  = f"time ({unit_lbl})"

        _plot_hist(vals, out_png, title, xlab, bins=DIST_BINS, log_x=USE_LOG_X)
        print(f"   ✓ wrote: {out_png}")


if RUN_MATRIX_DISTRIBUTION_PLOTS:
    import os
    # Default output: a subfolder under TDA_OUTPUT_DIR
    MATRIX_DIST_OUTPUT_DIR = os.path.join(TDA_OUTPUT_DIR, "matrix_distributions")
    os.makedirs(MATRIX_DIST_OUTPUT_DIR, exist_ok=True)

    print("\n=== OPTIONAL: Generating per-matrix distribution plots ===")
    for _city, _state in CITY_STATE_LIST:
        generate_distribution_plots_for_city(
            city=_city,
            state=_state,
            raw_dir=RAW_TRAVEL_DIR,
            wdm_dir=WDM_OUTPUT_DIR,
            out_dir=MATRIX_DIST_OUTPUT_DIR
        )


