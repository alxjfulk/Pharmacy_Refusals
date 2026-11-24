# -*- coding: utf-8 -*-
"""Step 7 - compare each removal scenario's persistence diagram with the full baseline.

Step 7 does not rerun persistent homology. It reads the persistence diagrams that
Step 5 saved and measures how different each scenario's diagram is from the
baseline diagram (scenario 'full', with every pharmacy). This is done separately
for each selected matrix type (e.g. walk_d) and each homology dimension: H0
(separate groups of pharmacies, which join up as travel time grows) and H1
(loops, i.e. "holes" in coverage). Two distances are reported, both calculated
by the GUDHI library: the bottleneck distance and the Wasserstein-1 distance
(see HOW THE TWO DISTANCES WORK below).

What is compared with what:
- every scenario is compared only with 'full', never with another scenario;
- each random-removal replicate gets its own row;
- nothing is averaged across replicates; any summary is left to the analysis.

WHAT STEP 7 READS (from Step 5, in step05_tda/)
-----------------------------------------------
- scenario_plan/plan.json: the saved scenario plan. Step 7 uses its plan_hash,
  its selected_matrix_types and ph_settings["max_homology_dim"] (1 by default,
  so H0 and H1 are compared).
- scenario_plan/scenario_index.csv: one row per scenario (scenario_key, type,
  how many pharmacies were kept and removed, which ones, random seed, ...).
- <matrix_type>/<scenario_key>/persistence_diagram.npy: one row per class, with
  three columns: birth, death, homology_dimension (0 or 1). Birth and death are
  in Step 5's output unit (minutes by default). A class that never dies has
  death = +inf.
- <matrix_type>/<scenario_key>/result_metadata.json: must say processing_status
  "complete" and carry the same plan_hash as plan.json.

WHAT STEP 7 WRITES (in step07_comparisons/)
-------------------------------------------
- <matrix_type>/scenario_to_baseline_comparisons.csv: one row per scenario other
  than 'full'. Each row first describes the scenario (scenario_key, type,
  replicate number, how many and which pharmacies were removed, random seed,
  and the paths of the two diagrams compared). Then it has four columns for
  each dimension, shown here for H0 (the H1 columns are the same with H1_):
      H0_baseline_point_count    number of H0 rows in the baseline diagram
      H0_scenario_point_count    number of H0 rows in the scenario diagram
      H0_bottleneck_distance     in the diagrams' unit (minutes by default)
      H0_wasserstein_1_distance  in the diagrams' unit (minutes by default)
- <matrix_type>/comparison_metadata.json: the plan hash, a SHA-256 checksum of
  every diagram that was used, and the number of rows. The next run uses it to
  decide whether the table can be reused.
- step07_metadata.json: one summary for the city, listing the metadata of every
  matrix type.

HOW THE PIECES FIT TOGETHER (compare_city_scenarios, at the bottom)
-------------------------------------------------------------------
1. Read the saved Step 5 plan and scenario index, and check them: same
   ANALYSIS_RUN_TAG, and exactly one baseline, named 'full'.
2. For each selected matrix type, load the baseline
   diagram; it must be complete and from this plan .... _require_complete_step05_result,
                                                        _load_diagram
3. Take a checksum of the baseline diagram and of
   every scenario diagram (each must be complete) ..... file_sha256
4. Skip this matrix type if an earlier Step 7 table was made from the same
   plan and the same diagrams; stop if anything else is already there.
5. For each scenario and each dimension, take that
   dimension's points from both diagrams .............. _dimension_points
   and calculate the two distances .................... default_distance_runner
6. Save the table, then its metadata file.
7. After every matrix type, write the city summary, step07_metadata.json.

HOW THE TWO DISTANCES WORK
--------------------------
A persistence diagram is a set of points (birth, death), one per class. To
compare two diagrams, GUDHI pairs up their points one to one, and each pair has
a cost:

- A point of one diagram can be matched with a point of the other. The cost is
  the larger of the two differences (GUDHI's ''internal_p'' = inf):
      cost = max(|birth_1 - birth_2|, |death_1 - death_2|)
- A point can instead be matched to the diagonal, the line birth = death, where
  a class with zero lifetime would sit. This is how a class with no partner in
  the other diagram is counted. The cost is half the class's lifetime, which is
  how far the point is from the nearest point on the diagonal:
      cost = (death - birth) / 2

Of all possible matchings, each distance uses the best one for itself:
- bottleneck distance = the largest single cost, in the matching that makes
  this largest cost as small as possible. It reports the one biggest change.
- Wasserstein-1 distance = the total of all the costs, in the matching that
  makes this total as small as possible (''order'' = 1: the costs are simply
  added). It adds up every change, large and small.
Both are 0 when the two diagrams have exactly the same points.

WORKED EXAMPLE 1 (H1, minutes)
    baseline:  A = (10, 14), B = (20, 21)        scenario:  A' = (10, 16)
    best matching: A with A'      cost max(|10 - 10|, |14 - 16|) = 2
                   B to diagonal  cost (21 - 20) / 2             = 0.5
    bottleneck = max(2, 0.5) = 2 min        Wasserstein-1 = 2 + 0.5 = 2.5 min
    (Sending A and A' both to the diagonal instead would cost 2 and 3: worse.)

WORKED EXAMPLE 2 (H0, minutes). Every H0 class is born at 0, so a point's cost
to the diagonal is half its death.
    baseline deaths: 4, 6, 10        scenario (one pharmacy removed): 6, 12
    best matching: 6 with 6 (cost 0), 10 with 12 (cost 2), 4 to diagonal (cost 2)
    bottleneck = 2 min        Wasserstein-1 = 0 + 2 + 2 = 4 min
    The scenario has one point fewer, so one baseline point has to go to the
    diagonal whichever pharmacy was removed (see Known issue (g) below).

CLASSES THAT NEVER DIE (''keep_essential_parts'' = True)
--------------------------------------------------------
A class with death = +inf can only be matched with another such class of the
same dimension, at a cost equal to the difference in birth (0 for H0, where
every class is born at 0). If the two diagrams have different numbers of them,
no matching is possible and both distances are +inf. GUDHI then prints the
warning "Cardinality of essential parts differs" for the Wasserstein-1
calculation; the bottleneck is +inf without a warning. Normally there are no
such classes: Giotto drops the one H0 class that never dies, and every loop is
filled in once all edges are present. They appear when a pharmacy can never be
reached (Known issue (a) below).

UNITS
-----
The distances are in the same unit as the Step 5 diagrams: minutes by default
(TDA_OUTPUT_TIME_UNIT in Step 0). They are also on the same scale. In Giotto's
filtration an edge between two pharmacies appears at their travel time d[i, j],
while Hickok et al. grow balls of radius t, so the edge appears at t = d/2.
Every value here is therefore twice the value on the paper's t scale: a
bottleneck distance of 3 minutes here is 1.5 minutes in the paper's terms.

SKIP AND STOP RULES (for each matrix type)
------------------------------------------
- Every Step 5 result in the plan must be complete. If one is missing or
  incomplete, Step 7 stops before writing a table.
- An existing table is reused (printed as [SKIP]) only if its metadata says
  "complete", its plan hash matches plan.json, and the checksum of every source
  diagram (the baseline and every scenario) is still the same.
- If only one of the table and its metadata exists (an earlier run stopped
  partway), or both exist but something above differs (for example, a Step 5
  result was made again), Step 7 stops and leaves the files as they are.
  Nothing is overwritten or deleted.
- step07_metadata.json is only a summary and is rewritten on every Step 7 run,
  even when every matrix type was skipped.

Step 0 runs Step 7 for every city, one city after another, once Step 6 has run
for every city.

NOTES AND KNOWN ISSUES
----------------------
- Known issue (a): Step 3 can leave a pharmacy unreachable by car in one
  direction, so its whole row and column of drive_d and multimodal_d are NaN.
  Step 5 turns NaN into +inf, so the pharmacy never connects to the others: it
  is an extra H0 class that never dies in every diagram that keeps it (and it
  drops out of H1). A scenario that removes it has one fewer such class than the
  baseline, so both of its H0 distances are +inf; its H1 distances stay finite.
  This is why, in test_261011, Austin's random-removal replicate_003 has H0
  distances of +inf for drive_d and multimodal_d: that replicate removed the
  pharmacy by chance. A fix in Step 3 has been proposed but not applied.
- Known issue (g): removing k pharmacies normally leaves the scenario with k
  fewer H0 points, so at least k baseline H0 points must be matched to the
  diagonal, each costing half its death. This sets a floor that depends only
  on k: the H0 bottleneck is at least half the k-th smallest baseline H0 death,
  and the H0 Wasserstein-1 is at least half the sum of the k smallest. So H0
  distances partly reflect how many pharmacies were removed, not which ones.
  To judge a targeted removal (e.g. a chain) by H0, compare it with random
  removals of the same size. H1 is largely unaffected.
- If a scenario's diagram has no H1 rows (no loops, or Step 5 limited the
  dimensions because very few pharmacies were left), every baseline H1 point is
  matched to the diagonal.
- GUDHI is loaded only if it is installed (see the top of the code), so the
  validation script can test Step 7's file handling with a stand-in distance
  function. A real Step 7 run always needs GUDHI and POT (Python Optimal
  Transport, which GUDHI's Wasserstein function uses).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from pipeline_common import (
    ANALYSIS_TAG_COLUMN,
    CityRunPaths,
    file_sha256,
    normalize_matrix_type,
    read_json,
    utc_now_iso,
    validate_analysis_tag,
    write_json,
)


# =============================================================================
# GUDHI, the library that calculates the distances
# =============================================================================

# Try to load GUDHI's two distance functions. If that fails, keep the error
# instead of stopping, so this file can still be imported (for example by the
# validation script, which uses a stand-in distance function). _require_gudhi
# reports the saved error when a real distance is first needed.
try:  # GUDHI (and POT, for the Wasserstein function) may not be installed everywhere
    from gudhi import bottleneck_distance as _gudhi_bottleneck_distance
    from gudhi.wasserstein import wasserstein_distance as _gudhi_wasserstein_distance
    _GUDHI_IMPORT_ERROR = None
except Exception as exc:  # reached only if GUDHI or POT is missing or fails to load
    _gudhi_bottleneck_distance = None
    _gudhi_wasserstein_distance = None
    _GUDHI_IMPORT_ERROR = exc


def _require_gudhi() -> None:
    """Stop with a clear message if GUDHI (or POT) could not be loaded.

    The import at the top of this file does not fail when GUDHI is missing; it
    saves the error instead. This check runs just before the first distance is
    calculated, so Step 7 stops there, with the original import error in the
    message. A stand-in distance function (as in the validation script) never
    calls it.
    """
    if _GUDHI_IMPORT_ERROR is not None:
        raise ImportError(
            "Step 7 requires GUDHI and POT for bottleneck/Wasserstein distances. "
            f"Original import error: {_GUDHI_IMPORT_ERROR}"
        )


# =============================================================================
# Reading a Step 5 persistence diagram
# =============================================================================

def _load_diagram(path: Path) -> np.ndarray:
    """Read one Step 5 persistence_diagram.npy and check its shape.

    Returns an array with one row per class and three columns: birth, death,
    homology_dimension. Birth and death are in Step 5's output unit (minutes by
    default); death is +inf for a class that never dies.

    ''allow_pickle=False'' means only plain numbers are read, never stored
    Python objects. Stops if the array does not have exactly three columns.
    """
    diagram = np.load(path, allow_pickle=False)
    diagram = np.asarray(diagram, dtype=float)
    if diagram.ndim != 2 or diagram.shape[1] != 3:
        raise ValueError(f"Persistence diagram must have shape (n, 3): {path}")
    return diagram


def _dimension_points(diagram: np.ndarray, dimension: int) -> np.ndarray:
    """Return the (birth, death) points of one homology dimension, as an (n, 2) array.

    The distances compare one dimension at a time, so the H0 rows and the H1
    rows of a Step 5 diagram are separated first: matching an H0 class (a group
    of pharmacies) with an H1 class (a loop) would mean nothing. The third
    column (homology_dimension, stored as 0.0, 1.0, ...) is compared as a whole
    number.

    Example: rows (0, 4, 0), (0, 6, 0) and (10, 14, 1) give
        dimension 0 -> [[0, 4], [0, 6]]        dimension 1 -> [[10, 14]]

    If the diagram has no rows of that dimension (or no rows at all), the result
    is an empty (0, 2) array, and every point of the other diagram is then
    matched to the diagonal. A copy is returned, so the full diagram is never
    changed.
    """
    if diagram.size == 0:
        return np.empty((0, 2), dtype=float)
    mask = diagram[:, 2].astype(int) == int(dimension)
    return diagram[mask, :2].astype(float, copy=True)


# =============================================================================
# The two distances: bottleneck and Wasserstein-1
# =============================================================================

def default_distance_runner(
    baseline_points: np.ndarray,
    scenario_points: np.ndarray,
) -> tuple[float, float]:
    """Return (bottleneck, Wasserstein-1) between two diagrams of the same dimension.

    Each input is an (n, 2) array of (birth, death) points for ONE homology
    dimension, from _dimension_points. Both results are in the diagrams' unit
    (minutes by default) and can be +inf (see below). Swapping the two diagrams
    gives the same values. How the distances work, with worked examples, is
    explained at the top of this file.

    GUDHI settings, and why:
    - bottleneck_distance(baseline, scenario): with no third argument GUDHI
      calculates the exact value. Its cost between two points is always
      max(|difference in birth|, |difference in death|).
    - wasserstein_distance(..., ''order''=1.0): the costs are simply added up.
      (With order 2 it would be the square root of the sum of squared costs.)
    - ''internal_p''=np.inf: the same cost between two points as the
      bottleneck, so the two distances measure change in the same way and
      differ only in "largest cost" versus "total cost".
    - ''keep_essential_parts''=True: classes that never die (death = +inf) are
      matched only with each other. If the two diagrams have different numbers
      of them, the result is +inf and GUDHI prints "Cardinality of essential
      parts differs". The bottleneck function handles them the same way (+inf,
      without the warning).

    Step 0 always uses this function. compare_city_scenarios accepts any
    function with the same inputs and outputs, so the validation script can
    pass a simple stand-in and test Step 7 without GUDHI.
    """
    _require_gudhi()
    bottleneck = float(
        _gudhi_bottleneck_distance(baseline_points, scenario_points)
    )
    wasserstein = float(
        _gudhi_wasserstein_distance(
            baseline_points,
            scenario_points,
            order=1.0,
            internal_p=np.inf,
            keep_essential_parts=True,
        )
    )
    return bottleneck, wasserstein


# =============================================================================
# Checking that a Step 5 result is complete
# =============================================================================

def _require_complete_step05_result(result_dir: Path) -> dict[str, Any]:
    """Check that one Step 5 result folder is complete, and return its metadata.

    A result counts as complete when both persistence_diagram.npy and
    result_metadata.json exist and the metadata says ''processing_status'' is
    "complete". Step 5 writes result_metadata.json last, so "complete" there
    means the diagram was saved in full. Otherwise this stops with an error:
    Step 7 never compares a result that may be partial.

    This does not check the plan hash. compare_city_scenarios compares the
    returned metadata's ''plan_hash'' with plan.json itself.
    """
    diagram_path = result_dir / "persistence_diagram.npy"
    metadata_path = result_dir / "result_metadata.json"
    if not diagram_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Step 7 requires a complete Step 5 result: {result_dir}"
        )
    metadata = read_json(metadata_path)
    if metadata.get("processing_status") != "complete":
        raise RuntimeError(f"Step 5 result is not complete: {result_dir}")
    return metadata


# =============================================================================
# Step 7 for one city: runs the pieces above in order
# =============================================================================

def compare_city_scenarios(
    paths: CityRunPaths,
    *,
    distance_runner: Callable[[np.ndarray, np.ndarray], tuple[float, float]] = default_distance_runner,
) -> dict[str, Any]:
    """Compare every scenario in the saved Step 5 plan with 'full', for one city.

    Step 0 calls this once per city. The numbered comments below match the
    numbered list at the top of this file.

    For each selected matrix type it writes scenario_to_baseline_comparisons.csv
    (one row per scenario other than 'full') and comparison_metadata.json. At
    the end it writes the city summary, step07_metadata.json, and returns its
    contents.

    An existing table is reused only when its metadata says "complete" and its
    plan hash and every source diagram checksum still match. Partial or
    mismatched outputs stop the run (see SKIP AND STOP RULES at the top of this
    file).

    ''distance_runner'' is the function that turns two (n, 2) diagrams into
    (bottleneck, Wasserstein-1). Step 0 leaves the default, which uses GUDHI;
    the validation script passes a stand-in.
    """
    # 1. Read the saved Step 5 plan and scenario index, and check them.
    if not paths.scenario_plan_json.exists() or not paths.scenario_index_csv.exists():
        raise FileNotFoundError(
            f"Step 7 requires the locked Step 5 scenario plan: {paths.scenario_plan_dir}"
        )
    plan = read_json(paths.scenario_plan_json)
    # The plan must belong to this run (ANALYSIS_RUN_TAG).
    if plan.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
        raise ValueError("Step 5 plan belongs to a different analysis_run_tag.")
    scenario_index = pd.read_csv(paths.scenario_index_csv)
    if scenario_index.empty:
        raise ValueError("Step 5 scenario index is empty.")
    if "scenario_key" not in scenario_index.columns:
        raise ValueError("Step 5 scenario index lacks scenario_key.")
    # Step 5 adds the baseline to every plan by itself; every comparison is made
    # against it, so there must be exactly one, with the key 'full'.
    baseline_rows = scenario_index[scenario_index["scenario_type"] == "baseline"]
    if len(baseline_rows) != 1 or str(baseline_rows.iloc[0]["scenario_key"]) != "full":
        raise ValueError("Step 7 requires exactly one automatic baseline with scenario_key='full'.")

    # Every Step 5 result must carry this plan hash. max_homology_dim is the
    # value requested in Step 0 (1 by default), so dimensions 0 and 1 are compared.
    plan_hash = str(plan["plan_hash"])
    max_dimension = int(plan["ph_settings"]["max_homology_dim"])
    # The metadata of each matrix type, collected for step07_metadata.json.
    summaries = []

    # Steps 2 to 6 run once for each matrix type that Step 5 analyzed.
    for raw_matrix_type in plan["selected_matrix_types"]:
        matrix_type = normalize_matrix_type(raw_matrix_type)
        matrix_root = paths.tda_matrix_dir(matrix_type)
        # 2. Load the baseline diagram. It must be complete and made from this plan.
        baseline_dir = matrix_root / "full"
        baseline_metadata = _require_complete_step05_result(baseline_dir)
        if baseline_metadata.get("plan_hash") != plan_hash:
            raise ValueError(f"Baseline result does not match the saved plan: {baseline_dir}")
        baseline_diagram_path = baseline_dir / "persistence_diagram.npy"
        baseline_diagram = _load_diagram(baseline_diagram_path)
        baseline_hash = file_sha256(baseline_diagram_path)

        # Where this matrix type's table and metadata go:
        # step07_comparisons/<matrix_type>/.
        comparison_dir = paths.comparison_matrix_dir(matrix_type)
        comparison_dir.mkdir(parents=True, exist_ok=True)
        comparison_csv = comparison_dir / "scenario_to_baseline_comparisons.csv"
        comparison_metadata_path = comparison_dir / "comparison_metadata.json"

        # 3. A checksum of every diagram this table depends on: the baseline (taken
        #    just above) and every scenario. These go into comparison_metadata.json,
        #    so a later run can tell whether any diagram has changed since. Every
        #    scenario result must be complete before anything is compared.
        expected_source_hashes = {"full": baseline_hash}
        for _, row in scenario_index.iterrows():
            key = str(row["scenario_key"])
            if key == "full":
                continue
            # Random-removal and leave-one-out keys contain a slash (e.g.
            # remove_10_percent_random/replicate_003), so those results sit one
            # folder deeper.
            result_dir = matrix_root / key
            _require_complete_step05_result(result_dir)
            expected_source_hashes[key] = file_sha256(result_dir / "persistence_diagram.npy")

        # 4. Is this matrix type already done?
        #    Only one of the two files exists: an earlier run stopped partway. Stop.
        if comparison_csv.exists() != comparison_metadata_path.exists():
            raise RuntimeError(
                f"Step 7 has partial outputs in {comparison_dir}; both the comparison "
                "table and metadata must exist together."
            )
        #    Both exist: reuse them only if they were made from the same plan and
        #    exactly the same diagrams. Otherwise stop rather than overwrite them.
        if comparison_csv.exists() and comparison_metadata_path.exists():
            existing_metadata = read_json(comparison_metadata_path)
            if (
                existing_metadata.get("processing_status") == "complete"
                and existing_metadata.get("plan_hash") == plan_hash
                and existing_metadata.get("source_diagram_hashes") == expected_source_hashes
            ):
                print(f"[SKIP] Step 7 complete: {paths.city_state} | {matrix_type}")
                summaries.append(existing_metadata)
                continue  # on to the next matrix type
            raise RuntimeError(
                f"Existing Step 7 output conflicts with current Step 5 results: {comparison_dir}. "
                "Use a new ANALYSIS_RUN_TAG rather than mixing comparison plans."
            )

        # 5. One row per scenario: first what the scenario is, then the two
        #    distances for each dimension.
        rows: list[dict[str, Any]] = []
        for _, scenario_row in scenario_index.iterrows():
            scenario_key = str(scenario_row["scenario_key"])
            if scenario_key == "full":
                continue  # the baseline is not compared with itself
            scenario_dir = matrix_root / scenario_key
            result_metadata = _require_complete_step05_result(scenario_dir)
            # The scenario must come from the same plan as the baseline.
            if result_metadata.get("plan_hash") != plan_hash:
                raise ValueError(f"Scenario result does not match the saved plan: {scenario_dir}")
            scenario_diagram_path = scenario_dir / "persistence_diagram.npy"
            scenario_diagram = _load_diagram(scenario_diagram_path)

            row: dict[str, Any] = {
                # Which run, city and matrix type.
                ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
                "city": paths.city,
                "state": paths.state.upper(),
                "city_state": paths.city_state,
                "matrix_type": matrix_type,
                "baseline_scenario_key": "full",
                # The scenario, copied from scenario_index.csv (included_count and
                # excluded_count are renamed). parent_scenario_name is filled for
                # random-removal and leave-one-out scenarios; replicate_number and
                # random_seed for random removals only. Otherwise they are blank.
                "scenario_name": scenario_row.get("scenario_name"),
                "scenario_key": scenario_key,
                "scenario_type": scenario_row.get("scenario_type"),
                "scenario_scope": scenario_row.get("scenario_scope"),
                "parent_scenario_name": scenario_row.get("parent_scenario_name"),
                "replicate_number": scenario_row.get("replicate_number"),
                "included_pharmacy_count": scenario_row.get("included_count"),
                "excluded_pharmacy_count": scenario_row.get("excluded_count"),
                # The removed pharmacy IDs, joined with "|".
                "excluded_pharmacy_ids": scenario_row.get("excluded_pharmacy_ids"),
                "random_seed": scenario_row.get("random_seed"),
                # Full paths of the two diagrams compared, so any row can be traced.
                "baseline_diagram": str(baseline_diagram_path.resolve()),
                "scenario_diagram": str(scenario_diagram_path.resolve()),
            }
            # H0 first, then H1 (up to max_homology_dim). Each dimension is compared
            # on its own (see _dimension_points and default_distance_runner).
            for dimension in range(max_dimension + 1):
                baseline_points = _dimension_points(baseline_diagram, dimension)
                scenario_points = _dimension_points(scenario_diagram, dimension)
                bottleneck, wasserstein = distance_runner(
                    baseline_points, scenario_points
                )
                # The point counts help when reading the distances: removing k
                # pharmacies normally leaves k fewer H0 points, and that alone
                # raises the H0 distances (Known issue (g), top of this file).
                row[f"H{dimension}_baseline_point_count"] = len(baseline_points)
                row[f"H{dimension}_scenario_point_count"] = len(scenario_points)
                row[f"H{dimension}_bottleneck_distance"] = bottleneck
                row[f"H{dimension}_wasserstein_1_distance"] = wasserstein
            rows.append(row)

        # 6. Save the table, then its metadata. The columns are listed here so their
        #    order is fixed and the header is complete even when there are no rows.
        comparison_columns = [
            ANALYSIS_TAG_COLUMN, "city", "state", "city_state", "matrix_type",
            "baseline_scenario_key", "scenario_name", "scenario_key",
            "scenario_type", "scenario_scope", "parent_scenario_name",
            "replicate_number", "included_pharmacy_count",
            "excluded_pharmacy_count", "excluded_pharmacy_ids", "random_seed",
            "baseline_diagram", "scenario_diagram",
        ]
        # Four columns for each dimension, in the same order as in each row above.
        for dimension in range(max_dimension + 1):
            comparison_columns.extend(
                [
                    f"H{dimension}_baseline_point_count",
                    f"H{dimension}_scenario_point_count",
                    f"H{dimension}_bottleneck_distance",
                    f"H{dimension}_wasserstein_1_distance",
                ]
            )
        result_table = pd.DataFrame(rows, columns=comparison_columns)
        # Check that every row carries this run's tag. validate_analysis_tag would
        # reject a table with no rows (no tags at all), so it is checked only when
        # there are rows.
        if not result_table.empty:
            validate_analysis_tag(result_table, paths.analysis_run_tag, str(comparison_csv))
        else:
            # The plan has only 'full', so there is nothing to compare. A table with
            # just the header is still written, so this matrix type counts as done.
            print(f"[INFO] Step 7: {paths.city_state} | {matrix_type} has only the full baseline; no comparisons were generated.")
        result_table.to_csv(comparison_csv, index=False)
        # The metadata is written last. If the run stops between the two writes,
        # the next run finds a table without metadata and stops (step 4 above).
        metadata = {
            "pipeline_step": 7,
            ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
            "city": paths.city,
            "state": paths.state.upper(),
            "matrix_type": matrix_type,
            "processing_status": "complete",
            "completed_utc": utc_now_iso(),
            "plan_hash": plan_hash,
            "comparison_count": len(result_table),
            "comparison_scope": "each non-baseline scenario versus full only",
            # Checksums of every diagram used; the next run compares them (step 4).
            "source_diagram_hashes": expected_source_hashes,
            "comparison_table": str(comparison_csv.resolve()),
        }
        write_json(comparison_metadata_path, metadata)
        summaries.append(metadata)

    # 7. The city summary: the metadata of every matrix type, whether made now or
    #    reused. This file is rewritten on every run.
    overall = {
        "pipeline_step": 7,
        ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
        "city": paths.city,
        "state": paths.state.upper(),
        "completed_utc": utc_now_iso(),
        "matrix_results": summaries,
    }
    write_json(paths.step07_dir / "step07_metadata.json", overall)
    return overall
