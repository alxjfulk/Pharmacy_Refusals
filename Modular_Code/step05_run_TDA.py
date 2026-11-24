# -*- coding: utf-8 -*-
"""Step 5 - lock a removal-scenario plan for each city, then run persistent homology.

Step 5 asks what the "shape" of pharmacy access looks like with every pharmacy
(the baseline) and with some pharmacies removed (the scenarios). It has two halves:

1. The scenario plan. For each city, the scenario settings in Step 0 are turned
   into concrete lists: for every scenario, which pharmacies are kept and which
   are removed. The plans for all cities are checked and saved before any
   persistent homology starts, and from then on each plan is locked (see below).
2. Persistent homology (PH). For every selected Step 4 matrix and every scenario
   in the saved plan, Step 5 keeps only the rows and columns of the kept
   pharmacies, runs Giotto-TDA's Vietoris-Rips persistence on that smaller matrix,
   and saves the persistence diagram, plots and summary statistics.

The saved plan links the two halves: the PH half never works out the scenarios
again; it reads which pharmacies to keep from the saved membership files. Step 6
(scenario maps) and Step 7 (comparison with the baseline) read the same files, so
every step uses exactly the same scenario membership.

WHAT STEP 5 READS
-----------------
- step03_raw_travel/manifest.csv: one row per pharmacy. Its matrix_index order is
  the row/column order of every matrix; its OrganizationName column is used for
  chain matching.
- step04_derived_matrices/<matrix_type>/matrix.npy for each selected matrix type,
  in seconds. Step 0's ANALYSIS_TRAVEL_MODES x ANALYSIS_MATRIX_FORMS choose them;
  by default walk_d, drive_d and multimodal_d.
- Step 0 settings: GLOBAL_SCENARIOS (used for every city), CITY_SCENARIOS (used for
  one city), TDA_INPUT_TIME_UNIT ("seconds"), TDA_OUTPUT_TIME_UNIT ("minutes") and
  TDA_MAX_HOMOLOGY_DIM (1, so H0 and H1).

WHAT STEP 5 WRITES (in <CityState>/step05_tda/)
-----------------------------------------------
    scenario_plan/
        plan.json            the locked plan: settings, scenario definitions, one
                             entry per scenario, and the plan's fingerprint (plan_hash)
        scenario_index.csv   one row per scenario: key, type, number kept and
                             removed, removed pharmacy IDs (joined by "|"), and a
                             fingerprint of its membership (membership_hash)
        memberships/<scenario_key>.csv
                             one row per manifest pharmacy: the manifest columns
                             plus scenario columns; included = True if kept
    <matrix_type>/<scenario_key>/      e.g. walk_d/full/ or
                                       drive_d/remove_10_percent_random/replicate_002/
        persistence_diagram.npy    one row per class: birth, death, homology dimension
        persistence_diagram.csv    the same, for opening in Excel
        birth_death.npz            births and deaths split by dimension
        persistence_diagram.png    the diagram as a scatter plot
        persistence_landscape.png  the persistence landscape (or, if it could not
                                   be drawn, persistence_landscape_error.txt)
        summary.json               statistics for each dimension, plus warnings
        result_metadata.json       where the result came from; written last, so
                                   it marks the folder as complete
    step05_execution_summary.json  how many results were run, skipped and failed
                                   (latest run only)

Births and deaths are in TDA_OUTPUT_TIME_UNIT (minutes by default).

SCENARIO TYPES (the 'type' of each definition in Step 0)
--------------------------------------------------------
- full: the baseline, keeps every pharmacy. Added automatically; never listed.
- exclude_ids (CITY_SCENARIOS only): removes the listed pharmacy_ids, which must be
  copied from this run's Step 3 manifest. Key: the scenario name.
- chain_exclusion: removes every pharmacy whose OrganizationName contains one of
  the organization_name_terms, ignoring capital letters. Key: the name.
- random_exclusion: removes a share ('proportion') of the city's pharmacies at
  random, 'replicates' times, starting from a fixed 'seed'. No two replicates
  remove the same set. Keys: <name>/replicate_001, <name>/replicate_002, ...
  Random removals are a reference for targeted removals of the same size (see
  known issue (g) in Step 7).
- leave_one_out (GLOBAL_SCENARIOS only): one scenario per pharmacy, each removing
  only that pharmacy. Keys: <name>/<pharmacy_id>.
These other spellings are also accepted: explicit_exclusion, remove_chain,
random_removal and loo.

HOW THE PIECES FIT TOGETHER
---------------------------
Step 0 runs build_scenario_plan_draft and validate_or_accept_existing_plan for
every city, then save_scenario_plan for each city that has no saved plan yet, and
only then execute_saved_scenario_plan, city by city. So a mistake in any city's
scenarios (a chain that matches nothing, an unknown ID, too many replicates) stops
Step 5 before anything is saved or computed.

Part 1 - the plan (1-5 in build_scenario_plan_draft; Step 0 then calls 6 and 7)
1. Read and check the manifest; check that the
   selected Step 4 matrices exist.
2. Merge the global and city definitions ........... merge_scenario_definitions
3. Expand each definition into concrete
   scenarios, the baseline first ................... expand_scenario_definitions
4. Check every scenario ............................ validate_expanded_scenario
5. Fingerprint the plan (plan_hash); build the
   scenario index and membership tables ............ membership_dataframe
6. Compare with the saved plan, if any ............. validate_or_accept_existing_plan
7. Save the plan, once every city passed 1-6 ....... save_scenario_plan

Part 2 - persistent homology (execute_saved_scenario_plan, one city at a time)
8. Read the saved plan, scenario index and manifest.
9. For each selected matrix and each scenario, read
   the membership file; skip the pair if complete .. _result_complete
10. Keep only the kept pharmacies; run PH on them .. process_single_matrix
    a. clean the matrix (NaN, symmetry, negatives) . _prepare_matrix_for_ph
    b. Giotto's Vietoris-Rips persistence
    c. convert seconds to minutes .................. time_conversion_factor
    d. save the diagram, plots and statistics ...... _dimension_statistics
11. Save step05_execution_summary.json; stop with
    an error if any pair failed.

WHY THE PLAN IS LOCKED, AND HOW RESUMING WORKS
----------------------------------------------
For reproducibility, every later run with the same ANALYSIS_RUN_TAG must use
exactly the same scenario membership. Drawing the random replicates again, or
matching chain names again after a settings change, would quietly mix results
from two different plans. So the first saved plan is locked:
- plan_hash is a fingerprint (SHA-256) of everything the plan depends on: the
  manifest, the selected matrix types, the PH settings, the enabled scenario
  definitions and the resulting memberships. Every run builds the plan again from
  the current settings and compares its hash with the saved one.
- Same hash: the saved plan is used and PH carries on where it stopped. A
  matrix/scenario result that is complete (all files present, and
  result_metadata.json records the same plan, matrix and membership) is skipped.
  Missing or incomplete results are computed again, in the same folder (unlike
  Steps 1-4, which stop when they find partial output).
- Different hash (a scenario added, removed or edited, a matrix type added, a
  unit changed, ...): Step 5 stops. Go back to the original settings, or start a
  new run with a new ANALYSIS_RUN_TAG.
Disabled scenarios are not part of the plan, so editing one does not change the
hash; enabling one does.

WORKED EXAMPLES
---------------
- Removal count (half_up_removal_count): 25 pharmacies x 0.10 = 2.5 -> 3 removed.
  Python's round(2.5) gives 2, so .5 is rounded up explicitly. 24 x 0.10 = 2.4 -> 2.
- Unit conversion (time_conversion_factor): a death at 1359 s is saved as
  1359 / 60 = 22.65 minutes.
- Giotto vs. Hickok et al.: a loop that dies at 30 minutes here dies at
  t = 15 minutes in the paper's terms (see below).

PERSISTENT HOMOLOGY, AND HOW IT RELATES TO HICKOK ET AL.
--------------------------------------------------------
Giotto's VietorisRipsPersistence is given the matrix itself (metric="precomputed").
As the filtration value grows, the edge between pharmacies i and j appears when
the value reaches d[i, j], and a triangle fills in once all three of its edges are
there.
- H0 classes (separate groups of pharmacies): every class is born at 0 and dies
  when its group joins another, so the H0 deaths are the edge lengths of a minimum
  spanning tree. Giotto uses reduced homology: the one group that never dies (the
  whole city, once everything is connected) is not reported.
- H1 classes (loops: "holes" in coverage): born when a ring of edges closes,
  dead when triangles fill it in.
- Classes of zero length (birth = death) are not reported.
Hickok et al. grow a ball of radius t around each site, and two balls touch at
t = d / 2, so Giotto's births and deaths are twice the paper's t. Unlike the paper,
there are no vertex weights here (no waiting time at each site).

KNOWN ISSUES
------------
- (h) Every analyzed matrix is cleaned before PH (_prepare_matrix_for_ph): NaN
  becomes +inf (the pair is never joined), the matrix is made symmetric with
  min(M, M.T), and negatives become 0. The warnings are only recorded in the
  output files (summary.json and result_metadata.json), never printed.
- (a) Step 3 can leave a pharmacy unreachable by car in one direction, so its drive
  and multimodal values are NaN (4 pharmacies in test_261011: Atlanta 2,
  Albuquerque 1, Austin 1). Here that pharmacy never joins the others: it adds an
  H0 class that never dies, and it drops out of H1. The H0 persistence landscape
  then cannot be computed (numpy/gtda RuntimeWarnings). A fix in Step 3 has been
  proposed but not applied.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common_260801 import (
    ANALYSIS_TAG_COLUMN,
    CityRunPaths,
    canonical_json,
    file_sha256,
    json_safe,
    normalize_matrix_type,
    object_sha256,
    ordered_id_checksum,
    read_json,
    safe_component,
    utc_now_iso,
    validate_analysis_tag,
    validate_pharmacy_ids,
    write_json,
)

# giotto-tda (gtda) is needed only for the PH half. If the import fails, the error is
# kept and reported by _require_gtda when PH starts, so the plans can still be built
# and checked on a computer without it.
try:
    from gtda.diagrams import PersistenceLandscape
    from gtda.homology import VietorisRipsPersistence
    _GTDA_IMPORT_ERROR = None
except Exception as exc:
    PersistenceLandscape = None
    VietorisRipsPersistence = None
    _GTDA_IMPORT_ERROR = exc

# Version label for the layout of plan.json. It is part of plan_hash, so changing it
# makes every saved plan look different and stops the run for existing tags.
PLAN_SCHEMA_VERSION = "2026-08-01"
# Scenario names a user cannot choose: "full" is the automatic baseline.
RESERVED_SCENARIO_NAMES = {"full"}


# =============================================================================
# Time units
# =============================================================================

def normalize_time_unit(value: Any) -> str:
    """Turn a time-unit setting into "seconds", "minutes" or "hours".

    Short forms are accepted, ignoring capital letters and spaces at either end:
    s, sec, second; m, min, minute; h, hr, hour. Anything else stops the run, so
    a typo in TDA_OUTPUT_TIME_UNIT is caught while the plan is built, before any
    PH runs.
    """
    aliases = {
        "s": "seconds", "sec": "seconds", "second": "seconds", "seconds": "seconds",
        "m": "minutes", "min": "minutes", "minute": "minutes", "minutes": "minutes",
        "h": "hours", "hr": "hours", "hour": "hours", "hours": "hours",
    }
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported time unit: {value!r}.")
    return normalized


def time_conversion_factor(input_unit: str, output_unit: str) -> float:
    """Return the number to multiply stored times by to express them in the output unit.

        factor = (seconds in one input unit) / (seconds in one output unit)

    Examples: seconds -> minutes gives 1 / 60, so 1359 s becomes 22.65 minutes;
    hours -> minutes gives 3600 / 60 = 60.

    Step 5 runs PH on the matrix in seconds and converts the diagram afterwards.
    This gives the same result as converting the matrix first: multiplying every
    distance by the same number multiplies every birth and death by that number.
    """
    seconds = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}
    source = normalize_time_unit(input_unit)
    target = normalize_time_unit(output_unit)
    return seconds[source] / seconds[target]


# =============================================================================
# Random removals: how many pharmacies, and which sets
# =============================================================================

def half_up_removal_count(total: int, proportion: float) -> int:
    """Number of pharmacies a random scenario removes: n * p, with .5 rounded up.

        removal_count = floor(n * p + 0.5), but at least 1 when p > 0

    Why not round()? Python's round() sends .5 to the nearest even number
    (round(2.5) == 2, round(3.5) == 4). Adding 0.5 and rounding down instead
    rounds every .5 up.

    Examples with p = 0.10: n = 25 gives 2.5 -> 3; n = 24 gives 2.4 -> 2; n = 4
    gives 0.4 -> 0, raised to 1 so that a positive share always removes someone.

    p = 0 returns 0, and expand_scenario_definitions then skips the scenario,
    since it would only repeat the full baseline. p must be between 0 and 1 and n
    at least 1. p = 1 returns n, which expand_scenario_definitions rejects because
    no pharmacy would be left.

    Limitation: n * p is a floating-point number, so a few exact halves come out a
    hair below .5 and round down, e.g. 90 * 0.35 = 31.499999999999996 -> 31.
    p = 0.05, 0.10, 0.20, 0.25 and 0.30 are not affected (checked up to n = 5000).
    """
    if total < 1:
        raise ValueError("Random scenarios require at least one pharmacy.")
    proportion = float(proportion)
    if not 0.0 <= proportion <= 1.0:
        raise ValueError("Random removal proportion must be in [0, 1].")
    if proportion == 0.0:
        return 0
    return max(1, int(math.floor(total * proportion + 0.5)))

# How the random replicates relate to each other: each replicate removes k
# different pharmacies (no pharmacy twice within a replicate; see STEP 5
# REMOVAL-SCENARIO SETTINGS in Step 0). Each replicate comes from a different
# combination rank, so no two replicates remove exactly the same set. A pharmacy
# can still be removed in several replicates, because every replicate is drawn
# from all of the city's pharmacies.
def unrank_combination(n: int, k: int, rank: int) -> tuple[int, ...]:
    """Return the combination (k positions out of n) that has the given rank.

    There are C(n, k) = math.comb(n, k) ways to choose k of n items. Listed in
    dictionary (lexicographic) order they are numbered 0 to C(n, k) - 1, and
    this function goes straight from a number (the rank) to its combination,
    without listing the others. The positions are 0-based places in the
    manifest order, so position 0 is the pharmacy with matrix_index 0.

    Example, n = 5 and k = 2 (10 combinations):
        rank:  0     1     2     3     4     5     6     7     8     9
              (0,1) (0,2) (0,3) (0,4) (1,2) (1,3) (1,4) (2,3) (2,4) (3,4)
    Finding rank 6:
    - First position. 4 combinations start with 0 (ranks 0-3): 6 >= 4, so skip
      them and 6 - 4 = 2 is left. 3 start with 1: 2 < 3, so take 1.
    - Second position (after 1). One combination each starts 1,2 / 1,3 / 1,4:
      skip 2 and 3 (2 - 1 - 1 = 0 left), take 4. Result: (1, 4).

    Why: a random scenario draws distinct random ranks
    (sample_unique_combination_ranks) and turns each one into a set of
    pharmacies here. Different ranks give different sets, so the replicates
    never repeat, with no draw-and-check loop and without storing all C(n, k)
    combinations (C(150, 15) is about 1.6e20).
    """
    # Total number of possible groups of k items chosen from n items.
    total = math.comb(n, k)
    # A valid rank must identify one of those possible combinations.
    if not 0 <= rank < total:
        raise ValueError(f"Combination rank {rank} is outside [0, {total}).")

    result: list[int] = []
    start = 0
    remaining_rank = int(rank)

    # Choose one position at a time, from left to right.
    for remaining_to_choose in range(k, 0, -1):
        for candidate in range(start, n):

            # How many combinations take this candidate next: the other
            # remaining_to_choose - 1 positions come from the n - candidate - 1
            # positions after it.
            count = math.comb(n - candidate - 1, remaining_to_choose - 1)

            # The rank falls in this candidate's block: keep it and move on.
            if remaining_rank < count:
                result.append(candidate)
                start = candidate + 1
                break

            # Otherwise skip this candidate's whole block of combinations.
            remaining_rank -= count
    if len(result) != k:
        raise RuntimeError("Combination unranking failed.")
    return tuple(result)


def sample_unique_combination_ranks(total_combinations: int, count: int, seed: int) -> list[int]:
    """Draw ''count'' different random ranks between 0 and total_combinations - 1.

    Each rank stands for one set of pharmacies (see unrank_combination), so
    different ranks mean different removal sets. Every rank is equally likely,
    so every set of k pharmacies is equally likely to be picked.

    The draws come from Python's random.Random(seed): the same seed, manifest and
    removal count always give the same ranks, and so the same replicates. (Every
    city uses the same seed, but the cities have different pharmacies, so their
    removals differ.)

    How the ranks are drawn depends on the number of possible sets, C(n, k):
    - Up to sys.maxsize (about 9.2e18 on a 64-bit computer): random.sample over
      range(total_combinations). The range is never stored as a list, so this
      is fast and uses little memory. Ranks come back in the order drawn.
    - Larger: random.sample cannot take a range this long (it needs the length
      to fit in sys.maxsize). Ranks are drawn one at a time with randrange and
      repeats are dropped until there are enough; with so many possible sets a
      repeat is practically impossible. Ranks come back sorted, smallest first.
      This branch is not rare: 150 pharmacies with 10% removed gives C(150, 15),
      about 1.6e20.
    Asking for more distinct ranks than exist stops the run.
    """
    if count > total_combinations:
        raise ValueError(
            f"Requested {count} unique subsets, but only {total_combinations} are possible."
        )
    rng = random.Random(int(seed))
    if total_combinations <= sys.maxsize:
        # range() produces numbers on demand, so even billions of ranks take no memory.
        return list(rng.sample(range(total_combinations), count))
    # Too many possible sets for random.sample: draw one rank at a time; the set
    # drops any repeat, so keep drawing until there are enough.
    selected: set[int] = set()
    while len(selected) < count:
        selected.add(rng.randrange(total_combinations))
    return sorted(selected)


# =============================================================================
# Reading and merging the Step 0 scenario definitions
# =============================================================================

def _normalize_definition(definition: Mapping[str, Any], scope: str) -> dict[str, Any]:
    """Tidy one scenario definition from Step 0 and record where it came from.

    Returns a copy with:
    - every value made JSON-safe, so it can be saved in plan.json and hashed;
    - ''scope'': "global" (from GLOBAL_SCENARIOS) or "city" (from CITY_SCENARIOS);
    - ''enabled'': True unless the definition says otherwise;
    - ''name'' with spaces trimmed from both ends, and ''type'' trimmed and in
      lower case.
    Stops if an enabled definition has no name, or if any definition (even a
    disabled one) is named "full" in any capitalization, which is reserved for
    the automatic baseline.
    """
    if not isinstance(definition, Mapping):
        raise ValueError(f"Scenario definition must be a mapping, not {type(definition).__name__}.")
    normalized = {str(key): json_safe(value) for key, value in definition.items()}
    normalized["scope"] = scope
    normalized["enabled"] = bool(normalized.get("enabled", True))
    normalized["name"] = str(normalized.get("name", "")).strip()
    normalized["type"] = str(normalized.get("type", "")).strip().lower()
    if normalized["enabled"] and not normalized["name"]:
        raise ValueError("Every enabled scenario requires a nonblank name.")
    if normalized["name"].lower() in RESERVED_SCENARIO_NAMES:
        raise ValueError("'full' is reserved for the automatic baseline.")
    return normalized


def merge_scenario_definitions(
    global_scenarios: Iterable[Mapping[str, Any]],
    city_scenarios: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Combine GLOBAL_SCENARIOS and one city's CITY_SCENARIOS into one list (stage 2).

    Global definitions come first, then the city's, each in the order written in
    Step 0. Disabled definitions are dropped here, so they never reach the plan
    and do not affect plan_hash.

    Every scenario name becomes a folder name, so names must be unique:
    - ignoring capital letters ("Remove_CVS" and "remove_cvs" clash), because
      Windows folder names ignore capital letters;
    - after cleaning for the file system, where every run of characters other
      than letters, digits, '.', '_' and '-' becomes '_' ("remove CVS" and
      "remove_CVS" both become "remove_CVS").
    Any clash stops the run.
    """
    definitions = [
        *[_normalize_definition(item, "global") for item in global_scenarios],
        *[_normalize_definition(item, "city") for item in city_scenarios],
    ]
    definitions = [item for item in definitions if item["enabled"]]
    names = [item["name"] for item in definitions]
    # casefold() is a thorough lower(): it lets names be compared ignoring case.
    folded_names = [name.casefold() for name in names]
    duplicates = sorted({names[i] for i, value in enumerate(folded_names) if folded_names.count(value) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate scenario names after merging global and city definitions: {duplicates}."
        )
    # The same check on the folder-safe versions of the names.
    safe_names = [safe_component(name, "scenario") for name in names]
    folded_safe_names = [name.casefold() for name in safe_names]
    safe_duplicates = sorted({safe_names[i] for i, value in enumerate(folded_safe_names) if folded_safe_names.count(value) > 1})
    if safe_duplicates:
        raise ValueError(
            f"Scenario names not unique after filesystem cleaning: {safe_duplicates}."
        )
    return definitions


# =============================================================================
# Expanding definitions into concrete scenarios (who is kept, who is removed)
# =============================================================================

def _base_scenario(
    name: str,
    scenario_type: str,
    scope: str,
    all_ids: list[str],
    excluded_ids: Iterable[str],
    exclusion_reason: str,
    **extra: Any,
) -> dict[str, Any]:
    """Build one concrete scenario: the kept and removed pharmacy lists plus labels.

    ''included_ids'' (kept) and ''excluded_ids'' (removed) are both in manifest
    order, whatever order the removed IDs were given in. ''extra'' holds the
    details for each type (seed, replicate number, matched terms, the original
    definition, ...), copied in as JSON-safe values. ''scenario_key'' starts as
    the folder-safe name; random and leave-one-out scenarios replace it with a
    two-level key afterwards. Stops if the scenario would remove every pharmacy.
    """
    excluded_set = {str(value) for value in excluded_ids}
    included = [pharmacy_id for pharmacy_id in all_ids if pharmacy_id not in excluded_set]
    excluded = [pharmacy_id for pharmacy_id in all_ids if pharmacy_id in excluded_set]
    if not included:
        raise ValueError(f"Scenario {name!r} removes every pharmacy.")
    return {
        "scenario_name": name,
        "scenario_type": scenario_type,
        "scenario_scope": scope,
        "scenario_key": safe_component(name, "scenario"),
        "included_ids": included,
        "excluded_ids": excluded,
        "exclusion_reason": exclusion_reason,
        **json_safe(extra),
    }


def expand_scenario_definitions(
    manifest: pd.DataFrame,
    definitions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn the merged definitions into concrete scenarios (stage 3).

    The automatic baseline "full" (every pharmacy kept) always comes first,
    followed by each definition in turn:
    - exclude_ids / explicit_exclusion (CITY_SCENARIOS only): removes the listed
      ''pharmacy_ids''. Stops if the list is empty, has repeats, or names an ID
      that is not in this run's Step 3 manifest.
    - chain_exclusion / remove_chain: removes every pharmacy whose
      OrganizationName contains any of the ''organization_name_terms'' (or
      ''match_terms''), as plain text and ignoring capital letters. Stops if
      nothing matches, so a global chain scenario must match in every city.
    - random_exclusion / random_removal: removes half_up_removal_count(n,
      ''proportion'') pharmacies, in ''replicates'' different random sets drawn
      from ''seed'' (123 if not given). One scenario per replicate.
    - leave_one_out / loo (GLOBAL_SCENARIOS only): one scenario per pharmacy.
    Any other type stops the run: only removal scenarios are supported.

    Returns (expanded_scenarios, skipped_definitions). A random definition whose
    proportion gives 0 removals goes into skipped_definitions instead of making
    a copy of the baseline. Finally every scenario key must be unique, since
    each key is a folder name.
    """
    all_ids = manifest["pharmacy_id"].astype(str).tolist()
    all_id_set = set(all_ids)
    # The baseline: every pharmacy kept, nothing removed.
    expanded: list[dict[str, Any]] = [
        {
            "scenario_name": "full",
            "scenario_type": "baseline",
            "scenario_scope": "automatic",
            "scenario_key": "full",
            "included_ids": all_ids.copy(),
            "excluded_ids": [],
            "exclusion_reason": "",
        }
    ]
    skipped: list[dict[str, Any]] = []

    for definition in definitions:
        name = definition["name"]
        scenario_type = definition["type"]
        scope = definition["scope"]

        # Explicit IDs ------------------------------------------------------------
        if scenario_type in {"exclude_ids", "explicit_exclusion"}:
            # A pharmacy_id belongs to one city's manifest, so these scenarios can
            # only be defined for one city.
            if scope != "city":
                raise ValueError(
                    f"Explicit-ID scenario {name!r} must be defined in CITY_SCENARIOS, "
                    "not GLOBAL_SCENARIOS."
                )
            requested = [str(value) for value in definition.get("pharmacy_ids", [])]
            if not requested:
                raise ValueError(
                    f"Explicit scenario {name!r} has no pharmacy IDs. Disable it instead."
                )
            if len(requested) != len(set(requested)):
                raise ValueError(f"Explicit scenario {name!r} contains duplicate IDs.")
            missing = sorted(set(requested) - all_id_set)
            if missing:
                raise ValueError(
                    f"Explicit scenario {name!r} references IDs not present in the Step 3 "
                    f"manifest: {missing}. Review the manifest for this city and analysis run."
                )
            expanded.append(
                _base_scenario(
                    name,
                    "exclude_ids",
                    scope,
                    all_ids,
                    requested,
                    "user_selected_explicit_removal",
                    source_definition=definition,
                )
            )

        # Chain names -------------------------------------------------------------
        elif scenario_type in {"chain_exclusion", "remove_chain"}:
            terms = definition.get("organization_name_terms", definition.get("match_terms", []))
            # A single term written as text counts as a one-term list; blank
            # terms are dropped.
            if isinstance(terms, str):
                terms = [terms]
            terms = [str(term).strip() for term in terms if str(term).strip()]
            if not terms:
                raise ValueError(f"Chain scenario {name!r} requires organization-name terms.")
            if "OrganizationName" not in manifest.columns:
                raise ValueError("Step 3 manifest lacks OrganizationName for chain matching.")
            names = manifest["OrganizationName"].astype("string")
            mask = pd.Series(False, index=manifest.index)
            matched_terms: dict[str, str] = {}
            for term in terms:
                # Plain-text match (regex=False) anywhere in the name, ignoring
                # capital letters; a missing name never matches. "CVS" matches
                # "CVS PHARMACY #1234" and "cvs/pharmacy". Because it matches inside
                # words too, short terms can catch other names: "RITE" also
                # matches "FAVORITE PHARMACY".
                term_mask = names.str.contains(term, case=False, na=False, regex=False)
                mask |= term_mask
                # Remember the first term that matched each pharmacy (it is shown
                # in the pharmacy's ''matched_term'' membership column).
                for pharmacy_id in manifest.loc[term_mask, "pharmacy_id"].astype(str):
                    matched_terms.setdefault(pharmacy_id, term)
            excluded = manifest.loc[mask, "pharmacy_id"].astype(str).tolist()
            if not excluded:
                raise ValueError(
                    f"Chain scenario {name!r} matched zero pharmacies using OrganizationName "
                    f"terms {terms}. Check the Step 3 manifest or use a new scenario definition."
                )
            expanded.append(
                _base_scenario(
                    name,
                    "chain_exclusion",
                    scope,
                    all_ids,
                    excluded,
                    "organization_name_chain_match",
                    organization_name_terms=terms,
                    matched_terms=matched_terms,
                    source_definition=definition,
                )
            )

        # Random removals ---------------------------------------------------------
        elif scenario_type in {"random_exclusion", "random_removal"}:
            proportion = float(definition.get("proportion", definition.get("remove_proportion", 0)))
            removal_count = half_up_removal_count(len(all_ids), proportion)
            # Removing nobody would just repeat the baseline: record it as skipped.
            if removal_count == 0:
                skipped.append(
                    {
                        "scenario_name": name,
                        "scenario_type": "random_exclusion",
                        "reason": "zero_removal_equivalent_to_full_baseline",
                        "source_definition": definition,
                    }
                )
                continue
            if removal_count >= len(all_ids):
                raise ValueError(
                    f"Random scenario {name!r} would remove {removal_count} of "
                    f"{len(all_ids)} pharmacies and leave an empty subset."
                )
            replicates = int(definition.get("replicates", 1))
            if replicates < 1:
                raise ValueError(f"Random scenario {name!r} requires replicates >= 1.")
            seed = int(definition.get("seed", definition.get("random_seed", 123)))
            # Number of different sets of removal_count pharmacies, C(n, k). The
            # replicates must all differ, so there cannot be more of them than this
            # (e.g. removing 1 of 8 pharmacies allows at most 8 replicates).
            possible = math.comb(len(all_ids), removal_count)
            if replicates > possible:
                raise ValueError(
                    f"Random scenario {name!r} requests {replicates} unique replicates, "
                    f"but only {possible} unique subsets are possible when removing "
                    f"{removal_count} of {len(all_ids)} pharmacies."
                )
            ranks = sample_unique_combination_ranks(possible, replicates, seed)
            for replicate_number, rank in enumerate(ranks, start=1):
                # rank -> positions in manifest order -> pharmacy IDs to remove.
                indices = unrank_combination(len(all_ids), removal_count, rank)
                removed = [all_ids[index] for index in indices]
                scenario = _base_scenario(
                    name,
                    "random_exclusion",
                    scope,
                    all_ids,
                    removed,
                    "random_proportional_removal",
                    parent_scenario_name=name,
                    replicate_number=replicate_number,
                    random_seed=seed,
                    combination_rank=rank,
                    removal_proportion=proportion,
                    removal_count=removal_count,
                    maximum_unique_subsets=possible,
                    source_definition=definition,
                )
                # One sub-folder per replicate: <name>/replicate_001, ...
                scenario["scenario_key"] = (
                    f"{safe_component(name, 'random')}/replicate_{replicate_number:03d}"
                )
                expanded.append(scenario)

        # Leave one out -----------------------------------------------------------
        elif scenario_type in {"leave_one_out", "loo"}:
            if scope != "global":
                raise ValueError(
                    f"Leave-one-out scenario {name!r} must be defined in GLOBAL_SCENARIOS."
                )
            # One scenario per pharmacy, so a city with 150 pharmacies gets 150
            # scenarios (and 150 PH runs for each selected matrix type).
            for pharmacy_id in all_ids:
                scenario = _base_scenario(
                    name,
                    "leave_one_out",
                    scope,
                    all_ids,
                    [pharmacy_id],
                    "leave_one_out",
                    parent_scenario_name=name,
                    excluded_pharmacy_id=pharmacy_id,
                    source_definition=definition,
                )
                # One sub-folder per removed pharmacy: <name>/<pharmacy_id>
                scenario["scenario_key"] = (
                    f"{safe_component(name, 'leave_one_out')}/{safe_component(pharmacy_id, 'pharmacy')}"
                )
                expanded.append(scenario)
        else:
            raise ValueError(
                f"Unsupported scenario type {scenario_type!r} for {name!r}. The revised "
                "pipeline supports removal-based scenarios only."
            )

    # Every key becomes a folder (and a membership file name), so keys must be unique.
    keys = [scenario["scenario_key"] for scenario in expanded]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"Expanded scenario keys are not unique: {duplicate_keys}.")
    return expanded, skipped


def validate_expanded_scenario(scenario: Mapping[str, Any], all_ids: list[str]) -> None:
    """Check that a scenario splits the manifest cleanly into kept and removed (stage 4).

    Every manifest pharmacy must be in exactly one of the two lists, with no
    repeats and at least one pharmacy kept, and the kept list must be in manifest
    order, so the cut-down matrix keeps the row order of the full matrix. Any
    problem stops the run.
    """
    included = [str(value) for value in scenario["included_ids"]]
    excluded = [str(value) for value in scenario["excluded_ids"]]
    if len(included) != len(set(included)) or len(excluded) != len(set(excluded)):
        raise ValueError(f"Scenario {scenario['scenario_name']!r} contains duplicate IDs.")
    if not included:
        raise ValueError(f"Scenario {scenario['scenario_name']!r} produces an empty matrix.")
    # Kept + removed must together be the whole manifest ...
    if set(included) | set(excluded) != set(all_ids):
        raise ValueError(f"Scenario {scenario['scenario_name']!r} does not cover the full manifest.")
    # ... with no pharmacy in both lists.
    if set(included) & set(excluded):
        raise ValueError(f"Scenario {scenario['scenario_name']!r} overlaps included/excluded IDs.")
    if included != [pharmacy_id for pharmacy_id in all_ids if pharmacy_id in set(included)]:
        raise ValueError(f"Scenario {scenario['scenario_name']!r} changed manifest order.")


def membership_dataframe(
    manifest: pd.DataFrame,
    scenario: Mapping[str, Any],
    analysis_run_tag: str,
) -> pd.DataFrame:
    """Build the membership table for one scenario: one row per manifest pharmacy.

    Each row has all of the manifest's columns, plus:
    - ''analysis_run_tag'', ''scenario_name'', ''scenario_key'', ''scenario_type''
      and ''scenario_scope'';
    - ''included'': True if the scenario keeps the pharmacy, False if it removes it;
    - ''exclusion_reason'': blank for kept pharmacies, otherwise why the pharmacy
      was removed (e.g. "random_proportional_removal");
    - ''matched_term'': for chain scenarios, the term that matched (else empty);
    - ''replicate_number'' and ''random_seed'': for random scenarios (else empty).
    It is saved as memberships/<scenario_key>.csv. The PH half and Step 6 read its
    ''included'' column to know which pharmacies the scenario keeps.
    """
    included = set(str(value) for value in scenario["included_ids"])
    matched_terms = scenario.get("matched_terms", {}) or {}
    rows = []
    for _, row in manifest.iterrows():
        pharmacy_id = str(row["pharmacy_id"])
        is_included = pharmacy_id in included
        record = row.to_dict()
        record.update(
            {
                ANALYSIS_TAG_COLUMN: analysis_run_tag,
                "scenario_name": scenario["scenario_name"],
                "scenario_key": scenario["scenario_key"],
                "scenario_type": scenario["scenario_type"],
                "scenario_scope": scenario["scenario_scope"],
                "included": bool(is_included),
                "exclusion_reason": "" if is_included else scenario.get("exclusion_reason", "excluded"),
                "matched_term": matched_terms.get(pharmacy_id),
                "replicate_number": scenario.get("replicate_number"),
                "random_seed": scenario.get("random_seed"),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


# =============================================================================
# Building, checking and saving the locked plan
# =============================================================================

def build_scenario_plan_draft(
    *,
    paths: CityRunPaths,
    selected_matrix_types: Iterable[str],
    global_scenarios: Iterable[Mapping[str, Any]],
    city_scenarios: Iterable[Mapping[str, Any]],
    ph_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and check one city's complete scenario plan in memory; nothing is saved.

    Step 0 calls this for every city first, so a problem in any city stops Step 5
    before any plan is saved. The numbered comments below match stages 1-5 at the
    top of this file.

    ph_settings holds ''input_time_unit'', ''output_time_unit'' and
    ''max_homology_dim'' (Step 0's TDA_* settings); a missing entry defaults to
    seconds, minutes and 1.

    Returns a dict with:
    - ''plan'': the content of plan.json, including ''plan_hash'';
    - ''scenario_index'': the table saved as scenario_index.csv;
    - ''memberships'': {scenario_key: membership table};
    - ''expanded_scenarios'' (with the full ID lists) and ''manifest'', which
      Step 0 does not use.

    What plan_hash covers: the plan layout version, the run tag and city, the
    manifest (a checksum of the file and of the pharmacy order), the selected
    matrix types, the PH settings, the enabled and skipped definitions, and every
    expanded scenario with its membership hash. ''created_utc'' is left out, so
    the same plan built on another day has the same hash. The Step 4 matrix files
    are not in the hash; each result records its matrix's checksum instead (see
    _result_complete). Lists count in order, so reordering ANALYSIS_TRAVEL_MODES or
    the scenario definitions in Step 0 also changes the hash.
    """
    # 1. Read the Step 3 manifest (IDs as text, so leading zeros are kept) and check
    #    its run tag, its pharmacy IDs, and that matrix_index runs 0, 1, 2, ... in
    #    order (row k of every matrix is the pharmacy with matrix_index k).
    manifest = pd.read_csv(
        paths.step03_manifest,
        dtype={"pharmacy_id": "string", "NPI": "string", "google_place_id": "string"},
    )
    validate_analysis_tag(manifest, paths.analysis_run_tag, str(paths.step03_manifest))
    all_ids = validate_pharmacy_ids(manifest, str(paths.step03_manifest))
    if "matrix_index" not in manifest.columns:
        raise ValueError("Step 3 manifest is missing matrix_index.")
    indices = pd.to_numeric(manifest["matrix_index"], errors="coerce")
    if indices.isna().any() or not np.array_equal(indices.to_numpy(), np.arange(len(manifest))):
        raise ValueError("Step 3 manifest matrix_index is not sequential from zero.")

    #    Selected matrix types: tidy the names, drop repeats (keeping the order),
    #    and check that each Step 4 matrix file exists.
    matrix_types = [normalize_matrix_type(value) for value in selected_matrix_types]
    matrix_types = list(dict.fromkeys(matrix_types))
    if not matrix_types:
        raise ValueError("Step 5 requires at least one selected Step 4 matrix type.")
    for matrix_type in matrix_types:
        if not paths.derived_matrix_path(matrix_type).exists():
            raise FileNotFoundError(
                f"Selected Step 4 matrix does not exist for {paths.city_state}: "
                f"{paths.derived_matrix_path(matrix_type)}"
            )

    # 2. Merge the global and city definitions (enabled ones only).
    definitions = merge_scenario_definitions(global_scenarios, city_scenarios)
    # 3. Expand them into concrete scenarios, the baseline first.
    expanded, skipped = expand_scenario_definitions(manifest, definitions)
    # 4. Check every scenario.
    for scenario in expanded:
        validate_expanded_scenario(scenario, all_ids)

    # PH settings in standard spellings, so "min" and "minutes" give the same hash.
    normalized_ph = {
        "input_time_unit": normalize_time_unit(ph_settings.get("input_time_unit", "seconds")),
        "output_time_unit": normalize_time_unit(ph_settings.get("output_time_unit", "minutes")),
        "max_homology_dim": int(ph_settings.get("max_homology_dim", 1)),
    }
    if normalized_ph["max_homology_dim"] < 0:
        raise ValueError("max_homology_dim cannot be negative.")

    # 5. Everything the plan depends on. plan_hash is the SHA-256 of this material.
    manifest_checksum = ordered_id_checksum(all_ids)
    plan_material = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
        "city": paths.city,
        "state": paths.state.upper(),
        "city_state": paths.city_state,
        "source_manifest": str(paths.step03_manifest.relative_to(paths.city_dir)),
        "source_manifest_sha256": file_sha256(paths.step03_manifest),
        "ordered_pharmacy_id_checksum": manifest_checksum,
        "selected_matrix_types": matrix_types,
        "ph_settings": normalized_ph,
        "global_scenario_definitions": [
            item for item in definitions if item["scope"] == "global"
        ],
        "city_scenario_definitions": [
            item for item in definitions if item["scope"] == "city"
        ],
        "skipped_definitions": skipped,
        # One entry per scenario: its labels and details, the numbers kept and
        # removed, and its membership hash (the SHA-256 of its kept and removed ID
        # lists). The ID lists themselves are left out of plan.json; they are in
        # the membership files.
        "expanded_scenarios": [
            {
                key: value
                for key, value in scenario.items()
                if key not in {"included_ids", "excluded_ids", "matched_terms"}
            }
            | {
                "included_count": len(scenario["included_ids"]),
                "excluded_count": len(scenario["excluded_ids"]),
                "membership_hash": object_sha256(
                    {
                        "included_ids": scenario["included_ids"],
                        "excluded_ids": scenario["excluded_ids"],
                    }
                ),
            }
            for scenario in expanded
        ],
    }
    plan_hash = object_sha256(plan_material)
    # The creation time is added after hashing, so it does not affect the hash.
    plan = {
        **plan_material,
        "plan_hash": plan_hash,
        "created_utc": utc_now_iso(),
        "scenario_count": len(expanded),
    }
    #    One membership table per scenario, and the index (one row per scenario)
    #    that the PH half, Step 6 and Step 7 loop over.
    memberships = {
        scenario["scenario_key"]: membership_dataframe(
            manifest, scenario, paths.analysis_run_tag
        )
        for scenario in expanded
    }
    scenario_index = pd.DataFrame(
        [
            {
                "scenario_name": scenario["scenario_name"],
                "scenario_key": scenario["scenario_key"],
                "scenario_type": scenario["scenario_type"],
                "scenario_scope": scenario["scenario_scope"],
                "parent_scenario_name": scenario.get("parent_scenario_name"),
                "replicate_number": scenario.get("replicate_number"),
                "included_count": len(scenario["included_ids"]),
                "excluded_count": len(scenario["excluded_ids"]),
                "excluded_pharmacy_ids": "|".join(scenario["excluded_ids"]),
                "random_seed": scenario.get("random_seed"),
                "membership_relative_path": f"memberships/{scenario['scenario_key']}.csv",
                # Same calculation as the membership hash in the plan above.
                "membership_hash": object_sha256(
                    {
                        "included_ids": scenario["included_ids"],
                        "excluded_ids": scenario["excluded_ids"],
                    }
                ),
            }
            for scenario in expanded
        ]
    )
    return {
        "plan": plan,
        "scenario_index": scenario_index,
        "memberships": memberships,
        "expanded_scenarios": expanded,
        "manifest": manifest,
    }


def validate_or_accept_existing_plan(paths: CityRunPaths, draft: Mapping[str, Any]) -> bool:
    """Compare the plan just built with the saved plan, if there is one (stage 6).

    - No saved plan.json: returns False, and Step 0 saves the new plan.
    - A saved plan with the same plan_hash: returns True, and the saved plan is
      used (PH resumes), provided scenario_index.csv and the memberships folder
      exist; if they don't, the saved plan is incomplete and the run stops.
    - A saved plan with a different plan_hash: the run stops. The plan is locked
      (scenarios or matrix types are never added to it), so every run with this
      ANALYSIS_RUN_TAG analyzes exactly the same scenario membership. Go back to the
      original settings, or use a new ANALYSIS_RUN_TAG.
    """
    if not paths.scenario_plan_json.exists():
        return False
    saved = read_json(paths.scenario_plan_json)
    saved_hash = saved.get("plan_hash")
    requested_hash = draft["plan"]["plan_hash"]
    if saved_hash != requested_hash:
        raise ValueError(
            f"The saved Step 5 scenario plan for {paths.city_state} does not match the "
            "current scenario definitions, selected matrix types, manifest, or PH settings. "
            "The pipeline does not add scenarios or matrix types to a locked plan. "
            "Use the original configuration or select a new ANALYSIS_RUN_TAG."
        )
    if not paths.scenario_index_csv.exists() or not paths.scenario_membership_dir.exists():
        raise RuntimeError(
            f"The Step 5 plan exists but its index/memberships are incomplete: {paths.scenario_plan_dir}"
        )
    return True


def save_scenario_plan(paths: CityRunPaths, draft: Mapping[str, Any]) -> None:
    """Write one city's plan: plan.json, scenario_index.csv and memberships/ (stage 7).

    Step 0 calls this only after every city's plan has been built and checked,
    and only for cities that have no saved plan yet. It never overwrites: an
    existing plan.json stops the run. Random and leave-one-out keys contain a
    '/', so their membership files go in a sub-folder, e.g.
    memberships/remove_10_percent_random/replicate_001.csv. plan.json is written
    first, then the index, then the membership files.
    """
    paths.scenario_plan_dir.mkdir(parents=True, exist_ok=True)
    if paths.scenario_plan_json.exists():
        raise FileExistsError(f"Scenario plan already exists: {paths.scenario_plan_json}")
    write_json(paths.scenario_plan_json, draft["plan"])
    draft["scenario_index"].to_csv(paths.scenario_index_csv, index=False)
    for scenario_key, membership in draft["memberships"].items():
        target = paths.scenario_membership_dir / f"{scenario_key}.csv"
        # Creates the sub-folder for two-level keys such as <name>/replicate_001.
        target.parent.mkdir(parents=True, exist_ok=True)
        membership.to_csv(target, index=False)


# =============================================================================
# Persistent homology for one matrix
# =============================================================================

def _require_gtda() -> None:
    """Stop with a clear message if giotto-tda could not be imported.

    Called at the start of every PH run (process_single_matrix), so the planning
    half still works on a computer without giotto-tda.
    """
    if _GTDA_IMPORT_ERROR is not None:
        raise ImportError(
            "Step 5 requires giotto-tda for persistent homology. "
            f"Original import error: {_GTDA_IMPORT_ERROR}"
        )


def _prepare_matrix_for_ph(matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Clean one (cut-down) matrix so that Giotto can use it as a distance matrix.

    Returns (cleaned copy, warnings). The input is not changed. In order:
    1. It must be square and not empty, otherwise the run stops.
    2. NaN (no travel time, e.g. no drive route) becomes +inf, so Giotto never
       joins that pair.
    3. The diagonal becomes 0 (a pharmacy is at distance 0 from itself).
    4. It is made symmetric by keeping the shorter direction of each pair:
           M[i, j] = M[j, i] = min(M[i, j], M[j, i])
       The d matrices are already symmetric, so this changes nothing for them;
       only a dtilde matrix (if analyzed) can change. If any pair's two
       directions differ by more than 1e-8, the largest difference is recorded
       as a warning. (Pairs that are infinite in either direction are left out
       of that check.)
    5. Negative values become 0. Travel times cannot be negative; this is a
       safety net.

    Example: [[0, 5, NaN], [7, 0, 4], [NaN, 4, 0]] becomes
    [[0, 5, inf], [5, 0, 4], [inf, 4, 0]], with the warnings "converted 2 NaN
    entries to +inf" and "symmetrized matrix with maximum finite asymmetry 2".

    Known issue (h): the warnings are not printed; they are only saved, in
    summary.json and result_metadata.json. Known issue (a): this is where a
    pharmacy with NaN drive times becomes infinitely far from the others.
    """
    prepared = np.asarray(matrix, dtype=float).copy()
    if prepared.ndim != 2 or prepared.shape[0] != prepared.shape[1]:
        raise ValueError(f"PH input must be square; received {prepared.shape}.")
    if prepared.shape[0] == 0:
        raise ValueError("PH input is empty.")
    warnings: list[str] = []
    # NaN -> +inf: an edge that never appears.
    nan_count = int(np.isnan(prepared).sum())
    if nan_count:
        prepared[np.isnan(prepared)] = np.inf
        warnings.append(f"converted {nan_count} NaN entries to +inf")
    np.fill_diagonal(prepared, 0.0)
    # Measure the asymmetry over pairs that are finite both ways (inf - inf would
    # give NaN).
    finite_pairs = np.isfinite(prepared) & np.isfinite(prepared.T)
    if np.any(finite_pairs):
        asymmetry = float(np.max(np.abs(prepared[finite_pairs] - prepared.T[finite_pairs])))
        if asymmetry > 1e-8:
            warnings.append(f"symmetrized matrix with maximum finite asymmetry {asymmetry:g}")
    # Keep the shorter direction of each pair (known issue (h)).
    prepared = np.minimum(prepared, prepared.T)
    negative_count = int((prepared < 0).sum())
    if negative_count:
        prepared[prepared < 0] = 0.0
        warnings.append(f"clipped {negative_count} negative entries to zero")
    return prepared, warnings


def _dimension_statistics(diagram: np.ndarray, requested_max_dim: int) -> dict[str, Any]:
    """Summary numbers for each homology dimension 0..requested_max_dim of a diagram.

    The diagram has one row per class: (birth, death, dimension). For each
    dimension k:
    - H{k}_point_count: the number of rows for dimension k.
    - H{k}_finite_feature_count: rows whose death is finite.
    - H{k}_infinite_feature_count: rows whose death is +inf (classes that never
      die). Normally 0 for H0, because reduced homology drops the one group that
      never dies; more than 0 when a pharmacy can never be reached (known issue
      (a)).
    - H{k}_mean_finite_lifetime and H{k}_max_finite_lifetime: the average and the
      largest lifetime (death - birth) over the finite rows; None if there are
      none. For H1 the maximum belongs to the most persistent hole.
    - H{k}_total_finite_persistence: the sum of those lifetimes (0.0 if none). For
      H0 every birth is 0, so this is the total length of the minimum spanning
      tree.
    Units are those of the diagram (minutes by default).

    Example: H1 rows (10, 16) and (12, 13) give point count 2, finite count 2,
    infinite count 0, mean lifetime 3.5, maximum 6 and total 7.

    Notes:
    - The requested dimension is used, so if process_single_matrix had to lower
      it (very few pharmacies), the dimension it skipped shows a count of 0.
    - When a dimension has no classes at all, Giotto puts one placeholder point
      with birth = death in the diagram, so that every dimension appears. That
      point is counted here: a point count of 1 with maximum lifetime 0 means
      "no classes".
    """
    statistics: dict[str, Any] = {}
    for dimension in range(requested_max_dim + 1):
        # The rows for this dimension (column 2 holds the dimension).
        points = (
            diagram[diagram[:, 2].astype(int) == dimension]
            if diagram.size
            else np.empty((0, 3))
        )
        finite = np.isfinite(points[:, 1]) if len(points) else np.array([], dtype=bool)
        # Lifetime = death - birth, for classes that do die.
        lifetimes = points[finite, 1] - points[finite, 0] if len(points) else np.array([])
        statistics.update(
            {
                f"H{dimension}_point_count": int(len(points)),
                f"H{dimension}_finite_feature_count": int(finite.sum()),
                f"H{dimension}_infinite_feature_count": int((~finite).sum()),
                f"H{dimension}_mean_finite_lifetime": (
                    float(np.mean(lifetimes)) if lifetimes.size else None
                ),
                f"H{dimension}_max_finite_lifetime": (
                    float(np.max(lifetimes)) if lifetimes.size else None
                ),
                f"H{dimension}_total_finite_persistence": (
                    float(np.sum(lifetimes)) if lifetimes.size else 0.0
                ),
            }
        )
    return statistics


def process_single_matrix(
    matrix: np.ndarray,
    output_dir: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any],
    input_time_unit: str = "seconds",
    output_time_unit: str = "minutes",
    max_homology_dim: int = 1,
) -> dict[str, Any]:
    """Run persistent homology on one cut-down matrix and save every result file.

    Input: one scenario's matrix (kept pharmacies only, in manifest order, in
    input_time_unit), the folder to write to, and the details to record in
    result_metadata.json. The defaults (seconds, minutes, dimension 1) apply only
    when this is called on its own; execute_saved_scenario_plan always passes the
    plan's PH settings.

    Stage 10 at the top of this file:
    a. Clean the matrix (_prepare_matrix_for_ph).
    b. Run Giotto's VietorisRipsPersistence with metric="precomputed" for
       dimensions 0..max_dim. The edge between i and j appears when the
       filtration value reaches d[i, j], which is twice Hickok et al.'s t.
       Giotto reports reduced homology (the never-dying group is dropped) and
       leaves out classes of zero length.
    c. Convert births and deaths to output_time_unit (seconds / 60 = minutes).
    d. Save persistence_diagram.npy and .csv, birth_death.npz,
       persistence_diagram.png, persistence_landscape.png, summary.json and,
       last, result_metadata.json, whose ''processing_status'' "complete" marks
       the folder as finished.
    Files already in the folder are overwritten; this is how an incomplete
    result is redone. Returns the content of result_metadata.json.

    Very few pharmacies: H1 needs at least 3, so the dimension is capped at
    n - 2 (n = 2 gives H0 only), and a warning is recorded.
    """
    _require_gtda()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # a. Clean the matrix.
    prepared, warnings = _prepare_matrix_for_ph(matrix)
    # Highest dimension that n points allow: n - 2, and never below 0.
    max_dim = min(int(max_homology_dim), max(0, prepared.shape[0] - 2))
    if max_dim < int(max_homology_dim):
        warnings.append(
            f"limited homology dimensions to 0..{max_dim} for {prepared.shape[0]} points"
        )

    # b. Vietoris-Rips persistence. metric="precomputed" means the entries are the
    #    distances themselves; +inf means the pair is never joined.
    vr = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dim + 1)), metric="precomputed"
    )
    # Giotto works on a batch of matrices: add a first axis (a batch of one) and
    # take result 0. Each row of the diagram is (birth, death, dimension).
    diagram = vr.fit_transform(prepared[np.newaxis, :, :])[0]
    # c. Convert births and deaths (columns 0 and 1), e.g. seconds x 1/60 = minutes.
    #    Column 2 is the dimension and stays as it is; infinite deaths stay infinite.
    factor = time_conversion_factor(input_time_unit, output_time_unit)
    scaled = np.asarray(diagram, dtype=float).copy()
    if scaled.size:
        scaled[:, :2] *= factor

    # d. Save the results.
    #    The diagram as .npy (what Step 7 reads) and as .csv.
    np.save(output / "persistence_diagram.npy", scaled)
    pd.DataFrame(scaled, columns=["birth", "death", "homology_dimension"]).to_csv(
        output / "persistence_diagram.csv", index=False
    )
    #    Births and deaths split by dimension: dim_0_birth, dim_0_death,
    #    dim_1_birth, dim_1_death.
    birth_death = {}
    if scaled.size:
        for dimension in np.unique(scaled[:, 2]).astype(int):
            points = scaled[scaled[:, 2].astype(int) == dimension]
            birth_death[f"dim_{dimension}_birth"] = points[:, 0]
            birth_death[f"dim_{dimension}_death"] = points[:, 1]
    np.savez(output / "birth_death.npz", **birth_death)

    #    Persistence diagram plot: one colour per dimension, birth across and death
    #    up. Points far above the dashed line (birth = death) are classes that last
    #    a long time. Matplotlib leaves out points with an infinite death, so a
    #    class that never dies (known issue (a)) is in the .npy and .csv files but
    #    not in the plot.
    plt.figure(figsize=(6, 6))
    if scaled.size:
        for dimension in np.unique(scaled[:, 2]).astype(int):
            points = scaled[scaled[:, 2].astype(int) == dimension]
            plt.scatter(points[:, 0], points[:, 1], label=f"H{dimension}")
        # The dashed diagonal runs from the smallest to the largest finite value.
        finite_values = scaled[:, :2][np.isfinite(scaled[:, :2])]
        if finite_values.size:
            lower, upper = float(np.min(finite_values)), float(np.max(finite_values))
            plt.plot([lower, upper], [lower, upper], linestyle="--")
        plt.legend()
    plt.xlabel(f"Birth ({normalize_time_unit(output_time_unit)})")
    plt.ylabel(f"Death ({normalize_time_unit(output_time_unit)})")
    plt.tight_layout()
    plt.savefig(output / "persistence_diagram.png", dpi=300)
    plt.close()

    #    Persistence landscape. Each class (b, d) becomes a tent: it rises from 0
    #    at b to a peak of (d - b) / 2 at the midpoint (b + d) / 2 and falls back to
    #    0 at d. Layer 1 is the highest tent at each filtration value, layer 2 the
    #    second highest, and so on; n_layers=5 keeps the top five. Example: one H1
    #    class (10, 16) is a tent peaking at 13 minutes with height 3.
    #    Giotto samples each dimension on its own grid of 100 points (its default),
    #    running from that dimension's smallest birth to its largest death, so the
    #    x axis is the grid position (0-99), not time. Giotto stacks the layers
    #    dimension by dimension, so with H0 and H1 the plot has 10 curves:
    #    "Layer 1"-"Layer 5" are H0 and "Layer 6"-"Layer 10" are H1.
    #    Only the plot is saved, not the landscape values.
    #    Known issue (a): when an H0 class never dies, the H0 grid would run to
    #    +inf, so the H0 landscape cannot be computed and numpy/gtda print
    #    RuntimeWarnings.
    try:
        landscape = PersistenceLandscape(n_layers=5)
        values = landscape.fit_transform(scaled[np.newaxis, :, :])[0]
        plt.figure(figsize=(8, 6))
        for layer_index in range(values.shape[0]):
            plt.plot(values[layer_index], label=f"Layer {layer_index + 1}")
        if values.shape[0]:
            plt.legend()
        plt.xlabel("Grid position")
        plt.ylabel("Persistence landscape")
        plt.tight_layout()
        plt.savefig(output / "persistence_landscape.png", dpi=300)
        plt.close()
    except Exception as exc:
        # A failed landscape should not lose the PH results above: record the
        # problem and carry on. The landscape plot is not needed for the result
        # to count as complete (see _result_complete).
        plt.close("all")
        warnings.append(f"persistence landscape could not be written: {exc}")
        # Keep a record of any failure for later review.
        (output / "persistence_landscape_error.txt").write_text(str(exc), encoding="utf-8")

    #    Statistics for each requested dimension (see _dimension_statistics).
    statistics = _dimension_statistics(scaled, int(max_homology_dim))
    write_json(output / "summary.json", {"statistics": statistics, "warnings": warnings})
    #    Written last: result_metadata.json with processing_status "complete" is
    #    what marks this folder as finished.
    result_metadata = {
        **json_safe(dict(metadata)),
        "processing_status": "complete",
        "completed_utc": utc_now_iso(),
        "n_points": int(prepared.shape[0]),
        "requested_max_homology_dim": int(max_homology_dim),
        "actual_max_homology_dim": max_dim,
        "input_time_unit": normalize_time_unit(input_time_unit),
        "output_time_unit": normalize_time_unit(output_time_unit),
        "warnings": warnings,
        "statistics": statistics,
    }
    write_json(output / "result_metadata.json", result_metadata)
    return result_metadata


# =============================================================================
# Running the saved plan, with resume
# =============================================================================

def _result_complete(
    output_dir: Path,
    *,
    plan_hash: str,
    matrix_sha256: str,
    membership_hash: str,
) -> bool:
    """Return True if this matrix/scenario result is complete and belongs to this plan.

    Complete means: the six required files exist (persistence_diagram.npy, .csv
    and .png, birth_death.npz, summary.json and result_metadata.json);
    result_metadata.json says ''processing_status'' "complete"; and it records
    the same plan_hash, the same checksum of the Step 4 matrix file and the same
    membership hash as now. The landscape plot is not required, because it can
    fail (known issue (a)). Anything else returns False, and the combination is
    computed again.
    """
    required = [
        output_dir / "persistence_diagram.npy",
        output_dir / "persistence_diagram.csv",
        output_dir / "persistence_diagram.png",
        output_dir / "birth_death.npz",
        output_dir / "summary.json",
        output_dir / "result_metadata.json",
    ]
    if not all(path.exists() for path in required):
        return False
    metadata = read_json(output_dir / "result_metadata.json")
    return (
        metadata.get("processing_status") == "complete"
        and metadata.get("plan_hash") == plan_hash
        and metadata.get("source_matrix_sha256") == matrix_sha256
        and metadata.get("membership_hash") == membership_hash
    )


def _coerce_included_boolean(series: pd.Series) -> pd.Series:
    """Read the ''included'' column of a membership file as True/False.

    A CSV file stores True/False as text. pandas normally turns it back into
    True/False, but when it doesn't, a plain conversion would treat any
    non-empty text, including "False", as True and quietly keep removed
    pharmacies. Accepted text (any capitalization, spaces ignored): true/false,
    1/0 and yes/no. Any other value stops the run. A blank cell counts as False
    (removed).
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    valid = {"true", "false", "1", "0", "yes", "no"}
    unexpected = sorted(set(normalized.dropna()) - valid)
    if unexpected:
        raise ValueError(f"Membership file contains invalid included values: {unexpected}")
    return normalized.isin({"true", "1", "yes"})


def execute_saved_scenario_plan(
    *,
    paths: CityRunPaths,
    ph_runner: Callable[..., dict[str, Any]] = process_single_matrix,
) -> dict[str, Any]:
    """Run, or resume, PH for every selected matrix and every scenario in the saved plan.

    Step 0 calls this once per city, after the plans for all cities are saved.
    The numbered comments below match stages 8-11 at the top of this file.

    For each matrix type in the plan (outer loop) and each row of
    scenario_index.csv (inner loop): read the scenario's membership file; skip
    the pair if its result is already complete; otherwise cut the matrix down to
    the kept pharmacies and run ph_runner on it, writing to
    step05_tda/<matrix_type>/<scenario_key>/. ph_runner is process_single_matrix,
    unless a test passes a different function.

    Failures: a PH run that fails is recorded and the loop moves on, so one bad
    scenario does not hold up the others. At the end,
    step05_execution_summary.json lists the counts and the failures, and Step 5
    stops with an error if anything failed. Running again with the same settings
    then redoes only the failed or missing pairs. Problems with the inputs
    themselves (wrong run tag, a missing membership file, a matrix of the wrong
    size) stop the run at once.

    Returns the summary that is saved as step05_execution_summary.json.
    """
    # 8. Read the saved plan and check that it belongs to this run.
    plan = read_json(paths.scenario_plan_json)
    validate_tag = plan.get(ANALYSIS_TAG_COLUMN)
    if validate_tag != paths.analysis_run_tag:
        raise ValueError(
            f"Saved Step 5 plan belongs to {validate_tag!r}, not {paths.analysis_run_tag!r}."
        )
    #    The scenario index lists every scenario; the manifest gives the pharmacy order.
    scenario_index = pd.read_csv(paths.scenario_index_csv)
    manifest = pd.read_csv(
        paths.step03_manifest,
        dtype={"pharmacy_id": "string", "NPI": "string", "google_place_id": "string"},
    )
    validate_analysis_tag(manifest, paths.analysis_run_tag, str(paths.step03_manifest))
    ids = validate_pharmacy_ids(manifest, str(paths.step03_manifest))

    # Counters and failure list for step05_execution_summary.json.
    successful = skipped = failed = 0
    failure_rows: list[dict[str, Any]] = []
    # PH settings come from the saved plan, not from Step 0 directly.
    ph_settings = plan["ph_settings"]
    plan_hash = plan["plan_hash"]

    for matrix_type in plan["selected_matrix_types"]:
        # Load the full Step 4 matrix once per matrix type, and label its rows and
        # columns with pharmacy IDs so each scenario can pick pharmacies by ID.
        matrix_type = normalize_matrix_type(matrix_type)
        matrix_path = paths.derived_matrix_path(matrix_type)
        matrix = np.load(matrix_path, allow_pickle=False)
        if matrix.shape != (len(ids), len(ids)):
            raise ValueError(
                f"Step 4 matrix {matrix_type} shape {matrix.shape} does not match manifest."
            )
        # Checksum of the matrix file, recorded with each result so that a changed
        # matrix is noticed on the next run (see _result_complete).
        matrix_sha = file_sha256(matrix_path)
        full_df = pd.DataFrame(matrix, index=ids, columns=ids)

        for _, scenario_row in scenario_index.iterrows():
            # 9. Read this scenario's membership file and check it matches the
            #    manifest: same run tag, same pharmacies, same order.
            scenario_key = str(scenario_row["scenario_key"])
            membership_path = paths.scenario_membership_dir / f"{scenario_key}.csv"
            membership = pd.read_csv(
                membership_path,
                dtype={"pharmacy_id": "string", ANALYSIS_TAG_COLUMN: "string"},
            )
            validate_analysis_tag(membership, paths.analysis_run_tag, str(membership_path))
            membership_ids = validate_pharmacy_ids(membership, str(membership_path))
            if membership_ids != ids:
                raise ValueError(f"Membership row order differs from the Step 3 manifest: {membership_path}")
            # IDs of the pharmacies this scenario keeps, in manifest order.
            included_mask = _coerce_included_boolean(membership["included"])
            included = membership.loc[included_mask, "pharmacy_id"].astype(str).tolist()
            # Taken from scenario_index.csv; it is not recalculated from the file.
            membership_hash = str(scenario_row["membership_hash"])
            output_dir = paths.tda_matrix_dir(matrix_type) / scenario_key

            #    Skip the pair if its result is already complete for this plan.
            if _result_complete(
                output_dir,
                plan_hash=plan_hash,
                matrix_sha256=matrix_sha,
                membership_hash=membership_hash,
            ):
                print(f"[SKIP] Step 5 complete: {paths.city_state} | {matrix_type} | {scenario_key}")
                skipped += 1
                continue

            # 10. Cut the matrix down to the kept pharmacies' rows and columns, e.g.
            #     removing 1 pharmacy of 100 gives a 99 x 99 matrix in the same order.
            subset = full_df.loc[included, included].to_numpy(dtype=float, copy=True)
            #     Saved in result_metadata.json. plan_hash, source_matrix_sha256 and
            #     membership_hash are what _result_complete checks on the next run.
            metadata = {
                "pipeline_step": 5,
                ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
                "city": paths.city,
                "state": paths.state.upper(),
                "city_state": paths.city_state,
                "plan_hash": plan_hash,
                "matrix_type": matrix_type,
                "scenario_name": scenario_row["scenario_name"],
                "scenario_key": scenario_key,
                "scenario_type": scenario_row["scenario_type"],
                "scenario_scope": scenario_row["scenario_scope"],
                "replicate_number": scenario_row.get("replicate_number"),
                "source_matrix": str(matrix_path.resolve()),
                "source_matrix_sha256": matrix_sha,
                "source_manifest": str(paths.step03_manifest.relative_to(paths.city_dir)),
                "membership_file": str(membership_path.resolve()),
                "membership_hash": membership_hash,
                "included_pharmacy_count": len(included),
                "excluded_pharmacy_count": int((~included_mask).sum()),
            }
            #     Run PH. A failure is recorded and the loop moves on to the next pair.
            try:
                print(f"[RUN] Step 5: {paths.city_state} | {matrix_type} | {scenario_key}")
                ph_runner(
                    subset,
                    output_dir,
                    metadata=metadata,
                    input_time_unit=ph_settings["input_time_unit"],
                    output_time_unit=ph_settings["output_time_unit"],
                    max_homology_dim=ph_settings["max_homology_dim"],
                )
                successful += 1
            except Exception as exc:
                failed += 1
                failure_rows.append(
                    {
                        "matrix_type": matrix_type,
                        "scenario_key": scenario_key,
                        "error": str(exc),
                    }
                )
                print(f"[ERROR] Step 5 failed: {matrix_type} | {scenario_key}: {exc}")

    # 11. Save the summary of this run (replacing the previous one), then stop with
    #     an error if any pair failed.
    summary = {
        "pipeline_step": 5,
        ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
        "city_state": paths.city_state,
        "plan_hash": plan_hash,
        "successful": successful,
        "skipped_complete": skipped,
        "failed": failed,
        "failures": failure_rows,
        "updated_utc": utc_now_iso(),
    }
    write_json(paths.step05_dir / "step05_execution_summary.json", summary)
    if failures := failure_rows:
        raise RuntimeError(
            f"Step 5 completed with {len(failures)} failed matrix/scenario combinations. "
            f"See {paths.step05_dir / 'step05_execution_summary.json'}."
        )
    return summary


# Step 5 cannot be run on its own: Step 0 must first build and check the plans
# for all cities together.
if __name__ == "__main__":
    raise SystemExit(
        "Run Step 5 through step00_main_process_flow_260801.py so the complete "
        "multi-city scenario plan is validated before persistent homology begins."
    )
