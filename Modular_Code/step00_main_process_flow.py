# -*- coding: utf-8 -*-
"""Step 0 - the main file: the settings you edit, and main(), which runs the steps.

Edit the settings in sections 1-7 below, then run this file using whatever python IDE you like (I use Spyder). ''main'' runs the steps listed in ''RUN_STEPS'' for every city in
''CITY_STATE_LIST'':

    Step 1  pharmacies (NPI Registry + Google Places)
    Step 2  census tract data for each pharmacy (ACS); pharmacies far from a city’s center are removed
    Step 3  raw one-way walk and drive travel times between pharmacies (OSMnx)
    Step 4  calculate the six modified distance matrices using the raw matrices from Step 3 (walk, drive, multimodal; dtilde and d)
    Step 5  persistent homology (Giotto-TDA) for the baseline and removal scenarios
    Step 6  maps showing pharmacies in each city 
    Step 7  each scenario's persistence diagram compared with the baseline full set of pharmacies (using the GUDHI package)

ANALYSIS RUNS AND ''ANALYSIS_RUN_TAG''
-------------------------------------
''ANALYSIS_RUN_TAG'' is the one name that identifies a run. Every city in a run
shares it, and each city's outputs go in their own folder:

    <ANALYSIS_OUTPUT_ROOT>/analysis_runs/<ANALYSIS_RUN_TAG>/<CityState>/stepNN_.../

The tag is also part of every pharmacy ID (for example
MilwaukeeWI_test_261011_P0001, where test_261011 is the run tag), so outputs from different runs can't be mixed.

If there is no folder for the specified run tag yet, a new run starts. If the folder exists, the
run continues where it left off. For each requested step and city:

- outputs that are complete and pass their checks are skipped;
- outputs that don't exist yet are created;
- outputs that are partial, or don't match this run, stop the run;
- nothing is ever deleted or overwritten, so you can stop at any time and
  continue later with the same tag.

Missing earlier steps are not run automatically. For example, requesting only
{3, 4} when Step 2 has never run stops the run before anything is done (see
''RUN_STEPS'' below). To start over, or to change a Step 5 plan that is already
saved (see below), choose a new tag.

HOW TO USE ''RUN_STEPS''
------------------------
''RUN_STEPS'' can hold any of the step numbers 1 to 7. Each step needs the output of one
earlier step:

    Step 2 needs Step 1     Step 3 needs Step 2     Step 4 needs Step 3
    Step 5 needs Step 4     Step 6 needs Step 2     Step 7 needs Step 5's plan

Requesting only a step that does not have the necessary output from a prior step stops the run. For example,
{5, 6, 7} works once Steps 1-4 are complete for every city.

GENERAL OVERVIEW OF HOW A RUN FLOWS 
-------------------------------------------------------
1. Validation checks, before anything runs: the settings (''_validate_main_settings''), a
   printed run summary (''_print_run_summary''), then the saved files,
   prerequisites and API keys (''_preflight''). If ''DRY_RUN'' = True (mainly used for testing log comments), the run
   ends here.
2. Steps 1-4, city by city: run every requested step among 1-4 for the first city,
   then the same for the next city, and so on.
3. Step 5, across all cities. It first builds and checks every city's scenario
   plan (which helps define which pharmacies each scenario removes) and saves the plan; only
   then it runs persistent homology calculations, city by city. A bad scenario setting
   stops the run before a plan is saved or any homology is computed.
4. Step 6, across all cities: maps.
5. Step 7, across all cities: compares each scenario's persistence diagram with
   the baseline.

After each step, the city's city_metadata.json records where the outputs are.
That file is only an index: whether a step is complete is always rechecked from
the output files themselves.

EXPLICIT PHARMACY-ID SCENARIOS NEED TWO EXECUTIONS
--------------------------------------------------
The ''exclude_ids'' scenario removes specific pharmacies you name by ''pharmacy_id''. IDs are
assigned at the end of Step 1, and Step 2 possibly drops some pharmacies (e.g., no
census tract match, a failed geocoder request, or too far from the city center).
So you only know the final pharmacies, and their IDs, once Step 3 has written its
manifest. To run this scenario successfully:

1. Run Steps 1-3 (''RUN_STEPS'' = {1, 2, 3}).
2. Open step03_raw_travel/manifest.csv for each city.
3. Copy the ''pharmacy_id'' values you want to exclude into ''CITY_SCENARIOS'' below and set
   that scenario's "enabled" variable to True.
4. Keep the SAME ''ANALYSIS_RUN_TAG'' and run Steps 4-7.

The first time Step 5 runs, it saves each city's scenario plan, and from then on
that plan is locked. A later Step 5 run continues any unfinished homology work
only while the settings still match the saved plan exactly; any difference stops
Step 5.

CHANGING SETTINGS PART-WAY THROUGH A RUN
----------------------------------------
- Steps 1-4 settings (sections 2-4): a city's completed outputs are kept and
  skipped. They are not checked against the new value, which only affects cities
  and steps that haven't run yet. Therefore, to have every city use the new value, use a
  new tag.
- Steps 5-7 matrix, homology and scenario settings (sections 5-6): saved in the
  Step 5 plan. Changing them after the plan is saved stops Step 5; go back to the
  saved values or use a new tag.
- Step 6 map settings (section 7): saved with the maps. Changing them after the
  maps exist stops Step 6; go back to the saved values or use a new tag.
- Safe to change at any time: ''RUN_STEPS'', ''DRY_RUN'', the API keys,
  ''GOOGLE_REQUEST_DELAY_SECONDS'', ''STEP3_DOWNLOAD_ATTEMPTS'',
  ''STEP3_RETRY_WAIT_SECONDS'' and ''MAKE_SCENARIO_MAPS''.
"""

from __future__ import annotations

import os
from datetime import date  # only for the date.today() option of ANALYSIS_RUN_TAG
from pathlib import Path
# Folder that holds this file and the other pipeline modules (pipeline_common.py,
# step01_..., and so on). Changing the working directory to it lets the imports
# below find those modules when this file is run from Spyder, and makes OSMnx put
# its download cache (./cache) in the code folder. If the folder doesn't exist
# (for example on another computer), nothing changes, and the imports only work
# if Python can already find the modules (e.g. when run from the code folder). This may need to be updated once I start using Git more consistently
DEFAULT_CODE_DIR = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\modular_code\testing\Pharmacy_Refusals_Pipeline_260801\pharmacy_pipeline_260801"
if os.path.isdir(DEFAULT_CODE_DIR):
    os.chdir(DEFAULT_CODE_DIR)

# Shared helpers: the folder layout for one city (CityRunPaths), the matrix type
# names, and the city_metadata.json index.
from pipeline_common import (
    ALL_DERIVED_MATRIX_TYPES,
    CityRunPaths,
    resolve_matrix_types,
    update_city_metadata,
)
# Checks on saved outputs. Each validate_stepN returns "complete" or "absent" (or
# "skipped" for Step 1) and stops the run if the outputs are partial or don't match.
from pipeline_validation import (
    validate_requested_step_prerequisites,
    validate_step1,
    validate_step2,
    validate_step3,
    validate_step4,
)


# =============================================================================
# 1. GENERAL RUN SETTINGS
# =============================================================================

# Folder under which the main file creates analysis_runs/<ANALYSIS_RUN_TAG>/
# <CityState>/ for all outputs.
ANALYSIS_OUTPUT_ROOT = Path(
    r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\pipeline_results"
)

# Name of this analysis run (see the top of this file). Reuse the same tag to
# continue a run, on any day. Use a new tag for a new data extraction, or for a
# Step 5 plan with different matrix, homology or scenario settings. Today's date
# (date.today().isoformat(), e.g. "2026-10-18") is a convenient default, but any
# name works. Use only letters, digits, ".", "_" and "-": other characters are
# replaced by "-" in folder names and pharmacy IDs.
ANALYSIS_RUN_TAG = 'test_261011' # or date.today().isoformat()
# Example override:
# ANALYSIS_RUN_TAG = "pharmacy_baseline_v1"

# Which steps to run: any combination of 1-7, e.g. {1, 2, 3} or {5, 6, 7}. They
# always run in number order. A step whose required earlier step is neither
# complete nor in this set stops the run before anything is processed (see HOW TO
# USE RUN_STEPS at the top of this file).
RUN_STEPS = {1,2,3,4,5,6,7}

# Cities to process, as (city name, two-letter state code). The city name is used
# to search the NPI Registry and to find the city boundary, so spell it the way
# those services know it. All cities share ANALYSIS_RUN_TAG, but each gets its own
# folder, city_metadata.json, matrices, scenarios and results, named by its city
# key: the city name without spaces or punctuation plus the state, e.g.
# ("Las Vegas", "NV") -> LasVegasNV.
#
# Limitation (Step 1): the NPI Registry returns at most 1,200 records per city,
# sorted by organization name, so a very large city is cut off alphabetically.
# Step 1 writes WARNING_NPI_CEILING.txt when this happens.
CITY_STATE_LIST = [
    # ("Brooklyn", "NY"),
    ("Milwaukee", "WI"),
    ("Las Vegas", "NV"),
    ("Atlanta", "GA"),
    ("Pittsburgh", "PA"),
    ("Albuquerque", "NM"),
    ("Austin", "TX"),
    # ("Kansas City", "MO"),  # Step 3 could not complete for this city; cause unknown.
]

# True = check the settings and the prerequisites (the same checks main() makes
# before any step runs), print the run summary, then stop without running any
# step. A dry run creates no files or folders. It does not check whether a saved
# Step 5 plan still matches the settings; Step 5 itself checks that.
DRY_RUN = False


# =============================================================================
# 2. API SETTINGS
# =============================================================================

# API keys are read from Windows environment variables, so they never appear in
# this file. To add one: type "edit environment variables" in the Windows search
# bar, open "Edit environment variables for your account", choose New, and enter
# the name below with your key as the value. Restart Spyder afterwards: it only
# sees variables that existed when it started. A missing key becomes "" (blank),
# and spaces around a pasted key are removed.
#
# GOOGLE_PLACES_API_KEY (Step 1) and CENSUS_API_KEY (Step 2) are read here. Each is
# needed only when its step actually has to run for at least one city; complete
# outputs are skipped without it. _preflight checks this before anything runs.
#
# A third key, CARTO_BASEMAP_API_KEY, is used by Step 6 for the light-grey CARTO
# basemap under every map. It is a free CARTO basemap key for non-commercial use
# (request one at carto.com/basemaps/apikey). Step 6 reads it directly from the
# Windows environment variable of that name, so no setting is needed in this file.
# Set it like the other keys and restart Spyder after setting it. It is not
# checked before the run starts: Step 6 stops with an error when it first needs a
# basemap and the key is missing.
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "").strip()

# Step 1: which of a pharmacy's NPI addresses must be in the chosen city and
# state. The NPI Registry applies this filter itself, before sending the results.
# Allowed values: "PRIMARY" (primary practice location; the default, so pharmacies
# are found by where they operate, not where their mail goes), "SECONDARY" (a
# secondary practice location), "LOCATION" (practice location) or "MAILING"
# (mailing address). Step 1 always asks for organizations (NPI-2), never for
# individual pharmacists (NPI-1).
NPI_ADDRESS_PURPOSE = "PRIMARY"  # LOCATION | MAILING | PRIMARY | SECONDARY
# Step 1: pause after each Google Places request, in seconds. This buffer makes
# rate-limit failures less likely, which matters because Step 1 drops any pharmacy
# whose request fails (it keeps only results marked OPERATIONAL). The cost is about
# 1 second per pharmacy, e.g. 5 minutes for 300 pharmacies. Step 1's own default
# is 0.5.
GOOGLE_REQUEST_DELAY_SECONDS = 1.0 # buffer so requests are less likely to fail


# =============================================================================
# 3. STEP 2 SETTINGS
# =============================================================================
# Changing a value here after a city's Step 2 is complete has no effect on that
# city (see CHANGING SETTINGS at the top of this file).

# Step 2 removes pharmacies farther than this from the city centre, in miles,
# measured as a straight line over the Earth's surface (great-circle distance).
# The centre is the centroid of the city boundary that OSMnx's geocoder returns.
# This keeps the analysis to the city and its edges: an NPI address can name the
# city while Google places the pharmacy far outside it. Removed pharmacies are
# listed in far_pharmacies.csv and take no further part in the run. If the centre
# can't be found, Step 2 prints a warning and removes no pharmacies.
MAX_DISTANCE_MILES = 20.0
# Step 2: the American Community Survey (ACS) 5-year release to use, named by its
# last year (2022 = the 2018-2022 estimates). It supplies tract population,
# household vehicles and poverty counts. 2022 was the most recent release the last
# time I checked, but that was a while ago. (Step 2's own default is 2024; this
# value is the one Step 0 passes.)
ACS_YEAR = 2022


# =============================================================================
# 4. STEPS 3-4 SETTINGS
# =============================================================================

# Step 3 always builds both raw one-way matrices (walk and drive), because Step 4
# always makes walk, drive and multimodal matrices from them. The raw Step 3
# matrices are inputs to Step 4 only; Steps 5-7 use Step 4's matrices.
#
# Changing STEP3_BBOX_BUFFER_DEGREES, WALK_SPEED_MPS or DRIVE_PARKING_PENALTY_MINUTES
# after a city's Step 3 (or Step 4) is complete has no effect on that city (see
# CHANGING SETTINGS at the top of this file).
#
# Known issue (Step 3): each pharmacy is attached to the nearest point of the drive
# network, which can be a one-way dead end. The pharmacy is then unreachable by car
# in one direction, its drive times are NaN (4 pharmacies in test_261011), and its
# drive and multimodal distances are missing downstream (a fix has been proposed).

# Step 3 downloads the street network for the smallest box around all of a city's
# pharmacies, widened by this many degrees on every side, so routes between
# pharmacies near the edge can use streets just outside the box. 0.02 degrees is
# about 2.2 km (1.4 miles) north-south, and about 1.6 km east-west at Milwaukee's
# latitude.
STEP3_BBOX_BUFFER_DEGREES = 0.02
# Walking speed in metres per second. Step 3 turns each shortest walking route
# into a time: walk time (s) = route length (m) / WALK_SPEED_MPS. 1.42 m/s (about
# 3.2 mph or 5.1 km/h) is a typical adult walking pace; at that speed 1 km takes
# 704 s, just under 12 minutes.
WALK_SPEED_MPS = 1.42

# Step 3 downloads the street networks from the public Overpass (OpenStreetMap)
# server, which sometimes fails to respond. STEP3_DOWNLOAD_ATTEMPTS is the total
# number of tries for each download (1 turns retries off). STEP3_RETRY_WAIT_SECONDS
# is the wait before the first retry; it doubles after each subsequent failure, so
# 4 attempts with a 60-second wait pause for 60, 120 and 240 seconds (7 minutes
# at most). If every attempt fails, Step 3 stops without saving anything for that
# city, and the run can be restarted with the same ANALYSIS_RUN_TAG. These two
# settings only change how hard Step 3 tries, not its results, so they can be
# changed at any time.
STEP3_DOWNLOAD_ATTEMPTS = 4
STEP3_RETRY_WAIT_SECONDS = 60

# Step 4: minutes added to every one-way drive (each pair that has a drive time)
# for parking and for walking to and from the car, before the round trips are
# formed. A round trip has two legs, so it gets twice this value: 2.5 adds 5
# minutes to every drive round trip. Since the penalty applies to BOTH the trip
# there and the trip back, set it to half of the intended total. Step 4 converts
# it to seconds (2.5 min = 150 s). Multimodal times include it too, because they
# are built from these drive times.
DRIVE_PARKING_PENALTY_MINUTES = 2.5


# =============================================================================
# 5. STEPS 5-7 MATRIX AND PH SETTINGS
# =============================================================================

# Step 4 always writes all six matrix types listed in ALL_DERIVED_MATRIX_TYPES
# (walk_d, walk_dtilde, drive_d, drive_dtilde, multimodal_d, multimodal_dtilde).
# The next two settings choose which of them Steps 5-7 analyze: every travel mode
# is paired with every matrix form, so ["walk", "drive", "multimodal"] with ["d"]
# gives walk_d, drive_d and multimodal_d. Each extra matrix type adds a full set
# of Step 5 runs (one per scenario).
#
# These two settings, and the three TDA_ settings below, are saved in the Step 5
# plan. Changing them after Step 5 has saved its plan stops Step 5; go back to the
# saved values or use a new ANALYSIS_RUN_TAG.

# Any non-empty subset of ["walk", "drive", "multimodal"].
ANALYSIS_TRAVEL_MODES = ["walk", "drive", "multimodal"]
# "d" = the round trips in both directions combined into one symmetric value,
# weighted by tract population (Hickok et al. eq. 3.1; see Step 4).
ANALYSIS_MATRIX_FORMS = ["d"]
# Advanced option: add "dtilde" to also analyze the round-trip matrices before
# population weighting. For walk and drive these equal d anyway (except pairs where
# both tracts have population 0), so only multimodal differs. Step 5 makes a dtilde
# matrix symmetric by keeping the smaller of the two directions for each pair, and
# records this only in its summary.json.
# ANALYSIS_MATRIX_FORMS = ["d", "dtilde"]

# Unit of the Step 3 and Step 4 matrices. Step 3 always writes seconds, so leave
# this as "seconds": Step 4 uses it to convert the parking penalty, and Step 5 to
# convert to TDA_OUTPUT_TIME_UNIT, so any other value would scale times wrongly.
TDA_INPUT_TIME_UNIT = "seconds" # TDA input time is always seconds, DO NOT CHANGE THIS
# Unit of Step 5's persistence diagrams (birth and death values), and therefore of
# Step 7's distances: "seconds", "minutes" or "hours". Times are converted after
# the homology is computed, so this choice only rescales the numbers.
TDA_OUTPUT_TIME_UNIT = "minutes" # This can be changed to "seconds" or "hours" if desired
# Highest homology dimension Step 5 computes. 1 gives H0 (separate groups of
# pharmacies) and H1 (loops, i.e. "holes" in coverage), which is all this analysis
# needs. 0 would give H0 only; 2 would add voids, which take much longer to
# compute and mean little for points on a map.
TDA_MAX_HOMOLOGY_DIM = 1 # This is the maximum dimension we need to analyze


# =============================================================================
# 6. STEP 5 REMOVAL-SCENARIO SETTINGS
# =============================================================================

# A scenario removes some pharmacies. Step 5 then computes persistent homology for
# the pharmacies that remain, and Step 7 compares the result with the baseline.
# Step 5 always analyzes the baseline (all pharmacies, scenario name "full") by
# itself: don't list it, and don't use the name "full".
#
# Each scenario is a dictionary with:
#   "name"     a unique name (names that differ only in capitals count as the
#              same); it also becomes the scenario's folder name
#   "type"     one of the four types below
#   "enabled"  True to run it; False to ignore it completely
# plus the settings its type needs:
#   "leave_one_out"     one scenario per pharmacy, each removing just that one
#                       pharmacy. GLOBAL_SCENARIOS only. That is one Step 5 run per
#                       pharmacy and matrix type, so it is slow for large cities.
#   "random_exclusion"  removes "proportion" (0 to 1) of the pharmacies at random,
#                       n * proportion rounded to a whole number (.5 rounds up, and
#                       at least 1 if proportion > 0). "replicates" is how many
#                       different random sets to draw, and "seed" makes the draws
#                       repeatable. Example: 10% of 85 pharmacies = 8.5 -> 9
#                       removed. No two replicates remove exactly the same set, but
#                       one pharmacy can be removed in several replicates.
#                       A proportion of 0 is skipped (it would equal the baseline).
#   "chain_exclusion"   removes every pharmacy whose OrganizationName contains any
#                       of the "organization_name_terms", ignoring capitals; e.g.
#                       "CVS" matches "CVS PHARMACY #1234".
#   "exclude_ids"       removes the pharmacies listed in "pharmacy_ids".
#                       CITY_SCENARIOS only (see below).
#
# GLOBAL_SCENARIOS apply to every city. If a global chain scenario (e.g. removing
# all CVS pharmacies) matches no pharmacy in one of the selected cities, that is an
# error, and Step 5 stops before any plan is saved.
#
# Changing an enabled scenario, or switching a scenario on or off, after Step 5 has
# saved its plan stops Step 5; go back to the saved values or use a new
# ANALYSIS_RUN_TAG.
#
# Limitation (Step 7): a scenario that removes k pharmacies has k fewer H0 classes
# than the baseline, so at least k baseline classes are left without a partner.
# H0 distances therefore partly reflect how many pharmacies were removed, not which
# ones. For H0, compare a targeted removal with a random removal of the same size.
# H1 is largely unaffected.

GLOBAL_SCENARIOS = [
    {
        "name": "leave_one_out",
        "type": "leave_one_out",
        "enabled": False,
    },
    {
        # 10% of each city's pharmacies, 3 different random sets, repeatable seed.
        "name": "remove_10_percent_random",
        "type": "random_exclusion",
        "proportion": 0.10,
        "replicates": 3,
        "seed": 12345,
        "enabled": True,
    },
    {
        "name": "remove_all_CVS",
        "type": "chain_exclusion",
        "organization_name_terms": ["CVS"],
        "enabled": False,
    },
]

# CITY_SCENARIOS add scenarios for one city only. Each key is a city key (city name
# without spaces or punctuation plus the state, e.g. "MilwaukeeWI") and must be a
# city in CITY_STATE_LIST. Scenario names must be unique across the global and
# city scenarios together.
#
# EXPLICIT-ID LIMITATION:
# Don't fill in an exclude_ids scenario until Step 3 has written that city's
# manifest (step03_raw_travel/manifest.csv). Copy the IDs from the manifest of this
# same ANALYSIS_RUN_TAG (the tag is part of each ID). Unknown or duplicate IDs stop
# Step 5, and so does an enabled exclude_ids scenario with an empty list (switch it
# off instead). See the top of this file for the two-execution routine.
# Example of another city key: "BrooklynNY"
CITY_SCENARIOS = {
    "MilwaukeeWI": [
        {
            "name": "remove_selected_pharmacies",
            "type": "exclude_ids",
            "pharmacy_ids": [
                # ID format: <CityKey>_<ANALYSIS_RUN_TAG>_P<number>. These two
                # examples come from an earlier Brooklyn run.
                # "BrooklynNY_pharmacy_baseline_v1_P0001",
                # "BrooklynNY_pharmacy_baseline_v1_P0008",
            ],
            "enabled": False,
        },
    ],
}


# =============================================================================
# 7. STEP 6 MAP SETTINGS
# =============================================================================

# Every Step 6 map is drawn on the CARTO light-grey basemap, so it needs the
# CARTO_BASEMAP_API_KEY environment variable (see section 2) and an internet
# connection. Step 6 runs whenever it is requested: map files that already exist
# are skipped one by one, and missing ones are drawn.
#
# The settings from MAKE_CONTEXTILY_MAP to UNION_GIF_SHOW_EDGES are saved with the
# maps (step06_mapping/mapping_metadata.json). Changing any of them after a city's
# maps exist, even switching a map off, stops Step 6; go back to the saved values
# or use a new ANALYSIS_RUN_TAG. MAP_POINT_SIZE is also saved with the scenario
# maps.

# Static PNG of all of the city's pharmacies (from Step 2's tract_data.csv) on the
# basemap: step06_mapping/pharmacies_contextily.png.
MAKE_CONTEXTILY_MAP = True
# Animated GIF in which a disk of radius r = epsilon / 2 grows around every
# pharmacy (the union of balls): step06_mapping/union_balls_contextily.gif.
# An illustration only: it uses straight-line distances in Web Mercator metres,
# not travel times, and Web Mercator stretches distances by about 1/cos(latitude)
# (about 1.37x at Milwaukee).
MAKE_UNION_BALLS_GIF = True
# Colour of the pharmacy dots on the PNG and the GIF: any matplotlib colour name or
# hex code. (Scenario maps use their own blue and orange.)
MAP_POINT_COLOR = "red"
# Dot size, as matplotlib marker area (in points squared). The PNG and the scenario
# maps use it as it is; the GIF uses 0.6 times it.
MAP_POINT_SIZE = 30.0
# Number of GIF frames; epsilon rises in equal steps from 0 to its largest value.
UNION_GIF_FRAMES = 140
# GIF playback speed in frames per second: 140 frames at 3 fps last about 47 s.
UNION_GIF_FPS = 3
# Largest epsilon when UNION_GIF_EPS_MAX_M is None: this quantile of the straight-
# line distances between all pairs of pharmacies. 0.70 ends the animation once 70%
# of the pairs are within epsilon of each other.
UNION_GIF_EPS_QUANTILE = 0.70
# Largest epsilon in (Web Mercator) metres, used instead of the quantile above.
# None = use the quantile.
UNION_GIF_EPS_MAX_M = None
# Draw a line between two pharmacies once epsilon reaches the distance between
# them, i.e. once their disks touch.
UNION_GIF_SHOW_EDGES = True
# One static map per removal scenario: pharmacies the scenario keeps are blue and
# those it removes are orange. The maps are drawn from the saved Step 5 scenario
# plan and saved in step06_mapping/scenario_maps/. The full baseline (the same as
# the main map) and leave-one-out scenarios (one map per pharmacy) are not mapped.
# Step 5 must have saved its plan first (include 5 in RUN_STEPS the first time
# Step 6 runs with this switched on). This switch is not saved with the maps, so
# it can be turned on or off at any time.
MAKE_SCENARIO_MAPS = True


# =============================================================================
# Checks and run summary, before any step runs
# =============================================================================

# Not a setting: the step numbers RUN_STEPS may contain (1 to 7).
VALID_STEPS = set(range(1, 8))


def _paths_for_all_cities() -> list[CityRunPaths]:
    """Return one ''CityRunPaths'' per city in ''CITY_STATE_LIST'', in list order.

    ''CityRunPaths'' (in pipeline_common) knows every file and folder path for one
    city in this run; for example ''paths.step03_manifest'' is
    <ANALYSIS_OUTPUT_ROOT>/analysis_runs/<tag>/MilwaukeeWI/step03_raw_travel/manifest.csv.
    Nothing is created on disk here.
    """
    return [
        CityRunPaths(ANALYSIS_OUTPUT_ROOT, ANALYSIS_RUN_TAG, city, state)
        for city, state in CITY_STATE_LIST
    ]


def _validate_main_settings() -> tuple[set[int], list[str]]:
    """Check the settings in this file before anything else happens.

    Stops with a ValueError if:
    - ''RUN_STEPS'' is empty or contains a number outside 1-7;
    - ''CITY_STATE_LIST'' is empty, has a state that isn't a two-letter code, or
      lists the same city twice. Two entries with the same city key count as the
      same city, e.g. ("Las Vegas", "NV") and ("LasVegas", "NV") are both
      LasVegasNV;
    - ''STEP3_DOWNLOAD_ATTEMPTS'' is below 1 or ''STEP3_RETRY_WAIT_SECONDS'' is
      negative;
    - ''ANALYSIS_TRAVEL_MODES'' or ''ANALYSIS_MATRIX_FORMS'' is empty or has an
      unknown value;
    - ''CITY_SCENARIOS'' has a key that isn't one of the selected cities.

    Only the settings are checked here; saved files are checked in ''_preflight''.
    Returns the requested steps as a set of whole numbers, and the matrix types
    Steps 5-7 will analyze, e.g. ["walk_d", "drive_d", "multimodal_d"].
    """
    steps = {int(value) for value in RUN_STEPS}
    invalid = steps - VALID_STEPS
    if invalid:
        raise ValueError(f"RUN_STEPS contains unsupported values: {sorted(invalid)}")
    if not steps:
        raise ValueError("RUN_STEPS cannot be empty.")
    if not CITY_STATE_LIST:
        raise ValueError("CITY_STATE_LIST cannot be empty.")
    if int(STEP3_DOWNLOAD_ATTEMPTS) < 1:
        raise ValueError("STEP3_DOWNLOAD_ATTEMPTS must be at least 1.")
    if float(STEP3_RETRY_WAIT_SECONDS) < 0:
        raise ValueError("STEP3_RETRY_WAIT_SECONDS cannot be negative.")
    # The city keys (e.g. MilwaukeeWI) name the city folders, so two entries with
    # the same key would share one folder. Building a key also checks the state.
    keys = [paths.city_state for paths in _paths_for_all_cities()]
    if len(keys) != len(set(keys)):
        raise ValueError(f"CITY_STATE_LIST contains duplicate cities: {keys}")
    # Every mode paired with every form, in the order typed; stops on an unknown
    # name such as "bike" or "D-tilde".
    matrix_types = resolve_matrix_types(
        ANALYSIS_TRAVEL_MODES, ANALYSIS_MATRIX_FORMS
    )
    # A city scenario for a city that isn't being run is most likely a typo in
    # the key, so stop rather than silently ignore it.
    unknown_city_scenario_keys = set(CITY_SCENARIOS) - set(keys)
    if unknown_city_scenario_keys:
        raise ValueError(
            "CITY_SCENARIOS contains city keys not selected in CITY_STATE_LIST: "
            f"{sorted(unknown_city_scenario_keys)}"
        )
    return steps, matrix_types


def _print_run_summary(steps: set[int], matrix_types: list[str]) -> None:
    """Print what this run will do: tag, folder, new or continuing, steps, cities.

    "Run status" says "existing run / continuation" when the tag's folder already
    exists, and "new run" otherwise.

    Note: this uses ''ANALYSIS_RUN_TAG'' exactly as typed, while the real folders
    use a cleaned version (characters other than letters, digits, ".", "_" and
    "-" become "-"). For a tag made only of those characters the two are the same.
    """
    run_dir = ANALYSIS_OUTPUT_ROOT / "analysis_runs" / ANALYSIS_RUN_TAG
    existing = run_dir.exists()
    print("\n" + "=" * 78)
    print("PHARMACY REFUSALS PIPELINE")
    print(f"Analysis run tag: {ANALYSIS_RUN_TAG}")
    print(f"Run directory:    {run_dir}")
    print("Run status:       " + ("existing run / continuation" if existing else "new run"))
    print(f"Selected steps:   {sorted(steps)}")
    print(f"Selected matrices for Steps 5-7: {matrix_types}")
    print(f"Step 4 standard outputs: {list(ALL_DERIVED_MATRIX_TYPES)}")
    print("Cities: " + ", ".join(paths.city_state for paths in _paths_for_all_cities()))
    print("=" * 78 + "\n")


def _preflight(paths_list: list[CityRunPaths], steps: set[int]) -> None:
    """Check that every requested step can start, before any step runs.

    A run can take hours, so it shouldn't fail halfway for a reason that could
    have been found at the start. These checks look at the saved files and stop
    with an error if:
    1. a requested step's required earlier step (2 needs 1, 3 needs 2, 4 needs 3,
       5 needs 4, 6 needs 2, 7 needs Step 5's saved plan) is neither requested in
       this run nor already complete for every city;
    2. Step 1 has to run for at least one city but ''GOOGLE_PLACES_API_KEY'' is
       blank;
    3. Step 6 is requested with ''MAKE_SCENARIO_MAPS'' on, Step 5 is not
       requested, and some city has no saved Step 5 plan;
    4. Step 2 has to run for at least one city but ''CENSUS_API_KEY'' is blank.

    A key isn't needed when its step's outputs already exist, since they are
    skipped. The checks only read files; a partial output they come across also
    stops the run here. Not checked here: CARTO_BASEMAP_API_KEY (Step 6 checks
    it itself), and whether a saved Step 5 plan still matches the settings (Step 5
    checks that).
    """
    # 1. Prerequisites, for every city and requested step. A prerequisite that is
    #    also requested is accepted, since it will run first.
    for paths in paths_list:
        for step in sorted(steps):
            validate_requested_step_prerequisites(paths, step, steps)
    # 2. Google Places key, only needed if some city still lacks Step 1 output.
    if 1 in steps and not GOOGLE_PLACES_API_KEY:
        # Existing valid Step 1 outputs can still be skipped without a key.
        if any(validate_step1(paths) == "absent" for paths in paths_list):
            raise ValueError(
                "GOOGLE_PLACES_API_KEY is required because Step 1 is requested for a city "
                "without a complete existing Step 1 output."
            )
    # 3. Scenario maps are drawn from the saved Step 5 plan, so if Step 5 isn't
    #    part of this run to make it, the plan must already exist for every city.
    if 6 in steps and MAKE_SCENARIO_MAPS and 5 not in steps:
        missing_plans = [
            paths.city_state for paths in paths_list if not paths.scenario_plan_json.exists()
        ]
        if missing_plans:
            raise ValueError(
                "MAKE_SCENARIO_MAPS needs a saved Step 5 scenario plan, which is missing for: "
                f"{missing_plans}. Add 5 to RUN_STEPS or set MAKE_SCENARIO_MAPS = False."
            )
    # 4. Census key, only needed if some city still lacks Step 2 output.
    if 2 in steps and not CENSUS_API_KEY:
        if any(validate_step2(paths) == "absent" for paths in paths_list):
            raise ValueError(
                "CENSUS_API_KEY is required because Step 2 is requested for a city "
                "without a complete existing Step 2 output."
            )


# =============================================================================
# main(): runs the requested steps
# =============================================================================

def main() -> None:
    """Run the requested steps for every selected city.

    The numbered comments below match HOW A RUN FLOWS at the top of this file. For
    each city, every step either skips (outputs complete), runs (outputs absent)
    or stops the run (outputs partial or mismatched); see ANALYSIS RUNS at the top.
    """
    # 1. Checks before anything runs: settings, run summary, then saved files,
    #    prerequisites and API keys. Any problem stops the run here.
    steps, selected_matrix_types = _validate_main_settings()
    paths_list = _paths_for_all_cities()
    _print_run_summary(steps, selected_matrix_types)
    _preflight(paths_list, steps)
    if DRY_RUN:
        print("DRY_RUN=True: validation completed; no processing functions were called.")
        return

    # The step modules are imported here, after the checks, rather than at the top
    # of the file. A dry run then needs none of them, and Steps 5, 6 and 7 (which
    # need Giotto-TDA, the mapping packages and GUDHI) are imported only when they
    # are requested. Steps 1-4 are imported together for the city loop below.
    from step01_extract_PharmacyCityState import extract_pharmacy_data
    from step02_attach_censustract_info import attach_census_info
    from step03_build_raw_walk_and_drive_time_matrices import (
        compute_matrices_dijkstra,
    )
    from step04_build_travel_matrices import build_all_derived_matrices

    # 2. Steps 1-4, city by city --------------------------------------------------
    # Each step below follows the same pattern:
    #   validate_stepN -> "complete": print [SKIP] and move on;
    #                  -> "absent": check the step it needs, run the step, then
    #                     validate the new output the same way;
    #                  -> partial or mismatched outputs: validate_stepN stops the run.
    # Finally update_city_metadata records the output location in the city's
    # city_metadata.json (also after a skip).
    for paths in paths_list:
        paths.ensure_base_dirs()  # creates analysis_runs/<tag>/<CityState>/ if needed
        print(f"\n### {paths.city}, {paths.state.upper()} ###")

        # Step 1: pharmacies for this city.
        if 1 in steps:
            state = validate_step1(paths)
            if state == "complete":
                print(f"[SKIP] Step 1 complete: {paths.city_state}")
            elif state == "skipped":
                # An older Step 1 could mark a city as skipped (metadata plus a marker
                # file). The current Step 1 stops with an error instead, so this only
                # applies to older outputs. Step 2 stops for such a city.
                print(f"[SKIP] Step 1 city previously marked skipped: {paths.city_state}")
            else:
                extract_pharmacy_data(
                    city=paths.city,
                    state=paths.state,
                    output_folder=paths.step01_dir,
                    api_key=GOOGLE_PLACES_API_KEY,
                    analysis_run_tag=ANALYSIS_RUN_TAG,
                    npi_address_purpose=NPI_ADDRESS_PURPOSE,
                    google_request_delay_seconds=GOOGLE_REQUEST_DELAY_SECONDS,
                )
                validate_step1(paths)
            update_city_metadata(
                paths,
                {"artifacts": {"step01_pharmacy_csv": str(paths.step01_pharmacy_csv)}},
            )

        # Step 2: census tract data, and removal of far pharmacies.
        if 2 in steps:
            state = validate_step2(paths)
            if state == "complete":
                print(f"[SKIP] Step 2 complete: {paths.city_state}")
            else:
                # Step 1 may have been complete already or have just run above;
                # either way it must be complete now.
                if validate_step1(paths) != "complete":
                    raise RuntimeError(f"Step 2 requires complete Step 1 for {paths.city_state}.")
                attach_census_info(
                    paths.step01_pharmacy_csv,
                    paths.step02_dir,
                    city=paths.city,
                    state=paths.state,
                    analysis_run_tag=ANALYSIS_RUN_TAG,
                    census_api_key=CENSUS_API_KEY,
                    max_distance_miles=MAX_DISTANCE_MILES,
                    acs_year=ACS_YEAR,
                )
                validate_step2(paths)
            update_city_metadata(
                paths,
                {"artifacts": {"step02_tract_csv": str(paths.step02_tract_csv)}},
            )

        # Step 3: raw one-way walk and drive times, and the manifest that fixes the
        # pharmacy order of every matrix.
        if 3 in steps:
            state = validate_step3(paths)
            if state == "complete":
                print(f"[SKIP] Step 3 complete: {paths.city_state}")
            else:
                if validate_step2(paths) != "complete":
                    raise RuntimeError(f"Step 3 requires complete Step 2 for {paths.city_state}.")
                compute_matrices_dijkstra(
                    paths.step02_tract_csv,
                    paths.step03_dir,
                    paths.city,
                    paths.state,
                    analysis_run_tag=ANALYSIS_RUN_TAG,
                    do_drive=True,  # both modes, because Step 4 needs both
                    do_walk=True,
                    bbox_buffer_degrees=STEP3_BBOX_BUFFER_DEGREES,
                    walk_speed_mps=WALK_SPEED_MPS,
                    download_attempts=STEP3_DOWNLOAD_ATTEMPTS,
                    retry_wait_seconds=STEP3_RETRY_WAIT_SECONDS,
                )
                validate_step3(paths)
            update_city_metadata(
                paths,
                {
                    "artifacts": {
                        "step03_manifest": str(paths.step03_manifest),
                        "step03_walk_matrix": str(paths.raw_matrix_path("walk")),
                        "step03_drive_matrix": str(paths.raw_matrix_path("drive")),
                    }
                },
            )

        # Step 4: the six distance matrices.
        if 4 in steps:
            state = validate_step4(paths)
            if state == "complete":
                print(f"[SKIP] Step 4 complete: {paths.city_state}")
            else:
                if validate_step3(paths) != "complete":
                    raise RuntimeError(f"Step 4 requires complete Step 3 for {paths.city_state}.")
                build_all_derived_matrices(
                    paths.step03_manifest,
                    paths.raw_matrix_path("walk"),
                    paths.raw_matrix_path("drive"),
                    paths.step04_dir,
                    analysis_run_tag=ANALYSIS_RUN_TAG,
                    drive_parking_penalty_minutes=DRIVE_PARKING_PENALTY_MINUTES,
                    # The unit of the Step 3 matrices, so the penalty (in minutes)
                    # is converted correctly (2.5 min -> 150 s).
                    input_time_unit=TDA_INPUT_TIME_UNIT,
                )
                validate_step4(paths)
            update_city_metadata(
                paths,
                {
                    "artifacts": {"step04_directory": str(paths.step04_dir)},
                    "settings": {"step04_matrix_types": list(ALL_DERIVED_MATRIX_TYPES)},
                },
            )

    # 3. Step 5, across all cities ------------------------------------------------
    if 5 in steps:
        from step05_run_TDA import (
            build_scenario_plan_draft,
            execute_saved_scenario_plan,
            save_scenario_plan,
            validate_or_accept_existing_plan,
        )

        # 3a. Build every city's plan in memory first. A bad global chain rule, an
        #     unknown ID, a duplicate name, an impossible replicate request, or a
        #     plan that differs from one already saved stops the run here, before
        #     any new plan is saved or any homology is computed.
        drafts = {}
        existing_plan = {}
        # Saved in each plan, so changing any of these later stops Step 5.
        ph_settings = {
            "input_time_unit": TDA_INPUT_TIME_UNIT,
            "output_time_unit": TDA_OUTPUT_TIME_UNIT,
            "max_homology_dim": TDA_MAX_HOMOLOGY_DIM,
        }
        for paths in paths_list:
            if validate_step4(paths) != "complete":
                raise RuntimeError(f"Step 5 requires complete Step 4 for {paths.city_state}.")
            draft = build_scenario_plan_draft(
                paths=paths,
                selected_matrix_types=selected_matrix_types,
                global_scenarios=GLOBAL_SCENARIOS,
                city_scenarios=CITY_SCENARIOS.get(paths.city_state, []),
                ph_settings=ph_settings,
            )
            drafts[paths.city_state] = draft
            # True if an identical plan is already saved, False if none is saved
            # yet. A saved plan that differs (settings changed after it was saved)
            # stops the run.
            existing_plan[paths.city_state] = validate_or_accept_existing_plan(paths, draft)

        # 3b. Every city's plan has passed, so save the new ones.
        for paths in paths_list:
            if not existing_plan[paths.city_state]:
                save_scenario_plan(paths, drafts[paths.city_state])
        # 3c. Persistent homology for every matrix type and scenario in each saved
        #     plan. Complete results are skipped, so an interrupted Step 5 resumes.
        #     If a combination fails, the others still run, and then that city
        #     stops the run with an error (failures are listed in
        #     step05_execution_summary.json).
        for paths in paths_list:
            result = execute_saved_scenario_plan(paths=paths)
            update_city_metadata(
                paths,
                {
                    "artifacts": {"step05_directory": str(paths.step05_dir)},
                    "settings": {
                        "selected_step05_matrix_types": selected_matrix_types,
                        "step05_plan_hash": result["plan_hash"],
                    },
                },
            )

    # 4. Step 6, across all cities ------------------------------------------------
    # There is no "complete" check here: create_city_maps and create_scenario_maps
    # skip map files that already exist, draw the missing ones, and stop if the
    # saved map settings differ from the current ones.
    if 6 in steps:
        from step06_pharmacy_mapping import create_city_maps, create_scenario_maps

        for paths in paths_list:
            # The main map and the GIF use the pharmacy coordinates from Step 2.
            if validate_step2(paths) != "complete":
                raise RuntimeError(f"Step 6 requires complete Step 2 for {paths.city_state}.")
            create_city_maps(
                paths,
                make_contextily=MAKE_CONTEXTILY_MAP,
                make_unionballs_contextily_gif=MAKE_UNION_BALLS_GIF,
                point_color=MAP_POINT_COLOR,
                point_size=MAP_POINT_SIZE,
                union_gif_frames=UNION_GIF_FRAMES,
                union_gif_fps=UNION_GIF_FPS,
                union_gif_eps_quantile=UNION_GIF_EPS_QUANTILE,
                union_gif_eps_max_m=UNION_GIF_EPS_MAX_M,
                union_gif_show_edges=UNION_GIF_SHOW_EDGES,
            )
            if MAKE_SCENARIO_MAPS:
                create_scenario_maps(paths, point_size=MAP_POINT_SIZE)
            update_city_metadata(
                paths,
                {"artifacts": {"step06_directory": str(paths.step06_dir)}},
            )

    # 5. Step 7, across all cities ------------------------------------------------
    # For each matrix type, compare_city_scenarios skips comparisons that are
    # complete and still match the Step 5 results, and stops if a Step 5 result
    # it needs is missing or incomplete.
    if 7 in steps:
        from step07_compare_TDA_scenarios import compare_city_scenarios

        for paths in paths_list:
            result = compare_city_scenarios(paths)
            update_city_metadata(
                paths,
                {"artifacts": {"step07_directory": str(paths.step07_dir)}},
            )

    print("\nAll requested steps completed or were validly skipped.")


# Running this file (e.g. Run in Spyder) runs main().
if __name__ == "__main__":
    main()

