# -*- coding: utf-8 -*-
"""Step 3 - build the raw one-way walk and drive travel-time matrices.

Step 3 measures how long it takes to get from every pharmacy to every other
pharmacy along real streets, once on foot and once by car. The results are "raw":
one-way times in seconds, with no parking overhead, no round trips and no
population weighting. Step 4 turns them into the six distance matrices.

WHAT STEP 3 READS
-----------------
- step02_census/tract_data.csv: one row per pharmacy kept by Step 2, with its
  pharmacy_id, latitude, longitude, analysis_run_tag and the census values for the
  tract it sits in. Every row must have coordinates.
- The street networks around the pharmacies, downloaded from OpenStreetMap
  (see STREET NETWORKS below).

WHAT STEP 3 WRITES (in step03_raw_travel/; times in seconds)
------------------------------------------------------------
- manifest.csv: the Step 2 rows in the same order, with a matrix_index column
  (0, 1, 2, ...) added at the front and the name and path of the Step 2 file added
  at the end. It fixes the row and column order of every later matrix: row k and
  column k are always the pharmacy whose matrix_index is k. Step 4 reads the tract
  populations and car shares from it, and Step 5 scenarios use its pharmacy_id values.
- walk/matrix.npy:  walk[i, j]  = one-way walking time from pharmacy i to pharmacy j.
  Walking is the same both ways, so walk[i, j] == walk[j, i].
- drive/matrix.npy: drive[i, j] = one-way driving time from i to j. It can differ
  by direction because of one-way streets.
- step03_metadata.json: the source file, the settings used (buffer, walking speed),
  checksums of the files above, a checksum of the pharmacy order, the OSMnx version
  and Overpass server, and how many download attempts each network needed. Step 0
  uses the checksums to confirm the files before skipping or using them.

In both matrices the diagonal (a pharmacy to itself) is 0, and a pair that can't
be connected on the network is NaN ("not a number", i.e. missing).

STREET NETWORKS
---------------
Both networks come from OpenStreetMap (OSM), the free, volunteer-built world map.
OSMnx (a Python package) downloads them from an Overpass server (a public service
that answers queries on OSM data) and turns each into a graph: nodes are points on
the streets (intersections, dead ends and, in the drive network, every point that
shapes a street) and edges are the street segments between them, with their
length in metres.

- Drive network (network_type="drive"): public streets cars can use, without
  service roads such as driveways and parking aisles. It is directed: a one-way
  street only has an edge in its legal direction.
- Walk network (network_type="walk"): streets and paths people can walk on. OSMnx
  ignores one-way rules for walking, so every walk edge goes both ways.

Download area: the smallest latitude/longitude box that contains every pharmacy,
widened on every side by STEP3_BBOX_BUFFER_DEGREES = 0.02 degrees. That is about
2.2 km north-south; east-west it is a little less, because lines of longitude get
closer toward the poles (about 1.6 km at Milwaukee, 1.9 km at Austin). The buffer
lets routes between pharmacies near the edge use streets just beyond the outermost
pharmacies; a route that would leave the box can't be used.

The public Overpass server is sometimes busy or slow. A download that fails for a
temporary reason is retried, with a wait that doubles after each subsequent
failure (download_graph_with_retries). Nothing is saved until every requested
matrix has been built, so a failed download leaves no files behind and the city
can simply be rerun with the same ANALYSIS_RUN_TAG.

HOW THE TRAVEL TIMES ARE MEASURED
---------------------------------
Each pharmacy is attached ("snapped") to the closest node of each network, by
straight-line distance from its coordinates (ox.distance.nearest_nodes). Times
are then measured between those nodes along the best route, using Dijkstra's
algorithm (a standard shortest-path search, from the networkx package).

- Drive: OSMnx gives every street segment a speed (ox.add_edge_speeds): the speed
  limit recorded in OSM, or, where that is missing, the average limit of the roads
  of the same type (residential, primary, ...) in the downloaded network. A
  segment's time is its length divided by that speed (ox.add_edge_travel_times),
  and drive[i, j] is the total time of the quickest route. This is free-flow
  driving at the speed limit: no traffic, no stops at lights or signs.
      Example: a 400 m block with a 25 mph limit (40.2 km/h = 11.18 m/s) takes
      400 / 11.18 = 35.8 s.
- Walk: the shortest walking distance in metres divided by WALK_SPEED_MPS = 1.42 m/s
  (85.2 metres per minute, about 3.2 mph), the walking speed Hickok et al. use.
      Example: a shortest walk of 1,200 m takes 1200 / 1.42 = 845 s (14.1 minutes).

Why walk is symmetric and drive is not: walking uses the same streets in both
directions, so walk[i, j] == walk[j, i]. Driving must follow one-way streets, so
the quickest route from i to j can take longer or shorter than the route back.

Not done here: the parking/overhead penalty (DRIVE_PARKING_PENALTY_MINUTES) is NOT
added in Step 3; Step 4 adds it to every one-way drive. Round trips and population
weighting are also Step 4. Times stay in seconds until Step 5 converts them.

HOW THE PIECES FIT TOGETHER (compute_matrices_dijkstra, at the bottom)
----------------------------------------------------------------------
1. Stop if any Step 3 file already exists for this city (safety check).
2. Read the Step 2 file and check it: this run's tag, unique pharmacy IDs, and a
   latitude and longitude for every pharmacy.
3. Build and check the manifest .............. build_matrix_manifest, validate_manifest
   and a checksum of the pharmacy order (both kept in memory until stage 7).
4. Set the download box: all pharmacies plus the 0.02-degree buffer.
5. Drive: download the network ............... download_graph_with_retries
   add speeds and segment times (OSMnx), snap each pharmacy to a node, then
   time every pair ........................... compute_full_drive_matrix
6. Walk: download the network ................ download_graph_with_retries
   snap each pharmacy to a node, then
   time every pair ........................... compute_full_walk_matrix
7. Save the manifest and the matrices, then the metadata file last.

NOTES AND LIMITATIONS
---------------------
- The distance between a pharmacy and the node it is snapped to is not counted.
  The drive network keeps every point along each street, so a pharmacy snaps to a
  nearby point on the street; the walk network keeps only intersections and dead
  ends, so a pharmacy in the middle of a long block snaps to the nearest corner.
- Two pharmacies snapped to the same node (for example, in the same building) get
  a travel time of 0 to each other.
- OSM is edited all the time, so rerunning Step 3 later can give slightly
  different times. The metadata records the OSMnx version and server, not the date
  of the map data.
- Hickok et al. also use public transit; this pipeline has no transit times.

KNOWN ISSUE: PHARMACIES THAT CAN'T BE REACHED BY CAR IN ONE DIRECTION
---------------------------------------------------------------------
The drive network is downloaded with retain_all=False, which keeps only its
largest WEAKLY connected part: the streets that join up when one-way rules are
ignored. Within that part, a node can still sit on a one-way stub that a car can
enter but not leave, or leave but not enter. A pharmacy snapped to such a node has
NaN drive times in one direction. In test_261011 this affected 4 pharmacies
(Atlanta 2, Albuquerque 1, Austin 1). What happens downstream:
- Step 4: a round trip needs both directions, so all of that pharmacy's drive and
  multimodal values become NaN.
- Step 5: NaN becomes +inf, so the pharmacy never connects to the others. It adds
  an H0 class (a separate group of pharmacies) that never dies and drops out of H1
  (loops), and the H0 persistence landscape can't be computed (numpy/gtda
  RuntimeWarnings).
- Step 7: H0 distances are +inf when the baseline and a scenario have different
  numbers of never-dying classes.
A fix has been proposed but not applied: snap pharmacies to the largest STRONGLY
connected part, in which every node can reach every other while obeying one-way
rules. Walking is not affected, because the walk network is two-way.
"""


from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd
import requests

from pipeline_common import (
    ANALYSIS_TAG_COLUMN,
    file_sha256,
    ordered_id_checksum,
    utc_now_iso,
    validate_analysis_tag,
    validate_pharmacy_ids,
    write_json,
)

# OSMnx is only needed when Step 3 actually runs. If it is missing, this file can
# still be imported (by Step 0 or by tests); compute_matrices_dijkstra then stops
# with an ImportError that includes the original error.
try:
    import osmnx as ox
    _OSMNX_IMPORT_ERROR = None
except Exception as exc:
    ox = None
    _OSMNX_IMPORT_ERROR = exc

# Defaults used when Step 3 is run on its own. The main file (Step 0) passes its
# STEP3_DOWNLOAD_ATTEMPTS and STEP3_RETRY_WAIT_SECONDS settings instead. With these
# values a download is tried up to 4 times, waiting 60, 120 and 240 seconds between
# tries (7 minutes at most).
DEFAULT_DOWNLOAD_ATTEMPTS = 4
DEFAULT_RETRY_WAIT_SECONDS = 60.0


# =============================================================================
# Downloading the street networks (with retries)
# =============================================================================

def _retryable_download_errors() -> tuple[type[BaseException], ...]:
    """Return the download errors that can clear up on their own, so are worth a retry.

    These are failed or dropped connections, responses that arrive too slowly, and
    error replies from the server. OSMnx already pauses and retries "server busy"
    replies (HTTP 429 and 504) by itself. Anything else, such as an area with no
    street data or a mistake in the code, is not retried and stops Step 3 at once.

    Returned as a tuple so that download_graph_with_retries can catch any of them
    with a single ''except'' line.
    """
    errors: list[type[BaseException]] = [
        requests.exceptions.ConnectionError,  # includes connection timeouts
        requests.exceptions.Timeout,  # includes slow responses (read timeouts)
        requests.exceptions.ChunkedEncodingError,  # connection dropped mid-download
    ]
    try:  # OSMnx raises this for server error replies it does not retry itself
        from osmnx._errors import ResponseStatusCodeError

        errors.append(ResponseStatusCodeError)
    except Exception:  # reached only if this OSMnx version lacks that error type
        pass
    return tuple(errors)


def download_graph_with_retries(
    bbox: tuple[float, float, float, float],
    *,
    network_type: str,
    place_label: str,
    attempts: int = DEFAULT_DOWNLOAD_ATTEMPTS,
    retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    **graph_options: Any,
) -> tuple[Any, int]:
    """Download one OSMnx street network, retrying temporary network failures.

    Inputs:
    - ''bbox'': the download box in degrees, as (west, south, east, north), the
      order OSMnx 2.x expects (left, bottom, right, top).
    - ''network_type'': "drive" or "walk".
    - ''place_label'': the city name, used only in messages.
    - ''attempts'': the total number of tries; 1 turns retries off.
    - ''retry_wait_seconds'': the wait before the first retry. It doubles after
      each subsequent failure: four attempts with a 60-second wait pause for 60,
      120 and 240 seconds (420 seconds, or 7 minutes, at most).
    - ''sleep'': the function that does the waiting. Tests replace it so that they
      don't actually wait.
    - ''graph_options'': passed unchanged to ox.graph_from_bbox (for example
      ''simplify'' and ''retain_all''; see compute_matrices_dijkstra).

    Returns the graph and the number of attempts it took (saved in the metadata).

    Only the errors listed in _retryable_download_errors are retried; any other
    error stops at once. If every attempt fails, raises a RuntimeError that names
    the last error and says that nothing was saved for this city.
    """
    attempts = int(attempts)
    retry_wait_seconds = float(retry_wait_seconds)
    if attempts < 1:
        raise ValueError("The number of Step 3 download attempts must be at least 1.")
    if retry_wait_seconds < 0:
        raise ValueError("The Step 3 retry wait time cannot be negative.")
    retryable = _retryable_download_errors()
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            graph = ox.graph_from_bbox(bbox, network_type=network_type, **graph_options)
            return graph, attempt
        except retryable as exc:
            last_error = exc
            # No wait after the last attempt: go straight to the error below.
            if attempt == attempts:
                break
            # The wait doubles each time: 60 s after the 1st failure, 120 s after
            # the 2nd, 240 s after the 3rd (with the 60-second default).
            wait_seconds = retry_wait_seconds * 2 ** (attempt - 1)
            print(
                f"Step 3 {network_type} download failed for {place_label} "
                f"(attempt {attempt} of {attempts}): {type(exc).__name__}. "
                f"Waiting {wait_seconds:g} s before retrying."
            )
            sleep(wait_seconds)
    # Every attempt failed. "from last_error" keeps the original error in the
    # traceback, so its details are not lost.
    raise RuntimeError(
        f"Step 3 could not download the {network_type} street network for {place_label} "
        f"after {attempts} attempt(s). Last error: {type(last_error).__name__}: "
        f"{last_error}. No Step 3 files were saved for this city, so it can be rerun "
        "with the same ANALYSIS_RUN_TAG once the Overpass server responds."
    ) from last_error


def _osmnx_download_settings() -> dict[str, Any]:
    """Return the OSMnx version and Overpass server settings, for the metadata file.

    OSM data and Overpass servers change over time, so recording which software
    and server produced the networks helps explain why a later rerun may give
    slightly different travel times. Returns ''osmnx_version'', ''overpass_url''
    (the server address) and ''overpass_timeout_seconds'' (how long one request
    may take before OSMnx gives up); a value is None if it can't be found.

    OSMnx 2.x names these settings ''overpass_url'' and ''requests_timeout'';
    OSMnx 1.x named them ''overpass_endpoint'' and ''timeout''. Both are tried.
    """
    settings = ox.settings
    return {
        "osmnx_version": getattr(ox, "__version__", None),
        # Try the OSMnx 2.x name first, then the 1.x name, then give None.
        "overpass_url": getattr(
            settings, "overpass_url", getattr(settings, "overpass_endpoint", None)
        ),
        "overpass_timeout_seconds": getattr(
            settings, "requests_timeout", getattr(settings, "timeout", None)
        ),
    }


# =============================================================================
# All-pairs travel times on a street network
# =============================================================================

def compute_full_drive_matrix(graph, nodes, weight: str = "travel_time") -> np.ndarray:
    """Return drive[i, j], the quickest driving time (s), for every pair of pharmacies.

    ''nodes'' holds one network node per pharmacy, in manifest order (from
    ox.distance.nearest_nodes), so row i and column i of the result belong to the
    pharmacy with matrix_index i. ''weight'' names the edge value added up along a
    route: ''travel_time'' is the time in seconds that ox.add_edge_travel_times gave
    each street segment (length divided by speed).

    One Dijkstra search from each pharmacy's node finds the quickest route to every
    node it can reach, obeying one-way streets:

        drive[i, j] = total travel_time along the quickest route from node i to node j

    Row i comes from the search that starts at pharmacy i, so drive[i, j] and
    drive[j, i] come from different searches and can differ. Where two edges join
    the same pair of nodes, networkx uses the quicker one.

    - A pharmacy that can't be reached from i stays NaN in row i (see the known
      issue at the top of this file).
    - The diagonal is set to 0.
    - Two pharmacies snapped to the same node get 0 for each other.
    - These are raw times: the parking overhead is added in Step 4.

    The result is an n x n array of floats (n = number of pharmacies). Each of the
    n searches covers the whole network, which is the slow part of Step 3 for
    cities with many pharmacies.
    """
    n = len(nodes)
    # Start with every pair missing (NaN); pairs are filled in as routes are found.
    matrix = np.full((n, n), np.nan, dtype=float)
    for row_index, source in enumerate(nodes):
        # Quickest time from this pharmacy's node to every node it can reach,
        # as {node: seconds}. Unreachable nodes are not in the result.
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight=weight)
        for column_index, target in enumerate(nodes):
            matrix[row_index, column_index] = lengths.get(target, np.nan)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def compute_full_walk_matrix(
    graph,
    nodes,
    length_weight: str = "length",
    speed_mps: float = 1.42,
) -> np.ndarray:
    """Return walk[i, j], the walking time in seconds for every pair of pharmacies.

        walk[i, j] = shortest walking distance from node i to node j (m) / speed_mps

    ''nodes'' holds one network node per pharmacy, in manifest order. ''length_weight''
    names the edge value added up along a route: ''length'' is each segment's length
    in metres, set by OSMnx at download. ''speed_mps'' is the walking speed in metres
    per second; Step 0 passes WALK_SPEED_MPS = 1.42 (85.2 metres per minute, about
    3.2 mph), the walking speed Hickok et al. use. The 1.42 default here only
    applies when this function is called on its own.

    Example: a shortest walk of 1,200 m takes 1200 / 1.42 = 845 s (14.1 minutes).

    Walking is the same both ways (OSMnx makes every walking street two-way), so
    each time found is written to both walk[i, j] and walk[j, i], and the result is
    symmetric. Otherwise it works like compute_full_drive_matrix: one Dijkstra
    search per pharmacy; pairs that can't be connected stay NaN; the diagonal is 0;
    two pharmacies snapped to the same node get 0 for each other.

    Raises ValueError if ''speed_mps'' is 0 or negative.
    """
    if speed_mps <= 0:
        raise ValueError("walk_speed_mps must be positive.")
    n = len(nodes)
    # Start with every pair missing (NaN); pairs are filled in as routes are found.
    matrix = np.full((n, n), np.nan, dtype=float)
    for row_index, source in enumerate(nodes):
        # Shortest walking distance in metres from this pharmacy's node to every
        # node it can reach, as {node: metres}.
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight=length_weight)
        for column_index, target in enumerate(nodes):
            distance_m = lengths.get(target)
            if distance_m is not None:
                seconds = float(distance_m) / float(speed_mps)  # metres / (m/s) = seconds
                # Fill both directions, so the matrix is symmetric by construction.
                matrix[row_index, column_index] = seconds
                matrix[column_index, row_index] = seconds
    np.fill_diagonal(matrix, 0.0)
    return matrix


# =============================================================================
# The manifest: fixes the row and column order of every matrix
# =============================================================================

def build_matrix_manifest(
    filtered_df: pd.DataFrame,
    source_csv_path: str | os.PathLike[str],
    analysis_run_tag: str,
) -> pd.DataFrame:
    """Create the manifest: the pharmacy list whose row order every matrix follows.

    The manifest is a copy of the Step 2 rows (''filtered_df''), in the same order,
    with:
    - ''matrix_index'' (0, 1, 2, ...) added as the first column. Row k and column k
      of every Step 3 and later matrix are the pharmacy with matrix_index k.
    - ''source_tract_data_file'' and ''source_tract_data_path'' (the name and full
      path of the Step 2 file) added at the end, to record where the rows came from.

    Checks first that every row has this run's tag and a unique pharmacy_id. Only
    builds the table; compute_matrices_dijkstra writes it once both matrices exist.
    """
    # Renumber the rows 0, 1, 2, ... so matrix_index matches the row position.
    manifest = filtered_df.copy().reset_index(drop=True)
    validate_analysis_tag(manifest, analysis_run_tag, "Step 3 filtered input")
    validate_pharmacy_ids(manifest, "Step 3 filtered input")
    manifest.insert(0, "matrix_index", np.arange(len(manifest), dtype=int))
    manifest["source_tract_data_file"] = Path(source_csv_path).name
    manifest["source_tract_data_path"] = str(Path(source_csv_path).resolve())
    return manifest


def validate_manifest(manifest: pd.DataFrame, expected_tag: str, context: str) -> list[str]:
    """Check a manifest and return its pharmacy IDs in matrix order.

    Checks that every row has a pharmacy_id and no ID appears twice, that every row
    has this run's tag (''expected_tag''), and that matrix_index exists and runs
    0, 1, 2, ... in order. ''context'' names the manifest in error messages. Stops
    with a ValueError if any check fails.

    Step 4 has its own version of this check, which also requires the population
    column.
    """
    ids = validate_pharmacy_ids(manifest, context)
    validate_analysis_tag(manifest, expected_tag, context)
    if "matrix_index" not in manifest.columns:
        raise ValueError(f"{context} is missing matrix_index.")
    # Text or blank values become NaN, which fails the check below.
    indices = pd.to_numeric(manifest["matrix_index"], errors="coerce")
    expected = np.arange(len(manifest))
    if indices.isna().any() or not np.array_equal(indices.to_numpy(), expected):
        raise ValueError(f"{context} matrix_index is not sequential from zero.")
    return ids


# =============================================================================
# Step 3 for one city: runs the pieces above in order
# =============================================================================

def compute_matrices_dijkstra(
    input_csv_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    city: str,
    state: str,
    *,
    analysis_run_tag: str,
    do_drive: bool = True,
    do_walk: bool = True,
    bbox_buffer_degrees: float = 0.02,
    walk_speed_mps: float = 1.42,
    download_attempts: int = DEFAULT_DOWNLOAD_ATTEMPTS,
    retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str | int]:
    """Build the Step 3 manifest and the raw walk and drive matrices for one city.

    Step 0 calls this once per city whose Step 3 folder is still empty. The
    numbered comments below match the numbered list at the top of this file. The
    name refers to Dijkstra's shortest-path method, used for the travel times.

    Inputs:
    - ''input_csv_path'': Step 2's tract_data.csv for this city.
    - ''output_dir'': the city's step03_raw_travel folder.
    - ''city'', ''state'': used only in messages and the metadata.
    - ''do_drive'', ''do_walk'': which matrices to build. Step 0 always asks for
      both, because Step 4 needs both and Step 0's check of this folder expects
      both matrix files.
    - ''bbox_buffer_degrees'', ''walk_speed_mps'', ''download_attempts'',
      ''retry_wait_seconds'': the defaults apply only when Step 3 is run on its own.
      Step 0 passes STEP3_BBOX_BUFFER_DEGREES, WALK_SPEED_MPS,
      STEP3_DOWNLOAD_ATTEMPTS and STEP3_RETRY_WAIT_SECONDS.
    - ''sleep'': the waiting function used between download attempts (tests
      replace it).

    Both street networks are downloaded and both matrices built before anything is
    saved. If a download still fails after ''download_attempts'' tries, Step 3 stops
    and leaves no files behind for this city, so it can be rerun with the same
    ANALYSIS_RUN_TAG.

    Returns the paths of the files written (None for a mode that was not requested)
    and the number of pharmacies. Step 0 does not use the returned values; it checks
    the folder itself afterwards.
    """
    if not do_drive and not do_walk:
        raise ValueError("At least one of do_drive or do_walk must be True.")
    if ox is None:
        raise ImportError(f"Step 3 requires OSMnx: {_OSMNX_IMPORT_ERROR}")

    # 1. Stop if any requested Step 3 file already exists. Step 0 only calls this
    #    for a city with no Step 3 files, so this is a safety check (mainly for
    #    running Step 3 on its own). Nothing is ever overwritten.
    input_path = Path(input_csv_path)
    root = Path(output_dir)
    manifest_path = root / "manifest.csv"
    metadata_path = root / "step03_metadata.json"
    walk_path = root / "walk" / "matrix.npy"
    drive_path = root / "drive" / "matrix.npy"
    requested_outputs = [manifest_path, metadata_path]
    if do_walk:
        requested_outputs.append(walk_path)
    if do_drive:
        requested_outputs.append(drive_path)
    if any(path.exists() for path in requested_outputs):
        raise FileExistsError(
            f"Step 3 output already exists in {root}; the main should validate and skip it."
        )

    # 2. Read the Step 2 file and check it.
    #    IDs are read as text so values such as NPIs keep any leading zeros.
    frame = pd.read_csv(
        input_path,
        dtype={"pharmacy_id": "string", "NPI": "string", "google_place_id": "string"},
    )
    validate_analysis_tag(frame, analysis_run_tag, str(input_path))
    validate_pharmacy_ids(frame, str(input_path))
    required = {"latitude", "longitude"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Step 3 input is missing columns: {missing}.")
    # Text or blank coordinates become NaN and are caught just below.
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    # A pharmacy without coordinates can't be placed on the map. Step 3 stops
    # rather than drop it, because dropping a row here would quietly change the
    # list of pharmacies (the manifest).
    if frame[["latitude", "longitude"]].isna().any(axis=1).any():
        raise ValueError(
            "Step 3 input contains missing coordinates. Step 2 should remove or reject them "
            "before matrix construction so manifest identity is not changed without user input."
        )
    if frame.empty:
        raise ValueError("Step 3 input contains no pharmacies.")

    # 3. Build and check the manifest now, but write it only once both matrices
    #    exist (stage 7). The checksum is a fingerprint of the pharmacy IDs in
    #    order. It goes in the metadata (Steps 4 and 5 save it too), so Step 0 and
    #    a reader can confirm a matrix uses the same pharmacies in the same order.
    manifest = build_matrix_manifest(frame, input_path, analysis_run_tag)
    ids = validate_manifest(manifest, analysis_run_tag, str(manifest_path))
    checksum = ordered_id_checksum(ids)

    # 4. The download box: the smallest box around all pharmacies, widened by the
    #    buffer on every side (0.02 degrees is about 2 km; see the top of this file).
    #    The order (west, south, east, north) is what OSMnx 2.x expects.
    min_lon, max_lon = frame["longitude"].min(), frame["longitude"].max()
    min_lat, max_lat = frame["latitude"].min(), frame["latitude"].max()
    buffer = float(bbox_buffer_degrees)
    bbox = (min_lon - buffer, min_lat - buffer, max_lon + buffer, max_lat + buffer)
    place_label = f"{city}, {state.upper()}"  # used only in messages
    # Both downloads use the same retry settings.
    retry_settings = {
        "attempts": download_attempts,
        "retry_wait_seconds": retry_wait_seconds,
        "sleep": sleep,
    }
    # How many tries each download needed, saved in the metadata.
    attempts_used: dict[str, int] = {}
    # A matrix stays None if its mode was not requested.
    drive = None
    walk = None

    # 5. Drive times.
    if do_drive:
        print(f"Step 3 drive matrix: {city}, {state}")
        # Download options:
        # - simplify=False keeps every OSM node, including the points that only
        #   shape a curving street, so a pharmacy can snap to a point along the
        #   street, not only to an intersection.
        # - retain_all=False keeps only the largest weakly connected part of the
        #   network (connected when one-way rules are ignored). Known issue: some
        #   nodes in it can't be reached, or can't be left, by car; see the
        #   nearest_nodes comment below and the top of this file.
        # - truncate_by_edge=False drops nodes outside the box, so streets are cut
        #   off at its edge.
        graph, attempts_used["drive"] = download_graph_with_retries(
            bbox,
            network_type="drive",
            place_label=place_label,
            simplify=False,
            retain_all=False,
            truncate_by_edge=False,
            **retry_settings,
        )
        # Speed (km/h) for every street segment: its OSM speed limit, or, where
        # that is missing, the average limit of roads of the same type in this
        # network. Limits given in mph are converted to km/h.
        ox.add_edge_speeds(graph)
        # Time (seconds) for every segment = length (m) / speed (converted to m/s).
        # Free-flow driving: no traffic, no stops at lights or signs.
        ox.add_edge_travel_times(graph)
        # Snap each pharmacy to the closest network node, by straight-line distance
        # (X = longitude, Y = latitude). Gives one node per pharmacy, in manifest
        # order. For a latitude/longitude network OSMnx needs scikit-learn for this.
        # Known issue: the closest node can be on a one-way stub that a car can't
        # leave or can't enter, which gives that pharmacy NaN drive times in one
        # direction (4 pharmacies in test_261011: Atlanta 2, Albuquerque 1,
        # Austin 1). A fix has been proposed (snap to the largest strongly connected
        # part, where every node can reach every other) but not applied.
        nodes = ox.distance.nearest_nodes(
            graph, frame["longitude"].to_numpy(), frame["latitude"].to_numpy()
        )
        drive = compute_full_drive_matrix(graph, list(nodes), weight="travel_time")
        if drive.shape != (len(manifest), len(manifest)):
            raise ValueError("Drive matrix dimension does not match manifest length.")
        del graph  # free the drive network before downloading the walk network

    # 6. Walk times.
    if do_walk:
        print(f"Step 3 walk matrix: {city}, {state}")
        # Download options:
        # - The walk network ignores one-way rules, so every edge goes both ways.
        # - simplify=True keeps only intersections and dead ends as nodes (segment
        #   lengths are kept in full), so a pharmacy snaps to the nearest corner.
        # - retain_all is left at OSMnx's default (False): only the largest
        #   connected part is kept. Because every walking street is two-way, every
        #   node in it can reach every other, so missing walk times are not expected.
        graph, attempts_used["walk"] = download_graph_with_retries(
            bbox,
            network_type="walk",
            place_label=place_label,
            simplify=True,
            **retry_settings,
        )
        # Snap each pharmacy to the closest walk-network node (same method as for
        # driving; the node can differ from the drive one).
        nodes = ox.distance.nearest_nodes(
            graph, frame["longitude"].to_numpy(), frame["latitude"].to_numpy()
        )
        # Shortest walking distance (m) / walking speed (m/s) = seconds.
        walk = compute_full_walk_matrix(
            graph,
            list(nodes),
            length_weight="length",
            speed_mps=float(walk_speed_mps),
        )
        if walk.shape != (len(manifest), len(manifest)):
            raise ValueError("Walk matrix dimension does not match manifest length.")
        del graph

    # 7. Every requested matrix exists, so write the outputs. The metadata file goes
    #    last: it holds checksums of the files above, so they must exist first, and
    #    if the run is interrupted while saving there is no metadata file, so Step 0
    #    reports the Step 3 outputs as partial and stops instead of using them.
    root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    if drive is not None:
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(drive_path, drive)
    if walk is not None:
        walk_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(walk_path, walk)

    # The "_sha256" values are checksums (fingerprints) of each file. Step 0
    # recomputes them to confirm the files have not changed since they were made.
    metadata = {
        "pipeline_step": 3,
        "city": city,
        "state": state.upper(),
        ANALYSIS_TAG_COLUMN: analysis_run_tag,
        "created_utc": utc_now_iso(),
        "source_tract_file": str(input_path.resolve()),
        "source_tract_sha256": file_sha256(input_path),
        "manifest_file": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "matrix_dimension": len(manifest),
        "ordered_pharmacy_id_checksum": checksum,
        "bbox_buffer_degrees": buffer,
        "walk_speed_mps": float(walk_speed_mps),
        "walk_matrix": str(walk_path.resolve()) if do_walk else None,
        "walk_matrix_sha256": file_sha256(walk_path) if do_walk else None,
        "drive_matrix": str(drive_path.resolve()) if do_drive else None,
        "drive_matrix_sha256": file_sha256(drive_path) if do_drive else None,
        # Download details: which server and settings produced the street networks,
        # and how many attempts each download needed.
        **_osmnx_download_settings(),
        "download_attempts_allowed": int(download_attempts),
        "retry_wait_seconds": float(retry_wait_seconds),
        "download_attempts_used": attempts_used,
    }
    write_json(metadata_path, metadata)
    print("\n=== Step 3 summary ===")
    print(f"Manifest rows: {len(manifest)}")
    print(f"Ordered ID checksum: {checksum}")
    return {
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "walk_path": str(walk_path) if do_walk else None,
        "drive_path": str(drive_path) if do_drive else None,
        "matrix_dimension": len(manifest),
    }


# Run Step 3 for one city from the command line, without Step 0. For example:
#     python step03_build_raw_walk_and_drive_time_matrices.py
#         <tract_data.csv> <step03_raw_travel folder> Milwaukee WI test_261011
# The option defaults match Step 0's settings (0.02-degree buffer, 1.42 m/s walking
# speed, 4 download attempts, 60-second first wait). Add --no-drive or --no-walk to
# build only one matrix.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build raw walk and drive matrices.")
    parser.add_argument("input_csv")
    parser.add_argument("output_dir")
    parser.add_argument("city")
    parser.add_argument("state")
    parser.add_argument("analysis_run_tag")
    parser.add_argument("--bbox-buffer-degrees", type=float, default=0.02)
    parser.add_argument("--walk-speed-mps", type=float, default=1.42)
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--no-walk", action="store_true")
    parser.add_argument("--download-attempts", type=int, default=DEFAULT_DOWNLOAD_ATTEMPTS)
    parser.add_argument("--retry-wait-seconds", type=float, default=DEFAULT_RETRY_WAIT_SECONDS)
    args = parser.parse_args()
    compute_matrices_dijkstra(
        args.input_csv,
        args.output_dir,
        args.city,
        args.state,
        analysis_run_tag=args.analysis_run_tag,
        do_drive=not args.no_drive,
        do_walk=not args.no_walk,
        bbox_buffer_degrees=args.bbox_buffer_degrees,
        walk_speed_mps=args.walk_speed_mps,
        download_attempts=args.download_attempts,
        retry_wait_seconds=args.retry_wait_seconds,
    )


