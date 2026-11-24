# -*- coding: utf-8 -*-
"""Step 6 - draw maps of each city's pharmacies and of the removal scenarios.

Step 6 only makes pictures. Steps 5 and 7 do not read anything it writes, so the
maps can be made, remade in a new run or left out without changing any result.
Step 0 runs Step 6 for every city before Step 7 starts.

WHAT STEP 6 READS
-----------------
- step02_census/tract_data.csv: one row per pharmacy kept by Step 2, with its
  latitude and longitude in degrees. Used for the pharmacy map and the GIF.
- For the scenario maps only, the saved Step 5 scenario plan in
  step05_tda/scenario_plan/:
  - plan.json: the plan, including its plan_hash (a fingerprint of the plan);
  - scenario_index.csv: one row per scenario, with the IDs of the pharmacies it
    removes joined by "|" (excluded_pharmacy_ids);
  - memberships/<scenario_key>.csv: one row per pharmacy, in manifest order,
    with included = True (kept) or False (removed).
- For the scenario maps only, step03_raw_travel/manifest.csv: the pharmacy
  positions, in the same row order as the membership files. Step 3 keeps every
  Step 2 row, so both kinds of map show the same pharmacies.

WHAT STEP 6 WRITES (in <CityState>/step06_mapping/)
---------------------------------------------------
- pharmacies_contextily.png: every pharmacy as a dot (red by default) on a light
  grey street map, 300 dpi.
- union_balls_contextily.gif: the union-of-balls animation (see below).
- mapping_metadata.json: the Step 2 file used and its SHA-256 checksum, the map
  settings, and the paths of the two maps above.
- scenario_maps/<scenario_key>.png: one map per removal scenario, 150 dpi.
  Blue = pharmacies the scenario keeps, orange = pharmacies it removes; the
  legend gives both counts. Example:
  scenario_maps/remove_10_percent_random/replicate_001.png
- scenario_maps/scenario_maps_metadata.json: the Step 5 plan_hash, the
  scenario-map settings, the map paths, and the kept/removed counts per scenario.

HOW THE PIECES FIT TOGETHER
---------------------------
Step 0 calls create_city_maps for every city, then create_scenario_maps for the
same city when MAKE_SCENARIO_MAPS is True.

create_city_maps (the pharmacy map and the GIF):
1. Check the Step 2 file exists and compare with any earlier Step 6 run (see
   SKIP AND STOP RULES below).
2. Read the pharmacies, check the run tag and IDs, keep valid coordinates
   ................................................. _clean_points
3. Load the mapping packages ........................ _load_optional_mapping_packages
4. Project the points to Web Mercator metres, the projection of the map tiles.
5. Draw the pharmacy PNG, unless it already exists.
6. Draw the union-of-balls GIF, unless it already exists ... _pairwise_distances
7. Write mapping_metadata.json.

create_scenario_maps (one map per removal scenario):
1. Check the Step 5 plan, scenario index and Step 3 manifest exist.
2. Read the positions from the manifest and fix one extent for all of the
   city's scenario maps ............................. _web_mercator, _padded_bounds
3. Choose the scenarios to map (all but the baseline and leave-one-out).
4. Compare with any earlier scenario maps (see SKIP AND STOP RULES below).
5. For each scenario:
   a. read and check its membership file ........... _read_scenario_membership
   b. if its PNG does not exist yet, download the basemap (first map drawn
      only) ........................................ _fetch_basemap
      and draw the map ............................. _draw_scenario_map,
                                                     _scenario_subtitle
6. Write scenario_maps_metadata.json and print a one-line summary.

THE BASEMAP (CARTO Positron)
----------------------------
Every map is drawn on CARTO's light grey "Positron" tiles (light_all), which
need a free API key read from the CARTO_BASEMAP_API_KEY environment variable
(details next to CARTO_BASEMAP_URL below). The key is needed only when a map is
actually drawn, so a run in which every map already exists makes no tile
requests and does not need it.

THE UNION-OF-BALLS GIF
----------------------
Each frame uses one distance, epsilon, in metres. Every pharmacy gets a faint
shaded circle of radius r = epsilon / 2, and a line (an edge) joins two
pharmacies that are at most epsilon apart. Two circles of radius r touch exactly
when their centres are 2r = epsilon apart, so an edge appears at the moment its
two circles meet. epsilon runs evenly from 0 up to a chosen maximum: by default
the 70th percentile of all pairwise distances (UNION_GIF_EPS_QUANTILE = 0.70),
so by the last frame about 70% of all pairs are joined. UNION_GIF_EPS_MAX_M
sets the maximum in metres instead. A label in the bottom left corner shows
epsilon and r for each frame.

Worked example (made-up numbers): if the 70th percentile is 13,900 m, the 140
frames go from 0 to 13,900 m in steps of 13,900 / 139 = 100 m. Frame index 50 has
epsilon = 5,000 m and r = 2,500 m: two pharmacies 4,800 m apart are joined (their
circles overlap), two that are 5,200 m apart are not yet. At 3 frames per second
the GIF lasts about 47 seconds.

The GIF illustrates the idea behind the Vietoris-Rips filtration that Step 5
computes: as epsilon grows, groups of pharmacies join up (H0 classes, the gaps
between groups, die) and loops form and fill in (H1 classes, holes in coverage).
epsilon plays the part of Giotto's filtration value, and r the part of Hickok et
al.'s ball radius t (t = epsilon / 2). It is an illustration only, not the input
to persistent homology:
- Its distances are straight lines in Web Mercator metres, not travel times.
  Web Mercator stretches distances by about 1/cos(latitude): about 1.37x at
  Milwaukee (43 degrees north), so a label of 5,000 m there is about 3,650 m on
  the ground.
- Step 5 uses Step 4's d matrices (round-trip travel times, population
  weighted), in which two pharmacies close on the map can be far apart (for
  example across a river), and the other way round.
- Known issue: a pharmacy that Step 3 could not reach by car (NaN drive times)
  never joins up in the drive and multimodal PH results, but it joins normally
  in the GIF.
- Vietoris-Rips fills in a triangle of three pharmacies as soon as its three
  edges are present, so a small uncovered gap in the middle of three circles
  that touch only in pairs is not a hole there, even though the GIF shows one.

SKIP AND STOP RULES FOR EXISTING FILES
--------------------------------------
As elsewhere in the pipeline: complete outputs are skipped, missing ones are
made, and anything that doesn't match stops the run. Nothing is overwritten or
deleted.

Pharmacy PNG and GIF (create_city_maps):
- If mapping_metadata.json exists, it must have this run's tag, the same Step 2
  checksum and exactly the same settings, or Step 6 stops before drawing
  anything. If they match, each requested map is skipped if its file exists and
  drawn if not, and the metadata file is rewritten.
- If there is no mapping_metadata.json but either map file exists, Step 6 stops:
  there is no record of how the file was made. The metadata is written last, so
  this also happens when an earlier run stopped after drawing the PNG.
- An existing map file is accepted as it is (it is not opened or checked).

Scenario maps (create_scenario_maps):
- If scenario_maps_metadata.json exists, it must have this run's tag, the same
  Step 5 plan_hash and the same scenario-map settings, or Step 6 stops. If they
  match, each scenario's PNG is skipped if it exists and drawn if not.
- If there is no scenario_maps_metadata.json but any PNG exists in
  scenario_maps/ (subfolders included), Step 6 stops.
- Every membership file is read and checked, even when its map is skipped.

WHY THE SCENARIO MAPS HAVE THEIR OWN METADATA FILE
--------------------------------------------------
create_city_maps compares the saved settings in mapping_metadata.json with the
requested ones as a whole. Adding any new setting there (such as a scenario-map
switch) would make every existing mapping_metadata.json look different and stop
Step 6 on every run that already has maps. So the scenario maps live in their own
folder with their own metadata file, and mapping_metadata.json and the existing
maps are left as they were. The scenario-map metadata records the Step 5
plan_hash rather than the Step 2 checksum, because those maps are drawn from the
plan.

NOTES AND LIMITATIONS
---------------------
- The mapping packages (geopandas, contextily, matplotlib, pillow) are imported
  inside functions, not at the top of this file, so the rest of the pipeline and
  its offline checks can run on a computer that does not have them.
- Limitation: mapping_metadata.json does not record which basemap was used, so
  an existing PNG or GIF drawn on a different basemap is kept as it is.
  scenario_maps_metadata.json does record it.
- Changing any create_city_maps setting for a city that already has maps, even
  switching one map off, stops Step 6; use a new ANALYSIS_RUN_TAG.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_common_260801 import (
    ANALYSIS_TAG_COLUMN,
    CityRunPaths,
    file_sha256,
    read_json,
    utc_now_iso,
    validate_analysis_tag,
    validate_pharmacy_ids,
    write_json,
)


# =============================================================================
# Basemap: CARTO Positron tiles (needs a free API key)
# =============================================================================
# Every Step 6 map uses this basemap. OpenStreetMap's own tile servers now block
# automated downloads like those used in this pipeline, and CARTO's basemaps have
# needed a free API key since 25 September 2026 (free for non-commercial use up to
# 5 million tiles a month; request one at carto.com/basemaps/apikey). The key is
# read from the CARTO_BASEMAP_API_KEY environment variable, like the other API keys.
# In the address, contextily fills in {z} (zoom level) and {x}, {y} (tile column
# and row); _basemap_source adds the key after "?key=".
CARTO_BASEMAP_URL = "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png?key="
# Credit line printed on every map, as the tiles' terms of use ask.
CARTO_ATTRIBUTION = "(C) OpenStreetMap contributors (C) CARTO"


def _basemap_source() -> str:
    """Return the CARTO Positron (light grey) tile address, including the API key.

    The key comes from the CARTO_BASEMAP_API_KEY environment variable (spaces at
    either end are trimmed). Step 0 does not pass it in; Step 6 reads it here. If
    it is missing or empty, Step 6 stops with a message saying how to get one.
    A newly saved environment variable is seen only after Spyder (or whatever
    program runs the pipeline) is restarted.

    This is called only when a map is about to be drawn, so a run in which every
    map already exists does not need the key.
    """
    key = os.getenv("CARTO_BASEMAP_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Step 6 maps need a CARTO basemap API key. Request a free one at "
            "carto.com/basemaps/apikey, save it in the CARTO_BASEMAP_API_KEY "
            "environment variable, and restart Spyder."
        )
    return CARTO_BASEMAP_URL + key


# =============================================================================
# Helpers for the pharmacy map and the union-of-balls GIF
# =============================================================================

def _load_optional_mapping_packages():
    """Import the mapping packages and return them in one dictionary.

    The imports are done here rather than at the top of the file so that the
    rest of the pipeline (and its offline checks) can run on a computer without
    these packages; only drawing maps needs them.

    Dictionary keys are the short names create_city_maps uses: "cx" (contextily,
    downloads and draws the basemap tiles), "gpd" (GeoPandas, the points and their
    projection), "plt" (Matplotlib), and three Matplotlib pieces for the GIF:
    "FuncAnimation" (builds the animation frame by frame), "LineCollection" (the
    edges) and "Circle" (the balls). Pillow is not imported here; Matplotlib uses
    it to write the GIF.

    If any import fails, Step 6 stops with an ImportError that names the needed
    packages and repeats the original error.
    """
    try:
        import contextily as cx
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle
    except Exception as exc:  # reached only if a mapping package is missing or broken
        raise ImportError(
            "Step 6 requires geopandas, contextily, matplotlib, and pillow. "
            f"Original import error: {exc}"
        ) from exc
    return {
        "cx": cx,
        "gpd": gpd,
        "plt": plt,
        "FuncAnimation": FuncAnimation,
        "LineCollection": LineCollection,
        "Circle": Circle,
    }


def _clean_points(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the coordinates into numbers and drop rows that can't be put on a map.

    Needs ''latitude'' and ''longitude'' columns in degrees, and stops if either
    is missing. Text or blank values become missing (NaN). Rows with a missing
    coordinate, a latitude outside -90..90 or a longitude outside -180..180 are
    dropped, and the remaining rows are renumbered from 0. Stops if no rows are
    left. Returns a new table; the input is not changed.

    Dropped rows are not counted or reported, so a pharmacy with a bad
    coordinate would simply be missing from the pharmacy map and the GIF. Step 3
    stops on any missing coordinate, so in a run that got past Step 3 none
    should be missing here.
    """
    required = {"latitude", "longitude"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Step 6 input is missing coordinate columns: {missing}.")
    output = df.copy()
    output["latitude"] = pd.to_numeric(output["latitude"], errors="coerce")
    output["longitude"] = pd.to_numeric(output["longitude"], errors="coerce")
    output = output.dropna(subset=["latitude", "longitude"])
    # Keep only coordinates that exist on Earth.
    output = output[
        output["latitude"].between(-90, 90)
        & output["longitude"].between(-180, 180)
    ].reset_index(drop=True)
    if output.empty:
        raise ValueError("Step 6 input contains no valid coordinate pairs.")
    return output


def _pairwise_distances(points: np.ndarray):
    """Straight-line distance between every pair of points, in the points' units.

    ''points'' has one row per pharmacy and two columns, x and y (for the GIF,
    Web Mercator metres). For every pair:

        distance[i, j] = sqrt((x_i - x_j)**2 + (y_i - y_j)**2)

    Example: points (0, 0) and (3000, 4000) are 5000 m apart.

    Returns three things:
    - ''full'': the n x n matrix of all distances (0 on the diagonal);
    - ''upper'': the row and column numbers of every pair counted once (i < j),
      as two arrays; with 3 points the pairs are (0, 1), (0, 2) and (1, 2);
    - ''full[upper]'': the distances of those pairs in the same order,
      n * (n - 1) / 2 values. The GIF uses them to choose its largest epsilon and
      to decide which edges to draw in each frame.

    These are map distances, not travel times; see the top of this file.
    """
    # differences[i, j] = (x_i - x_j, y_i - y_j) for every pair, shape (n, n, 2).
    differences = points[:, None, :] - points[None, :, :]
    full = np.sqrt(np.sum(differences**2, axis=2))
    # Pairs above the diagonal: each pair once, and no pharmacy paired with itself.
    upper = np.triu_indices(points.shape[0], k=1)
    return full, upper, full[upper]


# =============================================================================
# Step 6 for one city: pharmacy map and union-of-balls GIF
# =============================================================================

def create_city_maps(
    paths: CityRunPaths,
    *,
    make_contextily: bool = True,
    make_unionballs_contextily_gif: bool = True,
    point_color: str = "red",
    point_size: float = 30.0,
    union_gif_frames: int = 140,
    union_gif_fps: int = 3,
    union_gif_eps_quantile: float = 0.70,
    union_gif_eps_max_m: float | None = None,
    union_gif_show_edges: bool = True,
) -> dict[str, Any]:
    """Draw the pharmacy map and the union-of-balls GIF for one city.

    Step 0 calls this once per city with the settings in its STEP 6 MAP SETTINGS
    section (MAKE_CONTEXTILY_MAP, MAKE_UNION_BALLS_GIF, MAP_POINT_COLOR, ...). The
    defaults here are the same as Step 0's and only matter when this is run on
    its own. The numbered comments below match the first numbered list at the
    top of this file.

    Settings:
    - ''make_contextily'': draw pharmacies_contextily.png.
    - ''make_unionballs_contextily_gif'': draw union_balls_contextily.gif.
    - ''point_color'', ''point_size'': colour and size of the pharmacy dots
      (Matplotlib marker size, in points squared). The GIF draws its dots at 60%
      of ''point_size''.
    - ''union_gif_frames'': the number of frames, that is, of epsilon values from
      0 to the maximum.
    - ''union_gif_fps'': frames per second (at least 1).
    - ''union_gif_eps_quantile'': used when ''union_gif_eps_max_m'' is None; the
      last epsilon is this percentile of all pairwise distances, so about this
      share of all pairs is joined by the last frame (0.70 = 70%).
    - ''union_gif_eps_max_m'': the last epsilon in Web Mercator metres; None uses
      the quantile instead.
    - ''union_gif_show_edges'': also draw a line between every pair of pharmacies
      that are at most epsilon apart.

    Skips and stops (see SKIP AND STOP RULES at the top of this file): if
    mapping_metadata.json exists, its tag, Step 2 checksum and settings must all
    match or Step 6 stops before drawing anything; an existing map is then
    skipped and a missing one drawn. Map files without mapping_metadata.json stop
    Step 6. All nine settings above are compared, including the two on/off
    switches.

    Returns the metadata dictionary that is also saved as mapping_metadata.json.
    Its "outputs" lists only the maps requested in this call.
    """
    # 1. The Step 2 file must exist; then compare with any earlier Step 6 run.
    if not paths.step02_tract_csv.exists():
        raise FileNotFoundError(
            f"Step 6 requires the Step 2 tract file: {paths.step02_tract_csv}"
        )
    # Every setting that changes the maps. It is saved in mapping_metadata.json and
    # compared on later runs. The values are turned into plain True/False, numbers
    # and text so they are stored the same way in the JSON file every time.
    requested_settings = {
        "make_contextily": bool(make_contextily),
        "make_unionballs_contextily_gif": bool(make_unionballs_contextily_gif),
        "point_color": str(point_color),
        "point_size": float(point_size),
        "union_gif_frames": int(union_gif_frames),
        "union_gif_fps": int(union_gif_fps),
        "union_gif_eps_quantile": float(union_gif_eps_quantile),
        "union_gif_eps_max_m": union_gif_eps_max_m,
        "union_gif_show_edges": bool(union_gif_show_edges),
    }
    # Fingerprint of the Step 2 file: if tract_data.csv has changed since the maps
    # were drawn, they are out of date, and the different checksum stops Step 6.
    source_hash = file_sha256(paths.step02_tract_csv)
    named_outputs = {
        "contextily_png": paths.step06_dir / "pharmacies_contextily.png",
        "union_balls_gif": paths.step06_dir / "union_balls_contextily.gif",
    }
    if paths.step06_metadata.exists():
        # An earlier Step 6 run finished for this city, so it must match this one.
        from pipeline_common_260801 import read_json
        saved = read_json(paths.step06_metadata)
        if saved.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
            raise ValueError("Step 6 metadata belongs to another analysis_run_tag.")
        # The settings are compared as a whole dictionary, so an added or missing
        # setting also counts as a difference. This is why the scenario maps keep
        # their settings in a separate metadata file.
        if saved.get("source_tract_sha256") != source_hash or saved.get("settings") != requested_settings:
            raise RuntimeError(
                "Existing Step 6 outputs were created from different source data or map "
                "settings. Use the original settings or a new ANALYSIS_RUN_TAG."
            )
    else:
        # No metadata file. On a first run there are no map files either, which is
        # fine. A map file without metadata means an earlier run stopped before the
        # metadata was written (it is written last), or the file came from
        # elsewhere; either way its settings are unknown, so stop rather than mix it
        # with new maps.
        existing_outputs = [str(path) for path in named_outputs.values() if path.exists()]
        if existing_outputs:
            raise RuntimeError(
                "Step 6 map files exist without matching mapping_metadata.json: "
                f"{existing_outputs}."
            )

    # 2. Read the pharmacies (IDs as text), check the run tag and the IDs, and keep
    #    the rows with valid coordinates.
    data = pd.read_csv(
        paths.step02_tract_csv,
        dtype={"pharmacy_id": "string", ANALYSIS_TAG_COLUMN: "string"},
        quoting=csv.QUOTE_MINIMAL,
    )
    validate_analysis_tag(data, paths.analysis_run_tag, str(paths.step02_tract_csv))
    validate_pharmacy_ids(data, str(paths.step02_tract_csv))
    data = _clean_points(data)

    # 3. Load the mapping packages (see _load_optional_mapping_packages).
    packages = _load_optional_mapping_packages()
    cx = packages["cx"]
    gpd = packages["gpd"]
    plt = packages["plt"]
    FuncAnimation = packages["FuncAnimation"]
    LineCollection = packages["LineCollection"]
    Circle = packages["Circle"]

    paths.step06_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    # e.g. "Milwaukee, WI pharmacies"
    title = f"{paths.city}, {paths.state.upper()} pharmacies"

    # 4. Turn the coordinates into map points. EPSG:4326 is longitude/latitude in
    #    degrees (as from a GPS); EPSG:3857 is Web Mercator in metres, the projection
    #    the basemap tiles use, so the points and the tiles line up.
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
        crs="EPSG:4326",
    )
    projected = gdf.to_crs(epsg=3857)

    # 5. Pharmacy map (PNG), unless it already exists.
    if make_contextily:
        output = named_outputs["contextily_png"]
        if not output.exists():
            fig, axis = plt.subplots(figsize=(10, 10))
            projected.plot(ax=axis, color=point_color, markersize=point_size, alpha=0.85)
            # Download the CARTO tiles for the area on screen and draw them under the
            # points, with the credit line (needs the API key; see _basemap_source).
            cx.add_basemap(axis, source=_basemap_source(), attribution=CARTO_ATTRIBUTION)
            axis.set_axis_off()
            axis.set_title(title)
            # Show all the points plus a 5% margin on every side. If every point has
            # the same x (or y), the width there is 0, so use a fixed 1,000 m margin.
            xmin, ymin, xmax, ymax = projected.total_bounds
            pad_x = (xmax - xmin) * 0.05 if xmax > xmin else 1000
            pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 1000
            axis.set_xlim(xmin - pad_x, xmax + pad_x)
            axis.set_ylim(ymin - pad_y, ymax + pad_y)
            fig.tight_layout()
            fig.savefig(output, dpi=300)  # 10 x 10 inches at 300 dpi = 3000 x 3000 pixels
            plt.close(fig)
        # Listed whether it was drawn just now or already existed.
        outputs["contextily_png"] = str(output.resolve())

    # 6. Union-of-balls GIF, unless it already exists (explained at the top of this file).
    if make_unionballs_contextily_gif:
        output = named_outputs["union_balls_gif"]
        if not output.exists():
            # Pharmacy positions as an n x 2 array of Web Mercator metres (x, y).
            coordinates = np.column_stack(
                [projected.geometry.x.to_numpy(), projected.geometry.y.to_numpy()]
            )
            # Every pair once: upper holds each pair's two row numbers and
            # distances its straight-line distance in metres.
            _, upper, distances = _pairwise_distances(coordinates)
            # The largest epsilon, shown in the last frame:
            if distances.size == 0:
                # Only one pharmacy, so no pairs: use the setting, or 1,000 m.
                epsilon_max = float(union_gif_eps_max_m or 1000.0)
            elif union_gif_eps_max_m is None:
                # Default: a percentile of all pairwise distances (the 70th unless
                # changed), so about that share of pairs is joined by the end.
                epsilon_max = float(np.quantile(distances, float(union_gif_eps_quantile)))
            else:
                # A fixed maximum in metres from the settings.
                epsilon_max = float(union_gif_eps_max_m)
            # Evenly spaced values from 0 to epsilon_max, one per frame (both ends
            # included); e.g. 140 frames up to 13,900 m go up in steps of 100 m.
            epsilons = np.linspace(0.0, epsilon_max, int(union_gif_frames))
            fig, axis = plt.subplots(figsize=(10, 10))
            # Layers from top to bottom: dots (zorder 7), edges (6), circles, basemap.
            axis.scatter(
                coordinates[:, 0], coordinates[:, 1],
                s=float(point_size) * 0.6, c=point_color, alpha=0.9, zorder=7,
            )
            cx.add_basemap(axis, source=_basemap_source(), attribution=CARTO_ATTRIBUTION)
            axis.set_axis_off()
            axis.set_title(f"{title} — union of balls (r = ε/2)")
            # As for the PNG, but with a 6% margin.
            xmin, ymin, xmax, ymax = projected.total_bounds
            pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1000
            pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1000
            axis.set_xlim(xmin - pad_x, xmax + pad_x)
            axis.set_ylim(ymin - pad_y, ymax + pad_y)
            # One circle per pharmacy, starting at radius 0; update() grows them.
            # Each is very faint (alpha 0.10), so places covered by several circles
            # look darker.
            circles = []
            for x_value, y_value in coordinates:
                circle = Circle((x_value, y_value), radius=0.0, fill=True, alpha=0.10)
                axis.add_patch(circle)
                circles.append(circle)
            # One object holds all the edge lines; update() replaces them every frame.
            # It is only added to the map when edges are switched on.
            edges = LineCollection([], linewidths=0.8, alpha=0.35, zorder=6)
            if union_gif_show_edges:
                axis.add_collection(edges)
            # Text in the bottom left corner, filled in by update() for each frame.
            label = axis.text(0.02, 0.02, "", transform=axis.transAxes)

            def update(frame_index: int):
                """Draw one frame: circles of radius epsilon / 2, plus the edges.

                For this frame's epsilon, every circle gets radius epsilon / 2, and
                (if edges are on) a line joins every pair of pharmacies at most
                epsilon apart, that is, every pair whose circles touch or overlap.
                Returns the pieces that changed, as FuncAnimation expects.
                """
                epsilon = float(epsilons[frame_index])
                radius = epsilon / 2.0
                for circle in circles:
                    circle.set_radius(radius)
                artists = circles + [label]
                if union_gif_show_edges:
                    # True for every pair joined in this frame.
                    mask = distances <= epsilon
                    # The two pharmacies of each joined pair.
                    row_indices, column_indices = upper[0][mask], upper[1][mask]
                    # One line per joined pair, from one pharmacy to the other:
                    # shape (number of pairs, 2 end points, x and y). Early frames
                    # have no pairs yet, so an empty array of that shape is used.
                    segments = (
                        np.stack([coordinates[row_indices], coordinates[column_indices]], axis=1)
                        if len(row_indices)
                        else np.empty((0, 2, 2))
                    )
                    edges.set_segments(segments)
                    artists.append(edges)
                # e.g. "ε=5,000 m | r=2,500 m" (Web Mercator metres, not ground metres).
                label.set_text(f"ε={epsilon:,.0f} m | r={radius:,.0f} m")
                return artists

            # interval is the time between frames in milliseconds (1000 / 3 = about
            # 333 ms); the GIF itself is saved at fps frames per second. max(1, ...)
            # turns a setting of 0 into 1, so there is no division by zero.
            animation = FuncAnimation(
                fig,
                update,
                frames=len(epsilons),
                interval=1000 / max(1, int(union_gif_fps)),
                blit=False,
            )
            animation.save(output, writer="pillow", fps=max(1, int(union_gif_fps)))
            plt.close(fig)
        outputs["union_balls_gif"] = str(output.resolve())

    # 7. Write mapping_metadata.json. It is rewritten on every run, so created_utc
    #    is the time of the latest run, not of the first drawing.
    metadata = {
        "pipeline_step": 6,
        ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
        "city": paths.city,
        "state": paths.state.upper(),
        "created_utc": utc_now_iso(),
        "source_tract_file": str(paths.step02_tract_csv.resolve()),
        "source_tract_sha256": source_hash,
        "outputs": outputs,
        "settings": requested_settings,
    }
    write_json(paths.step06_metadata, metadata)
    return metadata


# =============================================================================
# Scenario maps: which pharmacies each removal scenario keeps and removes
# =============================================================================
# One static PNG per removal scenario, drawn from the saved Step 5 scenario plan
# (scenario_index.csv and the membership files). The maps need only the plan, not
# the persistent homology results, so they can be made as soon as Step 5 has saved
# its plan. They sit in their own folder with their own metadata file, so the
# pharmacy map, the GIF and mapping_metadata.json are not affected (the top of this
# file explains why).

SCENARIO_MAP_FOLDER = "scenario_maps"  # inside step06_mapping/
SCENARIO_MAP_INCLUDED_COLOR = "#2a78d6"  # blue: pharmacies the scenario keeps
SCENARIO_MAP_EXCLUDED_COLOR = "#eb6834"  # orange: pharmacies the scenario removes
SCENARIO_MAP_TEXT_COLOR = "#222222"  # near-black, for the title and legend
SCENARIO_MAP_DPI = 150  # the pharmacy PNG uses 300 dpi; lower here to save space
# Label for the basemap (CARTO Positron; the tiles themselves come from
# _basemap_source). It is saved with the settings in scenario_maps_metadata.json,
# so if the basemap and this label change, existing maps drawn on the old basemap
# stop the run instead of being mixed with new ones.
SCENARIO_MAP_BASEMAP = "CARTO light_all (API key)"
# Not mapped: the full baseline removes nothing, so its map would only repeat
# pharmacies_contextily.png; leave-one-out would make one map per pharmacy, each
# with a single orange dot.
SCENARIO_TYPES_NOT_MAPPED = ("baseline", "leave_one_out")

# Earth radius used by Web Mercator (the WGS 84 equatorial radius), in metres.
_WEB_MERCATOR_RADIUS_M = 6378137.0


# =============================================================================
# Scenario maps: projection, map extent and basemap
# =============================================================================

def _web_mercator(longitude, latitude) -> tuple[np.ndarray, np.ndarray]:
    """Convert longitude and latitude (degrees) to Web Mercator x, y (metres, EPSG:3857).

        x = R * longitude_in_radians
        y = R * ln(tan(pi / 4 + latitude_in_radians / 2)),   R = 6,378,137 m

    Example: Milwaukee (longitude -87.9065, latitude 43.0389) is at about
    x = -9,785,707 m, y = 5,317,895 m.

    This is the projection of the basemap tiles, and the same conversion the
    pharmacy map gets from GeoPandas with ''to_crs(epsg=3857)''; written out here,
    it means the scenario maps do not need GeoPandas. Distances in these metres
    are stretched by about 1/cos(latitude) compared with the ground (about 1.37x
    at Milwaukee), which is fine for drawing but not for measuring.
    """
    lon = np.radians(np.asarray(longitude, dtype=float))
    lat = np.radians(np.asarray(latitude, dtype=float))
    x = _WEB_MERCATOR_RADIUS_M * lon
    y = _WEB_MERCATOR_RADIUS_M * np.log(np.tan(np.pi / 4.0 + lat / 2.0))
    return x, y


def _padded_bounds(x: np.ndarray, y: np.ndarray, pad_share: float = 0.05):
    """Return the map extent (xmin, ymin, xmax, ymax) with a margin on every side.

    The margin on each side is ''pad_share'' of the width (or height): 5% by
    default, as on the pharmacy map. Example: points from x = 0 to x = 20,000 m
    give xmin = -1,000 and xmax = 21,000. If every point has the same x (or y),
    the width there is 0, so a fixed 1,000 m margin is used instead.
    """
    xmin, xmax, ymin, ymax = float(x.min()), float(x.max()), float(y.min()), float(y.max())
    pad_x = (xmax - xmin) * pad_share if xmax > xmin else 1000.0
    pad_y = (ymax - ymin) * pad_share if ymax > ymin else 1000.0
    return xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y


def _fetch_basemap(bounds, provider_name: str):
    """Download the basemap once per city; every scenario map of the city reuses it.

    ''bounds'' is the city's extent from _padded_bounds, in Web Mercator metres
    (ll=False tells contextily these are metres, not longitude/latitude).
    contextily chooses the zoom level from the size of the area (zoom="auto"),
    downloads the tiles that cover it and joins them into one image. Every
    scenario map of a city has the same extent, so one download serves them all:
    quicker, and fewer tiles counted towards CARTO's free monthly limit. Needs
    the CARTO API key (see _basemap_source).

    Returns (image, extent, attribution): the image as an array of pixels, the
    area it covers in Web Mercator metres (whole tiles, so usually a little more
    than ''bounds''), and the credit line to print on the map.

    ''provider_name'' is not used; the tile address always comes from
    _basemap_source.
    """
    import contextily as cx

    xmin, ymin, xmax, ymax = bounds
    image, extent = cx.bounds2img(
        xmin, ymin, xmax, ymax, zoom="auto", source=_basemap_source(), ll=False
    )
    return image, extent, CARTO_ATTRIBUTION


def _add_attribution(axis, text: str) -> None:
    """Print the basemap credit line on a map, as contextily.add_basemap does.

    Nothing is added if ''text'' is empty. If contextily's own function fails, a
    small grey caption is written in the bottom left corner instead.
    """
    if not text:
        return
    try:
        import contextily as cx

        cx.add_attribution(axis, text)
    except Exception:  # reached only if contextily's add_attribution fails: plain caption
        axis.text(0.005, 0.005, text, transform=axis.transAxes, fontsize=7, color="#555555")


# =============================================================================
# Scenario maps: reading the plan and drawing one map
# =============================================================================

def _read_scenario_membership(
    paths: CityRunPaths,
    scenario_key: str,
    manifest_ids: list[str],
    expected_excluded_ids: list[str],
) -> pd.DataFrame:
    """Read one membership file and check it against the manifest and the index.

    The file is step05_tda/scenario_plan/memberships/<scenario_key>.csv, written
    by Step 5: one row per pharmacy in manifest order, with an ''included''
    column (True = kept, False = removed). These are the same files Step 5 used
    for its PH results, so the map shows exactly the pharmacies those results
    left out. Checks, in order (any failure stops Step 6):

    1. the file exists;
    2. every row has this run's tag;
    3. the pharmacy IDs are unique and in exactly the same order as the Step 3
       manifest. This matters because the map takes the positions from the
       manifest and the True/False flags from this file, row by row, so a
       different order would colour the wrong pharmacies;
    4. ''included'' holds only True/False (pandas reads the column that way only
       when every cell is True or False, so a blank or other text fails);
    5. the removed pharmacies are the same set as the scenario's
       ''excluded_pharmacy_ids'' in scenario_index.csv.

    Returns the membership table.
    """
    path = paths.scenario_membership_dir / f"{scenario_key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Scenario membership file is missing: {path}")
    membership = pd.read_csv(
        path, dtype={"pharmacy_id": "string", ANALYSIS_TAG_COLUMN: "string"}
    )
    validate_analysis_tag(membership, paths.analysis_run_tag, str(path))
    ids = validate_pharmacy_ids(membership, str(path))
    # Same IDs in the same order (a list comparison, so order counts).
    if ids != manifest_ids:
        raise ValueError(f"Pharmacies in {path} do not match the Step 3 manifest.")
    if not pd.api.types.is_bool_dtype(membership["included"]):
        raise ValueError(f"The 'included' column in {path} must contain only True/False.")
    excluded = membership.loc[~membership["included"], "pharmacy_id"].astype(str).tolist()
    # Sorted on both sides so only the set of IDs is compared, not their order.
    if sorted(excluded) != sorted(expected_excluded_ids):
        raise ValueError(
            f"Excluded pharmacies in {path} differ from scenario_index.csv for {scenario_key!r}."
        )
    return membership


def _scenario_subtitle(row: pd.Series, membership: pd.DataFrame, replicate_total: int) -> str:
    """Build the grey line under a map's title that says how the pharmacies were chosen.

    Parts are joined by " | ", each only when it applies:
    - random removals: "Replicate 2 of 3" (''replicate_total'' is the number of
      replicates of that scenario) and "random seed 12345" (the seed set in
      Step 0; all replicates of a scenario share it, so it is the same on each);
    - chain removals: "name matches CVS" (the search terms that matched at least
      one removed pharmacy's name, from the ''matched_term'' column).

    Example: "Replicate 2 of 3 | random seed 12345". An explicit-ID scenario has
    none of these, so its subtitle is "" and no subtitle line is drawn.
    """
    parts = []
    if pd.notna(row.get("replicate_number")):
        parts.append(f"Replicate {int(row['replicate_number'])} of {replicate_total}")
    if pd.notna(row.get("random_seed")):
        parts.append(f"random seed {int(row['random_seed'])}")
    if "matched_term" in membership.columns:
        terms = sorted(membership["matched_term"].dropna().astype(str).unique())
        if terms:
            parts.append("name matches " + ", ".join(terms))
    return " | ".join(parts)


def _draw_scenario_map(
    *,
    x: np.ndarray,
    y: np.ndarray,
    included: np.ndarray,
    bounds,
    basemap,
    title: str,
    subtitle: str,
    point_size: float,
    output: Path,
) -> None:
    """Draw and save one scenario map: kept pharmacies blue, removed ones orange on top.

    Inputs: ''x'' and ''y'' are every pharmacy's Web Mercator position, in
    manifest order; ''included'' is True for the pharmacies the scenario keeps;
    ''bounds'' is the city's shared extent; ''basemap'' is the (image, extent,
    attribution) from _fetch_basemap; ''output'' is the PNG path (its folder is
    created if needed). If the image is None, no basemap is drawn;
    create_scenario_maps always passes a downloaded one.

    Removed pharmacies are drawn after the kept ones, so an orange dot is never
    hidden under a blue one, and every dot has a thin white outline. The legend
    gives both counts, e.g. "Included (37)" and "Excluded (4)". The figure is
    saved at SCENARIO_MAP_DPI with the empty border trimmed (bbox_inches="tight").
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    image, extent, attribution = basemap
    xmin, ymin, xmax, ymax = bounds
    fig, axis = plt.subplots(figsize=(10, 10))
    if image is not None:
        # The basemap at its true position in metres, at the bottom (zorder 0).
        axis.imshow(image, extent=extent, interpolation="bilinear", zorder=0)
    marker_style = {"s": float(point_size), "edgecolors": "white", "linewidths": 0.6}
    axis.scatter(
        x[included], y[included], color=SCENARIO_MAP_INCLUDED_COLOR, zorder=2, **marker_style
    )
    axis.scatter(
        x[~included], y[~included], color=SCENARIO_MAP_EXCLUDED_COLOR, zorder=3, **marker_style
    )
    # Show exactly the shared extent, so every scenario map of a city lines up.
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")  # one metre across is drawn as long as one metre up
    axis.set_axis_off()
    _add_attribution(axis, attribution)

    # The legend sits below the map so it never covers a pharmacy.
    legend_marker = {"marker": "o", "linestyle": "", "markersize": 8, "markeredgecolor": "white"}
    handles = [
        Line2D([], [], markerfacecolor=SCENARIO_MAP_INCLUDED_COLOR,
               label=f"Included ({int(included.sum())})", **legend_marker),
        Line2D([], [], markerfacecolor=SCENARIO_MAP_EXCLUDED_COLOR,
               label=f"Excluded ({int((~included).sum())})", **legend_marker),
    ]
    axis.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2,
        frameon=False, fontsize=11, labelcolor=SCENARIO_MAP_TEXT_COLOR,
    )
    # A larger gap under the title leaves room for the subtitle line, if there is one.
    axis.set_title(title, fontsize=14, color=SCENARIO_MAP_TEXT_COLOR, pad=22 if subtitle else 8)
    if subtitle:
        # Centred just above the top edge of the map.
        axis.text(
            0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", va="bottom",
            fontsize=10, color="#555555",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=SCENARIO_MAP_DPI, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Scenario maps for one city: runs the pieces above in order
# =============================================================================

def create_scenario_maps(paths: CityRunPaths, *, point_size: float = 30.0) -> dict[str, Any]:
    """Draw one map per removal scenario for one city, from the saved Step 5 plan.

    Step 0 calls this after create_city_maps when MAKE_SCENARIO_MAPS is True,
    passing MAP_POINT_SIZE. Maps are written to
    ''step06_mapping/scenario_maps/<scenario_key>.png''; for example
    ''scenario_maps/remove_10_percent_random/replicate_001.png''. The numbered
    comments below match the second numbered list at the top of this file.

    Which scenarios: every row of scenario_index.csv except the baseline and the
    leave-one-out scenarios (see SCENARIO_TYPES_NOT_MAPPED). If nothing is left,
    no maps are drawn and the metadata lists none.

    One extent and one basemap per city: the extent is set from all pharmacies
    in the manifest, not only the kept ones, so every map of a city shows the
    same area at the same scale and the maps can be flipped through and compared
    directly. The shared extent also means a single basemap download serves every
    map, and it is downloaded only if at least one map has to be drawn.

    Skips and stops (see SKIP AND STOP RULES at the top of this file): an existing
    scenario_maps_metadata.json must have this run's tag, the same Step 5
    plan_hash and the same settings, or Step 6 stops; then existing maps are
    skipped and missing ones drawn. PNG files without the metadata file stop
    Step 6. Every membership file is checked, even when its map is skipped.

    Returns the metadata dictionary that is also saved as
    scenario_maps_metadata.json, and prints how many maps were made, already
    existed, or were not mapped.
    """
    # 1. The Step 5 plan, the scenario index and the Step 3 manifest must exist.
    for required, label in [
        (paths.scenario_plan_json, "Step 5 scenario plan"),
        (paths.scenario_index_csv, "Step 5 scenario index"),
        (paths.step03_manifest, "Step 3 manifest"),
    ]:
        if not required.exists():
            raise FileNotFoundError(
                f"Scenario maps need the {label} for {paths.city_state}: {required}. "
                "Run Step 5 first, or set MAKE_SCENARIO_MAPS = False."
            )
    plan = read_json(paths.scenario_plan_json)
    # A plan with no tag recorded passes this check (the default is this run's tag).
    if plan.get(ANALYSIS_TAG_COLUMN, paths.analysis_run_tag) != paths.analysis_run_tag:
        raise ValueError("Step 5 scenario plan belongs to another analysis_run_tag.")
    # The plan's fingerprint (a SHA-256 hash of its contents). It is saved with the
    # maps so that maps from a different plan are never mixed in.
    plan_hash = plan.get("plan_hash")

    # 2. Pharmacy positions from the manifest (the membership files use the same
    #    row order), and one extent for every scenario map of this city.
    manifest = pd.read_csv(
        paths.step03_manifest, dtype={"pharmacy_id": "string", ANALYSIS_TAG_COLUMN: "string"}
    )
    manifest_ids = validate_pharmacy_ids(manifest, str(paths.step03_manifest))
    x_all, y_all = _web_mercator(manifest["longitude"], manifest["latitude"])
    bounds = _padded_bounds(x_all, y_all)  # same extent for every scenario in a city

    # 3. The scenarios to map: every row of the index except the baseline and
    #    leave-one-out. IDs are read as text; an empty cell (nothing removed)
    #    becomes missing.
    index = pd.read_csv(paths.scenario_index_csv, dtype={"excluded_pharmacy_ids": "string"})
    mapped = index[~index["scenario_type"].isin(SCENARIO_TYPES_NOT_MAPPED)]
    # Rows per scenario name, for the "Replicate 2 of 3" subtitle: the number of
    # replicates of a random scenario, and 1 for any other scenario.
    replicate_totals = mapped["scenario_name"].value_counts().to_dict()

    # 4. Compare with any earlier scenario maps for this city.
    output_dir = paths.step06_dir / SCENARIO_MAP_FOLDER
    metadata_path = output_dir / "scenario_maps_metadata.json"
    # Everything that changes how a scenario map looks. Compared with the saved
    # settings on later runs; any difference stops Step 6.
    settings = {
        "included_color": SCENARIO_MAP_INCLUDED_COLOR,
        "excluded_color": SCENARIO_MAP_EXCLUDED_COLOR,
        "point_size": float(point_size),
        "dpi": SCENARIO_MAP_DPI,
        "basemap": SCENARIO_MAP_BASEMAP,
        "scenario_types_not_mapped": list(SCENARIO_TYPES_NOT_MAPPED),
    }
    if metadata_path.exists():
        saved = read_json(metadata_path)
        if saved.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
            raise ValueError("Scenario map metadata belongs to another analysis_run_tag.")
        if saved.get("plan_hash") != plan_hash or saved.get("settings") != settings:
            raise RuntimeError(
                f"Existing scenario maps in {output_dir} were made from a different Step 5 "
                "plan or with different map settings. Use a new ANALYSIS_RUN_TAG, or move "
                "that folder elsewhere to make the maps again."
            )
    elif output_dir.exists() and any(output_dir.rglob("*.png")):
        # PNGs but no metadata: an earlier run stopped before writing it (it is
        # written last), or the files came from elsewhere. Their plan and settings
        # are unknown, so stop. rglob also searches subfolders such as
        # remove_10_percent_random/.
        raise RuntimeError(
            f"Scenario map files exist in {output_dir} without scenario_maps_metadata.json."
        )

    # 5. One map per scenario.
    city_label = f"{paths.city}, {paths.state.upper()}"
    outputs: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    basemap = None
    made = skipped = 0
    for _, row in mapped.iterrows():
        scenario_key = str(row["scenario_key"])
        # The removed pharmacy IDs, stored joined by "|"; a missing or empty cell
        # means none.
        excluded_ids = (
            str(row["excluded_pharmacy_ids"]).split("|")
            if pd.notna(row["excluded_pharmacy_ids"]) and str(row["excluded_pharmacy_ids"])
            else []
        )
        # Checked even when the map already exists, so the counts below are always
        # filled in and a membership file that no longer matches is caught.
        membership = _read_scenario_membership(paths, scenario_key, manifest_ids, excluded_ids)
        # True = kept (blue), False = removed (orange), in manifest order.
        included = membership["included"].to_numpy(dtype=bool)
        # scenario_key can include a folder, e.g. remove_10_percent_random/replicate_001.
        output = output_dir / f"{scenario_key}.png"
        if output.exists():
            skipped += 1
        else:
            if basemap is None:  # download the basemap only if a map needs drawing
                basemap = _fetch_basemap(bounds, SCENARIO_MAP_BASEMAP)
            _draw_scenario_map(
                x=x_all,
                y=y_all,
                included=included,
                bounds=bounds,
                basemap=basemap,
                # e.g. "Milwaukee, WI: remove_10_percent_random"
                title=f"{city_label}: {row['scenario_name']}",
                subtitle=_scenario_subtitle(
                    row, membership, replicate_totals.get(row["scenario_name"], 1)
                ),
                point_size=point_size,
                output=output,
            )
            made += 1
        # Listed for every mapped scenario, whether drawn now or on an earlier run.
        outputs[scenario_key] = str(output.resolve())
        counts[scenario_key] = {
            "included": int(included.sum()),
            "excluded": int((~included).sum()),
        }

    # 6. Write scenario_maps_metadata.json (rewritten on every run) and print a
    #    one-line summary.
    metadata = {
        "pipeline_step": 6,
        ANALYSIS_TAG_COLUMN: paths.analysis_run_tag,
        "city": paths.city,
        "state": paths.state.upper(),
        "created_utc": utc_now_iso(),
        "plan_hash": plan_hash,
        "scenario_index_file": str(paths.scenario_index_csv.resolve()),
        "settings": settings,
        "outputs": outputs,
        "pharmacy_counts": counts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, metadata)
    print(
        f"Step 6 scenario maps for {city_label}: {made} made, {skipped} already existed, "
        f"{len(index) - len(mapped)} scenario(s) not mapped (baseline/leave-one-out)."
    )
    return metadata
