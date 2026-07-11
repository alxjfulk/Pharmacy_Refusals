# -*- coding: utf-8 -*-
"""
Variant 2: Single-threaded Dijkstra loop without multiprocessing.

Updated for pipeline consistency:
- Step 3 now accepts a pipeline run tag from the master process flow.
- Raw travel matrices are saved as:
      <City><State>_drive_<TAG>.npy
      <City><State>_walk_<TAG>.npy
  which matches Step 4's expected input pattern.
"""

import os
from datetime import date
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox


def compute_full_drive_matrix(G, nodes, weight='travel_time'):
    """
    Compute a full directed drive-time matrix using Dijkstra's algorithm.
    Returns an n x n numpy array, where n = len(nodes).
    """
    n = len(nodes)
    D = np.full((n, n), np.nan, dtype=float)
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=weight)
        for j, tgt in enumerate(nodes):
            D[i, j] = lengths.get(tgt, np.nan)
    return D


def compute_full_walk_matrix(G, nodes, length_weight='length', speed_mps=1.42):
    """
    Compute a full walk-time matrix.

    The shortest path is calculated using edge length in meters and then converted
    to seconds using speed_mps. Walking paths are treated as symmetric in this
    implementation, so W[i, j] and W[j, i] are filled with the same value.
    """
    n = len(nodes)
    W = np.full((n, n), np.nan, dtype=float)
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=length_weight)
        for j, tgt in enumerate(nodes):
            dist_m = lengths.get(tgt)
            if dist_m is not None:
                secs = dist_m / speed_mps
                W[i, j] = secs
                W[j, i] = secs
    return W


def _matrix_prefix(city: str, state: str, output_prefix: str | None = None) -> str:
    """
    Return the filename prefix used across Steps 2-4.

    Step 2 writes tract files as <City><State>_tract_data.csv. Step 3 should use
    that same <City><State> prefix so Step 4 can pair tract files with raw travel
    matrices using the pattern <City><State>_{drive|walk}_<TAG>.npy.
    """
    if output_prefix is not None:
        return str(output_prefix).strip()
    return f"{str(city).strip()}{str(state).strip()}"


def compute_matrices_dijkstra(
    input_csv_path: str,
    output_dir: str,
    city: str,
    state: str,
    do_drive: bool = True,
    do_walk: bool = True,
    travel_tag: str | None = None,
    output_prefix: str | None = None,
    bbox_buffer_degrees: float = 0.02,
):
    """
    Read a CSV with latitude/longitude, snap points to OSMnx graphs built from a
    bounding box, and compute full walk/drive travel-time matrices via Dijkstra.

    Parameters
    ----------
    input_csv_path : str
        Path to <City><State>_tract_data.csv from Step 2.
    output_dir : str
        Folder where raw .npy travel matrices are saved.
    city, state : str
        City name and state abbreviation, used for messages and default filenames.
    do_drive, do_walk : bool
        Whether to compute each travel mode.
    travel_tag : str or None
        Run tag supplied by the master process flow. If None, today's date is used.
    output_prefix : str or None
        Optional explicit prefix for output files. Defaults to <City><State>.
    bbox_buffer_degrees : float
        Degree buffer added around the point bounding box before requesting OSMnx
        graphs. The default preserves the prior behavior.

    Output filenames
    ----------------
    <prefix>_drive_<travel_tag>.npy
    <prefix>_walk_<travel_tag>.npy
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = str(travel_tag).strip() if travel_tag else date.today().isoformat()
    prefix = _matrix_prefix(city, state, output_prefix=output_prefix)

    # Load points.
    df = pd.read_csv(input_csv_path)
    df = df.dropna(subset=['latitude', 'longitude']).copy()
    df['lat'] = df['latitude'].astype(float)
    df['lon'] = df['longitude'].astype(float)
    n = len(df)
    if n == 0:
        raise ValueError("No valid coordinates found in input CSV.")

    # Build bbox (west, south, east, north). This avoids city-boundary graph
    # exclusions when pharmacies fall outside the municipal polygon.
    minx, maxx = df['lon'].min(), df['lon'].max()
    miny, maxy = df['lat'].min(), df['lat'].max()
    buf = float(bbox_buffer_degrees)
    west, south, east, north = minx - buf, miny - buf, maxx + buf, maxy + buf
    bbox = (west, south, east, north)

    # DRIVE
    if do_drive:
        print(f"→ Computing drive-time for {city}, {state}")
        Gd = ox.graph_from_bbox(
            bbox,
            network_type='drive',
            simplify=False,
            retain_all=False,
            truncate_by_edge=False
        )
        ox.add_edge_speeds(Gd)
        ox.add_edge_travel_times(Gd)
        nodes_drive = ox.distance.nearest_nodes(Gd, df['lon'], df['lat'])
        print("  ✔ Drive network loaded & snapped")

        D = compute_full_drive_matrix(Gd, list(nodes_drive), weight='travel_time')
        drive_path = os.path.join(output_dir, f"{prefix}_drive_{tag}.npy")
        np.save(drive_path, D)
        print(f"  ✔ Saved drive matrix → {drive_path}")

    # WALK
    if do_walk:
        print(f"→ Computing walk-time for {city}, {state}")
        Gw = ox.graph_from_bbox(
            bbox,
            network_type='walk',
            simplify=True
        )
        nodes_walk = ox.distance.nearest_nodes(Gw, df['lon'], df['lat'])
        print("  ✔ Walk network loaded & snapped")

        W = compute_full_walk_matrix(Gw, list(nodes_walk), length_weight='length', speed_mps=1.42)
        walk_path = os.path.join(output_dir, f"{prefix}_walk_{tag}.npy")
        np.save(walk_path, W)
        print(f"  ✔ Saved walk matrix → {walk_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute raw walk & drive time matrices via Dijkstra loops"
    )
    parser.add_argument("input_csv", help="CSV file with latitude & longitude columns")
    parser.add_argument("output_dir", help="Directory to save .npy matrices")
    parser.add_argument("city", help="City name for messages")
    parser.add_argument("state", help="State abbreviation for messages")
    parser.add_argument("--travel-tag", default=None, help="Tag appended to output filenames; defaults to today's date")
    parser.add_argument("--output-prefix", default=None, help="Optional output filename prefix; defaults to <City><State>")
    parser.add_argument("--bbox-buffer-degrees", type=float, default=0.02, help="Buffer around point bbox in degrees")
    parser.add_argument("--no-drive", action='store_true', help="Skip drive-time computation")
    parser.add_argument("--no-walk",  action='store_true', help="Skip walk-time computation")
    args = parser.parse_args()
    compute_matrices_dijkstra(
        args.input_csv,
        args.output_dir,
        args.city,
        args.state,
        do_drive=not args.no_drive,
        do_walk=not args.no_walk,
        travel_tag=args.travel_tag,
        output_prefix=args.output_prefix,
        bbox_buffer_degrees=args.bbox_buffer_degrees,
    )
