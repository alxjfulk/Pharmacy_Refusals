# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 08:57:39 2025

@author: bigfo
"""

# -*- coding: utf-8 -*-
"""
Variant 2: Single-threaded Dijkstra loop without multiprocessing
"""
import os
from datetime import date
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox


def compute_full_drive_matrix(G, nodes, weight='travel_time'):
    """
    Compute full drive-time matrix using Dijkstra's algorithm.
    Returns an n x n numpy array, with n = len(nodes).
    """
    n = len(nodes)
    D = np.full((n, n), np.nan, dtype=float)
    for i, src in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, src, weight=weight) #https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length.html
        for j, tgt in enumerate(nodes):
            D[i, j] = lengths.get(tgt, np.nan)
    return D


def compute_full_walk_matrix(G, nodes, length_weight='length', speed_mps=1.42):
    """
    Compute full walk-time matrix: Dijkstra on 'length' (meters),
    then convert to seconds via speed (m/s).
    Returns symmetric n x n numpy array.
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


def compute_matrices_dijkstra(
    input_csv_path: str,
    output_dir: str,
    city: str,
    state: str,
    do_drive: bool = True,
    do_walk: bool = True
):
    """
    Reads CSV with 'latitude' & 'longitude', snaps to OSMnx graph via bbox,
    then computes full travel-time matrices via Dijkstra (no multiprocessing).
    Saves .npy files for drive & walk matrices to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    today = date.today().isoformat()

    # load points
    df = pd.read_csv(input_csv_path)
    df = df.dropna(subset=['latitude', 'longitude'])
    df['lat'] = df['latitude'].astype(float)
    df['lon'] = df['longitude'].astype(float)
    n = len(df)
    if n == 0:
        raise ValueError("No valid coordinates found in input CSV.")

    # build bbox (west, south, east, north)
    minx, maxx = df['lon'].min(), df['lon'].max()
    miny, maxy = df['lat'].min(), df['lat'].max()
    buf = 0.02  # ~2km buffer
    west, south, east, north = minx - buf, miny - buf, maxx + buf, maxy + buf
    bbox = (west, south, east, north)

    # DRIVE
    if do_drive:
        print(f"→ Computing drive‐time for {city}, {state}")
        Gd = ox.graph_from_bbox(
            bbox,
            network_type='drive',
            simplify=False,
            retain_all=False,
            truncate_by_edge=False
        )
        # add travel-time attributes
        ox.add_edge_speeds(Gd)
        ox.add_edge_travel_times(Gd)
        nodes_drive = ox.distance.nearest_nodes(Gd, df['lon'], df['lat'])
        print("  ✔ Drive network loaded & snapped")

        D = compute_full_drive_matrix(Gd, list(nodes_drive), weight='travel_time')
        drive_path = os.path.join(output_dir, f"{city}{state}_drive_{today}.npy")
        np.save(drive_path, D)
        print(f"  ✔ Saved drive matrix → {drive_path}")

    # WALK
    if do_walk:
        print(f"→ Computing walk‐time for {city}, {state}")
        Gw = ox.graph_from_bbox(
            bbox,
            network_type='walk',
            simplify=True
        )
        nodes_walk = ox.distance.nearest_nodes(Gw, df['lon'], df['lat'])
        print("  ✔ Walk network loaded & snapped")

        W = compute_full_walk_matrix(Gw, list(nodes_walk), length_weight='length', speed_mps=1.42)
        walk_path = os.path.join(output_dir, f"{city}{state}_walk_{today}.npy")
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
    parser.add_argument("--no-drive", action='store_true', help="Skip drive-time computation")
    parser.add_argument("--no-walk",  action='store_true', help="Skip walk-time computation")
    args = parser.parse_args()
    compute_matrices_dijkstra(
        args.input_csv,
        args.output_dir,
        args.city,
        args.state,
        do_drive=not args.no_drive,
        do_walk= not args.no_walk
    )
