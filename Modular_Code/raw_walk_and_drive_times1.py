# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:58:51 2025

@author: bigfo
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import multiprocess as mp
from datetime import date

def compute_matrices(
    input_csv_path: str,
    output_dir: str,
    city: str,
    state: str,
    do_drive: bool = True,
    do_walk: bool = True
):
    """
    Compute & save drive‐time and/or walk‐time matrices from a CSV of points,
    parametrized by city & state for both network lookup and file naming.

    Parameters
    ----------
    input_csv_path : str
        Path to a CSV with 'latitude' and 'longitude' columns.
    output_dir : str
        Directory where the .npy outputs will be saved.
    city : str
        City (or county) name for OSMnx to pull the network for.
    state : str
        State (abbreviation or full name) for OSMnx network lookup.
    do_drive : bool
        If True, compute and save the drive‐time matrix.
    do_walk : bool
        If True, compute and save the walk‐time matrix.
    """
    # ———————————————
    # Setup & I/O prep
    # ———————————————
    # place = f"{city}, {state}"
    place = f"{city} County"
    today = date.today().isoformat()
    file_name_city  = city.replace(' ', '_') #replace spaces in city and state names for saving files
    file_name_state = state.replace(' ', '_')
    os.makedirs(output_dir, exist_ok=True)

    # ———————————————
    # Load & sanitize points
    # ———————————————
    df = pd.read_csv(input_csv_path)
    df['lat']  = df['latitude'].astype(float)
    df['long'] = df['longitude'].astype(float)
    df = df.dropna(subset=['lat', 'long'])
    n = len(df)
    print(f"✔ Loaded {n} points from {input_csv_path}")

    # precompute index pairs
    pairs_walk  = [(i, j) for i in range(n) for j in range(i+1, n)]
    pairs_drive = [(i, j) for i in range(n) for j in range(n) if i != j]

    # ———————————————
    # DRIVE‐TIME SEGMENT
    # ———————————————
    if do_drive:
        print(f"→ Computing drive‐time for {place}")
        Gd = ox.graph_from_place(place, network_type='drive', simplify=False)
        ox.routing.add_edge_speeds(Gd, fallback=40.2336)
        ox.routing.add_edge_travel_times(Gd)
        nodes_drive = ox.distance.nearest_nodes(Gd, df.long, df.lat)
        print("  ✔ Drive network loaded & points snapped")

        def _drive_time(pair):
            i, j = pair
            return nx.shortest_path_length(
                Gd,
                nodes_drive[i],
                nodes_drive[j],
                weight='travel_time'
            )

        with mp.Pool(round(1.4 * mp.cpu_count())) as pool:
            drive_results = pool.map(_drive_time, pairs_drive)
        D = np.zeros((n, n))
        for idx, (i, j) in enumerate(pairs_drive):
            D[i, j] = drive_results[idx]
        drive_file = os.path.join(
            output_dir,
            f"drive_time_{file_name_city}_{file_name_state}_{today}.npy"
        )
        np.save(drive_file, D)
        print(f"  ✔ Saved drive matrix → {drive_file}")

    # ———————————————
    # WALK‐TIME SEGMENT
    # ———————————————
    if do_walk:
        print(f"→ Computing walk‐time for {place}")
        Gw = ox.graph_from_place(place, network_type='walk', simplify=True)
        nodes_walk = ox.distance.nearest_nodes(Gw, df.long, df.lat)
        print("  ✔ Walk network loaded & points snapped")

        def _walk_time(pair):
            i, j = pair
            length_m = nx.shortest_path_length(
                Gw,
                nodes_walk[i],
                nodes_walk[j],
                weight='length'
            )
            return length_m / 1.42

        with mp.Pool(round(1.4 * mp.cpu_count())) as pool:
            walk_results = pool.map(_walk_time, pairs_walk)
        W = np.zeros((n, n))
        for idx, (i, j) in enumerate(pairs_walk):
            W[i, j] = W[j, i] = walk_results[idx]
        walk_file = os.path.join(
            output_dir,
            f"walk_time_{file_name_city}_{file_name_state}_{today}.npy"
        )
        np.save(walk_file, W)
        print(f"  ✔ Saved walk matrix → {walk_file}")
