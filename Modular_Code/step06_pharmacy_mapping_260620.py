# -*- coding: utf-8 -*-
"""
pharmacy_mapping_modular_20260208_full.py

Purpose
-------
Create maps for every "*_tract_data.csv" in a directory using FOUR methods:

1) Static PNG via GeoPandas + Contextily (basemap tiles w/ roads)
2) Interactive HTML via Folium (Leaflet) with clustered markers
3) Animated GIF via Contextily showing union-of-balls ("growing circles") over the basemap
4) Interactive HTML via Folium with a time slider approximating "growing circles"

Key changes (2026-02-08)
-----------------------
- Adds union-of-balls outputs to the existing mapping module without requiring copy/paste.
- Adds batch-level toggles to enable/disable any output type per run.

Expected input CSV columns
--------------------------
At minimum, each CSV must contain latitude/longitude columns.
This module tries these column name pairs (in order):
  - ("latitude", "longitude")
  - ("lat", "lon")
  - ("LATITUDE", "LONGITUDE")

Optional columns used for popup/tooltips (Folium):
  - OrganizationName
  - AddressLocation
  - TRACT
  - Poverty_Rate
  - Vehicles_per_capita

Notes
-----
Union-of-balls mapping for Vietoris–Rips:
- VR edges appear when dist(i,j) <= ε
- The union-of-balls intuition uses circle radius r = ε/2 (balls touch when dist=2r)

Geo warning:
- For the Contextily GIF overlay we project points to EPSG:3857 to match map tiles; distances are
  computed in that projected coordinate space (meters-ish).
- For the Folium slider we keep points in lat/lon and Leaflet interprets circle radius in meters;
  the included slider implementation is an *approximation* (see function docstring).
"""

from __future__ import annotations

import os
import glob
import csv
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# NOTE: geopandas/contextily are required for the static plot + contextily GIF
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx

# Matplotlib animation bits (contextily GIF)
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

# NOTE: folium is only required for the interactive maps
import folium
from folium.plugins import MarkerCluster, TimestampedGeoJson


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _find_latlon_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (lat_col, lon_col) from df using a few common conventions."""
    candidates = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("LATITUDE", "LONGITUDE"),
    ]
    cols = set(df.columns)
    for latc, lonc in candidates:
        if latc in cols and lonc in cols:
            return latc, lonc
    raise ValueError(
        "Could not find latitude/longitude columns. Expected one of: "
        "('latitude','longitude'), ('lat','lon'), ('LATITUDE','LONGITUDE')."
    )


def _clean_points(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    """Coerce lat/lon to numeric and drop invalid rows."""
    out = df.copy()

    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    pre = len(out)
    out = out.dropna(subset=[lat_col, lon_col]).copy()
    post = len(out)

    # Range sanity check (cheap + avoids weird inputs)
    out = out[(out[lat_col].between(-90, 90)) & (out[lon_col].between(-180, 180))].copy()
    post2 = len(out)

    print(f"   - rows: {pre:,} → {post:,} after NaN drop → {post2:,} after range filter")
    return out


def _stem_from_csv_path(csv_path: str) -> str:
    """Build a friendly stem for outputs based on the CSV filename."""
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return base.replace("_tract_data", "")


def _compute_pairwise_distances(X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Compute full pairwise Euclidean distances D for X (n,2)."""
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(X.shape[0], k=1)
    dists = D[iu]
    return D, iu, dists


# -----------------------------------------------------------------------------
# Contextily (static) mapping
# -----------------------------------------------------------------------------

def plot_points_contextily(
    csv_path: str,
    out_png: str,
    point_color: str = "red",
    point_size: float = 30.0,
    basemap: object = None,
    title: Optional[str] = None,
) -> Optional[str]:
    """Create a static PNG map (roads basemap) for a single CSV."""
    basemap = basemap or cx.providers.OpenStreetMap.Mapnik

    print(f"\n[contextily] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)

    lat_col, lon_col = _find_latlon_cols(df)
    df = _clean_points(df, lat_col, lon_col)
    if df.empty:
        print("   - no valid points; skipping.")
        return None

    # GeoDataFrame in WGS84 then project to Web Mercator for tiles
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    gdf_3857 = gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_3857.plot(ax=ax, color=point_color, markersize=point_size, alpha=0.85)

    # Basemap beneath points
    cx.add_basemap(ax, source=basemap)

    ax.set_axis_off()
    ax.set_title(title or f"{_stem_from_csv_path(csv_path)} (pharmacies)", fontsize=14)

    # Zoom to points with padding
    xmin, ymin, xmax, ymax = gdf_3857.total_bounds
    pad_x = (xmax - xmin) * 0.05 if xmax > xmin else 1000
    pad_y = (ymax - ymin) * 0.05 if ymax > ymin else 1000
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"   - wrote: {out_png}")
    return out_png


# -----------------------------------------------------------------------------
# Folium (interactive) mapping
# -----------------------------------------------------------------------------

def plot_points_folium(
    csv_path: str,
    out_html: str,
    point_color: str = "red",
    circle_radius: int = 5,
    zoom_start: int = 11,
    title: Optional[str] = None,
) -> Optional[str]:
    """Create an interactive HTML map for a single CSV."""
    print(f"\n[folium] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)

    lat_col, lon_col = _find_latlon_cols(df)
    df = _clean_points(df, lat_col, lon_col)
    if df.empty:
        print("   - no valid points; skipping.")
        return None

    center_lat = float(df[lat_col].mean())
    center_lon = float(df[lon_col].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Optional title block
    if title is None:
        title = f"{_stem_from_csv_path(csv_path)} (pharmacies)"
    title_html = f"""
        <div style="
            position: fixed;
            top: 10px; left: 50px;
            z-index: 9999;
            background-color: white;
            padding: 6px 10px;
            border: 1px solid #999;
            border-radius: 4px;
            font-size: 14px;
        ">
            {title}
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    cluster = MarkerCluster(name="Pharmacies (clustered)").add_to(m)

    # Add markers
    for _, row in df.iterrows():
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        org = row.get("OrganizationName", "Unknown Organization")
        addr = row.get("AddressLocation", "Unknown Address")
        tract = row.get("TRACT", "Unknown Tract")

        popup_html = f"""
        <b>{org}</b><br>
        {addr}<br>
        Tract: {tract}<br>
        """

        if "Poverty_Rate" in df.columns and pd.notna(row.get("Poverty_Rate")):
            try:
                popup_html += f"Poverty_Rate: {float(row['Poverty_Rate']):.2f}%<br>"
            except Exception:
                pass

        if "Vehicles_per_capita" in df.columns and pd.notna(row.get("Vehicles_per_capita")):
            try:
                popup_html += f"Vehicles_per_capita: {float(row['Vehicles_per_capita']):.3f}<br>"
            except Exception:
                pass

        folium.CircleMarker(
            location=[lat, lon],
            radius=int(circle_radius),
            color=point_color,
            fill=True,
            fill_color=point_color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=str(org),
        ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print(f"   - wrote: {out_html}")
    return out_html


# -----------------------------------------------------------------------------
# Union-of-balls on Contextily basemap (animated GIF)
# -----------------------------------------------------------------------------

def plot_union_balls_contextily_gif(
    csv_path: str,
    out_gif: str,
    basemap: object = None,
    n_frames: int = 140,
    fps: int = 12,
    eps_max_m: Optional[float] = None,
    eps_quantile: float = 0.70,
    show_edges: bool = False,
    circle_alpha: float = 0.10,
    edge_alpha: float = 0.35,
    edge_lw: float = 0.8,
    point_color: str = "red",
    point_size: float = 18.0,
    title: Optional[str] = None,
) -> Optional[str]:
    """Animated GIF with Contextily basemap + growing circles (r = ε/2)."""
    basemap = basemap or cx.providers.OpenStreetMap.Mapnik

    print(f"\n[contextily union-balls GIF] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)

    lat_col, lon_col = _find_latlon_cols(df)
    df = _clean_points(df, lat_col, lon_col)
    if df.empty:
        print("   - no valid points; skipping.")
        return None

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    gdf_3857 = gdf.to_crs(epsg=3857)

    X = np.column_stack([gdf_3857.geometry.x.to_numpy(), gdf_3857.geometry.y.to_numpy()])
    _, iu, dists = _compute_pairwise_distances(X)

    if eps_max_m is None:
        eps_max_m = float(np.quantile(dists, float(eps_quantile)))

    eps_values = np.linspace(0.0, float(eps_max_m), int(n_frames))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X[:, 0], X[:, 1], s=float(point_size), c=point_color, alpha=0.9)
    cx.add_basemap(ax, source=basemap)

    ax.set_axis_off()
    ax.set_title(title or f"{_stem_from_csv_path(csv_path)} — Union-of-balls (r=ε/2)", fontsize=14)

    xmin, ymin, xmax, ymax = gdf_3857.total_bounds
    pad_x = (xmax - xmin) * 0.06 if xmax > xmin else 1000
    pad_y = (ymax - ymin) * 0.06 if ymax > ymin else 1000
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    circles: List[Circle] = []
    for i in range(X.shape[0]):
        c = Circle((X[i, 0], X[i, 1]), radius=0.0, fill=True, alpha=float(circle_alpha))
        ax.add_patch(c)
        circles.append(c)

    edge_collection = LineCollection(
    [],
    linewidths=float(edge_lw),
    alpha=float(edge_alpha),
    colors="gold",          # try: "cyan", "magenta", "white", "black"
    zorder=6,)
    if show_edges:
        ax.add_collection(edge_collection)

    txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, ha="left", va="bottom")

    def update(frame_idx: int):
        eps = float(eps_values[frame_idx])
        r = eps / 2.0

        for c in circles:
            c.set_radius(r)

        if show_edges:
            mask = dists <= eps
            ii = iu[0][mask]
            jj = iu[1][mask]
            if len(ii) > 0:
                segments = np.stack([X[ii], X[jj]], axis=1)
            else:
                segments = np.empty((0, 2, 2))
            edge_collection.set_segments(segments)

        txt.set_text(f"ε={eps:,.0f} m | r={r:,.0f} m")
        artists = circles + [txt]
        if show_edges:
            artists.append(edge_collection)
        return artists

    anim = FuncAnimation(fig, update, frames=len(eps_values), interval=1000 / max(1, int(fps)), blit=False)

    os.makedirs(os.path.dirname(out_gif), exist_ok=True)
    anim.save(out_gif, writer="pillow", fps=int(fps))
    plt.close(fig)

    print(f"   - wrote: {out_gif}")
    return out_gif


# -----------------------------------------------------------------------------
# Union-of-balls Folium time slider (interactive HTML, approximate)
# -----------------------------------------------------------------------------

def plot_union_balls_folium_slider(
    csv_path: str,
    out_html: str,
    n_steps: int = 40,
    eps_max_m: float = 6000.0,
    zoom_start: int = 11,
    circle_color: str = "red",
    circle_fill_opacity: float = 0.15,
    show_points: bool = True,
    point_radius_px: int = 3,
) -> Optional[str]:
    """Folium time slider approximation for growing circles."""
    print(f"\n[folium union-balls slider] {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path, quoting=csv.QUOTE_MINIMAL)

    lat_col, lon_col = _find_latlon_cols(df)
    df = _clean_points(df, lat_col, lon_col)
    if df.empty:
        print("   - no valid points; skipping.")
        return None

    center_lat = float(df[lat_col].mean())
    center_lon = float(df[lon_col].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=int(zoom_start),
        tiles="OpenStreetMap",
        control_scale=True,
    )

    title = f"{_stem_from_csv_path(csv_path)} — Union-of-balls slider (approx)"
    title_html = f"""
        <div style="
            position: fixed;
            top: 10px; left: 50px;
            z-index: 9999;
            background-color: white;
            padding: 6px 10px;
            border: 1px solid #999;
            border-radius: 4px;
            font-size: 14px;
        ">
            {title}
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    if show_points:
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[float(row[lat_col]), float(row[lon_col])],
                radius=int(point_radius_px),
                color=circle_color,
                fill=True,
                fill_color=circle_color,
                fill_opacity=0.85,
                opacity=0.85,
            ).add_to(m)

    features = []
    eps_values = np.linspace(0.0, float(eps_max_m), int(n_steps))

    for step_idx, eps in enumerate(eps_values):
        r = float(eps) / 2.0
        t = f"2026-01-01T00:{step_idx:02d}:00"

        for _, row in df.iterrows():
            features.append({
                "type": "Feature",
                "properties": {
                    "time": t,
                    "style": {
                        "color": circle_color,
                        "fillColor": circle_color,
                        "fillOpacity": float(circle_fill_opacity),
                        "opacity": 0.6,
                        "weight": 1,
                    },
                    "eps_m": float(eps),
                    "r_m": float(r),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row[lon_col]), float(row[lat_col])],
                }
            })

    geojson = {"type": "FeatureCollection", "features": features}

    TimestampedGeoJson(
        data=geojson,
        transition_time=200,
        period="PT1M",
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=5,
        loop_button=True,
        date_options="HH:mm",
        time_slider_drag_update=True,
    ).add_to(m)

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)
    print(f"   - wrote: {out_html}")
    return out_html


# -----------------------------------------------------------------------------
# Folder-level batch wrapper (includes union-of-balls)
# -----------------------------------------------------------------------------

def batch_map_folder(
    input_folder: str,
    output_folder: str,
    pattern: str = "*_tract_data.csv",

    # Existing outputs
    make_contextily: bool = True,
    make_folium: bool = True,

    # New outputs
    make_unionballs_contextily_gif: bool = True,
    make_unionballs_folium_slider: bool = True,

    # Styling / settings shared across outputs
    point_color: str = "red",
    point_size: float = 30.0,
    circle_radius: int = 5,
    basemap: object = None,

    # Union-of-balls controls: Contextily GIF
    ub_gif_frames: int = 140,
    ub_gif_fps: int = 12,
    ub_gif_eps_quantile: float = 0.70,
    ub_gif_eps_max_m: Optional[float] = None,
    ub_gif_show_edges: bool = False,

    # Union-of-balls controls: Folium slider
    ub_html_steps: int = 40,
    ub_html_eps_max_m: float = 6000.0,
    ub_html_zoom_start: int = 11,
) -> Dict:
    """Apply mapping to every CSV matching pattern in input_folder."""
    os.makedirs(output_folder, exist_ok=True)
    basemap = basemap or cx.providers.OpenStreetMap.Mapnik

    csv_paths = sorted(glob.glob(os.path.join(input_folder, pattern)))
    print(f"\n=== Mapping batch ===")
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Matched files: {len(csv_paths)}")

    written_png: List[str] = []
    written_html: List[str] = []
    written_union_gif: List[str] = []
    written_union_html: List[str] = []
    skipped: List[str] = []
    errors: List[str] = []

    for csv_path in csv_paths:
        stem = _stem_from_csv_path(csv_path)

        if make_contextily:
            out_png = os.path.join(output_folder, f"{stem}_contextily.png")
            try:
                r = plot_points_contextily(
                    csv_path=csv_path,
                    out_png=out_png,
                    point_color=point_color,
                    point_size=point_size,
                    basemap=basemap,
                )
                if r:
                    written_png.append(r)
            except Exception as e:
                msg = f"contextily failed for {stem}: {e}"
                print(f"   - [ERROR] {msg}")
                errors.append(msg)

        if make_folium:
            out_html = os.path.join(output_folder, f"{stem}_folium.html")
            try:
                r = plot_points_folium(
                    csv_path=csv_path,
                    out_html=out_html,
                    point_color=point_color,
                    circle_radius=circle_radius,
                )
                if r:
                    written_html.append(r)
            except Exception as e:
                msg = f"folium failed for {stem}: {e}"
                print(f"   - [ERROR] {msg}")
                errors.append(msg)

        if make_unionballs_contextily_gif:
            out_gif = os.path.join(output_folder, f"{stem}_union_balls_contextily.gif")
            try:
                r = plot_union_balls_contextily_gif(
                    csv_path=csv_path,
                    out_gif=out_gif,
                    basemap=basemap,
                    n_frames=int(ub_gif_frames),
                    fps=int(ub_gif_fps),
                    eps_max_m=ub_gif_eps_max_m,
                    eps_quantile=float(ub_gif_eps_quantile),
                    show_edges=bool(ub_gif_show_edges),
                    point_color=point_color,
                    point_size=float(point_size) * 0.6,
                )
                if r:
                    written_union_gif.append(r)
            except Exception as e:
                msg = f"union-balls contextily GIF failed for {stem}: {e}"
                print(f"   - [ERROR] {msg}")
                errors.append(msg)

        if make_unionballs_folium_slider:
            out_html2 = os.path.join(output_folder, f"{stem}_union_balls_folium.html")
            try:
                r = plot_union_balls_folium_slider(
                    csv_path=csv_path,
                    out_html=out_html2,
                    n_steps=int(ub_html_steps),
                    eps_max_m=float(ub_html_eps_max_m),
                    zoom_start=int(ub_html_zoom_start),
                    circle_color=point_color,
                )
                if r:
                    written_union_html.append(r)
            except Exception as e:
                msg = f"union-balls folium HTML failed for {stem}: {e}"
                print(f"   - [ERROR] {msg}")
                errors.append(msg)

        if (not make_contextily) and (not make_folium) and (not make_unionballs_contextily_gif) and (not make_unionballs_folium_slider):
            skipped.append(csv_path)

    summary = {
        "csv_files": csv_paths,
        "contextily_pngs": written_png,
        "folium_htmls": written_html,
        "union_balls_gifs": written_union_gif,
        "union_balls_htmls": written_union_html,
        "n_csv": len(csv_paths),
        "n_png": len(written_png),
        "n_html": len(written_html),
        "n_union_gif": len(written_union_gif),
        "n_union_html": len(written_union_html),
        "skipped": skipped,
        "errors": errors,
    }

    print("\n=== Mapping summary ===")
    print(f"   CSV files:        {summary['n_csv']}")
    print(f"   PNG maps:         {summary['n_png']}")
    print(f"   HTML maps:        {summary['n_html']}")
    print(f"   Union GIF maps:   {summary['n_union_gif']}")
    print(f"   Union HTML maps:  {summary['n_union_html']}")
    if errors:
        print(f"   Errors:           {len(errors)} (see summary['errors'])")

    return summary


if __name__ == "__main__":
    # Example usage (edit paths to your environment)
    INPUT = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_tract"
    OUTPUT = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_graphics"

    batch_map_folder(
        input_folder=INPUT,
        output_folder=OUTPUT,

        make_contextily=True,
        make_folium=True,

        make_unionballs_contextily_gif=True,
        make_unionballs_folium_slider=True,

        point_color="red",
        point_size=30,
        circle_radius=5,

        ub_gif_frames=140,
        ub_gif_fps=12,
        ub_gif_eps_quantile=0.70,
        ub_gif_eps_max_m=None,
        ub_gif_show_edges=False,

        ub_html_steps=40,
        ub_html_eps_max_m=6000.0,
        ub_html_zoom_start=11,
    )
