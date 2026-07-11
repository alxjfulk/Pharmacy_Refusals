# ph_computations_orig.py
# -*- coding: utf-8 -*-
"""
Originally from ph_computations_orig.ipynb, converted to script.

Parameters (edit here or override before import):
    matrix_folder     – folder containing *_d_*.npy and *_dtilde_*.npy files
    output_folder     – where PNGs and .npz files will be saved
    time_unit_divisor – e.g. 60 to convert seconds → minutes
    max_homology_dim  – maximum homology dimension to compute
    matrix_metrics    – which matrix families to process: d, dtilde, or both
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from gtda.homology      import VietorisRipsPersistence
from gtda.diagrams      import PersistenceLandscape
from gtda.plotting      import plot_diagram
import kaleido

# ── Parameters ───────────────────────────────────────────────────────────────
matrix_folder     = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_WDM"
output_folder     = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_TDA"
time_unit_divisor = 60
max_homology_dim  = 1
matrix_metrics    = {"d", "dtilde"}
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_matrix_metrics(metrics):
    """
    Normalize the matrix-family selection for Step 5.

    Accepted values:
      - None or "both"       -> {"d", "dtilde"}
      - "d"                  -> {"d"}
      - "dtilde"             -> {"dtilde"}
      - list/tuple/set values -> any combination of "d" and "dtilde"

    Step 4 still computes both dtilde and d. This selector only controls which
    saved matrices are passed into the TDA / Vietoris-Rips step.
    """
    if metrics is None:
        return {"d", "dtilde"}

    if isinstance(metrics, str):
        m = metrics.strip().lower()
        if m in {"both", "all", "d_and_dtilde", "d+dtilde"}:
            return {"d", "dtilde"}
        metrics = [m]

    out = {str(m).strip().lower() for m in metrics}
    invalid = out - {"d", "dtilde"}
    if invalid:
        raise ValueError(f"Unsupported TDA matrix metric(s): {sorted(invalid)}. Use 'd', 'dtilde', or both.")
    if not out:
        raise ValueError("At least one TDA matrix metric must be selected: 'd', 'dtilde', or both.")
    return out


def extract_birth_death(diagram):
    """
    Return birth/death arrays by homology dimension.

    giotto-tda persistence diagrams are represented as columns:
        birth, death, homology_dimension

    The saved key names are kept as dim_<d>_birth / dim_<d>_death for backward
    compatibility with earlier output files.
    """
    result = {}
    if diagram.size == 0:
        return result

    dims = np.unique(diagram[:, 2]).astype(int)
    for d in dims:
        pts = diagram[diagram[:, 2].astype(int) == d]
        result[f"dim_{int(d)}_birth"] = pts[:, 0]
        result[f"dim_{int(d)}_death"] = pts[:, 1]
    return result

def process_all_matrices(matrix_folder_=None,
                        output_folder_=None,
                        input_time_unit_=None,
                        output_time_unit_=None,
                        max_homology_dim_=None,
                        matrix_metrics_=None):
    # Allow the master process flow to override module defaults via keyword args.
    # If no args are provided, the module-level defaults below are used.
    global matrix_folder, output_folder, time_unit_divisor, max_homology_dim, matrix_metrics

    if matrix_folder_ is not None:
        matrix_folder = matrix_folder_
    if output_folder_ is not None:
        output_folder = output_folder_
    if max_homology_dim_ is not None:
        max_homology_dim = max_homology_dim_
    matrix_metrics = _normalize_matrix_metrics(matrix_metrics_)
    print(f"TDA matrix metrics selected: {sorted(matrix_metrics)}")

    def _normalize_unit(u):
        if u is None:
            return None
        u = str(u).strip().lower()
        aliases = {
            "s": "seconds", "sec": "seconds", "secs": "seconds", "second": "seconds", "seconds": "seconds",
            "m": "minutes", "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
            "h": "hours", "hr": "hours", "hrs": "hours", "hour": "hours", "hours": "hours",
        }
        return aliases.get(u, u)

    in_unit = _normalize_unit(input_time_unit_)
    out_unit = _normalize_unit(output_time_unit_)

    # Convert diagram units: we scale birth/death by dividing by a divisor.
    # Example: seconds -> minutes => divisor=60 (divide by 60).
    if in_unit and out_unit:
        unit_to_seconds = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}
        if in_unit not in unit_to_seconds or out_unit not in unit_to_seconds:
            raise ValueError(
                f"Unsupported time unit(s): input={in_unit!r}, output={out_unit!r}. "
                "Use seconds/minutes/hours."
            )
    
        # if units match, don't scale
        if in_unit == out_unit:
            time_unit_divisor = 1.0
        else:
            time_unit_divisor = unit_to_seconds[out_unit] / unit_to_seconds[in_unit]
    else:
        # Fall back to legacy labeling if units are not provided.
        out_unit = "minutes" if time_unit_divisor == 60 else "time units"
    os.makedirs(output_folder, exist_ok=True)
    for fname in sorted(os.listdir(matrix_folder)):
        # only .npy files containing _d_ or _dtilde_
        if not fname.endswith(".npy") or not ("_d_" in fname or "_dtilde_" in fname):
            continue

        base = os.path.splitext(fname)[0]

        # Parse filenames robustly. Expected pattern (flexible):
        #   <CityState>_<metric>_<mode>_<tag...>.npy
        # Examples:
        #   AlbuquerqueNM_d_drive_2025-12-30.npy
        #   MilwaukeeWI_dtilde_walk_2025-12-30.npy
        #   LasVegasNV_dtilde_multimodal_2025-12-30.npy
        parts = base.split("_")

        # Identify metric position and split the rest from there.
        metric_idx = next((i for i, p in enumerate(parts) if p in ("d", "dtilde")), None)
        if metric_idx is None:
            metric = "unknown"
            city = parts[0] if parts else base
            travel = "unknown"
            tag = ""
        else:
            metric = parts[metric_idx]
            city = "_".join(parts[:metric_idx]) if metric_idx > 0 else "unknown"
            travel = parts[metric_idx + 1] if (metric_idx + 1) < len(parts) else "unknown"
            tag = "_".join(parts[metric_idx + 2:]) if (metric_idx + 2) < len(parts) else ""

        if metric not in matrix_metrics:
            continue

        # Build an output stem that preserves the travel tag to avoid overwrites.
        out_stem = f"{city}_{travel}_{metric}" + (f"_{tag}" if tag else "")

        mat_fp = os.path.join(matrix_folder, fname)
        print(f"\n▶ Processing {city} | {travel} | {metric}" + (f" | {tag}" if tag else ""))

        # 1) Compute persistence diagram

        # Load matrix
        M = np.load(mat_fp)
        
        # --- sanitize distance matrix for Giotto/Sklearn ---
        M = np.asarray(M, dtype=float)
        
        nan_ct = int(np.isnan(M).sum())
        if nan_ct > 0:
            print(f"   [WARN] {fname}: found {nan_ct:,} NaNs; converting NaN -> +inf")
            M[np.isnan(M)] = np.inf
        
        # Ensure zero diagonal (distance to self should be 0).
        # Giotto's precomputed VR calculation expects a symmetric distance matrix.
        # The final d matrix from Step 4 is designed to be symmetric. dtilde may be
        # asymmetric because it is origin-weighted; if dtilde is selected for TDA,
        # this code makes the required symmetric input explicit using the smaller
        # of the two directed values.
        if M.shape[0] == M.shape[1]:
            np.fill_diagonal(M, 0.0)

            finite_pair = np.isfinite(M) & np.isfinite(M.T)
            if np.any(finite_pair):
                asym = np.nanmax(np.abs(M[finite_pair] - M.T[finite_pair]))
                if asym > 1e-8:
                    print(f"   [WARN] {fname}: matrix is asymmetric; symmetrizing with min(M, M.T) for VR input")

            M = np.minimum(M, M.T)
        
        # Optional: clamp negative values if any (distances shouldn't be negative)
        if np.any(M < 0):
            neg_ct = int((M < 0).sum())
            print(f"   [WARN] {fname}: found {neg_ct:,} negative entries; clipping to 0")
            M[M < 0] = 0.0
        # --- end sanitize ---
        
        # Add batch dimension for Giotto
        D = M[np.newaxis, :, :]
        VR  = VietorisRipsPersistence(
                  homology_dimensions=list(range(max_homology_dim+1)),
                  metric="precomputed"
              )

        diag = VR.fit_transform(D)[0]
        # giotto-tda diagram columns are [birth, death, homology_dimension].
        # Convert only birth/death values to the requested output time unit.
        if diag.size > 0:
            diag[:, 0:2] /= time_unit_divisor
        print(f"   • {len(diag)} points in diagram")

        # 2) Standard Giotto‐TDA diagram
        out1 = os.path.join(
            output_folder,
            f"{out_stem}_diagram.png"
        )
        fig = plot_diagram(diag)
        # requires kaleido
        fig.write_image(out1)
        print("   • diagram →", out1)

        # 3) Custom minute‐scaled scatter
        out2 = os.path.join(
            output_folder,
            f"{out_stem}_custom.png"
        )
        plt.figure(figsize=(5,5))
        for d in np.unique(diag[:, 2]).astype(int):
            pts   = diag[diag[:, 2].astype(int) == d]
            birth = pts[:, 0]
            death = pts[:, 1]
            mkr   = "." if d == 0 else "X"
            clr   = "red" if d == 0 else "tab:blue"
            sz    = 10 if d == 0 else 5
            plt.scatter(birth, death, marker=mkr, color=clr, s=sz)
        red  = mlines.Line2D([], [], color="red", marker=".", linestyle="None",
                             markersize=10, label="0D")
        blu  = mlines.Line2D([], [], color="tab:blue", marker="X", linestyle="None",
                             markersize=5,  label="1D")
        plt.legend(handles=[red, blu])
        plt.xlabel(f"Birth ({out_unit})")
        plt.ylabel(f"Death ({out_unit})")
        plt.tight_layout()
        plt.savefig(out2, dpi=300)
        plt.close()
        print("   • custom →", out2)

        # 4) Persistence landscape
        landscape = PersistenceLandscape(n_layers=5)
        vals      = landscape.fit_transform(diag[np.newaxis, :, :])[0]
        plt.figure(figsize=(8,6))
        for i in range(vals.shape[0]):
            plt.plot(vals[i], label=f"Layer {i+1}")
        plt.xlabel("Time")
        plt.ylabel("Persistence")
        plt.legend()
        plt.tight_layout()
        out3 = os.path.join(
            output_folder,
            f"{out_stem}_landscape.png"
        )
        plt.savefig(out3, dpi=300)
        plt.close()
        print("   • landscape →", out3)

        # 5) Save birth/death arrays
        bd   = extract_birth_death(diag)
        out4 = os.path.join(
            output_folder,
            f"{out_stem}_birth_death.npz"
        )
        np.savez(out4, **bd)
        print("   • birth/death →", out4)

        # 6) Print mean lifetimes
        print("   • mean lifetimes:")
        for k in sorted(bd):
            if k.endswith("_birth"):
                d    = k.split("_")[1]
                lm = bd[f"dim_{d}_death"] - bd[k]
                lm = lm[np.isfinite(lm)]
                mean_lm = float(np.mean(lm)) if lm.size else float("nan")
                print(f"      H{d}: {mean_lm:.2f}")
                
def generate_pharmacy_birth_death_gif(
    matrix_fp,
    pharmacy_csv,
    streets_shp=None,
    tracts_shp=None,
    n_frames=100,
    dt=1.0,
    t_max=None,
    tol=None,
    out_gif=None
):
    """
    Parameters
    ----------
    matrix_fp : str
        Path to your travel-time .npy matrix (seconds).
    pharmacy_csv : str
        Path to your pharmacy CSV with 'latitude' and 'longitude' columns.
    streets_shp : str or None
        Optional: path to a street network shapefile for background.
    tracts_shp : str or None
        Optional: path to a census-tract shapefile for background.
    n_frames : int
        Number of frames in the GIF (default 100).
    dt : float
        Minutes between successive frames (default 1.0).
    t_max : float or None
        Max time (min) to animate; if None, uses the largest death time.
    tol : float or None
        Matching tolerance (min) for highlighting events; if None, set to dt/2.
    out_gif : str or None
        Output GIF path; if None, uses matrix_fp base + '_bd.gif'.
    """
    import os
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import imageio
    from shapely.geometry import LineString, Polygon
    from gtda.homology import VietorisRipsPersistence

    # 1) load pharmacy points
    df = pd.read_csv(pharmacy_csv)
    df = df.dropna(subset=['latitude','longitude'])
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # 2) load & scale matrix → minutes
    M = np.load(matrix_fp)            # shape (N,N)
    D = (M[np.newaxis] / 60.0)        # batch dim + minutes

    # 3) fit VR complex, get simplex tree
    VR = VietorisRipsPersistence(
        homology_dimensions=[0,1],
        metric="precomputed",
        return_simplex_tree=True
    )
    diagrams, sts = VR.fit_transform(D)  # returns (diagrams, simplex_trees) when return_simplex_tree=True
    st = sts[0]
    pairs = st.persistence_simplex_pairs()  # list of (birth_sx, death_sx)

    # 4) time grid
    death_times = [st.filtration(ds) for _, ds in pairs]
    if t_max is None:
        t_max = max(death_times) if death_times else 0
    times = np.linspace(0, t_max, n_frames)
    if tol is None:
        # If not provided, default to half of the frame spacing so events are not missed.
        tol = (times[1] - times[0]) / 2 if len(times) > 1 else 0

    # 5) build frames
    frames, tmp = [], []
    for i, t in enumerate(times):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_axis_off()

        # background (comment out if unwanted)
        if streets_shp:
            gpd.read_file(streets_shp).plot(ax=ax, edgecolor='lightgray', facecolor='none')
        if tracts_shp:
            gpd.read_file(tracts_shp).plot(ax=ax, edgecolor='gray', linewidth=0.5, facecolor='none')

        # pharmacies
        pts.plot(ax=ax, markersize=10, color='blue', alpha=0.6)

        # highlight events at this t
        for bx, ds in pairs:
            dtime = st.filtration(ds)
            if abs(dtime - t) <= tol:
                coords = [pts.geometry.iloc[j].coords[0] for j in ds]
                dim = len(ds) - 1
                if dim == 1:
                    line = LineString(coords)
                    ax.plot(*line.xy, color='red', linewidth=2)
                elif dim == 2:
                    poly = Polygon(coords)
                    gpd.GeoSeries([poly]).plot(ax=ax, edgecolor='green', facecolor='none', linewidth=2)

        ax.set_title(f"t = {t:.1f} min")
        plt.tight_layout()

        fn = f"__tmp_frame_{i:03d}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        tmp.append(fn)
        frames.append(imageio.imread(fn))

    # 6) assemble GIF
    if out_gif is None:
        base = os.path.splitext(matrix_fp)[0]
        out_gif = base + "_bd.gif"
    imageio.mimsave(out_gif, frames, fps=1)

    # cleanup
    for fn in tmp:
        try: os.remove(fn)
        except: pass

    print(f"Saved birth–death GIF to {out_gif}")
    
import os
import glob

def batch_generate_birth_death_gifs(
    matrix_folder,
    pharmacy_folder,
    streets_shp=None,
    tracts_shp=None,
    n_frames=100,
    dt=1.0,
    t_max=None,
    tol=None,
    output_gif_folder=None
):
    """
    For every Pharmacy_Data_<CityState>.csv in pharmacy_folder and
    every <CityState>_d_<travel>.npy in matrix_folder, build a
    birth–death GIF of H₀ merges vs. H₁ fillings.

    Parameters
    ----------
    matrix_folder : str
        Path containing *_d_*.npy files.
    pharmacy_folder : str
        Path containing Pharmacy_Data_*.csv files.
    streets_shp, tracts_shp : str or None
        Optional shapefile paths for background layers.
    n_frames, dt, t_max, tol : as in generate_pharmacy_birth_death_gif.
    output_gif_folder : str or None
        Where to save .gif outputs (defaults to matrix_folder).
    """
    # 0) prep output directory
    if output_gif_folder is None:
        output_gif_folder = matrix_folder
    os.makedirs(output_gif_folder, exist_ok=True)

    # 1) find all pharmacy CSVs
    pharm_paths = glob.glob(os.path.join(pharmacy_folder, "Pharmacy_Data_*.csv"))  # :contentReference[oaicite:2]{index=2}

    for pharm_csv in pharm_paths:
        city_state = os.path.splitext(os.path.basename(pharm_csv))[0].replace("Pharmacy_Data_", "")

        # 2) find matching persistence-matrix files
        #    we look for any <CityState>_*.npy then filter for "_d_"
        candidates = glob.glob(os.path.join(matrix_folder, f"{city_state}_*.npy"))
        matrix_files = [m for m in candidates if "_d_" in os.path.basename(m)]  # :contentReference[oaicite:3]{index=3}

        for matrix_fp in matrix_files:
            # derive a sensible GIF name
            base = os.path.splitext(os.path.basename(matrix_fp))[0]
            out_gif = os.path.join(output_gif_folder, f"{base}_birth_death.gif")

            # 3) call the existing GIF generator
            generate_pharmacy_birth_death_gif(
                matrix_fp=matrix_fp,
                pharmacy_csv=pharm_csv,
                streets_shp=streets_shp,
                tracts_shp=tracts_shp,
                n_frames=n_frames,
                dt=dt,
                t_max=t_max,
                tol=tol,
                out_gif=out_gif
            )
            print(f"→ Generated GIF for {city_state}: {os.path.basename(out_gif)}")

if __name__ == "__main__":
    process_all_matrices()