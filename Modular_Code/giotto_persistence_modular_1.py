# ph_computations_orig.py
# -*- coding: utf-8 -*-
"""
Originally from ph_computations_orig.ipynb, converted to script.

Parameters (edit here or override before import):
    matrix_folder     – folder containing *_d_*.npy and *_dtilde_*.npy files
    output_folder     – where PNGs and .npz files will be saved
    time_unit_divisor – e.g. 60 to convert seconds → minutes
    max_homology_dim  – maximum homology dimension to compute
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
# ──────────────────────────────────────────────────────────────────────────────

def extract_birth_death(diagram):
    """
    Returns a dict mapping:
      'dim_<d>_birth' → array of birth times,
      'dim_<d>_death' → array of death times
    """
    result = {}
    for d in np.unique(diagram[:, 0]):
        pts = diagram[diagram[:, 0] == d]
        result[f"dim_{int(d)}_birth"] = pts[:, 1]
        result[f"dim_{int(d)}_death"] = pts[:, 2]
    return result

def process_all_matrices():
    os.makedirs(output_folder, exist_ok=True)
    for fname in sorted(os.listdir(matrix_folder)):
        # only .npy files containing _d_ or _dtilde_
        if not fname.endswith(".npy") or not ("_d_" in fname or "_dtilde_" in fname):
            continue

        base   = os.path.splitext(fname)[0]      # e.g. 'AlbuquerqueNM_d_drive'
        parts  = base.split("_")
        metric = next((p for p in parts if p in ("d","dtilde")), "unknown")
        travel = next((p for p in parts if p in ("drive","walk")),  "unknown")
        city   = base.replace(f"_{metric}_{travel}", "")

        mat_fp = os.path.join(matrix_folder, fname)
        print(f"\n▶ Processing {city} | {travel} | {metric}")

        # 1) Compute persistence diagram
        D   = np.load(mat_fp)[np.newaxis, :, :]
        VR  = VietorisRipsPersistence(
                  homology_dimensions=list(range(max_homology_dim+1)),
                  metric="precomputed"
              )
        diag = VR.fit_transform(D)[0]
        #print("diag.shape =", diag.shape)
        #print("First 5 rows (dim, birth, death):")
        #print(diag[:5, :3])
        diag[:, :2] /= time_unit_divisor
        print(f"   • {len(diag)} points in diagram")

        # 2) Standard Giotto‐TDA diagram
        out1 = os.path.join(
            output_folder,
            f"{city}_{travel}_{metric}_diagram.png"
        )
        fig = plot_diagram(diag)
        # requires kaleido
        fig.write_image(out1)
        print("   • diagram →", out1)

        # 3) Custom minute‐scaled scatter
        out2 = os.path.join(
            output_folder,
            f"{city}_{travel}_{metric}_custom.png"
        )
        plt.figure(figsize=(5,5))
        for d in np.unique(diag[:,0]):
            pts   = diag[diag[:,0]==d]
            birth = pts[:,1]/time_unit_divisor
            death = pts[:,2]/time_unit_divisor
            mkr   = "." if d==0 else "X"
            clr   = "red" if d==0 else "tab:blue"
            sz    = 10 if d==0 else 5
            plt.scatter(birth, death, marker=mkr, color=clr, s=sz)
        red  = mlines.Line2D([], [], color="red", marker=".", linestyle="None",
                             markersize=10, label="0D")
        blu  = mlines.Line2D([], [], color="tab:blue", marker="X", linestyle="None",
                             markersize=5,  label="1D")
        plt.legend(handles=[red, blu])
        plt.xlabel("Birth (minutes)")
        plt.ylabel("Death (minutes)")
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
            f"{city}_{travel}_{metric}_landscape.png"
        )
        plt.savefig(out3, dpi=300)
        plt.close()
        print("   • landscape →", out3)

        # 5) Save birth/death arrays
        bd   = extract_birth_death(diag)
        out4 = os.path.join(
            output_folder,
            f"{city}_{travel}_{metric}_birth_death.npz"
        )
        np.savez(out4, **bd)
        print("   • birth/death →", out4)

        # 6) Print mean lifetimes
        print("   • mean lifetimes:")
        for k in sorted(bd):
            if k.endswith("_birth"):
                d    = k.split("_")[1]
                lm   = bd[f"dim_{d}_death"] - bd[k]
                print(f"      H{d}: {np.mean(lm):.2f}")

if __name__ == "__main__":
    process_all_matrices()
