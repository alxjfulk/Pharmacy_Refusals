# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 11:14:37 2025

@author: bigfo
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram, plot_barcodes, plot_landscape


def load_labeled_matrix(npy_path, tract_csv_path):
    """
    Load a .npy matrix and label it with tract IDs from a matching CSV file.
    """
    matrix = np.load(npy_path)
    df = pd.read_csv(tract_csv_path)
    ids = df['ID'].reset_index(drop=True)
    if matrix.shape[0] != len(ids):
        raise ValueError("Matrix size and number of IDs do not match.")
    return pd.DataFrame(matrix, index=ids, columns=ids), df


def compute_accessibility_metrics(matrix_folder, tract_folder, output_folder=None, matrix_type='d', threshold=10):
    """
    Compute summary accessibility metrics from distance matrices and generate persistence diagrams.

    Parameters:
        matrix_folder (str): Folder containing distance matrices (.npy)
        tract_folder (str): Folder containing corresponding *_tract_data.csv files
        output_folder (str): Where to save computed metrics (optional)
        matrix_type (str): 'd' or 'dtilde'
        threshold (float): Distance threshold (e.g., minutes or km) for coverage calculation
    """
    os.makedirs(output_folder, exist_ok=True)
    matrix_files = [f for f in os.listdir(matrix_folder) if f.endswith(f"_{matrix_type}_drive.npy")]

    for file in matrix_files:
        matrix_path = os.path.join(matrix_folder, file)
        base_name = file.replace(f"_{matrix_type}_drive.npy", "")
        tract_csv_path = os.path.join(tract_folder, f"{base_name}_tract_data.csv")

        try:
            matrix_df, tract_df = load_labeled_matrix(matrix_path, tract_csv_path)
            d = matrix_df.values
            ids = matrix_df.index

            # Metrics
            min_distances = np.min(d + np.eye(d.shape[0]) * 1e6, axis=1)
            avg_distances = np.mean(d, axis=1)
            covered = (min_distances <= threshold).astype(int)

            # Merge with tract data
            tract_df = tract_df.set_index('ID').loc[ids]
            pop = pd.to_numeric(tract_df['B01003_001E'], errors='coerce').fillna(0)
            pov = pd.to_numeric(tract_df['Poverty_Rate'], errors='coerce').fillna(0)
            county = tract_df['county'] if 'county' in tract_df.columns else pd.Series(index=tract_df.index, dtype='str')

            total_pop = pop.sum()
            covered_pop = pop[covered == 1].sum()
            coverage_pct = 100 * covered_pop / total_pop if total_pop > 0 else 0

            result_df = pd.DataFrame({
                'ID': ids,
                'min_distance': min_distances,
                'avg_distance': avg_distances,
                'covered_within_threshold': covered,
                'population': pop,
                'poverty_rate': pov,
                'county': county
            })

            summary_stats = {
                'Total_Population': total_pop,
                'Covered_Population': covered_pop,
                'Coverage_Percent': coverage_pct,
                'Average_Min_Distance': np.average(min_distances, weights=pop),
                'Average_Avg_Distance': np.average(avg_distances, weights=pop)
            }

            # Disaggregate by poverty rate brackets
            poverty_bins = [0, 10, 20, 30, 40, 100]
            labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40%+']
            result_df['poverty_bin'] = pd.cut(result_df['poverty_rate'], bins=poverty_bins, labels=labels, include_lowest=True)

            disagg_summary = []
            for label in labels:
                group = result_df[result_df['poverty_bin'] == label]
                pop_sum = group['population'].sum()
                covered_sum = group.loc[group['covered_within_threshold'] == 1, 'population'].sum()
                disagg_summary.append({
                    'Poverty_Bracket': label,
                    'Group_Population': pop_sum,
                    'Group_Covered_Population': covered_sum,
                    'Group_Coverage_Percent': 100 * covered_sum / pop_sum if pop_sum > 0 else 0
                })

            # Disaggregate by county
            county_summary = []
            for county_name, group in result_df.groupby('county'):
                pop_sum = group['population'].sum()
                covered_sum = group.loc[group['covered_within_threshold'] == 1, 'population'].sum()
                county_summary.append({
                    'County': county_name,
                    'County_Population': pop_sum,
                    'County_Covered_Population': covered_sum,
                    'County_Coverage_Percent': 100 * covered_sum / pop_sum if pop_sum > 0 else 0
                })

            # Persistence diagram, barcode, and landscape
            persistence = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1])
            d_reshaped = d[np.newaxis, :, :]
            diagrams = persistence.fit_transform(d_reshaped)

            diag_fig = plot_diagram(diagrams[0])
            diag_fig.write_image(os.path.join(output_folder, f"{base_name}_{matrix_type}_drive_persistence.png"))

            barcode_fig = plot_barcodes(diagrams[0])
            barcode_fig.write_image(os.path.join(output_folder, f"{base_name}_{matrix_type}_drive_barcodes.png"))

            landscape_fig = plot_landscape(diagrams[0])
            landscape_fig.write_image(os.path.join(output_folder, f"{base_name}_{matrix_type}_drive_landscape.png"))

            if output_folder:
                result_path = os.path.join(output_folder, f"{base_name}_{matrix_type}_drive_metrics.csv")
                summary_path = os.path.join(output_folder, f"{base_name}_{matrix_type}_drive_summary.csv")
                disagg_path = os.path.join(output_folder, f"{base_name}_{matrix_type}_drive_poverty_disagg.csv")
                county_path = os.path.join(output_folder, f"{base_name}_{matrix_type}_drive_county_disagg.csv")
                result_df.to_csv(result_path, index=False)
                pd.DataFrame([summary_stats]).to_csv(summary_path, index=False)
                pd.DataFrame(disagg_summary).to_csv(disagg_path, index=False)
                pd.DataFrame(county_summary).to_csv(county_path, index=False)
                print(f"Saved metrics to {result_path}, summary to {summary_path}, and disaggregations to {disagg_path}, {county_path}")

        except Exception as e:
            print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    matrix_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\weight_travel_time_Results"
    tract_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\test_tract"
    output_folder = r"C:\Users\bigfo\OneDrive\Desktop\Pharmacy Refusals Stuff\test_modular\Accessibility_Results"

    compute_accessibility_metrics(matrix_folder, tract_folder, output_folder, matrix_type='d', threshold=10)
