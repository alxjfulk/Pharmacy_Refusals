# -*- coding: utf-8 -*-
"""Reusable validation helpers for the revised Pharmacy Refusals pipeline.

The master uses these checks to decide whether a requested step should be:
- run because its complete output group is absent;
- skipped because the complete output group exists and validates; or
- stopped because outputs are partial, incompatible, or belong to another run.

These checks do not contact NPI, Google Places, Census, OSMnx, Giotto-TDA, or
GUDHI. They inspect only saved files and pure-Python metadata relationships.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pipeline_common_260801 import (
    ALL_DERIVED_MATRIX_TYPES,
    ANALYSIS_TAG_COLUMN,
    CityRunPaths,
    file_sha256,
    normalize_matrix_type,
    ordered_id_checksum,
    read_json,
    validate_analysis_tag,
    validate_pharmacy_ids,
)


class PipelinePrerequisiteError(RuntimeError):
    """Raised when a requested step cannot safely start."""


class PartialOutputError(RuntimeError):
    """Raised when only part of an expected output group exists."""


def _group_state(paths: Iterable[Path], context: str) -> str:
    paths = list(paths)
    present = [path for path in paths if path.exists()]
    absent = [path for path in paths if not path.exists()]
    if present and absent:
        raise PartialOutputError(
            f"{context} has partial outputs. Existing: {[str(p) for p in present]}; "
            f"missing: {[str(p) for p in absent]}. Use a new ANALYSIS_RUN_TAG or "
            "manually repair/remove the incomplete group."
        )
    return "complete" if present else "absent"


def _ensure_absent_directory_is_empty(directory: Path, state: str, context: str) -> None:
    """Treat untracked files in an otherwise absent output group as partial work."""
    if state == "absent" and directory.exists() and any(directory.iterdir()):
        raise PartialOutputError(
            f"{context} has files in {directory} but no complete recognized output group. "
            "Use a new ANALYSIS_RUN_TAG or manually inspect the interrupted output."
        )


def validate_step1(paths: CityRunPaths) -> str:
    """Validate the complete Step 1 output group, or report that it is absent/skipped."""
    csv_exists = paths.step01_pharmacy_csv.exists()
    metadata_exists = paths.step01_metadata.exists()
    if not csv_exists and not metadata_exists:
        _ensure_absent_directory_is_empty(
            paths.step01_dir, "absent", f"Step 1 for {paths.city_state}"
        )
        return "absent"
    if metadata_exists and not csv_exists:
        metadata = read_json(paths.step01_metadata)
        if (
            metadata.get("processing_status") == "skipped"
            and metadata.get(ANALYSIS_TAG_COLUMN) == paths.analysis_run_tag
        ):
            marker = Path(str(metadata.get("skip_marker", "")))
            if not marker.exists():
                raise PartialOutputError("Step 1 skip metadata exists but its marker file is missing.")
            return "skipped"
        raise PartialOutputError(
            f"Step 1 for {paths.city_state} has metadata without pharmacy_data.csv."
        )
    if csv_exists and not metadata_exists:
        raise PartialOutputError(
            f"Step 1 for {paths.city_state} has pharmacy_data.csv without metadata."
        )
    frame = pd.read_csv(
        paths.step01_pharmacy_csv,
        dtype={"pharmacy_id": "string", "NPI": "string", ANALYSIS_TAG_COLUMN: "string"},
    )
    validate_analysis_tag(frame, paths.analysis_run_tag, str(paths.step01_pharmacy_csv))
    ids = validate_pharmacy_ids(frame, str(paths.step01_pharmacy_csv))
    metadata = read_json(paths.step01_metadata)
    if metadata.get("processing_status", "complete") != "complete":
        raise ValueError("Step 1 pharmacy CSV exists but metadata is not marked complete.")
    if metadata.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
        raise ValueError("Step 1 metadata analysis_run_tag mismatch.")
    if int(metadata.get("final_pharmacy_count", -1)) != len(frame):
        raise ValueError("Step 1 metadata count does not match pharmacy_data.csv.")
    if ids and not ids[0].startswith(f"{paths.city_state}_{paths.analysis_run_tag}_P"):
        raise ValueError("Step 1 pharmacy IDs do not use the selected analysis_run_tag.")
    return "complete"


def validate_step2(paths: CityRunPaths) -> str:
    """Validate the complete Step 2 tract, far-pharmacy, and metadata output group."""
    state = _group_state(
        [paths.step02_tract_csv, paths.step02_far_csv, paths.step02_metadata],
        f"Step 2 for {paths.city_state}",
    )
    if state == "absent":
        _ensure_absent_directory_is_empty(paths.step02_dir, state, f"Step 2 for {paths.city_state}")
        return state
    tract = pd.read_csv(
        paths.step02_tract_csv,
        dtype={"pharmacy_id": "string", "NPI": "string", ANALYSIS_TAG_COLUMN: "string"},
    )
    if tract.empty:
        raise ValueError("Step 2 tract_data.csv is empty.")
    validate_analysis_tag(tract, paths.analysis_run_tag, str(paths.step02_tract_csv))
    validate_pharmacy_ids(tract, str(paths.step02_tract_csv))
    required = {"GEOID", "TRACT", "B01003_001E", "latitude", "longitude"}
    missing = sorted(required - set(tract.columns))
    if missing:
        raise ValueError(f"Step 2 tract data is missing columns: {missing}.")
    far = pd.read_csv(
        paths.step02_far_csv,
        dtype={"pharmacy_id": "string", "NPI": "string", ANALYSIS_TAG_COLUMN: "string"},
    )
    if not far.empty:
        validate_analysis_tag(far, paths.analysis_run_tag, str(paths.step02_far_csv))
        validate_pharmacy_ids(far, str(paths.step02_far_csv))
    metadata = read_json(paths.step02_metadata)
    if metadata.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
        raise ValueError("Step 2 metadata analysis_run_tag mismatch.")
    if int(metadata.get("retained_pharmacy_count", -1)) != len(tract):
        raise ValueError("Step 2 metadata count does not match tract_data.csv.")
    return state


def _load_manifest(paths: CityRunPaths) -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(
        paths.step03_manifest,
        dtype={"pharmacy_id": "string", "NPI": "string", ANALYSIS_TAG_COLUMN: "string"},
    )
    validate_analysis_tag(manifest, paths.analysis_run_tag, str(paths.step03_manifest))
    ids = validate_pharmacy_ids(manifest, str(paths.step03_manifest))
    if "matrix_index" not in manifest.columns:
        raise ValueError("Step 3 manifest lacks matrix_index.")
    indices = pd.to_numeric(manifest["matrix_index"], errors="coerce")
    if indices.isna().any() or not np.array_equal(indices.to_numpy(), np.arange(len(manifest))):
        raise ValueError("Step 3 manifest matrix_index is not sequential from zero.")
    return manifest, ids


def validate_step3(paths: CityRunPaths) -> str:
    """Validate raw matrices, the authoritative manifest, dimensions, and provenance."""
    outputs = [
        paths.step03_manifest,
        paths.step03_metadata,
        paths.raw_matrix_path("walk"),
        paths.raw_matrix_path("drive"),
    ]
    state = _group_state(outputs, f"Step 3 for {paths.city_state}")
    if state == "absent":
        _ensure_absent_directory_is_empty(paths.step03_dir, state, f"Step 3 for {paths.city_state}")
        return state
    manifest, ids = _load_manifest(paths)
    expected_shape = (len(ids), len(ids))
    for mode in ("walk", "drive"):
        matrix = np.load(paths.raw_matrix_path(mode), allow_pickle=False)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Step 3 {mode} matrix shape {matrix.shape} does not match manifest {expected_shape}."
            )
    metadata = read_json(paths.step03_metadata)
    if metadata.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
        raise ValueError("Step 3 metadata analysis_run_tag mismatch.")
    if metadata.get("ordered_pharmacy_id_checksum") != ordered_id_checksum(ids):
        raise ValueError("Step 3 metadata ordered pharmacy-ID checksum mismatch.")
    if int(metadata.get("matrix_dimension", -1)) != len(ids):
        raise ValueError("Step 3 metadata matrix dimension mismatch.")
    expected_hashes = {
        "manifest_sha256": file_sha256(paths.step03_manifest),
        "walk_matrix_sha256": file_sha256(paths.raw_matrix_path("walk")),
        "drive_matrix_sha256": file_sha256(paths.raw_matrix_path("drive")),
    }
    for key, expected_hash in expected_hashes.items():
        # Older synthetic fixtures may omit these optional provenance values; live
        # revised outputs always write them. A present value must match exactly.
        if metadata.get(key) is not None and metadata.get(key) != expected_hash:
            raise ValueError(f"Step 3 metadata checksum mismatch for {key}.")
    return state


def validate_step4(paths: CityRunPaths) -> str:
    """Validate all six Step 4 matrices and their metadata against the manifest."""
    outputs = [paths.step04_metadata]
    for matrix_type in ALL_DERIVED_MATRIX_TYPES:
        outputs.extend(
            [
                paths.derived_matrix_path(matrix_type),
                paths.derived_matrix_csv(matrix_type),
                paths.derived_matrix_metadata(matrix_type),
            ]
        )
    state = _group_state(outputs, f"Step 4 for {paths.city_state}")
    if state == "absent":
        _ensure_absent_directory_is_empty(paths.step04_dir, state, f"Step 4 for {paths.city_state}")
        return state
    _, ids = _load_manifest(paths)
    expected = (len(ids), len(ids))
    for matrix_type in ALL_DERIVED_MATRIX_TYPES:
        matrix_type = normalize_matrix_type(matrix_type)
        matrix = np.load(paths.derived_matrix_path(matrix_type), allow_pickle=False)
        if matrix.shape != expected:
            raise ValueError(
                f"Step 4 {matrix_type} shape {matrix.shape} does not match manifest {expected}."
            )
        metadata = read_json(paths.derived_matrix_metadata(matrix_type))
        if metadata.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
            raise ValueError(f"Step 4 metadata tag mismatch for {matrix_type}.")
        if metadata.get("matrix_sha256") != file_sha256(paths.derived_matrix_path(matrix_type)):
            raise ValueError(f"Step 4 matrix checksum mismatch for {matrix_type}.")
    overall = read_json(paths.step04_metadata)
    if overall.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
        raise ValueError("Step 4 overall metadata analysis_run_tag mismatch.")
    return state


def validate_step5_plan_exists(paths: CityRunPaths) -> str:
    """Validate the saved Step 5 scenario plan, index, and membership files."""
    state = _group_state(
        [
            paths.scenario_plan_json,
            paths.scenario_index_csv,
        ],
        f"Step 5 scenario plan for {paths.city_state}",
    )
    if state == "absent":
        _ensure_absent_directory_is_empty(paths.step05_dir, state, f"Step 5 for {paths.city_state}")
        return state
    if not paths.scenario_membership_dir.exists():
        raise PartialOutputError("Step 5 plan exists but memberships directory is missing.")
    plan = read_json(paths.scenario_plan_json)
    if plan.get(ANALYSIS_TAG_COLUMN) != paths.analysis_run_tag:
        raise ValueError("Step 5 plan analysis_run_tag mismatch.")
    index = pd.read_csv(paths.scenario_index_csv)
    required = {"scenario_key", "scenario_type", "membership_hash"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"Step 5 scenario index is missing columns: {missing}.")
    for key in index["scenario_key"].astype(str):
        if not (paths.scenario_membership_dir / f"{key}.csv").exists():
            raise PartialOutputError(f"Step 5 membership is missing for scenario {key!r}.")
    return state


def require_prerequisite(condition: bool, message: str) -> None:
    """Raise a pipeline prerequisite error when a required condition is false."""
    if not condition:
        raise PipelinePrerequisiteError(message)


def validate_requested_step_prerequisites(
    paths: CityRunPaths,
    step: int,
    selected_steps: set[int],
) -> None:
    """Check whether a requested step has existing or planned direct prerequisites."""
    if step == 1:
        return
    dependency = {2: 1, 3: 2, 4: 3, 5: 4, 6: 2, 7: 5}[step]
    if dependency in selected_steps:
        return
    validators = {
        1: validate_step1,
        2: validate_step2,
        3: validate_step3,
        4: validate_step4,
        5: validate_step5_plan_exists,
    }
    state = validators[dependency](paths)
    require_prerequisite(
        state == "complete",
        f"Cannot run Step {step} for {paths.city_state}: required Step {dependency} "
        f"outputs do not exist. Add Step {dependency} to RUN_STEPS or complete it first.",
    )
