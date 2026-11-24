# -*- coding: utf-8 -*-
"""Shared path, metadata, hashing, and identity helpers for the Pharmacy Refusals pipeline.

The pipeline stores every workspace under::

    <ANALYSIS_OUTPUT_ROOT>/analysis_runs/<ANALYSIS_RUN_TAG>/<CityState>/

"analysis_run_tag" is the only run identifier used by the revised pipeline.
The user-facing setting, row-level provenance, matrix manifest, scenario plan,
and output paths all use that same concept and field name.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

ANALYSIS_TAG_COLUMN = "analysis_run_tag"
PIPELINE_SCHEMA_VERSION = "2026-08-01"


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def normalize_analysis_run_tag(value: Any) -> str:
    """Return a filesystem- and identifier-safe analysis-run tag.

    The main file supplies a date-based default, but users may replace it with a
    descriptive tag such as "brooklyn_baseline_v1". A previously used tag may
    be selected again to continue the same analysis. To start a new run, use a 
    new tag.
    """
    raw = str(value).strip()
    if not raw:
        raise ValueError("ANALYSIS_RUN_TAG cannot be blank.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    if not safe:
        raise ValueError("ANALYSIS_RUN_TAG did not contain usable characters.")
    return safe


def city_state_key(city: str, state: str) -> str:
    """Return the compact "<City><State>" identifier used in folder names and IDs."""
    city_part = re.sub(r"[^A-Za-z0-9]+", "", str(city).strip())
    state_part = re.sub(r"[^A-Za-z0-9]+", "", str(state).strip().upper())
    if not city_part or len(state_part) != 2:
        raise ValueError(f"Invalid city/state values: city={city!r}, state={state!r}.")
    return f"{city_part}{state_part}"


def safe_component(value: Any, default: str = "item") -> str:
    """Return a safe directory/file component without changing it too much."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return safe or default


@dataclass(frozen=True)
class CityRunPaths:
    """Centralized paths for one city within one analysis run.

    Individual steps receive explicit paths rather than a previous iteration
    that searched a folder for relevant files.
    """

    analysis_output_root: Path
    analysis_run_tag: str
    city: str
    state: str

    @property
    def analysis_runs_root(self) -> Path:
        return self.analysis_output_root / "analysis_runs"

    @property
    def analysis_run_dir(self) -> Path:
        return self.analysis_runs_root / normalize_analysis_run_tag(self.analysis_run_tag)

    @property
    def city_state(self) -> str:
        return city_state_key(self.city, self.state)

    @property
    def city_dir(self) -> Path:
        return self.analysis_run_dir / self.city_state

    @property
    def city_metadata(self) -> Path:
        return self.city_dir / "city_metadata.json"

    @property
    def step01_dir(self) -> Path:
        return self.city_dir / "step01_pharmacy"

    @property
    def step01_pharmacy_csv(self) -> Path:
        return self.step01_dir / "pharmacy_data.csv"

    @property
    def step01_metadata(self) -> Path:
        return self.step01_dir / "extraction_metadata.json"

    @property
    def step02_dir(self) -> Path:
        return self.city_dir / "step02_census"

    @property
    def step02_tract_csv(self) -> Path:
        return self.step02_dir / "tract_data.csv"

    @property
    def step02_far_csv(self) -> Path:
        return self.step02_dir / "far_pharmacies.csv"

    @property
    def step02_metadata(self) -> Path:
        return self.step02_dir / "step02_metadata.json"

    @property
    def step03_dir(self) -> Path:
        return self.city_dir / "step03_raw_travel"

    @property
    def step03_manifest(self) -> Path:
        return self.step03_dir / "manifest.csv"

    @property
    def step03_metadata(self) -> Path:
        return self.step03_dir / "step03_metadata.json"

    def raw_matrix_dir(self, travel_mode: str) -> Path:
        mode = normalize_travel_mode(travel_mode, allow_multimodal=False)
        return self.step03_dir / mode

    def raw_matrix_path(self, travel_mode: str) -> Path:
        return self.raw_matrix_dir(travel_mode) / "matrix.npy"

    @property
    def step04_dir(self) -> Path:
        return self.city_dir / "step04_derived_matrices"

    @property
    def step04_metadata(self) -> Path:
        return self.step04_dir / "step04_metadata.json"

    def derived_matrix_dir(self, matrix_type: str) -> Path:
        return self.step04_dir / normalize_matrix_type(matrix_type)

    def derived_matrix_path(self, matrix_type: str) -> Path:
        return self.derived_matrix_dir(matrix_type) / "matrix.npy"

    def derived_matrix_csv(self, matrix_type: str) -> Path:
        return self.derived_matrix_dir(matrix_type) / "matrix.csv"

    def derived_matrix_metadata(self, matrix_type: str) -> Path:
        return self.derived_matrix_dir(matrix_type) / "metadata.json"

    @property
    def step05_dir(self) -> Path:
        return self.city_dir / "step05_tda"

    @property
    def scenario_plan_dir(self) -> Path:
        return self.step05_dir / "scenario_plan"

    @property
    def scenario_plan_json(self) -> Path:
        return self.scenario_plan_dir / "plan.json"

    @property
    def scenario_index_csv(self) -> Path:
        return self.scenario_plan_dir / "scenario_index.csv"

    @property
    def scenario_membership_dir(self) -> Path:
        return self.scenario_plan_dir / "memberships"

    def tda_matrix_dir(self, matrix_type: str) -> Path:
        return self.step05_dir / normalize_matrix_type(matrix_type)

    @property
    def step06_dir(self) -> Path:
        return self.city_dir / "step06_mapping"

    @property
    def step06_metadata(self) -> Path:
        return self.step06_dir / "mapping_metadata.json"

    @property
    def step07_dir(self) -> Path:
        return self.city_dir / "step07_comparisons"

    def comparison_matrix_dir(self, matrix_type: str) -> Path:
        return self.step07_dir / normalize_matrix_type(matrix_type)

    def ensure_base_dirs(self) -> None:
        self.analysis_runs_root.mkdir(parents=True, exist_ok=True)
        self.analysis_run_dir.mkdir(parents=True, exist_ok=True)
        self.city_dir.mkdir(parents=True, exist_ok=True)


VALID_TRAVEL_MODES = {"walk", "drive", "multimodal"}
VALID_MATRIX_FORMS = {"d", "dtilde"}
ALL_DERIVED_MATRIX_TYPES = tuple(
    f"{mode}_{form}"
    for mode in ("walk", "drive", "multimodal")
    for form in ("d", "dtilde")
)


def normalize_travel_mode(value: Any, *, allow_multimodal: bool = True) -> str:
    """Normalize a travel-mode setting and validate it against supported values."""
    mode = str(value).strip().lower()
    aliases = {"multi": "multimodal"}
    mode = aliases.get(mode, mode)
    allowed = VALID_TRAVEL_MODES if allow_multimodal else {"walk", "drive"}
    if mode not in allowed:
        raise ValueError(f"Unsupported travel mode {value!r}; use {sorted(allowed)}.")
    return mode


def normalize_matrix_form(value: Any) -> str:
    """Normalize a Step 4 matrix-form setting to "d" or "dtilde"."""
    form = str(value).strip().lower()
    if form not in VALID_MATRIX_FORMS:
        raise ValueError(f"Unsupported matrix form {value!r}; use {sorted(VALID_MATRIX_FORMS)}.")
    return form


def normalize_matrix_type(value: Any) -> str:
    """Normalize and validate a canonical "<travel_mode>_<matrix_form>" name."""
    raw = str(value).strip().lower().replace("multi_", "multimodal_")
    match = re.fullmatch(r"(walk|drive|multimodal)_(d|dtilde)", raw)
    if not match:
        raise ValueError(
            f"Unsupported matrix type {value!r}; use one of {list(ALL_DERIVED_MATRIX_TYPES)}."
        )
    return raw


def resolve_matrix_types(travel_modes: Iterable[Any], matrix_forms: Iterable[Any]) -> list[str]:
    """Return the ordered Cartesian product used by Steps 5-7."""
    modes = [normalize_travel_mode(value) for value in travel_modes]
    forms = [normalize_matrix_form(value) for value in matrix_forms]
    if not modes or not forms:
        raise ValueError("At least one travel mode and one matrix form must be selected.")
    # Remove duplicates while preserving the order supplied in the main file.
    modes = list(dict.fromkeys(modes))
    forms = list(dict.fromkeys(forms))
    return [f"{mode}_{form}" for mode in modes for form in forms]


def validate_analysis_tag(
    df: pd.DataFrame,
    expected_tag: str,
    context: str,
    *,
    require_nonempty: bool = True,
) -> str:
    """Require exactly one nonmissing analysis tag and verify its expected value."""
    if ANALYSIS_TAG_COLUMN not in df.columns:
        raise ValueError(f"{context} is missing required column {ANALYSIS_TAG_COLUMN!r}.")
    tags = df[ANALYSIS_TAG_COLUMN].astype("string")
    if tags.isna().any() or tags.str.strip().eq("").any():
        raise ValueError(f"{context} contains missing {ANALYSIS_TAG_COLUMN} values.")
    unique = tags.astype(str).unique().tolist()
    if require_nonempty and not unique:
        raise ValueError(f"{context} contains no analysis-run tags.")
    if len(unique) != 1:
        raise ValueError(f"{context} contains mixed analysis-run tags: {unique}.")
    actual = normalize_analysis_run_tag(unique[0])
    expected = normalize_analysis_run_tag(expected_tag)
    if actual != expected:
        raise ValueError(
            f"{context} belongs to ANALYSIS_RUN_TAG={actual!r}, not the selected {expected!r}. "
            "Select the matching tag or create a new analysis run."
        )
    return actual


def validate_pharmacy_ids(df: pd.DataFrame, context: str) -> list[str]:
    """Validate unique, nonmissing run-specific pharmacy IDs and return them in row order."""
    if "pharmacy_id" not in df.columns:
        raise ValueError(f"{context} is missing required column 'pharmacy_id'.")
    ids = df["pharmacy_id"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError(f"{context} contains missing pharmacy_id values.")
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated(False)].astype(str).tolist()
        raise ValueError(f"{context} contains duplicate pharmacy_id values: {duplicates}.")
    return ids.astype(str).tolist()


def ordered_id_checksum(ids: Iterable[Any]) -> str:
    """Return a SHA-256 checksum for an ordered pharmacy-ID sequence."""
    payload = "\n".join(str(value) for value in ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: os.PathLike[str] | str) -> str:
    """Return the SHA-256 digest of a saved artifact without loading it all at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert NumPy, pandas, Path, and nonfinite values into JSON-safe objects."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def canonical_json(value: Any) -> str:
    """Serialize an object deterministically for hashing and plan comparison."""
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def object_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 hash for a JSON-compatible object."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    """Read a JSON object from disk and reject non-object top-level values."""
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def write_json(path: os.PathLike[str] | str, value: Mapping[str, Any]) -> Path:
    """Write JSON atomically so an interrupted write does not look complete."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(dict(value)), handle, indent=2, ensure_ascii=False)
    temporary.replace(target)
    return target


def update_city_metadata(paths: CityRunPaths, updates: Mapping[str, Any]) -> Path:
    """Merge high-level provenance into the city's metadata index.

    Step completeness is always revalidated from the actual files. This metadata
    file is just an index and history summary.
    """
    paths.ensure_base_dirs()
    if paths.city_metadata.exists():
        current = read_json(paths.city_metadata)
        existing_tag = current.get(ANALYSIS_TAG_COLUMN)
        if existing_tag != normalize_analysis_run_tag(paths.analysis_run_tag):
            raise ValueError(
                f"City metadata at {paths.city_metadata} belongs to tag {existing_tag!r}."
            )
    else:
        current = {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "city": paths.city,
            "state": paths.state.upper(),
            "city_state": paths.city_state,
            ANALYSIS_TAG_COLUMN: normalize_analysis_run_tag(paths.analysis_run_tag),
            "created_utc": utc_now_iso(),
            "artifacts": {},
            "validation": {},
        }
    for key, value in updates.items():
        if key in {"artifacts", "validation", "settings"} and isinstance(value, Mapping):
            existing = current.setdefault(key, {})
            existing.update(json_safe(dict(value)))
        else:
            current[key] = json_safe(value)
    current["last_updated_utc"] = utc_now_iso()
    return write_json(paths.city_metadata, current)


def required_files_state(paths: Iterable[os.PathLike[str] | str]) -> tuple[list[Path], list[Path]]:
    """Return "(existing, missing)" for a required output group."""
    normalized = [Path(path) for path in paths]
    existing = [path for path in normalized if path.exists()]
    missing = [path for path in normalized if not path.exists()]
    return existing, missing


def require_no_partial_outputs(required_paths: Iterable[os.PathLike[str] | str], context: str) -> bool:
    """Return True when complete, False when absent, and error when partially present."""
    existing, missing = required_files_state(required_paths)
    if existing and missing:
        raise RuntimeError(
            f"{context} is incomplete. Existing files: {[str(p) for p in existing]}; "
            f"missing files: {[str(p) for p in missing]}. Complete or remove the partial "
            "analysis manually, or use a new ANALYSIS_RUN_TAG."
        )
    return bool(existing) and not missing
