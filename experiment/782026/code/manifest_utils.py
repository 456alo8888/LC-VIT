from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml


SPLIT_NAMES = ("train", "valid", "test")
VIEW_PATH_COLUMNS = ("axial_path", "coronal_path", "sagittal_path")
TARGET_COLUMNS = ("gs_rankin_6isdeath", "nihss")
TABULAR_FEATURE_COLUMNS = (
    "sex",
    "age",
    "race",
    "acuteischaemicstroke",
    "priorstroke",
    "bmi",
    "etiology",
)
REQUIRED_SOURCE_COLUMNS = (
    "participant_id",
    "source_split",
    *VIEW_PATH_COLUMNS,
    "all_views_present",
    *TABULAR_FEATURE_COLUMNS,
    *TARGET_COLUMNS,
)
EXPECTED_EXCLUSIONS = ("sub-235", "sub-335")
EXPECTED_FOLD_COUNTS = {
    0: {"train": 396, "valid": 99, "test": 125},
    1: {"train": 396, "valid": 99, "test": 125},
    2: {"train": 399, "valid": 98, "test": 123},
    3: {"train": 398, "valid": 99, "test": 123},
    4: {"train": 398, "valid": 98, "test": 124},
}


class ManifestError(ValueError):
    """Raised when a manifest violates the experiment contract."""


def find_repository_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    matches = [candidate for candidate in (current, *current.parents) if (candidate / ".git").exists()]
    if matches:
        # LC-VIT and baseline_encoder are nested repositories. Experiment paths
        # are intentionally relative to the outer stroke-prediction repository.
        return matches[-1]
    raise ManifestError(f"Could not locate repository root from {current}")


def resolve_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ManifestError(f"Configuration must be a YAML mapping: {config_path}")
    if not isinstance(config.get("paths"), dict):
        raise ManifestError("Configuration must define a 'paths' mapping")

    # Paths in experiment configs are repository-relative.  Resolve from this
    # module rather than the config location so temporary/external configs with
    # absolute fixture paths remain supported as well.
    repository_root = find_repository_root(Path(__file__))
    config = dict(config)
    config["paths"] = {
        key: str(resolve_path(value, repository_root))
        for key, value in config["paths"].items()
    }
    config["_config_path"] = str(config_path)
    config["_repository_root"] = str(repository_root)
    return config


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ManifestError(f"{label} is missing required columns: {missing}")


def require_unique_ids(frame: pd.DataFrame, label: str) -> None:
    if frame["participant_id"].isna().any():
        raise ManifestError(f"{label} contains null participant_id values")
    duplicate_ids = sorted(
        frame.loc[frame["participant_id"].duplicated(keep=False), "participant_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    if duplicate_ids:
        raise ManifestError(f"{label} contains duplicate participant IDs: {duplicate_ids[:10]}")


def _normalise_id_series(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ManifestError("participant_id contains null values")
    values = series.astype(str).str.strip()
    if (values == "").any():
        raise ManifestError("participant_id contains empty values")
    return values


def load_canonical_fold(fold_dir: str | Path) -> pd.DataFrame:
    fold_dir = Path(fold_dir)
    frames: list[pd.DataFrame] = []
    split_sets: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        csv_path = fold_dir / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Canonical split CSV not found: {csv_path}")
        split_frame = pd.read_csv(csv_path)
        if "subject_id" not in split_frame.columns:
            if "participant_id" not in split_frame.columns:
                raise ManifestError(f"{csv_path} has neither subject_id nor participant_id")
            id_column = "participant_id"
        else:
            id_column = "subject_id"
        participant_ids = _normalise_id_series(split_frame[id_column])
        if participant_ids.duplicated().any():
            duplicates = sorted(participant_ids[participant_ids.duplicated(keep=False)].unique())
            raise ManifestError(f"Duplicate IDs in canonical {split} split: {duplicates[:10]}")
        split_sets[split] = set(participant_ids)
        frames.append(pd.DataFrame({"participant_id": participant_ids, "split": split}))

    assert_pairwise_disjoint(split_sets, label=f"canonical fold {fold_dir.name}")
    result = pd.concat(frames, ignore_index=True)
    require_unique_ids(result, f"canonical fold {fold_dir.name}")
    return sort_manifest(result)


def _coerce_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = series.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        bad = sorted(series[converted.isna()].astype(str).unique().tolist())
        raise ManifestError(f"{label} has invalid boolean values: {bad[:10]}")
    return converted.astype(bool)


def load_source_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(path)
    if "participant_id" not in frame.columns:
        if "subject_id" not in frame.columns:
            raise ManifestError(f"Source manifest has neither subject_id nor participant_id: {path}")
        frame = frame.rename(columns={"subject_id": "participant_id"})
    if "source_split" not in frame.columns:
        if "split" not in frame.columns:
            raise ManifestError(f"Source manifest has neither split nor source_split: {path}")
        frame = frame.rename(columns={"split": "source_split"})

    require_columns(frame, REQUIRED_SOURCE_COLUMNS, "source manifest")
    frame = frame.copy()
    frame["participant_id"] = _normalise_id_series(frame["participant_id"])
    require_unique_ids(frame, "source manifest")
    invalid_splits = sorted(set(frame["source_split"].astype(str)) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ManifestError(f"Source manifest has invalid source_split values: {invalid_splits}")
    frame["all_views_present"] = _coerce_bool(
        frame["all_views_present"], "source manifest all_views_present"
    )
    if not frame["all_views_present"].all():
        ids = frame.loc[~frame["all_views_present"], "participant_id"].tolist()
        raise ManifestError(f"Source manifest contains incomplete three-view rows: {ids[:10]}")

    for column in (*VIEW_PATH_COLUMNS, *TABULAR_FEATURE_COLUMNS, *TARGET_COLUMNS):
        if frame[column].isna().any():
            ids = frame.loc[frame[column].isna(), "participant_id"].tolist()
            raise ManifestError(f"Source column '{column}' contains null values for: {ids[:10]}")
    for column in (*TABULAR_FEATURE_COLUMNS, *TARGET_COLUMNS):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.map(math.isfinite)
        if invalid.any():
            ids = frame.loc[invalid, "participant_id"].tolist()
            raise ManifestError(f"Source column '{column}' contains non-finite values for: {ids[:10]}")
    return frame.sort_values("participant_id").reset_index(drop=True)


def assert_pairwise_disjoint(split_sets: Mapping[str, set[str]], label: str) -> None:
    names = list(split_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(split_sets[left] & split_sets[right])
            if overlap:
                raise ManifestError(
                    f"{label}: splits '{left}' and '{right}' overlap: {overlap[:10]}"
                )


def assert_image_files(frame: pd.DataFrame, decode: bool = False) -> None:
    require_columns(frame, VIEW_PATH_COLUMNS, "manifest")
    for column in VIEW_PATH_COLUMNS:
        for participant_id, value in frame[["participant_id", column]].itertuples(index=False):
            path = Path(str(value))
            if not path.is_file():
                raise ManifestError(f"Missing {column} for {participant_id}: {path}")
            if decode:
                try:
                    from PIL import Image

                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        if image.width <= 0 or image.height <= 0:
                            raise ManifestError(f"Invalid image dimensions: {path}")
                except ManifestError:
                    raise
                except Exception as exc:
                    raise ManifestError(f"Could not decode image {path}: {exc}") from exc


def sort_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    split_order = {name: index for index, name in enumerate(SPLIT_NAMES)}
    result["_split_order"] = result["split"].map(split_order)
    return (
        result.sort_values(["_split_order", "participant_id"])
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        writer(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    _atomic_replace(output_path, lambda temporary: frame.to_csv(temporary, index=False))


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)

    def write(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    _atomic_replace(output_path, write)
