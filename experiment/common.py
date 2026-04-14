from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - only hit outside the experiment env
    yaml = None


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_ROOT = Path("/mnt/disk2/SOOP_views")
DEFAULT_TABULAR_CSV = Path(
    "../../../preprocess_MRI/processed_tabular/clinical_encoded.csv"
)
DEFAULT_ARTIFACT_ROOT = EXPERIMENT_DIR / "artifacts"
DEFAULT_MANIFEST_DIR = DEFAULT_ARTIFACT_ROOT / "manifest_fixed_split"
DEFAULT_FEATURE_DIR = DEFAULT_ARTIFACT_ROOT / "features"
DEFAULT_MERGED_DIR = DEFAULT_ARTIFACT_ROOT / "merged"
DEFAULT_RUNS_DIR = EXPERIMENT_DIR / "runs"
VIEW_NAMES = ("Axial", "Coronal", "Sagittal")
SPLIT_NAMES = ("train", "valid", "test")
TARGET_COLUMNS = ("gs_rankin_6isdeath", "nihss")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required to read yaml files.")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    if yaml is None:
        raise ImportError("PyYAML is required to write yaml files.")
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def standardize_id_column_name(column_name: str) -> str:
    lowered = column_name.strip().lower()
    if lowered in {"participant_id", "patient_id", "patientid", "patient_id "}:
        return "participant_id"
    if lowered == "patient_id":
        return "participant_id"
    return column_name


def feature_column_prefix(view_name: str) -> str:
    return view_name.lower()


def format_feature_columns(view_name: str, feature_dim: int) -> list[str]:
    prefix = feature_column_prefix(view_name)
    return [f"{prefix}_feature_{index:04d}" for index in range(feature_dim)]
