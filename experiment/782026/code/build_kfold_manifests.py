from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from manifest_utils import (
    EXPECTED_EXCLUSIONS,
    SPLIT_NAMES,
    TABULAR_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    VIEW_PATH_COLUMNS,
    ManifestError,
    assert_image_files,
    atomic_write_csv,
    atomic_write_json,
    load_canonical_fold,
    load_config,
    load_source_manifest,
    sha256_file,
    sort_manifest,
)


OUTPUT_FILENAMES = (
    "manifest.json",
    "all_subjects.csv",
    "train.csv",
    "valid.csv",
    "test.csv",
    "dropped_subjects.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical LC-VIT three-view K-fold manifests")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _fold_index(value: Any) -> int:
    if isinstance(value, str) and value.startswith("fold_"):
        value = value[len("fold_") :]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Invalid fold index: {value!r}") from exc


def _validate_output_guard(manifest_root: Path, folds: list[int], overwrite: bool) -> None:
    existing = [
        manifest_root / f"fold_{fold}" / filename
        for fold in folds
        for filename in OUTPUT_FILENAMES
        if (manifest_root / f"fold_{fold}" / filename).exists()
    ]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        raise FileExistsError(f"K-fold artifacts already exist ({preview}); pass --overwrite to replace")


def build_fold(
    *,
    fold: int,
    canonical_root: Path,
    source_frame: pd.DataFrame,
    source_manifest_path: Path,
    manifest_root: Path,
    config_path: Path,
    experiment_id: str,
    seed: int,
) -> dict[str, Any]:
    canonical_dir = canonical_root / f"fold_{fold}"
    canonical = load_canonical_fold(canonical_dir)
    canonical_ids = set(canonical["participant_id"])
    source_ids = set(source_frame["participant_id"])
    missing_ids = sorted(canonical_ids - source_ids)

    if len(canonical_ids) == 622 and tuple(missing_ids) != EXPECTED_EXCLUSIONS:
        raise ManifestError(
            f"Fold {fold}: expected exclusions {list(EXPECTED_EXCLUSIONS)}, got {missing_ids}"
        )

    merged = canonical.merge(source_frame, on="participant_id", how="inner", validate="one_to_one")
    if len(merged) != len(canonical_ids & source_ids):
        raise ManifestError(f"Fold {fold}: ID join produced an unexpected row count")
    assert_image_files(merged, decode=False)
    merged = sort_manifest(merged)

    dropped = canonical.loc[canonical["participant_id"].isin(missing_ids), ["participant_id", "split"]].copy()
    dropped["reason"] = "missing_source_manifest"
    dropped = sort_manifest(dropped) if not dropped.empty else pd.DataFrame(
        columns=["participant_id", "split", "reason"]
    )

    output_dir = manifest_root / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {
        split: int((merged["split"] == split).sum()) for split in SPLIT_NAMES
    }
    canonical_counts = {
        split: int((canonical["split"] == split).sum()) for split in SPLIT_NAMES
    }

    canonical_files = {
        split: str(canonical_dir / f"{split}.csv") for split in SPLIT_NAMES
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": experiment_id,
        "fold": fold,
        "seed": seed,
        "split_protocol": "canonical_5fold_seed42_lcvit_3view_intersection",
        "canonical_subject_count": int(len(canonical)),
        "usable_subject_count": int(len(merged)),
        "excluded_subjects": missing_ids,
        "view_names": ["Axial", "Coronal", "Sagittal"],
        "view_path_columns": list(VIEW_PATH_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "tabular_feature_cols": list(TABULAR_FEATURE_COLUMNS),
        "counts": {
            "canonical_split_counts": canonical_counts,
            "split_counts": split_counts,
            "dropped_subjects": int(len(dropped)),
        },
        "sources": {
            "canonical_fold_dir": str(canonical_dir),
            "canonical_split_csvs": canonical_files,
            "source_manifest": str(source_manifest_path),
            "config": str(config_path),
        },
        "checksums": {
            "config_sha256": sha256_file(config_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "canonical_split_sha256": {
                split: sha256_file(path) for split, path in canonical_files.items()
            },
        },
        "files": {
            "all_subjects_csv": str(output_dir / "all_subjects.csv"),
            "train_csv": str(output_dir / "train.csv"),
            "valid_csv": str(output_dir / "valid.csv"),
            "test_csv": str(output_dir / "test.csv"),
            "dropped_subjects_csv": str(output_dir / "dropped_subjects.csv"),
        },
    }

    atomic_write_csv(merged, output_dir / "all_subjects.csv")
    for split in SPLIT_NAMES:
        atomic_write_csv(merged.loc[merged["split"] == split].reset_index(drop=True), output_dir / f"{split}.csv")
    atomic_write_csv(dropped, output_dir / "dropped_subjects.csv")
    atomic_write_json(manifest, output_dir / "manifest.json")
    return manifest


def build_manifests(config_path: str | Path, overwrite: bool = False) -> list[dict[str, Any]]:
    config = load_config(config_path)
    paths = config["paths"]
    required_paths = ("canonical_root", "source_manifest", "manifest_root")
    missing = [key for key in required_paths if key not in paths]
    if missing:
        raise ManifestError(f"Configuration is missing path keys: {missing}")

    canonical_root = Path(paths["canonical_root"])
    source_manifest_path = Path(paths["source_manifest"])
    manifest_root = Path(paths["manifest_root"])
    folds = [_fold_index(value) for value in config.get("folds", range(5))]
    if len(set(folds)) != len(folds):
        raise ManifestError(f"Configuration contains duplicate folds: {folds}")
    _validate_output_guard(manifest_root, folds, overwrite)

    source_frame = load_source_manifest(source_manifest_path)
    results = []
    for fold in folds:
        results.append(
            build_fold(
                fold=fold,
                canonical_root=canonical_root,
                source_frame=source_frame,
                source_manifest_path=source_manifest_path,
                manifest_root=manifest_root,
                config_path=Path(config["_config_path"]),
                experiment_id=str(config.get("experiment_id", "LCVIT_782026")),
                seed=int(config.get("seed", 42)),
            )
        )
    return results


def main() -> None:
    args = parse_args()
    manifests = build_manifests(args.config, overwrite=args.overwrite)
    counts = {f"fold_{item['fold']}": item["counts"]["split_counts"] for item in manifests}
    print(f"Built {len(manifests)} folds: {counts}")


if __name__ == "__main__":
    main()
