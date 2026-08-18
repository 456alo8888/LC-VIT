from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from manifest_utils import (
    EXPECTED_EXCLUSIONS,
    EXPECTED_FOLD_COUNTS,
    REQUIRED_SOURCE_COLUMNS,
    SPLIT_NAMES,
    TABULAR_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    VIEW_PATH_COLUMNS,
    ManifestError,
    assert_image_files,
    assert_pairwise_disjoint,
    atomic_write_json,
    load_canonical_fold,
    load_config,
    require_columns,
    require_unique_ids,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LC-VIT canonical K-fold manifests")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-images", action="store_true", help="Decode images in addition to existence checks")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestError(f"Expected JSON object: {path}")
    return value


def _finite_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = values.isna() | ~values.map(math.isfinite)
        if invalid.any():
            ids = frame.loc[invalid, "participant_id"].tolist()
            raise ManifestError(f"Column '{column}' has non-finite values for {ids[:10]}")


def validate_fold(
    *,
    fold: int,
    canonical_root: Path,
    manifest_root: Path,
    check_images: bool,
) -> tuple[dict[str, Any], set[str]]:
    fold_dir = manifest_root / f"fold_{fold}"
    required_files = [
        fold_dir / "manifest.json",
        fold_dir / "all_subjects.csv",
        *(fold_dir / f"{split}.csv" for split in SPLIT_NAMES),
        fold_dir / "dropped_subjects.csv",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise ManifestError(f"Fold {fold} is missing artifacts: {missing_files}")

    metadata = _load_json(fold_dir / "manifest.json")
    all_subjects = pd.read_csv(fold_dir / "all_subjects.csv")
    require_columns(all_subjects, ("participant_id", "split", *REQUIRED_SOURCE_COLUMNS[1:]), f"fold {fold}")
    require_unique_ids(all_subjects, f"fold {fold}")
    invalid_splits = sorted(set(all_subjects["split"].astype(str)) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ManifestError(f"Fold {fold} has invalid split values: {invalid_splits}")
    if "source_split" not in all_subjects or all_subjects["source_split"].isna().any():
        raise ManifestError(f"Fold {fold} does not preserve source_split")
    invalid_source_splits = sorted(
        set(all_subjects["source_split"].astype(str)) - set(SPLIT_NAMES)
    )
    if invalid_source_splits:
        raise ManifestError(
            f"Fold {fold} has invalid source_split values: {invalid_source_splits}"
        )
    if not all_subjects["all_views_present"].astype(str).str.lower().isin(("true", "1")).all():
        raise ManifestError(f"Fold {fold} contains all_views_present=False")
    _finite_numeric(all_subjects, (*TABULAR_FEATURE_COLUMNS, *TARGET_COLUMNS))
    assert_image_files(all_subjects, decode=check_images)

    output_sets: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}
    for split in SPLIT_NAMES:
        split_frame = pd.read_csv(fold_dir / f"{split}.csv")
        require_columns(split_frame, all_subjects.columns, f"fold {fold} {split}.csv")
        if set(split_frame["split"].astype(str)) != {split}:
            raise ManifestError(f"Fold {fold} {split}.csv contains rows with another split role")
        output_sets[split] = set(split_frame["participant_id"].astype(str))
        expected_ids = set(all_subjects.loc[all_subjects["split"] == split, "participant_id"].astype(str))
        if output_sets[split] != expected_ids:
            raise ManifestError(f"Fold {fold} {split}.csv does not match all_subjects.csv")
        split_counts[split] = len(output_sets[split])
    assert_pairwise_disjoint(output_sets, label=f"output fold {fold}")
    output_union = set().union(*output_sets.values())
    if output_union != set(all_subjects["participant_id"].astype(str)):
        raise ManifestError(f"Fold {fold} split union does not match all_subjects.csv")

    canonical = load_canonical_fold(canonical_root / f"fold_{fold}")
    canonical_ids = set(canonical["participant_id"])
    dropped = pd.read_csv(fold_dir / "dropped_subjects.csv")
    require_columns(dropped, ("participant_id", "split", "reason"), f"fold {fold} dropped subjects")
    if dropped["participant_id"].duplicated().any():
        raise ManifestError(f"Fold {fold} dropped_subjects.csv contains duplicate IDs")
    dropped_ids = set(dropped["participant_id"].astype(str))
    if output_union != canonical_ids - dropped_ids:
        raise ManifestError(f"Fold {fold} output membership is not canonical minus dropped IDs")
    expected_dropped_roles = canonical.set_index("participant_id")["split"].to_dict()
    for row in dropped.itertuples(index=False):
        if expected_dropped_roles.get(str(row.participant_id)) != str(row.split):
            raise ManifestError(f"Fold {fold} has wrong dropped role for {row.participant_id}")
        if str(row.reason) != "missing_source_manifest":
            raise ManifestError(f"Fold {fold} has an unexpected drop reason for {row.participant_id}")
    for split in SPLIT_NAMES:
        canonical_split_ids = set(
            canonical.loc[canonical["split"] == split, "participant_id"].astype(str)
        )
        dropped_split_ids = set(
            dropped.loc[dropped["split"] == split, "participant_id"].astype(str)
        )
        if output_sets[split] != canonical_split_ids - dropped_split_ids:
            raise ManifestError(
                f"Fold {fold} {split} membership differs from canonical membership minus exclusions"
            )

    if len(canonical_ids) == 622:
        if dropped_ids != set(EXPECTED_EXCLUSIONS):
            raise ManifestError(f"Fold {fold} exclusions differ from {list(EXPECTED_EXCLUSIONS)}")
        if split_counts != EXPECTED_FOLD_COUNTS[fold]:
            raise ManifestError(
                f"Fold {fold} counts are {split_counts}, expected {EXPECTED_FOLD_COUNTS[fold]}"
            )

    if int(metadata.get("fold", -1)) != fold:
        raise ManifestError(f"Fold {fold} manifest.json has an incorrect fold index")
    if int(metadata.get("usable_subject_count", -1)) != len(all_subjects):
        raise ManifestError(f"Fold {fold} manifest usable_subject_count is stale")
    if metadata.get("counts", {}).get("split_counts") != split_counts:
        raise ManifestError(f"Fold {fold} manifest split_counts are stale")
    if set(metadata.get("excluded_subjects", [])) != dropped_ids:
        raise ManifestError(f"Fold {fold} manifest excluded_subjects are stale")
    checksums = metadata.get("checksums", {})
    source_manifest = Path(metadata.get("sources", {}).get("source_manifest", ""))
    if not source_manifest.is_file() or checksums.get("source_manifest_sha256") != sha256_file(source_manifest):
        raise ManifestError(f"Fold {fold} source-manifest checksum is stale")
    canonical_checksums = checksums.get("canonical_split_sha256", {})
    for split in SPLIT_NAMES:
        canonical_path = canonical_root / f"fold_{fold}" / f"{split}.csv"
        if canonical_checksums.get(split) != sha256_file(canonical_path):
            raise ManifestError(f"Fold {fold} canonical {split} checksum is stale")

    return {
        "fold": fold,
        "status": "passed",
        "canonical_subject_count": len(canonical_ids),
        "usable_subject_count": len(output_union),
        "split_counts": split_counts,
        "excluded_subjects": sorted(dropped_ids),
        "images_decoded": bool(check_images),
    }, output_sets["test"]


def validate_manifests(
    config_path: str | Path,
    *,
    check_images: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    paths = config["paths"]
    canonical_root = Path(paths["canonical_root"])
    manifest_root = Path(paths["manifest_root"])
    folds = [
        int(str(value)[len("fold_") :] if str(value).startswith("fold_") else str(value))
        for value in config.get("folds", range(5))
    ]
    report: dict[str, Any] = {
        "validated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "config": str(config["_config_path"]),
        "check_images": check_images,
        "status": "passed",
        "folds": [],
        "errors": [],
    }
    test_sets: dict[int, set[str]] = {}
    for fold in folds:
        try:
            fold_report, test_ids = validate_fold(
                fold=fold,
                canonical_root=canonical_root,
                manifest_root=manifest_root,
                check_images=check_images,
            )
            report["folds"].append(fold_report)
            test_sets[fold] = test_ids
        except Exception as exc:
            report["status"] = "failed"
            report["errors"].append({"fold": fold, "error": str(exc)})

    if not report["errors"] and len(folds) > 1:
        canonical_universe = set(load_canonical_fold(canonical_root / f"fold_{folds[0]}")["participant_id"])
        excluded = set(EXPECTED_EXCLUSIONS) if len(canonical_universe) == 622 else set()
        expected_universe = canonical_universe - excluded
        frequencies = Counter(participant_id for ids in test_sets.values() for participant_id in ids)
        bad_frequency = sorted(
            participant_id for participant_id in expected_universe if frequencies[participant_id] != 1
        )
        unexpected = sorted(set(frequencies) - expected_universe)
        if bad_frequency or unexpected:
            report["status"] = "failed"
            report["errors"].append(
                {
                    "scope": "cross_fold",
                    "error": "Test-fold coverage is not exactly once per usable subject",
                    "bad_frequency_ids": bad_frequency[:20],
                    "unexpected_ids": unexpected[:20],
                }
            )
        report["cross_fold"] = {
            "expected_subject_count": len(expected_universe),
            "test_union_count": len(frequencies),
            "all_test_frequency_one": not bad_frequency and not unexpected,
        }

    final_output = Path(output_path) if output_path else manifest_root / "validation_report.json"
    atomic_write_json(report, final_output)
    return report


def main() -> None:
    args = parse_args()
    report = validate_manifests(
        args.config,
        check_images=args.check_images,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
