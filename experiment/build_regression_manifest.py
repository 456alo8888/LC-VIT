from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_TABULAR_CSV,
    SPLIT_NAMES,
    TARGET_COLUMNS,
    VIEW_NAMES,
    ensure_dir,
    save_json,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LC-VIT fixed-split regression manifest")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--tabular-csv", type=Path, default=DEFAULT_TABULAR_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    return parser.parse_args()


def collect_image_records(image_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    dropped: list[dict] = []

    for split in SPLIT_NAMES:
        split_dir = image_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for subject_dir in sorted(split_dir.iterdir()):
            if not subject_dir.is_dir():
                continue

            view_paths = {view: subject_dir / f"{view}.png" for view in VIEW_NAMES}
            missing_views = [view for view, path in view_paths.items() if not path.is_file()]

            row = {
                "participant_id": subject_dir.name,
                "split": split,
                "axial_path": str(view_paths["Axial"]),
                "coronal_path": str(view_paths["Coronal"]),
                "sagittal_path": str(view_paths["Sagittal"]),
                "all_views_present": len(missing_views) == 0,
            }
            rows.append(row)

            if missing_views:
                dropped.append(
                    {
                        "participant_id": subject_dir.name,
                        "split": split,
                        "reason": "missing_view",
                        "missing_views": missing_views,
                    }
                )

    return pd.DataFrame(rows), dropped


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    image_df, dropped_records = collect_image_records(args.image_root)
    complete_df = image_df[image_df["all_views_present"]].copy()

    tabular_df = pd.read_csv(args.tabular_csv)
    if "participant_id" not in tabular_df.columns:
        raise ValueError(f"'participant_id' column not found in {args.tabular_csv}")

    merged_df = complete_df.merge(tabular_df, on="participant_id", how="inner")
    merged_df = merged_df.sort_values(["split", "participant_id"]).reset_index(drop=True)

    dropped_ids = {record["participant_id"] for record in dropped_records}
    tabular_only_ids = sorted(set(tabular_df["participant_id"]) - set(image_df["participant_id"]))
    missing_tabular_ids = sorted(set(complete_df["participant_id"]) - set(tabular_df["participant_id"]))

    for participant_id in missing_tabular_ids:
        split = complete_df.loc[complete_df["participant_id"] == participant_id, "split"].iloc[0]
        dropped_records.append(
            {
                "participant_id": participant_id,
                "split": split,
                "reason": "missing_tabular",
                "missing_views": [],
            }
        )

    tabular_feature_cols = [
        column
        for column in tabular_df.columns
        if column not in {"participant_id", *TARGET_COLUMNS}
    ]

    manifest = {
        "created_at": utc_now_iso(),
        "image_root": str(args.image_root),
        "tabular_csv": str(args.tabular_csv),
        "split_protocol": "fixed_soop_views",
        "view_names": list(VIEW_NAMES),
        "target_columns": list(TARGET_COLUMNS),
        "tabular_feature_cols": tabular_feature_cols,
        "counts": {
            "image_subjects_total": int(image_df["participant_id"].nunique()),
            "image_subjects_complete": int(complete_df["participant_id"].nunique()),
            "tabular_subjects_total": int(tabular_df["participant_id"].nunique()),
            "merged_subjects_total": int(merged_df["participant_id"].nunique()),
            "split_counts": {
                split: int(merged_df.loc[merged_df["split"] == split, "participant_id"].nunique())
                for split in SPLIT_NAMES
            },
        },
        "target_configs": {
            target: {
                "target_col": target,
                "tabular_feature_cols": tabular_feature_cols,
            }
            for target in TARGET_COLUMNS
        },
        "dropped_subjects": dropped_records,
        "tabular_only_subjects": tabular_only_ids,
        "subjects_missing_tabular": missing_tabular_ids,
        "files": {
            "all_subjects_csv": str(output_dir / "all_subjects.csv"),
            "train_csv": str(output_dir / "train.csv"),
            "valid_csv": str(output_dir / "valid.csv"),
            "test_csv": str(output_dir / "test.csv"),
            "dropped_subjects_csv": str(output_dir / "dropped_subjects.csv"),
        },
    }

    merged_df.to_csv(output_dir / "all_subjects.csv", index=False)
    for split in SPLIT_NAMES:
        merged_df.loc[merged_df["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)
    pd.DataFrame(dropped_records).to_csv(output_dir / "dropped_subjects.csv", index=False)
    save_json(output_dir / "manifest.json", manifest)

    print(f"Saved manifest: {output_dir / 'manifest.json'}")
    print(f"Merged subjects: {manifest['counts']['merged_subjects_total']}")
    print(f"Split counts: {manifest['counts']['split_counts']}")


if __name__ == "__main__":
    main()
