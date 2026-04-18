from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


TABULAR_NUMERIC_FEATURES = ["bmi", "age"]
TABULAR_CATEGORICAL_FEATURES = ["etiology"]
TABULAR_ETIOLOGY_CATEGORIES = [1, 2, 3, 4, 5]
TABULAR_PROCESSED_ETIOLOGY_COLUMNS = [
    f"etiology_{category}" for category in TABULAR_ETIOLOGY_CATEGORIES
]
TABULAR_PROCESSED_COLUMNS = [*TABULAR_NUMERIC_FEATURES, *TABULAR_PROCESSED_ETIOLOGY_COLUMNS]


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


def _build_tabular_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[TABULAR_ETIOLOGY_CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, TABULAR_NUMERIC_FEATURES),
            ("cat", categorical_transformer, TABULAR_CATEGORICAL_FEATURES),
        ]
    )


def _build_preprocessed_split_df(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
    final_df = original_df.copy()
    for column in TABULAR_NUMERIC_FEATURES:
        final_df[column] = processed_df[column]

    final_df = final_df.drop(columns=TABULAR_CATEGORICAL_FEATURES)
    final_df = pd.concat([final_df, processed_df[TABULAR_PROCESSED_ETIOLOGY_COLUMNS]], axis=1)
    return final_df


def preprocess_tabular(
    merged_df: pd.DataFrame,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    required_columns = {"split", *TABULAR_NUMERIC_FEATURES, *TABULAR_CATEGORICAL_FEATURES}
    missing_columns = sorted(required_columns - set(merged_df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns for tabular preprocessing: {missing_columns}")

    split_frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_NAMES:
        split_df = merged_df.loc[merged_df["split"] == split].copy()
        if split_df.empty:
            raise ValueError(f"No rows found for split '{split}'")
        split_frames[split] = split_df

    preprocessor = _build_tabular_preprocessor()
    train_features = split_frames["train"][TABULAR_NUMERIC_FEATURES + TABULAR_CATEGORICAL_FEATURES].copy()
    preprocessor.fit(train_features)

    preprocessed_frames: list[pd.DataFrame] = []
    for split in SPLIT_NAMES:
        split_df = split_frames[split]
        split_features = split_df[TABULAR_NUMERIC_FEATURES + TABULAR_CATEGORICAL_FEATURES].copy()
        processed = preprocessor.transform(split_features)
        processed_df = pd.DataFrame(processed, columns=TABULAR_PROCESSED_COLUMNS, index=split_df.index)
        preprocessed_frames.append(_build_preprocessed_split_df(split_df, processed_df))

    merged_preprocessed_df = (
        pd.concat(preprocessed_frames, axis=0)
        .sort_values(["split", "participant_id"])
        .reset_index(drop=True)
    )
    return merged_preprocessed_df, preprocessor


def build_target_dataframe(preprocessed_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    target_columns_to_drop = [column for column in TARGET_COLUMNS if column != target_col]
    return preprocessed_df.drop(columns=target_columns_to_drop, errors="ignore")


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
    preprocessed_df, tabular_preprocessor = preprocess_tabular(merged_df)

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
    preprocessed_feature_cols = [
        column
        for column in tabular_feature_cols
        if column not in {"etiology", "bmi", "age"}
    ]
    preprocessed_feature_cols.extend(["bmi", "age", *TABULAR_PROCESSED_ETIOLOGY_COLUMNS])

    target_dataframes = {
        target: build_target_dataframe(preprocessed_df, target_col=target)
        for target in TARGET_COLUMNS
    }

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
                "tabular_feature_cols": list(preprocessed_feature_cols),
                "all_subjects_csv": str(output_dir / f"all_subjects_preprocessed_{target}.csv"),
                "train_csv": str(output_dir / f"train_preprocessed_{target}.csv"),
                "valid_csv": str(output_dir / f"valid_preprocessed_{target}.csv"),
                "test_csv": str(output_dir / f"test_preprocessed_{target}.csv"),
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
            "all_subjects_preprocessed_gs_rankin_6isdeath_csv": str(
                output_dir / "all_subjects_preprocessed_gs_rankin_6isdeath.csv"
            ),
            "all_subjects_preprocessed_nihss_csv": str(
                output_dir / "all_subjects_preprocessed_nihss.csv"
            ),
            "tabular_preprocessor_joblib": str(output_dir / "tabular_preprocessor.joblib"),
            "dropped_subjects_csv": str(output_dir / "dropped_subjects.csv"),
        },
    }

    merged_df.to_csv(output_dir / "all_subjects.csv", index=False)
    for split in SPLIT_NAMES:
        merged_df.loc[merged_df["split"] == split].to_csv(output_dir / f"{split}.csv", index=False)

    for target, target_df in target_dataframes.items():
        target_df.to_csv(output_dir / f"all_subjects_preprocessed_{target}.csv", index=False)
        for split in SPLIT_NAMES:
            target_df.loc[target_df["split"] == split].to_csv(
                output_dir / f"{split}_preprocessed_{target}.csv", index=False
            )

    joblib.dump(tabular_preprocessor, output_dir / "tabular_preprocessor.joblib")
    pd.DataFrame(dropped_records).to_csv(output_dir / "dropped_subjects.csv", index=False)
    save_json(output_dir / "manifest.json", manifest)

    print(f"Saved manifest: {output_dir / 'manifest.json'}")
    print(f"Merged subjects: {manifest['counts']['merged_subjects_total']}")
    print(f"Split counts: {manifest['counts']['split_counts']}")


if __name__ == "__main__":
    main()
