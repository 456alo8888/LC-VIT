from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MERGED_DIR,
    VIEW_NAMES,
    ensure_dir,
    feature_column_prefix,
    format_feature_columns,
    load_json,
    save_json,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LC-VIT 3-view features with tabular manifest")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MERGED_DIR)
    return parser.parse_args()


def _load_feature_csv(path: Path, view_name: str) -> tuple[pd.DataFrame, list[str]]:
    dataframe = pd.read_csv(path)
    if "participant_id" not in dataframe.columns:
        if "Patient_ID" in dataframe.columns:
            dataframe = dataframe.rename(columns={"Patient_ID": "participant_id"})
        else:
            raise ValueError(f"participant_id column not found in {path}")

    raw_feature_cols = [column for column in dataframe.columns if column != "participant_id"]
    prefixed_cols = format_feature_columns(view_name, len(raw_feature_cols))
    rename_map = {old: new for old, new in zip(raw_feature_cols, prefixed_cols)}
    dataframe = dataframe.rename(columns=rename_map)
    return dataframe, prefixed_cols


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest_dir / "manifest.json")
    feature_manifest = load_json(args.feature_dir / "feature_manifest.json")
    merged_df = pd.read_csv(args.manifest_dir / "all_subjects.csv")

    view_feature_cols: dict[str, list[str]] = {}
    for view_name in VIEW_NAMES:
        feature_path = Path(feature_manifest["files"][view_name])
        feature_df, feature_cols = _load_feature_csv(feature_path, view_name=view_name)
        merged_df = merged_df.merge(feature_df, on="participant_id", how="inner")
        view_feature_cols[view_name] = feature_cols

    output_dir = ensure_dir(args.output_dir)
    merged_csv = output_dir / "merged_features.csv"
    merged_df = merged_df.sort_values(["split", "participant_id"]).reset_index(drop=True)
    merged_df.to_csv(merged_csv, index=False)

    merged_manifest = {
        "created_at": utc_now_iso(),
        "source_manifest": str(args.manifest_dir / "manifest.json"),
        "source_feature_manifest": str(args.feature_dir / "feature_manifest.json"),
        "merged_csv": str(merged_csv),
        "feature_extractor": feature_manifest["extractor"],
        "model_name": feature_manifest["model_name"],
        "view_feature_cols": view_feature_cols,
        "view_feature_dims": {view: len(columns) for view, columns in view_feature_cols.items()},
        "tabular_feature_cols": manifest["tabular_feature_cols"],
        "target_columns": manifest["target_columns"],
        "split_counts": {
            split: int(merged_df.loc[merged_df["split"] == split, "participant_id"].nunique())
            for split in ("train", "valid", "test")
        },
        "n_subjects": int(merged_df["participant_id"].nunique()),
        "files": {
            "merged_csv": str(merged_csv),
        },
    }
    save_json(output_dir / "merged_manifest.json", merged_manifest)

    print(f"Saved merged manifest: {output_dir / 'merged_manifest.json'}")
    print(f"Merged subjects: {merged_manifest['n_subjects']}")
    print(f"Split counts: {merged_manifest['split_counts']}")


if __name__ == "__main__":
    main()
