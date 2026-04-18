from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_MERGED_DIR,
    TARGET_COLUMNS,
    VIEW_NAMES,
    ensure_dir,
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


def _resolve_target_source_csv(manifest: dict, target: str) -> Path:
    target_config = manifest.get("target_configs", {}).get(target, {})
    target_csv = target_config.get("all_subjects_csv")
    if target_csv:
        return Path(target_csv)

    if target == "gs_rankin_6isdeath":
        legacy_gs_csv = manifest.get("files", {}).get("all_subjects_gs_rankin_6isdeath_preprocessed_csv")
        if legacy_gs_csv:
            return Path(legacy_gs_csv)
    return Path(manifest["files"]["all_subjects_csv"])


def _merge_view_features(
    source_df: pd.DataFrame,
    feature_dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    merged_df = source_df.copy()
    for view_name in VIEW_NAMES:
        merged_df = merged_df.merge(feature_dataframes[view_name], on="participant_id", how="inner")
    return merged_df


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest_dir / "manifest.json")
    feature_manifest = load_json(args.feature_dir / "feature_manifest.json")

    feature_dataframes: dict[str, pd.DataFrame] = {}
    view_feature_cols: dict[str, list[str]] = {}
    for view_name in VIEW_NAMES:
        feature_path = Path(feature_manifest["files"][view_name])
        feature_df, feature_cols = _load_feature_csv(feature_path, view_name=view_name)
        feature_dataframes[view_name] = feature_df
        view_feature_cols[view_name] = feature_cols

    output_dir = ensure_dir(args.output_dir)

    merged_by_target: dict[str, pd.DataFrame] = {}
    merged_csv_by_target: dict[str, str] = {}
    merged_target_configs: dict[str, dict] = {}

    target_columns = [str(target) for target in manifest.get("target_columns", list(TARGET_COLUMNS))]
    for target in target_columns:
        source_csv = _resolve_target_source_csv(manifest=manifest, target=target)
        source_df = pd.read_csv(source_csv)
        merged_df = _merge_view_features(source_df=source_df, feature_dataframes=feature_dataframes)
        merged_df = merged_df.sort_values(["split", "participant_id"]).reset_index(drop=True)

        target_merged_csv = output_dir / f"merged_features_{target}.csv"
        merged_df.to_csv(target_merged_csv, index=False)

        merged_by_target[target] = merged_df
        merged_csv_by_target[target] = str(target_merged_csv)

        source_target_config = manifest.get("target_configs", {}).get(target, {})
        merged_target_configs[target] = {
            "target_col": target,
            "tabular_feature_cols": source_target_config.get(
                "tabular_feature_cols", manifest.get("tabular_feature_cols", [])
            ),
            "merged_csv": str(target_merged_csv),
        }

    default_target = target_columns[0]
    merged_df = merged_by_target[default_target]
    merged_csv = Path(merged_csv_by_target[default_target])
    default_tabular_cols = merged_target_configs[default_target]["tabular_feature_cols"]

    merged_manifest = {
        "created_at": utc_now_iso(),
        "source_manifest": str(args.manifest_dir / "manifest.json"),
        "source_feature_manifest": str(args.feature_dir / "feature_manifest.json"),
        "merged_csv": str(merged_csv),
        "feature_extractor": feature_manifest["extractor"],
        "model_name": feature_manifest["model_name"],
        "view_feature_cols": view_feature_cols,
        "view_feature_dims": {view: len(columns) for view, columns in view_feature_cols.items()},
        "tabular_feature_cols": default_tabular_cols,
        "target_columns": target_columns,
        "target_configs": merged_target_configs,
        "split_counts": {
            split: int(merged_df.loc[merged_df["split"] == split, "participant_id"].nunique())
            for split in ("train", "valid", "test")
        },
        "n_subjects": int(merged_df["participant_id"].nunique()),
        "files": {
            "merged_csv": str(merged_csv),
            "merged_csv_by_target": merged_csv_by_target,
        },
    }
    save_json(output_dir / "merged_manifest.json", merged_manifest)

    print(f"Saved merged manifest: {output_dir / 'merged_manifest.json'}")
    print(f"Merged subjects: {merged_manifest['n_subjects']}")
    print(f"Split counts: {merged_manifest['split_counts']}")


if __name__ == "__main__":
    main()
