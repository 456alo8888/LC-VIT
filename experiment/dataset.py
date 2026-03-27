from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import TARGET_COLUMNS, load_json


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    tabular_feature_cols: list[str]
    view_feature_cols: Dict[str, list[str]]


def load_dataset_bundle(merged_manifest_path: Path) -> DatasetBundle:
    merged_manifest = load_json(merged_manifest_path)
    dataframe = pd.read_csv(merged_manifest["merged_csv"])
    tabular_feature_cols = list(merged_manifest["tabular_feature_cols"])
    view_feature_cols = {key: list(value) for key, value in merged_manifest["view_feature_cols"].items()}
    return DatasetBundle(dataframe=dataframe, tabular_feature_cols=tabular_feature_cols, view_feature_cols=view_feature_cols)


def split_dataframe(dataframe: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        split: dataframe.loc[dataframe["split"] == split].copy().reset_index(drop=True)
        for split in ("train", "valid", "test")
    }


def compute_tabular_stats(dataframe: pd.DataFrame, columns: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    values = dataframe.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def debug_shapes(dataframe: pd.DataFrame, tabular_cols: list[str], view_feature_cols: Dict[str, list[str]]) -> Dict[str, int]:
    if dataframe.empty:
        return {
            "n_samples": 0,
            "clinical_dim": len(tabular_cols),
            "axial_dim": len(view_feature_cols["Axial"]),
            "coronal_dim": len(view_feature_cols["Coronal"]),
            "sagittal_dim": len(view_feature_cols["Sagittal"]),
        }

    return {
        "n_samples": int(len(dataframe)),
        "clinical_dim": int(len(tabular_cols)),
        "axial_dim": int(len(view_feature_cols["Axial"])),
        "coronal_dim": int(len(view_feature_cols["Coronal"])),
        "sagittal_dim": int(len(view_feature_cols["Sagittal"])),
    }


class LCVITRegressionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_col: str,
        tabular_feature_cols: list[str],
        view_feature_cols: Dict[str, list[str]],
        tabular_mean: np.ndarray | None = None,
        tabular_std: np.ndarray | None = None,
    ):
        if target_col not in TARGET_COLUMNS:
            raise ValueError(f"Unsupported target column: {target_col}")

        self.dataframe = dataframe.reset_index(drop=True)
        self.target_col = target_col
        self.tabular_feature_cols = list(tabular_feature_cols)
        self.view_feature_cols = {key: list(value) for key, value in view_feature_cols.items()}

        tabular = (
            self.dataframe.loc[:, self.tabular_feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        if tabular_mean is not None and tabular_std is not None:
            tabular = (tabular - tabular_mean.astype(np.float32)) / tabular_std.astype(np.float32)

        self.tabular = tabular
        self.targets = self.dataframe[target_col].to_numpy(dtype=np.float32)
        self.participant_ids = self.dataframe["participant_id"].astype(str).tolist()
        self.axial = self.dataframe[self.view_feature_cols["Axial"]].to_numpy(dtype=np.float32)
        self.coronal = self.dataframe[self.view_feature_cols["Coronal"]].to_numpy(dtype=np.float32)
        self.sagittal = self.dataframe[self.view_feature_cols["Sagittal"]].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.participant_ids)

    def __getitem__(self, index: int):
        return {
            "participant_id": self.participant_ids[index],
            "clinical": torch.from_numpy(self.tabular[index]),
            "axial": torch.from_numpy(self.axial[index]),
            "coronal": torch.from_numpy(self.coronal[index]),
            "sagittal": torch.from_numpy(self.sagittal[index]),
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
        }
