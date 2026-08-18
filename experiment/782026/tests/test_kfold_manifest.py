from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from build_kfold_manifests import build_manifests  # noqa: E402
from manifest_utils import ManifestError, load_canonical_fold, load_source_manifest  # noqa: E402
from validate_kfold_manifests import validate_manifests  # noqa: E402


def _make_fixture(tmp_path: Path) -> Path:
    canonical_root = tmp_path / "canonical"
    source_rows = []
    subjects = ["sub-a", "sub-b", "sub-c", "sub-d", "sub-e", "sub-f"]
    roles = {
        0: {"train": subjects[2:5], "valid": [subjects[5]], "test": subjects[:2]},
        1: {"train": [subjects[0], subjects[1], subjects[5]], "valid": [subjects[4]], "test": subjects[2:4]},
        2: {"train": subjects[:3], "valid": [subjects[3]], "test": subjects[4:]},
    }
    for fold, split_map in roles.items():
        fold_dir = canonical_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        for split, ids in split_map.items():
            pd.DataFrame({"subject_id": list(reversed(ids)), "unused": range(len(ids))}).to_csv(
                fold_dir / f"{split}.csv", index=False
            )

    for index, participant_id in enumerate(reversed(subjects)):
        image_dir = tmp_path / "views" / participant_id
        image_dir.mkdir(parents=True)
        paths = {}
        for view in ("Axial", "Coronal", "Sagittal"):
            image_path = image_dir / f"{view}.png"
            Image.new("L", (4, 5), color=index).save(image_path)
            paths[f"{view.lower()}_path"] = str(image_path)
        source_rows.append(
            {
                "subject_id": participant_id,
                "split": ("train", "valid", "test")[index % 3],
                **paths,
                "all_views_present": True,
                "sex": index % 2,
                "age": 40 + index,
                "race": 0,
                "acuteischaemicstroke": 1,
                "priorstroke": 0,
                "bmi": 20 + index,
                "etiology": index % 5,
                "nihss": index,
                "gs_rankin_6isdeath": index % 7,
            }
        )
    source_path = tmp_path / "source.csv"
    pd.DataFrame(source_rows).to_csv(source_path, index=False)
    config = {
        "experiment_id": "test",
        "seed": 42,
        "paths": {
            "canonical_root": str(canonical_root),
            "source_manifest": str(source_path),
            "manifest_root": str(tmp_path / "output"),
        },
        "folds": [0, 1, 2],
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_builder_joins_by_id_preserves_paths_and_source_split(tmp_path: Path) -> None:
    config_path = _make_fixture(tmp_path)
    manifests = build_manifests(config_path)
    assert len(manifests) == 3
    output = pd.read_csv(tmp_path / "output/fold_0/all_subjects.csv")
    source = pd.read_csv(tmp_path / "source.csv").set_index("subject_id")

    assert "source_split" in output.columns
    assert output["participant_id"].is_unique
    for row in output.itertuples(index=False):
        expected = source.loc[row.participant_id]
        assert row.source_split == expected["split"]
        assert row.axial_path == expected["axial_path"]
        assert row.coronal_path == expected["coronal_path"]
        assert row.sagittal_path == expected["sagittal_path"]

    report = validate_manifests(config_path, check_images=True)
    assert report["status"] == "passed"
    assert report["cross_fold"]["all_test_frequency_one"] is True


def test_overwrite_guard(tmp_path: Path) -> None:
    config_path = _make_fixture(tmp_path)
    build_manifests(config_path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        build_manifests(config_path)
    build_manifests(config_path, overwrite=True)


def test_canonical_duplicate_and_overlap_fail(tmp_path: Path) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    pd.DataFrame({"subject_id": ["a", "a"]}).to_csv(fold_dir / "train.csv", index=False)
    pd.DataFrame({"subject_id": ["b"]}).to_csv(fold_dir / "valid.csv", index=False)
    pd.DataFrame({"subject_id": ["c"]}).to_csv(fold_dir / "test.csv", index=False)
    with pytest.raises(ManifestError, match="Duplicate IDs"):
        load_canonical_fold(fold_dir)

    pd.DataFrame({"subject_id": ["a"]}).to_csv(fold_dir / "train.csv", index=False)
    pd.DataFrame({"subject_id": ["a"]}).to_csv(fold_dir / "valid.csv", index=False)
    with pytest.raises(ManifestError, match="overlap"):
        load_canonical_fold(fold_dir)


@pytest.mark.parametrize("mutation, message", [
    ("missing_column", "missing required columns"),
    ("duplicate", "duplicate participant IDs"),
    ("null_target", "null values"),
])
def test_invalid_source_schema_fails(tmp_path: Path, mutation: str, message: str) -> None:
    config_path = _make_fixture(tmp_path)
    source_path = tmp_path / "source.csv"
    frame = pd.read_csv(source_path)
    if mutation == "missing_column":
        frame = frame.drop(columns="axial_path")
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    else:
        frame.loc[0, "nihss"] = None
    frame.to_csv(source_path, index=False)
    with pytest.raises(ManifestError, match=message):
        load_source_manifest(source_path)


def test_missing_image_fails_before_artifact_write(tmp_path: Path) -> None:
    config_path = _make_fixture(tmp_path)
    source = pd.read_csv(tmp_path / "source.csv")
    missing_path = Path(source.loc[0, "axial_path"])
    missing_path.unlink()
    with pytest.raises(ManifestError, match="Missing axial_path"):
        build_manifests(config_path)


def test_missing_source_subject_is_recorded_as_exclusion(tmp_path: Path) -> None:
    config_path = _make_fixture(tmp_path)
    source = pd.read_csv(tmp_path / "source.csv")
    source = source.loc[source["subject_id"] != "sub-a"]
    source.to_csv(tmp_path / "source.csv", index=False)
    build_manifests(config_path)
    dropped = pd.read_csv(tmp_path / "output/fold_0/dropped_subjects.csv")
    assert dropped.to_dict("records") == [
        {"participant_id": "sub-a", "split": "test", "reason": "missing_source_manifest"}
    ]
