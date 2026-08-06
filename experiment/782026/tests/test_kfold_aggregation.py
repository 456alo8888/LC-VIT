from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from aggregate_kfold_results import (  # noqa: E402
    AggregationError,
    REPORT_METRICS,
    aggregate_group,
    summarize_fold_metrics,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from metrics import compute_regression_metrics  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_fold(
    root: Path,
    *,
    fold: int,
    ids: list[str],
    target: str = "nihss",
    mode: str = "fusion",
    seed: int = 42,
    y_true: list[float] | None = None,
    y_pred: list[float] | None = None,
) -> None:
    y_true = y_true if y_true is not None else [float(fold + i) for i in range(len(ids))]
    y_pred = y_pred if y_pred is not None else [value + 0.5 for value in y_true]
    manifest_dir = root / "manifests" / f"fold_{fold}"
    run_dir = root / "runs" / target / mode / f"fold_{fold}" / f"seed{seed}"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"participant_id": ids, "split": "test"}).to_csv(
        manifest_dir / "test.csv", index=False
    )
    predictions = pd.DataFrame(
        {"participant_id": ids, "y_true": y_true, "y_pred": y_pred}
    )
    predictions["abs_error"] = (predictions["y_pred"] - predictions["y_true"]).abs()
    predictions["squared_error"] = (
        predictions["y_pred"] - predictions["y_true"]
    ).pow(2)
    predictions_path = run_dir / "predictions" / "test_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    metrics = compute_regression_metrics(np.asarray(y_true), np.asarray(y_pred))
    _write_json(run_dir / "metrics" / "test_metrics.json", metrics)
    _write_json(
        run_dir / "manifest.json",
        {
            "target_col": target,
            "model_mode": mode,
            "seed": seed,
            "final_eval": False,
            "manifest_dir": str(manifest_dir),
        },
    )


def _aggregate(root: Path, folds: list[int], **kwargs):
    return aggregate_group(
        target=kwargs.pop("target", "nihss"),
        mode=kwargs.pop("mode", "fusion"),
        seed=kwargs.pop("seed", 42),
        folds=folds,
        manifest_root=root / "manifests",
        runs_root=root / "runs",
        aggregate_root=root / "aggregate",
        require_complete=kwargs.pop("require_complete", True),
        **kwargs,
    )


def test_summary_uses_unweighted_mean_and_sample_std() -> None:
    dataframe = pd.DataFrame(
        {
            "fold": [0, 1, 2],
            **{metric: [1.0, 2.0, 4.0] for metric in REPORT_METRICS},
        }
    )
    summary, payload = summarize_fold_metrics(dataframe)

    expected_mean = np.mean([1.0, 2.0, 4.0])
    expected_std = np.std([1.0, 2.0, 4.0], ddof=1)
    assert np.allclose(summary["mean"], expected_mean)
    assert np.allclose(summary["std"], expected_std)
    assert payload["std_ddof"] == 1
    assert payload["metrics"]["mae"]["std"] == pytest.approx(expected_std)


def test_pooled_metrics_and_unequal_fold_sizes(tmp_path: Path) -> None:
    _make_fold(tmp_path, fold=0, ids=["sub-1"], y_true=[0.0], y_pred=[1.0])
    _make_fold(
        tmp_path,
        fold=1,
        ids=["sub-2", "sub-3", "sub-4"],
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 2.0, 5.0],
    )

    output = _aggregate(tmp_path, [0, 1])
    assert output is not None
    oof = pd.read_csv(output / "oof_predictions.csv")
    pooled = json.loads((output / "oof_metrics.json").read_text(encoding="utf-8"))
    expected = compute_regression_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())

    assert len(oof) == 4
    assert oof["participant_id"].nunique() == 4
    assert pooled["n_subjects"] == 4
    for metric, value in expected.items():
        assert pooled["metrics"][metric] == pytest.approx(value)
    fold_metrics = pd.read_csv(output / "fold_metrics.csv")
    assert fold_metrics["n_test"].tolist() == [1, 3]


def test_require_complete_rejects_missing_fold(tmp_path: Path) -> None:
    _make_fold(tmp_path, fold=0, ids=["sub-1"])
    with pytest.raises(AggregationError, match="missing folds=\\[1\\]"):
        _aggregate(tmp_path, [0, 1])


def test_duplicate_oof_subject_is_rejected(tmp_path: Path) -> None:
    _make_fold(tmp_path, fold=0, ids=["sub-1"])
    _make_fold(tmp_path, fold=1, ids=["sub-1"])
    with pytest.raises(AggregationError, match="more than one test fold"):
        _aggregate(tmp_path, [0, 1])


def test_wrong_fold_membership_is_rejected(tmp_path: Path) -> None:
    _make_fold(tmp_path, fold=0, ids=["sub-1"])
    _make_fold(tmp_path, fold=1, ids=["sub-2"])
    fold_one_predictions = (
        tmp_path / "runs" / "nihss" / "fusion" / "fold_1" / "seed42"
        / "predictions" / "test_predictions.csv"
    )
    dataframe = pd.read_csv(fold_one_predictions)
    dataframe["participant_id"] = ["sub-unexpected"]
    dataframe.to_csv(fold_one_predictions, index=False)
    with pytest.raises(AggregationError, match="Wrong test membership"):
        _aggregate(tmp_path, [0, 1])


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("target_col", "gs_rankin_6isdeath", "target_col"),
        ("model_mode", "image_only", "model_mode"),
        ("seed", 7, "seed"),
        ("final_eval", True, "final_eval=false"),
    ],
)
def test_run_identity_mismatch_is_rejected(
    tmp_path: Path, field: str, wrong_value: object, message: str
) -> None:
    _make_fold(tmp_path, fold=0, ids=["sub-1"])
    _make_fold(tmp_path, fold=1, ids=["sub-2"])
    manifest_path = (
        tmp_path / "runs" / "nihss" / "fusion" / "fold_1" / "seed42" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = wrong_value
    _write_json(manifest_path, manifest)
    with pytest.raises(AggregationError, match=message):
        _aggregate(tmp_path, [0, 1])


def test_saved_metrics_must_match_predictions(tmp_path: Path) -> None:
    _make_fold(tmp_path, fold=0, ids=["sub-1"])
    _make_fold(tmp_path, fold=1, ids=["sub-2"])
    metrics_path = (
        tmp_path / "runs" / "nihss" / "fusion" / "fold_0" / "seed42"
        / "metrics" / "test_metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["mae"] += 1.0
    _write_json(metrics_path, metrics)
    with pytest.raises(AggregationError, match="Saved mae does not match"):
        _aggregate(tmp_path, [0, 1])
