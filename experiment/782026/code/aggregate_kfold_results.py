#!/usr/bin/env python3
"""Aggregate LC-VIT K-fold test artifacts into fold and pooled OOF reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from metrics import compute_regression_metrics  # noqa: E402


REPORT_METRICS = ("mse", "rmse", "mae", "mape", "r2")
PREDICTION_COLUMNS = ("participant_id", "y_true", "y_pred")
METRIC_RTOL = 1e-6
METRIC_ATOL = 1e-8


class AggregationError(RuntimeError):
    """Raised when run artifacts violate the K-fold experiment contract."""


def _repository_root(start: Path) -> Path:
    candidates = (start, *start.parents)
    # This experiment lives in the LC-VIT Git submodule, while configured paths
    # are relative to the enclosing stroke-outcome repository. Prefer its stable
    # workspace markers before falling back to the nearest Git boundary.
    for candidate in candidates:
        if (candidate / "code" / "datasets").is_dir() and (
            candidate / "code" / "baseline_encoder"
        ).is_dir():
            return candidate
    for candidate in candidates:
        if (candidate / ".git").exists():
            return candidate
    return start.parent


def _resolve_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_config(config_path: Path) -> tuple[dict[str, Any], Path]:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise AggregationError(f"Config must contain a YAML mapping: {config_path}")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise AggregationError("Config is missing the 'paths' mapping.")
    for required in ("manifest_root", "runs_root", "aggregate_root"):
        if required not in paths:
            raise AggregationError(f"Config is missing paths.{required}.")

    repo_root = _repository_root(config_path.parent)
    return config, repo_root


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise AggregationError(f"Missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AggregationError(f"Invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AggregationError(f"Expected a JSON object: {path}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    _atomic_write_text(path, dataframe.to_csv(index=False))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_unique_strings(values: Iterable[Any], *, label: str) -> list[str]:
    result = [str(value) for value in values]
    duplicates = pd.Series(result, dtype="string").duplicated(keep=False)
    if bool(duplicates.any()):
        duplicate_ids = sorted(pd.Series(result, dtype="string")[duplicates].unique().tolist())
        raise AggregationError(f"Duplicate participant_id in {label}: {duplicate_ids[:10]}")
    return result


def _load_test_membership(manifest_root: Path, fold: int) -> set[str]:
    fold_dir = manifest_root / f"fold_{fold}"
    test_path = fold_dir / "test.csv"
    if test_path.exists():
        dataframe = pd.read_csv(test_path, dtype={"participant_id": "string"})
    else:
        all_path = fold_dir / "all_subjects.csv"
        try:
            dataframe = pd.read_csv(all_path, dtype={"participant_id": "string"})
        except FileNotFoundError as exc:
            raise AggregationError(
                f"Missing fold membership; expected {test_path} or {all_path}"
            ) from exc
        if "split" not in dataframe.columns:
            raise AggregationError(f"Missing 'split' column in {all_path}")
        dataframe = dataframe.loc[dataframe["split"].astype(str) == "test"]

    if "participant_id" not in dataframe.columns:
        raise AggregationError(f"Missing 'participant_id' column in fold membership: {fold_dir}")
    ids = _as_unique_strings(dataframe["participant_id"].tolist(), label=str(fold_dir))
    return set(ids)


def _validate_run_manifest(
    manifest: Mapping[str, Any], *, target: str, mode: str, seed: int, fold: int, path: Path
) -> None:
    expected = {"target_col": target, "model_mode": mode, "seed": seed}
    for key, wanted in expected.items():
        actual = manifest.get(key)
        if actual != wanted:
            raise AggregationError(
                f"Run identity mismatch in {path}: {key}={actual!r}, expected {wanted!r}"
            )
    if manifest.get("final_eval") is not False:
        raise AggregationError(f"Run must record final_eval=false: {path}")

    manifest_dir = manifest.get("manifest_dir")
    if manifest_dir is not None:
        name = Path(str(manifest_dir)).name
        if name != f"fold_{fold}":
            raise AggregationError(
                f"Run fold mismatch in {path}: manifest_dir ends with {name!r}, "
                f"expected 'fold_{fold}'"
            )


def _load_predictions(path: Path, *, fold: int) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(path, dtype={"participant_id": "string"})
    except FileNotFoundError as exc:
        raise AggregationError(f"Missing required artifact: {path}") from exc
    missing = [column for column in PREDICTION_COLUMNS if column not in dataframe.columns]
    if missing:
        raise AggregationError(f"Missing prediction columns {missing} in {path}")
    if dataframe.empty:
        raise AggregationError(f"Prediction artifact is empty: {path}")
    if dataframe[list(PREDICTION_COLUMNS)].isna().any().any():
        raise AggregationError(f"Null values in required prediction columns: {path}")

    _as_unique_strings(dataframe["participant_id"].tolist(), label=str(path))
    for column in ("y_true", "y_pred"):
        dataframe[column] = pd.to_numeric(dataframe[column], errors="raise")
        if not np.isfinite(dataframe[column].to_numpy(dtype=np.float64)).all():
            raise AggregationError(f"Non-finite {column} values in {path}")
    dataframe["participant_id"] = dataframe["participant_id"].astype(str)
    dataframe["fold"] = int(fold)
    return dataframe


def _reported_metrics(path: Path) -> dict[str, float]:
    payload = _read_json(path)
    result: dict[str, float] = {}
    for metric in REPORT_METRICS:
        if metric not in payload:
            raise AggregationError(f"Missing metric {metric!r} in {path}")
        try:
            value = float(payload[metric])
        except (TypeError, ValueError) as exc:
            raise AggregationError(f"Metric {metric!r} is not numeric in {path}") from exc
        if not np.isfinite(value):
            raise AggregationError(f"Metric {metric!r} is not finite in {path}")
        result[metric] = value
    return result


def _recomputed_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    computed = compute_regression_metrics(
        predictions["y_true"].to_numpy(dtype=np.float64),
        predictions["y_pred"].to_numpy(dtype=np.float64),
    )
    return {metric: float(computed[metric]) for metric in REPORT_METRICS}


def _assert_metrics_match(
    reported: Mapping[str, float], recomputed: Mapping[str, float], *, source: Path
) -> None:
    for metric in REPORT_METRICS:
        if not np.isclose(
            float(reported[metric]),
            float(recomputed[metric]),
            rtol=METRIC_RTOL,
            atol=METRIC_ATOL,
        ):
            raise AggregationError(
                f"Saved {metric} does not match predictions in {source}: "
                f"saved={reported[metric]}, recomputed={recomputed[metric]}"
            )


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return unweighted mean and sample standard deviation across fold rows."""
    if len(fold_metrics) < 2:
        raise AggregationError("At least two completed folds are required to compute sample std.")
    rows: list[dict[str, float | int | str]] = []
    metrics_json: dict[str, dict[str, float]] = {}
    for metric in REPORT_METRICS:
        values = pd.to_numeric(fold_metrics[metric], errors="raise").to_numpy(dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        rows.append({"metric": metric, "mean": mean, "std": std, "n_folds": len(values)})
        metrics_json[metric] = {"mean": mean, "std": std}
    return pd.DataFrame(rows), {
        "aggregation": "unweighted_across_folds",
        "std_ddof": 1,
        "n_folds": int(len(fold_metrics)),
        "metrics": metrics_json,
        "primary_metrics": ["mae", "rmse", "r2"],
        "secondary_metrics": ["mape"],
    }


def _run_dir(runs_root: Path, target: str, mode: str, fold: int, seed: int) -> Path:
    return runs_root / target / mode / f"fold_{fold}" / f"seed{seed}"


def aggregate_group(
    *,
    target: str,
    mode: str,
    seed: int,
    folds: Sequence[int],
    manifest_root: Path,
    runs_root: Path,
    aggregate_root: Path,
    require_complete: bool,
    config_path: Path | None = None,
) -> Path | None:
    """Validate and aggregate one target/mode group; return its output directory."""
    available: list[tuple[int, Path]] = []
    missing_folds: list[int] = []
    for fold in folds:
        run_dir = _run_dir(runs_root, target, mode, int(fold), seed)
        required = (
            run_dir / "manifest.json",
            run_dir / "metrics" / "test_metrics.json",
            run_dir / "predictions" / "test_predictions.csv",
        )
        if all(path.exists() for path in required):
            available.append((int(fold), run_dir))
        else:
            missing_folds.append(int(fold))

    if missing_folds and require_complete:
        raise AggregationError(
            f"Incomplete runs for target={target}, mode={mode}; missing folds={missing_folds}"
        )
    if not available:
        return None
    if len(available) < 2:
        raise AggregationError(
            f"Need at least two completed folds for target={target}, mode={mode}; "
            f"found folds={[fold for fold, _ in available]}"
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    source_artifacts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for fold, run_dir in available:
        run_manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics" / "test_metrics.json"
        predictions_path = run_dir / "predictions" / "test_predictions.csv"
        run_manifest = _read_json(run_manifest_path)
        _validate_run_manifest(
            run_manifest, target=target, mode=mode, seed=seed, fold=fold, path=run_manifest_path
        )
        predictions = _load_predictions(predictions_path, fold=fold)
        prediction_ids = set(predictions["participant_id"])
        expected_ids = _load_test_membership(manifest_root, fold)
        if prediction_ids != expected_ids:
            missing = sorted(expected_ids - prediction_ids)
            unexpected = sorted(prediction_ids - expected_ids)
            raise AggregationError(
                f"Wrong test membership for fold_{fold}: missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )
        duplicated_across_folds = sorted(seen_ids & prediction_ids)
        if duplicated_across_folds:
            raise AggregationError(
                f"OOF participants appear in more than one test fold: {duplicated_across_folds[:10]}"
            )
        seen_ids.update(prediction_ids)

        reported = _reported_metrics(metrics_path)
        recomputed = _recomputed_metrics(predictions)
        _assert_metrics_match(reported, recomputed, source=predictions_path)
        metric_rows.append({"fold": fold, "n_test": len(predictions), **reported})
        prediction_frames.append(predictions)
        source_artifacts.append(
            {
                "fold": fold,
                "run_dir": str(run_dir),
                "run_manifest_sha256": _sha256(run_manifest_path),
                "test_metrics_sha256": _sha256(metrics_path),
                "test_predictions_sha256": _sha256(predictions_path),
            }
        )

    fold_metrics = pd.DataFrame(metric_rows).sort_values("fold").reset_index(drop=True)
    oof = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["participant_id", "fold"]
    ).reset_index(drop=True)
    if oof["participant_id"].duplicated().any():
        raise AggregationError("OOF predictions contain duplicate participant_id values.")

    summary_csv, summary_json = summarize_fold_metrics(fold_metrics)
    pooled_metrics = compute_regression_metrics(
        oof["y_true"].to_numpy(dtype=np.float64),
        oof["y_pred"].to_numpy(dtype=np.float64),
    )
    pooled_payload = {
        "aggregation": "pooled_out_of_fold",
        "n_subjects": int(len(oof)),
        "metrics": {key: float(value) for key, value in pooled_metrics.items()},
        "primary_metrics": ["mae", "rmse", "r2"],
        "secondary_metrics": ["mape"],
        "mape_note": "Secondary metric because regression targets may be zero.",
    }

    output_dir = aggregate_root / target / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fold_metrics.csv", fold_metrics)
    _write_csv(output_dir / "summary_mean_std.csv", summary_csv)
    _write_json(output_dir / "summary_mean_std.json", summary_json)
    _write_csv(output_dir / "oof_predictions.csv", oof)
    _write_json(output_dir / "oof_metrics.json", pooled_payload)
    _write_json(
        output_dir / "aggregation_manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_col": target,
            "model_mode": mode,
            "seed": seed,
            "requested_folds": [int(fold) for fold in folds],
            "aggregated_folds": [fold for fold, _ in available],
            "missing_folds": missing_folds,
            "require_complete": require_complete,
            "complete": not missing_folds,
            "n_oof_subjects": int(len(oof)),
            "config_path": str(config_path) if config_path else None,
            "config_sha256": _sha256(config_path) if config_path else None,
            "source_artifacts": source_artifacts,
            "outputs": [
                "fold_metrics.csv",
                "summary_mean_std.csv",
                "summary_mean_std.json",
                "oof_predictions.csv",
                "oof_metrics.json",
            ],
        },
    )
    return output_dir


def aggregate_from_config(config_path: Path, *, require_complete: bool) -> list[Path]:
    config, repo_root = load_config(config_path)
    paths = config["paths"]
    manifest_root = _resolve_path(paths["manifest_root"], repo_root=repo_root)
    runs_root = _resolve_path(paths["runs_root"], repo_root=repo_root)
    aggregate_root = _resolve_path(paths["aggregate_root"], repo_root=repo_root)
    try:
        folds = [int(fold) for fold in config["folds"]]
        targets = [str(target) for target in config["targets"]]
        modes = [str(mode) for mode in config["modes"]]
        seed = int(config["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AggregationError("Config must define integer seed and non-empty folds/targets/modes.") from exc
    if not folds or not targets or not modes:
        raise AggregationError("Config folds, targets, and modes must be non-empty.")
    if len(set(folds)) != len(folds):
        raise AggregationError("Config folds must be unique.")

    outputs: list[Path] = []
    for target in targets:
        for mode in modes:
            output = aggregate_group(
                target=target,
                mode=mode,
                seed=seed,
                folds=folds,
                manifest_root=manifest_root,
                runs_root=runs_root,
                aggregate_root=aggregate_root,
                require_complete=require_complete,
                config_path=config_path.expanduser().resolve(),
            )
            if output is not None:
                outputs.append(output)
                print(f"Aggregated target={target} mode={mode}: {output}")
            else:
                print(f"Skipped target={target} mode={mode}: no completed folds")
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to kfold.yaml")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every configured target/mode has all configured folds.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        aggregate_from_config(args.config, require_complete=args.require_complete)
    except AggregationError as exc:
        raise SystemExit(f"Aggregation failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
