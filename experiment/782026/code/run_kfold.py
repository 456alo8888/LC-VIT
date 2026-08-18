#!/usr/bin/env python3
"""Validate and launch the canonical LC-VIT K-fold experiment matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment failure
    raise ImportError("PyYAML is required to run the K-fold launcher.") from exc


SCRIPT_PATH = Path(__file__).resolve()
EXPERIMENT_DIR = SCRIPT_PATH.parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "kfold.yaml"
VALIDATOR = SCRIPT_PATH.parent / "validate_kfold_manifests.py"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def find_repository_root(start: Path) -> Path:
    matches: list[Path] = []
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            matches.append(candidate)
    if matches:
        # LC-VIT is itself a nested Git checkout. Config paths intentionally use
        # the surrounding stroke-outcome workspace as their common root.
        return matches[-1]
    raise RuntimeError(f"Could not locate repository root above {start}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return payload


def resolve_config_path(value: str | Path, repository_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate manifests and launch LC-VIT canonical K-fold runs sequentially."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fold", type=int, action="append", help="Fold filter; repeat to select multiple folds.")
    parser.add_argument("--target", action="append", help="Target filter; repeat to select multiple targets.")
    parser.add_argument("--mode", action="append", help="Model-mode filter; repeat to select multiple modes.")
    parser.add_argument("--device", default="auto", help="Torch device passed to the trainer (default: auto).")
    parser.add_argument("--wandb-enable", action="store_true", help="Enable W&B regardless of config default.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--print-only", action="store_true", help="Print commands after preflight without running them.")
    parser.add_argument("--resume", action="store_true", help="Skip only complete runs with matching identity.")
    parser.add_argument("--dry-run", action="store_true", help="Run the trainer's tensor/forward-pass dry run.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples; use 9 with --dry-run.")
    parser.add_argument("--runs-root", type=Path, default=None, help="Override output root (useful for smoke runs).")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip manifest validation (intended only for launcher unit/debug checks).",
    )
    return parser.parse_args()


def select_values(
    configured: Iterable[Any], requested: list[Any] | None, label: str
) -> list[Any]:
    available = list(configured)
    if requested is None:
        return available
    invalid = sorted(set(requested).difference(available), key=str)
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}; configured values are {available}")
    requested_set = set(requested)
    return [value for value in available if value in requested_set]


def run_preflight(config_path: Path) -> None:
    if not VALIDATOR.is_file():
        raise FileNotFoundError(f"Manifest validator is missing: {VALIDATOR}")
    command = [sys.executable, str(VALIDATOR), "--config", str(config_path)]
    print(f"[preflight] {shlex.join(command)}", file=sys.stderr, flush=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, file=sys.stderr, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def add_option(command: list[str], name: str, value: Any) -> None:
    command.extend((name, str(value)))


def build_command(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    fold: int,
    target: str,
    mode: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    train = config["train"]
    batch_size = args.batch_size if args.batch_size is not None else train["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else train["num_workers"]
    max_epochs = args.max_epochs if args.max_epochs is not None else train["max_epochs"]
    patience = args.patience if args.patience is not None else train["patience"]

    command = [sys.executable, str(paths["trainer"])]
    options = (
        ("--manifest-dir", paths["manifest_root"] / f"fold_{fold}"),
        ("--target-col", target),
        ("--model-mode", mode),
        ("--output-dir", output_dir),
        ("--tcformer-repo", paths["tcformer_repo"]),
        ("--checkpoint", paths["checkpoint"]),
        ("--seed", config["seed"]),
        ("--device", args.device),
        ("--batch-size", batch_size),
        ("--num-workers", num_workers),
        ("--max-epochs", max_epochs),
        ("--patience", patience),
        ("--selection-metric", train["selection_metric"]),
        ("--optimizer", train["optimizer"]),
        ("--head-lr", train["head_lr"]),
        ("--backbone-lr", train["backbone_lr"]),
        ("--weight-decay", train["weight_decay"]),
    )
    for name, value in options:
        add_option(command, name, value)

    if args.dry_run:
        command.append("--dry-run")
    if args.limit is not None:
        add_option(command, "--limit", args.limit)

    wandb = config.get("wandb", {})
    wandb_mode = args.wandb_mode or str(wandb.get("mode", "online"))
    wandb_enabled = bool(args.wandb_enable or wandb.get("enabled", False))
    if wandb_enabled and wandb_mode != "disabled":
        command.extend(("--wandb-enable", "--wandb-mode", wandb_mode))
        add_option(command, "--wandb-project", wandb.get("project", "LC-VIT-stroke-outcome-prediction"))
        if wandb.get("entity"):
            add_option(command, "--wandb-entity", wandb["entity"])
        run_name = f"LCVIT_{target.upper()}_{mode.upper()}_fold_{fold}_seed{config['seed']}"
        add_option(command, "--wandb-run-name", run_name)

    forbidden = {"--final-eval", "--freeze-backbone", "--unfreeze-after-epoch"}
    present_forbidden = forbidden.intersection(command)
    if present_forbidden:  # defensive invariant
        raise AssertionError(f"Forbidden trainer options generated: {sorted(present_forbidden)}")
    return command


def expected_identity(config: dict[str, Any], fold: int, target: str, mode: str) -> dict[str, Any]:
    return {"fold": fold, "target_col": target, "model_mode": mode, "seed": int(config["seed"])}


def run_is_complete(
    output_dir: Path,
    identity: dict[str, Any],
    config_sha256: str,
    expected_argv: list[str],
) -> bool:
    required = (
        output_dir / "checkpoints" / "best.ckpt",
        output_dir / "metrics" / "test_metrics.json",
        output_dir / "predictions" / "test_predictions.csv",
        output_dir / "manifest.json",
        output_dir / "launch_manifest.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        with (output_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            trainer_manifest = json.load(handle)
        with (output_dir / "launch_manifest.json").open("r", encoding="utf-8") as handle:
            launch_manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return False

    trainer_matches = all(
        trainer_manifest.get(key) == value
        for key, value in identity.items()
        if key != "fold"
    ) and trainer_manifest.get("final_eval") is False
    launcher_matches = all(launch_manifest.get(key) == value for key, value in identity.items())
    launcher_matches = launcher_matches and launch_manifest.get("config_sha256") == config_sha256
    launcher_matches = launcher_matches and launch_manifest.get("argv") == expected_argv
    return bool(trainer_matches and launcher_matches and launch_manifest.get("status") == "completed")


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    repository_root = find_repository_root(SCRIPT_PATH.parent)

    required_sections = ("paths", "folds", "targets", "modes", "train", "seed")
    missing_sections = [key for key in required_sections if key not in config]
    if missing_sections:
        raise KeyError(f"Config is missing required keys: {missing_sections}")

    paths = {
        key: resolve_config_path(value, repository_root)
        for key, value in config["paths"].items()
    }
    required_path_keys = ("manifest_root", "runs_root", "trainer", "tcformer_repo", "checkpoint")
    missing_path_keys = [key for key in required_path_keys if key not in paths]
    if missing_path_keys:
        raise KeyError(f"Config paths are missing: {missing_path_keys}")
    for key in ("trainer", "tcformer_repo", "checkpoint"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Configured {key} does not exist: {paths[key]}")

    folds = select_values(config["folds"], args.fold, "fold")
    targets = select_values(config["targets"], args.target, "target")
    modes = select_values(config["modes"], args.mode, "mode")
    if not args.skip_preflight:
        run_preflight(config_path)

    runs_root = (
        args.runs_root.expanduser().resolve()
        if args.runs_root is not None
        else paths["runs_root"]
    )
    config_sha256 = sha256_file(config_path)
    jobs: list[tuple[int, str, str, Path, list[str]]] = []
    for target in targets:
        for mode in modes:
            for fold in folds:
                output_dir = runs_root / target / mode / f"fold_{fold}" / f"seed{config['seed']}"
                command = build_command(
                    config=config,
                    paths=paths,
                    fold=fold,
                    target=target,
                    mode=mode,
                    output_dir=output_dir,
                    args=args,
                )
                jobs.append((fold, target, mode, output_dir, command))

    if args.print_only:
        for _, _, _, _, command in jobs:
            print(shlex.join(command))
        print(f"Generated {len(jobs)} command(s).", file=sys.stderr)
        return 0

    for job_index, (fold, target, mode, output_dir, command) in enumerate(jobs, start=1):
        identity = expected_identity(config, fold, target, mode)
        if args.resume and not args.dry_run and run_is_complete(
            output_dir, identity, config_sha256, command
        ):
            print(f"[{job_index}/{len(jobs)}] skip complete: {output_dir}", flush=True)
            continue
        if output_dir.exists() and not args.resume:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Use --resume to retry/skip safely, "
                "or choose a different --runs-root."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        launch_manifest_path = output_dir / "launch_manifest.json"
        launch_payload: dict[str, Any] = {
            "experiment_id": config.get("experiment_id"),
            **identity,
            "status": "running",
            "config_path": str(config_path),
            "config_sha256": config_sha256,
            "manifest_dir": str(paths["manifest_root"] / f"fold_{fold}"),
            "output_dir": str(output_dir),
            "argv": command,
            "command": shlex.join(command),
            "dry_run": bool(args.dry_run),
            "final_eval": False,
            "started_at": utc_now_iso(),
            "completed_at": None,
            "exit_status": None,
        }
        atomic_write_json(launch_manifest_path, launch_payload)
        print(f"[{job_index}/{len(jobs)}] {shlex.join(command)}", flush=True)
        try:
            result = subprocess.run(command, check=False)
        except BaseException:
            launch_payload.update(status="interrupted", completed_at=utc_now_iso(), exit_status=None)
            atomic_write_json(launch_manifest_path, launch_payload)
            raise
        launch_payload.update(
            status="completed" if result.returncode == 0 else "failed",
            completed_at=utc_now_iso(),
            exit_status=int(result.returncode),
        )
        atomic_write_json(launch_manifest_path, launch_payload)
        if result.returncode != 0:
            return int(result.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
