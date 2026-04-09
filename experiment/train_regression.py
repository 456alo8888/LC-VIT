from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import wandb

# try:
#     wandb.login()
# except Exception:
#     os.environ.setdefault("WANDB_MODE", "disabled")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_RUNS_DIR, load_json, load_yaml, save_json, save_yaml, set_seed, utc_now_iso
from dataset import (
    LCVITRegressionDataset,
    compute_tabular_stats,
    debug_shapes,
    load_dataset_bundle,
    split_dataframe,
)
from metrics import compute_regression_metrics
from model import MODEL_MODES, build_regression_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LC-VIT multimodal regression model")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config_regression.yaml")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to merged_manifest.json")
    parser.add_argument("--target-col", type=str, required=True, choices=["gs_rankin_6isdeath", "nihss"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-mode", type=str, default=None, choices=list(MODEL_MODES))
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(str(output_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = output_dir / "logs" / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def apply_overrides(config: dict, args: argparse.Namespace, output_dir: Path) -> dict:
    config = dict(config)
    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("optim", {})
    config.setdefault("train", {})
    config.setdefault("logging", {})

    config["data"]["manifest_path"] = str(args.manifest)
    config["data"]["target_col"] = args.target_col
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.max_epochs is not None:
        config["train"]["max_epochs"] = args.max_epochs
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.model_mode is not None:
        config["model"]["mode"] = args.model_mode

    config["logging"]["output_dir"] = str(output_dir)
    return config


def build_dataloaders(bundle, target_col: str, batch_size: int, num_workers: int):
    split_dfs = split_dataframe(bundle.dataframe)
    tabular_mean, tabular_std = compute_tabular_stats(split_dfs["train"], bundle.tabular_feature_cols)

    datasets = {
        split: LCVITRegressionDataset(
            dataframe=dataframe,
            target_col=target_col,
            tabular_feature_cols=bundle.tabular_feature_cols,
            view_feature_cols=bundle.view_feature_cols,
            tabular_mean=tabular_mean,
            tabular_std=tabular_std,
        )
        for split, dataframe in split_dfs.items()
    }
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
    return split_dfs, datasets, dataloaders, tabular_mean, tabular_std


def forward_batch(model, batch, device, model_mode: str):
    clinical = batch["clinical"].to(device)
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    sagittal = batch["sagittal"].to(device)

    if model_mode == "fusion":
        return model(clinical, axial, coronal, sagittal)
    if model_mode == "image_only":
        return model(axial, coronal, sagittal)
    if model_mode == "clinical_only":
        return model(clinical)
    raise ValueError(f"Unsupported model_mode: {model_mode}")


def run_epoch(model, dataloader, criterion, optimizer, device, model_mode: str):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        target = batch["target"].to(device)

        optimizer.zero_grad()
        prediction = forward_batch(model, batch, device, model_mode).squeeze(1)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        batch_size = int(target.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(1, total_samples)


def evaluate(model, dataloader, criterion, device, model_mode: str):
    model.eval()
    rows: list[dict] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            target = batch["target"].to(device)

            prediction = forward_batch(model, batch, device, model_mode).squeeze(1)
            loss = criterion(prediction, target)

            pred_np = prediction.detach().cpu().numpy()
            true_np = target.detach().cpu().numpy()
            losses.append(float(loss.item()))
            y_true.append(true_np)
            y_pred.append(pred_np)

            for participant_id, gt, pdv in zip(batch["participant_id"], true_np, pred_np):
                rows.append(
                    {
                        "participant_id": str(participant_id),
                        "y_true": float(gt),
                        "y_pred": float(pdv),
                        "abs_error": float(abs(pdv - gt)),
                        "squared_error": float((pdv - gt) ** 2),
                    }
                )

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    metrics = compute_regression_metrics(y_true_np, y_pred_np)
    metrics["loss"] = float(np.mean(losses))
    return metrics, rows


def save_predictions(output_path: Path, rows: list[dict]) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config = apply_overrides(config, args, output_dir=Path(""))
    model_mode = str(config["model"].get("mode", "fusion"))
    if model_mode not in MODEL_MODES:
        raise ValueError(f"Unsupported model.mode: {model_mode}")

    output_dir = args.output_dir or (DEFAULT_RUNS_DIR / f"{args.target_col}_{model_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    config = apply_overrides(config, args, output_dir=output_dir)

    seed = int(config["train"].get("seed", 42))
    set_seed(seed)
    configured_device = args.device or config["train"].get("device", "auto")
    if configured_device == "auto":
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    if configured_device == "cuda" and not torch.cuda.is_available():
        configured_device = "cpu"
    device = torch.device(configured_device)
    logger.info("device=%s seed=%s model_mode=%s", device, seed, model_mode)

    bundle = load_dataset_bundle(args.manifest)
    batch_size = int(config["data"].get("batch_size", 8))
    num_workers = int(config["data"].get("num_workers", 0))
    split_dfs, datasets, dataloaders, tabular_mean, tabular_std = build_dataloaders(
        bundle=bundle,
        target_col=args.target_col,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    debug_payload = {
        split: debug_shapes(
            dataframe=dataframe,
            tabular_cols=bundle.tabular_feature_cols,
            view_feature_cols=bundle.view_feature_cols,
        )
        for split, dataframe in split_dfs.items()
    }
    save_json(output_dir / "debug_shapes.json", debug_payload)

    image_embed_dim = len(bundle.view_feature_cols["Axial"])
    model = build_regression_model(
        model_mode=model_mode,
        clinical_dim=len(bundle.tabular_feature_cols),
        image_embed_dim=image_embed_dim,
        fusion_embed_dim=int(config["model"].get("fusion_embed_dim", image_embed_dim)),
        hidden_dim=int(config["model"].get("hidden_dim", 256)),
        num_heads=int(config["model"].get("num_heads", 4)),
        dropout=float(config["model"].get("dropout", 0.2)),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer_name = str(config["optim"].get("optimizer", "adamw")).lower()
    learning_rate = float(config["optim"].get("learning_rate", 8e-4))
    weight_decay = float(config["optim"].get("weight_decay", 1e-4))
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(config["optim"].get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    max_epochs = int(config["train"].get("max_epochs", 30))
    patience = int(config["train"].get("patience", 30))
    selection_metric = str(config["train"].get("selection_metric", "val_mae"))
    mode = "min"

    best_metric = float("inf")
    best_state = None
    no_improve_count = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, dataloaders["train"], criterion, optimizer, device, model_mode)
        val_metrics, _ = evaluate(model, dataloaders["valid"], criterion, device, model_mode)
        current_metric = float(val_metrics["mae"] if selection_metric == "val_mae" else val_metrics["mse"])
        logger.info(
            "epoch=%s train_loss=%.6f val_mse=%.6f val_rmse=%.6f val_mae=%.6f val_mape=%.6f val_r2=%.6f",
            epoch,
            train_loss,
            val_metrics["mse"],
            val_metrics["rmse"],
            val_metrics["mae"],
            val_metrics["mape"],
            val_metrics["r2"],
        )

        if current_metric < best_metric:
            best_metric = current_metric
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            logger.info("Early stopping triggered at epoch=%s", epoch)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.ckpt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_mode": model_mode,
            "target_col": args.target_col,
            "tabular_feature_cols": bundle.tabular_feature_cols,
            "view_feature_cols": bundle.view_feature_cols,
            "tabular_mean": tabular_mean,
            "tabular_std": tabular_std,
            "config": config,
            "image_embed_dim": image_embed_dim,
            "merged_manifest_path": str(args.manifest),
            "saved_at": utc_now_iso(),
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state)
    val_metrics, val_rows = evaluate(model, dataloaders["valid"], criterion, device, model_mode)
    test_metrics, test_rows = evaluate(model, dataloaders["test"], criterion, device, model_mode)

    save_json(output_dir / "metrics" / "val_metrics.json", val_metrics)
    save_json(output_dir / "metrics" / "test_metrics.json", test_metrics)
    save_predictions(output_dir / "predictions" / "valid_predictions.csv", val_rows)
    save_predictions(output_dir / "predictions" / "test_predictions.csv", test_rows)

    run_manifest = {
        "created_at": utc_now_iso(),
        "target_col": args.target_col,
        "model_mode": model_mode,
        "output_dir": str(output_dir),
        "merged_manifest_path": str(args.manifest),
        "feature_extractor": load_json(args.manifest).get("feature_extractor"),
        "selection_metric": selection_metric,
        "seed": seed,
        "device": str(device),
        "split_counts": {split: int(len(df)) for split, df in split_dfs.items()},
        "tabular_feature_cols": bundle.tabular_feature_cols,
        "view_feature_dims": {view: len(columns) for view, columns in bundle.view_feature_cols.items()},
        "best_checkpoint": str(checkpoint_path),
    }
    save_json(output_dir / "manifest.json", run_manifest)
    save_yaml(output_dir / "config_used.yaml", config)

    logger.info("Saved checkpoint=%s", checkpoint_path)
    logger.info("Saved val metrics=%s", output_dir / "metrics" / "val_metrics.json")
    logger.info("Saved test metrics=%s", output_dir / "metrics" / "test_metrics.json")


if __name__ == "__main__":
    main()
