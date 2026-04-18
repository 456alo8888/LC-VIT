from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import load_json, save_json, utc_now_iso
from dataset import LCVITRegressionDataset, load_dataset_bundle_for_target, split_dataframe
from model import MODEL_MODES, build_regression_model
from train_regression import evaluate, save_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LC-VIT multimodal regression checkpoint")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to merged_manifest.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    target_col = checkpoint["target_col"]

    bundle = load_dataset_bundle_for_target(args.manifest, target_col=target_col)
    split_dfs = split_dataframe(bundle.dataframe)
    split_df = split_dfs[args.split]

    missing_checkpoint_tabular_cols = sorted(
        set(checkpoint["tabular_feature_cols"]) - set(split_df.columns)
    )
    if missing_checkpoint_tabular_cols:
        raise ValueError(
            "Checkpoint/data schema mismatch for target "
            f"'{target_col}'. Missing columns: {missing_checkpoint_tabular_cols}"
        )
    if target_col not in split_df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from selected split dataframe.")

    dataset = LCVITRegressionDataset(
        dataframe=split_df,
        target_col=target_col,
        tabular_feature_cols=checkpoint["tabular_feature_cols"],
        view_feature_cols=checkpoint["view_feature_cols"],
        tabular_mean=checkpoint["tabular_mean"],
        tabular_std=checkpoint["tabular_std"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"
    device = torch.device(resolved_device)
    model_mode = str(checkpoint.get("model_mode") or checkpoint["config"].get("model", {}).get("mode", "fusion"))
    if model_mode not in MODEL_MODES:
        raise ValueError(f"Unsupported model_mode in checkpoint: {model_mode}")

    image_embed_dim = int(checkpoint.get("image_embed_dim", len(checkpoint["view_feature_cols"]["Axial"])))
    config = checkpoint["config"]
    model = build_regression_model(
        model_mode=model_mode,
        clinical_dim=len(checkpoint["tabular_feature_cols"]),
        image_embed_dim=image_embed_dim,
        fusion_embed_dim=int(config["model"].get("fusion_embed_dim", image_embed_dim)),
        hidden_dim=int(config["model"].get("hidden_dim", 256)),
        num_heads=int(config["model"].get("num_heads", 4)),
        dropout=float(config["model"].get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.MSELoss()
    metrics, rows = evaluate(model, dataloader, criterion, device, model_mode)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"{args.split}_predictions.csv"
    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    save_predictions(predictions_path, rows)
    save_json(
        metrics_path,
        {
            "created_at": utc_now_iso(),
            "split": args.split,
            "target_col": target_col,
            "model_mode": model_mode,
            "metrics": metrics,
            "predictions_csv": str(predictions_path),
            "checkpoint": str(args.checkpoint),
        },
    )

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
