
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm
import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if "object" not in np.__dict__:
    np.object = object  # type: ignore[attr-defined]


CURRENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CURRENT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from common import (  # noqa: E402
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RUNS_DIR,
    TARGET_COLUMNS,
    ensure_dir,
    load_json,
    save_json,
    set_seed,
    utc_now_iso,
)
from metrics import compute_regression_metrics  # noqa: E402
from model import MODEL_MODES, build_regression_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end LC-VIT finetuning from raw 3-view images")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--target-col", type=str, required=True, choices=list(TARGET_COLUMNS))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tcformer-repo", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=EXPERIMENT_DIR / "classification" / "tcformer_light-edacd9e5_20220606.pth",
    )
    parser.add_argument("--model-name", type=str, default="tcformer_light")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--selection-metric", type=str, default="val_mae", choices=["val_mae", "val_mse"])
    parser.add_argument("--model-mode", type=str, default="fusion", choices=list(MODEL_MODES))
    parser.add_argument("--fusion-embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-after-epoch", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def _import_runtime_modules():
    import cv2

    return cv2


def _prepare_tcformer_repo(repo_path: Path | None) -> None:
    if repo_path is None:
        return

    candidates = [repo_path, repo_path / "classification"]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _build_tcformer_backbone(args: argparse.Namespace) -> nn.Module:
    if args.tcformer_repo is not None:
        _prepare_tcformer_repo(args.tcformer_repo)

    import timm

    try:
        import tcformer  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tcformer module is required for end-to-end finetuning. "
            "Pass --tcformer-repo to a local TCFormer checkout."
        ) from exc

    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError("A valid --checkpoint path is required.")

    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = model.state_dict()

    for key in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]

    model.load_state_dict(checkpoint_model, strict=False)
    if hasattr(model, "head"):
        model.head = nn.Identity()
    return model


def _build_normalization_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return mean, std


def _crop_foreground(cv2, image: np.ndarray):
    _, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
    non_zero_columns = np.sum(thresh > 0, axis=0)
    if not (non_zero_columns > 0).any():
        return image

    left_bound = int(np.argmax(non_zero_columns > 0))
    right_bound = int(len(non_zero_columns) - np.argmax(non_zero_columns[::-1] > 0))
    margin = 20
    left_bound = max(0, left_bound - margin)
    right_bound = min(image.shape[1], right_bound + margin)
    return image[:, left_bound:right_bound]


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


@dataclass
class EndToEndBundle:
    dataframe: pd.DataFrame
    tabular_feature_cols: list[str]


def load_end_to_end_bundle(manifest_dir: Path, limit: int | None = None) -> EndToEndBundle:
    manifest = load_json(manifest_dir / "manifest.json")
    dataframe = pd.read_csv(manifest_dir / "all_subjects.csv")
    if limit is not None:
        per_split = max(1, limit // 3)
        split_chunks = []
        for split_name in ("train", "valid", "test"):
            split_df = dataframe.loc[dataframe["split"] == split_name].head(per_split)
            if not split_df.empty:
                split_chunks.append(split_df)
        if split_chunks:
            dataframe = pd.concat(split_chunks, axis=0, ignore_index=True)
            if len(dataframe) > limit:
                dataframe = dataframe.head(limit).copy()
        else:
            dataframe = dataframe.head(limit).copy()
    tabular_feature_cols = list(manifest["tabular_feature_cols"])
    return EndToEndBundle(dataframe=dataframe, tabular_feature_cols=tabular_feature_cols)


def split_dataframe(dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: dataframe.loc[dataframe["split"] == split].copy().reset_index(drop=True)
        for split in ("train", "valid", "test")
    }


def compute_tabular_stats(dataframe: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = dataframe.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


class EndToEndLCVITDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_col: str,
        tabular_feature_cols: list[str],
        tabular_mean: np.ndarray,
        tabular_std: np.ndarray,
        cv2,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.target_col = target_col
        self.tabular_feature_cols = list(tabular_feature_cols)
        self.tabular_mean = tabular_mean.astype(np.float32)
        self.tabular_std = tabular_std.astype(np.float32)
        self.cv2 = cv2
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def _load_view(self, image_path: str) -> torch.Tensor:
        image = self.cv2.imread(image_path, self.cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        cropped = _crop_foreground(self.cv2, image)
        resized = self.cv2.resize(cropped, (224, 224), interpolation=self.cv2.INTER_AREA)
        rgb = np.stack([resized, resized, resized], axis=-1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        tensor = (tensor - self.mean) / self.std
        return tensor

    def _get_tabular(self, row: pd.Series) -> torch.Tensor:
        values = pd.to_numeric(row.loc[self.tabular_feature_cols], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        values = (values - self.tabular_mean[0]) / self.tabular_std[0]
        return torch.from_numpy(values)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        axial = self._load_view(str(row["axial_path"]))
        coronal = self._load_view(str(row["coronal_path"]))
        sagittal = self._load_view(str(row["sagittal_path"]))
        clinical = self._get_tabular(row)
        target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return {
            "participant_id": str(row["participant_id"]),
            "clinical": clinical,
            "axial_img": axial,
            "coronal_img": coronal,
            "sagittal_img": sagittal,
            "target": target,
        }


def build_dataloaders(bundle: EndToEndBundle, args: argparse.Namespace, cv2):
    split_dfs = split_dataframe(bundle.dataframe)
    tabular_mean, tabular_std = compute_tabular_stats(split_dfs["train"], bundle.tabular_feature_cols)
    mean, std = _build_normalization_tensors()

    datasets = {
        split: EndToEndLCVITDataset(
            dataframe=dataframe,
            target_col=args.target_col,
            tabular_feature_cols=bundle.tabular_feature_cols,
            tabular_mean=tabular_mean,
            tabular_std=tabular_std,
            cv2=cv2,
            mean=mean,
            std=std,
        )
        for split, dataframe in split_dfs.items()
    }
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
        )
        for split, dataset in datasets.items()
    }
    return split_dfs, datasets, dataloaders, tabular_mean, tabular_std


class EndToEndRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, model_mode: str):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.model_mode = model_mode

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim > 2:
            out = torch.flatten(out, 1)
        return out

    def forward(self, clinical, axial_img, coronal_img, sagittal_img):
        if self.model_mode == "clinical_only":
            return self.head(clinical)

        axial_feat = self._encode(axial_img)
        coronal_feat = self._encode(coronal_img)
        sagittal_feat = self._encode(sagittal_img)
        if self.model_mode == "image_only":
            return self.head(axial_feat, coronal_feat, sagittal_feat)
        return self.head(clinical, axial_feat, coronal_feat, sagittal_feat)


def _toggle_backbone_grad(backbone: nn.Module, enabled: bool) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = enabled


def _build_optimizer(model: EndToEndRegressor, args: argparse.Namespace):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr})
    if not param_groups:
        raise ValueError("No trainable parameters were found.")

    if args.optimizer == "adam":
        return torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay)
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def _resolve_device(device_arg: str | None) -> torch.device:
    configured_device = device_arg or "auto"
    if configured_device == "auto":
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    if configured_device == "cuda" and not torch.cuda.is_available():
        configured_device = "cpu"
    return torch.device(configured_device)


def _get_prediction(model: EndToEndRegressor, batch: dict, device: torch.device) -> torch.Tensor:
    clinical = batch["clinical"].to(device)
    axial_img = batch["axial_img"].to(device)
    coronal_img = batch["coronal_img"].to(device)
    sagittal_img = batch["sagittal_img"].to(device)
    return model(clinical, axial_img, coronal_img, sagittal_img).squeeze(1)

def run_epoch(model, dataloader, criterion, optimizer, device, epoch: int | None = None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(
        dataloader,
        desc=f"Train" if epoch is None else f"Train Epoch {epoch}",
        leave=True,
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for step, batch in enumerate(progress, start=1):
        target = batch["target"].to(device)

        optimizer.zero_grad()
        prediction = _get_prediction(model, batch, device)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        batch_size = int(target.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        avg_loss = total_loss / max(1, total_samples)
        progress.set_postfix(
            batch_loss=f"{loss.item():.4f}"
        )

    progress.close()
    return total_loss / max(1, total_samples)


def evaluate(model, dataloader, criterion, device, split_name: str = "Eval", epoch: int | None = None):
    model.eval()
    rows: list[dict] = []
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        progress = tqdm(
            dataloader,
            desc=split_name if epoch is None else f"{split_name} Epoch {epoch}",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for step, batch in enumerate(progress, start=1):
            target = batch["target"].to(device)
            prediction = _get_prediction(model, batch, device)
            loss = criterion(prediction, target)

            batch_size = int(target.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            avg_loss = total_loss / max(1, total_samples)

            progress.set_postfix(
                batch_loss=f"{loss.item():.4f}"
            )

            pred_np = prediction.detach().cpu().numpy()
            true_np = target.detach().cpu().numpy()
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

        progress.close()

    y_true_np = np.concatenate(y_true, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    metrics = compute_regression_metrics(y_true_np, y_pred_np)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics, rows

def save_predictions(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _infer_image_embed_dim(model: EndToEndRegressor, dataloader: DataLoader, device: torch.device) -> int:
    batch = next(iter(dataloader))
    with torch.no_grad():
        axial_img = batch["axial_img"].to(device)
        feats = model._encode(axial_img)
    return int(feats.shape[1])


def _run_dry_run(model: EndToEndRegressor, dataloaders: dict[str, DataLoader], device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        for split_name, loader in dataloaders.items():
            batch = next(iter(loader))
            prediction = _get_prediction(model, batch, device)
            print(f"dry_run split={split_name} batch={prediction.shape[0]} prediction_shape={tuple(prediction.shape)}")


def _build_wandb_payload(prefix: str, metrics: Dict[str, float], samples: int, iteration: Optional[int] = None) -> Dict[str, float]:
    payload = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
    payload[f"{prefix}/samples"] = float(samples)
    if iteration is not None:
        payload[f"{prefix}/iteration"] = float(iteration)
    return payload


def _build_wandb_run(args: argparse.Namespace, output_dir: Path):
    if not args.wandb_enable or args.wandb_mode == "disabled":
        return None

    try:
        wandb = importlib.import_module("wandb")
    except Exception as exc:
        raise ImportError("wandb is required for experiment tracking. Install with `pip install wandb`.") from exc
    
    config_payload = {
        "model_name": args.model_name,
        "model_mode": args.model_mode,
        "target_col": args.target_col,
        "manifest_dir": str(args.manifest_dir),
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "selection_metric": args.selection_metric,
        "optimizer": args.optimizer,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "freeze_backbone": bool(args.freeze_backbone),
        "unfreeze_after_epoch": args.unfreeze_after_epoch,
        "seed": args.seed,
    }

    run = wandb.init(
        project=args.wandb_project or "LC-VIT-stroke-outcome-prediction",
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        mode=args.wandb_mode,
        config=config_payload,
    )

    return run


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (DEFAULT_RUNS_DIR / f"finetune_{args.target_col}_{args.model_mode}")
    output_dir = ensure_dir(output_dir)
    logger = setup_logger(output_dir)
    wandb_run = _build_wandb_run(args, output_dir)
    set_seed(args.seed)
    device = _resolve_device(args.device)
    logger.info("device=%s seed=%s model_mode=%s", device, args.seed, args.model_mode)

    cv2 = _import_runtime_modules()

    bundle = load_end_to_end_bundle(args.manifest_dir, limit=args.limit)
    split_dfs, _, dataloaders, tabular_mean, tabular_std = build_dataloaders(
        bundle=bundle,
        args=args,
        cv2=cv2,
    )
    if len(split_dfs["train"]) == 0:
        raise RuntimeError("Training split is empty after filtering.")

    if wandb_run is not None:
        wandb_run.log(
            {
                "data/train_samples": float(len(split_dfs["train"])),
                "data/valid_samples": float(len(split_dfs["valid"])),
                "data/test_samples": float(len(split_dfs["test"])),
                "data/tabular_dim": float(len(bundle.tabular_feature_cols)),
            }
        )

    backbone = _build_tcformer_backbone(args)
    if args.freeze_backbone:
        _toggle_backbone_grad(backbone, enabled=False)

    backbone = backbone.to(device)

    # Infer feature dimension from one training batch so head dimensions match the active backbone.
    temp_model = EndToEndRegressor(backbone=backbone, head=nn.Identity(), model_mode="image_only").to(device)
    image_embed_dim = _infer_image_embed_dim(temp_model, dataloaders["train"], device)

    head = build_regression_model(
        model_mode=args.model_mode,
        clinical_dim=len(bundle.tabular_feature_cols),
        image_embed_dim=image_embed_dim,
        fusion_embed_dim=args.fusion_embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = EndToEndRegressor(backbone=backbone, head=head, model_mode=args.model_mode).to(device)

    if args.dry_run:
        _run_dry_run(model, dataloaders, device)
        dry_run_manifest_path = output_dir / "dry_run_manifest.json"
        save_json(
            dry_run_manifest_path,
            {
                "created_at": utc_now_iso(),
                "manifest_dir": str(args.manifest_dir),
                "target_col": args.target_col,
                "model_mode": args.model_mode,
                "device": str(device),
                "limit": args.limit,
                "split_counts": {split: int(len(df)) for split, df in split_dfs.items()},
                "tabular_feature_cols": bundle.tabular_feature_cols,
                "image_embed_dim": image_embed_dim,
            },
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "dry_run/enabled": 1.0,
                    "dry_run/train_samples": float(len(split_dfs["train"])),
                    "dry_run/valid_samples": float(len(split_dfs["valid"])),
                    "dry_run/test_samples": float(len(split_dfs["test"])),
                    "dry_run/image_embed_dim": float(image_embed_dim),
                }
            )
            wandb_run.summary["dry_run_manifest"] = str(dry_run_manifest_path)
            wandb_run.finish()
        logger.info("Dry run complete.")
        return

    criterion = nn.MSELoss()
    optimizer = _build_optimizer(model, args)

    best_metric = float("inf")
    best_state = None
    no_improve_count = 0

    for epoch in range(1, args.max_epochs + 1):
        if args.unfreeze_after_epoch is not None and epoch > args.unfreeze_after_epoch:
            if not any(param.requires_grad for param in model.backbone.parameters()):
                _toggle_backbone_grad(model.backbone, enabled=True)
                optimizer = _build_optimizer(model, args)
                logger.info("Backbone unfrozen at epoch=%s", epoch)

        train_loss = run_epoch(model, dataloaders["train"], criterion, optimizer, device, epoch=epoch)
        val_metrics, _ = evaluate(model, dataloaders["valid"], criterion, device, split_name="Valid", epoch=epoch)
        current_metric = float(val_metrics["mae"] if args.selection_metric == "val_mae" else val_metrics["mse"])
        
        tqdm.write(
            "epoch=%s train_loss=%.6f val_mse=%.6f val_rmse=%.6f val_mae=%.6f val_mape=%.6f val_r2=%.6f"
            % (
                epoch,
                train_loss,
                val_metrics["mse"],
                val_metrics["rmse"],
                val_metrics["mae"],
                val_metrics["mape"],
                val_metrics["r2"],
            )
        )
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
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": float(train_loss),
                    "train/epoch": float(epoch),
                    **_build_wandb_payload(
                        prefix="val",
                        metrics=val_metrics,
                        samples=len(split_dfs["valid"]),
                        iteration=epoch,
                    ),
                }
            )

        if current_metric < best_metric:
            best_metric = current_metric
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= args.patience:
            logger.info("Early stopping triggered at epoch=%s", epoch)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    checkpoint_path = checkpoint_dir / "best.ckpt"
    torch.save(
        {
            "model_state_dict": best_state,
            "target_col": args.target_col,
            "model_mode": args.model_mode,
            "tabular_feature_cols": bundle.tabular_feature_cols,
            "tabular_mean": tabular_mean,
            "tabular_std": tabular_std,
            "manifest_dir": str(args.manifest_dir),
            "image_embed_dim": image_embed_dim,
            "config": vars(args),
            "saved_at": utc_now_iso(),
        },
        checkpoint_path,
    )

    model.load_state_dict(best_state)
    val_metrics, val_rows = evaluate(model, dataloaders["valid"], criterion, device)
    test_metrics, test_rows = evaluate(model, dataloaders["test"], criterion, device)

    save_json(output_dir / "metrics" / "val_metrics.json", val_metrics)
    save_json(output_dir / "metrics" / "test_metrics.json", test_metrics)
    save_predictions(output_dir / "predictions" / "valid_predictions.csv", val_rows)
    save_predictions(output_dir / "predictions" / "test_predictions.csv", test_rows)

    run_manifest = {
        "created_at": utc_now_iso(),
        "target_col": args.target_col,
        "model_mode": args.model_mode,
        "output_dir": str(output_dir),
        "manifest_dir": str(args.manifest_dir),
        "selection_metric": args.selection_metric,
        "seed": args.seed,
        "device": str(device),
        "split_counts": {split: int(len(df)) for split, df in split_dfs.items()},
        "tabular_feature_cols": bundle.tabular_feature_cols,
        "image_embed_dim": image_embed_dim,
        "freeze_backbone": bool(args.freeze_backbone),
        "unfreeze_after_epoch": args.unfreeze_after_epoch,
        "best_checkpoint": str(checkpoint_path),
    }
    save_json(output_dir / "manifest.json", run_manifest)

    if wandb_run is not None:
        wandb_run.log(_build_wandb_payload("final/val", val_metrics, len(split_dfs["valid"])))
        wandb_run.log(_build_wandb_payload("final/test", test_metrics, len(split_dfs["test"])))
        wandb_run.summary["best_checkpoint"] = str(checkpoint_path)
        wandb_run.summary["best_selection_metric"] = float(best_metric)
        wandb_run.summary["output_dir"] = str(output_dir)
        wandb_run.finish()

    logger.info("Saved checkpoint=%s", checkpoint_path)
    logger.info("Saved val metrics=%s", output_dir / "metrics" / "val_metrics.json")
    logger.info("Saved test metrics=%s", output_dir / "metrics" / "test_metrics.json")


if __name__ == "__main__":
    main()
