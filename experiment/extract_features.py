from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from common import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_MANIFEST_DIR,
    VIEW_NAMES,
    ensure_dir,
    format_feature_columns,
    load_json,
    save_json,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 3-view features for LC-VIT regression")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--views", nargs="+", default=list(VIEW_NAMES), choices=list(VIEW_NAMES))
    parser.add_argument("--extractor", type=str, default="tcformer", choices=["tcformer", "simple_stats"])
    parser.add_argument("--tcformer-repo", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="tcformer_light")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--simple-height", type=int, default=16)
    parser.add_argument("--simple-width", type=int, default=32)
    return parser.parse_args()


def _import_torch_modules():
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    return cv2, np, torch, nn, Image, DataLoader, Dataset, transforms


def _prepare_tcformer_repo(repo_path: Path | None) -> None:
    if repo_path is None:
        return

    candidates = [repo_path, repo_path / "classification"]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _build_tcformer_model(args: argparse.Namespace, torch, nn):
    if args.tcformer_repo is not None:
        _prepare_tcformer_repo(args.tcformer_repo)

    import timm

    try:
        import tcformer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "tcformer module is required for --extractor tcformer. "
            "Pass --tcformer-repo to a local TCFormer checkout."
        ) from exc

    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError(
            "A valid --checkpoint path is required for --extractor tcformer."
        )

    model = timm.create_model(
        args.model_name,
        pretrained=False,
        num_classes=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_model = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = model.state_dict()

    for key in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]

    model.load_state_dict(checkpoint_model, strict=False)
    if hasattr(model, "head"):
        model.head = nn.Identity()
    model.eval()
    return model


def _build_transform(transforms, image_size=(224, 224)):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _crop_foreground(cv2, np, image):
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


class ViewDataset:
    def __init__(self, dataframe, view_name, cv2, np, Image, transform):
        self.df = dataframe.reset_index(drop=True)
        self.view_name = view_name
        self.cv2 = cv2
        self.np = np
        self.Image = Image
        self.transform = transform
        self.column_name = f"{view_name.lower()}_path"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row[self.column_name]
        image = self.cv2.imread(image_path, self.cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        cropped = _crop_foreground(self.cv2, self.np, image)
        resized = self.cv2.resize(cropped, (224, 224), interpolation=self.cv2.INTER_AREA)
        tensor = self.transform(self.Image.fromarray(resized))
        return row["participant_id"], tensor


def _extract_with_simple_stats(data_loader, torch, height: int, width: int):
    participant_ids: list[str] = []
    features: list[list[float]] = []

    for ids, batch in data_loader:
        gray_batch = batch[:, :1]
        pooled = torch.nn.functional.interpolate(
            gray_batch,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        flattened = pooled.flatten(start_dim=1).cpu().numpy().tolist()
        participant_ids.extend([str(value) for value in ids])
        features.extend(flattened)

    return participant_ids, features


def _extract_with_tcformer(data_loader, model, device, torch):
    participant_ids: list[str] = []
    features: list[list[float]] = []
    model = model.to(device)

    with torch.no_grad():
        for ids, batch in data_loader:
            batch = batch.to(device)
            outputs = model(batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            outputs = outputs.detach().cpu()
            participant_ids.extend([str(value) for value in ids])
            features.extend(outputs.numpy().tolist())

    return participant_ids, features


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest_dir / "manifest.json")
    dataframe = pd.read_csv(args.manifest_dir / "all_subjects.csv")
    if args.limit is not None:
        dataframe = dataframe.head(args.limit).copy()

    output_dir = ensure_dir(args.output_dir)
    cv2, np, torch, nn, Image, DataLoader, _, transforms = _import_torch_modules()
    transform = _build_transform(transforms)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = None
    if args.extractor == "tcformer":
        model = _build_tcformer_model(args, torch, nn)

    metadata = {
        "created_at": utc_now_iso(),
        "manifest_dir": str(args.manifest_dir),
        "extractor": args.extractor,
        "views": args.views,
        "limit": args.limit,
        "device": str(device),
        "tcformer_repo": str(args.tcformer_repo) if args.tcformer_repo else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "model_name": args.model_name if args.extractor == "tcformer" else "simple_stats",
        "files": {},
    }

    for view_name in args.views:
        dataset = ViewDataset(dataframe, view_name=view_name, cv2=cv2, np=np, Image=Image, transform=transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        if args.extractor == "tcformer":
            participant_ids, features = _extract_with_tcformer(
                data_loader=data_loader,
                model=model,
                device=device,
                torch=torch,
            )
        else:
            participant_ids, features = _extract_with_simple_stats(
                data_loader=data_loader,
                torch=torch,
                height=args.simple_height,
                width=args.simple_width,
            )

        feature_dim = len(features[0]) if features else args.simple_height * args.simple_width
        feature_columns = format_feature_columns(view_name, feature_dim)
        feature_df = pd.DataFrame(features, columns=feature_columns)
        feature_df.insert(0, "participant_id", participant_ids)

        feature_path = output_dir / f"features_{view_name.lower()}.csv"
        feature_df.to_csv(feature_path, index=False)

        metadata["files"][view_name] = str(feature_path)
        metadata.setdefault("feature_dims", {})[view_name] = int(feature_dim)

    save_json(output_dir / "feature_manifest.json", metadata)
    print(f"Saved feature manifest: {output_dir / 'feature_manifest.json'}")
    print(f"Extractor: {args.extractor}")
    print(f"Subjects processed: {len(dataframe)}")
    print(f"Reference merged subjects in manifest: {manifest['counts']['merged_subjects_total']}")


if __name__ == "__main__":
    main()
