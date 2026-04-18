from __future__ import annotations

import torch
import torch.nn as nn

MODEL_MODES = ("fusion", "image_only", "clinical_only")


class MutualCrossAttentionModule(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        output_a, attn_weights_a = self.mha(x1, x2, x2)
        output_b, _ = self.mha(x2, x1, x1)
        output = output_a + output_b
        # out1 = self.layer_norm1(x1 + output)
        out1 = self.layer_norm1(output)
        ff_output = self.feed_forward(out1)
        ff_output = self.dropout(ff_output)
        out2 = self.layer_norm2(out1 + ff_output)
        return out2, attn_weights_a


class ClinicalRegressor(nn.Module):
    def __init__(self, clinical_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, clinical_features: torch.Tensor):
        return self.mlp(clinical_features)


class ImageOnlyRegressor(nn.Module):
    def __init__(self, image_embed_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        merged_dim = image_embed_dim * 3
        self.mlp = nn.Sequential(
            nn.Linear(merged_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        axial_features: torch.Tensor,
        coronal_features: torch.Tensor,
        sagittal_features: torch.Tensor,
    ):
        merged_features = torch.cat([axial_features, coronal_features, sagittal_features], dim=1)
        return self.mlp(merged_features)


class ClinicalImageFusionRegressor(nn.Module):
    def __init__(
        self,
        clinical_dim: int,
        image_embed_dim: int,
        fusion_embed_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_embed_dim),
        )
        self.image_projection = (
            nn.Identity()
            if image_embed_dim == fusion_embed_dim
            else nn.Linear(image_embed_dim, fusion_embed_dim)
        )
        self.cross_attn = MutualCrossAttentionModule(
            embed_dim=fusion_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.regressor = nn.Sequential(
            nn.Linear(fusion_embed_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        clinical_features: torch.Tensor,
        axial_features: torch.Tensor,
        coronal_features: torch.Tensor,
        sagittal_features: torch.Tensor,
        return_attention: bool = False,
    ):
        clinical_embed = self.clinical_mlp(clinical_features)
        clinical_embed = clinical_embed.unsqueeze(1).repeat(1, 3, 1)

        image_features = torch.stack(
            [
                self.image_projection(axial_features),
                self.image_projection(coronal_features),
                self.image_projection(sagittal_features),
            ],
            dim=1,
        )
        fused_features, attn_weights = self.cross_attn(clinical_embed, image_features)
        pooled = fused_features.mean(dim=1)
        prediction = self.regressor(pooled)

        if return_attention:
            return prediction, attn_weights
        return prediction


def build_regression_model(
    model_mode: str,
    clinical_dim: int,
    image_embed_dim: int,
    fusion_embed_dim: int = 512,
    hidden_dim: int = 256,
    num_heads: int = 4,
    dropout: float = 0.2,
):
    if clinical_dim <= 0:
        raise ValueError(f"clinical_dim must be > 0, got {clinical_dim}")

    if model_mode == "fusion":
        return ClinicalImageFusionRegressor(
            clinical_dim=clinical_dim,
            image_embed_dim=image_embed_dim,
            fusion_embed_dim=fusion_embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    if model_mode == "image_only":
        return ImageOnlyRegressor(
            image_embed_dim=image_embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_mode == "clinical_only":
        return ClinicalRegressor(
            clinical_dim=clinical_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_mode: {model_mode}")
