"""MSP-Net classification path."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

from .fpcsa import FPCSA
from .priors import NullPrior, OSMPriorEncoder, PositionPriorEncoder, ScalePriorEncoder


class ResNet18VisualBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(x)
        visual_feature = self.pool(feature_map).flatten(1)
        return feature_map, visual_feature


class MSPNet(nn.Module):
    """Core MSP-Net classifier.

    The Stage-1 size estimator is intentionally external.  ``size_stats`` are
    its frozen predictions ``[mu_L, mu_W, logvar_L, logvar_W]`` generated with
    the outer-fold/internal-cross-fitting protocol described in the paper.
    """

    def __init__(self, osm_dim: int, num_classes: int = 3, osm_availability_dim: int = 0, token_dim: int = 512) -> None:
        super().__init__()
        self.visual = ResNet18VisualBackbone()
        self.scale_encoder = ScalePriorEncoder(token_dim=token_dim)
        self.position_encoder = PositionPriorEncoder(token_dim=token_dim)
        self.osm_encoder = OSMPriorEncoder(osm_dim, token_dim=token_dim, availability_dim=osm_availability_dim)
        self.null_prior = NullPrior(token_dim=token_dim)
        self.fpcsa = FPCSA(feature_channels=512, token_dim=token_dim)
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(
        self,
        sar_input: torch.Tensor,
        size_stats: torch.Tensor,
        lonlat: torch.Tensor,
        local_position: torch.Tensor,
        osm_features: torch.Tensor,
        osm_availability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        feature_map, visual_feature = self.visual(sar_input)
        scale_token = self.scale_encoder(size_stats.detach())
        position_token = self.position_encoder(lonlat, local_position)
        osm_token = self.osm_encoder(osm_features, osm_availability)
        null_token = self.null_prior(sar_input.shape[0])
        fused = self.fpcsa(
            feature_map,
            visual_feature,
            {"scale": scale_token, "position": position_token, "osm": osm_token, "null": null_token},
        )
        return {
            "logits": self.classifier(fused.feature),
            "feature": fused.feature,
            "attention": fused.attention,
            "band_weights": fused.band_weights,
            "routing": fused.routing,
            "null_mass": fused.null_mass,
            "visual_alignment": fused.visual_alignment,
            "prior_alignment": fused.prior_alignment,
            "source_names": fused.source_names,
        }

