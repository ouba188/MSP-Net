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
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.layer2(self.layer1(self.stem(x)))
        frequency_map = self.layer3(x)
        visual_feature = self.pool(self.layer4(frequency_map)).flatten(1)
        return frequency_map, visual_feature


class MSPNet(nn.Module):
    """Core MSP-Net classifier.

    The Stage-1 size estimator is intentionally external.  ``size_stats`` are
    its frozen predictions ``[mu_L, mu_W, logvar_L, logvar_W]`` generated with
    the outer-fold/internal-cross-fitting protocol described in the paper.
    """

    def __init__(self, osm_dim: int, num_classes: int = 3, osm_availability_dim: int = 0, token_dim: int = 512, fusion_kind: str = "fpcsa", **fusion_options) -> None:
        super().__init__()
        if token_dim != 512:
            raise ValueError("The manuscript classification path uses token_dim=512")
        if fusion_kind not in ("fpcsa", "visual", "concat", "film", "vanilla"):
            raise ValueError("Unknown fusion_kind")
        self.fusion_kind = fusion_kind
        self.visual = ResNet18VisualBackbone()
        if fusion_kind == "visual":
            self.classifier = nn.Linear(token_dim, num_classes)
            return
        self.scale_encoder = ScalePriorEncoder(token_dim=token_dim)
        self.position_encoder = PositionPriorEncoder(token_dim=token_dim)
        self.osm_encoder = OSMPriorEncoder(osm_dim, token_dim=token_dim, availability_dim=osm_availability_dim)
        if fusion_kind == "fpcsa":
            self.null_prior = NullPrior(token_dim=token_dim)
            self.fpcsa = FPCSA(feature_channels=256, token_dim=token_dim, **fusion_options)
        elif fusion_kind == "film":
            self.film = nn.Sequential(nn.Linear(3 * token_dim, 128), nn.GELU(), nn.Linear(128, 2 * token_dim))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        elif fusion_kind == "vanilla":
            self.query = nn.Linear(token_dim, token_dim)
            self.key = nn.Linear(token_dim, token_dim, bias=False)
            self.value = nn.Linear(token_dim, token_dim, bias=False)
            self.attention_output = nn.Linear(token_dim, token_dim)
        self.classifier = nn.Linear(4 * token_dim if fusion_kind == "concat" else token_dim, num_classes)

    def forward(
        self,
        sar_input: torch.Tensor,
        size_stats: torch.Tensor,
        lonlat: torch.Tensor,
        local_position: torch.Tensor,
        osm_features: torch.Tensor,
        osm_availability: torch.Tensor | None = None,
        return_alignment: bool | None = None,
    ) -> dict[str, torch.Tensor | tuple[str, ...] | None]:
        feature_map, visual_feature = self.visual(sar_input)
        if self.fusion_kind == "visual":
            return {"logits": self.classifier(visual_feature), "feature": visual_feature,
                    "visual_alignment": None, "prior_alignment": None}
        scale_token = self.scale_encoder(size_stats.detach())
        position_token = self.position_encoder(lonlat, local_position)
        osm_token = self.osm_encoder(osm_features, osm_availability)
        if self.fusion_kind != "fpcsa":
            source = torch.stack((scale_token, position_token, osm_token), 1)
            if self.fusion_kind == "concat":
                feature = torch.cat((visual_feature, source.flatten(1)), 1)
            elif self.fusion_kind == "film":
                gamma, beta = self.film(source.flatten(1)).chunk(2, -1)
                feature = visual_feature + gamma * visual_feature + beta
            else:
                batch = source.shape[0]
                query = self.query(visual_feature).reshape(batch, 4, 1, 128)
                key = self.key(source).reshape(batch, 3, 4, 128).transpose(1, 2)
                value = self.value(source).reshape(batch, 3, 4, 128).transpose(1, 2)
                weights = (query @ key.transpose(-2, -1) / (0.8 * 128**0.5)).softmax(-1)
                attended = (weights @ value).reshape(batch, 512)
                feature = visual_feature + self.attention_output(attended)
            return {"logits": self.classifier(feature), "feature": feature,
                    "visual_alignment": None, "prior_alignment": None}
        null_token = self.null_prior(sar_input.shape[0])
        fused = self.fpcsa(
            feature_map,
            visual_feature,
            {"scale": scale_token, "position": position_token, "osm": osm_token, "null": null_token},
            return_alignment=return_alignment,
        )
        return {
            "logits": self.classifier(fused.feature),
            "feature": fused.feature,
            "attention": fused.attention,
            "band_weights": fused.band_weights,
            "routing": fused.routing,
            "null_mass": fused.null_mass,
            "attention_per_head": fused.attention_per_head,
            "null_mass_per_head": fused.null_mass_per_head,
            "visual_alignment": fused.visual_alignment,
            "prior_alignment": fused.prior_alignment,
            "source_names": fused.source_names,
            "band_names": fused.band_names,
        }

