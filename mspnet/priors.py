from __future__ import annotations

import math
import warnings

import torch
from torch import nn


class ScalePriorEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 512,
        hidden_dim: int = 128,
        descriptor_mean: torch.Tensor | None = None,
        descriptor_std: torch.Tensor | None = None,
        confidence_logvar_bound: float = 6.0,
    ) -> None:
        super().__init__()
        if not confidence_logvar_bound > 0:
            raise ValueError("confidence_logvar_bound must be positive")
        mean = torch.zeros(6) if descriptor_mean is None else torch.as_tensor(descriptor_mean, dtype=torch.float32)
        std = torch.ones(6) if descriptor_std is None else torch.as_tensor(descriptor_std, dtype=torch.float32)
        if mean.numel() != 6 or std.numel() != 6:
            raise ValueError("Scale descriptor statistics must each contain six values")
        self.register_buffer("descriptor_mean", mean.reshape(6))
        self.register_buffer("descriptor_std", std.reshape(6).clamp_min(1.0e-6))
        if (descriptor_mean is None) != (descriptor_std is None):
            raise ValueError("Provide both descriptor mean and standard deviation")
        self.register_buffer("statistics_fitted", torch.tensor(descriptor_mean is not None))
        self.confidence_half_width = 0.5 * float(confidence_logvar_bound)
        self.projection = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    @staticmethod
    def descriptor(size_stats: torch.Tensor) -> torch.Tensor:
        if size_stats.ndim != 2 or size_stats.shape[1] != 4:
            raise ValueError("size_stats must have shape [batch, 4] = [mu_L, mu_W, logvar_L, logvar_W]")
        mu_l, mu_w, logvar_l, logvar_w = size_stats.unbind(dim=1)
        pred_l = torch.expm1(mu_l).clamp_min(0.0)
        pred_w = torch.expm1(mu_w).clamp_min(0.0)
        confidence_l = torch.exp(-0.5 * logvar_l)
        confidence_w = torch.exp(-0.5 * logvar_w)
        eps = torch.finfo(size_stats.dtype).eps
        return torch.stack(
            (
                torch.log1p(pred_l),
                torch.log1p(pred_w),
                torch.log((pred_l + eps) / (pred_w + eps)),
                torch.log1p(pred_l * pred_w),
                confidence_l,
                confidence_w,
            ),
            dim=1,
        )

    def confidence(self, descriptor: torch.Tensor) -> torch.Tensor:
        upper = math.exp(self.confidence_half_width)
        lower = math.exp(-self.confidence_half_width)
        clamped = descriptor[:, 4:6].clamp(min=lower, max=upper)
        return torch.sqrt((clamped[:, 0] * clamped[:, 1]).clamp_min(0.0))

    def forward(self, size_stats: torch.Tensor) -> torch.Tensor:
        descriptor = self.descriptor(size_stats)
        if not bool(self.statistics_fitted):
            warnings.warn(
                "ScalePriorEncoder descriptor statistics were never fitted; "
                "call set_descriptor_statistics() with training out-of-fold size "
                "predictions, otherwise the descriptor is effectively unnormalised.",
                RuntimeWarning,
                stacklevel=2,
            )
        normalized = (descriptor - self.descriptor_mean) / self.descriptor_std
        confidence = self.confidence(descriptor).unsqueeze(1)
        return confidence * self.projection(normalized)

    @torch.no_grad()
    def set_descriptor_statistics(self, training_size_stats: torch.Tensor) -> None:
        descriptor = self.descriptor(training_size_stats)
        self.descriptor_mean.copy_(descriptor.mean(0))
        self.descriptor_std.copy_(descriptor.std(0, correction=0).clamp_min(1.0e-6))
        self.statistics_fitted.fill_(True)


class PositionPriorEncoder(nn.Module):
    def __init__(self, token_dim: int = 512, hidden_dim: int = 128, harmonics: tuple[int, ...] = (1, 2, 4)) -> None:
        super().__init__()
        self.harmonics = tuple(int(k) for k in harmonics)
        global_dim = 3 + 6 * len(self.harmonics)
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim))
        self.local_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim))
        self.route = nn.Linear(2 * token_dim, 1)
        self.output = nn.Linear(token_dim, token_dim, bias=False)

    def forward(self, lonlat: torch.Tensor, local_position: torch.Tensor) -> torch.Tensor:
        if lonlat.ndim != 2 or lonlat.shape[1] != 2:
            raise ValueError("lonlat must have shape [batch, 2] = [longitude, latitude]")
        if local_position.ndim != 2 or local_position.shape[1] != 3:
            raise ValueError("local_position must have shape [batch, 3] = [x_local, y_local, radial_distance]")
        lon = torch.deg2rad(lonlat[:, 0:1])
        lat = torch.deg2rad(lonlat[:, 1:2])
        sphere = torch.cat((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()), 1)
        components = [sphere]
        for k in self.harmonics:
            components.extend((torch.sin(math.pi * k * sphere), torch.cos(math.pi * k * sphere)))
        global_feature = self.global_mlp(torch.cat(components, dim=1))
        local_feature = self.local_mlp(local_position.float())
        route = torch.sigmoid(self.route(torch.cat((global_feature, local_feature), dim=1)))
        return self.output(route * global_feature + (1.0 - route) * local_feature)


class OSMPriorEncoder(nn.Module):
    def __init__(self, feature_dim: int, token_dim: int = 512, hidden_dim: int = 128, availability_dim: int = 0) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.availability_dim = int(availability_dim)
        input_dim = self.feature_dim + self.availability_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim))
        self.feature_gate = nn.Sequential(nn.Linear(token_dim, token_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, features: torch.Tensor, availability: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"OSM features must have shape [batch, {self.feature_dim}]")
        if self.availability_dim:
            if availability is None:
                availability = features.new_zeros(features.shape[0], self.availability_dim)
            elif availability.ndim == 1:
                availability = availability.unsqueeze(1)
            if availability.shape != (features.shape[0], self.availability_dim):
                raise ValueError(f"availability must have shape [batch, {self.availability_dim}]")
            features = torch.cat((features, availability), dim=1)
        encoded = self.norm(self.encoder(features.float()))
        return self.feature_gate(encoded) * encoded


class NullPrior(nn.Module):
    def __init__(self, token_dim: int = 512) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, token_dim))
        nn.init.normal_(self.token, mean=0.0, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.token.expand(int(batch_size), -1)
