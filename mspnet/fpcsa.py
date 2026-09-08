"""Frequency-conditioned prior cross-attention (FPCSA)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class FPCSAOutput:
    feature: torch.Tensor
    attention: torch.Tensor
    band_weights: torch.Tensor
    routing: torch.Tensor
    null_mass: torch.Tensor
    prior_mix: torch.Tensor
    frequency_tokens: torch.Tensor
    visual_alignment: torch.Tensor
    prior_alignment: torch.Tensor
    source_names: tuple[str, ...]


class FrequencyDecomposer(nn.Module):
    """Create low-, mid-, and high-frequency query tokens from a feature map."""

    bands = ("low", "mid", "high")

    def __init__(self, channels: int, token_dim: int, rho1: float = 0.22, rho2: float = 0.55) -> None:
        super().__init__()
        if not 0.0 < rho1 < rho2 < 1.0:
            raise ValueError("Require 0 < rho1 < rho2 < 1")
        self.rho1 = float(rho1)
        self.rho2 = float(rho2)
        self.projection = nn.ModuleDict(
            {
                band: nn.Sequential(nn.Linear(channels, token_dim), nn.GELU(), nn.LayerNorm(token_dim))
                for band in self.bands
            }
        )

    @staticmethod
    def _mask_mean(spectrum: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=spectrum.dtype, device=spectrum.device)
        denominator = weight.sum().clamp_min(1.0)
        return (spectrum * weight.view(1, 1, *weight.shape)).sum(dim=(-2, -1)) / denominator

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [batch, channels, height, width]")
        x = feature_map.float()
        _, _, height, width = x.shape
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
        magnitude = torch.log1p(spectrum.abs())
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=x.device)).view(height, 1)
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=x.device)).view(1, width)
        radius = torch.sqrt(fx.square() + fy.square()) / math.sqrt(0.5**2 + 0.5**2)
        masks = (
            radius <= self.rho1,
            (radius > self.rho1) & (radius <= self.rho2),
            radius > self.rho2,
        )
        descriptors = []
        for band, mask in zip(self.bands, masks):
            if not bool(mask.any()):
                mask = radius == radius.min() if band == "low" else radius == radius.max()
            descriptors.append(self.projection[band](self._mask_mean(magnitude, mask).to(feature_map.dtype)))
        return torch.stack(descriptors, dim=1)


class FPCSA(nn.Module):
    """Route frequency queries over scale, position, OSM, and null priors."""

    source_order = ("scale", "position", "osm", "null")

    def __init__(
        self,
        feature_channels: int = 512,
        token_dim: int = 512,
        hidden_dim: int = 128,
        num_heads: int = 4,
        temperature: float = 0.8,
        rho1: float = 0.22,
        rho2: float = 0.55,
        modulation_scale: float = 0.1,
        contrast_dim: int = 128,
    ) -> None:
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.temperature = float(temperature)
        self.modulation_scale = float(modulation_scale)
        self.frequency = FrequencyDecomposer(feature_channels, token_dim, rho1=rho1, rho2=rho2)
        self.query = nn.Linear(token_dim, token_dim, bias=False)
        self.key = nn.Linear(token_dim, token_dim, bias=False)
        self.value = nn.Linear(token_dim, token_dim, bias=False)
        self.output = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        pair_dim = 4 * token_dim
        self.compatibility = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.source_gate = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim), nn.Sigmoid())
        self.band_score = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.channel_gate = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim), nn.Sigmoid())
        self.beta = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.visual_projection = nn.Sequential(nn.Linear(3 * token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, contrast_dim))
        self.prior_projection = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, contrast_dim))

    def forward(self, feature_map: torch.Tensor, visual_feature: torch.Tensor, priors: dict[str, torch.Tensor]) -> FPCSAOutput:
        frequency_tokens = self.frequency(feature_map)
        names = tuple(name for name in self.source_order if name in priors)
        if names != self.source_order:
            raise ValueError(f"FPCSA requires priors {self.source_order}, got {names}")
        prior_tokens = torch.stack([priors[name] for name in names], dim=1)
        if prior_tokens.shape[-1] != self.token_dim:
            raise ValueError("All prior tokens must have the configured token dimension")
        batch, bands, _ = frequency_tokens.shape
        sources = prior_tokens.shape[1]

        q = self.query(frequency_tokens).view(batch, bands, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(prior_tokens).view(batch, sources, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhnd,bhsd->bhns", q, k) / (self.temperature * math.sqrt(self.head_dim))

        band = frequency_tokens.unsqueeze(2).expand(-1, -1, sources, -1)
        prior = prior_tokens.unsqueeze(1).expand(-1, bands, -1, -1)
        pair = torch.cat((band, prior, band * prior, (band - prior).abs()), dim=-1)
        logits = logits + self.compatibility(pair).squeeze(-1).unsqueeze(1)
        attention = torch.softmax(logits, dim=-1)
        attention_mean = attention.mean(dim=1)

        values = self.value(prior_tokens).unsqueeze(1).expand(-1, bands, -1, -1)
        gates = self.source_gate(pair)
        null_index = names.index("null")
        values = values * gates
        values[:, :, null_index, :] = 0.0
        band_prior = torch.einsum("bns,bnsd->bnd", attention_mean, values)
        band_prior = self.output(band_prior)
        band_weights = torch.softmax(self.band_score(frequency_tokens).squeeze(-1), dim=1)
        prior_mix = (band_weights.unsqueeze(-1) * band_prior).sum(dim=1)

        null_mass = (band_weights * attention_mean[:, :, null_index]).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        keep = 1.0 - null_mass
        gate = self.channel_gate(prior_mix)
        beta = self.beta(prior_mix)
        modulated = self.norm((1.0 + keep * self.modulation_scale * gate) * visual_feature + keep * self.modulation_scale * beta)
        output_feature = keep * modulated + null_mass * visual_feature

        routing = band_weights.unsqueeze(-1) * attention_mean
        visual_alignment = F.normalize(self.visual_projection(frequency_tokens.reshape(batch, -1)), dim=-1)
        prior_alignment = F.normalize(self.prior_projection(prior_mix), dim=-1)
        return FPCSAOutput(
            feature=output_feature,
            attention=attention_mean,
            band_weights=band_weights,
            routing=routing,
            null_mass=null_mass,
            prior_mix=prior_mix,
            frequency_tokens=frequency_tokens,
            visual_alignment=visual_alignment,
            prior_alignment=prior_alignment,
            source_names=names,
        )

