"""Manuscript-aligned frequency-conditioned prior cross-attention.

This revision is not checkpoint-compatible with the September core release.
Historical performance does not validate the revised architecture.
"""
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
    visual_alignment: torch.Tensor | None
    prior_alignment: torch.Tensor | None
    source_names: tuple[str, ...]


class FrequencyDecomposer(nn.Module):
    """Three descriptors and a shared affine projection; parameter-matched controls."""
    bands = ("low", "mid", "high")
    modes = ("frequency", "gap_repeated", "spatial_statistics", "full_spectrum_repeated")

    def __init__(self, channels, token_dim, rho1=0.22, rho2=0.55, query_mode="frequency"):
        super().__init__()
        if not 0 < rho1 < rho2 < 1:
            raise ValueError("Require 0 < rho1 < rho2 < 1")
        if query_mode not in self.modes:
            raise ValueError(f"query_mode must be one of {self.modes}")
        self.rho1, self.rho2, self.query_mode = rho1, rho2, query_mode
        self.projection = nn.Linear(channels, token_dim)

    def masks(self, height, width, device=None):
        y = torch.arange(height, device=device) - height // 2
        x = torch.arange(width, device=device) - width // 2
        r = torch.sqrt(y[:, None].float().square() + x[None, :].float().square())
        r = r / r.max().clamp_min(1.0)
        return torch.stack((r <= self.rho1, (r > self.rho1) & (r <= self.rho2), r > self.rho2))

    def descriptors(self, feature_map):
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B,C,H,W]")
        x = feature_map.float()
        if self.query_mode == "gap_repeated":
            return x.mean((-2, -1)).unsqueeze(1).expand(-1, 3, -1)
        if self.query_mode == "spatial_statistics":
            return torch.stack((x.mean((-2, -1)), x.amax((-2, -1)), x.std((-2, -1), correction=0)), 1)
        spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm="backward"), dim=(-2, -1))
        magnitude = torch.log1p(spectrum.abs())
        if self.query_mode == "full_spectrum_repeated":
            return magnitude.mean((-2, -1)).unsqueeze(1).expand(-1, 3, -1)
        masks = self.masks(*x.shape[-2:], device=x.device)
        counts = masks.sum((-2, -1))
        if bool((counts == 0).any()):
            raise ValueError("Empty frequency mask; use the manuscript's 14x14 map")
        return torch.einsum("bchw,nhw->bnc", magnitude, masks.float()) / counts[None, :, None]

    def forward(self, feature_map):
        return self.projection(self.descriptors(feature_map).to(feature_map.dtype))


class FPCSA(nn.Module):
    source_order = ("scale", "position", "osm", "null")

    def __init__(self, feature_channels=256, token_dim=512, hidden_dim=128,
                 num_heads=4, temperature=0.8, rho1=0.22, rho2=0.55,
                 contrast_dim=128, query_mode="frequency", active_sources=None,
                 compatibility_bias=True, source_gating=True):
        super().__init__()
        if token_dim % num_heads or temperature <= 0:
            raise ValueError("Require divisible head dimensions and positive temperature")
        names = self.source_order if active_sources is None else tuple(active_sources)
        if not names or len(set(names)) != len(names) or any(n not in self.source_order for n in names):
            raise ValueError("active_sources must contain unique known source names")
        self.active_sources = tuple(n for n in self.source_order if n in names)
        self.token_dim, self.num_heads = token_dim, num_heads
        self.head_dim, self.temperature = token_dim // num_heads, temperature
        self.compatibility_bias, self.source_gating = compatibility_bias, source_gating
        self.frequency = FrequencyDecomposer(feature_channels, token_dim, rho1, rho2, query_mode)
        self.query = nn.Linear(token_dim, token_dim)
        self.key, self.value = nn.Linear(token_dim, token_dim, bias=False), nn.Linear(token_dim, token_dim, bias=False)
        self.output = nn.Linear(token_dim, token_dim)
        self.compatibility = nn.Sequential(nn.Linear(4 * token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.source_gate = nn.Sequential(nn.Linear(2 * token_dim, token_dim), nn.Sigmoid())
        self.band_score = nn.Linear(token_dim, 1, bias=False)
        self.film = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2 * token_dim))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.visual_projection = nn.Sequential(nn.Linear(3 * token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, contrast_dim))
        self.prior_projection = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, contrast_dim))

    @staticmethod
    def residual_film(visual, gamma, beta, null_mass):
        return visual + (1 - null_mass) * (gamma * visual + beta)

    def forward(self, feature_map, visual_feature, priors, return_alignment=None):
        tokens = self.frequency(feature_map)
        names = self.active_sources
        if any(n not in priors for n in names):
            raise ValueError(f"Required sources: {names}")
        source = torch.stack([priors[n] for n in names], 1)
        batch, bands, dim = tokens.shape
        if source.shape != (batch, len(names), dim) or visual_feature.shape != (batch, dim):
            raise ValueError("Prior and visual feature dimensions must match token_dim")
        queries = self.query(tokens)
        q = queries.reshape(batch, bands, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(source).reshape(batch, len(names), self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.einsum("bhnd,bhsd->bhns", q, k) / (self.temperature * math.sqrt(self.head_dim))
        band = tokens.unsqueeze(2).expand(-1, -1, len(names), -1)
        prior = source.unsqueeze(1).expand(-1, bands, -1, -1)
        if self.compatibility_bias:
            pair = torch.cat((band, prior, band * prior, (band - prior).abs()), -1)
            logits = logits + self.compatibility(pair).squeeze(-1).unsqueeze(1)
        attention = logits.softmax(-1)
        values = self.value(source).unsqueeze(1).expand(-1, bands, -1, -1)
        gates = self.source_gate(torch.cat((band, prior), -1)) if self.source_gating else torch.ones_like(values)
        # All four sources have learned values; null mass scales residual FiLM.
        values = (values * gates).reshape(batch, bands, len(names), self.num_heads, self.head_dim)
        head_output = torch.einsum("bhns,bnshd->bnhd", attention, values)
        band_prior = self.output(head_output.reshape(batch, bands, dim))
        band_weights = self.band_score(tokens).squeeze(-1).softmax(-1)
        prior_mix = (band_weights.unsqueeze(-1) * band_prior).sum(1)
        attention_mean = attention.mean(1)
        if "null" in names:
            null_mass = (band_weights * attention_mean[:, :, names.index("null")]).sum(1, keepdim=True)
        else:
            null_mass = visual_feature.new_zeros(batch, 1)
        gamma, beta = self.film(prior_mix).chunk(2, -1)
        feature = self.residual_film(visual_feature, gamma, beta, null_mass)
        return_alignment = self.training if return_alignment is None else return_alignment
        visual_alignment = F.normalize(self.visual_projection(queries.reshape(batch, -1)), dim=-1) if return_alignment else None
        prior_alignment = F.normalize(self.prior_projection(prior_mix), dim=-1) if return_alignment else None
        return FPCSAOutput(feature=feature, attention=attention_mean, band_weights=band_weights,
                           routing=band_weights.unsqueeze(-1) * attention_mean, null_mass=null_mass,
                           prior_mix=prior_mix, frequency_tokens=tokens,
                           visual_alignment=visual_alignment, prior_alignment=prior_alignment, source_names=names)
