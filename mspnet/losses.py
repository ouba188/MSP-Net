from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F


def heteroscedastic_size_loss(
    size_stats: torch.Tensor,
    length_m: torch.Tensor,
    width_m: torch.Tensor,
) -> torch.Tensor:
    if size_stats.ndim != 2 or size_stats.shape[1] != 4:
        raise ValueError("size_stats must have shape [batch, 4]")
    target = torch.stack((torch.log1p(length_m.clamp_min(0.0)), torch.log1p(width_m.clamp_min(0.0))), dim=1)
    mean = size_stats[:, :2]
    logvar = size_stats[:, 2:]
    return 0.5 * (torch.exp(-logvar) * (target - mean).square() + logvar).sum(dim=1).mean()


def frequency_vca_loss(visual_alignment: torch.Tensor, prior_alignment: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    if visual_alignment.shape != prior_alignment.shape:
        raise ValueError("visual and prior alignment tensors must have the same shape")
    batch = visual_alignment.shape[0]
    if batch < 2:
        warnings.warn(
            "frequency_vca_loss needs at least two samples per batch; returning 0. "
            "Merge a singleton tail batch into the preceding training batch.",
            RuntimeWarning,
            stacklevel=2,
        )
        return visual_alignment.new_zeros(())
    visual = F.normalize(visual_alignment, dim=-1)
    prior = F.normalize(prior_alignment, dim=-1)
    logits = visual @ prior.t() / float(temperature)
    target = torch.arange(batch, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def mspnet_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    visual_alignment: torch.Tensor | None,
    prior_alignment: torch.Tensor | None,
    lambda_v: float = 5e-4,
    temperature: float = 0.5,
    label_smoothing: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    classification = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
    if lambda_v == 0:
        contrastive = logits.new_zeros(())
    else:
        if visual_alignment is None or prior_alignment is None:
            raise ValueError("The joint loss requires alignment embeddings; use return_alignment=True or lambda_v=0")
        contrastive = frequency_vca_loss(visual_alignment, prior_alignment, temperature=temperature)
    total = classification + float(lambda_v) * contrastive
    return total, {"classification": classification.detach(), "freq_vca": contrastive.detach()}
