from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


class SARSizeEstimator(nn.Module):
    def __init__(self, logvar_bounds: tuple[float, float] = (-12.0, 12.0)) -> None:
        super().__init__()
        if not logvar_bounds[0] < logvar_bounds[1]:
            raise ValueError("Require increasing log-variance bounds")
        self.network = resnet18(weights=None)
        self.network.fc = nn.Linear(self.network.fc.in_features, 4)
        self.logvar_bounds = logvar_bounds

    def forward(self, sar_input: torch.Tensor) -> torch.Tensor:
        raw = self.network(sar_input)
        means = F.softplus(raw[:, :2])
        logvars = raw[:, 2:].clamp(*self.logvar_bounds)
        return torch.cat((means, logvars), 1)
