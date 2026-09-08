"""Small CPU smoke test for the released core implementation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mspnet import MSPNet, build_three_channel, mspnet_loss


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    batch = 2
    raw_vv = torch.rand(batch, 64, 64)
    raw_vh = torch.rand(batch, 64, 64)
    sar = build_three_channel(raw_vv, raw_vh, output_size=224)
    model = MSPNet(osm_dim=12, num_classes=3)
    output = model(
        sar,
        size_stats=torch.tensor([[5.0, 3.0, -1.0, -1.0], [4.5, 3.2, -0.5, -0.8]]),
        lonlat=torch.tensor([[129.64, 35.33], [121.50, 31.24]]),
        local_position=torch.randn(batch, 3),
        osm_features=torch.randn(batch, 12),
    )
    attention = output["attention"]
    assert output["logits"].shape == (batch, 3)
    assert attention.shape == (batch, 3, 4)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(batch, 3), atol=1e-5)
    assert torch.isfinite(output["logits"]).all()
    labels = torch.tensor([0, 2])
    total, parts = mspnet_loss(output["logits"], labels, output["visual_alignment"], output["prior_alignment"])
    total.backward()
    assert torch.isfinite(total)
    print("MSP-Net core smoke test passed")
    print(f"logits={tuple(output['logits'].shape)}, routing={tuple(output['routing'].shape)}, loss={float(total.detach()):.4f}")
    print({name: float(value) for name, value in parts.items()})


if __name__ == "__main__":
    main()
