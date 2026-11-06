from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected a 2-D, 3-D, or 4-D tensor, got shape {tuple(x.shape)}")


def build_three_channel(
    vv: torch.Tensor,
    vh: torch.Tensor,
    output_size: int | tuple[int, int] | None = None,
) -> torch.Tensor:
    vv = _as_bchw(vv).float()
    vh = _as_bchw(vh).float()
    if vv.shape[1] != 1 or vh.shape[1] != 1:
        raise ValueError("Each polarization must have one channel")
    if vv.shape != vh.shape:
        raise ValueError(f"VV and VH shapes must match, got {vv.shape} and {vh.shape}")

    vv_log = torch.log1p(vv.clamp_min(0.0))
    vh_log = torch.log1p(vh.clamp_min(0.0))
    x = torch.cat((vv_log, vh_log, vh_log - vv_log), dim=1)

    if output_size is not None:
        size = (output_size, output_size) if isinstance(output_size, int) else output_size
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x


def load_png_pair(
    vv_path: str | Path,
    vh_path: str | Path,
    output_size: int | tuple[int, int] | None = 224,
    normalize_uint8: bool = True,
) -> torch.Tensor:
    from PIL import Image

    def read(path: str | Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)

        tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
        if normalize_uint8:
            tensor = tensor / 255.0
        return tensor

    return build_three_channel(read(vv_path), read(vh_path), output_size=output_size)
