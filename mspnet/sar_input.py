"""SAR input construction used by the classification and size branches."""

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
    """Build ``[log(1+VV), log(1+VH), log(1+VH)-log(1+VV)]``.

    The inputs may be ``H x W``, ``B x H x W``, or ``B x 1 x H x W`` tensors.
    Values are treated as non-negative linear-intensity images.  Resizing is
    optional and is applied after the three channels have been constructed.
    """

    vv = _as_bchw(vv).float()
    vh = _as_bchw(vh).float()
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
    """Load a VV/VH PNG pair and return a single ``1 x 3 x H x W`` tensor.

    PNG chips in the companion SAR-AIS release are 8-bit exports.  When
    ``normalize_uint8`` is true, values above one are divided by 255 before
    log compression.  Match this convention to the preprocessing used for a
    particular experiment when loading other SAR products.
    """

    from PIL import Image

    def read(path: str | Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
        tensor = torch.from_numpy(array)
        if normalize_uint8 and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor

    return build_three_channel(read(vv_path), read(vh_path), output_size=output_size)

