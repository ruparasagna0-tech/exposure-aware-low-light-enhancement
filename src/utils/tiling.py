"""Tiled full-resolution inference with linear blending (for very large images, e.g. SICE).

Approved setting: tile 512, overlap 64, linear blending.  Tiles are placed with
stride tile - overlap; the last tile in each direction is aligned to the image
border.  Each tile's weight ramps linearly over ``overlap`` pixels on every edge
that borders another tile (weights (i+1)/(overlap+1)), and is 1 elsewhere (also at
image borders).  The output is sum(w * tile_out) / sum(w), accumulated on CPU,
so memory on the compute device is bounded by one tile.
Images no larger than one tile are processed in a single forward pass.
"""

from typing import Callable, List

import torch


def tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ramp(size: int, overlap: int, left: bool, right: bool) -> torch.Tensor:
    w = torch.ones(size)
    if overlap > 0:
        r = torch.arange(1, overlap + 1, dtype=torch.float32) / (overlap + 1)
        if left:
            w[:overlap] = torch.minimum(w[:overlap], r)
        if right:
            w[-overlap:] = torch.minimum(w[-overlap:], r.flip(0))
    return w


@torch.no_grad()
def tiled_forward(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, tile: int = 512,
                  overlap: int = 64, device: torch.device = None) -> torch.Tensor:
    """Apply ``fn`` (e.g. a model) to [1, C, H, W] ``x`` tile by tile; returns [1, C_out, H, W] on CPU."""
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError(f"expected [1, C, H, W], got {tuple(x.shape)}")
    if not 0 <= overlap < tile:
        raise ValueError(f"overlap {overlap} must be in [0, tile)")
    device = device or x.device
    _, _, H, W = x.shape
    if H <= tile and W <= tile:
        return fn(x.to(device)).float().cpu()
    ys, xs = tile_starts(H, tile, overlap), tile_starts(W, tile, overlap)
    out, wsum = None, torch.zeros(1, 1, H, W)
    for iy, y0 in enumerate(ys):
        th = min(tile, H)
        wy = _ramp(th, overlap, left=iy > 0, right=iy < len(ys) - 1)
        for ix, x0 in enumerate(xs):
            tw = min(tile, W)
            wx = _ramp(tw, overlap, left=ix > 0, right=ix < len(xs) - 1)
            pred = fn(x[:, :, y0:y0 + th, x0:x0 + tw].to(device)).float().cpu()
            if out is None:
                out = torch.zeros(1, pred.shape[1], H, W)
            w = (wy[:, None] * wx[None, :])[None, None]
            out[:, :, y0:y0 + th, x0:x0 + tw] += pred * w
            wsum[:, :, y0:y0 + th, x0:x0 + tw] += w
    return out / wsum
