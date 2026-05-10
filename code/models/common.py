from __future__ import annotations

import torch


def to_channel_first(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor [B, C, L] or [B, L, C], got shape {tuple(x.shape)}")

    if x.shape[1] == 2:
        return x
    if x.shape[2] == 2:
        return x.transpose(1, 2)

    raise ValueError(f"Could not infer channel dimension for shape {tuple(x.shape)}")
