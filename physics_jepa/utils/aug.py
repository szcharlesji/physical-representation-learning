"""Sample-level JEPA augmentations.

All augmentations are applied *identically* to context and target so the
JEPA prediction task stays well-posed. The augmenter draws random choices
once per sample and replays them on both tensors.

Tensor shape convention: (C, T, H, W) torch float tensor (one sample).

Default configuration is a no-op, so omitting the `augment` block in the
train config reproduces prior behavior exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class AugmentConfig:
    noise_std: float = 0.0
    channel_dropout_p: float = 0.0
    rotations: List[int] = field(default_factory=list)   # subset of {0,90,180,270}
    reflections: bool = False
    translations_px: int = 0  # wrap-around shift magnitude; 0 disables
    periodic_bcs: bool = False  # only allow translations when both spatial dims are periodic

    @classmethod
    def from_cfg(cls, cfg_block, periodic_bcs: bool = False) -> "AugmentConfig":
        """Build from an OmegaConf DictConfig / dict / None."""
        if cfg_block is None:
            return cls(periodic_bcs=periodic_bcs)
        def _get(k, default):
            if hasattr(cfg_block, "get"):
                v = cfg_block.get(k, default)
            else:
                v = cfg_block.get(k, default) if isinstance(cfg_block, dict) else default
            return v
        rotations = list(_get("rotations", []) or [])
        # sanitize
        rotations = [int(r) for r in rotations if int(r) in (0, 90, 180, 270)]
        return cls(
            noise_std=float(_get("noise_std", 0.0) or 0.0),
            channel_dropout_p=float(_get("channel_dropout_p", 0.0) or 0.0),
            rotations=rotations,
            reflections=bool(_get("reflections", False)),
            translations_px=int(_get("translations_px", 0) or 0),
            periodic_bcs=bool(periodic_bcs),
        )

    def is_noop(self) -> bool:
        return (
            self.noise_std <= 0.0
            and self.channel_dropout_p <= 0.0
            and not self.rotations
            and not self.reflections
            and self.translations_px <= 0
        )


class SampleAugmenter:
    """Draw random augmentation choices once per sample, apply to ctx+tgt.

    Noise is added *independently* to ctx and tgt (they should see different
    draws, like two measurements of the same underlying state). Rotations,
    reflections, translations, and channel-dropout are drawn once and
    applied identically to both tensors so the JEPA pred target is
    self-consistent.
    """

    def __init__(self, cfg: AugmentConfig):
        self.cfg = cfg

    def __call__(self, ctx: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        if c.is_noop():
            # still honor legacy noise_std if set (handled by caller), but here
            # no-op fast path avoids extra allocations.
            return ctx, tgt

        # -- shared geometric transforms --
        if c.rotations:
            k_choices = [r // 90 for r in c.rotations]
            k = int(k_choices[torch.randint(low=0, high=len(k_choices), size=(1,)).item()])
            if k != 0:
                ctx = torch.rot90(ctx, k=k, dims=(-2, -1))
                tgt = torch.rot90(tgt, k=k, dims=(-2, -1))

        if c.reflections:
            if torch.rand(()).item() < 0.5:
                ctx = torch.flip(ctx, dims=(-1,))
                tgt = torch.flip(tgt, dims=(-1,))
            if torch.rand(()).item() < 0.5:
                ctx = torch.flip(ctx, dims=(-2,))
                tgt = torch.flip(tgt, dims=(-2,))

        # Translations use torch.roll (wrap-around). Only physically sensible
        # when both spatial dims are periodic; data.py sets `periodic_bcs`
        # from a per-dataset default, so non-periodic datasets skip the op.
        if c.translations_px > 0 and c.periodic_bcs:
            H, W = ctx.shape[-2], ctx.shape[-1]
            dh = int(torch.randint(low=-c.translations_px, high=c.translations_px + 1, size=(1,)).item())
            dw = int(torch.randint(low=-c.translations_px, high=c.translations_px + 1, size=(1,)).item())
            if dh != 0 or dw != 0:
                ctx = torch.roll(ctx, shifts=(dh, dw), dims=(-2, -1))
                tgt = torch.roll(tgt, shifts=(dh, dw), dims=(-2, -1))

        # -- shared channel dropout --
        if c.channel_dropout_p > 0.0:
            C = ctx.shape[0]
            mask = (torch.rand(C) >= c.channel_dropout_p).to(dtype=ctx.dtype).view(C, 1, 1, 1)
            ctx = ctx * mask
            tgt = tgt * mask

        # -- independent noise on ctx / tgt --
        if c.noise_std > 0.0:
            ctx = ctx + torch.randn_like(ctx) * c.noise_std
            tgt = tgt + torch.randn_like(tgt) * c.noise_std

        return ctx, tgt
