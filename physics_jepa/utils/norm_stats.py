"""Per-channel normalization stats with disk caching.

`mode=none` is the default and a strict no-op (behavior-preserving for
existing YAMLs). `mode=per_channel_zscore` computes (mean, std) once per
(dataset, resolution, resize_mode, num_frames) on the training split and
caches to disk.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch


@dataclass
class NormStats:
    mode: str            # "none" | "per_channel_zscore"
    mean: Optional[torch.Tensor]   # (C,) float32 or None
    std: Optional[torch.Tensor]    # (C,) float32 or None

    def is_noop(self) -> bool:
        return self.mode == "none" or self.mean is None or self.std is None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply to (C, T, H, W) tensor. No-op when mode=none."""
        if self.is_noop():
            return x
        C = x.shape[0]
        m = self.mean.to(dtype=x.dtype, device=x.device).view(C, 1, 1, 1)
        s = self.std.to(dtype=x.dtype, device=x.device).clamp_min(1e-8).view(C, 1, 1, 1)
        return (x - m) / s


def cache_key(dataset_name: str, resolution, resize_mode: str, num_frames: int) -> str:
    """Stable 16-char hex hash for the cache filename."""
    parts = [
        str(dataset_name),
        str(tuple(resolution) if resolution is not None else None),
        str(resize_mode),
        str(int(num_frames)),
    ]
    s = "|".join(parts)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def compute_per_channel_stats(dataset, max_samples: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Streaming (mean, std) computation using Welford's algorithm.

    Takes up to `max_samples` items from `dataset`. Each item is expected to
    yield a dict with a 'context' tensor of shape (C, T, H, W).
    """
    n = 0
    mean = None
    m2 = None

    total = min(len(dataset), int(max_samples))
    for i in range(total):
        item = dataset[i]
        x = item["context"] if isinstance(item, dict) else item[0]
        # x: (C, T, H, W) -> per-sample, per-channel stats over (T,H,W)
        C = x.shape[0]
        x = x.float().reshape(C, -1)
        batch_n = x.shape[1]
        batch_mean = x.mean(dim=1)
        batch_var = x.var(dim=1, unbiased=False)
        if mean is None:
            mean = batch_mean.clone()
            m2 = batch_var * batch_n
            n = batch_n
            continue
        delta = batch_mean - mean
        new_n = n + batch_n
        mean = mean + delta * (batch_n / new_n)
        m2 = m2 + batch_var * batch_n + (delta ** 2) * (n * batch_n / new_n)
        n = new_n

    if n == 0 or mean is None:
        raise RuntimeError("compute_per_channel_stats: no samples seen")
    var = m2 / n
    std = var.clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


def load_or_compute_stats(
    dataset,
    dataset_name: str,
    resolution,
    resize_mode: str,
    num_frames: int,
    cache_dir: str | Path,
    max_samples: int = 256,
    rank: int = 0,
) -> NormStats:
    """Return NormStats in per_channel_zscore mode, using disk cache if present.

    Stats are computed once (rank 0 only); other ranks block on file presence.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(dataset_name, resolution, resize_mode, num_frames)
    path = cache_dir / f"norm_stats_{dataset_name}_{key}.pt"

    if path.exists():
        blob = torch.load(path, map_location="cpu")
        return NormStats(mode="per_channel_zscore", mean=blob["mean"], std=blob["std"])

    if rank != 0:
        # non-rank-0: defer and re-try a few times
        import time
        for _ in range(600):
            if path.exists():
                blob = torch.load(path, map_location="cpu")
                return NormStats(mode="per_channel_zscore", mean=blob["mean"], std=blob["std"])
            time.sleep(1)
        raise RuntimeError(f"rank {rank}: norm stats not produced at {path}")

    mean, std = compute_per_channel_stats(dataset, max_samples=max_samples)
    torch.save({"mean": mean, "std": std, "max_samples": int(max_samples)}, path)
    print(f"[norm_stats] computed + cached to {path}: mean={mean.tolist()} std={std.tolist()}", flush=True)
    return NormStats(mode="per_channel_zscore", mean=mean, std=std)


def build_norm_stats(
    mode: str,
    dataset_factory,
    dataset_name: str,
    resolution,
    resize_mode: str,
    num_frames: int,
    cache_dir: str | Path,
    max_samples: int = 256,
    rank: int = 0,
) -> NormStats:
    """Entry point used by data loaders.

    `dataset_factory()` must return an un-normalized training dataset
    (used only to compute the stats; we don't want normalization to be
    applied during stat collection).
    """
    if mode in (None, "none"):
        return NormStats(mode="none", mean=None, std=None)
    if mode == "per_channel_zscore":
        ds = dataset_factory()
        return load_or_compute_stats(
            ds,
            dataset_name=dataset_name,
            resolution=resolution,
            resize_mode=resize_mode,
            num_frames=num_frames,
            cache_dir=cache_dir,
            max_samples=max_samples,
            rank=rank,
        )
    raise ValueError(f"unknown dataset.normalize: {mode!r}; expected none|per_channel_zscore")
