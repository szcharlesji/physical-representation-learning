"""Diagnostic: reproduce the old half-of-val MSE from cached features.

Old eval_frozen.run() did `val_all = extract_features("val")` (full 1200-sample
data/valid/), then sub-sampled to first 600 (seed-42 randperm) for val and
last 600 for test. The new run() extracts data/valid/ (1200) for val and
data/test/ (1300) for test — different splits, no half mixing.

If the old number (e.g. ai3ktxmv val=0.0857 on ViT3DEncoder_5) was just due to
the seed-42 first half of valid being easier, this script will reproduce
~0.0857 from the SAME cached features the new code uses. If instead it
reports the new ~0.155, the half-vs-full subsetting is NOT the explanation
and something else changed (data on disk, feature extraction path, etc.).

Usage on the HPC:
  cd /scratch/$USER/physical-representation-learning
  python scripts/diag_old_val_split.py \
    /scratch/$USER/physical-representation-learning/checkpoints_vit3d/active_matter-16frames-vit3d-jepa-vit3d-depth6_2026-04-25_11-11-48/ViT3DEncoder_5.pth \
    --feature-cache feature_cache --seed 42

  # (optional) point at a different feature_cache_dir if not the default.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def cache_key(
    ckpt_path: str,
    split: str,
    dataset_name: str,
    num_frames: int,
    resolution,
    resize_mode: str,
    feature_pool: str,
    noise_std: float,
) -> str:
    """Mirror physics_jepa.eval_frozen.FrozenEvaluator._cache_key (current)."""
    s = "|".join([
        str(Path(ckpt_path).resolve()),
        split,
        dataset_name,
        str(num_frames),
        str(resolution),
        resize_mode,
        feature_pool,
        f"noise={float(noise_std)}",
    ])
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def lstsq_fit(X_tr: torch.Tensor, Y_tr: torch.Tensor):
    Xb = torch.cat([X_tr, torch.ones(X_tr.size(0), 1, dtype=X_tr.dtype)], dim=1)
    sol = torch.linalg.lstsq(Xb, Y_tr).solution  # (D+1, P)
    W = sol[:-1].T  # (P, D)
    b = sol[-1]    # (P,)
    return W, b


def per_param_mse(pred: torch.Tensor, y: torch.Tensor) -> dict:
    err = (pred - y).pow(2).mean(0)  # (P,)
    return {
        "mse_per_param": err.tolist(),
        "mse_mean": err.mean().item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--feature-cache", default="feature_cache")
    p.add_argument("--dataset", default="active_matter")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--resolution", default="[256, 256]",
                   help="String form as it appears in config (e.g. `[256, 256]` or `224`)")
    p.add_argument("--resize-mode", default="fft")
    p.add_argument("--pool", default="gap")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Resolved noise_std used in the run that wrote the cache")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cache_dir = Path(args.feature_cache)
    train_path = cache_dir / f"{cache_key(args.ckpt, 'train', args.dataset, args.num_frames, args.resolution, args.resize_mode, args.pool, args.noise)}.pt"
    val_path = cache_dir / f"{cache_key(args.ckpt, 'val', args.dataset, args.num_frames, args.resolution, args.resize_mode, args.pool, args.noise)}.pt"

    print(f"train cache: {train_path}  exists={train_path.exists()}")
    print(f"val   cache: {val_path}    exists={val_path.exists()}")
    if not (train_path.exists() and val_path.exists()):
        raise SystemExit("missing caches; check --resolution / --resize-mode / --noise match the run that wrote them")

    train_f = torch.load(train_path)
    val_f = torch.load(val_path)
    print(f"train features={tuple(train_f['features'].shape)} labels={tuple(train_f['labels_raw'].shape)}")
    print(f"val   features={tuple(val_f['features'].shape)} labels={tuple(val_f['labels_raw'].shape)}")

    # z-score using train stats, same as eval_frozen.run()
    mean = train_f["labels_raw"].mean(0)
    std = train_f["labels_raw"].std(0).clamp_min(1e-8)
    Y_tr = (train_f["labels_raw"] - mean) / std
    Y_va = (val_f["labels_raw"] - mean) / std

    W, b = lstsq_fit(train_f["features"], Y_tr)
    pred_full = val_f["features"] @ W.T + b

    # Full val (what new code reports as "val")
    full = per_param_mse(pred_full, Y_va)

    # Old code's val: first half of seed-42 randperm
    N = val_f["features"].shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    half = N // 2
    val_idx = perm[:half]
    test_idx = perm[half:]

    val_half = per_param_mse(pred_full[val_idx], Y_va[val_idx])
    test_half = per_param_mse(pred_full[test_idx], Y_va[test_idx])

    print()
    print("=" * 70)
    print("LINEAR PROBE (lstsq on train, predict on val)")
    print("=" * 70)
    print(f"  full val   (N={N})    mse_mean={full['mse_mean']:.4f}   per_param={[f'{x:.4f}' for x in full['mse_per_param']]}")
    print(f"  old 'val'  (first {half} of seed-{args.seed} randperm)  mse_mean={val_half['mse_mean']:.4f}   per_param={[f'{x:.4f}' for x in val_half['mse_per_param']]}")
    print(f"  old 'test' (last {N-half} of seed-{args.seed} randperm)  mse_mean={test_half['mse_mean']:.4f}   per_param={[f'{x:.4f}' for x in test_half['mse_per_param']]}")
    print("=" * 70)
    print()
    print("If 'old val' here matches the Apr-25 ai3ktxmv val number, the half-vs-full")
    print("subsetting is the entire delta — full valid is genuinely harder than the")
    print("seed-42 first half. If 'old val' here is close to 'full val', something")
    print("else (data on disk, feature extraction) changed and the diagnostic clears it.")


if __name__ == "__main__":
    main()
