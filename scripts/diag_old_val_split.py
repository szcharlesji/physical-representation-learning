"""Diagnostic: reproduce the old half-of-val MSE from cached features.

Old eval_frozen.run() did `val_all = extract_features("val")` (full 1200-sample
data/valid/), then sub-sampled to first 600 (seed-42 randperm) for val and
last 600 for test. The new run() extracts data/valid/ (1200) for val and
data/test/ (1300) for test.

Usage: pass cache file paths directly (no hash reproduction needed):

  python scripts/diag_old_val_split.py \
    --train feature_cache/7e10dd9e7e6f7539.pt \
    --val   feature_cache/61724e8be6e62d7b.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def lstsq_fit(X_tr: torch.Tensor, Y_tr: torch.Tensor):
    Xb = torch.cat([X_tr, torch.ones(X_tr.size(0), 1, dtype=X_tr.dtype)], dim=1)
    sol = torch.linalg.lstsq(Xb, Y_tr).solution
    return sol[:-1].T, sol[-1]


def per_param_mse(pred: torch.Tensor, y: torch.Tensor) -> dict:
    err = (pred - y).pow(2).mean(0)
    return {"mse_per_param": err.tolist(), "mse_mean": err.mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train cache .pt")
    p.add_argument("--val", required=True, help="Path to val cache .pt")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_f = torch.load(args.train)
    val_f = torch.load(args.val)
    print(f"train features={tuple(train_f['features'].shape)} labels={tuple(train_f['labels_raw'].shape)}")
    print(f"val   features={tuple(val_f['features'].shape)} labels={tuple(val_f['labels_raw'].shape)}")

    mean = train_f["labels_raw"].mean(0)
    std = train_f["labels_raw"].std(0).clamp_min(1e-8)
    Y_tr = (train_f["labels_raw"] - mean) / std
    Y_va = (val_f["labels_raw"] - mean) / std

    W, b = lstsq_fit(train_f["features"], Y_tr)
    pred_full = val_f["features"] @ W.T + b

    full = per_param_mse(pred_full, Y_va)

    # Old code: torch.Generator().manual_seed(cfg.seed); randperm(N)
    N = val_f["features"].shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    half = N // 2
    val_half = per_param_mse(pred_full[perm[:half]], Y_va[perm[:half]])
    test_half = per_param_mse(pred_full[perm[half:]], Y_va[perm[half:]])

    print()
    print("=" * 70)
    print("LINEAR PROBE (lstsq on train, predict on val)")
    print("=" * 70)
    print(f"  full val   (N={N})    mse_mean={full['mse_mean']:.4f}   per_param={[f'{x:.4f}' for x in full['mse_per_param']]}")
    print(f"  old 'val'  (first {half} of seed-{args.seed} randperm)  mse_mean={val_half['mse_mean']:.4f}   per_param={[f'{x:.4f}' for x in val_half['mse_per_param']]}")
    print(f"  old 'test' (last {N-half} of seed-{args.seed} randperm)  mse_mean={test_half['mse_mean']:.4f}   per_param={[f'{x:.4f}' for x in test_half['mse_per_param']]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
