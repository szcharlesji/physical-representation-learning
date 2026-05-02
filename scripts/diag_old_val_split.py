"""Diagnostic: reproduce the old half-of-val MSE from cached features.

Old eval_frozen.run() did `val_all = extract_features("val")` (full 1200-sample
data/valid/), then sub-sampled to first 600 (seed-42 randperm) for val and
last 600 for test. The new run() extracts data/valid/ (1200) for val and
data/test/ (1300) for test — different splits, no half mixing.

If the old number (e.g. ai3ktxmv val=0.0857 on ViT3DEncoder_5) was just due to
the seed-42 first half of valid being easier, this script will reproduce
~0.0857 from the SAME cached features the new code uses. If instead it
reports the new ~0.155, the half-vs-full subsetting is NOT the explanation
and something else changed.

Resolution: this version constructs the cache hash via FrozenEvaluator._cache_key
on the run's actual config.yaml (with the same overrides applied as the eval
run), which avoids drift from manually stringifying resolution/resize_mode/etc.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from physics_jepa.eval_frozen import FrozenEvaluator
from physics_jepa.post_train_probes import _load_ft_block


def lstsq_fit(X_tr: torch.Tensor, Y_tr: torch.Tensor):
    Xb = torch.cat([X_tr, torch.ones(X_tr.size(0), 1, dtype=X_tr.dtype)], dim=1)
    sol = torch.linalg.lstsq(Xb, Y_tr).solution
    W = sol[:-1].T
    b = sol[-1]
    return W, b


def per_param_mse(pred: torch.Tensor, y: torch.Tensor) -> dict:
    err = (pred - y).pow(2).mean(0)
    return {"mse_per_param": err.tolist(), "mse_mean": err.mean().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Run config.yaml path")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--frozen-config",
        default="configs/train_activematter_frozen.yaml",
        help="ft block to swap in (matches post_train_probes.run_frozen_probes)",
    )
    p.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="Hydra-style dotlist overrides applied on top of the loaded config "
             "(should match the eval run that wrote the caches, e.g. "
             "`train.noise_std=0.0 ft.noise_std=0.0`)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Reproduce the cfg construction in post_train_probes.run_frozen_probes,
    # post-CLI-override:
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, False)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
        OmegaConf.set_struct(cfg, False)

    # post_train_probes.run_frozen_probes deep-copies pretrain_cfg, swaps
    # ft = frozen_config.ft, sets eval_mode/run_name/out_dir/seed/dry_run.
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.ft = copy.deepcopy(_load_ft_block(args.frozen_config))
    cfg.ft.eval_mode = "linear_and_knn"
    cfg.ft.run_name = None
    base_out = Path(cfg.ft.get("out_dir", "./frozen_eval_out"))
    ckpt = Path(args.checkpoint)
    cfg.ft.out_dir = str(base_out / ckpt.parent.name / ckpt.stem)
    cfg.seed = cfg.get("seed", 42)
    cfg.dry_run = cfg.get("dry_run", False)

    # IMPORTANT: re-apply ft.* overrides from the CLI dotlist, since `cfg.ft`
    # was just replaced wholesale by frozen_config's ft block. Mirrors the
    # subtle bug in post_train_probes that we don't want to recreate here.
    if args.overrides:
        ft_overrides = [o for o in args.overrides if o.startswith("ft.")]
        if ft_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(ft_overrides))
            OmegaConf.set_struct(cfg, False)

    # Construct evaluator just to get _cache_key + cache_dir; don't run anything.
    # FrozenEvaluator.__init__ doesn't load the encoder, so this is cheap.
    ev = FrozenEvaluator(cfg, checkpoint_path=str(ckpt))

    train_path = ev.cache_dir / f"{ev._cache_key('train')}.pt"
    val_path = ev.cache_dir / f"{ev._cache_key('val')}.pt"
    test_path = ev.cache_dir / f"{ev._cache_key('test')}.pt"

    print(f"feature_cache_dir: {ev.cache_dir}")
    print(f"resolved noise:    {ev._resolved_noise_std()}")
    print(f"feature_pool:      {ev.feature_pool}")
    print(f"resolution:        {cfg.dataset.get('resolution', None)}")
    print(f"resize_mode:       {cfg.dataset.get('resize_mode', None)}")
    print()
    print(f"train cache: {train_path}  exists={train_path.exists()}")
    print(f"val   cache: {val_path}    exists={val_path.exists()}")
    print(f"test  cache: {test_path}   exists={test_path.exists()}")

    if not (train_path.exists() and val_path.exists()):
        print()
        print("Listing all *.pt under feature_cache to help match by hand:")
        for f in sorted(ev.cache_dir.glob("*.pt")):
            try:
                d = torch.load(f, map_location="cpu", weights_only=True)
                shape = tuple(d["features"].shape) if "features" in d else "?"
            except Exception as e:
                shape = f"<load err: {e}>"
            print(f"  {f.name}  features={shape}")
        sys.exit(1)

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
