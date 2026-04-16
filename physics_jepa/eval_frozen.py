"""Frozen-encoder evaluation: linear probe + kNN.

Standalone alternative to finetuner.py's attentive-probe path. The existing
attentive-probe flow in finetuner.py is untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_dataset
from .model import get_model_and_loss_cnn
from .utils.hydra import compose


class FrozenEvaluator:
    def __init__(self, cfg, checkpoint_path: str):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_mode = cfg.ft.eval_mode
        if self.eval_mode == "attentive":
            raise SystemExit("use physics_jepa.finetune for attentive mode")
        if self.eval_mode not in ("linear", "knn", "linear_and_knn"):
            raise ValueError(f"unknown eval_mode: {self.eval_mode}")

        self.feature_pool = cfg.ft.get("feature_pool", "gap")
        assert self.feature_pool in ("gap", "flatten")

        self.cache_dir = Path(cfg.ft.feature_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(cfg.ft.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def load_encoder(self) -> nn.Module:
        encoder, _, _ = get_model_and_loss_cnn(
            self.cfg.model.dims,
            self.cfg.model.num_res_blocks,
            self.cfg.dataset.num_frames,
            in_chans=self.cfg.dataset.num_chans,
        )
        print(f"loading encoder state dict from {self.checkpoint_path}", flush=True)
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        assert all(not p.requires_grad for p in encoder.parameters()), \
            "encoder params must be frozen"
        return encoder.to(self.device)

    def _cache_key(self, split: str) -> str:
        s = "|".join([
            str(Path(self.checkpoint_path).resolve()),
            split,
            self.cfg.dataset.name,
            str(self.cfg.dataset.num_frames),
            str(self.cfg.dataset.get("resolution", None)),
            self.feature_pool,
        ])
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def _make_loader(self, split: str) -> DataLoader:
        dataset = get_dataset(
            self.cfg.dataset.name,
            self.cfg.dataset.num_frames,
            split=split,
            include_labels=True,
            resolution=self.cfg.dataset.get("resolution", None),
            offset=self.cfg.dataset.get("offset", None),
            noise_std=0.0,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.ft.batch_size,
            shuffle=False,
            num_workers=self.cfg.ft.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, H, W)
        if self.feature_pool == "gap":
            return z.mean(dim=(-2, -1))
        return z.flatten(1)

    def extract_features(self, encoder: nn.Module, split: str) -> Dict[str, torch.Tensor]:
        cache_path = self.cache_dir / f"{self._cache_key(split)}.pt"
        if cache_path.exists():
            print(f"loading cached features from {cache_path}", flush=True)
            return torch.load(cache_path)

        print(f"extracting features for split={split}", flush=True)
        loader = self._make_loader(split)
        feats, raw_labels = [], []
        for batch in tqdm(loader, desc=f"encode {split}"):
            ctx = batch["context"].to(self.device, non_blocking=True)
            if ctx.shape[2] < 4:
                ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))
            with torch.no_grad():
                z = encoder(ctx)
                if torch.isnan(z).any():
                    raise ValueError(f"NaN in encoder output: split={split}")
                z = self._pool(z)
            feats.append(z.float().cpu())
            raw_labels.append(batch["physical_params"].float())

        out = {
            "features": torch.cat(feats, dim=0),
            "labels_raw": torch.cat(raw_labels, dim=0),
        }
        torch.save(out, cache_path)
        print(
            f"cached {split}: features={tuple(out['features'].shape)} "
            f"labels={tuple(out['labels_raw'].shape)} -> {cache_path}",
            flush=True,
        )
        return out

    def run(self):
        encoder = self.load_encoder()
        train_f = self.extract_features(encoder, "train")
        val_all = self.extract_features(encoder, "val")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        N = val_all["features"].shape[0]
        g = torch.Generator().manual_seed(self.cfg.seed)
        perm = torch.randperm(N, generator=g)
        half = N // 2
        val_f = {k: v[perm[:half]] for k, v in val_all.items()}
        test_f = {k: v[perm[half:]] for k, v in val_all.items()}

        mean = train_f["labels_raw"].mean(0)
        std = train_f["labels_raw"].std(0).clamp_min(1e-8)
        for s in (train_f, val_f, test_f):
            s["labels"] = (s["labels_raw"] - mean) / std

        print(
            f"label z-score: mean={mean.tolist()} std={std.tolist()}",
            flush=True,
        )
        print(
            f"train={tuple(train_f['features'].shape)} "
            f"val={tuple(val_f['features'].shape)} "
            f"test={tuple(test_f['features'].shape)}",
            flush=True,
        )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.seed = args.seed
    cfg.dry_run = args.dry_run
    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.model.objective != "jepa":
        raise ValueError(
            f"eval_frozen currently supports jepa only, got {cfg.model.objective}"
        )

    evaluator = FrozenEvaluator(cfg, checkpoint_path=args.checkpoint)
    evaluator.run()


if __name__ == "__main__":
    main()
