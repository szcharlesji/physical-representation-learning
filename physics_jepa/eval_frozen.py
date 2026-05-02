"""Frozen-encoder evaluation: linear probe + kNN.

Standalone alternative to finetuner.py's attentive-probe path. The existing
attentive-probe flow in finetuner.py is untouched.

Reports per-param + mean MSE on **both val and test** splits. Linear is fit
once via lstsq on train and predicted on val and test; kNN sweeps (k, metric),
selects the best pair by val mean MSE, and reports val+test metrics for every
swept pair so the wandb panel shows the val/test gap directly.
"""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_dataset, _build_norm_stats_from_cfg
from .model import get_model_and_loss_cnn
from .utils.hydra import compose
from .utils.wandb_utils import init_run as wandb_init_run, group_from_checkpoint


PARAM_NAMES = ["alpha", "zeta"]  # scalars are sorted alphabetically in data.py:_build_index


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
        # Dispatch on cfg.model.backbone so checkpoints trained with a
        # vit3d/conv3d_next_attn backbone load into the matching encoder.
        encoder, _, _ = get_model_and_loss_cnn(
            self.cfg.model.dims,
            self.cfg.model.num_res_blocks,
            self.cfg.dataset.num_frames,
            in_chans=self.cfg.dataset.num_chans,
            model_cfg=self.cfg.model,
            img_size=self.cfg.dataset.get("resolution", None),
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

    def _resolved_noise_std(self) -> float:
        # Match attentive's preprocessing: probe the encoder under the noise
        # distribution it was pretrained on. Source order:
        #   1) cfg.ft.noise_std (set when ft block is the pretrain ft, e.g.
        #      attentive flow keeps the un-swapped pretrain config)
        #   2) cfg.train.noise_std (the pretrain noise; what frozen.yaml-swapped
        #      paths fall back to)
        #   3) 0.0 if neither is set
        ft_noise = self.cfg.ft.get("noise_std", None)
        if ft_noise is not None:
            return float(ft_noise)
        return float(self.cfg.get("train", {}).get("noise_std", 0.0))

    def _cache_key(self, split: str) -> str:
        s = "|".join([
            str(Path(self.checkpoint_path).resolve()),
            split,
            self.cfg.dataset.name,
            str(self.cfg.dataset.num_frames),
            str(self.cfg.dataset.get("resolution", None)),
            str(self.cfg.dataset.get("resize_mode", "bilinear")),
            self.feature_pool,
            f"noise={self._resolved_noise_std()}",
        ])
        return hashlib.sha1(s.encode()).hexdigest()[:16]

    def _make_loader(self, split: str) -> DataLoader:
        # Match the pretrain-time per-channel normalization (if any) so the
        # frozen encoder sees the pixel distribution it was trained on.
        norm_stats = _build_norm_stats_from_cfg(self.cfg, rank=0)
        noise_std = self._resolved_noise_std()
        print(
            f"[FrozenEvaluator] split={split} noise_std={noise_std}",
            flush=True,
        )
        dataset = get_dataset(
            self.cfg.dataset.name,
            self.cfg.dataset.num_frames,
            split=split,
            include_labels=True,
            resolution=self.cfg.dataset.get("resolution", None),
            offset=self.cfg.dataset.get("offset", None),
            noise_std=noise_std,
            resize_mode=self.cfg.dataset.get("resize_mode", "bilinear"),
            augment_cfg=None,
            norm_stats=norm_stats,
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

    def _init_wandb(self, job_type: str):
        """Start a single wandb run that hosts every probe in this evaluator.

        `job_type` is one of `probe_linear`, `probe_knn`, `probe_frozen`. When
        eval_mode runs both linear and knn we use `probe_frozen` so they land
        in one run and metrics live under `linear/...` and `knn/...` prefixes.
        """
        if self.cfg.get("dry_run", False):
            self._wandb_on = False
            return
        group = group_from_checkpoint(self.checkpoint_path)
        ckpt_stem = Path(self.checkpoint_path).stem  # e.g. ConvEncoder_11
        name = self.cfg.ft.get("run_name") or f"{group}-{job_type}-{ckpt_stem}"
        wandb_init_run(
            self.cfg,
            job_type=job_type,
            group=group,
            name=name,
            extra_config={
                "probe_type": job_type.replace("probe_", ""),
                "eval_mode": self.eval_mode,
                "checkpoint_path": str(Path(self.checkpoint_path).resolve()),
            },
        )
        self._wandb_on = True

    def _finish_wandb(self):
        if getattr(self, "_wandb_on", False):
            wandb.finish()
            self._wandb_on = False

    def run(self):
        encoder = self.load_encoder()
        train_f = self.extract_features(encoder, "train")
        val_f = self.extract_features(encoder, "val")
        # Held-out test split (The Well exposes train/valid/test natively).
        # Cached separately from val via _cache_key's split component.
        test_f = self.extract_features(encoder, "test")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mean = train_f["labels_raw"].mean(0)
        std = train_f["labels_raw"].std(0).clamp_min(1e-8)
        self._y_mean = mean
        self._y_std = std
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

        # One wandb run hosts every probe FrozenEvaluator runs in this call:
        # `probe_frozen` when both linear and knn fire, otherwise the
        # single-probe job_type. Metrics are namespaced by `linear/` and
        # `knn/` so they coexist on one run's chart panel.
        if self.eval_mode == "linear_and_knn":
            job_type = "probe_frozen"
        elif self.eval_mode == "linear":
            job_type = "probe_linear"
        else:
            job_type = "probe_knn"
        self._init_wandb(job_type)

        results: List[Dict] = []
        if self.eval_mode in ("linear", "linear_and_knn"):
            results += self.run_linear(train_f, val_f, test_f)
        if self.eval_mode in ("knn", "linear_and_knn"):
            results += self.run_knn(train_f, val_f, test_f)
        self._finish_wandb()

        self._report(results, mean=mean.tolist(), std=std.tolist())
        return results

    def _report(self, results: List[Dict], mean, std):
        header = ["probe_type", "k", "metric", "split",
                  "mse_alpha", "mse_zeta", "mse_mean",
                  "mse_alpha_raw", "mse_zeta_raw", "mse_mean_raw"]
        widths = [10, 5, 10, 5, 10, 10, 10, 14, 14, 14]

        def fmt(v, w):
            if v is None:
                return f"{'-':<{w}}"
            if isinstance(v, float):
                return f"{v:<{w}.4f}"
            return f"{str(v):<{w}}"

        line = "  ".join(f"{h:<{w}}" for h, w in zip(header, widths))
        sep = "-" * len(line)
        print()
        print(sep)
        print(line)
        print(sep)
        for r in results:
            row = [r.get(h) for h in header]
            print("  ".join(fmt(v, w) for v, w in zip(row, widths)))
        print(sep)

        # Headline TEST MSE (real held-out test split). Linear has a single
        # row; kNN reports the (k, metric) selected by val mean (knn_best tag
        # on the test row). Printed prominently so the user can paste these
        # numbers straight into the report table without grepping the table.
        linear_test = next(
            (r for r in results
             if r["probe_type"] == "linear" and r["split"] == "test"),
            None,
        )
        knn_at_best_val_test = next(
            (r for r in results
             if r["probe_type"] == "knn_best" and r["split"] == "test"),
            None,
        )
        print()
        print("=" * 70)
        print("HEADLINE TEST MSE (real held-out data/test/ split)")
        print("=" * 70)
        if linear_test is not None:
            print(
                f"  linear:  mean={linear_test['mse_mean']:.4f}  "
                f"alpha={linear_test['mse_alpha']:.4f}  "
                f"zeta={linear_test['mse_zeta']:.4f}"
            )
        if knn_at_best_val_test is not None:
            print(
                f"  knn   :  mean={knn_at_best_val_test['mse_mean']:.4f}  "
                f"alpha={knn_at_best_val_test['mse_alpha']:.4f}  "
                f"zeta={knn_at_best_val_test['mse_zeta']:.4f}  "
                f"(k={knn_at_best_val_test['k']}, "
                f"metric={knn_at_best_val_test['metric']}, selected by val)"
            )
        print("=" * 70)

        payload = {
            "checkpoint": str(Path(self.checkpoint_path).resolve()),
            "dataset": self.cfg.dataset.name,
            "eval_mode": self.eval_mode,
            "feature_pool": self.feature_pool,
            "label_order": PARAM_NAMES,
            "label_mean": mean,
            "label_std": std,
            "methodology": {"linear": "lstsq", "knn": "inv_distance"},
            "results": results,
        }
        out_path = self.out_dir / "results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"results saved to {out_path}", flush=True)

    # ------------------------------------------------------------------ linear
    def _per_param_mse(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        err = (pred - target) ** 2
        per = err.mean(dim=0).tolist()
        out = {f"mse_{PARAM_NAMES[i]}": float(per[i]) for i in range(len(PARAM_NAMES))}
        out["mse_mean"] = float(err.mean().item())
        mean = self._y_mean.to(pred.device, dtype=pred.dtype)
        std = self._y_std.to(pred.device, dtype=pred.dtype)
        pred_r = pred * std + mean
        target_r = target * std + mean
        err_r = (pred_r - target_r) ** 2
        per_r = err_r.mean(dim=0).tolist()
        for i in range(len(PARAM_NAMES)):
            out[f"mse_{PARAM_NAMES[i]}_raw"] = float(per_r[i])
        out["mse_mean_raw"] = float(err_r.mean().item())
        return out

    def run_linear(self, train_f, val_f, test_f) -> List[Dict]:
        lin_cfg = self.cfg.ft.linear
        X = train_f["features"]
        Y = train_f["labels"]
        design = torch.cat(
            [X, torch.ones(X.size(0), 1, dtype=X.dtype)], dim=1
        ) if lin_cfg.get("bias", True) else X
        sol = torch.linalg.lstsq(design, Y).solution
        if lin_cfg.get("bias", True):
            W = sol[:-1].T.contiguous()
            b = sol[-1].contiguous()
        else:
            W = sol.T.contiguous()
            b = torch.zeros(Y.size(1), dtype=Y.dtype)

        out: List[Dict] = []
        # Predict on val and test using the same lstsq solution.
        val_pred = val_f["features"] @ W.T + b
        val_metrics = self._per_param_mse(val_pred, val_f["labels"])
        out.append({"probe_type": "linear", "k": None, "metric": None,
                    "split": "val", **val_metrics})
        print(
            f"[linear] split=val   mse_mean={val_metrics['mse_mean']:.4f} "
            f"mse_mean_raw={val_metrics['mse_mean_raw']:.4f}",
            flush=True,
        )

        test_pred = test_f["features"] @ W.T + b
        test_metrics = self._per_param_mse(test_pred, test_f["labels"])
        out.append({"probe_type": "linear", "k": None, "metric": None,
                    "split": "test", **test_metrics})
        print(
            f"[linear] split=test  mse_mean={test_metrics['mse_mean']:.4f} "
            f"mse_mean_raw={test_metrics['mse_mean_raw']:.4f}",
            flush=True,
        )

        ckpt_path = self.out_dir / "linear_lstsq.pt"
        torch.save({"weight": W, "bias": b}, ckpt_path)
        print(f"[linear] lstsq solution saved to {ckpt_path}", flush=True)

        if self._wandb_on:
            wandb.log({
                "linear/val_mse": val_metrics["mse_mean"],
                "linear/val_mse_raw": val_metrics["mse_mean_raw"],
                "linear/val_mse_alpha": val_metrics["mse_alpha"],
                "linear/val_mse_zeta": val_metrics["mse_zeta"],
                "linear/test_mse": test_metrics["mse_mean"],
                "linear/test_mse_raw": test_metrics["mse_mean_raw"],
                "linear/test_mse_alpha": test_metrics["mse_alpha"],
                "linear/test_mse_zeta": test_metrics["mse_zeta"],
            })
            wandb.run.summary["linear/val_mse"] = val_metrics["mse_mean"]
            wandb.run.summary["linear/val_mse_raw"] = val_metrics["mse_mean_raw"]
            wandb.run.summary["linear/test_mse"] = test_metrics["mse_mean"]
            wandb.run.summary["linear/test_mse_raw"] = test_metrics["mse_mean_raw"]
        return out


    # --------------------------------------------------------------------- knn
    def run_knn(self, train_f, val_f, test_f) -> List[Dict]:
        knn_cfg = self.cfg.ft.knn
        chunk = int(knn_cfg.get("chunk_size", 1024))
        eps = 1e-8

        x_tr = train_f["features"]
        y_tr = train_f["labels"]

        def predict(xq: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
                    k: int, metric: str) -> torch.Tensor:
            if metric == "cosine":
                xq = F.normalize(xq, dim=1)
                xt = F.normalize(xt, dim=1)
            chunks = []
            for qc in xq.split(chunk):
                d = torch.cdist(qc, xt)
                dk, ik = torch.topk(d, k=k, dim=1, largest=False)
                yk = yt[ik]  # (q, k, out)
                zero = dk <= eps
                w = dk.clamp_min(eps).reciprocal()
                w = w / w.sum(dim=1, keepdim=True)
                if zero.any():
                    zw = zero.to(dtype=w.dtype)
                    zw = zw / zw.sum(dim=1, keepdim=True).clamp_min(1.0)
                    exact = (zw.unsqueeze(-1) * yk).sum(1)
                    weighted = (w.unsqueeze(-1) * yk).sum(1)
                    use_exact = zero.any(dim=1, keepdim=True)
                    chunks.append(torch.where(use_exact, exact, weighted))
                else:
                    chunks.append((w.unsqueeze(-1) * yk).sum(1))
            return torch.cat(chunks, dim=0)

        out: List[Dict] = []
        best = None
        step = 0
        for metric in knn_cfg.metrics:
            for k in knn_cfg.ks:
                if k > x_tr.size(0):
                    print(f"[knn] skipping k={k} > n_train={x_tr.size(0)}", flush=True)
                    continue
                # Compute val and test predictions for the same (k, metric).
                # Selection is by val mean; test is reported alongside.
                val_pred = predict(val_f["features"], x_tr, y_tr, k, metric)
                val_metrics = self._per_param_mse(val_pred, val_f["labels"])
                out.append({"probe_type": "knn", "k": k, "metric": metric,
                            "split": "val", **val_metrics})

                test_pred = predict(test_f["features"], x_tr, y_tr, k, metric)
                test_metrics = self._per_param_mse(test_pred, test_f["labels"])
                out.append({"probe_type": "knn", "k": k, "metric": metric,
                            "split": "test", **test_metrics})

                if best is None or val_metrics["mse_mean"] < best["val_mse_mean"]:
                    best = {
                        "k": k, "metric": metric,
                        "val_mse_mean": val_metrics["mse_mean"],
                        "test_mse_mean": test_metrics["mse_mean"],
                    }
                print(
                    f"[knn] k={k:3d} metric={metric:9s}  "
                    f"val_mean={val_metrics['mse_mean']:.4f}  "
                    f"test_mean={test_metrics['mse_mean']:.4f}",
                    flush=True,
                )
                if self._wandb_on:
                    wandb.log({
                        f"knn/{metric}/k{k}_val_mse_mean": val_metrics["mse_mean"],
                        f"knn/{metric}/k{k}_val_mse_mean_raw": val_metrics["mse_mean_raw"],
                        f"knn/{metric}/k{k}_test_mse_mean": test_metrics["mse_mean"],
                        f"knn/{metric}/k{k}_test_mse_mean_raw": test_metrics["mse_mean_raw"],
                        "knn/sweep_val_mse": val_metrics["mse_mean"],
                        "knn/sweep_val_mse_raw": val_metrics["mse_mean_raw"],
                        "knn/sweep_test_mse": test_metrics["mse_mean"],
                        "knn/sweep_test_mse_raw": test_metrics["mse_mean_raw"],
                        "knn/sweep_k": k,
                        "knn/sweep_metric_is_cosine": 1 if metric == "cosine" else 0,
                        "knn/sweep_step": step,
                    })
                step += 1

        if best is not None:
            # Tag the val and test rows of the best-by-val pair with knn_best.
            # Both rows are kept so eval_run.py can pull either split.
            for r in out:
                if (
                    r["probe_type"] == "knn"
                    and r["k"] == best["k"]
                    and r["metric"] == best["metric"]
                ):
                    out.append({**r, "probe_type": "knn_best"})
            if self._wandb_on:
                wandb.log({
                    "knn/best_val_mse": best["val_mse_mean"],
                    "knn/best_test_mse": best["test_mse_mean"],
                })
                wandb.run.summary["knn/best_val_mse"] = best["val_mse_mean"]
                wandb.run.summary["knn/best_test_mse"] = best["test_mse_mean"]
                wandb.run.summary["knn/best_k"] = best["k"]
                wandb.run.summary["knn/best_metric"] = best["metric"]
        return out


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
