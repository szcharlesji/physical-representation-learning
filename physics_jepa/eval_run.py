"""Sweep every checkpoint in a run dir; emit one wandb run with 18 curves.

For each `Encoder_<N>.pth` under the run directory we compute, on **both
val and test** splits:
  - linear (closed-form lstsq)         -> linear/{val,test}/{alpha,zeta,mean}
  - kNN best (k, metric) by val mean   -> knn/{val,test}/{alpha,zeta,mean}
  - attentive probe best head-epoch    -> attentive/{val,test}/{alpha,zeta,mean}

The kNN (k, metric) and attentive head epoch are selected by **val** mean
MSE; test numbers are then reported at that same selection. This keeps the
test split fully held-out (no selection-on-test leakage) while still giving
a directly-comparable test number per checkpoint.

All metrics share x-axis `epoch` (the pretrain checkpoint number), so wandb
renders 18 curves on one run instead of 11+ scattered child runs.

Reuses `FrozenEvaluator.run()` for linear/kNN and a thin `JepaFinetuner`
subclass for attentive. Both run with `cfg.dry_run = True` so they don't
touch wandb -- this driver owns the single run.

Data parity: all three probes evaluate on the full val and test splits with
the same `noise_std` the encoder was pretrained on. FrozenEvaluator falls
back to `cfg.train.noise_std` when `cfg.ft` is the swapped frozen.yaml block,
so linear/kNN match attentive's preprocessing automatically.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from .eval_frozen import FrozenEvaluator
from .post_train_probes import _clear_dist_env, _load_ft_block, find_checkpoints
from .utils.wandb_utils import init_run as wandb_init_run


# --------------------------------------------------------------------- helpers

def _epoch_from_ckpt(ckpt: Path) -> Optional[int]:
    """Extract the epoch number from `<Class>Encoder_<N>.pth`. Skip step-checkpoints."""
    m = re.match(r"^[A-Za-z0-9]+Encoder_(\d+)$", ckpt.stem)
    return int(m.group(1)) if m else None


def _frozen_metrics(
    cfg: DictConfig, ckpt_path: str, frozen_config: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Run linear+kNN via FrozenEvaluator with wandb suppressed.

    Returns per-param + mean MSE on val and test for both probes. kNN's
    (k, metric) is selected by val mean; the test numbers are at that same
    selection (no selection-on-test peeking). Shape:

        {
          "linear": {"val": {alpha, zeta, mean}, "test": {alpha, zeta, mean}},
          "knn":    {"val": {alpha, zeta, mean}, "test": {alpha, zeta, mean},
                     "k": int, "metric": str},
        }
    """
    sub = copy.deepcopy(cfg)
    OmegaConf.set_struct(sub, False)
    sub.ft = copy.deepcopy(_load_ft_block(frozen_config))
    sub.ft.eval_mode = "linear_and_knn"
    sub.ft.run_name = None

    # Per-checkpoint out_dir mirrors post_train_probes' layout so results.json
    # files don't clobber each other.
    base_out = Path(sub.ft.get("out_dir", "./frozen_eval_out"))
    ckpt = Path(ckpt_path)
    sub.ft.out_dir = str(base_out / ckpt.parent.name / ckpt.stem)
    sub.seed = sub.get("seed", 42)
    sub.dry_run = True  # FrozenEvaluator._init_wandb -> no-op

    evaluator = FrozenEvaluator(sub, checkpoint_path=str(ckpt))
    results = evaluator.run()

    def _row(probe_type: str, split: str):
        return next(
            r for r in results
            if r["probe_type"] == probe_type and r["split"] == split
        )

    linear_val = _row("linear", "val")
    linear_test = _row("linear", "test")

    # Pick best kNN (k, metric) by val mean, then take the matching test row.
    knn_val_rows = [
        r for r in results if r["probe_type"] == "knn" and r["split"] == "val"
    ]
    if not knn_val_rows:
        raise RuntimeError(f"no kNN val rows for {ckpt_path}")
    best_knn_val = min(knn_val_rows, key=lambda r: r["mse_mean"])
    best_k, best_metric = int(best_knn_val["k"]), str(best_knn_val["metric"])
    best_knn_test = next(
        r for r in results
        if r["probe_type"] == "knn"
        and r["split"] == "test"
        and r["k"] == best_k
        and r["metric"] == best_metric
    )

    return {
        "linear": {
            "val": {
                "alpha": float(linear_val["mse_alpha"]),
                "zeta": float(linear_val["mse_zeta"]),
                "mean": float(linear_val["mse_mean"]),
            },
            "test": {
                "alpha": float(linear_test["mse_alpha"]),
                "zeta": float(linear_test["mse_zeta"]),
                "mean": float(linear_test["mse_mean"]),
            },
        },
        "knn": {
            "val": {
                "alpha": float(best_knn_val["mse_alpha"]),
                "zeta": float(best_knn_val["mse_zeta"]),
                "mean": float(best_knn_val["mse_mean"]),
            },
            "test": {
                "alpha": float(best_knn_test["mse_alpha"]),
                "zeta": float(best_knn_test["mse_zeta"]),
                "mean": float(best_knn_test["mse_mean"]),
            },
            "k": best_k,
            "metric": best_metric,
        },
    }


def _attentive_metrics(
    cfg: DictConfig, ckpt_path: str
) -> Dict[str, Dict[str, float]]:
    """Train attentive probe on this checkpoint; return best per-param val + test MSE.

    The head trains for `cfg.ft.num_epochs` epochs (default 100 from
    `configs/ft/linear.yaml`). We override `val()` to record per-param MSE
    on **both** val and test each epoch. The snapshot is selected by lowest
    val mean; the corresponding test numbers come from the same epoch (no
    selection on test).

    Both code paths are supported:
      - `not_from_embeddings=True`: raw forwards each epoch. We build a
        parallel raw test_loader alongside the parent's val_loader.
      - `not_from_embeddings=False` (default): cached HDF5 embeddings for
        train+val. We additionally compute test embeddings **once** in
        memory after the parent's get_embeddings() runs (encoder is
        re-loaded for that single forward pass, then dropped).

    Returns: {"val": {alpha, zeta, mean}, "test": {alpha, zeta, mean}}.
    """
    import gc

    import numpy as np

    from .finetuner import JepaFinetuner
    from .utils.data_utils import normalize_labels
    from .data import EmbeddingsDataset, get_dataset, _build_norm_stats_from_cfg

    sub = copy.deepcopy(cfg)
    OmegaConf.set_struct(sub, False)
    sub.ft.use_attentive_pooling = True
    sub.ft.task = "regression"
    sub.ft.run_name = None
    sub.ft.distributed = False
    sub.seed = sub.get("seed", 42)
    sub.dry_run = True  # JepaFinetuner.train + Trainer.training_loop -> no wandb

    class _AttCurve(JepaFinetuner):
        """Capture per-param val + test MSE; track best by val mean."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            inf3 = lambda: {
                "alpha": float("inf"),
                "zeta": float("inf"),
                "mean": float("inf"),
            }
            self.best_val = inf3()
            # Test stats taken from the same epoch where val_best was set.
            # No independent selection on test.
            self.best_test = inf3()
            # self.test_loader is filled by get_embeddings() (default path)
            # or get_encoder_and_raw_loaders() (raw path) before val() runs.
            self.test_loader = None

        # -- raw path ---------------------------------------------------------
        def get_encoder_and_raw_loaders(self):
            """Build raw train+val (parent) plus a raw test loader."""
            encoder = super().get_encoder_and_raw_loaders()
            norm_stats = _build_norm_stats_from_cfg(self.cfg, rank=self.rank or 0)
            test_ds = get_dataset(
                self.cfg.dataset.name,
                self.cfg.dataset.num_frames,
                split="test",
                include_labels=True,
                resolution=self.cfg.dataset.get("resolution", None),
                offset=self.cfg.dataset.get("offset", None),
                noise_std=self.cfg.ft.get("noise_std", 0.0),
                resize_mode=self.cfg.dataset.get("resize_mode", "bilinear"),
                augment_cfg=None,
                norm_stats=norm_stats,
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=self.cfg.ft.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )
            return encoder

        # -- embeddings path --------------------------------------------------
        def get_embeddings(self):
            """Parent caches train+val embeddings; we additionally produce
            test embeddings in memory using a freshly-loaded encoder.

            Test embeddings are NOT cached on disk -- they're recomputed
            per-checkpoint. Cheap relative to train (test is much smaller).
            """
            train_data, train_labels, val_data, val_labels = super().get_embeddings()

            # Build a raw test loader and forward through a freshly-loaded
            # encoder (the parent dropped its encoder after computing
            # train+val embeddings).
            norm_stats = _build_norm_stats_from_cfg(self.cfg, rank=self.rank or 0)
            test_ds_raw = get_dataset(
                self.cfg.dataset.name,
                self.cfg.dataset.num_frames,
                split="test",
                include_labels=True,
                resolution=self.cfg.dataset.get("resolution", None),
                offset=self.cfg.dataset.get("offset", None),
                noise_std=self.cfg.ft.get("noise_std", 0.0),
                resize_mode=self.cfg.dataset.get("resize_mode", "bilinear"),
                augment_cfg=None,
                norm_stats=norm_stats,
            )
            raw_test_loader = torch.utils.data.DataLoader(
                test_ds_raw,
                batch_size=self.cfg.ft.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )

            encoder = self.load_model()
            encoder.to(self.rank)
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

            test_emb_chunks: List[np.ndarray] = []
            test_lbl_chunks: List[np.ndarray] = []
            print("[eval_run] computing test embeddings (in-memory)", flush=True)
            with torch.no_grad():
                for batch in raw_test_loader:
                    enc_ctx, lbl = self.inference_step(batch, encoder)
                    test_emb_chunks.append(enc_ctx.detach().cpu().numpy())
                    test_lbl_chunks.append(lbl.detach().cpu().numpy())

            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            test_emb = np.concatenate(test_emb_chunks, axis=0)
            test_lbl = np.concatenate(test_lbl_chunks, axis=0)
            print(
                f"[eval_run] test embeddings: {test_emb.shape} labels: {test_lbl.shape}",
                flush=True,
            )

            test_dataset = EmbeddingsDataset(test_emb, test_lbl)
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.cfg.ft.batch_size,
                shuffle=False,
                num_workers=4,
            )
            return train_data, train_labels, val_data, val_labels

        # -- shared ----------------------------------------------------------
        def _compute_per_param_mse(self, loader, model_components):
            """Return (alpha, zeta, mean) MSE over `loader`.

            Branches on `not_from_embeddings` exactly like the parent's
            val() did, so the same loop iterates raw or cached batches.
            """
            sse = torch.zeros(2)
            n = 0
            with torch.no_grad():
                for batch in loader:
                    if self.cfg.ft.get("not_from_embeddings", False):
                        ctx_in = batch["context"].to(self.rank)
                        ctx = self._model_inference(ctx_in, model_components[0])
                        head = model_components[1]
                        labels = normalize_labels(
                            batch[self.label_name], stats=self.label_stats
                        ).to(self.rank)
                    else:
                        ctx = batch["embeddings"].to(self.rank)
                        head = model_components[0]
                        labels = batch["label"].to(self.rank)
                    pred = head(ctx)
                    err = ((pred - labels) ** 2).detach().cpu()
                    sse += err.sum(dim=0)
                    n += err.shape[0]
            per = (sse / max(n, 1)).tolist()
            return {
                "alpha": float(per[0]),
                "zeta": float(per[1]),
                "mean": float(sum(per) / len(per)),
            }

        def val(self, model_components, loss_fn, epoch):
            for c in model_components:
                c.eval()
            val_metrics = self._compute_per_param_mse(
                self.val_loader, model_components
            )
            # Test forward over the same model state. test_loader is built by
            # whichever path (embeddings or raw) ran during train(). Stays
            # None only if both setup hooks were skipped, which would itself
            # be a bug -- emit infs in that case so the bug is visible.
            if self.test_loader is not None:
                test_metrics = self._compute_per_param_mse(
                    self.test_loader, model_components
                )
            else:
                test_metrics = {
                    "alpha": float("inf"),
                    "zeta": float("inf"),
                    "mean": float("inf"),
                }
            # Selection by val mean: when val improves, snapshot BOTH val and
            # test from this epoch. test never drives selection.
            if val_metrics["mean"] < self.best_val["mean"]:
                self.best_val = val_metrics
                self.best_test = test_metrics
            return {"val/loss": torch.tensor(val_metrics["mean"])}

    finetuner = _AttCurve(
        sub, trained_model_path=str(ckpt_path), rank=0, world_size=1
    )
    finetuner.train()
    return {"val": finetuner.best_val, "test": finetuner.best_test}


# ------------------------------------------------------------------------ main

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_dir",
        required=True,
        help="Run directory containing ConvEncoder_*.pth and config.yaml",
    )
    p.add_argument(
        "--frozen_config",
        default=None,
        help="Override post_train_eval.frozen_config (needed if missing in cfg)",
    )
    p.add_argument(
        "--probes",
        nargs="+",
        default=["linear", "knn", "attentive"],
        help="Subset of {linear, knn, attentive}",
    )
    p.add_argument(
        "--name",
        default=None,
        help="W&B run name (default: <run_dir_name>-eval-curve)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.ckpt_dir).resolve()
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"missing config.yaml at {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, False)
    _clear_dist_env()

    # Prefer the run's own config when it specifies a frozen config (newer
    # runs do this via post_train_eval.frozen_config). Fall back to --frozen_config
    # for older runs whose config.yaml predates that block.
    frozen_config = cfg.get("post_train_eval", {}).get("frozen_config", None) \
        or args.frozen_config
    if frozen_config is None and ("linear" in args.probes or "knn" in args.probes):
        raise SystemExit(
            "--frozen_config or post_train_eval.frozen_config required for linear/knn"
        )
    print(f"[eval_run] frozen_config: {frozen_config}", flush=True)

    ckpts = find_checkpoints(run_dir)
    epoch_ckpts = [(c, _epoch_from_ckpt(c)) for c in ckpts]
    epoch_ckpts = [(c, e) for (c, e) in epoch_ckpts if e is not None]
    if not epoch_ckpts:
        raise SystemExit(f"no Encoder_<N>.pth found under {run_dir}")
    epoch_ckpts.sort(key=lambda x: x[1])
    print(
        f"[eval_run] {len(epoch_ckpts)} checkpoints; "
        f"epochs={[e for _, e in epoch_ckpts]}",
        flush=True,
    )

    import wandb

    run_name = args.name or f"{run_dir.name}-eval-curve"
    wandb_init_run(
        cfg,
        job_type="probe_eval_curve",
        group=run_dir.name,
        name=run_name,
        extra_config={
            "ckpt_dir": str(run_dir),
            "probes": list(args.probes),
            "n_checkpoints": len(epoch_ckpts),
        },
    )

    # Bind every probe metric to the pretrain epoch so wandb plots curves with
    # epoch on the x-axis instead of wandb's default global step. The wildcard
    # `<probe>/*` captures both the per-split namespace (`linear/val/alpha`,
    # `linear/test/alpha`) and the bookkeeping keys (`knn/best_k`).
    wandb.define_metric("epoch")
    for prefix in ("linear", "knn", "attentive"):
        wandb.define_metric(f"{prefix}/*", step_metric="epoch")

    rows: List[Dict] = []
    for ckpt, epoch in epoch_ckpts:
        print(f"\n[eval_run] === epoch={epoch} ckpt={ckpt.name} ===", flush=True)
        log: Dict = {"epoch": epoch}
        row: Dict = {"epoch": epoch, "checkpoint": str(ckpt)}

        if "linear" in args.probes or "knn" in args.probes:
            try:
                fm = _frozen_metrics(cfg, str(ckpt), frozen_config)
                if "linear" in args.probes:
                    for split in ("val", "test"):
                        for k in ("alpha", "zeta", "mean"):
                            log[f"linear/{split}/{k}"] = fm["linear"][split][k]
                    row["linear"] = fm["linear"]
                if "knn" in args.probes:
                    for split in ("val", "test"):
                        for k in ("alpha", "zeta", "mean"):
                            log[f"knn/{split}/{k}"] = fm["knn"][split][k]
                    log["knn/best_k"] = fm["knn"]["k"]
                    log["knn/best_metric_is_cosine"] = (
                        1 if fm["knn"]["metric"] == "cosine" else 0
                    )
                    row["knn"] = fm["knn"]
            except Exception as e:
                print(
                    f"[eval_run] frozen probes failed at epoch {epoch}: {e}",
                    flush=True,
                )

        if "attentive" in args.probes:
            try:
                am = _attentive_metrics(cfg, str(ckpt))
                for split in ("val", "test"):
                    for k in ("alpha", "zeta", "mean"):
                        log[f"attentive/{split}/{k}"] = am[split][k]
                row["attentive"] = am
            except Exception as e:
                print(
                    f"[eval_run] attentive probe failed at epoch {epoch}: {e}",
                    flush=True,
                )

        wandb.log(log)
        rows.append(row)
        print(f"[eval_run] epoch={epoch} -> {log}", flush=True)

    # Best-epoch summary per probe. Two parallel selections:
    #   best_val_*  -> epoch with lowest val mean (canonical model-selection)
    #   best_test_* -> epoch with lowest test mean (independent reference;
    #                  often the same epoch, but the gap is informative)
    # For each selection we record both the val and test numbers from THAT
    # epoch, plus knn's (k, metric).
    summary: Dict = {}

    def _emit(prefix: str, sel_split: str, picked: Dict):
        """Write summary keys for one (probe, selection-split) pair."""
        block = picked[prefix]
        tag = f"best_{sel_split}"
        summary[f"{prefix}/{tag}_epoch"] = picked["epoch"]
        for split in ("val", "test"):
            for stat in ("alpha", "zeta", "mean"):
                summary[f"{prefix}/{tag}_{split}_{stat}"] = block[split][stat]
        if prefix == "knn":
            summary[f"knn/{tag}_k"] = block["k"]
            summary[f"knn/{tag}_metric"] = block["metric"]

    for prefix in ("linear", "knn", "attentive"):
        ranked = [
            r for r in rows
            if prefix in r and "val" in r[prefix] and "mean" in r[prefix]["val"]
        ]
        if not ranked:
            continue
        best_val = min(ranked, key=lambda r: r[prefix]["val"]["mean"])
        best_test = min(ranked, key=lambda r: r[prefix]["test"]["mean"])
        _emit(prefix, "val", best_val)
        _emit(prefix, "test", best_test)
    for k, v in summary.items():
        wandb.run.summary[k] = v

    out_path = run_dir / "eval_curve.json"
    out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"[eval_run] wrote summary -> {out_path}", flush=True)

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
