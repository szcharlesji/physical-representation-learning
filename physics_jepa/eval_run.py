"""Sweep every checkpoint in a run dir; emit one wandb run with 9 curves.

For each `Encoder_<N>.pth` under the run directory we compute, on the val set:
  - linear (closed-form lstsq)         -> linear/{alpha,zeta,mean}
  - kNN best (k, metric) by val mean   -> knn/{alpha,zeta,mean}
  - attentive probe best head-epoch    -> attentive/{alpha,zeta,mean}

All metrics share x-axis `epoch` (the pretrain checkpoint number), so wandb
renders 9 curves on one run instead of 11+ scattered child runs.

Reuses `FrozenEvaluator.run()` for linear/kNN and a thin `JepaFinetuner`
subclass for attentive. Both run with `cfg.dry_run = True` so they don't
touch wandb -- this driver owns the single run.

Data parity: all three probes are evaluated on the full val split with the
same `noise_std` the encoder was pretrained on. FrozenEvaluator falls back
to `cfg.train.noise_std` when `cfg.ft` is the swapped frozen.yaml block, so
linear/kNN match attentive's preprocessing automatically. Numbers across
the three rows are directly comparable -- their gap reflects probe
expressivity, not data mismatch.
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
) -> Dict[str, Dict[str, float]]:
    """Run linear+kNN via FrozenEvaluator with wandb suppressed.

    Picks the val-split row for linear (single config) and the best (k, metric)
    pair for kNN by val mean MSE. Returns z-normalized per-param numbers --
    the same scale as `linear/val_mse` etc. on existing per-checkpoint runs.
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

    linear_val = next(
        r for r in results
        if r["probe_type"] == "linear" and r["split"] == "val"
    )
    knn_val_rows = [
        r for r in results if r["probe_type"] == "knn" and r["split"] == "val"
    ]
    if not knn_val_rows:
        raise RuntimeError(f"no kNN val rows for {ckpt_path}")
    best_knn = min(knn_val_rows, key=lambda r: r["mse_mean"])

    return {
        "linear": {
            "alpha": float(linear_val["mse_alpha"]),
            "zeta": float(linear_val["mse_zeta"]),
            "mean": float(linear_val["mse_mean"]),
        },
        "knn": {
            "alpha": float(best_knn["mse_alpha"]),
            "zeta": float(best_knn["mse_zeta"]),
            "mean": float(best_knn["mse_mean"]),
            "k": int(best_knn["k"]),
            "metric": str(best_knn["metric"]),
        },
    }


def _attentive_metrics(cfg: DictConfig, ckpt_path: str) -> Dict[str, float]:
    """Train attentive probe on this checkpoint; return best per-param val MSE.

    The head trains for `cfg.ft.num_epochs` epochs (default 100 from
    `configs/ft/linear.yaml`). We override `val()` to record per-param MSE
    each epoch and keep the snapshot whose mean is lowest.
    """
    from .finetuner import JepaFinetuner
    from .utils.data_utils import normalize_labels

    sub = copy.deepcopy(cfg)
    OmegaConf.set_struct(sub, False)
    sub.ft.use_attentive_pooling = True
    sub.ft.task = "regression"
    sub.ft.run_name = None
    sub.ft.distributed = False
    sub.seed = sub.get("seed", 42)
    sub.dry_run = True  # JepaFinetuner.train + Trainer.training_loop -> no wandb

    class _AttCurve(JepaFinetuner):
        """Capture per-param val MSE; track best by mean across head-epochs."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.best = {
                "alpha": float("inf"),
                "zeta": float("inf"),
                "mean": float("inf"),
            }

        def val(self, model_components, loss_fn, epoch):
            for c in model_components:
                c.eval()
            sse = torch.zeros(2)
            n = 0
            with torch.no_grad():
                for batch in self.val_loader:
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
            mean = sum(per) / len(per)
            if mean < self.best["mean"]:
                self.best = {
                    "alpha": float(per[0]),
                    "zeta": float(per[1]),
                    "mean": float(mean),
                }
            return {"val/loss": torch.tensor(mean)}

    finetuner = _AttCurve(
        sub, trained_model_path=str(ckpt_path), rank=0, world_size=1
    )
    finetuner.train()
    return finetuner.best


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
    # epoch on the x-axis instead of wandb's default global step.
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
                    for k in ("alpha", "zeta", "mean"):
                        log[f"linear/{k}"] = fm["linear"][k]
                    row["linear"] = fm["linear"]
                if "knn" in args.probes:
                    for k in ("alpha", "zeta", "mean"):
                        log[f"knn/{k}"] = fm["knn"][k]
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
                for k in ("alpha", "zeta", "mean"):
                    log[f"attentive/{k}"] = am[k]
                row["attentive"] = am
            except Exception as e:
                print(
                    f"[eval_run] attentive probe failed at epoch {epoch}: {e}",
                    flush=True,
                )

        wandb.log(log)
        rows.append(row)
        print(f"[eval_run] epoch={epoch} -> {log}", flush=True)

    # Best-epoch summary per probe (selected by mean MSE on val).
    summary: Dict = {}
    for prefix in ("linear", "knn", "attentive"):
        ranked = [r for r in rows if prefix in r and "mean" in r[prefix]]
        if not ranked:
            continue
        best = min(ranked, key=lambda r: r[prefix]["mean"])
        summary[f"{prefix}/best_epoch"] = best["epoch"]
        summary[f"{prefix}/best_mean"] = best[prefix]["mean"]
        summary[f"{prefix}/best_alpha"] = best[prefix]["alpha"]
        summary[f"{prefix}/best_zeta"] = best[prefix]["zeta"]
        if prefix == "knn":
            summary["knn/best_k"] = best["knn"]["k"]
            summary["knn/best_metric"] = best["knn"]["metric"]
    for k, v in summary.items():
        wandb.run.summary[k] = v

    out_path = run_dir / "eval_curve.json"
    out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"[eval_run] wrote summary -> {out_path}", flush=True)

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
