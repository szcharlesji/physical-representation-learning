"""Run probes automatically at end of pretraining.

Iterates saved checkpoints on rank 0 and launches FrozenEvaluator (linear +
kNN) and JepaFinetuner (attentive) as single-GPU, in-process evaluations.
Each probe uses its own W&B run whose `group` matches the pretrain group,
so pretrain + probes cluster together in the W&B UI.

This module is gated on `cfg.post_train_eval.enabled`; it is a no-op when
that flag is missing or false, so existing YAMLs without the block continue
to produce the same behavior as before.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf

from .utils.hydra import compose


def _maybe_wandb_finish() -> None:
    """Finish any in-flight wandb run so the next probe starts clean."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception as e:
        print(f"[post_train_eval] wandb.finish failed: {e}", flush=True)


def find_checkpoints(run_dir: Path) -> List[Path]:
    """Return sorted list of ConvEncoder_*.pth under run_dir.

    Sort order: step-checkpoints first (by step), then epoch-checkpoints
    (by epoch). Anything else falls to the end lexicographically.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    ckpts = list(run_dir.glob("ConvEncoder_*.pth"))

    def _sort_key(p: Path):
        stem = p.stem
        m = re.match(r"^ConvEncoder_step(\d+)$", stem)
        if m:
            return (0, int(m.group(1)), stem)
        m = re.match(r"^ConvEncoder_(\d+)$", stem)
        if m:
            return (1, int(m.group(1)), stem)
        return (2, 0, stem)

    return sorted(ckpts, key=_sort_key)


def _load_ft_block(frozen_config_path: str) -> DictConfig:
    """Load a full YAML via hydra compose and return only its `ft` block."""
    cfg = compose(frozen_config_path, [])
    OmegaConf.resolve(cfg)
    return cfg.ft


def _clear_dist_env() -> None:
    """Clear torchrun-injected env so the probes take the single-GPU path.

    After pretrain finishes we don't want JepaFinetuner's Trainer.__init__ to
    call ddp_setup() again (the process group is either already initialized
    and then destroyed, or initialized as a singleton; either way a second
    init_process_group call is unsafe).
    """
    for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"):
        os.environ.pop(k, None)


def run_frozen_probes(
    pretrain_cfg: DictConfig,
    ckpt_path: str,
    frozen_config: str,
    eval_mode: str = "linear_and_knn",
) -> None:
    """Run linear + kNN frozen probes on a single checkpoint.

    Deep-copies the pretrain cfg (so dataset/model match the trained encoder)
    and swaps in the `ft` block from the frozen config.
    """
    from .eval_frozen import FrozenEvaluator

    cfg = copy.deepcopy(pretrain_cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.ft = copy.deepcopy(_load_ft_block(frozen_config))
    cfg.ft.eval_mode = eval_mode
    cfg.ft.run_name = None

    # Scope out_dir per checkpoint so results.json doesn't overwrite.
    base_out = Path(cfg.ft.get("out_dir", "./frozen_eval_out"))
    ckpt = Path(ckpt_path)
    cfg.ft.out_dir = str(base_out / ckpt.parent.name / ckpt.stem)

    cfg.seed = cfg.get("seed", 42)
    cfg.dry_run = pretrain_cfg.get("dry_run", False)

    evaluator = FrozenEvaluator(cfg, checkpoint_path=str(ckpt_path))
    evaluator.run()
    _maybe_wandb_finish()


def run_attentive_probe(pretrain_cfg: DictConfig, ckpt_path: str) -> None:
    """Run the attentive probe on a single checkpoint.

    Reuses the pretrain cfg's `ft` block (already configured for an
    attentive probe in the train_*_small.yaml files), forcing
    `use_attentive_pooling=true` and `task=regression`. `run_name` is
    cleared so JepaFinetuner auto-generates a unique name per checkpoint.
    """
    from .finetuner import JepaFinetuner

    cfg = copy.deepcopy(pretrain_cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.ft.use_attentive_pooling = True
    cfg.ft.task = "regression"
    cfg.ft.run_name = None
    # Force single-GPU; avoid DDP re-init after pretrain teardown.
    cfg.ft.distributed = False
    cfg.seed = cfg.get("seed", 42)
    cfg.dry_run = pretrain_cfg.get("dry_run", False)

    finetuner = JepaFinetuner(cfg, trained_model_path=str(ckpt_path), rank=0, world_size=1)
    finetuner.train()
    _maybe_wandb_finish()


def run_post_train_probes(
    pretrain_cfg: DictConfig,
    run_dir: Path | str,
) -> None:
    """Entry point called from Trainer.train() on rank 0 after pretraining.

    Iterates every saved checkpoint under `run_dir` and runs the probes
    listed in `cfg.post_train_eval.probes` (default linear+knn+attentive).
    """
    post = pretrain_cfg.get("post_train_eval", None)
    if post is None or not post.get("enabled", False):
        return

    probes: Sequence[str] = list(post.get("probes", ["linear", "knn", "attentive"]))
    frozen_cfg: Optional[str] = post.get("frozen_config", None)

    ckpts = find_checkpoints(Path(run_dir))
    if not ckpts:
        print(
            f"[post_train_eval] no checkpoints under {run_dir}, skipping",
            flush=True,
        )
        return

    want_frozen = bool({"linear", "knn"} & set(probes))
    want_attentive = "attentive" in probes

    # linear_and_knn when both requested; otherwise run whichever the user asked for.
    if {"linear", "knn"} <= set(probes):
        frozen_mode = "linear_and_knn"
    elif "linear" in probes:
        frozen_mode = "linear"
    elif "knn" in probes:
        frozen_mode = "knn"
    else:
        frozen_mode = None

    _clear_dist_env()

    print(
        f"[post_train_eval] found {len(ckpts)} checkpoints under {run_dir}; "
        f"probes={list(probes)}",
        flush=True,
    )

    for ckpt in ckpts:
        ckpt_str = str(ckpt)
        if want_frozen:
            if frozen_cfg is None:
                print(
                    "[post_train_eval] linear/knn requested but "
                    "post_train_eval.frozen_config not set; skipping",
                    flush=True,
                )
            else:
                print(f"[post_train_eval] {frozen_mode} probe on {ckpt.name}", flush=True)
                try:
                    run_frozen_probes(pretrain_cfg, ckpt_str, frozen_cfg, eval_mode=frozen_mode)
                except Exception as e:
                    print(
                        f"[post_train_eval] frozen probes failed on {ckpt.name}: {e}",
                        flush=True,
                    )
                    _maybe_wandb_finish()
        if want_attentive:
            print(f"[post_train_eval] attentive probe on {ckpt.name}", flush=True)
            try:
                run_attentive_probe(pretrain_cfg, ckpt_str)
            except Exception as e:
                print(
                    f"[post_train_eval] attentive probe failed on {ckpt.name}: {e}",
                    flush=True,
                )
                _maybe_wandb_finish()

    print("[post_train_eval] done", flush=True)
