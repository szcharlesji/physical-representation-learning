"""Centralized wandb init for pretrain + probe runs.

Every run goes to a single project (default `physics-jepa-baseline`,
overridable via the WANDB_PROJECT env var) and carries a `job_type` plus a
shared `group` so a pretrain + its downstream probes cluster together.
"""
import os
from typing import Iterable, Optional

import wandb
from omegaconf import OmegaConf

DEFAULT_PROJECT = "physics-jepa-baseline"

# Allowed values for job_type; kept as a constant so callers don't drift.
JOB_TYPES = ("pretrain", "probe_linear", "probe_knn", "probe_attentive", "probe_mlp")


def build_tags(cfg, extra: Optional[Iterable[str]] = None) -> list:
    tags = []
    dataset_name = cfg.get("dataset", {}).get("name") if cfg is not None else None
    model_name = cfg.get("model", {}).get("name") if cfg is not None else None
    resize_mode = cfg.get("dataset", {}).get("resize_mode", "bilinear") if cfg is not None else None
    for t in (dataset_name, model_name, resize_mode):
        if t:
            tags.append(str(t))
    if extra:
        for t in extra:
            if t:
                tags.append(str(t))
    # de-dup while preserving order
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def init_run(
    cfg,
    *,
    job_type: str,
    group: str,
    name: str,
    tags: Optional[Iterable[str]] = None,
    extra_config: Optional[dict] = None,
    resume: str = "allow",
):
    """Initialize a wandb run with the project conventions above.

    `cfg` is the OmegaConf config (or None). If cfg has `dry_run=True` this is
    a no-op and returns None.
    """
    if cfg is not None and cfg.get("dry_run", False):
        return None

    if job_type not in JOB_TYPES:
        raise ValueError(f"unknown job_type {job_type!r}; expected one of {JOB_TYPES}")

    project = os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)
    config_payload = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
    if extra_config:
        config_payload = {**config_payload, **extra_config}

    resolved_tags = build_tags(cfg, extra=tags)

    return wandb.init(
        project=project,
        name=name,
        group=group,
        job_type=job_type,
        tags=resolved_tags,
        config=config_payload,
        resume=resume,
    )


def group_from_checkpoint(checkpoint_path: Optional[str]) -> str:
    """Derive a W&B group from a pretrain checkpoint path.

    Convention: pretrain writes checkpoints into
    `<out_path>/<run_name>_<timestamp>/ConvEncoder_<epoch>.pth`, so the
    parent dir name uniquely identifies the pretrain instance.
    """
    if checkpoint_path is None:
        return "randominit"
    from pathlib import Path
    return Path(checkpoint_path).parent.name
