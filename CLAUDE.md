# CLAUDE.md

Quick-reference context for Claude working in this repo. See `README.md` for user-facing install/usage.

## What this is

Research code for JEPA-style representation learning on PDE-simulation data from [The Well](https://github.com/PolymathicAI/the_well). Three target datasets: `shear_flow`, `rayleigh_benard`, `active_matter`. Trains a convolutional encoder with a VICReg-style objective, then evaluates via linear probe / kNN / attentive probe for physical-parameter regression.

## Layout

- `physics_jepa/` — library
  - `train.py` — pretraining loop (Trainer). Entry module: `python -m physics_jepa.train_jepa <config.yaml>`
  - `finetuner.py` — **attentive probe** (and from-embeddings linear/MLP head). Caches embeddings per-checkpoint in `ft.embeddings_dir`
  - `eval_frozen.py` — **kNN + linear probe** on frozen features. Caches features under `ft.feature_cache_dir`, keyed by SHA of (ckpt, split, dataset, num_frames, resolution, resize_mode, pool)
  - `data.py` — `WellDatasetForJEPA` (primary) and `WellDatasetForMPP` (baseline only). HDF5 shard-scanner + (context,target) windowing
  - `model.py`, `utils/model_utils.py` — `ConvEncoder` (default) and `ConvEncoderViTTiny` (when `model.vit_equivalency: tiny`)
  - `videomae.py`, `baselines/` — baselines (VideoMAE, DISCO, MPP). **Out of scope for most edits.**
- `configs/`
  - `train_*_small.yaml` — pretrain + attentive-probe path (ft: linear). Each has `dataset:`, `model:`, `train:`, `ft:` blocks.
  - `train_activematter_frozen.yaml` — kNN + linear-probe path (ft: frozen). Only exists for active_matter.
  - `dataset/{shear_flow,rayleigh_benard,active_matter}.yaml` — dataset defaults (native res, num_frames, num_chans). `resolution: 224` here; overridden by train configs.
  - `ft/{linear,mlp,frozen}.yaml` — ft-stage presets
- `pretrain.sbatch`, `pretrain_fft.sbatch`, `eval.sbatch`, `attentive.sbatch` — SLURM entrypoints for the HPC. Paths assume `/scratch/$USER/physical-representation-learning` and a Singularity overlay.
- `scripts/<dataset>/run_*.sh` — thin shell wrappers that `source scripts/env_setup.sh` then `torchrun` the relevant module.

## Data pipeline gotchas (`physics_jepa/data.py`)

- On-disk shapes differ from model input. Per-dataset crop in `_build_global_field_schema` and `__getitem__`:
  - shear_flow: 256×512 → crop to **256×256** (x-half)
  - rayleigh_benard: 512×128 → crop to **128×128** (middle strip `h[192:320]`)
  - active_matter: native 256×256 (no crop)
- After crop, optional resize to `cfg.dataset.resolution`. `resize_mode: "bilinear" | "fft" | "none"`:
  - `bilinear` (default): `F.interpolate`
  - `fft`: **bypasses the per-dataset crop** (reads full native) and uses `fft_resize_2d` from `utils/data_utils.py`. Target periodic BCs.
  - `none`: no resize
- Noise (`cfg[stage].noise_std`) is added *after* resize. No per-channel mean/std normalization anywhere on the training path.
- When adding new loader call sites, plumb **every** dataset field: `resolution`, `offset`, `resize_mode`, `noise_std`, `subset_config_path` (train only). Missing any silently defaults — this has bitten us in eval_frozen and finetuner.

## Models

- `conv_small` (default for all three train configs) — pure strided CNN, works at any H,W divisible by the total stride (16).
- `conv_vit_tiny` — has a **hard assert** in `train.py:40` requiring `resolution == 224` and `num_frames == 4`. Don't forget when switching.
- `videomae` — `PatchEmbed.forward` asserts input size matches `img_size` (default 224). Factories in `videomae.py` pass `img_size` via `**kwargs`.

## Running things

- Pretrain (HPC): `sbatch pretrain.sbatch` (defaults to activematter; edit `CONFIG=` to switch) or `sbatch pretrain_fft.sbatch [shearflow|rayleighbenard|activematter]`.
- kNN+linear eval (HPC, active_matter only): `sbatch eval.sbatch /path/to/ConvEncoder_<N>.pth` — uses `train_activematter_frozen.yaml`.
- Attentive probe (HPC, active_matter only): `sbatch attentive.sbatch /path/to/ConvEncoder_<N>.pth` — uses `train_activematter_small.yaml`.
- Local dev: no torch in the system Python on this Mac — code only runs on the HPC inside the Singularity image. Use `python3 -m py_compile <file>` to syntax-check.
- Config overrides via Hydra-style CLI args, e.g. `train.num_epochs=10 dataset.resize_mode=fft`.

## Auto-probe at end of pretraining (`post_train_eval`)

When `cfg.post_train_eval.enabled: true`, `Trainer.train()` iterates every saved `ConvEncoder_*.pth` under the run's checkpoint dir (rank 0 only) and runs each probe listed in `cfg.post_train_eval.probes`. Implementation lives in `physics_jepa/post_train_probes.py`; the hook is at the bottom of `Trainer.train()`.

- **linear / knn** → `FrozenEvaluator` with the `ft` block taken from `cfg.post_train_eval.frozen_config` (currently `configs/train_activematter_frozen.yaml`). `cfg.ft.out_dir` is scoped per-checkpoint (`<out_dir>/<run_dir_name>/<ckpt_stem>/`) so `results.json` doesn't overwrite.
- **attentive** → `JepaFinetuner` with `use_attentive_pooling=true`, reusing the pretrain cfg's `ft` block (which is already configured as an attentive probe in `train_*_small.yaml`).
- `cfg.ft.run_name` is nulled for both paths so each checkpoint gets a unique auto-generated W&B run name (`<group>-probe_<type>-<ckpt_stem>`). All probe runs reuse the pretrain `group` via `group_from_checkpoint`, so they cluster together in the UI.
- The hook pops `LOCAL_RANK`/`RANK`/`WORLD_SIZE` before running probes so `JepaFinetuner`'s `Trainer.__init__` takes the single-GPU branch and doesn't call `ddp_setup()` again.

**Currently wired up only for active_matter**. The `post_train_eval` block only lives in `configs/train_activematter_small.yaml`; shear_flow and rayleigh_benard configs have no block → gate is false → old behavior. Extending to those datasets requires adding matching frozen configs and fixing `FrozenEvaluator.PARAM_NAMES` in [physics_jepa/eval_frozen.py](physics_jepa/eval_frozen.py) (currently hardcoded to active_matter's `["alpha", "zeta"]`).

## W&B

All runs go through `physics_jepa/utils/wandb_utils.py::init_run`. One project, filter-by-tag.

- **project**: env `WANDB_PROJECT` → fallback `physics-jepa-baseline`. FFT and other variants are **tags**, not separate projects.
- **job_type** (how to split probes on the UI): `pretrain` | `probe_linear` | `probe_knn` | `probe_attentive` | `probe_mlp`. `eval_frozen.py` with `eval_mode=linear_and_knn` produces two sequential runs, one per `job_type`.
- **group**: pretrain uses `f"{run_name}_{timestamp}"` (matches the on-disk checkpoint dir). Probes reuse it via `group_from_checkpoint(cfg.ft.trained_model_path)`, which is `Path(ckpt).parent.name`. Pretrain + all its downstream probes cluster together in the W&B UI.
- **tags**: `build_tags` emits `[dataset.name, model.name, resize_mode, objective, backbone, regularizer]`. Filter the sidebar by `vit3d`/`conv3d_next`/`conv3d_next_attn` for architecture, or by `vicreg`/`sigreg` for the regularizer. Dedup keeps duplicate strings (e.g. `model.name=vit3d` and `model.backbone=vit3d` both resolve to one `vit3d` tag).
- **unified probe metric**: every probe path logs `probe/val_mse` (periodic) and sets `summary["probe/best_val_mse"]` so linear/knn/attentive runs overlay directly on a single W&B chart. `eval_frozen` also logs `probe/test_mse_final`. Attentive-probe path (inside `Trainer.training_loop`) only emits these on probe runs (`self.wandb_job_type.startswith("probe_")` and regression task) — pretrain is unaffected.

## Current branches / experiments

- `try-fft` — FFT-based resize replacing the per-dataset crop. All three train configs set `resize_mode: fft` and explicit `resolution: [H, W]`. Checkpoints land in `checkpoints_fft/` via `pretrain_fft.sbatch` + `WANDB_PROJECT=physics-jepa-fft`. When evaluating FFT checkpoints, the eval configs must match (`resize_mode: fft` + matching resolution) — the eval paths now honor `resize_mode`, but `eval.sbatch` points at `train_activematter_frozen.yaml` which is still at 224/bilinear and needs matching FFT overrides if used against an FFT checkpoint.

## Optional pretrain config axes

These knobs are all opt-in and default to no-ops; the three `train_*_small.yaml` baselines don't set any of them, so they keep their current behavior. Three dedicated configs demonstrate each axis in isolation:

- `configs/train_activematter_sigreg.yaml` — `train.regularizer: sigreg` (LeWM-style; `sim_coeff*MSE + bcs_coeff*SIGReg`, defaults `bcs_coeff=0.1, num_slices=1024`)
- `configs/train_activematter_vit3d.yaml` — `model.backbone: vit3d` (3D patch embed + transformer stack; output reshaped to 4D so ConvPredictor still works)
- `configs/train_activematter_convattn.yaml` — `model.backbone: conv3d_next_attn` + `attn_stages: [4]` (self-attention after the last conv stage)

Launch via `sbatch pretrain_experiment.sbatch {sigreg|vit3d|convattn|baseline}`.

### Regularizer (`train.regularizer`)

Single switch in [physics_jepa/train.py](physics_jepa/train.py) `get_model_components`:
- `vicreg` (default) — unchanged `vicreg_loss_3d` path.
- `sigreg` — uses `vicreg_loss_bcs` (sim * MSE + bcs * SIGReg projections/Epps-Pulley). Legacy alias: `model.loss: gaussian_matching`.

### Optimizer (`train.optim`)

`build_optimizer` in [physics_jepa/utils/model_utils.py](physics_jepa/utils/model_utils.py) reads:
- `optim.name: adamw | lion` (default `adamw`; `lion` requires `lion-pytorch`).
- `optim.betas: [b1, b2]` (default `[0.9, 0.95]` for adamw, `[0.9, 0.99]` for lion).
Also exposes `train.grad_clip_norm` (default None → no clipping) and new LR schedulers `linear` / `constant` alongside `cosine`.

### Data / preprocessing

- `dataset.normalize: none | per_channel_zscore` (default `none`). Stats cached at `cache_path/norm_stats_<dataset>_<sha>.pt`. When set, probes (`eval_frozen.py`, `finetuner.py` raw loaders) automatically apply the same stats so the frozen encoder sees matched inputs.
- `train.augment` (and `ft.augment`) block, handled by `physics_jepa/utils/aug.py::SampleAugmenter`. Supported knobs:
  - `noise_std` (preferred over the bare `train.noise_std` when augment block is present)
  - `channel_dropout_p`
  - `rotations: [0, 90, 180, 270]` subset (uniform sample)
  - `reflections: true|false`
  - `translations_px: int` (gated by `periodic_bcs`; per-dataset default in `_DATASET_PERIODIC_BCS` at [physics_jepa/data.py](physics_jepa/data.py): active_matter=true, shear_flow=true, rayleigh_benard=false).
  All geometric/channel transforms are drawn once per sample and applied identically to ctx and tgt. Noise is drawn independently per side.
- If the `augment` block is absent, the dataset applies only `cfg[stage].noise_std` (preserving the bare-noise behavior of the baseline configs).

### Encoder backbones (`model.backbone`)

Selected by `build_encoder` in [physics_jepa/model.py](physics_jepa/model.py):
- `conv3d_next` (default; omitting the key gives this) — unchanged `ConvEncoder`.
- `conv3d_next_attn` — same `ConvEncoder` with `attn_stages: [idx, ...]` inserting a transformer `Block` after the ResidualBlock stack at each listed stage. `attn_stages=[]` is structurally identical to `conv3d_next`.
- `vit3d` — new `ViT3DEncoder` (3D patch embed → N transformer blocks → time-collapsed (B, D, H', W') output). Config under `model.vit3d` (patch_size, embed_dim, depth, num_heads, mlp_ratio). Note: `model.dims[-1]` should equal the ViT `embed_dim` so `ConvPredictor(dims=reversed(encoder.dims)[:2])` receives matching channels.

### Known limitations

- `model.{patch_size, strides, norm}` knobs for `conv3d_next` are not wired; only `attn_stages` is exposed for that backbone.
- `train.early_stop.enabled` raises `NotImplementedError` — the periodic in-training probe needed to drive it isn't wired.
- Probe auto-run is active_matter only; `FrozenEvaluator.PARAM_NAMES` is hardcoded to `["alpha", "zeta"]`. Extending to shear_flow / rayleigh_benard requires swapping that for `get_dataset_metadata(...).constant_scalar_names`.

## Editing etiquette

- Don't edit `configs/dataset/*.yaml` for experiment-level changes — those are defaults consumed by FT/eval/baselines too. Override in the experiment's train config instead.
- Don't edit baselines (`videomae.py`, `baselines/*`, `train_*_videomae.yaml`) unless the task explicitly names them.
- Don't edit `WellDatasetForMPP` — it's used only by the MPP baseline.
