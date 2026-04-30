# `data/` — wandb metrics → tidy CSVs → figures

Pulls eval-probe metrics from
[ry2665-new-york-university/physics-jepa-baseline](https://wandb.ai/ry2665-new-york-university/physics-jepa-baseline)
and renders the figures used by [`../report/report.tex`](../report/report.tex).
This directory is the single source of truth for every number in the report.

## One-time setup

The system Python on this Mac doesn't have `pandas`/`matplotlib`. Create a
local venv at the repo root:

```bash
cd ..                                                  # repo root
python3 -m venv .venv-data
.venv-data/bin/pip install wandb pandas matplotlib numpy
```

You also need to be logged into the wandb CLI: `wandb login` (uses
`~/.netrc`).

## Reproduce everything

From this directory:

```bash
make all          # fetch (cached) + plots
make refresh      # bust the API cache and re-fetch
make clean        # nuke generated CSVs and figures
```

`make all` is idempotent and reads from `.cache/<run_id>.json` on the second
run, so it costs ~1 second after the first call. Pass `--refresh` (or use
`make refresh`) to ignore the cache and re-query wandb.

## Layout

```
data/
├── fetch.py                 ← Python wandb SDK → CSVs
├── plot.py                  ← matplotlib → PDF/PNG figures
├── presets.mplstyle         ← shared matplotlib style
├── Makefile                 ← thin wrapper around the two scripts
├── all_runs_long.csv        ← combined long format, plot input
├── best_summary.csv         ← one row per (label, probe) with best/best-epoch
├── <label>/curves.csv       ← per-experiment wide format (one folder each)
├── .cache/<run_id>.json     ← raw scan_history payloads (gitignored)
└── figures/fig_*.{pdf,png}  ← 6 figures embedded by the report
```

## Run map

| Label                | wandb run id | description |
|----------------------|--------------|-------------|
| `vicreg_bs2`         | [`ztffyvmc`](https://wandb.ai/ry2665-new-york-university/physics-jepa-baseline/runs/ztffyvmc) | VICReg, bilinear, bs=2 fp32 (original baseline, poor) |
| `vicreg_bs8`         | [`6mhmw4k6`](https://wandb.ai/ry2665-new-york-university/physics-jepa-baseline/runs/6mhmw4k6) | VICReg, bilinear, bs=8 bf16 (improved baseline) |
| `fft`                | [`5b2obx5w`](https://wandb.ai/ry2665-new-york-university/physics-jepa-baseline/runs/5b2obx5w) | VICReg, FFT preprocessing |
| `cnn_attn`           | [`ih2xhy8w`](https://wandb.ai/ry2665-new-york-university/physics-jepa-baseline/runs/ih2xhy8w) | conv3d_next_attn (attn at stage 4), FFT |
| `sigreg`             | [`wbo7xqs0`](https://wandb.ai/ry2665-new-york-university/physics-jepa-baseline/runs/wbo7xqs0) | SIGReg regulariser, FFT (did not converge) |
| `vit3d`              | per-checkpoint reconstruction | ViT3D depth-6, FFT — eval-curves crashed; reassembled from 15 `probe_frozen` runs in the group |

The first five are wandb `probe_eval_curve` runs that aggregate every
probe at every checkpoint into one tidy log; we just download
`run.scan_history()` for them. ViT3D never produced a valid eval-curve
(quota ran out), so `fetch_vit3d_per_ckpt()` walks the per-checkpoint
`probe_frozen` runs and stitches their summary values into the same
schema. Its attentive entry is `NaN` because the attentive probe crashed.

To add a new run, append an `EvalCurveRun(...)` to `EVAL_CURVE_RUNS` in
[`fetch.py`](fetch.py) and add a styling tuple to `LABELS` in
[`plot.py`](plot.py).

## CSV schemas

**`<label>/curves.csv`** — wide, one row per epoch:

```
epoch, linear/mean, linear/alpha, linear/zeta,
       knn/mean, knn/alpha, knn/zeta, knn/best_k, knn/best_metric_is_cosine,
       attentive/mean, attentive/alpha, attentive/zeta
```

Missing metrics (e.g. SIGReg has no attentive log; ViT3D has no per-param
k-NN) are blank.

**`all_runs_long.csv`** — long, ready for `seaborn`/`pandas` group-by:

```
label, epoch, probe, param, mse
```

`probe ∈ {linear, knn, attentive}`, `param ∈ {mean, alpha, zeta}`.

**`best_summary.csv`** — one row per (label, probe):

```
label, probe, best_mean, best_alpha, best_zeta, best_epoch
```

Read directly from each run's `summary` (the wandb-side aggregate; for ViT3D
we recompute it from the per-checkpoint values).

## Figures

| File | What it shows |
|------|---------------|
| `fig_linear_mean.pdf`       | Linear probe mean MSE vs epoch, all 6 configs overlaid. Annotates baseline best + overall best. |
| `fig_knn_mean.pdf`          | k-NN probe mean MSE vs epoch (best k/metric per epoch). |
| `fig_attentive_mean.pdf`    | Attentive probe mean MSE vs epoch. ViT3D omitted (probe crashed). |
| `fig_per_param_best.pdf`    | Linear probe MSE on α (top) vs ζ (bottom). |
| `fig_best_summary_bars.pdf` | Grouped bars: x = config, hue = probe, y = best mean MSE. |
| `fig_three_probes_bs8.pdf`  | Single-config breakdown of `vicreg_bs8` — anatomy of an eval-curve run. |
