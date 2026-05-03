"""Render report figures from `data/all_runs_long.csv` + `data/best_summary.csv`.

Outputs (PDF + PNG) to `data/figures/`:
  fig_linear_mean        linear/mean MSE vs epoch, all configs overlaid
  fig_knn_mean           knn/mean MSE vs epoch, all configs overlaid
  fig_attentive_mean     attentive/mean MSE vs epoch (no vit3d)
  fig_per_param_best     linear alpha | zeta side-by-side
  fig_best_summary_bars  best_mean across configs, grouped by probe
  fig_three_probes_bs8   single-config breakdown of vicreg_bs8 (anatomy)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
FIG_DIR = DATA_DIR / "figures"

plt.style.use(DATA_DIR / "presets.mplstyle")


# Display order, color and linestyle for each label
LABELS = [
    ("vicreg_bs2",   "VICReg, bs=2 (fp32)",         "#888888", "--", "o"),
    ("vicreg_bs8",   "VICReg, bs=8 (bf16)",         "#1f77b4", "-",  "o"),
    ("fft",          "VICReg + FFT preproc",        "#ff7f0e", "-",  "o"),
    ("cnn_attn",     "Conv+Attn (FFT)",             "#2ca02c", "-",  "o"),
    ("cnn_attn_d6",  "Conv+Attn ×6 (FFT, 5 ep)",    "#17becf", "-",  "^"),
    ("sigreg",       "SIGReg (FFT, did not conv.)", "#d62728", ":",  "o"),
    ("vit3d",        "ViT3D-d6 (FFT)",              "#9467bd", "-",  "s"),
]
LABEL_ORDER = [t[0] for t in LABELS]
STYLE = {t[0]: dict(label=t[1], color=t[2], linestyle=t[3], marker=t[4])
         for t in LABELS}


BASELINE_LABEL = "vicreg_bs8"  # what the user calls "the baseline" (improved CNN baseline)


def _annotate_min(ax: plt.Axes, df: pd.DataFrame, probe: str, param: str,
                  label: str, prefix: str, xytext: tuple[int, int]) -> None:
    """Mark and label the minimum point of (label, probe, param)."""
    sub = df[(df["probe"] == probe) & (df["param"] == param) & (df["label"] == label)]
    if sub.empty:
        return
    idx = sub["mse"].idxmin()
    x = float(sub.loc[idx, "epoch"])
    y = float(sub.loc[idx, "mse"])
    color = STYLE[label]["color"]
    ax.scatter([x], [y], s=110, facecolors="none", edgecolors=color,
               linewidths=2.0, zorder=5)
    ax.annotate(f"{prefix}: {y:.3f} @ ep{int(x)}",
                xy=(x, y), xytext=xytext, textcoords="offset points",
                fontsize=9, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.85),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color,
                          lw=0.8, alpha=0.92),
                zorder=6)


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path)
    print(f"[plot] wrote figures/{name}.{{pdf,png}}")
    plt.close(fig)


def _plot_curves(df: pd.DataFrame, probe: str, param: str,
                 ax: plt.Axes, include_labels: list[str] | None = None) -> None:
    """Overlay all (probe, param) curves on `ax`."""
    sub = df[(df["probe"] == probe) & (df["param"] == param)]
    labels = include_labels if include_labels is not None else LABEL_ORDER
    for lbl in labels:
        s = STYLE[lbl]
        d = sub[sub["label"] == lbl].sort_values("epoch")
        if d.empty:
            continue
        ax.plot(d["epoch"], d["mse"],
                color=s["color"], linestyle=s["linestyle"],
                marker=s["marker"], markersize=4, label=s["label"])
    ax.set_yscale("log")
    ax.set_xlabel("Pretrain epoch")
    ax.set_ylabel(f"{probe}/{param} MSE (log)")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")


def _overall_best_label(df: pd.DataFrame, probe: str, param: str,
                        candidates: list[str]) -> str | None:
    """Return the label among `candidates` whose curve has the lowest minimum."""
    best_lbl, best_val = None, float("inf")
    sub = df[(df["probe"] == probe) & (df["param"] == param)]
    for lbl in candidates:
        d = sub[sub["label"] == lbl]
        if d.empty:
            continue
        m = d["mse"].min()
        if m < best_val:
            best_lbl, best_val = lbl, m
    return best_lbl


def fig_probe_mean(df: pd.DataFrame, probe: str, fname: str,
                   include_labels: list[str] | None = None,
                   title_suffix: str = "",
                   baseline_xy: tuple[int, int] = (15, 28),
                   overall_xy: tuple[int, int] = (-110, 18)) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    _plot_curves(df, probe, "mean", ax, include_labels=include_labels)
    ax.set_title(f"{probe.capitalize()} probe — mean MSE vs pretrain epoch{title_suffix}")

    candidates = include_labels if include_labels is not None else LABEL_ORDER
    overall = _overall_best_label(df, probe, "mean", candidates)
    if BASELINE_LABEL in candidates:
        _annotate_min(ax, df, probe, "mean", BASELINE_LABEL, "baseline best",
                      xytext=baseline_xy)
    if overall is not None and overall != BASELINE_LABEL:
        _annotate_min(ax, df, probe, "mean", overall,
                      f"overall best ({STYLE[overall]['label'].split(' ')[0]})",
                      xytext=overall_xy)
    ax.legend(loc="upper right")
    _save(fig, fname)


def fig_per_param_best(df: pd.DataFrame) -> None:
    """linear/{alpha, zeta} stacked. Annotates baseline + overall minima per panel."""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    for ax, param in zip(axes, ("alpha", "zeta")):
        _plot_curves(df, "linear", param, ax)
        ax.set_title(f"linear/{param}")
        overall = _overall_best_label(df, "linear", param, LABEL_ORDER)
        _annotate_min(ax, df, "linear", param, BASELINE_LABEL,
                      "baseline best", xytext=(20, 30))
        if overall is not None and overall != BASELINE_LABEL:
            _annotate_min(ax, df, "linear", param, overall,
                          f"overall best ({STYLE[overall]['label'].split(' ')[0]})",
                          xytext=(40, 30))
    axes[1].set_xlabel("Pretrain epoch")
    axes[0].set_xlabel("")
    axes[0].legend(loc="upper right")
    fig.suptitle("Per-parameter linear-probe MSE — alpha is easier than zeta",
                 y=0.995)
    _save(fig, "fig_per_param_best")


def fig_best_summary_bars(summary: pd.DataFrame) -> None:
    """Grouped bar chart: x=label, hue=probe, y=best_mean. Linear y-axis."""
    pivot = summary.pivot_table(index="label", columns="probe",
                                values="best_mean").reindex(LABEL_ORDER)
    probes = ["linear", "knn", "attentive"]
    pivot = pivot[probes]
    n_labels = len(pivot)
    n_probes = len(probes)
    bar_w = 0.25
    x = np.arange(n_labels)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    probe_color = {"linear": "#4c72b0", "knn": "#dd8452", "attentive": "#55a868"}
    for i, p in enumerate(probes):
        vals = pivot[p].to_numpy()
        offsets = (i - (n_probes - 1) / 2) * bar_w
        bars = ax.bar(x + offsets, vals, bar_w, color=probe_color[p],
                      label=p.capitalize())
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[j] + offsets, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([STYLE[l]["label"].replace(", ", "\n") for l in pivot.index],
                       rotation=0, fontsize=8)
    ax.set_ylabel("best mean MSE")
    ax.set_title("Best validation MSE across configs and probes (lower is better)")
    ax.legend(loc="upper left")
    _save(fig, "fig_best_summary_bars")


def fig_three_probes_bs8(df: pd.DataFrame) -> None:
    """Single config breakdown — vicreg_bs8, the canonical CNN baseline."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sub = df[(df["label"] == "vicreg_bs8") & (df["param"] == "mean")]
    probe_color = {"linear": "#4c72b0", "knn": "#dd8452", "attentive": "#55a868"}
    for probe in ("linear", "knn", "attentive"):
        d = sub[sub["probe"] == probe].sort_values("epoch")
        ax.plot(d["epoch"], d["mse"], marker="o", color=probe_color[probe],
                label=probe.capitalize())
    ax.set_yscale("log")
    ax.set_xlabel("Pretrain epoch")
    ax.set_ylabel("mean MSE (log)")
    ax.set_title("VICReg bs=8 (bf16) — anatomy of an eval-curve run")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    _save(fig, "fig_three_probes_bs8")


def main():
    df = pd.read_csv(DATA_DIR / "all_runs_long.csv")
    summary = pd.read_csv(DATA_DIR / "best_summary.csv")

    # Per-plot annotation offsets, hand-tuned to avoid clipping/overlap.
    fig_probe_mean(df, "linear",  "fig_linear_mean",
                   baseline_xy=(20, 30), overall_xy=(40, 35))
    fig_probe_mean(df, "knn",     "fig_knn_mean",
                   baseline_xy=(-170, 28), overall_xy=(-200, 12))
    # ViT3D's attentive probe crashed → omit from this figure.
    fig_probe_mean(df, "attentive", "fig_attentive_mean",
                   include_labels=[l for l in LABEL_ORDER if l != "vit3d"],
                   title_suffix=" (ViT3D omitted: probe crashed)",
                   baseline_xy=(20, 30), overall_xy=(-170, 18))

    fig_per_param_best(df)
    fig_best_summary_bars(summary)
    fig_three_probes_bs8(df)


if __name__ == "__main__":
    main()
