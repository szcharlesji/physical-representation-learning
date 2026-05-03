"""Fetch eval-probe metrics from wandb and write tidy CSVs for plotting.

Outputs (under repo `data/`):
  - <label>/curves.csv          per-experiment wide-format curves
  - all_runs_long.csv           tidy long-format, plot-ready
  - best_summary.csv            one row per (label, probe) with best/best-epoch
  - .cache/<run_id>.json        cached scan_history payloads (run with --refresh to bust)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import wandb


ENTITY = "ry2665-new-york-university"
PROJECT = "physics-jepa-baseline"
DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / ".cache"


@dataclass(frozen=True)
class EvalCurveRun:
    label: str
    run_id: str
    note: str  # human description for provenance


EVAL_CURVE_RUNS: list[EvalCurveRun] = [
    EvalCurveRun("vicreg_bs2", "ztffyvmc",
                 "VICReg, bilinear, batch=2 fp32; original baseline (poor)"),
    EvalCurveRun("vicreg_bs8", "6mhmw4k6",
                 "VICReg, bilinear, batch=8 bf16; improved baseline"),
    EvalCurveRun("fft", "5b2obx5w",
                 "VICReg, FFT preprocessing, batch=8 bf16"),
    EvalCurveRun("cnn_attn", "ih2xhy8w",
                 "VICReg, FFT, conv3d_next_attn (attn at stage 4)"),
    EvalCurveRun("sigreg", "wbo7xqs0",
                 "SIGReg regularizer, FFT; did not converge"),
]

# ViT3D has no eval-curve run (all crashed). Reconstruct from per-checkpoint
# probe_frozen runs in this group.
VIT3D_GROUP = "active_matter-16frames-vit3d-jepa-vit3d-depth6_2026-04-25_11-11-48"
VIT3D_LABEL = "vit3d"

# Conv+Attn ×6 (deeper attention head): 6 transformer blocks at the end of
# the conv stack. HPC-quota cut training short at epoch 5; only 3 frozen
# probe_frozen runs exist (one per saved checkpoint) — listed by run id
# rather than scanned by group, since the runs are scattered across two
# launch attempts and group filtering picks up extras.
CNN_ATTN_D6_LABEL = "cnn_attn_d6"
CNN_ATTN_D6_RUNS: list[tuple[int, str]] = [
    (1, "hfnzs0zr"),
    (3, "0mguarjj"),
    (5, "zled9non"),
    (7, "j1bl1ph4"),
]

EVAL_CURVE_METRIC_KEYS = [
    "epoch",
    "linear/mean", "linear/alpha", "linear/zeta",
    "knn/mean", "knn/alpha", "knn/zeta",
    "knn/best_k", "knn/best_metric_is_cosine",
    "attentive/mean", "attentive/alpha", "attentive/zeta",
]


def _api():
    return wandb.Api(timeout=60)


def _cache_path(run_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{run_id}.json"


def fetch_eval_curve(run_id: str, refresh: bool) -> tuple[list[dict], dict]:
    """Return (history_rows, summary_dict) for an eval-curve run."""
    cache = _cache_path(run_id)
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text())
        return payload["history"], payload["summary"]
    run = _api().run(f"{ENTITY}/{PROJECT}/{run_id}")
    # Don't pass `keys=` — wandb requires every row to contain every listed key,
    # so a run that's missing one metric (e.g. SIGReg has no attentive curve)
    # returns zero rows. Pulling everything and filtering client-side is fine
    # because eval-curve runs only log ~15 rows.
    history_full = list(run.scan_history())
    history = [{k: row[k] for k in EVAL_CURVE_METRIC_KEYS if k in row}
               for row in history_full]
    history = [r for r in history if "epoch" in r]
    # Coerce summary to plain JSON-safe dict (drop nested `_wandb` etc.)
    summary = {k: v for k, v in dict(run.summary).items()
               if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
    cache.write_text(json.dumps({"history": history, "summary": summary}))
    return history, summary


def fetch_vit3d_per_ckpt(refresh: bool) -> tuple[list[dict], dict]:
    """Reconstruct a curve-shaped dict-list from the 15 ViT3D probe_frozen runs.

    Returns (history_rows_keyed_by_epoch, summary_dict).
    Summary mirrors eval-curve format (linear/best_mean, knn/best_mean, ...).
    """
    cache = _cache_path(f"{VIT3D_LABEL}_per_ckpt")
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text())
        return payload["history"], payload["summary"]

    api = _api()
    runs = api.runs(f"{ENTITY}/{PROJECT}",
                    filters={"group": VIT3D_GROUP, "jobType": "probe_frozen",
                             "state": "finished"})
    name_re = re.compile(r"-ViT3DEncoder_(\d+)$")
    rows: list[dict] = []
    for run in runs:
        m = name_re.search(run.name)
        if not m:
            continue
        epoch = int(m.group(1))
        s = run.summary
        rows.append({
            "epoch": epoch,
            "linear/mean": _f(s.get("linear/val_mse")),
            "linear/alpha": _f(s.get("linear/val_mse_alpha")),
            "linear/zeta": _f(s.get("linear/val_mse_zeta")),
            "knn/mean": _f(s.get("knn/best_val_mse")),
            "knn/alpha": _f(s.get("knn/best_val_mse_alpha")),
            "knn/zeta": _f(s.get("knn/best_val_mse_zeta")),
            "knn/best_k": _f(s.get("knn/best_k")),
            "knn/best_metric_is_cosine": _bool_to_int(s.get("knn/best_metric")),
            # Attentive crashed for ViT3D — leave NaN.
            "attentive/mean": None,
            "attentive/alpha": None,
            "attentive/zeta": None,
        })
    rows.sort(key=lambda r: r["epoch"])

    df = pd.DataFrame(rows)
    summary = {}
    for probe in ("linear", "knn", "attentive"):
        col = f"{probe}/mean"
        if col in df and df[col].notna().any():
            best_idx = df[col].idxmin()
            summary[f"{probe}/best_mean"] = float(df.loc[best_idx, col])
            summary[f"{probe}/best_alpha"] = _opt_float(df.loc[best_idx, f"{probe}/alpha"])
            summary[f"{probe}/best_zeta"] = _opt_float(df.loc[best_idx, f"{probe}/zeta"])
            summary[f"{probe}/best_epoch"] = int(df.loc[best_idx, "epoch"])
    cache.write_text(json.dumps({"history": rows, "summary": summary}))
    return rows, summary


def fetch_cnn_attn_d6_per_ckpt(refresh: bool) -> tuple[list[dict], dict]:
    """Reconstruct a curve for the conv+attn×6 group from 3 probe_frozen runs.

    Same shape as `fetch_vit3d_per_ckpt`, but the (epoch, run_id) mapping is
    explicit instead of group-scanned. Each run's summary is read for any
    linear/* and knn/* keys; missing keys come back as None and are dropped
    by `dropna(subset=["mse"])` downstream.
    """
    cache = _cache_path(f"{CNN_ATTN_D6_LABEL}_per_ckpt")
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text())
        return payload["history"], payload["summary"]

    api = _api()
    rows: list[dict] = []
    for epoch, run_id in CNN_ATTN_D6_RUNS:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        s = run.summary
        rows.append({
            "epoch": epoch,
            "linear/mean": _f(s.get("linear/val_mse")),
            "linear/alpha": _f(s.get("linear/val_mse_alpha")),
            "linear/zeta": _f(s.get("linear/val_mse_zeta")),
            "knn/mean": _f(s.get("knn/best_val_mse")),
            "knn/alpha": _f(s.get("knn/best_val_mse_alpha")),
            "knn/zeta": _f(s.get("knn/best_val_mse_zeta")),
            "knn/best_k": _f(s.get("knn/best_k")),
            "knn/best_metric_is_cosine": _bool_to_int(s.get("knn/best_metric")),
            "attentive/mean": None,
            "attentive/alpha": None,
            "attentive/zeta": None,
        })
    rows.sort(key=lambda r: r["epoch"])

    df = pd.DataFrame(rows)
    summary = {}
    for probe in ("linear", "knn", "attentive"):
        col = f"{probe}/mean"
        if col in df and df[col].notna().any():
            best_idx = df[col].idxmin()
            summary[f"{probe}/best_mean"] = float(df.loc[best_idx, col])
            summary[f"{probe}/best_alpha"] = _opt_float(df.loc[best_idx, f"{probe}/alpha"])
            summary[f"{probe}/best_zeta"] = _opt_float(df.loc[best_idx, f"{probe}/zeta"])
            summary[f"{probe}/best_epoch"] = int(df.loc[best_idx, "epoch"])
    cache.write_text(json.dumps({"history": rows, "summary": summary}))
    return rows, summary


def _f(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _opt_float(x):
    return None if pd.isna(x) else float(x)


def _bool_to_int(metric_name):
    if metric_name is None:
        return None
    return 1 if str(metric_name).lower() == "cosine" else 0


def history_to_long(label: str, history: list[dict]) -> pd.DataFrame:
    """Reshape a curve dict-list into long format: (label, epoch, probe, param, mse)."""
    df = pd.DataFrame(history)
    if df.empty:
        return df
    df = df.sort_values("epoch").reset_index(drop=True)
    pieces = []
    for probe in ("linear", "knn", "attentive"):
        for param in ("mean", "alpha", "zeta"):
            col = f"{probe}/{param}"
            if col not in df.columns:
                continue
            sub = df[["epoch", col]].rename(columns={col: "mse"}).copy()
            sub["label"] = label
            sub["probe"] = probe
            sub["param"] = param
            pieces.append(sub[["label", "epoch", "probe", "param", "mse"]])
    out = pd.concat(pieces, ignore_index=True)
    return out.dropna(subset=["mse"])


def write_curves_csv(label: str, history: list[dict]) -> Path:
    df = pd.DataFrame(history).sort_values("epoch").reset_index(drop=True)
    out = DATA_DIR / label / "curves.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def summary_rows(label: str, summary: dict) -> list[dict]:
    rows = []
    for probe in ("linear", "knn", "attentive"):
        if f"{probe}/best_mean" not in summary:
            continue
        rows.append({
            "label": label,
            "probe": probe,
            "best_mean": _f(summary.get(f"{probe}/best_mean")),
            "best_alpha": _f(summary.get(f"{probe}/best_alpha")),
            "best_zeta": _f(summary.get(f"{probe}/best_zeta")),
            "best_epoch": _f(summary.get(f"{probe}/best_epoch")),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cached payloads and re-query wandb.")
    args = parser.parse_args()

    long_frames: list[pd.DataFrame] = []
    summary_records: list[dict] = []

    for spec in EVAL_CURVE_RUNS:
        print(f"[fetch] {spec.label} ({spec.run_id}) — {spec.note}")
        history, summary = fetch_eval_curve(spec.run_id, args.refresh)
        path = write_curves_csv(spec.label, history)
        print(f"        wrote {path.relative_to(DATA_DIR)} ({len(history)} rows)")
        long_frames.append(history_to_long(spec.label, history))
        summary_records.extend(summary_rows(spec.label, summary))

    print(f"[fetch] {VIT3D_LABEL} (per-checkpoint reconstruction)")
    history, summary = fetch_vit3d_per_ckpt(args.refresh)
    path = write_curves_csv(VIT3D_LABEL, history)
    print(f"        wrote {path.relative_to(DATA_DIR)} ({len(history)} rows)")
    long_frames.append(history_to_long(VIT3D_LABEL, history))
    summary_records.extend(summary_rows(VIT3D_LABEL, summary))

    print(f"[fetch] {CNN_ATTN_D6_LABEL} (per-checkpoint reconstruction, 3 runs)")
    history, summary = fetch_cnn_attn_d6_per_ckpt(args.refresh)
    path = write_curves_csv(CNN_ATTN_D6_LABEL, history)
    print(f"        wrote {path.relative_to(DATA_DIR)} ({len(history)} rows)")
    long_frames.append(history_to_long(CNN_ATTN_D6_LABEL, history))
    summary_records.extend(summary_rows(CNN_ATTN_D6_LABEL, summary))

    long = pd.concat(long_frames, ignore_index=True)
    long_path = DATA_DIR / "all_runs_long.csv"
    long.to_csv(long_path, index=False)
    print(f"[fetch] wrote {long_path.relative_to(DATA_DIR)} ({len(long)} rows)")

    summary_df = pd.DataFrame(summary_records)
    summary_path = DATA_DIR / "best_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[fetch] wrote {summary_path.relative_to(DATA_DIR)} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
