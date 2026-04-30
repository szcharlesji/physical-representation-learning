# `report/` — LaTeX writeup

Compiles to a 6-page PDF summarising the JEPA pretraining experiments on
`active_matter`. All figures and numbers come from
[`../data/`](../data/README.md); this directory only owns the prose.

## Build

```bash
make pdf      # runs pdflatex twice (needed for cross-references)
make clean    # removes report.pdf and aux/log files
```

Requires `pdflatex` (TeX Live or MacTeX). The Makefile lists the figure
PDFs as prerequisites, so if you regenerate them in `../data/figures/` and
re-run `make pdf` the report will pick them up automatically.

## Updating numbers and figures

Numbers in the prose and Table 1 are hand-typed and need to be edited by
hand if the underlying data changes. To refresh the figures themselves:

```bash
cd ../data && make refresh && cd ../report && make pdf
```

`make refresh` ignores the wandb API cache and re-pulls every run, so use
it after a new pretrain finishes. After it completes, re-check Table 1 in
[`report.tex`](report.tex) against
[`../data/best_summary.csv`](../data/best_summary.csv) — that table is the
only place the numbers live as literals; everywhere else they're rendered
straight onto the figures by the annotation code in `../data/plot.py`.

## Layout

```
report/
├── report.tex   ← single-file source, ~7 sections
├── report.pdf   ← committed for convenience
└── Makefile
```

`\graphicspath{{../data/figures/}}` lets `\includegraphics{fig_*}` resolve
without copying figures into this directory.

## Section map

| § | Title | Figures |
|---|-------|---------|
| 1 | Introduction | — |
| 2 | Experimental setup | — |
| 3 | From the original baseline to a stable training recipe | `fig_linear_mean`, `fig_attentive_mean` |
| 4 | FFT preprocessing replaces the per-dataset crop | `fig_knn_mean` |
| 5 | Architectural variants (Conv+Attn, ViT3D, SIGReg) | (re-uses §3 / §4 figures) |
| 6 | Per-parameter difficulty | `fig_per_param_best` |
| 7 | Summary | `fig_best_summary_bars`, Table 1 |

`fig_three_probes_bs8.pdf` is rendered but not currently embedded — useful
as a "what does an eval-curve run actually look like" appendix figure if
the report grows.

## Editing notes

- All wandb run mentions use direct URLs of the form
  `https://wandb.ai/ry2665-new-york-university/physics-jepa-baseline/runs/<id>`
  via `\href{}{}`. New runs added to `../data/fetch.py` should also be
  added here if they're worth discussing.
- The two custom commands at the top of `report.tex` are
  `\probe{...}` (small caps for probe names) and `\code{...}` (typewriter
  font, alias for `\texttt{...}`).
