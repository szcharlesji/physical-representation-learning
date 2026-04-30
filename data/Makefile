PY ?= ../.venv-data/bin/python

.PHONY: all fetch plots refresh clean

all: fetch plots

fetch:
	$(PY) fetch.py

refresh:
	$(PY) fetch.py --refresh

plots:
	$(PY) plot.py

clean:
	rm -rf figures/ .cache/ \
	       vicreg_bs2/ vicreg_bs8/ fft/ cnn_attn/ sigreg/ vit3d/ \
	       all_runs_long.csv best_summary.csv
