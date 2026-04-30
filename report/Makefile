.PHONY: pdf clean

pdf: report.pdf

report.pdf: report.tex ../data/figures/fig_linear_mean.pdf \
            ../data/figures/fig_knn_mean.pdf \
            ../data/figures/fig_attentive_mean.pdf \
            ../data/figures/fig_per_param_best.pdf \
            ../data/figures/fig_best_summary_bars.pdf
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	pdflatex -interaction=nonstopmode -halt-on-error report.tex

clean:
	rm -f report.aux report.log report.out report.toc report.pdf
