# Final report

`A_01.pdf` is the submitted report (Section A, Group 1). Its source is
`main.tex`, an ACM `acmart` (sigconf) document.

To build it, run this from the `report/` folder:

```
latexmk -pdf main.tex
```

Running `pdflatex main.tex` twice works too; the second pass resolves the
references. The figures are read from `../figures/`, which are committed, so
the report builds without running any experiment. To rebuild a figure from
the result tables, see the main README (`scripts/make_report_figures.py`).

The project proposal is kept for reference in `docs/`.
