"""Figures for the report and the presentation.

Ownership is split so each person plots what they compute:

  style.py              shared look, colours, save helper   (Anisa)
  convergence_plots.py  residual histories, measured order  (Suchi)
  grid_plots.py         (e, M) heatmaps, robustness maps    (Mahdi)
  cost_plots.py         cost accounting, cost vs accuracy   (Dipit)
  propagation_plots.py  tolerance -> parameter shift        (Fariha)

Rule: a plotting function NEVER computes a metric. It takes a DataFrame that
evaluation/ produced and draws it. That way a figure can always be traced to
a result file.
"""
