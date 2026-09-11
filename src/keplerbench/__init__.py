"""keplerbench - benchmark harness for Newton-type solvers of Kepler's equation.

Project: "Rooting for Kepler" (CSE 402 Numerical Methods).

The package is organised so that every solver and every starting guess plugs
into one shared pipeline, which makes the comparison fair:

    core/        the shared contract: problem, cost counter, result objects
    guesses/     starting-guess strategies (the "guess layer" factor)
    solvers/     the five root finders being compared
    reference/   high-precision reference roots (ground truth)
    experiments/ the four studies from the Work Plan
    evaluation/  metrics computed from raw run records
    rv/          radial-velocity / downstream error-propagation model
    plotting/    figures for the report
    io/          config loading and result files
"""

__version__ = "0.1.0"
