"""Command-line entry point.

    keplerbench list
    keplerbench solve --solver newton --guess simple --e 0.5 --M 0.3
    keplerbench run verification      --config configs/verification.yaml
    keplerbench run grid              --config configs/grid_benchmark.yaml
    keplerbench run propagation       --config configs/error_propagation.yaml
    keplerbench run montecarlo        --config configs/monte_carlo.yaml

Owner: Anisa. Fully wired - the subcommands will start working as the
underlying experiment modules get implemented.
"""

from __future__ import annotations

import argparse
import sys

from keplerbench.core.registry import get_guess, get_solver, list_guesses, list_solvers
from keplerbench.experiments.runner import solve_one


def _cmd_list(args: argparse.Namespace) -> int:
    print("solvers:", ", ".join(list_solvers()))
    print("guesses:", ", ".join(list_guesses()))
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    """Solve a single (e, M) and print the trace - handy for debugging."""
    solver = get_solver(args.solver)
    guess = get_guess(args.guess)
    result = solve_one(
        solver, guess, e=args.e, M=args.M,
        tol=args.tol, max_iter=args.max_iter, record_history=True,
    )
    print(f"{result.solver} + {result.guess}  e={result.e}  M={result.M}")
    for rec in result.history:
        print(f"  n={rec.iteration:2d}  E={rec.E!r:<22}  |f|={rec.residual:.3e}")
    print(f"  converged={result.converged}  iterations={result.iterations}")
    print(f"  cost={result.cost}")
    if result.failure:
        print(f"  FAILURE: {result.failure}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from keplerbench.experiments import (
        error_propagation,
        grid_benchmark,
        monte_carlo,
        verification,
    )

    modules = {
        "verification": verification,
        "grid": grid_benchmark,
        "propagation": error_propagation,
        "montecarlo": monte_carlo,
    }
    modules[args.experiment].run(args.config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="keplerbench", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="show registered solvers and guesses").set_defaults(
        func=_cmd_list
    )

    s = sub.add_parser("solve", help="one solve, with the full iteration trace")
    s.add_argument("--solver", required=True)
    s.add_argument("--guess", default="simple")
    s.add_argument("--e", type=float, required=True)
    s.add_argument("--M", type=float, required=True)
    s.add_argument("--tol", type=float, default=1e-14)
    s.add_argument("--max-iter", type=int, default=50, dest="max_iter")
    s.set_defaults(func=_cmd_solve)

    r = sub.add_parser("run", help="run one of the four experiments")
    r.add_argument("experiment",
                   choices=["verification", "grid", "propagation", "montecarlo"])
    r.add_argument("--config", required=True)
    r.set_defaults(func=_cmd_run)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
