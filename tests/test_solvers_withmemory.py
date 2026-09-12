"""Tests for NWM9 and NWM11. Owner: Suchi."""

from __future__ import annotations

import math

import mpmath as mp
import pytest

from conftest import skip_if_unimplemented
from keplerbench.core.registry import get_guess, get_solver
from keplerbench.experiments.runner import solve_one
from keplerbench.solvers._withmemory_base import (
    hermite_derivative_estimate,
    newton_divided_differences,
)


def test_divided_differences_of_a_polynomial_terminate():
    """For a cubic, the 4th divided difference must be zero."""
    with skip_if_unimplemented():
        f = lambda x: 2 * x**3 - x + 5
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        coeffs = newton_divided_differences(xs, [f(x) for x in xs])
        assert coeffs[3] == pytest.approx(2.0)      # leading coefficient
        assert coeffs[4] == pytest.approx(0.0, abs=1e-9)


def test_hermite_derivative_estimate_is_accurate():
    with skip_if_unimplemented():
        f = lambda x: math.exp(x)
        xs = [0.9, 0.95, 1.0, 1.05, 1.1]
        est = hermite_derivative_estimate(xs, [f(x) for x in xs], order=1, at=1.0)
        assert est == pytest.approx(math.e, rel=1e-4)


# ----------------------------------------------------------------------
# Repeated (confluent) nodes.  The published parameter formulas interpolate
# over node lists like [s_k, s_k, t_{k-1}, ...], so a plain Newton table is
# not enough - see the module docstring of _withmemory_base.py.
# ----------------------------------------------------------------------
def test_divided_differences_with_a_repeated_node_use_the_derivative():
    """f[x, x] must equal f'(x), not blow up on a zero denominator."""
    with skip_if_unimplemented():
        f = lambda x: math.exp(x)
        xs = [1.0, 1.0, 2.0]
        fs = [f(1.0), f(1.0), f(2.0)]
        ds = [f(1.0), None, None]          # d/dx exp = exp
        coeffs = newton_divided_differences(xs, fs, ds)
        assert coeffs[0] == pytest.approx(math.e)
        assert coeffs[1] == pytest.approx(math.e)          # f[x0, x0] = f'(x0)
        assert coeffs[2] == pytest.approx(math.e**2 - 2 * math.e)


def test_hermite_interpolant_reproduces_a_cubic_exactly():
    """Four conditions (f, f' at a double node, f at two more) pin a cubic.

    With the interpolant exact, every derivative it reports must be the
    cubic's own - at any point, not just at a node.
    """
    with skip_if_unimplemented():
        f = lambda x: 2 * x**3 - x + 5
        fp = lambda x: 6 * x**2 - 1
        fpp = lambda x: 12 * x
        xs = [0.5, 0.5, 1.5, 2.5]
        fs = [f(x) for x in xs]
        ds = [fp(0.5), None, None, None]

        # Hermite condition at the double node.
        assert hermite_derivative_estimate(
            xs, fs, order=1, at=0.5, ds=ds) == pytest.approx(fp(0.5))
        # Exactness away from every node.
        assert hermite_derivative_estimate(
            xs, fs, order=0, at=1.7, ds=ds) == pytest.approx(f(1.7))
        assert hermite_derivative_estimate(
            xs, fs, order=2, at=1.7, ds=ds) == pytest.approx(fpp(1.7))
        # A cubic has no fourth derivative.
        assert hermite_derivative_estimate(
            xs, fs, order=4, at=1.7, ds=ds) == pytest.approx(0.0, abs=1e-9)


def test_repeated_node_without_a_derivative_fails_loudly():
    """A missing derivative must raise, never silently divide by zero."""
    with skip_if_unimplemented():
        with pytest.raises(ValueError, match="repeated"):
            newton_divided_differences([1.0, 1.0], [2.0, 2.0])


def test_repeated_nodes_must_be_adjacent():
    """[a, b, a] is a different polynomial, not the confluent one."""
    with skip_if_unimplemented():
        with pytest.raises(ValueError, match="adjacent"):
            newton_divided_differences([1.0, 2.0, 1.0], [1.0, 4.0, 1.0],
                                       [1.0, None, 1.0])


def test_divided_differences_work_at_extended_precision():
    """The tables are ill-conditioned once nodes collapse onto the root,
    which is why verification runs in mpmath. The code must be type-agnostic."""
    with skip_if_unimplemented():
        with mp.workdps(80):
            f = mp.exp
            xs = [mp.mpf(1), mp.mpf(1), mp.mpf(2)]
            fs = [f(x) for x in xs]
            ds = [f(mp.mpf(1)), None, None]
            coeffs = newton_divided_differences(xs, fs, ds)
            assert isinstance(coeffs[1], mp.mpf), "precision was silently lost"
            assert abs(coeffs[1] - mp.e) < mp.mpf(10) ** -70


@pytest.mark.parametrize("solver_name", ["nwm9", "nwm11"])
def test_carries_memory_between_iterations(solver_name):
    """A with-memory method must actually accumulate history, and USE it.

    The second assertion is the one that matters: if the accelerating
    parameters stay at their published initial value, the memory is not
    wired up and the method has silently degraded to its memoryless base -
    which would show up in the report as "with-memory does not help", and
    would be wrong.

    (e, M) is chosen away from the root so the iteration takes several steps;
    close to it the sub-steps collapse within one ulp and the method returns
    early by design.
    """
    solver = get_solver(solver_name)
    with skip_if_unimplemented():
        from keplerbench.core.types import KeplerProblem

        p = KeplerProblem(e=0.9, M=0.5)
        state = solver.init_state(p, 0.5)

        E1 = solver.step(p, 0.5, state)
        assert math.isfinite(E1)
        assert state["memory"].has_memory(), "state was not updated by step()"
        first = dict(state["memory"].params)
        assert first, "no accelerating parameters were recorded"

        E2 = solver.step(p, E1, state)
        assert math.isfinite(E2)
        second = state["memory"].params
        assert set(second) == set(first)
        changed = [k for k in first if second[k] != first[k]]
        assert changed, (
            f"parameters {first} unchanged after a second iteration - the "
            "Hermite interpolation is not feeding back into the scheme"
        )


def test_nwm9_memoryless_base_is_order_eight():
    """The two-stage rule, as a test.

    NWM9 is an optimal eighth-order scheme lifted to 8.8989 by one parameter.
    Freezing that parameter at zero must recover exactly 8. If this fails the
    bug is in the base scheme; if this passes and the order test fails, the
    bug is in the memory. Without both, a wrong order is unattributable.
    """
    from keplerbench.experiments.verification import verify_order_on_test_functions
    from keplerbench.solvers.nwm9 import NWM9Solver

    class MemorylessNWM9(NWM9Solver):
        theoretical_order = 8.0

        def _accelerating_parameter(self, *args, **kwargs):
            return 0.0

    with skip_if_unimplemented():
        rows = verify_order_on_test_functions(
            "nwm9", functions=None, n_iterations=8,
        )
        del rows  # ensures the real solver is implemented before we bother
        import keplerbench.core.registry as registry
        registry._SOLVERS["nwm9_memoryless"] = MemorylessNWM9
        try:
            base = verify_order_on_test_functions("nwm9_memoryless", n_iterations=8)
        finally:
            registry._SOLVERS.pop("nwm9_memoryless", None)

    for row in base:
        assert row["measured_order"] == pytest.approx(8.0, abs=1e-3), row


# ----------------------------------------------------------------------
# Error-constant guards.
#
# These pin the transcriptions the PDFs render ambiguously. Measuring the
# convergence order does NOT pin them: several misreadings of both schemes
# still converge at order 8, so order agreement proves almost nothing. The
# papers' error CONSTANTS do discriminate - a wrong grouping in NWM11's
# w'(t_k) missed the e^8 constant by about 60 orders of magnitude while
# still being an order-8 method.
#
# They drive the shipped solver modules through GenericProblem for exactly
# one step, so a change to nwm9.py or nwm11.py breaks them. An earlier
# version recomputed the algebra inline and would have passed no matter what
# those files did - a guard that guards nothing.
# ----------------------------------------------------------------------
CONSTANT_DPS = 400
#: Start this far from the root. The e^8 term is then 1e-192, while the
#: O(e^9) remainder is 1e-216 - so the measured constant is good to ~1e-24
#: and the tolerance below is not fighting truncation error.
CONSTANT_OFFSET = mp.mpf(10) ** -24
#: Worst observed across both solvers and all eight functions is 3.6e-22.
CONSTANT_TOL = mp.mpf("1e-20")


def _paper_functions():
    from keplerbench.experiments.verification import PAPER_TEST_FUNCTIONS

    return PAPER_TEST_FUNCTIONS


def _first_step_constant(solver, f, xi):
    """(x1 - xi) / (x0 - xi)**8 for ONE step of a real solver.

    max_iter=1 with tol=0 forces exactly one iteration, so this is the
    scheme's own eighth-order error constant and nothing else. The parameter
    used is whatever the solver itself supplies on a first step, where there
    is no history yet.
    """
    from keplerbench.experiments.verification import GenericProblem

    problem = GenericProblem(f, lambda x: mp.diff(f, x, 1), dps=CONSTANT_DPS)
    E0 = xi + CONSTANT_OFFSET
    result = solver.solve(problem, E0=E0, tol=mp.mpf(0), max_iter=1,
                          record_history=True, E_reference=xi)
    return (result.E - xi) / (E0 - xi) ** 8


def _taylor_coefficients(f, xi):
    """c_j = f^(j)(xi) / (j! f'(xi)), the papers' notation."""
    A = mp.diff(f, xi, 1)
    return {j: mp.diff(f, xi, j) / (mp.factorial(j) * A) for j in (2, 3, 4)}


@pytest.mark.parametrize("index", range(8))
def test_nwm9_first_step_matches_the_papers_error_constant(index):
    """NWM9 Eq. (7), on the shipped solver with its shipped ALPHA_0.

        e_{n+1} = c2^2 K [ (alpha + c2) K + c4 ] e^8 + O(e^9),
        K = c2 + 5 c2^2 - c3

    Nothing is stubbed out here: this is nwm9.py exactly as it ships, so the
    test also pins ALPHA_0 being wired into the third step. The formula's
    alpha-dependence is independently corroborated by the paper's own limit
    condition - the coefficient vanishes at alpha = -c2 - c4/K, which is
    precisely the limit Section 2 states for alpha_n.
    """
    from keplerbench.experiments.verification import refine_root
    from keplerbench.core.registry import get_solver
    from keplerbench.solvers.nwm9 import ALPHA_0

    name, f, published, _ = _paper_functions()[index]
    with skip_if_unimplemented():
        with mp.workdps(CONSTANT_DPS):
            xi = refine_root(f, published, dps=CONSTANT_DPS)
            c = _taylor_coefficients(f, xi)
            K = c[2] + 5 * c[2] ** 2 - c[3]
            expected = c[2] ** 2 * K * ((mp.mpf(ALPHA_0) + c[2]) * K + c[4])

            measured = _first_step_constant(get_solver("nwm9"), f, xi)
            relative = abs(measured - expected) / abs(expected)
            assert relative < CONSTANT_TOL, (
                f"{name}: measured {mp.nstr(measured, 12)} vs predicted "
                f"{mp.nstr(expected, 12)} (relative {mp.nstr(relative, 4)})"
            )


def test_nwm11_memoryless_base_is_order_eight():
    """Two-stage rule for the bi-parametric scheme: both parameters frozen
    at zero must recover the underlying optimal eighth-order method."""
    from keplerbench.experiments.verification import verify_order_on_test_functions
    from keplerbench.solvers.nwm11 import NWM11Solver

    class MemorylessNWM11(NWM11Solver):
        theoretical_order = 8.0

        def _alpha(self, *args, **kwargs):
            return 0.0, None

        def _beta(self, *args, **kwargs):
            return 0.0

    import keplerbench.core.registry as registry

    with skip_if_unimplemented():
        registry._SOLVERS["nwm11_memoryless"] = MemorylessNWM11
        try:
            rows = verify_order_on_test_functions("nwm11_memoryless", n_iterations=8)
        finally:
            registry._SOLVERS.pop("nwm11_memoryless", None)

    for row in rows:
        assert row["measured_order"] == pytest.approx(8.0, abs=1e-3), row


@pytest.mark.parametrize("index", range(8))
def test_nwm11_first_step_matches_the_papers_error_constant(index):
    """NWM11 Eq. (2.6), driving nwm11.py with both parameters held at zero.

        e_{k+1} = (alpha + c2)^2 c3 (c2 c3 - c4) e^8 + O(e^9)

    Only the two parameter getters are overridden; every line of arithmetic -
    q(v_k), R(s_k,v_k), w'(t_k) and the three sub-steps - is the shipped
    code. Those three auxiliaries are exactly what the PDF renders
    ambiguously, and this single constant pins all of them at once: the wrong
    w' groupings are out by ~60 orders of magnitude here.

    Held at zero rather than at the shipped values because Eq. (2.6) is the
    UNI-parametric constant. beta also enters at e^8 in the bi-parametric
    scheme (2.33) - see the test below - so comparing the shipped solver
    against (2.6) would only agree to O(beta_0), which is far too loose to
    catch a transcription error.
    """
    from keplerbench.experiments.verification import refine_root
    from keplerbench.solvers.nwm11 import NWM11Solver

    class MemorylessNWM11(NWM11Solver):
        def _alpha(self, *args, **kwargs):
            return mp.mpf(0), None

        def _beta(self, *args, **kwargs):
            return mp.mpf(0)

    name, f, published, _ = _paper_functions()[index]
    with skip_if_unimplemented():
        with mp.workdps(CONSTANT_DPS):
            xi = refine_root(f, published, dps=CONSTANT_DPS)
            c = _taylor_coefficients(f, xi)
            expected = c[2] ** 2 * c[3] * (c[2] * c[3] - c[4])

            measured = _first_step_constant(MemorylessNWM11(), f, xi)
            relative = abs(measured - expected) / abs(expected)
            assert relative < CONSTANT_TOL, (
                f"{name}: measured {mp.nstr(measured, 12)} vs predicted "
                f"{mp.nstr(expected, 12)} (relative {mp.nstr(relative, 4)})"
            )


def test_nwm11_beta_reaches_the_eighth_order_term_linearly():
    """beta must be wired into the third step, and enter e^8 at first order.

    beta sits in the denominator of the final sub-step, perturbing it by
    beta*f(t_k) against w' ~ f'(xi). Since f(t_k) ~ e^4, the correction to
    the iterate is O(beta e^8) - so beta shifts the eighth-order constant
    itself, linearly and proportionally to beta.

    This is why the test above pins beta at zero: measured against the
    uni-parametric Eq. (2.6), the shipped solver agrees only to 5e-7..8e-5,
    which looks like a transcription error and is not one.

    Checking linearity rather than a closed form keeps the test independent
    of Eq. (2.32), whose e^8 coefficient the PDF renders too poorly to
    transcribe with confidence.
    """
    from keplerbench.experiments.verification import refine_root
    from keplerbench.solvers.nwm11 import BETA_0, NWM11Solver

    def solver_with_beta(beta):
        class Pinned(NWM11Solver):
            def _alpha(self, *args, **kwargs):
                return mp.mpf(0), None

            def _beta(self, *args, **kwargs):
                return mp.mpf(beta)

        return Pinned()

    name, f, published, _ = _paper_functions()[0]
    with skip_if_unimplemented():
        with mp.workdps(CONSTANT_DPS):
            xi = refine_root(f, published, dps=CONSTANT_DPS)
            at_zero = _first_step_constant(solver_with_beta(0), f, xi)
            at_one = _first_step_constant(solver_with_beta(BETA_0), f, xi)
            at_two = _first_step_constant(solver_with_beta(2 * BETA_0), f, xi)

            first = at_one - at_zero
            second = at_two - at_one
            assert first != 0, "beta never reached the iterate; it is not wired in"
            # Equal increments for equal steps in beta => linear.
            assert abs(second - first) / abs(first) < mp.mpf("1e-10"), (
                f"{name}: beta's effect on the e^8 constant is not linear "
                f"({mp.nstr(first, 8)} then {mp.nstr(second, 8)})"
            )


@pytest.mark.parametrize("solver_name", ["nwm9", "nwm11"])
def test_costs_one_sincos_pair_per_iteration_on_kepler(solver_name):
    """The Section 4.1 claim for both schemes, measured rather than asserted.

    f and f' at the first node come from a single sincos pair; the other two
    nodes need f only. Three distinct points, one pair - which is the whole
    reason these methods might beat Danby on Kepler.

    Note sin_only carries one extra count per iteration that the solver never
    asked for: IterativeSolver.solve calls problem.f(E) itself to compute the
    residual. That harness call must be netted out before these numbers go
    into a cost table, or every method looks one sin more expensive than it is.
    """
    from keplerbench.core.registry import get_guess, get_solver
    from keplerbench.experiments.runner import solve_one

    with skip_if_unimplemented():
        result = solve_one(get_solver(solver_name), get_guess("simple"),
                           e=0.9, M=0.5, tol=1e-14, max_iter=60)
        n = result.iterations
        assert n >= 2
        cost = result.cost
        assert cost["sincos_pairs"] / n == pytest.approx(1.0, abs=0.05)
        assert cost["eval_points"] / n == pytest.approx(3.0, abs=0.6)
        assert cost["synthesised_derivatives"] > 0, "memory never engaged"


def test_nwm9_costs_one_sincos_pair_per_iteration_on_kepler():
    """The Section 4.1 claim, measured rather than asserted.

    f and f' at x_n come from a single sincos pair; y_n and z_n need f only.
    Three distinct points, one pair. Note that sin_only carries one extra
    count per iteration that the solver never asked for: IterativeSolver.solve
    calls problem.f(E) itself to compute the residual. That harness call must
    be netted out before these numbers go into a cost table.
    """
    from keplerbench.core.registry import get_guess, get_solver
    from keplerbench.experiments.runner import solve_one

    with skip_if_unimplemented():
        result = solve_one(get_solver("nwm9"), get_guess("simple"),
                           e=0.9, M=0.5, tol=1e-14, max_iter=60)
        n = result.iterations
        assert n >= 3, "need a few iterations for a meaningful average"
        cost = result.cost
        assert cost["sincos_pairs"] / n == pytest.approx(1.0, abs=0.05)
        # 2 from the solver (y_n, z_n) + 1 from the harness residual call.
        assert cost["sin_only"] / n == pytest.approx(3.0, abs=0.2)
        assert cost["synthesised_derivatives"] > 0, "memory never engaged"


@pytest.mark.parametrize("solver_name", ["nwm9", "nwm11"])
def test_reaches_tolerance_on_kepler(solver_name, sample_points):
    solver = get_solver(solver_name)
    guess = get_guess("simple")
    with skip_if_unimplemented():
        for e, M in sample_points:
            r = solve_one(solver, guess, e=e, M=M, tol=1e-13, max_iter=100)
            assert r.converged, (solver_name, e, M, r.failure)


@pytest.mark.parametrize("solver_name", ["nwm9", "nwm11"])
def test_measured_order_matches_the_paper(solver_name):
    """Work Plan step 2, as a test. This is the gate before the grid run."""
    from keplerbench.experiments.verification import verify_order_on_test_functions

    with skip_if_unimplemented():
        out = verify_order_on_test_functions(solver_name)
        assert out, "no test functions configured - fill PAPER_TEST_FUNCTIONS"
        for row in out:
            assert row["passed"], row


def test_verification_harness_recovers_newtons_order():
    """Calibrate the gate itself against a solver we already trust.

    Newton is fully implemented and provably second order, so if this fails
    the fault is in the verification harness - the adapter, the extended
    precision path, or the order estimator - not in a with-memory solver.
    Without this, a failure of the test above is ambiguous.
    """
    from keplerbench.experiments.verification import verify_order_on_test_functions

    rows = verify_order_on_test_functions("newton", n_iterations=6)
    assert len(rows) == 8, "expected the paper's eight test functions"
    for row in rows:
        assert row["passed"], row
        assert row["measured_order"] == pytest.approx(2.0, abs=1e-6), row
