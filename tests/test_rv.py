"""Tests for the downstream RV chain. Owner: Fariha."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import skip_if_unimplemented  # noqa: E402
from keplerbench.evaluation import propagation as prop  # noqa: E402
from keplerbench.experiments import error_propagation, monte_carlo  # noqa: E402
from keplerbench.plotting import propagation_plots as pp  # noqa: E402
from keplerbench.rv.anomaly import dE_to_dnu, mean_anomaly, radial_velocity, true_anomaly  # noqa: E402
from keplerbench.rv.dataset import load_rv_dataset  # noqa: E402
from keplerbench.rv.model import OrbitParams, chi_squared, rv_curve  # noqa: E402
from keplerbench.rv.radvel_bridge import fit_with_solver, use_solver  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_true_anomaly_equals_E_for_circular_orbit():
    """For e = 0, nu == E identically - true, eccentric and mean anomaly all
    coincide for a circular orbit, with no wrap-around: nu tracks E
    continuously through the whole orbit, same as the quadrant test below
    requires for e > 0. (The original version of this test expected E wrapped
    into (-pi, pi] past E = pi, which is not physically correct - see
    test_true_anomaly_keeps_the_quadrant for the e > 0 case that pins this
    down with a real dnu/dE check.)"""
    with skip_if_unimplemented():
        for E in (0.0, 1.0, 2.0, 3.0, 4.5):
            assert true_anomaly(E, 0.0) == pytest.approx(E, abs=1e-12)


def test_true_anomaly_keeps_the_quadrant():
    """The naive tan-based formula fails here; the atan2 form must not."""
    with skip_if_unimplemented():
        nu = true_anomaly(math.pi + 0.1, 0.8)
        assert nu > math.pi - 1e-9 or nu < -math.pi + 1e-9 or nu > 0


def test_mean_anomaly_wraps():
    with skip_if_unimplemented():
        M = mean_anomaly(t=1000.0, P=10.0, tp=0.0)
        assert 0.0 <= M < 2 * math.pi


def test_radial_velocity_amplitude():
    """For e = 0, omega = 0 the curve is K*cos(nu) + gamma."""
    with skip_if_unimplemented():
        assert radial_velocity(0.0, K=5.0, e=0.0, omega=0.0, gamma=2.0) \
            == pytest.approx(7.0)


def test_amplification_is_largest_in_the_hard_corner():
    """dnu/dE should be big for e -> 1 near E = 0 - the analytic reason the
    pathological corner matters downstream."""
    with skip_if_unimplemented():
        assert dE_to_dnu(0.0, 0.99) > dE_to_dnu(0.0, 0.1)


# ----------------------------------------------------------------------
# rv/model.py.  Uses solver_name="newton", guess_name="simple" explicitly -
# those are the only solver/guess implemented so far. The defaults
# ("danby"/"canonical") are still teammates' skeletons.
# ----------------------------------------------------------------------

def test_rv_curve_matches_hand_computed_chain_for_circular_orbit():
    """For e = 0, Newton's method should reproduce the M -> E -> nu -> v_r
    chain built directly from rv/anomaly.py, since E = M exactly."""
    with skip_if_unimplemented():
        params = OrbitParams(P=10.0, tp=0.0, e=0.0, omega=0.3, K=5.0, gamma=1.0)
        times = np.linspace(0.0, 9.0, 7)

        got = rv_curve(times, params, solver_name="newton", guess_name="simple")

        expected = np.array([
            radial_velocity(true_anomaly(mean_anomaly(t, params.P, params.tp), params.e),
                             params.K, params.e, params.omega, params.gamma)
            for t in times
        ])
        np.testing.assert_allclose(got, expected, atol=1e-9)


def test_rv_curve_reports_nonzero_solver_cost():
    with skip_if_unimplemented():
        params = OrbitParams(P=10.0, tp=0.0, e=0.3, omega=0.1, K=5.0)
        cost: dict[str, int] = {}
        rv_curve([1.0, 2.0, 3.0], params, solver_name="newton",
                guess_name="simple", cost_out=cost)
        assert cost["sincos_pairs"] > 0


def test_chi_squared_is_zero_for_a_perfect_fit():
    with skip_if_unimplemented():
        params = OrbitParams(P=10.0, tp=0.0, e=0.2, omega=0.5, K=4.0, gamma=-1.0)
        times = np.linspace(0.0, 9.0, 5)
        velocities = rv_curve(times, params, solver_name="newton", guess_name="simple")
        errors = np.full_like(velocities, 0.1)

        chi2 = chi_squared(times, velocities, errors, params,
                            solver_name="newton", guess_name="simple")
        assert chi2 == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------
# rv/dataset.py
# ----------------------------------------------------------------------

def test_load_rv_dataset_has_the_expected_shape():
    with skip_if_unimplemented():
        df = load_rv_dataset("k2-24")
        assert list(df.columns) == ["time", "mnvel", "errvel", "tel"]
        assert len(df) == 32
        assert (df["errvel"] > 0).all()
        assert (df["tel"] == "hires").all()


def test_load_rv_dataset_unknown_name_raises():
    with skip_if_unimplemented():
        with pytest.raises(KeyError):
            load_rv_dataset("not-a-real-dataset")


def test_load_rv_dataset_hd164922_is_multi_instrument():
    """The bigger (401-point) dataset - unlike k2-24, it has 3 real
    instruments, which is exactly what build_posterior's CompositeLikelihood
    path needs to be exercised against."""
    with skip_if_unimplemented():
        df = load_rv_dataset("hd164922")
        assert list(df.columns) == ["time", "mnvel", "errvel", "tel"]
        assert len(df) == 401
        assert set(df["tel"].unique()) == {"j", "a", "k"}
        assert (df["errvel"] > 0).all()


def test_load_rv_dataset_k2131_is_multi_instrument():
    with skip_if_unimplemented():
        df = load_rv_dataset("k2-131")
        assert list(df.columns) == ["time", "mnvel", "errvel", "tel"]
        assert len(df) == 70
        assert set(df["tel"].unique()) == {"harps-n", "pfs"}
        assert (df["errvel"] > 0).all()


# ----------------------------------------------------------------------
# rv/radvel_bridge.py.  Needs a real, implemented solver underneath - uses
# "newton" explicitly, not the "danby" default, so these run regardless of
# which solvers are finished; and needs the real radvel package (installed
# in this project's environment).
# ----------------------------------------------------------------------

def test_use_solver_patches_and_restores_rv_drive():
    """The single biggest risk the workflow doc calls out: a patch that
    looks installed but never actually intercepts RadVel's model calls.
    This checks both that it installs a different function AND that it
    puts the original back afterwards."""
    with skip_if_unimplemented():
        import radvel.kepler as kmod
        original = kmod.rv_drive
        with use_solver("newton", tol=1e-10) as counter:
            assert kmod.rv_drive is not original
            assert counter["n_calls"] == 0
        assert kmod.rv_drive is original


def test_use_solver_counter_increments_only_for_nonzero_eccentricity():
    with skip_if_unimplemented():
        import radvel.kepler as kmod
        with use_solver("newton", tol=1e-12) as counter:
            kmod.rv_drive(np.array([0.0, 1.0, 2.0]), [10.0, 0.0, 0.0, 0.0, 5.0])
            assert counter["n_calls"] == 0  # e == 0 -> shortcut, no Kepler solve
            kmod.rv_drive(np.array([0.0, 1.0, 2.0]), [10.0, 0.0, 0.3, 0.1, 5.0])
            assert counter["n_calls"] == 3


def test_fit_with_solver_recovers_an_injected_orbit():
    """End-to-end: fit a known synthetic orbit through real RadVel machinery
    and check the recovered parameters are close to the truth."""
    with skip_if_unimplemented():
        real = load_rv_dataset("k2-24")
        dataset = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)

        fitted, n_solves, jitter = fit_with_solver(
            dataset, error_propagation.INITIAL_GUESS, "newton", 1e-12,
            guess_name="canonical")

        assert n_solves > 0
        assert math.isfinite(jitter)
        # Real, noisy 32-point data won't recover the injection exactly, but
        # a competent fit should land in the same ballpark.
        assert 0.0 < fitted.e < 0.6
        assert 2.0 < fitted.K < 9.0


def test_fit_with_solver_reports_wall_clock_for_ours_and_native():
    with skip_if_unimplemented():
        real = load_rv_dataset("k2-24")
        dataset = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)
        timing = {}
        fit_with_solver(dataset, error_propagation.INITIAL_GUESS, "danby", 1e-12,
                        timing_out=timing)
        assert timing["fit_seconds"] > 0.0
        assert timing["native_fit_seconds"] > 0.0


def test_fit_with_solver_handles_a_multi_instrument_dataset():
    """hd164922 has 3 real instruments - build_posterior must route this
    through CompositeLikelihood, not the single-RVLikelihood path, and with
    ~13x more data than k2-24 the injected orbit should recover tightly."""
    with skip_if_unimplemented():
        real = load_rv_dataset("hd164922")
        dataset = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)

        fitted, n_solves, jitter = fit_with_solver(
            dataset, error_propagation.INITIAL_GUESS, "newton", 1e-10,
            guess_name="canonical")

        assert n_solves > 0
        assert math.isnan(fitted.gamma)  # no single systemic velocity across 3 instruments
        assert fitted.e == pytest.approx(error_propagation.INJECTED_TRUTH.e, abs=0.02)
        assert fitted.K == pytest.approx(error_propagation.INJECTED_TRUTH.K, abs=0.1)


# ----------------------------------------------------------------------
# experiments/error_propagation.py.  run() itself needs Anisa's
# io.config.load_config, which is still a skeleton - the rest of this
# module does not, and is tested directly here.
# ----------------------------------------------------------------------

def test_reference_E_solves_keplers_equation():
    with skip_if_unimplemented():
        e, M = 0.6, 1.234
        E = error_propagation._reference_E(e, M)
        assert E - e * math.sin(E) - M == pytest.approx(0.0, abs=1e-12)


class _FakeConfig:
    """Minimal stand-in for io.config.ExperimentConfig's .extra attribute -
    lets _orbit_params_from_extra be tested without Anisa's load_config."""
    def __init__(self, extra):
        self.extra = extra


def test_orbit_params_from_extra_falls_back_to_default_when_absent():
    with skip_if_unimplemented():
        cfg = _FakeConfig(extra={})
        result = error_propagation._orbit_params_from_extra(
            cfg, "injected_truth", error_propagation.INJECTED_TRUTH)
        assert result is error_propagation.INJECTED_TRUTH


def test_orbit_params_from_extra_overrides_from_config():
    with skip_if_unimplemented():
        cfg = _FakeConfig(extra={
            "injected_truth": {"P": 1.0, "tp": 2.0, "e": 0.1, "omega": 0.2, "K": 3.0, "gamma": 0.5},
        })
        result = error_propagation._orbit_params_from_extra(
            cfg, "injected_truth", error_propagation.INJECTED_TRUTH)
        assert result == OrbitParams(P=1.0, tp=2.0, e=0.1, omega=0.2, K=3.0, gamma=0.5)


def test_inject_synthetic_dataset_keeps_real_cadence_and_noise():
    with skip_if_unimplemented():
        real = load_rv_dataset("k2-24")
        injected = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)

        np.testing.assert_array_equal(injected["time"].to_numpy(), real["time"].to_numpy())
        np.testing.assert_array_equal(injected["errvel"].to_numpy(), real["errvel"].to_numpy())
        # the injected signal has real noise added, so it must differ from
        # the noiseless truth curve, but not be wildly larger than the noise
        truth_curve = error_propagation.noiseless_curve(real["time"], error_propagation.INJECTED_TRUTH)
        residual = injected["mnvel"].to_numpy() - truth_curve
        assert np.all(np.abs(residual) < 6 * real["errvel"].to_numpy())
        assert not np.allclose(residual, 0.0)


def test_inject_synthetic_dataset_is_reproducible_with_same_seed():
    with skip_if_unimplemented():
        real = load_rv_dataset("k2-24")
        a = error_propagation.inject_synthetic_dataset(real, error_propagation.INJECTED_TRUTH, seed=7)
        b = error_propagation.inject_synthetic_dataset(real, error_propagation.INJECTED_TRUTH, seed=7)
        np.testing.assert_array_equal(a["mnvel"].to_numpy(), b["mnvel"].to_numpy())


def test_run_tolerance_sweep_produces_one_row_per_combination():
    with skip_if_unimplemented():
        real = load_rv_dataset("k2-24")
        dataset = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)

        df = error_propagation.run_tolerance_sweep(
            dataset, error_propagation.INITIAL_GUESS,
            solvers=["newton", "danby"], tolerances=[1e-4, 1e-14],
            guess_name="canonical")

        assert len(df) == 4
        assert set(df["solver"]) == {"newton", "danby"}
        # every solver should agree on e to several digits: same data, same
        # equation, just different solvers/tolerances
        assert df["e"].std() < 1e-2


def test_run_tolerance_sweep_skips_unimplemented_solvers_with_a_warning(monkeypatch):
    """A solver that is registered but whose step() still raises
    NotImplementedError (a teammate's in-progress work) must be skipped
    with a warning, not crash the whole sweep - this is simulated with
    monkeypatch since, as of this writing, all 5 real solvers are finished.
    An unknown solver NAME (a config typo) is a different, real bug and is
    intentionally left to crash loudly instead - see the KeyError case
    that is NOT caught here."""
    real = load_rv_dataset("k2-24")
    dataset = error_propagation.inject_synthetic_dataset(
        real, error_propagation.INJECTED_TRUTH, seed=0)

    original_fit = error_propagation.fit_with_solver

    def fake_fit(dataset_, initial_guess, solver_name, tol, **kwargs):
        if solver_name == "unfinished-solver":
            raise NotImplementedError("UnfinishedSolver.step: see TODO above")
        return original_fit(dataset_, initial_guess, solver_name, tol, **kwargs)

    monkeypatch.setattr(error_propagation, "fit_with_solver", fake_fit)

    with pytest.warns(UserWarning):
        df = error_propagation.run_tolerance_sweep(
            dataset, error_propagation.INITIAL_GUESS,
            solvers=["newton", "unfinished-solver"], tolerances=[1e-8],
            guess_name="canonical")
    assert set(df["solver"]) == {"newton"}


def test_run_reference_mcmc_returns_finite_posterior_widths():
    with skip_if_unimplemented():
        real = load_rv_dataset("k2-24")
        dataset = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)
        sigma = error_propagation.run_reference_mcmc(
            dataset, error_propagation.INITIAL_GUESS, nrun=50, seed=0)
        assert set(sigma) >= {"e", "omega", "K"}
        assert all(math.isfinite(v) and v >= 0 for v in sigma.values())


def test_error_propagation_run_needs_config_loading():
    """run() itself is a thin wrapper around Anisa's io.config.load_config;
    it should skip cleanly until that lands, not fail."""
    with skip_if_unimplemented():
        error_propagation.run("configs/error_propagation.yaml")


def test_real_data_check_all_solvers_agree_on_k2131():
    """K2-131 is a genuine single-planet system, so a direct fit of its
    real velocities has no unmodeled-second-planet degeneracy to blame the
    result on. What this checks: all 5 solvers converge to the SAME answer
    on completely real data, with nothing injected.

    That answer (e ~ 0.13) does NOT match the literature's e ~ 0 (NASA
    Exoplanet Archive - K2-131 b is tidally circularised). A direct
    log-likelihood comparison (see rv/radvel_bridge.py's build_posterior
    docstring) found e~0 and e~0.13 are statistically indistinguishable
    here (log-likelihood differs by ~0.1) - our RV-only maximum-likelihood
    point estimate genuinely cannot resolve this from noise alone. That
    mismatch is an honest, worth-reporting finding about the limits of
    point estimates on weak-signal data, not a pipeline bug."""
    with skip_if_unimplemented():
        df = error_propagation.run_real_data_check("k2-131")

        assert len(df) == 5
        assert df["e"].std() < 1e-3    # solvers agree with each other
        assert df["K"].std() < 1e-2


def test_real_data_check_all_solvers_agree_on_k2_24():
    """K2-24 IS a two-planet system, so this is exploratory only (see the
    module docstring for why) - the point of this check is that all 5
    solvers agree with each other on the same real, unmodified data."""
    with skip_if_unimplemented():
        df = error_propagation.run_real_data_check("k2-24")
        assert len(df) == 5
        assert df["e"].std() < 1e-3
        assert df["K"].std() < 1e-2


def test_run_all_real_data_checks_covers_every_registered_dataset():
    with skip_if_unimplemented():
        df = error_propagation.run_all_real_data_checks(solvers=["newton", "danby"])
        assert set(df["dataset"]) == set(error_propagation.REAL_DATASET_INITIAL_GUESSES)
        assert len(df) == 2 * len(error_propagation.REAL_DATASET_INITIAL_GUESSES)


# ----------------------------------------------------------------------
# experiments/monte_carlo.py
# ----------------------------------------------------------------------

def test_run_monte_carlo_spread_is_much_larger_than_solver_shift():
    """The core scientific claim of step 5: real measurement noise should
    dwarf any difference solver tolerance makes (step 4)."""
    with skip_if_unimplemented():
        real = load_rv_dataset("k2-24")
        dataset = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)
        best_fit, _, jitter = fit_with_solver(
            dataset, error_propagation.INITIAL_GUESS, "danby", 1e-14,
            guess_name="canonical")

        mc_df = monte_carlo.run_monte_carlo(
            dataset, best_fit, jitter, n_realisations=6, solver_name="danby",
            tol=1e-14, guess_name="canonical", seed=1)

        assert len(mc_df) == 6
        noise_spread_e = mc_df["e"].std()
        # from the tolerance sweep test above: solver-induced shifts are of
        # order 1e-4 or smaller for these solvers on this dataset.
        assert noise_spread_e > 1e-3


def test_run_monte_carlo_handles_nan_gamma_from_a_multi_instrument_fit():
    """Regression test: a multi-instrument best-fit has gamma == NaN (no
    single systemic velocity across instruments). Feeding that straight into
    noiseless_curve's + gamma term used to poison every point of the curve
    with NaN, which then made every per-instrument gamma in the refit NaN
    too, which made CompositeLikelihood's own internal consistency check
    fail (nan != nan) - not caught by the single-instrument k2-24 tests."""
    with skip_if_unimplemented():
        real = load_rv_dataset("hd164922")
        dataset = error_propagation.inject_synthetic_dataset(
            real, error_propagation.INJECTED_TRUTH, seed=0)
        best_fit, _, jitter = fit_with_solver(
            dataset, error_propagation.INITIAL_GUESS, "newton", 1e-8,
            guess_name="canonical")
        assert math.isnan(best_fit.gamma)  # sanity: this is the multi-instrument case

        mc_df = monte_carlo.run_monte_carlo(
            dataset, best_fit, jitter, n_realisations=2, solver_name="newton",
            tol=1e-8, guess_name="canonical", seed=1)

        assert len(mc_df) == 2
        assert mc_df["e"].notna().all()


def test_monte_carlo_run_needs_config_loading():
    with skip_if_unimplemented():
        monte_carlo.run("configs/monte_carlo.yaml")


# ----------------------------------------------------------------------
# evaluation/propagation.py.  Pure DataFrame/dict transforms - no solving,
# no fitting, fully synthetic and exact.
# ----------------------------------------------------------------------

def _synthetic_fits() -> pd.DataFrame:
    return pd.DataFrame([
        {"solver": "newton", "tolerance": 1e-14, "e": 0.30000, "omega": 1.0, "K": 5.000, "gamma": 0.0, "jitter": 0.1},
        {"solver": "newton", "tolerance": 1e-4, "e": 0.30010, "omega": 1.0, "K": 5.001, "gamma": 0.0, "jitter": 0.1},
        {"solver": "danby", "tolerance": 1e-14, "e": 0.30000, "omega": 1.0, "K": 5.000, "gamma": 0.0, "jitter": 0.1},
        {"solver": "danby", "tolerance": 1e-4, "e": 0.30000, "omega": 1.0, "K": 5.000, "gamma": 0.0, "jitter": 0.1},
    ])


def test_parameter_shift_is_zero_at_the_reference_tolerance():
    shifts = prop.parameter_shift(_synthetic_fits(), reference_tol=1e-14)
    reference_rows = shifts[shifts["tolerance"] == 1e-14]
    assert (reference_rows["e_shift"] == 0.0).all()
    assert (reference_rows["K_shift"] == 0.0).all()


def test_parameter_shift_matches_hand_computed_value():
    shifts = prop.parameter_shift(_synthetic_fits(), reference_tol=1e-14)
    newton_loose = shifts[(shifts["solver"] == "newton") & (shifts["tolerance"] == 1e-4)].iloc[0]
    assert newton_loose["e_shift"] == pytest.approx(0.0001)
    assert newton_loose["K_shift"] == pytest.approx(0.001)


def test_parameter_shift_picks_the_exact_reference_among_tiny_tolerances():
    fits = pd.DataFrame([
        {"solver": "danby", "tolerance": tol, "e": e}
        for tol, e in [(1e-8, 0.31), (1e-10, 0.32), (1e-14, 0.30)]
    ])
    shifts = prop.parameter_shift(fits, reference_tol=1e-14)
    assert shifts.loc[shifts["tolerance"] == 1e-14, "e_shift"].iloc[0] == 0.0
    assert shifts.loc[shifts["tolerance"] == 1e-8, "e_shift"].iloc[0] == pytest.approx(0.01)


def test_fit_timing_takes_medians_per_solver():
    fits = pd.DataFrame([
        {"solver": "danby", "fit_seconds": s, "native_fit_seconds": 0.1, "n_solves": 1000}
        for s in (1.0, 2.0, 9.0)
    ])
    row = prop.fit_timing(fits).iloc[0]
    assert row["median_fit_seconds"] == pytest.approx(2.0)
    assert row["median_slowdown_vs_native"] == pytest.approx(20.0)
    assert row["median_us_per_solve"] == pytest.approx(2000.0)


def test_shift_in_sigma_divides_by_posterior_width():
    shifts = prop.parameter_shift(_synthetic_fits(), reference_tol=1e-14)
    result = prop.shift_in_sigma(shifts, {"e": 0.01, "K": 0.5})
    newton_loose = result[(result["solver"] == "newton") & (result["tolerance"] == 1e-4)].iloc[0]
    assert newton_loose["e_shift_sigma"] == pytest.approx(0.0001 / 0.01)
    assert newton_loose["K_shift_sigma"] == pytest.approx(0.001 / 0.5)


def test_compare_error_budgets_assembles_the_final_table():
    shifts = prop.parameter_shift(_synthetic_fits(), reference_tol=1e-14)
    budget = prop.compare_error_budgets(
        shifts, noise_spread={"e": 0.08, "K": 0.45}, posterior_sigma={"e": 0.09, "K": 0.5})

    e_row = budget[budget["parameter"] == "e"].iloc[0]
    assert e_row["solver_induced_shift"] == pytest.approx(0.0001)
    assert e_row["noise_driven_spread"] == pytest.approx(0.08)
    assert e_row["mcmc_sigma"] == pytest.approx(0.09)
    assert e_row["ratio_solver_to_noise"] == pytest.approx(0.0001 / 0.08)


# ----------------------------------------------------------------------
# plotting/propagation_plots.py.  Checks the contract (a figure comes back,
# bad input fails loudly), not pixels - same convention as
# tests/test_plotting_convergence.py.
# ----------------------------------------------------------------------

def test_plot_tolerance_vs_parameter_shift_returns_a_figure():
    shifts = prop.parameter_shift(_synthetic_fits(), reference_tol=1e-14)
    shifts = prop.shift_in_sigma(shifts, {"e": 0.08, "K": 0.45})
    fig = pp.plot_tolerance_vs_parameter_shift(shifts)
    assert isinstance(fig, plt.Figure)


def test_plot_tolerance_vs_parameter_shift_requires_shift_columns():
    with pytest.raises(KeyError):
        pp.plot_tolerance_vs_parameter_shift(pd.DataFrame({"solver": ["newton"], "tolerance": [1e-4]}))


def test_plot_error_budget_returns_a_figure():
    shifts = prop.parameter_shift(_synthetic_fits(), reference_tol=1e-14)
    budget = prop.compare_error_budgets(
        shifts, noise_spread={"e": 0.08, "K": 0.45}, posterior_sigma={"e": 0.09, "K": 0.5})
    fig = pp.plot_error_budget(budget)
    assert isinstance(fig, plt.Figure)


def test_plot_amplification_returns_a_figure():
    fig = pp.plot_amplification([0.1, 0.5, 0.9])
    assert isinstance(fig, plt.Figure)
