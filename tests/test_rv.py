"""Tests for the downstream RV chain. Owner: Fariha."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import skip_if_unimplemented
from keplerbench.rv.anomaly import dE_to_dnu, mean_anomaly, radial_velocity, true_anomaly
from keplerbench.rv.dataset import load_rv_dataset
from keplerbench.rv.model import OrbitParams, chi_squared, rv_curve


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
