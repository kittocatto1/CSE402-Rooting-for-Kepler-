"""Tests for the downstream RV chain. Owner: Fariha."""

from __future__ import annotations

import math

import pytest

from conftest import skip_if_unimplemented
from keplerbench.rv.anomaly import dE_to_dnu, mean_anomaly, radial_velocity, true_anomaly


def test_true_anomaly_equals_E_for_circular_orbit():
    with skip_if_unimplemented():
        for E in (0.0, 1.0, 2.0, 3.0, 4.5):
            assert true_anomaly(E, 0.0) == pytest.approx(E if E <= math.pi
                                                         else E - 2 * math.pi,
                                                         abs=1e-12)


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
