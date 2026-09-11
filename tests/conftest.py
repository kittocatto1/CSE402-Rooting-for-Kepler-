"""Shared pytest fixtures and the "not written yet" convention.

Convention for this project
---------------------------
While the five of us are working in parallel, most tests target code that is
still a skeleton.  Those tests must SKIP, not fail, so that ``pytest`` stays
green and a real regression is visible immediately.

Use the ``skip_if_unimplemented`` helper for that::

    def test_danby_matches_reference(...):
        with skip_if_unimplemented():
            ...code that may raise NotImplementedError...

Once your module is implemented the skip disappears on its own and the test
starts really checking something.  Do not delete tests to make them pass.

Owner: Anisa.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from keplerbench.core.types import KeplerProblem


@contextmanager
def skip_if_unimplemented():
    """Turn a NotImplementedError from a skeleton into a pytest skip."""
    try:
        yield
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {exc}")


@pytest.fixture
def easy_problem() -> KeplerProblem:
    """A well-behaved case: moderate e, M away from 0."""
    return KeplerProblem(e=0.3, M=1.0)


@pytest.fixture
def hard_problem() -> KeplerProblem:
    """The pathological corner: e near 1, M near 0."""
    return KeplerProblem(e=0.999, M=1e-4)


@pytest.fixture
def sample_points() -> list[tuple[float, float]]:
    """A handful of (e, M) covering the range, for quick checks."""
    return [
        (0.0, 0.5),
        (0.1, 3.0),
        (0.5, 1.0),
        (0.9, 0.5),
        (0.99, 1e-3),
        (0.999, 1e-5),
    ]
