"""Name -> class lookup, so configs can say ``solver: danby`` as a string.

Owner: Anisa.

Usage in your own module::

    from keplerbench.core.registry import register_solver

    @register_solver("danby")
    class DanbySolver(IterativeSolver):
        ...
"""

from __future__ import annotations

from typing import Callable, TypeVar

from keplerbench.core.base import InitialGuess, KeplerSolver

_SOLVERS: dict[str, type[KeplerSolver]] = {}
_GUESSES: dict[str, type[InitialGuess]] = {}

T = TypeVar("T")


def register_solver(name: str) -> Callable[[type[KeplerSolver]], type[KeplerSolver]]:
    """Class decorator that adds a solver to the registry under ``name``."""

    def deco(cls: type[KeplerSolver]) -> type[KeplerSolver]:
        key = name.lower()
        if key in _SOLVERS:
            raise KeyError(f"solver {name!r} already registered")
        cls.name = name
        _SOLVERS[key] = cls
        return cls

    return deco


def register_guess(name: str) -> Callable[[type[InitialGuess]], type[InitialGuess]]:
    """Class decorator that adds a starting guess to the registry."""

    def deco(cls: type[InitialGuess]) -> type[InitialGuess]:
        key = name.lower()
        if key in _GUESSES:
            raise KeyError(f"guess {name!r} already registered")
        cls.name = name
        _GUESSES[key] = cls
        return cls

    return deco


def _load_all() -> None:
    """Import the solver/guess packages so their decorators run.

    Imported lazily to avoid a circular import at package start-up.
    """
    import keplerbench.guesses  # noqa: F401
    import keplerbench.solvers  # noqa: F401


def get_solver(name: str, **kwargs) -> KeplerSolver:
    """Build a solver instance by name, e.g. ``get_solver("newton")``."""
    _load_all()
    try:
        cls = _SOLVERS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown solver {name!r}; known: {sorted(_SOLVERS)}") from None
    return cls(**kwargs)


def get_guess(name: str, **kwargs) -> InitialGuess:
    """Build a starting-guess instance by name, e.g. ``get_guess("napier")``."""
    _load_all()
    try:
        cls = _GUESSES[name.lower()]
    except KeyError:
        raise KeyError(f"unknown guess {name!r}; known: {sorted(_GUESSES)}") from None
    return cls(**kwargs)


def list_solvers() -> list[str]:
    """All registered solver names, sorted."""
    _load_all()
    return sorted(_SOLVERS)


def list_guesses() -> list[str]:
    """All registered guess names, sorted."""
    _load_all()
    return sorted(_GUESSES)
