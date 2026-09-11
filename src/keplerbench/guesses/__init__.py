"""Starting-guess layer (Proposal Section 3.1/3.2).

Owner: Mahdi.

The proposal treats the starting guess as an INDEPENDENT experimental factor:
every solver is run with every guess, so we can separate "better start" from
"better iteration".  Keep these classes free of any solver logic.
"""

from keplerbench.guesses.simple import SimpleGuess  # noqa: F401
from keplerbench.guesses.canonical import CanonicalGuess  # noqa: F401
from keplerbench.guesses.radvel_start import RadVelGuess  # noqa: F401
from keplerbench.guesses.napier import NapierGuess  # noqa: F401
