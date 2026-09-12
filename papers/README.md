# Source papers

The PDFs in this directory are **git-ignored** — they are not ours to
republish. Recorded below is exactly what each one is and where it came from,
so anyone on the team can fetch their own copy. Same convention as
`data/README.md`.

> **Read the corrections section before citing anything.** Four of the six
> references in `CSE402 - Project Proposal.pdf` are wrong, and two of them
> point at papers that do not exist. The citations in this file are the
> verified ones and are what the report must use.

## The six references

| # | Citation | DOI / ID | Needed by | Access |
|---|----------|----------|-----------|--------|
| 1 | B. J. Fulton, E. A. Petigura, S. Blunt & E. Sinukoff, "RadVel: The Radial Velocity **Modeling** Toolkit", *PASP* **130**(986), 044504, 2018 | `10.1088/1538-3873/aaaaa8` | Fariha | IOP; arXiv preprint available |
| 2 | J. M. A. Danby, "The solution of Kepler's equation, **III**", *Celestial Mechanics* **40**(3–4), 303–312, 1987 | `10.1007/BF01235847` | Dipit | Springer — needs library access |
| 3 | F. L. Markley, "Kepler Equation solver", *Celestial Mechanics and Dynamical Astronomy* **63**(1), 101–111, 1995 | `10.1007/BF00691917` | Dipit | Springer — needs library access |
| 4 | S. K. Mittal, S. Panday & L. Jäntschi, "Enhanced Ninth-Order Memory-Based Iterative Technique for Efficiently Solving Nonlinear Equations", *Mathematics* **12**(22), 3490, 2024 | `10.3390/math12223490` | Suchi | MDPI, open access (CC-BY) |
| 5 | S. K. Mittal, S. Panday, L. Jäntschi & L. C. Bolunduț, "Two novel efficient memory-based multi-point iterative methods for solving nonlinear equations", *AIMS Mathematics* **10**(3), 5421–5443, 2025 | `10.3934/math.2025250` | Suchi | AIMS, open access (CC-BY) |
| 6 | K. J. Napier, "Improved Initial Guesses for Numerical Solutions of Kepler's Equation", arXiv preprint, 2024 | `arXiv:2411.15374` | Mahdi | arXiv, free |

Every DOI above was resolved against Crossref (or arXiv for [6]) on
2026-09-12; the titles and author lists are what the registries return, not
what the proposal says.

## Local filenames

| File | Reference | Downloaded by | Date |
|------|-----------|---------------|------|
| `nwm9-mittal-2024.pdf` | [4] | Suchi | 2026-09-12 |
| `nwm11-mittal-2025.pdf` | [5] | Suchi | 2026-09-12 |
| _not yet fetched_ | [1] [2] [3] [6] | | |

## Corrections to the proposal's bibliography

The proposal's reference list needs four fixes before the report inherits it.

**[4] and [5] do not exist as cited.** Both DOIs resolve to unrelated papers:

| Proposal cites | That DOI actually is |
|---|---|
| Mittal et al., "An Optimal Higher-Order Derivative-Free with-Memory Iterative Scheme", *Symmetry* 16(5) 588 (`10.3390/sym16050588`) | "DAE-GAN: Underwater Image Super-Resolution…" — Gao, Li, Wang & Fan |
| Mittal et al., "High-Efficiency With-Memory Iterative Solvers…", *Mathematics* 13(2) 245 (`10.3390/math13020245`) | "Quantifying Uncertainty of Insurance Claims Based on Expert Judgments" — Handoko, Franty & Indrayatna |

The real sources are [4] and [5] in the table above. They were identified by
their convergence orders: the claimed 8.8989 and 10.7446 appear verbatim in
those two papers and nowhere else.

**[2] has the wrong part number and the wrong DOI.** The 1987 vol. 40,
pp. 303–312 paper is part **III**, DOI `10.1007/BF01235847`, sole author
Danby. Part **I** is a different paper — Danby & Burkardt, *Celestial
Mechanics* **31**(2), 95–107, 1983, `10.1007/BF01686811`. The DOI the
proposal gives (`10.1007/BF01230252`) is a 1980 three-body paper by Langlois.
The method the project implements is the one in part III.

**[6] has the wrong title.** It is "Improved Initial Guesses for Numerical
Solutions of Kepler's Equation", not "Analytic Starting Guesses for Kepler's
Equation via Symbolic Regression". The docstring in
`src/keplerbench/guesses/napier.py` already has it right.

**[1] has a one-word title error** — "Modeling", not "Fitting". DOI, volume
and pages are correct.

[3] Markley is correct as cited (the published title is lower-cased,
"Kepler Equation solver").

## Notes for whoever transcribes from these

**Do not read the formulas off the PDF text layer.** In both with-memory
papers the superscripts detach onto their own line, so `e^(s²)` extracts as
`e^s` followed by a stray `2`. Three of the eight test functions in [5] are
affected. Worse, several misreadings of the iteration formulas still produce
a scheme of the *correct order*, so measuring the order does not catch them.

Pin every transcription against the paper's own **error constants** instead —
[4] Eq. (6), [5] Eqs. (2.4)–(2.6). Those do discriminate: a wrong grouping in
[5]'s `w'(t_k)` missed the e⁸ constant by about 60 orders of magnitude while
still converging at order 8. Both checks are encoded in
`tests/test_solvers_withmemory.py`.

## Secondary references

Named inside the papers above, cited in the report for provenance, but not
needed as PDFs to do the work:

- The optimal eighth-order without-memory base of [4] is **Matthies, Salimi,
  Sharifi & Varona (2016)**, its ref [16].
- The optimal eighth-order without-memory base of [5] is **Solaiman & Hashim
  (2021)**, its ref [12].
- The K2-24 system itself: Petigura et al. 2016 (*AJ* 152, 28) and Petigura
  et al. 2018 (*AJ* 155, 21) — see `data/README.md`.

## Naming convention

`<method-or-topic>-<first-author>-<year>.pdf`, lower case, e.g.
`danby-1987.pdf`, `markley-1995.pdf`, `napier-2024.pdf`,
`radvel-fulton-2018.pdf`. Add a row to the local-filenames table when you
fetch one.
