# Source Papers

The PDFs in this directory are **git-ignored** — they are not ours to republish. Recorded below is exactly what each one is and where it came from so anyone on the team can fetch their own copy.

## The Six References

| # | Citation | DOI / ID | Needed by | Access |
|---|----------|----------|-----------|--------|
| 1 | B. J. Fulton, E. A. Petigura, S. Blunt & E. Sinukoff, "RadVel: The Radial Velocity **Modeling** Toolkit", *PASP* **130**(986), 044504, 2018 | `10.1088/1538-3873/aaaaa8` | Fariha | IOP; arXiv preprint available |
| 2 | J. M. A. Danby, "The solution of Kepler's equation, **III**", *Celestial Mechanics* **40**(3–4), 303–312, 1987 | `10.1007/BF01235847` | Dipit | Springer — needs library access |
| 3 | F. L. Markley, "Kepler Equation solver", *Celestial Mechanics and Dynamical Astronomy* **63**(1), 101–111, 1995 | `10.1007/BF00691917` | Dipit | Springer — needs library access |
| 4 | S. K. Mittal, S. Panday & L. Jäntschi, "Enhanced Ninth-Order Memory-Based Iterative Technique for Efficiently Solving Nonlinear Equations", *Mathematics* **12**(22), 3490, 2024 | `10.3390/math12223490` | Suchi | MDPI, open access (CC-BY) |
| 5 | S. K. Mittal, S. Panday, L. Jäntschi & L. C. Bolunduț, "Two novel efficient memory-based multi-point iterative methods for solving nonlinear equations", *AIMS Mathematics* **10**(3), 5421–5443, 2025 | `10.3934/math.2025250` | Suchi | AIMS, open access (CC-BY) |
| 6 | K. J. Napier, "Improved Initial Guesses for Numerical Solutions of Kepler's Equation", arXiv preprint, 2024 | `arXiv:2411.15374` | Mahdi | arXiv, free |

> **Note:** Every DOI above was resolved against Crossref (or arXiv for [6]) on 2026-09-12. The titles and author lists reflect official registry returns rather than the proposal text.

---

## Local Filenames

| File | Reference | Downloaded by | Date |
|------|-----------|---------------|------|
| `nwm9-mittal-2024.pdf` | [4] | Suchi | 2026-09-12 |
| `nwm11-mittal-2025.pdf` | [5] | Suchi | 2026-09-12 |
| *Not yet fetched* | [1], [2], [3], [6] | — | — |

---

## Transcription Notes

**Do not read formulas directly off the PDF text layer.** 

* **Detached Superscripts:** In both with-memory papers, superscripts detach onto their own lines (e.g., $e^{s^2}$ extracts as `e^s` followed by a stray `2`). This affects three of the eight test functions in [5].
* **Silent Errors:** Several misreadings of the iteration formulas still produce a scheme of the *correct order*, meaning measuring the order will not catch the errors.

### Verification Guardrail
Pin every transcription against the paper's own **error constants**:
* **Paper [4]:** Eq. (6)
* **Paper [5]:** Eqs. (2.4)–(2.6)

*Example:* A wrong grouping in [5]'s $w'(t_k)$ missed the $e^8$ constant by about 60 orders of magnitude while still converging at order 8. Both verification checks are encoded in `tests/test_solvers_withmemory.py`.

---

## Secondary References

Cited in the report for provenance (not required as local PDFs):

* **Matthies, Salimi, Sharifi & Varona (2016)** (Ref [16] in [4]): Optimal 8th-order without-memory base for [4].
* **Solaiman & Hashim (2021)** (Ref [12] in [5]): Optimal 8th-order without-memory base for [5].
* **Petigura et al. 2016** (*AJ* 152, 28) & **Petigura et al. 2018** (*AJ* 155, 21): K2-24 system details — see `data/README.md`.

---

## Naming Convention

Format: `<method-or-topic>-<first-author>-<year>.pdf` (all lowercase)

* Examples: `danby-1987.pdf`, `markley-1995.pdf`, `napier-2024.pdf`, `radvel-fulton-2018.pdf`
* Remember to add a row to the **Local Filenames** table whenever a new PDF is fetched.