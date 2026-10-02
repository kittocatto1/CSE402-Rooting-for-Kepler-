# Source papers

This directory holds the papers we implemented from. Four are CC BY 4.0. The
other three are arXiv preprints under the arXiv non-exclusive distribution
licence, included only as course reference copies (see the note below the
table). Every other work the report cites is listed further down with its DOI
or link only.

## Included PDFs

| File | Citation | DOI / ID | Licence | Used for |
|---|---|---|---|---|
| `nwm9_mittal_2024.pdf` | S. K. Mittal, S. Panday & L. Jäntschi, "Enhanced Ninth-Order Memory-Based Iterative Technique for Efficiently Solving Nonlinear Equations", *Mathematics* 12(22), 3490, 2024 | `10.3390/math12223490` | CC BY 4.0 (MDPI) | NWM9 (`solvers/nwm9.py`) |
| `nwm11_mittal_2025.pdf` | S. K. Mittal, S. Panday, L. Jäntschi & L. C. Bolunduț, "Two novel efficient memory-based multi-point iterative methods for solving nonlinear equations", *AIMS Mathematics* 10(3), 5421-5443, 2025 | `10.3934/math.2025250` | CC BY 4.0 (AIMS Press) | NWM11 (`solvers/nwm11.py`) |
| `solaiman_hashim_2021.pdf` | O. S. Solaiman & I. Hashim, "Optimal Eighth-Order Solver for Nonlinear Equations with Applications in Chemical Engineering", *Intelligent Automation & Soft Computing* 27(2), 379-390, 2021 | `10.32604/iasc.2021.015285` | CC BY 4.0 (Tech Science Press) | the eighth-order base scheme of NWM11 |
| `matthies_2016_arxiv1602.07026.pdf` | G. Matthies, M. Salimi, S. Sharifi & J. L. Varona, "An optimal three-point eighth-order iterative method without memory for solving nonlinear equations with its dynamics", arXiv preprint; published in *Japan J. Indust. Appl. Math.* 33(3), 751-766, 2016 | [arXiv:1602.07026](https://arxiv.org/abs/1602.07026); `10.1007/s13160-016-0229-5` | arXiv non-exclusive distribution licence | the eighth-order base scheme of NWM9 |
| `napier_2024_arxiv2411.15374.pdf` | K. J. Napier, "Improved Initial Guesses for Numerical Solutions of Kepler's Equation", arXiv preprint, 2024 | [arXiv:2411.15374](https://arxiv.org/abs/2411.15374) | CC BY 4.0 | Napier starting guess (`guesses/napier.py`) |
| `radvel_fulton_2018_arxiv1801.01947.pdf` | B. J. Fulton, E. A. Petigura, S. Blunt & E. Sinukoff, "RadVel: The Radial Velocity Modeling Toolkit", arXiv preprint; published in *PASP* 130(986), 044504, 2018 | [arXiv:1801.01947](https://arxiv.org/abs/1801.01947); `10.1088/1538-3873/aaaaa8` | arXiv non-exclusive distribution licence | RadVel, the RV study (`rv/`) |
| `kipping_2013_arxiv1306.4982.pdf` | D. M. Kipping, "Parametrizing the exoplanet eccentricity distribution with the Beta distribution", arXiv preprint; published in *MNRAS Letters* 434(1), L51-L55, 2013 | [arXiv:1306.4982](https://arxiv.org/abs/1306.4982); `10.1093/mnrasl/slt075` | arXiv non-exclusive distribution licence | the Beta(0.867, 3.03) eccentricity sample in the grid (`experiments/grid.py`) |

The licence of each arXiv preprint is the one shown on its abs page
(checked 2026-10-02). Napier's preprint is CC BY 4.0. The other three
(Matthies et al., Fulton et al. and Kipping) are under arXiv's default
non-exclusive distribution licence, which gives arXiv, not us, the right to
redistribute them. They are included here only as course reference copies for
the CSE 402 project, and the abs page linked above is the authoritative
source. The three journal PDFs carry a CC BY 4.0 notice on their first page.

## Cited but not included

These are under publisher copyright. Fetch them through a library.

| Citation | DOI | Used for |
|---|---|---|
| J. M. A. Danby, "The solution of Kepler's equation, III", *Celestial Mechanics* 40(3-4), 303-312, 1987 | `10.1007/BF01235847` | Danby solver |
| F. L. Markley, "Kepler Equation solver", *Celest. Mech. Dyn. Astron.* 63(1), 101-111, 1995 | `10.1007/BF00691917` | Markley solver |
| C. D. Murray & S. F. Dermott, *Solar System Dynamics*, Cambridge University Press, 1999 | `10.1017/CBO9781139174817` | the form of Danby's method that RadVel's `kepler.c` follows |
| E. A. Petigura et al., "Two Transiting Low Density Sub-Saturns from K2", *ApJ* 818(1), 36, 2016 | `10.3847/0004-637X/818/1/36` | K2-24 system |
| E. A. Petigura et al., "Dynamics and Formation of the Near-resonant K2-24 System", *AJ* 156, 89, 2018 | `10.3847/1538-3881/aaceac` | K2-24 RV masses |
| B. J. Fulton et al., "Three Temperate Neptunes Orbiting Nearby Stars", *ApJ* 830(1), 46, 2016 | `10.3847/0004-637X/830/1/46` | HD 164922 system |
| F. Dai et al., "The Discovery and Mass Measurement of a New Ultra-short-period Planet: K2-131b", *AJ* 154(6), 226, 2017 | `10.3847/1538-3881/aa9065` | K2-131 ephemeris |

All DOIs above were checked against Crossref on 2026-10-02. arXiv IDs were
checked against arxiv.org.

## Software cited in the report

| Citation | DOI / link | Used for |
|---|---|---|
| California Planet Search, RadVel source code (`src/kepler.c`, `radvel/kepler.py`) | https://github.com/California-Planet-Search/radvel | RadVel's own Danby solver and its 1e-12 tolerance, and the `radvel` starting guess (`guesses/`) |
| F. Johansson et al., mpmath 1.4.1 | https://mpmath.org | 50-digit reference roots (`reference/`) and the 2000-digit order measurements (`experiments/verification.py`) |
| C. R. Harris et al., "Array programming with NumPy", *Nature* 585, 357-362, 2020 | `10.1038/s41586-020-2649-2` | arrays and floating-point maths throughout |
| P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", *Nature Methods* 17, 261-272, 2020 | `10.1038/s41592-019-0686-2` | the bracketing solve that draws the injected RV curve (`experiments/error_propagation.py`), and the maximum-likelihood fits through RadVel |
| W. McKinney, "Data structures for statistical computing in Python", *Proc. 9th Python in Science Conf.*, 56-61, 2010 | `10.25080/Majora-92bf1922-00a` | result tables (`io/`, `evaluation/`) |
| J. D. Hunter, "Matplotlib: a 2D graphics environment", *Computing in Science & Engineering* 9(3), 90-95, 2007 | `10.1109/MCSE.2007.55` | every figure (`plotting/`) |
| D. Foreman-Mackey, D. W. Hogg, D. Lang & J. Goodman, "emcee: the MCMC hammer", *PASP* 125, 306-312, 2013 | `10.1086/670067` | the MCMC run behind `results/error_propagation/posterior_sigma.csv` (through RadVel) |

## Transcription notes

Do not read formulas directly off the PDF text layer.

- Detached superscripts. In both with-memory papers, superscripts detach onto
  their own lines (for example, $e^{s^2}$ extracts as `e^s` followed by a
  stray `2`). This affected six of the eight test functions in the NWM9
  paper (see `NWM9_TEST_FUNCTIONS` in `experiments/verification.py`) and
  three of the eight in the NWM11 paper ($\varphi_5$, $\varphi_7$ and
  $\varphi_8$). The check that caught them is that the root the paper
  publishes must zero our copy of the function
  (`tests/test_verification.py`).
- Silent errors. Several misreadings of the iteration formulas still produce
  a scheme of the correct order, so measuring the order alone will not catch
  them.

Pin every transcription against the paper's own error constants: Eq. (6) of
the NWM9 paper and Eqs. (2.4)-(2.6) of the NWM11 paper. For example, a wrong
grouping in NWM11's $w'(t_k)$ missed the $e^8$ constant by about 60 orders of
magnitude while still converging at order 8. Both checks are encoded in
`tests/test_solvers_withmemory.py`.

## Naming

`<method-or-topic>_<first-author>_<year>[_arxiv<id>].pdf`, all lowercase. Add a
row to the table above, and an entry to `.gitignore`, for every new PDF.
