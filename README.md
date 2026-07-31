# ca-fermentation-kinetics

A small, self-contained, well-documented tool that fits two classic
mechanistic models to clavulanic-acid (CA) fermentation data from
*Streptomyces clavuligerus*:

* **Logistic growth** for biomass `X(t)` — 3 parameters (`X0`, `Xmax`, `mu`)
* **Luedeking–Piret** for product `P(t)`, driven by the fitted biomass —
  2 parameters (`alpha`, `beta`)

It then compares parent strain **33** against 
**H**, reports every parameter with its standard error, and is deliberately
honest about what five unreplicated datapoints can and cannot support.


## Why mechanistic models, not machine learning?

The time-series is **5 points per strain with no replicates**. A neural
network or tree ensemble on that data would fit noise and report nothing
biologically meaningful. Two small models with **interpretable parameters**
(a growth rate, a carrying capacity, a growth-associated and a non-growth-
associated production coefficient) are the responsible choice — and the
parameters themselves tell the biological story.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and only NumPy, SciPy, and matplotlib (pytest optional).

---

## Run

Reproduce the entire analysis — fits, comparison table, interpretation, and
all figures — with one command:

```bash
python pipeline.py
```

This prints the fitted parameters with standard errors, automatic
uncertainty flags, and a plain-language biological reading, then writes four
figures into `./figures/`.

Run the tests:

```bash
python -m pytest tests/ -v      # if pytest is installed
python tests/test_models.py     # standalone, no pytest needed
```

---

## Project layout

| File | Purpose |
|---|---|
| `data.py` | Hard-coded data tables with extensive quirk annotations. |
| `models.py` | Logistic + Luedeking–Piret, implemented analytically **and** numerically (they cross-validate). |
| `fit.py` | `curve_fit` wrappers returning point estimates, standard errors, R², and an uncertainty-flagging function. |
| `plot.py` | Publication-quality figures (data = points, model = smooth curve). |
| `pipeline.py` | End-to-end orchestration: load → fit → table → interpret → plot. |
| `tests/test_models.py` | Sanity checks, analytic-vs-numeric cross-validation, parameter recovery. |
| `figures/` | Generated PNGs. |


## What the fit finds (point estimates)

| | Strain 33 | Strain H |
|---|---|---|
| Growth rate `mu` (1/h) | ~0.045 | ~0.037 |
| Carrying capacity `Xmax` (%PMV) | ~30.9 | ~27.4 |
| Non-growth CA coeff. `beta` | ~0.110 | ~0.066 *(rel. SE 49%)* |

Read narratively: strain 33 grows ~22% faster, reaches higher biomass, and
its standing biomass is the more productive CA factory (larger `beta`) —
consistent with CA being a stationary-phase secondary metabolite.

**But:** with five unreplicated points, strain H's `beta` carries a ~49%
relative standard error. The tool flags this automatically. We report the
direction of the effect; we do **not** claim statistical significance.

---

## Limitations 

* 5 timepoints per strain, no replicate flasks → wide error bars, no
  statistical testing possible.
* PMV% is a crude biomass proxy (not dry cell weight).
* The 5-strain endpoint table is single-timepoint → descriptive only, never
  fitted as a curve.
* The logistic `mu` is phenomenological, not a fundamental physiological
  growth constant.
* High R² on 5 points is weak evidence, never proof.


