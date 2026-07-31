# GUIDE — Fermentation Kinetics Modelling, From Scratch

This is the teaching document. It explains **every concept before it is
used**, at beginner level, and is brutally honest about what this tiny
dataset can support. If you read only one file, read this one.

Structure:

1. The problem and why we model it this way
2. Theory you need first (rates, ODEs, curve-fitting, metabolite kinetics)
3. Model 1 — logistic growth
4. Model 2 — Luedeking–Piret
5. How the fitting works (least squares, covariance, standard error, R², dof)
6. The sequential-fit strategy and why
7. Reading the results (strain 33 vs H)
8. Honest limitations — what a reviewer will ask
9. Swapping in richer real data
10. Interview-style questions (with what a good answer hits)
11. What to learn next — toward a full bioprocess optimisation pipeline
12. How this connects to your Garson tool and a future DoE screen

---

## 1. The problem and why we model it this way

*Streptomyces clavuligerus* is a filamentous soil bacterium grown in stirred
liquid broth. It does two things we care about: it **grows** (biomass
increases), and it **manufactures clavulanic acid (CA)** — a β-lactamase
inhibitor that protects antibiotics like amoxicillin from resistant bacteria.

We want to answer two linked questions:

* How fast does biomass grow, and where does it stop?
* How is CA production linked to that growth — is CA made *while* cells
  divide, or *after* growth has slowed?

We answer them with two small mechanistic models, each with a handful of
parameters that have direct biological meaning. This is the opposite of a
black-box fit. With only **five datapoints per strain**, a flexible ML model
would memorise noise; small interpretable models are the honest tool, and
the fitted parameters *are* the biological answer.

---

## 2. Theory you need first

### 2.1 What is a rate of change?

A rate of change is how fast something changes at a moment in time. If
biomass increases a little each hour, the *rate of change of biomass* is how
much it goes up per hour. Position changing over time is speed; biomass
changing over time is growth rate. Same idea: amount of change ÷ time taken.

We write it with the symbol `d`. The expression **`dX/dt`** reads "the rate
of change of biomass `X` with respect to time `t`" — plainly, "how fast is
biomass growing right now". It is not a fraction to divide; it is one idea:
an instantaneous rate.

### 2.2 What is a differential equation (an ODE)?

A differential equation is just an equation that **contains a rate of
change**. Instead of telling you the amount of something, it gives the
**rule** for how that amount changes moment to moment.

Analogy: a savings account whose rule is "each year, add 5% of the current
balance". That rule — rate of change of balance = 0.05 × balance — is a
differential equation. Knowing the starting balance and following the rule
forward reconstructs the balance at every future time. **"Ordinary"** (the O
in ODE) means the rule depends on one variable, here time.

**"Integrating"** or **"solving"** an ODE means going from the rule (`dP/dt`)
back to the quantity (`P`) — the reverse of taking a rate of change.

### 2.3 What does curve-fitting actually do? (least squares)

Fitting means finding the parameter values that make the model curve pass as
close as possible to the data. "As close as possible" is made precise:

* For each point, the **residual** is the vertical gap: `residual = data − model`.
* Residuals can be + or −; to stop them cancelling we **square** each one.
* Add them: `SS_res = Σ residual²` (the *sum of squared residuals*).
* The **best fit** is the parameters that **minimise `SS_res`** — hence
  **"least squares"**.

Squaring also punishes large misses far more than small ones, so the fitted
curve works hard to avoid any single badly-missed point. The routine that
performs the search is `scipy.optimize.curve_fit`.

### 2.4 Primary vs secondary metabolites (the biology that shapes the model)

Bacteria make two broad classes of molecule with very different timing:

* **Primary metabolites** are made *because* the cell is growing — amino
  acids, nucleotides, lipids needed to build a new cell. Production tracks
  growth; when growth stops, so does production.
* **Secondary metabolites** are made *when growth slows or stops* — during
  crowded, nutrient-limited stationary phase — often as stress response or
  chemical competition. Antibiotics (clavulanic acid, penicillin,
  streptomycin) are secondary metabolites.

This is the key to reading the data. CA is a secondary metabolite, so its
titre should keep climbing after biomass plateaus. **Watch for that
fingerprint in the data: biomass flat, CA still rising.** The second model
exists to capture and quantify exactly this.

> In the strain-33 data, biomass is stuck at 30 %PMV from 96→120 h, yet CA
> climbs 420→535 µg/ml. That *is* the secondary-metabolite signature.

---

## 3. Model 1 — Logistic growth

Biomass can't grow forever. Early on growth compounds (more cells → more new
cells); later, nutrients deplete and waste builds up, so growth brakes and
saturates at a ceiling. The curve — slow start, steep middle, flat top — is
the **S-shaped logistic**.

**The ODE** (the growth rule):

```
dX/dt = μ · X · (1 − X/Xmax)
```

Read it: growth rate = `μ·X` (the compounding part) × `(1 − X/Xmax)` (a
brake that shrinks to zero as `X` nears the ceiling `Xmax`).

**The closed-form solution** (what we actually fit):

```
X(t) = X0 · Xmax / ( X0 + (Xmax − X0) · exp(−μ·t) )
```

| Param | Meaning |
|---|---|
| `X0` | initial biomass (where the clock starts) |
| `Xmax` | carrying capacity — the ceiling |
| `μ` | specific growth rate (1/h) — steepness of the climb |

**Important caveat on μ.** This `μ` is **phenomenological**: the number that
makes the logistic curve fit. It is *not* guaranteed to equal the
fundamental physiological maximum growth rate `μ_max` a microbiologist would
define from nutrient-uptake (Monod) kinetics. The logistic `μ` blends "how
fast can these cells divide" with "how quickly does the flask fill up".
Useful and interpretable — but not a fundamental constant.

*(In `models.py`, `logistic_derivative` returns `dX/dt` exactly, by
evaluating the ODE right-hand side at `X(t)`. We need it for the next model.)*

---

## 4. Model 2 — Luedeking–Piret

In 1959, working on lactic-acid fermentation, Luedeking and Piret proposed a
single rule flexible enough to describe primary-like, secondary-like, or
mixed production. The **rate** of product formation has two additive parts:

```
dP/dt = α · (dX/dt) + β · X
```

* **`α · dX/dt` — growth-associated.** Product made in proportion to how
  fast biomass is made. When division stops, `dX/dt → 0` and this term
  vanishes. The primary-metabolite part.
* **`β · X` — non-growth-associated.** Product made in proportion to
  *standing* biomass, dividing or not. Largest when lots of biomass sits in
  stationary phase. The secondary-metabolite part.

Fitting `α` and `β` gives a **data-driven** verdict on CA's nature without
deciding it in advance. Because CA is a secondary metabolite tied to nutrient
limitation, we expect **β to dominate**.

**Units** (in our sparse-data world):
* `α` — µg/ml per %PMV: "for each unit of new biomass, how much CA comes
  with it?"
* `β` — µg/ml per %PMV per hour: "each hour, how much CA does each unit of
  standing biomass make?"

**Turning the rate rule into a curve.** To fit CA we need accumulated
product `P(t)`, so we **integrate** the rule. Because `X(t)` is the logistic
closed form, we can integrate term by term:

```
P(t) = P0 + α·(X(t) − X(t0))  +  β·∫X dt
       └── growth term ──┘      └ non-growth ┘
```

* The growth term integrates trivially: the integral of a derivative is just
  the change, `α·(X(t) − X(t0))`.
* The non-growth term needs `∫X dt`. We use the exact antiderivative of the
  logistic (see `models.py`), and — crucially — **cross-check it against a
  numerical ODE solver** (`solve_ivp`). The two independent implementations
  agree to ~1e-5 µg/ml (see the test suite), so we trust the maths.

To compare their *contributions* to the final titre:
* Total growth-associated CA = `α·(X_final − X0)`
* Total non-growth CA = `β·∫X dt` over the run

*(The pipeline prints this split per strain.)*

---

## 5. How the fitting works

### 5.1 Least squares (recap in practice)

`curve_fit(f, t, y, p0, bounds)` searches parameter space for the minimum of
`Σ(y − f(t))²`, starting from your initial guess `p0`, staying inside
`bounds`. For nonlinear models it uses a trust-region / Levenberg–Marquardt
optimiser.

### 5.2 Initial-guess sensitivity

Nonlinear fits can converge to a **poor local minimum** or fail entirely from
a bad start. Give physically-motivated guesses:
`X0 ≈ first biomass reading`, `Xmax ≈ plateau reading`, `μ ≈ 0.05/h`. The
code does this automatically from the data.

### 5.3 Covariance matrix and standard error

A fitted parameter is an **estimate**, not a fact. `curve_fit` returns the
**covariance matrix `pcov`** — a square table of how uncertain the
parameters are and how they trade off. The **standard error** of each
parameter is `sqrt(diagonal of pcov)`: roughly its "give or take".

`μ = 0.045 ± 0.005` is reasonably pinned; `μ = 0.045 ± 0.022` is barely
constrained. On five points, expect wide standard errors. A parameter whose
**relative** SE (SE ÷ value) exceeds ~50% is flagged by the code as
poorly-constrained.

### 5.4 R² and why it lies on tiny samples

```
R² = 1 − SS_res / SS_tot
```

`SS_tot` is the total variation of the data about its mean; `R²` is the
fraction "explained", 0→1. **On 5 points this is dangerous**: a flexible
enough curve hugs a few points and posts `R² ≈ 0.99` regardless of whether
the model is right. High R² here is **weak evidence, never proof**. We report
it, then immediately caveat it.

### 5.5 Degrees of freedom

`dof = points − parameters`. Logistic: `5 − 3 = 2`. Luedeking–Piret:
`5 − 2 = 3`. It's the information left to judge the fit after the parameters
take their share. Tiny dof is *why* every error bar here is wide and every
claim hedged.

---

## 6. The sequential-fit strategy (and why)

We fit in **two stages**, not all at once:

1. Fit the **logistic** to biomass alone → recover `X0, Xmax, μ`.
2. **Freeze** those, then fit only **`α, β`** to CA, using the fixed biomass
   curve to drive production.

Why not fit all five parameters simultaneously? With five points per series,
a five-parameter joint fit lets parameters **trade off** against one another
(a bigger `μ` compensated by a smaller `Xmax`, etc.), and the solution
becomes unstable and non-identifiable. Splitting keeps each fit small (3
params, then 2) and defensible. This is the standard, honest choice on
sparse data.

---

## 7. Reading the results (strain 33 vs H)

Point estimates from the pipeline:

| Parameter | Strain 33 | Strain H |
|---|---|---|
| `μ` (1/h) | ~0.045 | ~0.037 |
| `Xmax` (%PMV) | ~30.9 | ~27.4 |
| `α` (µg/ml/%PMV) | ~16.2 | ~16.8 |
| `β` (per %PMV per h) | ~0.110 | ~0.066 *(rel SE 49%)* |

**The story the model tells:** strain 33 grows ~22% faster, reaches a higher
biomass ceiling, and has a ~1.7× larger `β` — its standing biomass is the
more productive CA factory. That is exactly what you'd expect if CA is a
stationary-phase secondary metabolite. The `α` values are similar, so the
strains differ mainly in the *non-growth* route.

**The honesty check (this is the important part):** strain H's `β` carries a
**~49% relative standard error**. With five unreplicated points we **cannot**
claim the strain difference is statistically real. We can say the fitted
point estimates point in a biologically sensible direction — and no more. The
code flags this automatically rather than letting you overclaim.

---

## 8. Honest limitations — what a reviewer will ask

A good reviewer will probe these immediately; the project pre-empts them.

* **Only 5 timepoints per strain, no replicates.** No estimate of
  measurement scatter, so no weighting by real error and **no significance
  testing** of strain differences. Error bars are wide.
* **PMV% is a crude biomass proxy.** It's the settled mycelial volume
  fraction, not dry cell weight, and is distorted by pellet morphology
  (fluffy vs compact). Two cultures with equal true biomass can read
  different PMV.
* **Endpoint 5-strain table is single-timepoint.** One measurement per
  strain, no time-course → *cannot* fit a kinetic curve. Used descriptively
  only (final yields, specific productivity), walled off from the fits.
* **Phenomenological μ.** Not the fundamental physiological growth constant;
  a proper characterisation needs a Monod model with measured substrate
  uptake.
* **High R² on 5 points is not validation.** It cannot distinguish a correct
  model from a flexible one hugging a few points.

None of these sink the project. Named openly, they turn a naive fit into a
careful, defensible piece of modelling — which is the point of a portfolio.

---

## 9. Swapping in richer real data

The same two models, fed better data, give far firmer answers. In rough
order of value:

* **Dry cell weight (DCW) instead of PMV.** A direct, physically-meaningful
  biomass measure. Removes the morphology artefact and lets `Xmax`/`μ` mean
  what they should. *(Even better: DCW + offline OD calibration.)*
* **Replicate flasks at each condition.** Every point gets a real error bar;
  strain differences become **testable** (t-test / ANOVA on the parameters,
  or bootstrap CIs). This is the single biggest upgrade for defensibility.
* **More timepoints, denser near the growth→stationary transition.** That
  transition is exactly where the `α` and `β` contributions separate most
  clearly; sampling it well tightens both.
* **Dissolved oxygen (DO) and key nutrients (N, phosphate, Fe²⁺).** Lets you
  move from "*when* is CA made" to "*what controls how much*", and to extend
  the model beyond logistic + L–P toward substrate-linked (Monod) kinetics.

Concretely: with replicates and, say, 12–15 timepoints, degrees of freedom
rise, standard errors shrink, and today's cautious "points in this direction"
becomes a firm, testable conclusion.

---

## 10. Interview-style questions (with what a good answer hits)

**Q1. What do α and β mean, and which do you expect to dominate for
clavulanic acid?**
*Good answer:* α is the growth-associated coefficient (product made per unit
of new biomass, vanishes when growth stops); β is non-growth-associated
(product per unit standing biomass per hour, dominant in stationary phase).
CA is a **secondary metabolite**, so expect **β** to carry most production —
and the data show CA rising after biomass plateaus, consistent with that.

**Q2. Why not use a neural network or random forest on this data?**
*Good answer:* five unreplicated points. Flexible ML models would fit noise
and overfit catastrophically; they'd also give no interpretable biological
quantities. Mechanistic models have few parameters, each with meaning, and
are identifiable on small data. Match model complexity to data size.

**Q3. Your R² is 0.99. Is the model validated?**
*Good answer:* No. On five points a high R² is nearly guaranteed and does not
distinguish a correct model from a flexible one. Real validation needs
out-of-sample data, replicates, and residual analysis. R² here is reported
with an explicit caveat; the standard errors and degrees of freedom matter
more.

**Q4. Where do the standard errors come from, and why are they large?**
*Good answer:* from the covariance matrix returned by `curve_fit` — SE =
sqrt of its diagonal. They're large because degrees of freedom are tiny
(2 and 3) and there are no replicates to pin down scatter. A ~49% relative SE
on β means it's barely constrained.

**Q5. Why fit the logistic first and freeze it, rather than fitting
everything together?**
*Good answer:* joint fitting of five parameters on five points is
non-identifiable — parameters trade off and the solution is unstable.
Sequential fitting keeps each stage low-dimensional (3 then 2) and
defensible. The cost is that biomass-fit uncertainty isn't formally
propagated into the CA fit — a limitation worth stating (and fixable later
with a joint Bayesian fit or bootstrap).

**Q6. PMV vs dry cell weight — does it matter?**
*Good answer:* Yes. PMV is a crude volumetric proxy sensitive to pellet
morphology; it's not calibrated mass. It's fine for a first-pass shape, but
`Xmax` and absolute `α`/`β` values shouldn't be over-interpreted. DCW (or a
DCW-calibrated proxy) is the fix.

**Q7. How would you decide if strain 33 really produces more CA than H?**
*Good answer:* You can't from this data — no replicates, so no significance
test. You'd run replicate fermentations, fit per replicate, and compare the
parameter distributions (t-test/ANOVA or bootstrap CIs on β), ideally with
DCW and denser sampling.

**Q8. What is `dP/dt` actually saying in words?**
*Good answer:* the instantaneous rate at which CA accumulates equals a part
proportional to how fast biomass is growing (`α·dX/dt`) plus a part
proportional to how much biomass is present (`β·X`). Integrating it over the
run gives the CA titre curve.

---

## 11. What to learn next — toward a bioprocess optimisation pipeline

A sensible progression from here:

1. **Substrate-linked kinetics (Monod).** Replace the phenomenological
   logistic with `μ = μ_max·S/(Ks+S)` and a substrate balance. Now `μ_max`
   is a real physiological constant and you can model nutrient limitation
   explicitly. *(Needs substrate measurements over time.)*
2. **Proper uncertainty propagation.** Bootstrap the fits, or go Bayesian
   (e.g. `emcee`/`PyMC`) to get full posterior distributions on parameters
   and to propagate biomass-fit uncertainty into the CA fit — fixing the
   sequential-fit limitation.
3. **Multi-condition datasets + parameter regression.** Run fermentations
   across medium compositions, fit `α, β, μ, Xmax` per condition, then model
   *how those parameters depend on the medium*. This is the bridge to
   optimisation.
4. **Design of Experiments (DoE) + response-surface / ML surrogate.** Screen
   factors efficiently (below), fit a response surface or ML surrogate to
   the kinetic parameters or directly to CA yield, and optimise.
5. **Scale-up considerations.** kLa, oxygen transfer, mixing time — the
   process-engineering layer.

---

## 12. How this connects to your Garson tool and a future DoE screen

This project is designed to **compose** with the Garson sensitivity tool you
already built, and to feed a future designed-experiment screen.

**The composition, concretely:**

* This tool converts each fermentation into a few **interpretable kinetic
  parameters** — most importantly **β** (stationary-phase CA productivity)
  and `μ`, `Xmax`.
* Run a **DoE screen** over the factors you flagged — **glycerol, arginine,
  Fe²⁺, phosphate, oleic acid** — with, say, a fractional-factorial or
  Plackett–Burman design to keep the flask count feasible.
* For each run, fit these models to get `β` (and the others) per condition.
  Now you have a table: *medium composition → kinetic parameters*.
* Feed that table to a small feed-forward network and apply **Garson's
  connection-weight algorithm** to rank **which nutrients most drive β** —
  i.e. which medium components most control stationary-phase CA production.

So the pipeline is:

```
DoE screen (glycerol, arginine, Fe²⁺, phosphate, oleic acid)
        │   run fermentations
        ▼
this tool: fit logistic + Luedeking–Piret per condition
        │   extract β, μ, Xmax per run
        ▼
regression table: medium composition → kinetic parameters
        │
        ▼
Garson tool: rank nutrient importance for β (CA productivity)
        │
        ▼
optimise medium for maximum stationary-phase CA
```

Each piece is honest on its own, and together they go from raw fermentation
data to a ranked, actionable statement about *what to change in the medium to
make more clavulanic acid* — which is exactly the kind of end-to-end
bioprocess-modelling story a DTU Bioengineering / RAPIDFUNG reviewer wants to
see.
