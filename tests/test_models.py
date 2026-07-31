"""
tests/test_models.py
====================

Sanity and consistency checks for the models and fitting code.

These are NOT exhaustive unit tests. They are the checks that actually matter
for trusting the results:

  A. SANITY -- do the models return physically sensible values?
     (positive, monotonic where expected, correct asymptotes, correct limits)

  B. CROSS-VALIDATION -- do the two independent Luedeking-Piret
     implementations (analytic vs numeric) agree? If they do, the maths is
     almost certainly right.

  C. PARAMETER RECOVERY -- if we generate data FROM the model with known
     parameters and then fit it, do we get the known parameters back? This is
     the single most convincing test that the fitting layer works.

Run with:
    python -m pytest tests/ -v
or, without pytest:
    python tests/test_models.py
"""

import os
import sys
import numpy as np

# Make the project root importable whether run via pytest or directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import (logistic, logistic_derivative,
                    luedeking_piret_analytic, luedeking_piret_numeric)
from fit import fit_logistic, fit_luedeking_piret


# ======================================================================
# A. SANITY CHECKS
# ======================================================================
def test_logistic_starts_near_X0():
    """At t=0 the logistic value should equal X0 exactly."""
    X0, Xmax, mu = 5.0, 30.0, 0.05
    assert np.isclose(logistic(0.0, X0, Xmax, mu), X0)


def test_logistic_asymptotes_to_Xmax():
    """As t -> infinity, logistic should approach the carrying capacity Xmax."""
    X0, Xmax, mu = 5.0, 30.0, 0.05
    late = logistic(1e5, X0, Xmax, mu)
    assert np.isclose(late, Xmax, rtol=1e-6)


def test_logistic_is_monotonic_increasing():
    """With Xmax > X0, biomass should never decrease over time."""
    X0, Xmax, mu = 5.0, 30.0, 0.05
    t = np.linspace(0, 300, 500)
    X = logistic(t, X0, Xmax, mu)
    assert np.all(np.diff(X) >= -1e-9)   # allow tiny numerical noise


def test_logistic_stays_within_bounds():
    """Biomass should stay in [X0, Xmax] for an increasing logistic."""
    X0, Xmax, mu = 5.0, 30.0, 0.05
    t = np.linspace(0, 500, 500)
    X = logistic(t, X0, Xmax, mu)
    assert X.min() >= X0 - 1e-9
    assert X.max() <= Xmax + 1e-9


def test_logistic_derivative_matches_finite_difference():
    """Analytical dX/dt should agree with numerical differentiation.

    np.gradient uses one-sided differences at the array boundaries, which are
    less accurate; we compare only interior points.
    """
    t = np.linspace(1, 200, 400)
    X0, Xmax, mu = 5.0, 30.0, 0.05
    analytic = logistic_derivative(t, X0, Xmax, mu)
    numeric = np.gradient(logistic(t, X0, Xmax, mu), t)
    # skip the two boundary points on each side
    assert np.allclose(analytic[2:-2], numeric[2:-2], rtol=5e-3, atol=1e-6)


def test_ca_is_nonnegative_and_increasing():
    """With alpha,beta >= 0 and growing biomass, CA should rise from >=0."""
    t = np.linspace(24, 120, 100)
    P = luedeking_piret_analytic(t, alpha=1.5, beta=0.05,
                                 X0=15.0, Xmax=30.0, mu=0.05, P0=0.0)
    assert P[0] >= -1e-9
    assert np.all(np.diff(P) >= -1e-9)


# ======================================================================
# B. CROSS-VALIDATION: analytic vs numeric Luedeking-Piret
# ======================================================================
def test_lp_analytic_matches_numeric():
    """The two independent L-P implementations must agree closely.

    This is the key check that our exact integration is correct.
    """
    t = np.linspace(24, 120, 60)
    params = dict(alpha=2.0, beta=0.08, X0=12.0, Xmax=28.0, mu=0.04)
    Pa = luedeking_piret_analytic(t, **params)
    Pn = luedeking_piret_numeric(t, **params)
    assert np.allclose(Pa, Pn, rtol=1e-4, atol=1e-3)


def test_lp_beta_zero_is_pure_growth_associated():
    """With beta=0, all product is growth-associated: P = alpha*(X - X0).

    This checks the growth term in isolation.
    """
    t = np.linspace(24, 120, 50)
    X0, Xmax, mu, alpha = 15.0, 30.0, 0.05, 2.0
    P = luedeking_piret_analytic(t, alpha=alpha, beta=0.0,
                                 X0=X0, Xmax=Xmax, mu=mu, P0=0.0)
    expected = alpha * (logistic(t, X0, Xmax, mu)
                        - logistic(t[0], X0, Xmax, mu))
    assert np.allclose(P, expected, rtol=1e-8, atol=1e-8)


def test_lp_alpha_zero_is_pure_non_growth():
    """With alpha=0, product comes only from the beta * integral(X) term.

    We check it against a fine trapezoidal integral of X.
    """
    t = np.linspace(24, 120, 50)
    X0, Xmax, mu, beta = 15.0, 30.0, 0.05, 0.1
    P = luedeking_piret_analytic(t, alpha=0.0, beta=beta,
                                 X0=X0, Xmax=Xmax, mu=mu, P0=0.0)
    # Independent numerical integral of X from t0 to each t.
    expected = np.array([
        beta * np.trapezoid(logistic(np.linspace(t[0], ti, 4000),
                                     X0, Xmax, mu),
                            np.linspace(t[0], ti, 4000))
        for ti in t
    ])
    assert np.allclose(P, expected, rtol=1e-4, atol=1e-3)


# ======================================================================
# C. PARAMETER RECOVERY (the convincing test)
# ======================================================================
def test_recover_logistic_parameters():
    """Generate clean logistic data, fit it, recover the known parameters."""
    t = np.array([24.0, 48.0, 72.0, 96.0, 120.0])
    true = dict(X0=6.0, Xmax=30.0, mu=0.045)
    X = logistic(t, **true)

    fit = fit_logistic(t, X)
    got = dict(zip(fit.names, fit.values))
    assert np.isclose(got["Xmax"], true["Xmax"], rtol=1e-2)
    assert np.isclose(got["mu"], true["mu"], rtol=5e-2)
    # X0 is the hardest to pin from few points; allow a looser tolerance.
    assert np.isclose(got["X0"], true["X0"], rtol=2e-1)


def test_recover_luedeking_piret_parameters():
    """Generate clean CA data from known alpha,beta; fit; recover them.

    We give the fitter the TRUE logistic params (as a FitResult stand-in) so
    this isolates the L-P recovery.
    """
    from fit import FitResult
    t = np.array([24.0, 48.0, 72.0, 96.0, 120.0])
    logi = dict(X0=6.0, Xmax=30.0, mu=0.045)
    true_alpha, true_beta = 12.0, 0.09

    # Build clean CA data from the model.
    P = luedeking_piret_analytic(t, alpha=true_alpha, beta=true_beta,
                                 P0=0.0, **logi)

    # Wrap the true logistic params in a FitResult so fit_luedeking_piret
    # can consume them.
    logi_fit = FitResult(
        names=["X0", "Xmax", "mu"],
        values=np.array([logi["X0"], logi["Xmax"], logi["mu"]]),
        stderrs=np.zeros(3), pcov=np.zeros((3, 3)),
        r2=1.0, n_points=5, n_params=3, model_name="logistic (biomass)",
    )

    fit = fit_luedeking_piret(t, P, logi_fit)
    got = dict(zip(fit.names, fit.values))
    assert np.isclose(got["alpha"], true_alpha, rtol=1e-2)
    assert np.isclose(got["beta"], true_beta, rtol=1e-2)


# ======================================================================
# Allow running without pytest.
# ======================================================================
if __name__ == "__main__":
    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {test.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {test.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
