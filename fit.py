"""
Curve-fitting layer. Turns raw data + models into fitted parameters WITH
honest uncertainty, using scipy.optimize.curve_fit.

"""

import numpy as np
from scipy.optimize import curve_fit
from models import logistic, make_lp_fit_function
from dataclasses import dataclass


@dataclass
class FitResult:
    names: list           # parameter names
    values: np.ndarray    # fitted point estimates
    stderrs: np.ndarray   # standard errors (sqrt of covariance diagonal)
    pcov: np.ndarray      # full covariance matrix
    r2: float             # R-squared (report WITH caveat)
    n_points: int         # number of data points used
    n_params: int         # number of fitted parameters
    model_name: str       # human label

    @property
    def dof(self):
        return self.n_points - self.n_params

    def rel_stderr(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.abs(self.stderrs / self.values)


# R2 calculation
def _r_squared(y_data, y_model):
    
    y_data = np.asarray(y_data, dtype=float)
    y_model = np.asarray(y_model, dtype=float)
    ss_res = np.sum((y_data - y_model) ** 2)          # residual sum of squares
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)  # total sum of squares
    if ss_tot == 0:
        return float("nan")   # data is flat; R^2 undefined
    return 1.0 - ss_res / ss_tot


# Standard errors
def _stderrs_from_pcov(pcov):
    return np.sqrt(np.diag(pcov))



def fit_logistic(t, X, p0=None, bounds=None):

    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)

    if p0 is None:
        # Physically-motivated defaults from the data itself.
        p0 = [max(X[0], 1e-3), max(X[-1], X.max()), 0.05]
    if bounds is None:
        # Keep parameters positive and physically sane.
        bounds = ([1e-6, 1e-6, 1e-6], [np.inf, np.inf, 5.0])

    popt, pcov = curve_fit(
        logistic, t, X,
        p0=p0, bounds=bounds,
        maxfev=20000,   # allow plenty of iterations; the data is awkward
    )
    y_model = logistic(t, *popt)
    return FitResult(
        names=["X0", "Xmax", "mu"],
        values=popt,
        stderrs=_stderrs_from_pcov(pcov),
        pcov=pcov,
        r2=_r_squared(X, y_model),
        n_points=len(t),
        n_params=3,
        model_name="logistic (biomass)",
    )


# Sequential fitting
def fit_luedeking_piret(t, P, logistic_fit, p0=None, bounds=None,
                        method="analytic"):
    
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)

    X0, Xmax, mu = logistic_fit.values
    P0 = float(P[0])   

    lp_func = make_lp_fit_function(X0, Xmax, mu, P0=P0, method=method)

    if p0 is None:
        p0 = [1.0, 0.05]
    if bounds is None:
        bounds = ([0.0, 0.0], [np.inf, np.inf])

    popt, pcov = curve_fit(
        lp_func, t, P,
        p0=p0, bounds=bounds,
        maxfev=20000,
    )
    y_model = lp_func(t, *popt)
    return FitResult(
        names=["alpha", "beta"],
        values=popt,
        stderrs=_stderrs_from_pcov(pcov),
        pcov=pcov,
        r2=_r_squared(P, y_model),
        n_points=len(t),
        n_params=2,
        model_name="Luedeking-Piret (CA)",
    )


# This prevents overinterpreting noisy parameter estimates
def flag_uncertainty(fit, rel_threshold=0.5):
    msgs = []
    rel = fit.rel_stderr()
    for name, val, se, r in zip(fit.names, fit.values, fit.stderrs, rel):
        if not np.isfinite(r):
            msgs.append(f"[{fit.model_name}] {name}={val:.4g}: SE undefined "
                        f"(value ~0); cannot assess.")
        elif r > rel_threshold:
            msgs.append(
                f"[{fit.model_name}] {name}={val:.4g} +/- {se:.4g} "
                f"(relative SE {r:.0%}): POORLY CONSTRAINED -- do not claim "
                f"this value is well-determined."
            )
    return msgs


def summarise_fit(fit):
    """Return a formatted multi-line string summarising a FitResult."""
    lines = []
    lines.append(f"Model: {fit.model_name}")
    lines.append(f"  points={fit.n_points}, params={fit.n_params}, "
                 f"dof={fit.dof}")
    lines.append(f"  R^2 = {fit.r2:.4f}   (CAVEAT: unreliable at n="
                 f"{fit.n_points})")
    for name, val, se, r in zip(fit.names, fit.values, fit.stderrs,
                                fit.rel_stderr()):
        lines.append(f"  {name:>5} = {val:12.5g} +/- {se:10.4g} "
                     f"(rel SE {r:5.0%})")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-check on real data.
    from data import get_timeseries
    t, pmv, ca = get_timeseries("33")

    lf = fit_logistic(t, pmv)
    print(summarise_fit(lf))
    print()
    pf = fit_luedeking_piret(t, ca, lf)
    print(summarise_fit(pf))
    print()
    for m in flag_uncertainty(lf) + flag_uncertainty(pf):
        print("FLAG:", m)
