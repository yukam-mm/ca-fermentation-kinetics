"""
We implement:
  1. logistic(t, ...)             -- biomass X(t), closed form
  2. logistic_derivative(t, ...)  -- dX/dt, closed form (needed by L-P)
  3. luedeking_piret_numeric(...) -- product P(t) by numerical ODE integration
  4. luedeking_piret_analytic(...)-- product P(t) by exact integration

"""

import numpy as np
from scipy.integrate import solve_ivp

# Logistic equation
def logistic(t, X0, Xmax, mu):
    t = np.asarray(t, dtype=float)
    # exp(-mu*t) is the decaying term that carries X from X0 up toward Xmax.
    return X0 * Xmax / (X0 + (Xmax - X0) * np.exp(-mu * t))

def logistic_derivative(t, X0, Xmax, mu):
    X = logistic(t, X0, Xmax, mu)
    return mu * X * (1.0 - X / Xmax)




def luedeking_piret_numeric(t, alpha, beta, X0, Xmax, mu, P0=0.0):
    t = np.asarray(t, dtype=float)
    t0, t_end = float(t[0]), float(t[-1])

    def dPdt(tt, _P):
        # The right-hand side. Note it does not actually depend on P itself
        # here (product formation is driven by biomass), but solve_ivp still
        # wants the standard (t, y) signature.
        return alpha * logistic_derivative(tt, X0, Xmax, mu) \
            + beta * logistic(tt, X0, Xmax, mu)

    sol = solve_ivp(
        dPdt,
        t_span=(t0, t_end),
        y0=[P0],
        t_eval=t,
        method="RK45",
        rtol=1e-8, atol=1e-10,   # tight tolerances: we want this to be the "truth"
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.y[0]


def luedeking_piret_analytic(t, alpha, beta, X0, Xmax, mu, P0=0.0):
    t = np.asarray(t, dtype=float)
    t0 = float(t[0])

    X_t = logistic(t, X0, Xmax, mu)
    X_t0 = logistic(t0, X0, Xmax, mu)

    # Exact antiderivative of the logistic, F(t) with dF/dt = X(t).
    # F(t) = (Xmax/mu) * ln( e^{mu t} * X0 + (Xmax - X0) )
    def F(tt):
        tt = np.asarray(tt, dtype=float)
        return (Xmax / mu) * np.log(np.exp(mu * tt) * X0 + (Xmax - X0))

    growth_term = alpha * (X_t - X_t0)          # TERM 1
    nongrowth_term = beta * (F(t) - F(t0))       # TERM 2

    return P0 + growth_term + nongrowth_term


def make_lp_fit_function(X0, Xmax, mu, P0=0.0, method="analytic"):
    if method == "analytic":
        def f(t, alpha, beta):
            return luedeking_piret_analytic(t, alpha, beta, X0, Xmax, mu, P0)
    elif method == "numeric":
        def f(t, alpha, beta):
            return luedeking_piret_numeric(t, alpha, beta, X0, Xmax, mu, P0)
    else:
        raise ValueError("method must be 'analytic' or 'numeric'")
    return f


if __name__ == "__main__":
    # Quick smoke test: do the two L-P implementations agree?
    t = np.linspace(24, 120, 50)
    params = dict(alpha=1.5, beta=0.05, X0=15.0, Xmax=30.0, mu=0.05)
    Pa = luedeking_piret_analytic(t, **params)
    Pn = luedeking_piret_numeric(t, **params)
    print("max |analytic - numeric| =", np.max(np.abs(Pa - Pn)))
    print("analytic P(end) =", Pa[-1], " numeric P(end) =", Pn[-1])
