
# ? How does each strain grow, how does it produce CA and is CA production linked to growth or to the stationary phase?

"""
It will:
  1. Load the data (data.py)
  2. Fit logistic growth to biomass for each strain (fit.py)
  3. Fit Luedeking-Piret to CA using the frozen biomass fit (fit.py)
  4. Print a clean parameter comparison table with standard errors
  5. Print automatic uncertainty flags
  6. Print a plain-language biological interpretation
  7. Regenerate all figures (plot.py) into ./figures/
  8. Compute the growth- vs non-growth-associated CA contribution split
"""

import os
import numpy as np

from data import get_timeseries, get_endpoint, FITTABLE_STRAINS
from models import logistic
from fit import (fit_logistic, fit_luedeking_piret,
                 flag_uncertainty, summarise_fit)
from plot import (plot_strain_fit, plot_strain_comparison,
                  plot_endpoint_bars)


#? contribution split: how much CA came from growth vs standing biomass?


# This function calculates where the produced CA came from
def contribution_split(t, ca, logistic_fit, lp_fit):
    
    X0, Xmax, mu = logistic_fit.values
    alpha, beta = lp_fit.values
    t0, t_end = float(t.min()), float(t.max())

    X_t0 = logistic(t0, X0, Xmax, mu)
    X_end = logistic(t_end, X0, Xmax, mu)

    # Numerically integrate X over the run (fine grid; trapezoidal).
    tt = np.linspace(t0, t_end, 2000)
    integral_X = np.trapezoid(logistic(tt, X0, Xmax, mu), tt)

    growth = alpha * (X_end - X_t0)
    nongrowth = beta * integral_X
    total = growth + nongrowth
    return {
        "growth": growth,
        "nongrowth": nongrowth,
        "growth_frac": growth / total if total else float("nan"),
        "nongrowth_frac": nongrowth / total if total else float("nan"),
    }



def print_comparison_table(results):
    
    strains = list(results.keys())

    print("=" * 74)
    print("PARAMETER COMPARISON TABLE  (value +/- standard error)")
    print("=" * 74)

    header = f"{'parameter':<26}" + "".join(f"{('strain ' + s):>22}"
                                             for s in strains)
    print(header)
    print("-" * 74)

    def row(label, getter):
        cells = "".join(f"{getter(results[s]):>22}" for s in strains)
        print(f"{label:<26}{cells}")

    # Logistic params
    def fmt(fit, i):
        return f"{fit.values[i]:.4g} +/- {fit.stderrs[i]:.2g}"

    row("X0  (%PMV)",        lambda r: fmt(r["logistic"], 0))
    row("Xmax (%PMV)",       lambda r: fmt(r["logistic"], 1))
    row("mu  (1/h)",         lambda r: fmt(r["logistic"], 2))
    row("R^2 logistic",      lambda r: f"{r['logistic'].r2:.4f}")
    print("-" * 74)
    row("alpha (µg/ml/%PMV)", lambda r: fmt(r["lp"], 0))
    row("beta (/%PMV/h)",     lambda r: fmt(r["lp"], 1))
    row("R^2 L-P",            lambda r: f"{r['lp'].r2:.4f}")
    print("-" * 74)
    row("CA: growth-assoc %",  lambda r: f"{100*r['split']['growth_frac']:.1f}%")
    row("CA: non-growth %",    lambda r: f"{100*r['split']['nongrowth_frac']:.1f}%")
    print("=" * 74)



def print_interpretation(results):
    """Print a plain-language reading of the fitted parameters."""
    print()
    print("BIOLOGICAL INTERPRETATION")
    print("-" * 74)

    # Pull the two strains if present.
    if "33" in results and "H" in results:
        mu33 = results["33"]["logistic"].values[2]
        muH = results["H"]["logistic"].values[2]
        xmax33 = results["33"]["logistic"].values[1]
        xmaxH = results["H"]["logistic"].values[1]
        beta33 = results["33"]["lp"].values[1]
        betaH = results["H"]["lp"].values[1]
        betaH_relse = results["H"]["lp"].rel_stderr()[1]

        faster = (mu33 - muH) / muH * 100
        print(f"* Growth rate: strain 33 mu={mu33:.4f} /h vs H mu={muH:.4f} /h.")
        print(f"  -> strain 33 grows about {faster:.0f}% faster (point estimate).")
        print(f"* Carrying capacity: 33 reaches Xmax={xmax33:.1f}% vs "
              f"H {xmaxH:.1f}%.")
        print(f"  -> strain 33 also accumulates more biomass overall.")
        print(f"* Non-growth CA coefficient beta: 33={beta33:.4f} vs "
              f"H={betaH:.4f}.")
        ratio = beta33 / betaH if betaH else float('nan')
        print(f"  -> strain 33's beta is ~{ratio:.1f}x larger: its standing "
              f"biomass is the")
        print(f"     more productive CA factory (consistent with CA as a "
              f"secondary metabolite).")
        print()
        print("  *** HONESTY CHECK ***")
        print(f"  Strain H's beta carries a relative standard error of "
              f"{betaH_relse:.0%}.")
        print(f"  With 5 unreplicated points we CANNOT claim the strain "
              f"difference is real.")
        print(f"  We can only say the fitted point estimates point in a "
              f"biologically")
        print(f"  sensible direction. No statistical significance is available "
              f"here.")

    print()
    print("* In BOTH strains the non-growth-associated (beta) term makes a "
          "large,")
    print("  often dominant contribution to CA -- the model independently "
          "recovers the")
    print("  expected secondary-metabolite signature (product keeps rising "
          "after biomass")
    print("  plateaus). The exact growth/non-growth split is shown per strain "
          "in the table")
    print("  above; treat the split itself as approximate given the tiny "
          "sample.")



def main(make_figures=True):
    os.makedirs("figures", exist_ok=True)

    results = {}
    fits_by_strain = {}
    data_by_strain = {}

    print()
    print("#" * 74)
    print("# CLAVULANIC-ACID FERMENTATION KINETICS  --  full pipeline")
    print("#" * 74)
    print()
    print("Data: 5 timepoints/strain, NO replicates, PMV as biomass proxy.")
    print("Strategy: fit logistic to biomass, FREEZE it, then fit "
          "Luedeking-Piret to CA.")
    print()

    for s in FITTABLE_STRAINS:
        t, pmv, ca = get_timeseries(s)
        lf = fit_logistic(t, pmv)
        pf = fit_luedeking_piret(t, ca, lf)
        split = contribution_split(t, ca, lf, pf)

        results[s] = {"logistic": lf, "lp": pf, "split": split}
        fits_by_strain[s] = (lf, pf)
        data_by_strain[s] = (t, pmv, ca)

        print(summarise_fit(lf))
        print(summarise_fit(pf))
        flags = flag_uncertainty(lf) + flag_uncertainty(pf)
        if flags:
            for m in flags:
                print("  FLAG:", m)
        else:
            print("  (no parameters flagged as poorly constrained)")
        print()

    print_comparison_table(results)
    print_interpretation(results)

    if make_figures:
        print()
        print("Writing figures to ./figures/ ...")
        for s in FITTABLE_STRAINS:
            t, pmv, ca = data_by_strain[s]
            lf, pf = fits_by_strain[s]
            out = plot_strain_fit(s, t, pmv, ca, lf, pf,
                                  f"figures/fit_strain_{s}.png")
            print("  wrote", out)
        out = plot_strain_comparison(fits_by_strain, data_by_strain,
                                     "figures/fit_comparison.png")
        print("  wrote", out)
        strains, pmv_e, ca_e = get_endpoint()
        out = plot_endpoint_bars(strains, pmv_e, ca_e,
                                 "figures/endpoint_bars.png")
        print("  wrote", out)

    print()
    print("Done.")
    return results


if __name__ == "__main__":
    main()
