"""
plot.py
=======

Publication-quality figures: data as points, fitted models as smooth curves.
All figures are saved as PNG at 300 dpi.

Design choices for a portfolio piece:
  * Data = markers only (no connecting lines) -- the data does not tell us
    what happens between measurements; the MODEL does.
  * Model = smooth curve on a dense time grid.
  * Two-panel per-strain figure: biomass (top) and CA (bottom) share the
    x-axis, so you can visually line up "biomass plateaus" with "CA still
    rising".
  * A strain-comparison overlay for the 33-vs-H story.
  * A descriptive endpoint bar chart, clearly labelled as single-timepoint.

We use only matplotlib (no seaborn) to keep dependencies minimal.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend: render straight to file
import matplotlib.pyplot as plt

from models import logistic, luedeking_piret_analytic, make_lp_fit_function


# ----------------------------------------------------------------------
# Shared style
# ----------------------------------------------------------------------
def _apply_style():
    """Set a clean, readable global style for all figures."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,          # publication resolution
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,    # de-clutter: drop top/right spines
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    })


# Colour-blind-friendly palette (Wong 2011).
C_BIOMASS = "#0072B2"   # blue
C_PRODUCT = "#D55E00"   # vermillion
C_STRAIN1 = "#0072B2"   # 33
C_STRAIN2 = "#E69F00"   # H (orange)


# ----------------------------------------------------------------------
# 1. Per-strain two-panel figure
# ----------------------------------------------------------------------
def plot_strain_fit(strain, t, pmv, ca, logistic_fit, lp_fit, outpath):
    """Two stacked panels for one strain: biomass fit and CA fit.

    Parameters
    ----------
    strain : str            label, e.g. "33"
    t, pmv, ca : arrays     the data
    logistic_fit, lp_fit : FitResult objects
    outpath : str           where to save the PNG
    """
    _apply_style()

    # Dense grid for smooth model curves, spanning the observed window.
    t_grid = np.linspace(t.min(), t.max(), 300)

    X0, Xmax, mu = logistic_fit.values
    alpha, beta = lp_fit.values

    X_curve = logistic(t_grid, X0, Xmax, mu)
    # For CA we must use the SAME frozen biomass params and P0 anchor.
    lp_func = make_lp_fit_function(X0, Xmax, mu, P0=float(ca[0]),
                                   method="analytic")
    P_curve = lp_func(t_grid, alpha, beta)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    # --- Top: biomass ---
    ax1.plot(t, pmv, "o", color=C_BIOMASS, markersize=8,
             label="PMV data", zorder=3)
    ax1.plot(t_grid, X_curve, "-", color=C_BIOMASS, linewidth=2,
             label="logistic fit", zorder=2)
    ax1.axhline(Xmax, color=C_BIOMASS, linestyle=":", alpha=0.6,
                label=f"$X_{{max}}$ = {Xmax:.1f}%")
    ax1.set_ylabel("Biomass  (%PMV)")
    ax1.set_title(f"Strain {strain}: biomass growth")
    ax1.legend(loc="lower right")
    # Annotate mu with its uncertainty.
    mu_se = logistic_fit.stderrs[2]
    ax1.text(0.03, 0.95,
             f"$\\mu$ = {mu:.4f} $\\pm$ {mu_se:.4f} h$^{{-1}}$",
             transform=ax1.transAxes, va="top", fontsize=10)

    # --- Bottom: CA ---
    ax2.plot(t, ca, "s", color=C_PRODUCT, markersize=8,
             label="CA data", zorder=3)
    ax2.plot(t_grid, P_curve, "-", color=C_PRODUCT, linewidth=2,
             label="Luedeking-Piret fit", zorder=2)
    ax2.set_xlabel("Time  (h)")
    ax2.set_ylabel("Clavulanic acid  (µg/ml)")
    ax2.set_title(f"Strain {strain}: CA production")
    ax2.legend(loc="lower right")
    beta_se = lp_fit.stderrs[1]
    ax2.text(0.03, 0.95,
             f"$\\alpha$ = {alpha:.2f}, "
             f"$\\beta$ = {beta:.4f} $\\pm$ {beta_se:.4f}",
             transform=ax2.transAxes, va="top", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------
# 2. Strain comparison overlay
# ----------------------------------------------------------------------
def plot_strain_comparison(fits_by_strain, data_by_strain, outpath):
    """Overlay both strains' biomass and CA on a two-panel figure.

    Parameters
    ----------
    fits_by_strain : dict
        {strain: (logistic_fit, lp_fit)}
    data_by_strain : dict
        {strain: (t, pmv, ca)}
    outpath : str
    """
    _apply_style()
    colours = {"33": C_STRAIN1, "H": C_STRAIN2}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    for strain, (lf, pf) in fits_by_strain.items():
        t, pmv, ca = data_by_strain[strain]
        col = colours.get(strain, None)
        t_grid = np.linspace(t.min(), t.max(), 300)

        X0, Xmax, mu = lf.values
        alpha, beta = pf.values
        X_curve = logistic(t_grid, X0, Xmax, mu)
        lp_func = make_lp_fit_function(X0, Xmax, mu, P0=float(ca[0]),
                                       method="analytic")
        P_curve = lp_func(t_grid, alpha, beta)

        # biomass
        ax1.plot(t, pmv, "o", color=col, markersize=7, zorder=3)
        ax1.plot(t_grid, X_curve, "-", color=col, linewidth=2,
                 label=f"strain {strain}", zorder=2)
        # CA
        ax2.plot(t, ca, "s", color=col, markersize=7, zorder=3)
        ax2.plot(t_grid, P_curve, "-", color=col, linewidth=2,
                 label=f"strain {strain}", zorder=2)

    ax1.set_ylabel("Biomass  (%PMV)")
    ax1.set_title("Biomass: strain 33 vs H")
    ax1.legend(loc="lower right")

    ax2.set_xlabel("Time  (h)")
    ax2.set_ylabel("Clavulanic acid  (µg/ml)")
    ax2.set_title("CA production: strain 33 vs H")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    return outpath


# ----------------------------------------------------------------------
# 3. Descriptive endpoint bar chart (SINGLE TIMEPOINT -- not a fit)
# ----------------------------------------------------------------------
def plot_endpoint_bars(strains, pmv, ca, outpath):
    """Bar chart of endpoint PMV and CA per strain, plus specific CA yield.

    This is DESCRIPTIVE. The title and caption make explicit that these are
    single-timepoint measurements at 129 h and are NOT fitted.
    """
    _apply_style()
    x = np.arange(len(strains))
    specific = ca / pmv   # ug/ml per %PMV -- a crude "productivity per mass"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: CA and PMV as grouped bars (twin y-axes because different units).
    w = 0.4
    axL.bar(x - w/2, ca, width=w, color=C_PRODUCT, label="CA (µg/ml)")
    axL.set_ylabel("Clavulanic acid  (µg/ml)", color=C_PRODUCT)
    axL.tick_params(axis="y", labelcolor=C_PRODUCT)
    axL.set_xticks(x)
    axL.set_xticklabels(strains)
    axL.set_xlabel("Strain")

    axL2 = axL.twinx()
    axL2.bar(x + w/2, pmv, width=w, color=C_BIOMASS, label="PMV (%)")
    axL2.set_ylabel("Biomass  (%PMV)", color=C_BIOMASS)
    axL2.tick_params(axis="y", labelcolor=C_BIOMASS)
    axL2.grid(False)
    axL.set_title("Endpoint yields @129 h (single timepoint)")

    # Right: specific CA yield -- the more honest "is it productivity or mass?"
    bars = axR.bar(x, specific, color="#009E73")
    axR.set_xticks(x)
    axR.set_xticklabels(strains)
    axR.set_xlabel("Strain")
    axR.set_ylabel("Specific CA  (µg/ml per %PMV)")
    axR.set_title("Specific productivity @129 h")
    # Label the standout (33-8).
    for xi, val in zip(x, specific):
        axR.text(xi, val + 0.5, f"{val:.0f}", ha="center", fontsize=9)

    fig.suptitle("DESCRIPTIVE ONLY -- single-timepoint data, not a kinetic fit",
                 fontsize=10, style="italic", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    # Generate all figures from real data as a smoke test.
    import os
    from data import get_timeseries, get_endpoint, FITTABLE_STRAINS
    from fit import fit_logistic, fit_luedeking_piret

    os.makedirs("figures", exist_ok=True)
    fits, datas = {}, {}
    for s in FITTABLE_STRAINS:
        t, pmv, ca = get_timeseries(s)
        lf = fit_logistic(t, pmv)
        pf = fit_luedeking_piret(t, ca, lf)
        fits[s] = (lf, pf)
        datas[s] = (t, pmv, ca)
        plot_strain_fit(s, t, pmv, ca, lf, pf, f"figures/fit_strain_{s}.png")
        print("wrote", f"figures/fit_strain_{s}.png")

    plot_strain_comparison(fits, datas, "figures/fit_comparison.png")
    print("wrote figures/fit_comparison.png")

    strains, pmv_e, ca_e = get_endpoint()
    plot_endpoint_bars(strains, pmv_e, ca_e, "figures/endpoint_bars.png")
    print("wrote figures/endpoint_bars.png")
