"""
Biomass measured as PMV% (packed mycelial volume, %v/v) 
clavulanic-acid titre in micrograms per millilitre (ug/ml).
"""

import numpy as np

TIME_H = np.array([24.0, 48.0, 72.0, 96.0, 120.0])  # hours

# Strain 33 (parent)
CA_33 = np.array([0.0, 200.0, 310.0, 420.0, 535.0])   # ug/ml
PMV_33 = np.array([15.0, 22.0, 28.0, 30.0, 30.0])     # % (plateau at 30)

# Strain H (parent) 
CA_H = np.array([0.0, 150.0, 210.0, 320.0, 380.0])    # ug/ml
PMV_H = np.array([12.0, 18.0, 22.0, 26.0, 26.0])      # % (plateau at 26)


# Five-strain endpoint data at 129 h (DESCRIPTIVE ONLY)
ENDPOINT_STRAINS = ["33", "33-3", "33-4", "33-7", "33-8"]
ENDPOINT_PMV = np.array([22.0, 26.0, 28.0, 24.0, 32.0])       # 129 h
ENDPOINT_CA = np.array([496.0, 722.0, 919.0, 696.0, 1545.0])  # activity


# Organization
def get_timeseries(strain):
    
    if strain == "33":
        return TIME_H, PMV_33, CA_33
    elif strain == "H":
        return TIME_H, PMV_H, CA_H
    else:
        raise ValueError(
            f"Unknown strain {strain!r}. Time-series strains are '33' and 'H'. "
            f"(The 33-x mutants only exist in the single-timepoint endpoint table.)"
        )


def get_endpoint():
    return ENDPOINT_STRAINS, ENDPOINT_PMV, ENDPOINT_CA


FITTABLE_STRAINS = ["33", "H"]


if __name__ == "__main__":
    # self-check
    for s in FITTABLE_STRAINS:
        t, pmv, ca = get_timeseries(s)
        print(f"Strain {s}: n={len(t)} points")
        print(f"  t   (h): {t}")
        print(f"  PMV (%): {pmv}")
        print(f"  CA (ug/ml): {ca}")
    strains, pmv, ca = get_endpoint()
    print("\nEndpoint @129h (descriptive only):")
    for st, pv, c in zip(strains, pmv, ca):
        print(f"  {st:>5}: PMV={pv:>4} %, CA={c:>6} ug/ml, "
              f"specific CA={c/pv:6.1f} ug/ml/%PMV")
