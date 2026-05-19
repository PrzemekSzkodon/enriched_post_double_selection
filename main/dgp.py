"""
dgp.py
======
Data Generating Processes for the EPDS simulation study.

All DGPs follow the partially linear model:
    y_i = beta0 * d_i + g(x_i) + eps_i
    d_i = m(x_i) + v_i

where g() and m() vary across DGPs.

All functions return:
    X     : (n, p) numpy array of controls
    d     : (n,)   numpy array of treatment
    y     : (n,)   numpy array of outcome
    beta0 : float  true ATE

DGP summary:
    DGP 1 : linear, sparse, no confounding
    DGP 2 : quadratic + linear, no confounding
    DGP 3 : interactions + linear, no confounding
    DGP 4 : mixed nonlinear, no confounding
    DGP 5 : mixed nonlinear + confounding
"""

import numpy as np


def _base_X(n, p, seed):
    """Simulate iid N(0,1) control matrix."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)), rng


# ============================================================
# DGP 1 -- linear, sparse, no confounding
# ============================================================
def dgp1(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Outcome: y = beta0*d + X @ gamma + eps
    Treatment: d = noise (no confounding)
    Functional form: linear
    """
    X, rng = _base_X(n, p, seed)

    gamma = np.zeros(p)
    gamma[:s] = [5.0, -5.0, 3.0, -3.0, 1.0, -1.0]

    delta = np.zeros(p)   # no confounding

    d   = X @ delta + rng.standard_normal(n)
    y   = beta0 * d + X @ gamma + rng.standard_normal(n)

    return X, d, y, beta0


# ============================================================
# DGP 2 -- quadratic + linear, no confounding
# ============================================================
def dgp2(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Outcome: y = beta0*d + x1^2 + 0.8*x2 + x3^2 - 0.4*x4 + 0.3*x5 + eps
    Treatment: d = noise (no confounding)
    Functional form: quadratic main effects mixed with linear
    LASSO on raw X will miss x1^2 and x3^2
    """
    X, rng = _base_X(n, p, seed)

    x1, x2, x3, x4, x5 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4]

    g = (5.0 * x1**2 +
         -5.0 * x2 +
         3.0 * x3**2 +
        -3.0 * x4 +
         1.0 * x5)

    delta = np.zeros(p)   # no confounding

    d   = X @ delta + rng.standard_normal(n)
    y   = beta0 * d + g + rng.standard_normal(n)

    return X, d, y, beta0


# ============================================================
# DGP 3 -- interactions + linear, no confounding
# ============================================================
def dgp3(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Outcome: y = beta0*d + x1*x2 + 0.8*x3 - x4*x5 + 0.4*x6 + eps
    Treatment: d = noise (no confounding)
    Functional form: multiplicative interactions + linear terms
    LASSO on raw X will completely miss x1*x2 and x4*x5
    since x1, x2, x4, x5 have zero marginal linear effect
    """
    X, rng = _base_X(n, p, seed)

    x1, x2, x3, x4, x5, x6 = (X[:,0], X[:,1], X[:,2],
                                X[:,3], X[:,4], X[:,5])

    g = (5.0 * x1*x2 +
         -5.0 * x3 +
         3.0 * x4*x5 +
         -3.0 * x6)

    delta = np.zeros(p)   # no confounding

    d   = X @ delta + rng.standard_normal(n)
    y   = beta0 * d + g + rng.standard_normal(n)

    return X, d, y, beta0


# ============================================================
# DGP 4 -- mixed nonlinear, no confounding
# ============================================================
def dgp4(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Outcome: y = beta0*d + x1^2*x2 + log|x3| + x4*x5 + 0.8*x6 + eps
    Treatment: d = noise (no confounding)
    Functional form: mixed -- quadratic interaction, log, product, linear
    Inspired by Du et al. (2025) Case 5
    Hardest case for LASSO -- misses all nonlinear terms
    """
    X, rng = _base_X(n, p, seed)

    x1, x2, x3, x4, x5, x6 = (X[:,0], X[:,1], X[:,2],
                                X[:,3], X[:,4], X[:,5])

    g = (5 * x1**2 * x2 +
         -5.0 * np.log(np.abs(x3) + 1e-6) +
         3.0 * x4 * x5 +
         -3.0 * x6)

    delta = np.zeros(p)   # no confounding

    d   = X @ delta + rng.standard_normal(n)
    y   = beta0 * d + g + rng.standard_normal(n)

    return X, d, y, beta0


# ============================================================
# DGP 5 -- mixed nonlinear + confounding
# ============================================================
def dgp5(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Outcome: y = beta0*d + x1^2*x2 + log|x3| + x4*x5 + 0.8*x6 + eps
    Treatment: d = X @ delta + v  (confounding via shared x1...x5)
    Functional form: same as DGP 4 but now treatment is endogenous
    This is the key DGP -- tests whether enriched selection
    removes confounding bias under nonlinear DGP
    """
    X, rng = _base_X(n, p, seed)

    x1, x2, x3, x4, x5, x6 = (X[:,0], X[:,1], X[:,2],
                                X[:,3], X[:,4], X[:,5])

    g = (5.0 * x1**2 * x2 +
         -5.0 * np.log(np.abs(x3) + 1e-6) +
         3.0 * x4 * x5 +
         -3.0 * x6)

    # treatment now depends on same variables as outcome
    delta       = np.zeros(p)
    delta[:s]   = [1, 1, 1, 1, 1, 1]

    d   = X @ delta + rng.standard_normal(n)
    y   = beta0 * d + g + rng.standard_normal(n)

    return X, d, y, beta0


# ============================================================
# registry -- makes it easy to loop over all DGPs
# ============================================================
DGP_REGISTRY = {
    'dgp1': {'fn': dgp1, 'label': 'Linear sparse'},
    'dgp2': {'fn': dgp2, 'label': 'Quadratic + linear'},
    'dgp3': {'fn': dgp3, 'label': 'Interactions + linear'},
    'dgp4': {'fn': dgp4, 'label': 'Mixed nonlinear'},
    'dgp5': {'fn': dgp5, 'label': 'Mixed nonlinear + confounding'},
}


# ============================================================
# quick sanity check
# ============================================================
if __name__ == '__main__':
    for name, entry in DGP_REGISTRY.items():
        X, d, y, beta0 = entry['fn']()
        print(f"{name} ({entry['label']})")
        print(f"  X: {X.shape}  d: {d.shape}  y: {y.shape}")
        print(f"  y mean: {y.mean():.3f}  y std: {y.std():.3f}")
        print(f"  true beta0: {beta0}")
        print()