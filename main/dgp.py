"""
dgp.py
======
Data Generating Processes for the EPDS simulation study.

All DGPs follow the partially linear model:
    y_i = beta0 * d_i + g(x_i) + eps_i
    d_i = m(x_i) + v_i

All functions return (X, d, y, beta0).

DGP progression (confounding severity increases toward DGP 7):
    DGP 1 : linear g(X),           d = noise              no confounding
    DGP 2 : quadratic g(X),        d = noise              no confounding
    DGP 3 : interactions in g(X),  d = noise              no confounding
    DGP 4 : mixed nonlinear g(X),  d = noise              no confounding
    DGP 5 : mixed nonlinear g(X),  d = X @ delta          linear confounding
    DGP 6 : mixed nonlinear g(X),  d = 2*x4*x5            mild NL confounding
    DGP 7 : mixed nonlinear g(X),  d = 2*x1^2*x2+3*x4*x5 severe NL confounding

Analytical confounding magnitude:
    DGP 6: cov(D, g(X)) = 2*3*Var(x4*x5) = 6
    DGP 7: cov(D, g(X)) = 2*5*Var(x1^2*x2) + 3*3*Var(x4*x5) = 30+9 = 39
"""

import numpy as np

def _base_X(n, p, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)), rng


# ============================================================
# DGP 1 -- linear, sparse, no confounding
# ============================================================
def dgp1(n=500, p=50, s=6, beta0=0.5, seed=42):
    """g(X)=5x1-5x2+3x3-3x4+x5-x6, d=noise. Baseline."""
    X, rng = _base_X(n, p, seed)
    gamma = np.zeros(p)
    gamma[:s] = [5.0, -5.0, 3.0, -3.0, 1.0, -1.0]
    d = rng.standard_normal(n)
    y = beta0 * d + X @ gamma + rng.standard_normal(n)
    return X, d, y, beta0


# ============================================================
# DGP 2 -- quadratic + linear, no confounding
# ============================================================
def dgp2(n=500, p=50, s=6, beta0=0.5, seed=42):
    """g(X)=5x1^2-5x2+3x3^2-3x4+x5, d=noise. LASSO misses x1^2, x3^2."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    g = 5.0*x1**2 - 5.0*x2 + 3.0*x3**2 - 3.0*x4 + 1.0*x5
    d = rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


# ============================================================
# DGP 3 -- interactions + linear, no confounding
# ============================================================
def dgp3(n=500, p=50, s=6, beta0=0.5, seed=42):
    """g(X)=5x1*x2-5x3+3x4*x5-3x6, d=noise. x1,x2,x4,x5 have zero marginal effect."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = 5.0*x1*x2 - 5.0*x3 + 3.0*x4*x5 - 3.0*x6
    d = rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


# ============================================================
# DGP 4 -- mixed nonlinear, no confounding
# ============================================================
def dgp4(n=500, p=50, s=6, beta0=0.5, seed=42):
    """g(X)=5x1^2*x2-5log|x3|+3x4*x5-3x6, d=noise. Hardest predictive case."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = 5.0*x1**2*x2 - 5.0*np.log(np.abs(x3)+1e-6) + 3.0*x4*x5 - 3.0*x6
    d = rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


# ============================================================
# DGP 5 -- mixed nonlinear + linear confounding
# ============================================================
def dgp5(n=500, p=50, s=6, beta0=0.5, seed=42):
    """g(X) same as DGP4; m(X)=x1+x2+x3+x4+x5+x6 (linear, overlapping with g)."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = 5.0*x1**2*x2 - 5.0*np.log(np.abs(x3)+1e-6) + 3.0*x4*x5 - 3.0*x6
    delta = np.zeros(p)
    delta[:s] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    d = X @ delta + rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


# ============================================================
# DGP 6 -- mixed nonlinear + MILD nonlinear confounding  (new)
# ============================================================
def dgp6(n=500, p=50, s=6, beta0=0.5, seed=42):
    """g(X) same as DGP4; m(X)=2*x4*x5 -- shares one NL term with g.
    cov(D,g(X))=6. LASSO misses x4*x5 in propensity -> bias in PDS."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = 5.0*x1**2*x2 - 5.0*np.log(np.abs(x3)+1e-6) + 3.0*x4*x5 - 3.0*x6
    d = 2.0*x4*x5 + rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


# ============================================================
# DGP 7 -- mixed nonlinear + SEVERE nonlinear confounding  (new)
# ============================================================
def dgp7(n=500, p=50, s=6, beta0=0.5, seed=42):
    """g(X) same as DGP4; m(X)=2*x1^2*x2+3*x4*x5 -- two shared NL terms.
    cov(D,g(X))=39. Both PDS-LASSO and DML-LASSO are badly biased."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = 5.0*x1**2*x2 - 5.0*np.log(np.abs(x3)+1e-6) + 3.0*x4*x5 - 3.0*x6
    d = 2.0*x1**2*x2 + 3.0*x4*x5 + rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


# ============================================================
# registry
# ============================================================

DGP_REGISTRY = {
    'dgp1': {'fn': dgp1, 'label': 'Linear'},
    'dgp2': {'fn': dgp2, 'label': 'Quadratic'},
    'dgp3': {'fn': dgp3, 'label': 'Interactions'},
    'dgp4': {'fn': dgp4, 'label': 'Mixed nonlinear'},
    'dgp5': {'fn': dgp5, 'label': 'NL + linear conf.'},
    'dgp6': {'fn': dgp6, 'label': 'NL + mild NL conf.'},
    'dgp7': {'fn': dgp7, 'label': 'NL + severe NL conf.'},
}


# ============================================================
# Ground-truth nonlinear terms for recovery analysis
#
# truth_terms_y : nonlinear terms in g(X) -- target for PySR on (X,y)
# truth_coefs_y : true coefficients in the y-equation
# truth_terms_d : nonlinear terms in m(X) -- target for PySR on (X,d)
# truth_coefs_d : true coefficients in the d-equation
#
# Term tuple format (matching estimators.extract_features output):
#   ('sq',    i, None) -> x_i^2
#   ('int',   i, j)    -> x_i * x_j   (i < j, 0-based)
#   ('cubic', i, j)    -> x_i^2 * x_j
#   ('log',   i, None) -> log|x_i|
#   ('sqrt',  i, None) -> sqrt|x_i|
#
# Pure linear terms are excluded -- LASSO handles them; not PySR's task.
# truth_terms is kept as alias for truth_terms_y for backward compat.
# ============================================================

_MIXED_NL_Y_TERMS = [
    ('cubic', 0, 1),    # x1^2 * x2
    ('log',   2, None), # log|x3|
    ('int',   3, 4),    # x4*x5
]
_MIXED_NL_Y_COEFS = {
    ('cubic', 0, 1):     5.0,
    ('log',   2, None): -5.0,
    ('int',   3, 4):     3.0,
}

# DGP 1 -- all linear
DGP_REGISTRY['dgp1'].update({
    'truth_terms_y': [], 'truth_coefs_y': {},
    'truth_terms_d': [], 'truth_coefs_d': {},
})

# DGP 2 -- quadratic Y, linear D
DGP_REGISTRY['dgp2'].update({
    'truth_terms_y': [('sq', 0, None), ('sq', 2, None)],
    'truth_coefs_y': {('sq', 0, None): 5.0, ('sq', 2, None): 3.0},
    'truth_terms_d': [], 'truth_coefs_d': {},
})

# DGP 3 -- interactions Y, linear D
DGP_REGISTRY['dgp3'].update({
    'truth_terms_y': [('int', 0, 1), ('int', 3, 4)],
    'truth_coefs_y': {('int', 0, 1): 5.0, ('int', 3, 4): 3.0},
    'truth_terms_d': [], 'truth_coefs_d': {},
})

# DGP 4 -- mixed NL Y, no confounding
DGP_REGISTRY['dgp4'].update({
    'truth_terms_y': _MIXED_NL_Y_TERMS,
    'truth_coefs_y': _MIXED_NL_Y_COEFS,
    'truth_terms_d': [], 'truth_coefs_d': {},
})

# DGP 5 -- mixed NL Y, linear D (linear terms handled by LASSO, not tracked)
DGP_REGISTRY['dgp5'].update({
    'truth_terms_y': _MIXED_NL_Y_TERMS,
    'truth_coefs_y': _MIXED_NL_Y_COEFS,
    'truth_terms_d': [], 'truth_coefs_d': {},
})

# DGP 6 -- mixed NL Y, mild NL D
DGP_REGISTRY['dgp6'].update({
    'truth_terms_y': _MIXED_NL_Y_TERMS,
    'truth_coefs_y': _MIXED_NL_Y_COEFS,
    'truth_terms_d': [('int', 3, 4)],
    'truth_coefs_d': {('int', 3, 4): 2.0},
})

# DGP 7 -- mixed NL Y, severe NL D
DGP_REGISTRY['dgp7'].update({
    'truth_terms_y': _MIXED_NL_Y_TERMS,
    'truth_coefs_y': _MIXED_NL_Y_COEFS,
    'truth_terms_d': [('cubic', 0, 1), ('int', 3, 4)],
    'truth_coefs_d': {('cubic', 0, 1): 2.0, ('int', 3, 4): 3.0},
})

# backward-compat aliases
for _k in DGP_REGISTRY:
    DGP_REGISTRY[_k]['truth_terms'] = DGP_REGISTRY[_k]['truth_terms_y']
    DGP_REGISTRY[_k]['truth_coefs'] = DGP_REGISTRY[_k]['truth_coefs_y']


# ============================================================
# sanity check
# ============================================================
if __name__ == '__main__':
    for name, entry in DGP_REGISTRY.items():
        X, d, y, beta0 = entry['fn']()
        print(f"{name}  ({entry['label']})")
        print(f"  shapes X={X.shape} d={d.shape} y={y.shape}")
        print(f"  y: mean={y.mean():.2f} std={y.std():.2f}")
        print(f"  d: mean={d.mean():.2f} std={d.std():.2f}")
        print(f"  truth_terms_y: {entry['truth_terms_y']}")
        print(f"  truth_terms_d: {entry['truth_terms_d']}")
        print()
