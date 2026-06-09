"""
Data-generating processes for the simulation study.

The model is the partially-linear specification of Robinson (1988):

    y = beta0 * d + g(x) + eps
    d = m(x) + v

with iid Gaussian noise (and Gaussian x in DGPs 1-7; DGP 8 mixes in three
Bernoulli dummies). All eight DGPs share beta0, n, p, sparsity s; what
varies is the form of g and m.

Progression:
    1  linear g,                m = 0                              no confounding
    2  quadratic g,             m = 0                              no confounding
    3  interactions in g,       m = 0                              no confounding
    4  mixed nonlinear g,       m = 0                              no confounding
    5  mixed nonlinear g,       m = x @ delta                      linear confounding
    6  mixed nonlinear g,       m = 2 x4 x5                        mild NL confounding
    7  mixed nonlinear g,       m = 2 x1^2 x2 + 3 x4 x5            severe NL confounding
    8  g with dummies,          m = 0.5 x6_d x4                    weak NL conf. (mixed types)

Analytical confounding magnitude cov(D, g(X)) for DGPs 6-8:
    DGP 6: 2 * 3 * Var(x4 x5)                              = 6
    DGP 7: 2 * 5 * Var(x1^2 x2) + 3 * 3 * Var(x4 x5)       = 30 + 9 = 39
    DGP 8: 0.5 * 2 * Var(x6_d x4)                          = 1   (shared with g)
"""

import numpy as np

# columns we promote to Bernoulli dummies in DGP-8. DGPs 1-7 are fully Gaussian.
DUMMY_COLS_DGP8 = [6, 7, 8]


def _base_X(n, p, seed, dummy_cols=None):
    """Generate (n,p) covariates. Specified columns become centred Bernoulli(0.5)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))

    if dummy_cols:
        # centre + scale so dummy columns have mean 0 and unit variance
        # (matches Gaussian columns' moments, which keeps lambda well-behaved)
        for j in dummy_cols:
            X[:, j] = (rng.binomial(1, 0.5, size=n).astype(float) - 0.5) / 0.5

    return X, rng


# --- DGPs ----------------------------------------------------------------

def dgp1(n=500, p=50, s=6, beta0=0.5, seed=42):
    """Linear sparse outcome, treatment is pure noise."""
    X, rng = _base_X(n, p, seed)
    gamma = np.zeros(p)
    gamma[:s] = [5, -5, 3, -3, 1, -1]
    d = rng.standard_normal(n)
    y = beta0 * d + X @ gamma + rng.standard_normal(n)
    return X, d, y, beta0


def dgp2(n=500, p=50, s=6, beta0=0.5, seed=42):
    """Quadratic outcome (squares LASSO can't see), no confounding."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    g = 5 * x1 ** 2 - 5 * x2 + 3 * x3 ** 2 - 3 * x4 + x5
    d = rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


def dgp3(n=500, p=50, s=6, beta0=0.5, seed=42):
    """Interactions in g (x1*x2, x4*x5 have zero marginal effect)."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = 5 * x1 * x2 - 5 * x3 + 3 * x4 * x5 - 3 * x6
    d = rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


def dgp4(n=500, p=50, s=6, beta0=0.5, seed=42):
    """Mixed nonlinear g: cubic-like + log + interaction. Hard predictive case."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = (5 * x1 ** 2 * x2
         - 5 * np.log(np.abs(x3) + 1e-6)
         + 3 * x4 * x5
         - 3 * x6)
    d = rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


def dgp5(n=500, p=50, s=6, beta0=0.5, seed=42):
    """Mixed nonlinear g, propensity is a linear function of x1..x6. Linear confounding."""
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = (5 * x1 ** 2 * x2
         - 5 * np.log(np.abs(x3) + 1e-6)
         + 3 * x4 * x5
         - 3 * x6)
    delta = np.zeros(p)
    delta[:s] = 1.0
    d = X @ delta + rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


def dgp6(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Mild nonlinear confounding: propensity shares the x4*x5 term with g.
    cov(D, g(X)) = 6. LASSO can't pick up x4 or x5 in m (zero marginal corr),
    so PDS-LASSO inherits the omitted-variable bias.
    """
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = (5 * x1 ** 2 * x2
         - 5 * np.log(np.abs(x3) + 1e-6)
         + 3 * x4 * x5
         - 3 * x6)
    d = 2 * x4 * x5 + rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


def dgp7(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Severe nonlinear confounding: propensity shares both x1^2*x2 and x4*x5.
    cov(D, g(X)) = 39. Both PDS-LASSO and DML-LASSO are badly biased here.
    """
    X, rng = _base_X(n, p, seed)
    x1, x2, x3, x4, x5, x6 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
    g = (5 * x1 ** 2 * x2
         - 5 * np.log(np.abs(x3) + 1e-6)
         + 3 * x4 * x5
         - 3 * x6)
    d = 2 * x1 ** 2 * x2 + 3 * x4 * x5 + rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


def dgp8(n=500, p=50, s=6, beta0=0.5, seed=42):
    """
    Weak-SNR nonlinear confounding with mixed continuous and dummy covariates.

    x1..x5 are continuous, x6, x7, x8 are centred Bernoulli(0.5) dummies,
    the rest is Gaussian noise.

    g(X) includes a dummy main effect and a dummy-continuous interaction;
    m(X) is the dummy-modulated nonlinear confounder x6_d * x4. Confounding
    strength is approximately 1.0 -- much smaller than DGPs 6 (6) and 7 (39)
    -- so this tests both the low-SNR failure mode and the framework's
    behaviour with mixed continuous / binary covariates.

    Note: the propensity is x6_d * x4 rather than x6_d * x4 * x5. The triple
    product would zero out asymptotically because E[x_j] = 0 for the
    continuous covariate, making the confounding spurious. The two-variable
    dummy interaction is captured cleanly by the ('int', i, j) extractor.
    """
    X, rng = _base_X(n, p, seed, dummy_cols=DUMMY_COLS_DGP8)
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    x6d = X[:, 6]  # Bernoulli centred to {-1, +1}

    g = (5 * x1 ** 2 * x2
         - 5 * np.log(np.abs(x3) + 1e-6)
         + 3 * x4 * x5
         - 3 * x6d
         + 2 * x6d * x4)
    m = 0.5 * x6d * x4
    d = m + rng.standard_normal(n)
    y = beta0 * d + g + rng.standard_normal(n)
    return X, d, y, beta0


# --- registry -----------------------------------------------------------

DGP_REGISTRY = {
    'dgp1': {'fn': dgp1, 'label': 'Linear'},
    'dgp2': {'fn': dgp2, 'label': 'Quadratic'},
    'dgp3': {'fn': dgp3, 'label': 'Interactions'},
    'dgp4': {'fn': dgp4, 'label': 'Mixed nonlinear'},
    'dgp5': {'fn': dgp5, 'label': 'NL + linear conf.'},
    'dgp6': {'fn': dgp6, 'label': 'NL + mild NL conf.'},
    'dgp7': {'fn': dgp7, 'label': 'NL + severe NL conf.'},
    'dgp8': {'fn': dgp8, 'label': 'Weak NL conf. (dummies)'},
}


# --- ground-truth nonlinear terms for recovery analysis -----------------
#
# Format of each term: (kind, i, j) where kind matches estimators.extract_features:
#   'sq'    -> x_i^2
#   'int'   -> x_i * x_j         (i < j)
#   'cubic' -> x_i^2 * x_j
#   'log'   -> log|x_i|
#   'sqrt'  -> sqrt|x_i|
#
# Pure linear terms are excluded -- they're LASSO's job, not PySR's.

_MIXED_NL_Y_TERMS = [
    ('cubic', 0, 1),     # 5 * x1^2 * x2
    ('log',   2, None),  # -5 * log|x3|
    ('int',   3, 4),     # 3 * x4 * x5
]
_MIXED_NL_Y_COEFS = {
    ('cubic', 0, 1):  5.0,
    ('log',   2, None): -5.0,
    ('int',   3, 4):  3.0,
}


DGP_REGISTRY['dgp1'].update(
    truth_terms_y=[], truth_coefs_y={},
    truth_terms_d=[], truth_coefs_d={},
)

DGP_REGISTRY['dgp2'].update(
    truth_terms_y=[('sq', 0, None), ('sq', 2, None)],
    truth_coefs_y={('sq', 0, None): 5.0, ('sq', 2, None): 3.0},
    truth_terms_d=[], truth_coefs_d={},
)

DGP_REGISTRY['dgp3'].update(
    truth_terms_y=[('int', 0, 1), ('int', 3, 4)],
    truth_coefs_y={('int', 0, 1): 5.0, ('int', 3, 4): 3.0},
    truth_terms_d=[], truth_coefs_d={},
)

DGP_REGISTRY['dgp4'].update(
    truth_terms_y=_MIXED_NL_Y_TERMS,
    truth_coefs_y=_MIXED_NL_Y_COEFS,
    truth_terms_d=[], truth_coefs_d={},
)

DGP_REGISTRY['dgp5'].update(
    truth_terms_y=_MIXED_NL_Y_TERMS,
    truth_coefs_y=_MIXED_NL_Y_COEFS,
    truth_terms_d=[], truth_coefs_d={},
)

DGP_REGISTRY['dgp6'].update(
    truth_terms_y=_MIXED_NL_Y_TERMS,
    truth_coefs_y=_MIXED_NL_Y_COEFS,
    truth_terms_d=[('int', 3, 4)],
    truth_coefs_d={('int', 3, 4): 2.0},
)

DGP_REGISTRY['dgp7'].update(
    truth_terms_y=_MIXED_NL_Y_TERMS,
    truth_coefs_y=_MIXED_NL_Y_COEFS,
    truth_terms_d=[('cubic', 0, 1), ('int', 3, 4)],
    truth_coefs_d={('cubic', 0, 1): 2.0, ('int', 3, 4): 3.0},
)

# DGP-8: dummy-continuous interactions in both equations. x4 * x6_d appears
# in g (coefficient 2) and in m (coefficient 0.5). Both are captured cleanly
# by the ('int', i, j) extractor.
DGP_REGISTRY['dgp8'].update(
    truth_terms_y=_MIXED_NL_Y_TERMS + [('int', 3, 6)],   # x4 * x6_d in g
    truth_coefs_y={**_MIXED_NL_Y_COEFS, ('int', 3, 6): 2.0},
    truth_terms_d=[('int', 3, 6)],                       # x4 * x6_d in m
    truth_coefs_d={('int', 3, 6): 0.5},
)


# backward-compat aliases used in older notebooks
for _k in DGP_REGISTRY:
    DGP_REGISTRY[_k]['truth_terms'] = DGP_REGISTRY[_k]['truth_terms_y']
    DGP_REGISTRY[_k]['truth_coefs'] = DGP_REGISTRY[_k]['truth_coefs_y']


if __name__ == '__main__':
    for name, entry in DGP_REGISTRY.items():
        X, d, y, beta0 = entry['fn']()
        print(f"{name}  ({entry['label']})")
        print(f"  X={X.shape}  d std={d.std():.2f}  y std={y.std():.2f}")
        if name == 'dgp8':
            print(f"  dummy cols {DUMMY_COLS_DGP8}, unique values: {np.unique(X[:, 6])}")
        print(f"  truth_terms_y: {entry['truth_terms_y']}")
        print(f"  truth_terms_d: {entry['truth_terms_d']}")
        print()
