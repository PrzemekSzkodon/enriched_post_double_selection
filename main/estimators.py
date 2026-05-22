"""
estimators.py
=============
Estimation methods for the EPDS simulation study.

All estimators follow the same interface:
    Input:  X (n,p), d (n,), y (n,), n, p
    Output: dict with keys
                beta_hat  : float  point estimate of ATE
                se        : float  heteroskedasticity-robust SE
                ci_low    : float  lower bound of 95% CI
                ci_high   : float  upper bound of 95% CI

Methods:
    1. naive_ols       -- OLS of y on d only (no controls)
    2. full_ols        -- OLS of y on d + all X (infeasible benchmark)
    3. pds_lasso       -- Post-Double Selection with theoretical lambda
    4. dml_lasso       -- Double ML with LASSO nuisance estimators
    5. dml_nn          -- Double ML with NN nuisance estimators
    6. epds            -- Enriched PDS (NFSRD + PySR + enriched LASSO)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# helpers
# ============================================================

def _theoretical_lambda(n, p, c=1.1, alpha=0.05):
    """Belloni et al. (2014) theoretical lambda."""
    return (c / np.sqrt(n)) * norm.ppf(1 - alpha / (2 * p))


def _ols_result(y, X_with_d):
    """Run HC1-robust OLS and return standardised result dict."""
    X_c = sm.add_constant(X_with_d)
    res = sm.OLS(y, X_c).fit(cov_type='HC1')
    beta_hat = res.params[1]
    se       = res.HC1_se[1]
    return {
        'beta_hat' : beta_hat,
        'se'       : se,
        'ci_low'   : beta_hat - 1.96 * se,
        'ci_high'  : beta_hat + 1.96 * se,
    }


# ============================================================
# 1. naive OLS -- no controls
# ============================================================
def naive_ols(X, d, y, n, p):
    """OLS of y on d only. Biased if confounding present."""
    return _ols_result(y, d.reshape(-1, 1))


# ============================================================
# 2. full OLS -- all controls (infeasible benchmark)
# ============================================================
def full_ols(X, d, y, n, p):
    """OLS of y on d + all X. Infeasible when p > n."""
    X_with_d = np.column_stack([d, X])
    return _ols_result(y, X_with_d)


# ============================================================
# 3. PDS-LASSO -- Belloni et al. (2014)
# ============================================================
def pds_lasso(X, d, y, n, p):
    """
    Post-Double Selection with theoretical lambda.
    Step 1: LASSO(y ~ X) -> S_y
    Step 2: LASSO(d ~ X) -> S_d
    Step 3: OLS(y ~ d + X[S_y union S_d])
    """
    lam = _theoretical_lambda(n, p)

    # step 1
    lasso_y = Lasso(alpha=lam, max_iter=10000, fit_intercept=True)
    lasso_y.fit(X, y)
    S_y = set(np.where(lasso_y.coef_ != 0)[0])

    # step 2
    lasso_d = Lasso(alpha=lam, max_iter=10000, fit_intercept=True)
    lasso_d.fit(X, d)
    S_d = set(np.where(lasso_d.coef_ != 0)[0])

    # step 3 -- union
    S_union = sorted(S_y | S_d)

    if len(S_union) == 0:
        X_pds = d.reshape(-1, 1)
    else:
        X_pds = np.column_stack([d, X[:, S_union]])

    return _ols_result(y, X_pds)


# ============================================================
# 4. DML-LASSO -- Chernozhukov et al. (2018)
# ============================================================
def dml_lasso(X, d, y, n, p, n_folds=5):
    """
    Double ML with LASSO nuisance estimators.
    Uses cross-fitting to avoid overfitting bias.
    Step 1: residualise y on X via LASSO (cross-fitted)
    Step 2: residualise d on X via LASSO (cross-fitted)
    Step 3: regress y-residuals on d-residuals
    """
    kf  = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    lam = _theoretical_lambda(n, p)

    lasso_y = Lasso(alpha=lam, max_iter=10000)
    lasso_d = Lasso(alpha=lam, max_iter=10000)

    y_tilde = y - cross_val_predict(lasso_y, X, y, cv=kf)
    d_tilde = d - cross_val_predict(lasso_d, X, d, cv=kf)

    return _ols_result(y_tilde, d_tilde.reshape(-1, 1))


# ============================================================
# 5. DML-NN -- DML with neural network nuisance estimators
# ============================================================
def dml_nn(X, d, y, n, p, n_folds=5):
    """
    Double ML with MLP neural network nuisance estimators.
    Same cross-fitting structure as DML-LASSO.
    NN can capture nonlinear confounding -- key advantage over DML-LASSO.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    nn = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    y_tilde = y - cross_val_predict(nn, X_sc, y, cv=kf)

    nn2     = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    d_tilde = d - cross_val_predict(nn2, X_sc, d, cv=kf)

    return _ols_result(y_tilde, d_tilde.reshape(-1, 1))


# ============================================================
# 6. EPDS -- Enriched Post-Double Selection (your method)
# ============================================================
EPDS_LOG = []

def epds(X, d, y, n, p, pysr_iters=40, log=False):
    """
    Enriched Post-Double Selection (EPDS).

    Step 1: PySR on (y ~ X) and (d ~ X) -- full sample
    Step 2: Extract exact terms PySR found
    Step 3: Build x* = all original linear terms + exact PySR terms
    Step 4: PDS-LASSO on x* -- full sample
    Step 5: OLS on full sample -> valid beta_hat
    """
    try:
        from pysr import PySRRegressor
        import sympy as sp
    except ImportError:
        raise ImportError("PySR not installed. Run: pip install pysr")

    col_names = [f'x{i}' for i in range(p)]

    # ---- step 1: PySR on full X ----
    def run_pysr(X_in, y_in):
        model = PySRRegressor(
            niterations      = pysr_iters,
            binary_operators = ['+', '-', '*'],
            unary_operators  = ['square', 'log', 'sqrt'],
            maxsize          = 20,
            parsimony        = 0.0005,
            model_selection  = 'best',
            procs            = 0,
            random_state     = 42,
            verbosity        = 0,
        )
        model.fit(X_in, y_in, variable_names=col_names)
        return model

    model_y = run_pysr(X, y)
    model_d = run_pysr(X, d)

    # ---- step 2: extract exact terms ----
    def extract_features(model):
        cols = []
        syms = {f'x{i}': sp.Symbol(f'x{i}') for i in range(p)}
        try:
            eq = model.sympy()
        except Exception:
            return cols
        atoms = eq.atoms()

        for i in range(p):
            s = syms[f'x{i}']
            if sp.Pow(s, 2) in eq.atoms(sp.Pow) or s**2 in atoms:
                cols.append(('sq', i, None))

        for i in range(p):
            for j in range(i+1, p):
                si, sj = syms[f'x{i}'], syms[f'x{j}']
                if si*sj in eq.atoms(sp.Mul) or sj*si in atoms:
                    cols.append(('int', i, j))

        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                si, sj = syms[f'x{i}'], syms[f'x{j}']
                if sp.Pow(si, 2)*sj in atoms or si**2*sj in atoms:
                    cols.append(('cubic', i, j))

        for i in range(p):
            s = syms[f'x{i}']
            if sp.log(s) in atoms or sp.log(sp.Abs(s)) in atoms:
                cols.append(('log', i, None))

        for i in range(p):
            s = syms[f'x{i}']
            if sp.sqrt(s) in atoms or sp.Pow(s, sp.Rational(1,2)) in atoms:
                cols.append(('sqrt', i, None))

        return cols

    def build_features(term_list, X_in):
        cols = []
        for term in term_list:
            kind, i, j = term
            if kind == 'sq':
                cols.append(X_in[:, i]**2)
            elif kind == 'int':
                cols.append(X_in[:, i] * X_in[:, j])
            elif kind == 'cubic':
                cols.append(X_in[:, i]**2 * X_in[:, j])
            elif kind == 'log':
                cols.append(np.log(np.abs(X_in[:, i]) + 1e-6))
            elif kind == 'sqrt':
                cols.append(np.sqrt(np.abs(X_in[:, i])))
        return cols

    terms_y = extract_features(model_y)
    terms_d = extract_features(model_d)
    all_terms = list(set(terms_y) | set(terms_d))

    # ---- step 3: build enriched dictionary ----
    def build_dict(X_in):
        parts = [X_in]
        nonlin = build_features(all_terms, X_in)
        if nonlin:
            parts.extend([f.reshape(-1, 1) if f.ndim == 1 else f
                          for f in nonlin])
        return np.column_stack(parts)

    X_star = build_dict(X)

    # ---- step 4: PDS-LASSO on full enriched dictionary ----
    p_star   = X_star.shape[1]
    lam_star = _theoretical_lambda(n, p_star)

    lasso_y2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, y)
    lasso_d2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, d)

    S_y = set(np.where(lasso_y2.coef_ != 0)[0])
    S_d = set(np.where(lasso_d2.coef_ != 0)[0])
    S   = sorted(S_y | S_d)

    # ---- verbose ----
    if log:
        EPDS_LOG.append({
            'eq_y'        : str(model_y.sympy()),
            'eq_d'        : str(model_d.sympy()),
            'loss_y'      : model_y.equations_['loss'].min(),
            'loss_d'      : model_d.equations_['loss'].min(),
            'terms_added' : len(all_terms),
            'dict_size'   : X_star.shape[1],
            'n_selected'  : len(S),
            'S_y'         : sorted(S_y),
            'S_d'         : sorted(S_d),
            'S_union'     : S,
        })

    # ---- step 5: OLS on full sample ----
    if len(S) == 0:
        X_final = d.reshape(-1, 1)
    else:
        X_final = np.column_stack([d, X_star[:, S]])

    return _ols_result(y, X_final)


# ============================================================
# registry -- mirrors DGP_REGISTRY pattern
# ============================================================
ESTIMATOR_REGISTRY = {
    'naive_ols' : {'fn': naive_ols,  'label': 'Naive OLS'},
    'full_ols'  : {'fn': full_ols,   'label': 'Full OLS (infeasible)'},
    'pds_lasso' : {'fn': pds_lasso,  'label': 'PDS-LASSO'},
    'dml_lasso' : {'fn': dml_lasso,  'label': 'DML-LASSO'},
    'dml_nn'    : {'fn': dml_nn,     'label': 'DML-NN'},
    'epds'      : {'fn': epds,       'label': 'EPDS'},
}


# ============================================================
# sanity check
# ============================================================
if __name__ == '__main__':
    from dgp import dgp1

    X, d, y, beta0 = dgp1(n=500, p=50, s=5, beta0=0.5, seed=42)
    n, p = X.shape

    print(f"True beta0: {beta0}\n")

    for name, entry in ESTIMATOR_REGISTRY.items():
        if name == 'epds':
            continue   # skip epds in quick check - requires pysr
        res = entry['fn'](X, d, y, n, p)
        covers = res['ci_low'] < beta0 < res['ci_high']
        print(f"{entry['label']:<30} "
              f"beta_hat={res['beta_hat']:.4f}  "
              f"se={res['se']:.4f}  "
              f"CI=[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  "
              f"covers={covers}")