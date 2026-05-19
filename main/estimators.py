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
def epds(X, d, y, n, p, split=0.3, pysr_iters=40):
    """
    Enriched Post-Double Selection.

    Sample A: LASSO selection + PySR functional form recovery
              + enriched dictionary construction
    Sample B: PDS-LASSO on enriched dictionary + OLS

    Steps:
        1. Split data into Sample A and Sample B
        2. [Sample A] LASSO(y ~ X), LASSO(d ~ X) -> S_y, S_d, S_union
        3. [Sample A] PySR on (y ~ X[S_union]) -> recover functional forms
        4. [Sample A] Build enriched dictionary x*
        5. [Sample A] LASSO(y ~ x*), LASSO(d ~ x*) -> S*_y, S*_d
        6. [Sample B] OLS(y ~ d + X[S*_y union S*_d]) -> beta_hat
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        raise ImportError("PySR not installed. Run: pip install pysr")

    # ---- sample split ----
    n_a   = int(n * split)
    idx_a = np.arange(n_a)
    idx_b = np.arange(n_a, n)

    X_a, d_a, y_a = X[idx_a], d[idx_a], y[idx_a]
    X_b, d_b, y_b = X[idx_b], d[idx_b], y[idx_b]

    n_a, n_b = len(idx_a), len(idx_b)
    lam_a    = _theoretical_lambda(n_a, p)

    # ---- step 2: initial LASSO selection on Sample A ----
    lasso_y = Lasso(alpha=lam_a, max_iter=10000).fit(X_a, y_a)
    lasso_d = Lasso(alpha=lam_a, max_iter=10000).fit(X_a, d_a)

    S_y     = set(np.where(lasso_y.coef_ != 0)[0])
    S_d     = set(np.where(lasso_d.coef_ != 0)[0])
    S_union = sorted(S_y | S_d)

    if len(S_union) == 0:
        # nothing selected -- fall back to pds_lasso on full sample
        return pds_lasso(X, d, y, n, p)

    # ---- step 3: PySR functional form recovery on Sample A ----
    X_sel_a = np.column_stack([d_a, X_a[:, S_union]])
    col_names = ['d'] + [f'x{i}' for i in S_union]

    pysr_model = PySRRegressor(
        niterations=pysr_iters,
        binary_operators=['+', '-', '*'],
        unary_operators=['square', 'log', 'sqrt'],
        maxsize=12,
        parsimony=0.0001,
        model_selection='accuracy',
        procs=0,
        random_state=42,
        verbosity=0,
    )
    pysr_model.fit(X_sel_a, y_a)

    # ---- step 4: build enriched dictionary ----
    def _enrich(X_in, d_in, S, pysr_eq):
        """
        Add PySR-recovered nonlinear terms to feature matrix.
        Squares and pairwise interactions of selected variables.
        """
        cols = [d_in.reshape(-1, 1), X_in[:, S]]

        # add squares of selected controls
        for j in S:
            cols.append((X_in[:, j] ** 2).reshape(-1, 1))

        # add pairwise interactions of selected controls
        S_list = list(S)
        for i in range(len(S_list)):
            for k in range(i + 1, len(S_list)):
                j1, j2 = S_list[i], S_list[k]
                cols.append((X_in[:, j1] * X_in[:, j2]).reshape(-1, 1))

        # add log of abs of selected controls
        for j in S:
            cols.append(np.log(np.abs(X_in[:, j]) + 1e-6).reshape(-1, 1))

        return np.column_stack(cols)

    X_star_a = _enrich(X_a, d_a, S_union, pysr_model)
    X_star_b = _enrich(X_b, d_b, S_union, pysr_model)

    p_star   = X_star_a.shape[1]
    lam_star = _theoretical_lambda(n_a, p_star)

    # ---- step 5: PDS-LASSO on enriched dictionary on Sample A ----
    # note: first column is d -- skip it for LASSO selection
    X_controls_a = X_star_a[:, 1:]
    X_controls_b = X_star_b[:, 1:]

    lasso_y2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_controls_a, y_a)
    lasso_d2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_controls_a, d_a)

    S_star_y = set(np.where(lasso_y2.coef_ != 0)[0])
    S_star_d = set(np.where(lasso_d2.coef_ != 0)[0])
    S_star   = sorted(S_star_y | S_star_d)

    # in epds() -- replace the existing step 6 block
    if len(S_star) == 0:
        # enrichment added nothing -- fall back to original linear selection
        if len(S_union) > 0:
            X_final = np.column_stack([d_b, X_b[:, list(S_union)]])
        else:
            X_final = d_b.reshape(-1, 1)
    else:
        X_final = np.column_stack([d_b, X_controls_b[:, list(S_star)]])

    return _ols_result(y_b, X_final)

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
            continue   # skip epds in quick check -- requires pysr
        res = entry['fn'](X, d, y, n, p)
        covers = res['ci_low'] < beta0 < res['ci_high']
        print(f"{entry['label']:<30} "
              f"beta_hat={res['beta_hat']:.4f}  "
              f"se={res['se']:.4f}  "
              f"CI=[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  "
              f"covers={covers}")