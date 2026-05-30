"""
estimators.py
=============
Estimation methods for the EPDS simulation study.

Standard interface for all estimators:
    Input : X (n,p), d (n,), y (n,), n, p
    Output: dict with keys beta_hat, se, ci_low, ci_high
            EPDS variants also return recovery metadata:
              epds_variant, pre_lasso_terms_y, post_lasso_terms_y,
              pre_lasso_terms_d, post_lasso_terms_d, term_coefs_y,
              term_coefs_d, du_selected_y, du_selected_d

Methods:
    1. naive_ols   -- OLS on d only (biased benchmark)
    2. full_ols    -- OLS on d + all X (infeasible benchmark)
    3. pds_lasso   -- Post-Double Selection, Belloni et al. (2014)
    4. dml_lasso   -- Double ML with LASSO nuisance, Chernozhukov et al. (2018)
    5. dml_nn      -- Double ML with NN nuisance
    6. epds        -- Enriched PDS: PySR basis expansion + PDS-LASSO (no pre-selection)
    7. epds_du     -- Enriched PDS: Du et al. Stein pre-selection + PySR + PDS-LASSO
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import Lasso
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
    """HC1-robust OLS; returns standard result dict."""
    X_c  = sm.add_constant(X_with_d, has_constant='add')
    res  = sm.OLS(y, X_c).fit(cov_type='HC1')
    bhat = res.params[1]
    se   = res.HC1_se[1]
    return {
        'beta_hat': bhat,
        'se'      : se,
        'ci_low'  : bhat - 1.96 * se,
        'ci_high' : bhat + 1.96 * se,
    }


# ============================================================
# 1. naive OLS
# ============================================================
def naive_ols(X, d, y, n, p):
    return _ols_result(y, d.reshape(-1, 1))


# ============================================================
# 2. full OLS (infeasible)
# ============================================================
def full_ols(X, d, y, n, p):
    return _ols_result(y, np.column_stack([d, X]))


# ============================================================
# 3. PDS-LASSO
# ============================================================
def pds_lasso(X, d, y, n, p):
    lam = _theoretical_lambda(n, p)

    lasso_y = Lasso(alpha=lam, max_iter=10000, fit_intercept=True)
    lasso_y.fit(X, y)
    S_y = set(np.where(lasso_y.coef_ != 0)[0])

    lasso_d = Lasso(alpha=lam, max_iter=10000, fit_intercept=True)
    lasso_d.fit(X, d)
    S_d = set(np.where(lasso_d.coef_ != 0)[0])

    S = sorted(S_y | S_d)
    if len(S) == 0:
        return _ols_result(y, d.reshape(-1, 1))
    return _ols_result(y, np.column_stack([d, X[:, S]]))


# ============================================================
# 4. DML-LASSO
# ============================================================
def dml_lasso(X, d, y, n, p, n_folds=5):
    kf  = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    lam = _theoretical_lambda(n, p)
    y_tilde = y - cross_val_predict(Lasso(alpha=lam, max_iter=10000), X, y, cv=kf)
    d_tilde = d - cross_val_predict(Lasso(alpha=lam, max_iter=10000), X, d, cv=kf)
    return _ols_result(y_tilde, d_tilde.reshape(-1, 1))


# ============================================================
# 5. DML-NN
# ============================================================
def dml_nn(X, d, y, n, p, n_folds=5):
    kf    = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    X_sc  = StandardScaler().fit_transform(X)

    _nn = lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu',
        max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1,
    )
    y_tilde = y - cross_val_predict(_nn(), X_sc, y, cv=kf)
    d_tilde = d - cross_val_predict(_nn(), X_sc, d, cv=kf)
    return _ols_result(y_tilde, d_tilde.reshape(-1, 1))


# ============================================================
# Du et al. (2025) Stein-based feature selector
# ============================================================

def du_et_al_selector(X, y, n_screen=None, n_components=None,
                      threshold_factor=0.3, max_select=None, debug=False):
    """
    Stein-based nonlinear feature selection (Du et al. 2025).

    Improvements over naive per-eigenvector thresholding:
      - K determined by eigenvalues clearly above noise (not gap detection alone)
      - Per-feature importance = sum over top-K of |λ_k| * v_k[j]^2
        (single score, no union-of-unions amplification)
      - Hard cap at max_select (default p/4) prevents degenerate selections
      - debug=True prints eigenvalue spectrum and selection diagnostics
    """
    n, p = X.shape

    # ---- screening ----
    if n_screen is None:
        n_screen = min(p, max(10, int(2 * np.sqrt(n * np.log(max(p, 2))))))
    n_screen = min(n_screen, p)

    if n_screen < p:
        corrs      = np.abs(np.corrcoef(X.T, y)[:p, p])
        screen_idx = np.argsort(corrs)[-n_screen:]
    else:
        screen_idx = np.arange(p)

    Xs = X[:, screen_idx]
    ps = Xs.shape[1]

    # ---- Stein moment matrix ----
    M_hat = (Xs.T @ (y[:, None] * Xs)) / n - np.mean(y) * np.eye(ps)
    M_hat = (M_hat + M_hat.T) / 2

    # ---- eigendecompose ----
    eigenvalues, eigenvectors = np.linalg.eigh(M_hat)
    abs_eigs = np.abs(eigenvalues)
    order    = np.argsort(abs_eigs)[::-1]
    sorted_abs = abs_eigs[order]

    # ---- choose K: count eigenvalues clearly above noise ----
    if n_components is None:
        noise_level  = np.median(sorted_abs[ps // 2:]) if ps > 2 else 1e-6
        noise_level  = max(noise_level, 1e-6)
        signal_count = int(np.sum(sorted_abs > 3.0 * noise_level))
        K = max(1, min(signal_count, 10, ps // 2))
    else:
        K = n_components

    # ---- weighted-importance score per feature ----
    importance = np.zeros(ps)
    for idx in order[:K]:
        importance += abs_eigs[idx] * eigenvectors[:, idx] ** 2

    if importance.max() > 0:
        importance = importance / importance.max()

    # ---- select features above threshold ----
    selected_local = np.where(importance > threshold_factor)[0]

    # hard cap on selection size
    if max_select is None:
        max_select = max(5, ps // 4)
    if len(selected_local) > max_select:
        selected_local = np.argsort(importance)[-max_select:]

    # always keep at least 3
    if len(selected_local) < 3:
        selected_local = np.argsort(importance)[-3:]

    if debug:
        print(f"  [Du] n={n} p={p} ps_screened={ps}")
        print(f"  [Du] top 10 |eigs|: {sorted_abs[:10].round(2)}")
        print(f"  [Du] noise level (median bottom half): {noise_level:.3f}")
        print(f"  [Du] K (signal eigenvectors): {K}")
        print(f"  [Du] features above threshold ({threshold_factor}): "
              f"{(importance > threshold_factor).sum()}")
        print(f"  [Du] final selected: {len(selected_local)} / {ps}")

    return sorted(int(screen_idx[i]) for i in selected_local)


# ============================================================
# EPDS shared infrastructure
# ============================================================

EPDS_LOG = []


def _extract_features(eq_sympy, p, debug=False, label=''):
    """
    Extract nonlinear feature terms from a fitted PySR sympy expression.
    Returns sorted list of (kind, i, j) tuples with 0-based original indices.
    Works correctly even when PySR was fitted on a *subset* of features,
    because we use original variable names (x0..xp-1) throughout.
    """
    import sympy as sp

    cols = set()
    eq   = eq_sympy

    sym_to_idx = {sp.Symbol(f'x{i}'): i for i in range(p)}
    half       = sp.Rational(1, 2)

    # -- step A: log(x_i), log(|x_i|), log(x_i^2k) ----------
    for log_node in eq.atoms(sp.log):
        arg = log_node.args[0]
        if arg in sym_to_idx:
            cols.add(('log', sym_to_idx[arg], None))
        elif isinstance(arg, sp.Abs) and arg.args[0] in sym_to_idx:
            cols.add(('log', sym_to_idx[arg.args[0]], None))
        elif isinstance(arg, sp.Pow):
            base, exp = arg.args
            if (base in sym_to_idx and exp.is_integer
                    and exp > 0 and int(exp) % 2 == 0):
                cols.add(('log', sym_to_idx[base], None))

    # -- step B: sqrt(x_i) ------------------------------------
    for pow_node in eq.atoms(sp.Pow):
        base, exp = pow_node.args
        if exp == half:
            if base in sym_to_idx:
                cols.add(('sqrt', sym_to_idx[base], None))
            elif isinstance(base, sp.Abs) and base.args[0] in sym_to_idx:
                cols.add(('sqrt', sym_to_idx[base.args[0]], None))

    # -- step C: polynomial monomials via expand --------------
    try:
        eq_exp = sp.expand(eq)
    except Exception:
        eq_exp = eq

    for term in sp.Add.make_args(eq_exp):
        _, sym_part = term.as_coeff_Mul()
        if sym_part == 1:
            continue

        var_powers  = {}
        has_nonpoly = False

        for factor in sp.Mul.make_args(sym_part):
            if factor in sym_to_idx:
                idx = sym_to_idx[factor]
                var_powers[idx] = var_powers.get(idx, 0) + 1
            elif (isinstance(factor, sp.Pow)
                  and factor.args[0] in sym_to_idx):
                base, exp = factor.args
                if exp.is_integer and int(exp) > 0:
                    idx = sym_to_idx[base]
                    var_powers[idx] = var_powers.get(idx, 0) + int(exp)
                else:
                    has_nonpoly = True
            else:
                has_nonpoly = True

        if has_nonpoly or not var_powers:
            continue

        total_deg = sum(var_powers.values())
        items     = sorted(var_powers.items())

        if total_deg == 1:
            continue
        elif total_deg == 2:
            if len(items) == 1:
                cols.add(('sq', items[0][0], None))
            else:
                i, j = sorted([items[0][0], items[1][0]])
                cols.add(('int', i, j))
        elif total_deg == 3 and len(items) == 2:
            (a, pa), (b, pb) = items
            if pa == 2 and pb == 1:
                cols.add(('cubic', a, b))
            elif pa == 1 and pb == 2:
                cols.add(('cubic', b, a))

    cols = sorted(cols)

    if debug:
        print(f"  [{label}] extracted {len(cols)} terms: {cols}")

    return cols


def _build_nonlinear_cols(term_list, X_in):
    """Construct numpy columns corresponding to a list of (kind,i,j) terms."""
    cols = []
    for kind, i, j in term_list:
        if kind == 'sq':
            cols.append(X_in[:, i] ** 2)
        elif kind == 'int':
            cols.append(X_in[:, i] * X_in[:, j])
        elif kind == 'cubic':
            cols.append(X_in[:, i] ** 2 * X_in[:, j])
        elif kind == 'log':
            cols.append(np.log(np.abs(X_in[:, i]) + 1e-6))
        elif kind == 'sqrt':
            cols.append(np.sqrt(np.abs(X_in[:, i])))
    return cols


def _term_label(term):
    kind, i, j = term
    if kind == 'sq':     return f'x{i}^2'
    if kind == 'int':    return f'x{i}*x{j}'
    if kind == 'cubic':  return f'x{i}^2*x{j}'
    if kind == 'log':    return f'log|x{i}|'
    if kind == 'sqrt':   return f'sqrt|x{i}|'
    return str(term)


def _run_pysr(X_in, y_in, col_names, pysr_iters, debug, label):
    """Fit PySR; return model. X_in may be a subset; col_names must match."""
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations      = pysr_iters,
        binary_operators = ['+', '-', '*'],
        unary_operators  = ['log', 'sqrt'],
        maxsize          = 40,
        parsimony        = 0.001,
        model_selection  = 'best',
        procs            = 0,
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X_in, y_in, variable_names=col_names)

    if debug:
        eqs = model.equations_
        print(f"\n  Pareto ({label}, {len(eqs)} eqs):")
        print(f"  {'idx':>3} {'cplx':>5} {'loss':>12} {'score':>8}  equation")
        for idx, row in eqs.iterrows():
            s = str(row['equation'])[:60]
            print(f"  {idx:>3} {row['complexity']:>5} "
                  f"{row['loss']:>12.5f} {row.get('score',0):>8.4f}  {s}")
        try:
            import sympy as sp
            chosen = model.sympy()
            print(f"  chosen: {chosen}")
            expanded = sp.expand(chosen)
            if expanded != chosen:
                print(f"  expanded: {expanded}")
        except Exception:
            pass

    return model


def _epds_core(
    X_pysr_y, X_pysr_d, X_full, col_names_y, col_names_d,
    d, y, n, p,
    pysr_iters=40, log=False, debug=False, metadata=None,
    variant='epds',
):
    """
    Shared core for epds and epds_du.

    X_pysr_y   : array passed to PySR for the y-equation  (may be a column-subset of X_full)
    X_pysr_d   : array passed to PySR for the d-equation  (may be a column-subset of X_full)
    X_full     : the complete (n,p) feature matrix -- used for the enriched dictionary
    col_names_y: variable names passed to PySR for y (must use original x0..xp-1 names)
    col_names_d: variable names passed to PySR for d
    variant    : 'epds' or 'epds_du' -- logged for recovery analysis
    metadata   : optional dict injected into the EPDS_LOG entry (e.g. {'dgp':'dgp5','rep':3})
    """
    import sympy as sp

    if debug:
        print(f"\n{'='*60}\n  EPDS CORE [{variant}]  n={n} p={p}\n{'='*60}")

    # ---- Step 1: PySR on y and d --------------------------------
    model_y = _run_pysr(X_pysr_y, y, col_names_y, pysr_iters, debug, 'Y')
    model_d = _run_pysr(X_pysr_d, d, col_names_d, pysr_iters, debug, 'D')

    # ---- Step 2: extract nonlinear terms ------------------------
    try:
        terms_y = _extract_features(model_y.sympy(), p, debug, 'Y')
    except Exception:
        terms_y = []
    try:
        terms_d = _extract_features(model_d.sympy(), p, debug, 'D')
    except Exception:
        terms_d = []

    all_terms = sorted(set(terms_y) | set(terms_d))

    if debug:
        print(f"  union terms ({len(all_terms)}): {[_term_label(t) for t in all_terms]}")

    # ---- Step 3: build enriched dictionary on FULL X -----------
    parts  = [X_full]
    nonlin = _build_nonlinear_cols(all_terms, X_full)
    if nonlin:
        parts.extend([c.reshape(-1, 1) for c in nonlin])
    X_star = np.column_stack(parts)

    if debug:
        print(f"  X* shape: {X_star.shape}  (original {p} + {len(all_terms)} NL terms)")

    # ---- Step 4: PDS-LASSO on enriched dictionary ---------------
    p_star   = X_star.shape[1]
    lam_star = _theoretical_lambda(n, p_star)

    lasso_y2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, y)
    lasso_d2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, d)

    S_y = set(np.where(lasso_y2.coef_ != 0)[0])
    S_d = set(np.where(lasso_d2.coef_ != 0)[0])
    S   = sorted(S_y | S_d)

    post_terms_y = [all_terms[i - p] for i in S if i >= p and (i - p) < len(all_terms)]
    post_terms_d = [t for t in post_terms_y if t in terms_d]

    if debug:
        def _lbl(i):
            return f'x{i}' if i < p else _term_label(all_terms[i - p])

        labels_y = [_lbl(i) for i in sorted(S_y)]
        labels_d = [_lbl(i) for i in sorted(S_d)]
        labels_u = [_lbl(i) for i in S]

        print(f"\n  LASSO step 1 -- LASSO(Y ~ X*)   S_y ({len(S_y)} features):")
        print(f"    {labels_y}")
        print(f"\n  LASSO step 2 -- LASSO(D ~ X*)   S_d ({len(S_d)} features):")
        print(f"    {labels_d}")
        print(f"\n  PDS union     -- S = S_y ∪ S_d  ({len(S)} features):")
        print(f"    {labels_u}")
        print(f"  overlap |S_y ∩ S_d| = {len(S_y & S_d)}")

    # ---- Step 5: final OLS + extract per-term coefs -------------
    X_final = d.reshape(-1, 1) if len(S) == 0 else np.column_stack([d, X_star[:, S]])
    X_c     = sm.add_constant(X_final, has_constant='add')
    ols_res = sm.OLS(y, X_c).fit(cov_type='HC1')

    bhat = ols_res.params[1]
    se   = ols_res.HC1_se[1]
    result = {
        'beta_hat': bhat,
        'se'      : se,
        'ci_low'  : bhat - 1.96 * se,
        'ci_high' : bhat + 1.96 * se,
    }

    # per-term OLS coefficients (for coefficient-recovery analysis)
    # ols_res.params: [const, beta_d, coef_S[0], coef_S[1], ...]
    term_coefs_y, term_coefs_d = {}, {}
    if len(S) > 0:
        for offset, lasso_idx in enumerate(S):
            if lasso_idx >= p:
                t    = all_terms[lasso_idx - p]
                coef = float(ols_res.params[offset + 2])
                if t in terms_y:
                    term_coefs_y[t] = coef
                if t in terms_d:
                    term_coefs_d[t] = coef

    # ---- recovery metadata (returned in result dict) ------------
    result.update({
        'epds_variant'       : variant,
        'pre_lasso_terms_y'  : terms_y,
        'pre_lasso_terms_d'  : terms_d,
        'post_lasso_terms_y' : post_terms_y,
        'post_lasso_terms_d' : post_terms_d,
        'term_coefs_y'       : term_coefs_y,
        'term_coefs_d'       : term_coefs_d,
    })

    if debug:
        print(f"  beta_hat={bhat:.4f}  se={se:.4f}")
        if term_coefs_y:
            for t, c in term_coefs_y.items():
                print(f"  coef {_term_label(t)} = {c:.4f}")

    # ---- EPDS_LOG (for interactive use) -------------------------
    if log:
        entry = {
            'variant'            : variant,
            'eq_y'               : str(model_y.sympy()),
            'eq_d'               : str(model_d.sympy()),
            'pre_lasso_terms_y'  : terms_y,
            'pre_lasso_terms_d'  : terms_d,
            'post_lasso_terms_y' : post_terms_y,
            'post_lasso_terms_d' : post_terms_d,
            'term_coefs_y'       : term_coefs_y,
            'term_coefs_d'       : term_coefs_d,
            'n_selected'         : len(S),
            'dict_size'          : p_star,
        }
        if metadata:
            entry.update(metadata)
        EPDS_LOG.append(entry)

    return result


# ============================================================
# 6. EPDS -- no pre-selection
# ============================================================

def epds(X, d, y, n, p, pysr_iters=40, log=False, debug=False, metadata=None):
    """
    Enriched PDS without upstream feature selection.
    PySR sees all p covariates.
    """
    col_names = [f'x{i}' for i in range(p)]
    return _epds_core(
        X_pysr_y=X, X_pysr_d=X, X_full=X,
        col_names_y=col_names, col_names_d=col_names,
        d=d, y=y, n=n, p=p,
        pysr_iters=pysr_iters, log=log, debug=debug,
        metadata=metadata, variant='epds',
    )


# ============================================================
# 7. EPDS-DuEtAl -- Du et al. Stein pre-selection + EPDS
# ============================================================

def epds_du(X, d, y, n, p, pysr_iters=40, log=False, debug=False, metadata=None):
    """
    Enriched PDS with Du et al. (2025) Stein-based upstream selector.
    Reduces the feature space PySR searches over, improving discovery
    at moderate n and reducing spurious terms.

    Du et al. selection is run independently on (X,y) and (X,d);
    PySR sees the union of selected features (with ORIGINAL variable names
    so extract_features correctly maps back to original indices).
    """
    # upstream selection
    S_du_y = du_et_al_selector(X, y)
    S_du_d = du_et_al_selector(X, d)
    S_du   = sorted(set(S_du_y) | set(S_du_d))

    if debug:
        print(f"  Du et al. selected {len(S_du)} / {p} features: {S_du}")

    X_reduced     = X[:, S_du]
    col_names_red = [f'x{i}' for i in S_du]  # ORIGINAL names preserved

    result = _epds_core(
        X_pysr_y=X_reduced, X_pysr_d=X_reduced, X_full=X,
        col_names_y=col_names_red, col_names_d=col_names_red,
        d=d, y=y, n=n, p=p,
        pysr_iters=pysr_iters, log=log, debug=debug,
        metadata=metadata, variant='epds_du',
    )
    result['du_selected'] = S_du
    return result


# ============================================================
# registry
# ============================================================

ESTIMATOR_REGISTRY = {
    'naive_ols' : {'fn': naive_ols,  'label': 'Naive OLS',            'requires_serial': False},
    'full_ols'  : {'fn': full_ols,   'label': 'Full OLS (infeasible)', 'requires_serial': False},
    'pds_lasso' : {'fn': pds_lasso,  'label': 'PDS-LASSO',            'requires_serial': False},
    'dml_lasso' : {'fn': dml_lasso,  'label': 'DML-LASSO',            'requires_serial': False},
    'dml_nn'    : {'fn': dml_nn,     'label': 'DML-NN',               'requires_serial': False},
    'epds'      : {'fn': epds,       'label': 'EPDS',                  'requires_serial': True},
    'epds_du'   : {'fn': epds_du,    'label': 'EPDS + Du et al.',      'requires_serial': True},
}


# ============================================================
# sanity check (skips EPDS to avoid PySR requirement)
# ============================================================
if __name__ == '__main__':
    from dgp import dgp1
    X, d, y, beta0 = dgp1(n=500, p=50, s=6, beta0=0.5, seed=42)
    n, p = X.shape
    print(f"True beta0: {beta0}\n")
    for name, entry in ESTIMATOR_REGISTRY.items():
        if entry['requires_serial']:
            continue
        res    = entry['fn'](X, d, y, n, p)
        covers = res['ci_low'] < beta0 < res['ci_high']
        print(f"{entry['label']:<26} "
              f"beta_hat={res['beta_hat']:.4f}  "
              f"se={res['se']:.4f}  "
              f"CI=[{res['ci_low']:.3f},{res['ci_high']:.3f}]  "
              f"covers={covers}")
