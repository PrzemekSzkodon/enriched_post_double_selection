"""
Causal estimators for the simulation study.

All estimators follow the same interface:

    estimator(X, d, y, n, p) -> dict with keys
        beta_hat, se, ci_low, ci_high

SR-PDS variants additionally return term-level recovery metadata
(pre_lasso_terms_y, post_lasso_terms_y, term_coefs_y, ...) so that the
notebook can compute discovery rates and coefficient bias after the run.

The seven methods compared are:

    full_ols    OLS on (d, X). Infeasible when p >= n; falls back to PDS.
    pds_lasso   Belloni, Chernozhukov, Hansen (2014)
    dml_lasso   DML with LASSO nuisances, Chernozhukov et al. (2018)
    dml_rf      DML with Random Forest nuisances (applied default config)
    dml_nn      DML with MLP(64, 32) nuisances
    sr_pds      Primary contribution -- PySR discovers nonlinear basis
                terms, PDS-LASSO operates on the enriched dictionary.
    sr_pds_cf   Cross-fitted variant. Used as a robustness check.

PySR configuration is in PYSR_CONFIG below. The defaults follow the
dissertation's reported values: 5 operators, max complexity 30,
asymmetric parsimony (outcome 0.005, propensity 0.01), 40 evolutionary
iterations, seed 42.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Central PySR config so all SR-PDS callers stay consistent.
PYSR_CONFIG = {
    'binary_operators': ['+', '-', '*'],
    'unary_operators':  ['log', 'sqrt', 'abs'],
    'maxsize': 30,         # max expression-tree complexity (nodes)
    'parsimony_y': 0.005,  # outcome equation -- stronger signal
    'parsimony_d': 0.01,   # propensity equation -- weaker signal
    'niterations': 40,
    'random_state': 42,
}


# --- helpers ------------------------------------------------------------

def _theoretical_lambda(n, p, c=1.1, alpha=0.05):
    """BCH (2014) plug-in lambda."""
    return (c / np.sqrt(n)) * norm.ppf(1 - alpha / (2 * p))


def _ols_result(y, X_with_d):
    """Robust OLS (HC1) wrapper returning the standard estimator dict."""
    X_c = sm.add_constant(X_with_d, has_constant='add')
    res = sm.OLS(y, X_c).fit(cov_type='HC1')
    bhat = res.params[1]
    se   = res.HC1_se[1]
    return {
        'beta_hat': bhat,
        'se': se,
        'ci_low':  bhat - 1.96 * se,
        'ci_high': bhat + 1.96 * se,
    }


# --- 1. Full OLS (infeasible benchmark) ---------------------------------

def full_ols(X, d, y, n, p):
    """OLS on (d, X). When p >= n the system is singular, so fall back to PDS."""
    if p >= n:
        return pds_lasso(X, d, y, n, p)
    return _ols_result(y, np.column_stack([d, X]))


# --- 2. PDS-LASSO (BCH 2014) --------------------------------------------

def pds_lasso(X, d, y, n, p):
    lam = _theoretical_lambda(n, p)

    lasso_y = Lasso(alpha=lam, max_iter=10000, fit_intercept=True).fit(X, y)
    S_y = set(np.where(lasso_y.coef_ != 0)[0])

    lasso_d = Lasso(alpha=lam, max_iter=10000, fit_intercept=True).fit(X, d)
    S_d = set(np.where(lasso_d.coef_ != 0)[0])

    S = sorted(S_y | S_d)
    if len(S) == 0:
        # selection is empty -- fall back to naive OLS on d only
        return _ols_result(y, d.reshape(-1, 1))
    return _ols_result(y, np.column_stack([d, X[:, S]]))


# --- 3. DML-LASSO -------------------------------------------------------

def dml_lasso(X, d, y, n, p, n_folds=5):
    k = max(2, min(n_folds, n // 2))  # protect against tiny n
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    lam = _theoretical_lambda(n, p)

    y_tilde = y - cross_val_predict(Lasso(alpha=lam, max_iter=10000), X, y, cv=kf)
    d_tilde = d - cross_val_predict(Lasso(alpha=lam, max_iter=10000), X, d, cv=kf)
    return _ols_result(y_tilde, d_tilde.reshape(-1, 1))


# --- 4. DML-RF ----------------------------------------------------------

def dml_rf(X, d, y, n, p, n_folds=5):
    """Standard applied defaults: 500 trees, no max depth, leaf size 5."""
    k = max(2, min(n_folds, n // 2))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    rf = lambda: RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        random_state=42, n_jobs=1,
    )
    y_tilde = y - cross_val_predict(rf(), X, y, cv=kf)
    d_tilde = d - cross_val_predict(rf(), X, d, cv=kf)
    return _ols_result(y_tilde, d_tilde.reshape(-1, 1))


# --- 5. DML-NN ----------------------------------------------------------

def dml_nn(X, d, y, n, p, n_folds=5):
    """(64, 32) ReLU MLP with early stopping. Defaults essentially unchanged."""
    k = max(2, min(n_folds, n // 2))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    X_sc = StandardScaler().fit_transform(X)

    nn = lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu',
        max_iter=500, random_state=42,
        early_stopping=True, validation_fraction=0.1,
    )
    y_tilde = y - cross_val_predict(nn(), X_sc, y, cv=kf)
    d_tilde = d - cross_val_predict(nn(), X_sc, d, cv=kf)
    return _ols_result(y_tilde, d_tilde.reshape(-1, 1))


# --- SR-PDS shared infrastructure ---------------------------------------

# Append-only log of every SR-PDS run. Read out at the end of a simulation
# for recovery analysis. Not fork-safe -- callers must run SR-PDS serially.
SR_PDS_LOG = []
EPDS_LOG   = SR_PDS_LOG   # back-compat alias used by older code


def _term_label(term):
    kind, i, j = term
    if kind == 'sq':    return f'x{i}^2'
    if kind == 'int':   return f'x{i}*x{j}'
    if kind == 'cubic': return f'x{i}^2*x{j}'
    if kind == 'log':   return f'log|x{i}|'
    if kind == 'sqrt':  return f'sqrt|x{i}|'
    return str(term)


def _extract_features(eq_sympy, p, debug=False, label=''):
    """
    Walk a PySR sympy expression and pull out the nonlinear terms in a
    small taxonomy: (sq, int, cubic, log, sqrt). Returns sorted tuples
    of (kind, i, j) using 0-based ORIGINAL feature indices.

    The PySR variable names are kept as x0..xp-1 throughout so this still
    works when PySR was fitted on a column-subset of X. Pure linear
    monomials are deliberately not extracted: they belong to the LASSO
    step, not the enriched dictionary.
    """
    import sympy as sp

    cols = set()
    eq = eq_sympy
    if eq is None:
        return []

    sym_to_idx = {sp.Symbol(f'x{i}'): i for i in range(p)}
    half = sp.Rational(1, 2)

    # log atoms: log(x_i), log(|x_i|), and log(x_i^{2k}) which sympy simplifies
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

    # sqrt: any x^{1/2} either of a bare symbol or of |x|
    for pow_node in eq.atoms(sp.Pow):
        base, exp = pow_node.args
        if exp == half:
            if base in sym_to_idx:
                cols.add(('sqrt', sym_to_idx[base], None))
            elif isinstance(base, sp.Abs) and base.args[0] in sym_to_idx:
                cols.add(('sqrt', sym_to_idx[base.args[0]], None))

    # Polynomial monomials. Expand the expression first so x*(x + y) becomes
    # x^2 + x*y, then iterate over the additive terms and classify each.
    try:
        eq_exp = sp.expand(eq)
    except Exception:
        eq_exp = eq

    for term in sp.Add.make_args(eq_exp):
        _, sym_part = term.as_coeff_Mul()
        if sym_part == 1:
            continue

        var_powers = {}
        has_nonpoly = False

        for factor in sp.Mul.make_args(sym_part):
            if factor in sym_to_idx:
                var_powers[sym_to_idx[factor]] = var_powers.get(sym_to_idx[factor], 0) + 1
            elif isinstance(factor, sp.Pow) and factor.args[0] in sym_to_idx:
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
        items = sorted(var_powers.items())

        if total_deg == 1:
            continue  # pure linear -- not our job
        elif total_deg == 2:
            if len(items) == 1:
                cols.add(('sq', items[0][0], None))
            else:
                i, j = sorted([items[0][0], items[1][0]])
                cols.add(('int', i, j))
        elif total_deg == 3 and len(items) == 2:
            # cubic-style: x_i^2 * x_j
            for idx, pwr in items:
                if pwr == 2:
                    other = [k for k, _ in items if k != idx][0]
                    cols.add(('cubic', idx, other))
                    break
        # higher-order or three-variable terms are dropped -- not in taxonomy

    out = sorted(cols, key=lambda t: (t[0], t[1], t[2] if t[2] is not None else -1))
    if debug:
        print(f"  [{label}] extracted {len(out)}: {[_term_label(t) for t in out]}")
    return out


def _build_nonlinear_cols(terms, X):
    """Materialise the numeric basis columns for a list of term tuples."""
    cols = []
    for kind, i, j in terms:
        if kind == 'sq':
            cols.append(X[:, i] ** 2)
        elif kind == 'int':
            cols.append(X[:, i] * X[:, j])
        elif kind == 'cubic':
            cols.append((X[:, i] ** 2) * X[:, j])
        elif kind == 'log':
            # epsilon to avoid log(0); negligible for |x| >> 1e-6
            cols.append(np.log(np.abs(X[:, i]) + 1e-6))
        elif kind == 'sqrt':
            cols.append(np.sqrt(np.abs(X[:, i]) + 1e-6))
    return cols


def _run_pysr(X_in, y_in, col_names, parsimony,
              debug=False, label='', niterations=None,
              maxsize=None, random_state=None):
    """Fit one PySR model. Defaults read from PYSR_CONFIG."""
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations      = niterations or PYSR_CONFIG['niterations'],
        binary_operators = PYSR_CONFIG['binary_operators'],
        unary_operators  = PYSR_CONFIG['unary_operators'],
        maxsize          = maxsize or PYSR_CONFIG['maxsize'],
        parsimony        = parsimony,
        model_selection  = 'best',
        procs            = 0,
        random_state     = random_state or PYSR_CONFIG['random_state'],
        verbosity        = 0,
    )
    model.fit(X_in, y_in, variable_names=col_names)

    if debug:
        eqs = model.equations_
        print(f"\n  Pareto ({label}, {len(eqs)} eqs):")
        for idx, row in eqs.iterrows():
            print(f"    cplx={row['complexity']:>2}  "
                  f"loss={row['loss']:>10.5f}  "
                  f"{str(row['equation'])[:60]}")
        try:
            print(f"  chosen: {model.sympy()}")
        except Exception:
            pass

    return model


def _sr_pds_core(X_pysr_y, X_pysr_d, X_full,
                 col_names_y, col_names_d,
                 d, y, n, p,
                 log=False, debug=False,
                 metadata=None, variant='sr_pds'):
    """
    The shared core used by sr_pds. (The cross-fit variant has its own loop
    but reuses _extract_features and _build_nonlinear_cols.)

    Steps:
        1. Fit PySR on (X, y) and (X, d).
        2. Extract nonlinear terms from each fitted expression.
        3. Build the enriched dictionary on the full X.
        4. Run PDS-LASSO on that enriched dictionary.
        5. Final OLS on the union-selected features.
        6. Record discovery / coefficient metadata for recovery analysis.
    """
    import sympy as sp  # noqa: F401  -- needed indirectly by _extract_features

    if debug:
        print(f"\n{'=' * 60}\n  SR-PDS [{variant}]  n={n} p={p}\n{'=' * 60}")

    # PySR on outcome and propensity, then pull out the nonlinear terms each found
    model_y = _run_pysr(X_pysr_y, y, col_names_y,
                        PYSR_CONFIG['parsimony_y'], debug=debug, label='Y')
    model_d = _run_pysr(X_pysr_d, d, col_names_d,
                        PYSR_CONFIG['parsimony_d'], debug=debug, label='D')

    try:
        terms_y = _extract_features(model_y.sympy(), p, debug, 'Y')
    except Exception:
        terms_y = []
    try:
        terms_d = _extract_features(model_d.sympy(), p, debug, 'D')
    except Exception:
        terms_d = []

    all_terms = sorted(set(terms_y) | set(terms_d))

    # enriched feature matrix: original X + numeric columns for each discovered term
    parts = [X_full]
    nonlin = _build_nonlinear_cols(all_terms, X_full)
    if nonlin:
        parts.extend([c.reshape(-1, 1) for c in nonlin])
    X_star = np.column_stack(parts)

    # PDS-LASSO step on the enriched dictionary (lambda is scaled to p*)
    p_star = X_star.shape[1]
    lam_star = _theoretical_lambda(n, p_star)
    lasso_y2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, y)
    lasso_d2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, d)

    S_y = set(np.where(lasso_y2.coef_ != 0)[0])
    S_d = set(np.where(lasso_d2.coef_ != 0)[0])
    S = sorted(S_y | S_d)

    # nonlinear terms that survived the union LASSO selection
    survivors = [all_terms[i - p] for i in S
                 if i >= p and (i - p) < len(all_terms)]
    # By construction, post_terms_y must be a subset of pre_lasso_terms_y
    # (= terms_y), and likewise for d. A term that's a true y-equation term
    # but was discovered only by PySR-D belongs in post_terms_d, not _y.
    post_terms_y = [t for t in survivors if t in terms_y]
    post_terms_d = [t for t in survivors if t in terms_d]

    # final OLS for the headline estimate
    X_final = d.reshape(-1, 1) if len(S) == 0 else np.column_stack([d, X_star[:, S]])
    X_c = sm.add_constant(X_final, has_constant='add')
    ols_res = sm.OLS(y, X_c).fit(cov_type='HC1')

    bhat = ols_res.params[1]
    se = ols_res.HC1_se[1]
    out = {
        'beta_hat': bhat,
        'se': se,
        'ci_low':  bhat - 1.96 * se,
        'ci_high': bhat + 1.96 * se,
    }

    # per-term post-LASSO OLS coefficients, for the recovery-analysis plots
    term_coefs_y, term_coefs_d = {}, {}
    for offset, lasso_idx in enumerate(S):
        if lasso_idx >= p:
            t = all_terms[lasso_idx - p]
            c = float(ols_res.params[offset + 2])
            if t in terms_y:
                term_coefs_y[t] = c
            if t in terms_d:
                term_coefs_d[t] = c

    out.update({
        'sr_pds_variant':     variant,
        'epds_variant':       variant,  # back-compat
        'pre_lasso_terms_y':  terms_y,
        'pre_lasso_terms_d':  terms_d,
        'post_lasso_terms_y': post_terms_y,
        'post_lasso_terms_d': post_terms_d,
        'term_coefs_y':       term_coefs_y,
        'term_coefs_d':       term_coefs_d,
    })

    if debug:
        print(f"  beta_hat = {bhat:.4f}  se = {se:.4f}")

    if log:
        entry = {
            'variant': variant,
            'eq_y': str(model_y.sympy()),
            'eq_d': str(model_d.sympy()),
            'pre_lasso_terms_y':  terms_y,
            'pre_lasso_terms_d':  terms_d,
            'post_lasso_terms_y': post_terms_y,
            'post_lasso_terms_d': post_terms_d,
            'term_coefs_y': term_coefs_y,
            'term_coefs_d': term_coefs_d,
            'n_selected': len(S),
            'dict_size': p_star,
        }
        if metadata:
            entry.update(metadata)
        SR_PDS_LOG.append(entry)

    return out


# --- 6. SR-PDS (primary method) -----------------------------------------

def sr_pds(X, d, y, n, p, log=False, debug=False, metadata=None):
    """PySR sees the full X; downstream PDS-LASSO works on the enriched dictionary."""
    col_names = [f'x{i}' for i in range(p)]
    return _sr_pds_core(
        X_pysr_y=X, X_pysr_d=X, X_full=X,
        col_names_y=col_names, col_names_d=col_names,
        d=d, y=y, n=n, p=p,
        log=log, debug=debug,
        metadata=metadata, variant='sr_pds',
    )


# --- 7. SR-PDS-CF (cross-fitted; robustness check) ----------------------

def sr_pds_cf(X, d, y, n, p, n_folds=5, log=False, debug=False, metadata=None):
    """
    Cross-fitted SR-PDS.

    For each fold k:
        - fit PySR on the K-1 training folds (twice -- one per equation)
        - extract the discovered nonlinear terms
        - fit LASSO on the same training folds for y and d
        - predict on the held-out fold to get residuals

    Concatenate residuals across folds, then OLS for the final estimate.
    This restores the kind of sample-splitting validity DML relies on, at
    K-times the PySR compute cost. Used as a robustness check on the
    same-data SR-PDS, not as the headline procedure.
    """
    k = max(2, min(n_folds, n // 10))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    col_names = [f'x{i}' for i in range(p)]

    y_tilde = np.zeros(n)
    d_tilde = np.zeros(n)
    pre_y, pre_d, post_y, post_d = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        d_tr, d_te = d[train_idx], d[test_idx]
        n_tr = len(train_idx)

        try:
            mdl_y = _run_pysr(X_tr, y_tr, col_names,
                              PYSR_CONFIG['parsimony_y'],
                              debug=debug, label=f'Y/f{fold_idx}')
            t_y_k = _extract_features(mdl_y.sympy(), p, debug, 'Y')
        except Exception:
            t_y_k = []
        try:
            mdl_d = _run_pysr(X_tr, d_tr, col_names,
                              PYSR_CONFIG['parsimony_d'],
                              debug=debug, label=f'D/f{fold_idx}')
            t_d_k = _extract_features(mdl_d.sympy(), p, debug, 'D')
        except Exception:
            t_d_k = []

        pre_y.extend(t_y_k)
        pre_d.extend(t_d_k)
        terms_k = sorted(set(t_y_k) | set(t_d_k))

        # build enriched matrices on train and test sets using this fold's terms
        parts_tr = [X_tr]
        parts_te = [X_te]
        nl_tr = _build_nonlinear_cols(terms_k, X_tr)
        nl_te = _build_nonlinear_cols(terms_k, X_te)
        if nl_tr:
            parts_tr.extend([c.reshape(-1, 1) for c in nl_tr])
            parts_te.extend([c.reshape(-1, 1) for c in nl_te])
        X_star_tr = np.column_stack(parts_tr)
        X_star_te = np.column_stack(parts_te)

        lam_k = _theoretical_lambda(n_tr, X_star_tr.shape[1])
        ly = Lasso(alpha=lam_k, max_iter=10000).fit(X_star_tr, y_tr)
        ld = Lasso(alpha=lam_k, max_iter=10000).fit(X_star_tr, d_tr)

        y_tilde[test_idx] = y_te - ly.predict(X_star_te)
        d_tilde[test_idx] = d_te - ld.predict(X_star_te)

        # NL terms that survived the per-fold LASSO selection
        S_k = set(np.where(ly.coef_ != 0)[0]) | set(np.where(ld.coef_ != 0)[0])
        survivors_k = [terms_k[i - p] for i in S_k
                       if i >= p and (i - p) < len(terms_k)]
        # strict subset relationship: post must come from the equation's own
        # PySR discovery
        post_y.extend([t for t in survivors_k if t in t_y_k])
        post_d.extend([t for t in survivors_k if t in t_d_k])

    out = _ols_result(y_tilde, d_tilde.reshape(-1, 1))

    # aggregate term sets across folds via union (a term counts as "found"
    # if at least one fold discovered it)
    out.update({
        'sr_pds_variant':     'sr_pds_cf',
        'epds_variant':       'sr_pds_cf',
        'pre_lasso_terms_y':  sorted(set(pre_y)),
        'pre_lasso_terms_d':  sorted(set(pre_d)),
        'post_lasso_terms_y': sorted(set(post_y)),
        'post_lasso_terms_d': sorted(set(post_d)),
        'term_coefs_y': {},  # not well-defined under cross-fitting
        'term_coefs_d': {},
    })

    if log:
        entry = {
            'variant': 'sr_pds_cf',
            'pre_lasso_terms_y':  out['pre_lasso_terms_y'],
            'pre_lasso_terms_d':  out['pre_lasso_terms_d'],
            'post_lasso_terms_y': out['post_lasso_terms_y'],
            'post_lasso_terms_d': out['post_lasso_terms_d'],
            'n_folds': k,
        }
        if metadata:
            entry.update(metadata)
        SR_PDS_LOG.append(entry)

    return out


# --- registry -----------------------------------------------------------

ESTIMATOR_REGISTRY = {
    'full_ols':   {'fn': full_ols,  'label': 'Full OLS',     'requires_serial': False},
    'pds_lasso':  {'fn': pds_lasso, 'label': 'PDS-LASSO',    'requires_serial': False},
    'dml_lasso':  {'fn': dml_lasso, 'label': 'DML-LASSO',    'requires_serial': False},
    'dml_rf':     {'fn': dml_rf,    'label': 'DML-RF',       'requires_serial': False},
    'dml_nn':     {'fn': dml_nn,    'label': 'DML-NN',       'requires_serial': False},
    'sr_pds':     {'fn': sr_pds,    'label': 'SR-PDS',       'requires_serial': True},
    'sr_pds_cf':  {'fn': sr_pds_cf, 'label': 'SR-PDS (CF)',  'requires_serial': True},
}


if __name__ == '__main__':
    # smoke test the non-PySR estimators on DGP-1
    from dgp import dgp1
    X, d, y, beta0 = dgp1(n=500, p=50, s=6, beta0=0.5, seed=42)
    print(f"True beta0: {beta0}\n")
    for name, entry in ESTIMATOR_REGISTRY.items():
        if entry['requires_serial']:
            continue
        r = entry['fn'](X, d, y, X.shape[0], X.shape[1])
        covers = r['ci_low'] < beta0 < r['ci_high']
        print(f"{entry['label']:<22}  beta_hat={r['beta_hat']:+.4f}  "
              f"se={r['se']:.4f}  covers={covers}")
