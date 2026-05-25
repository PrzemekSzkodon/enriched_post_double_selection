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

def epds(X, d, y, n, p, pysr_iters=40, log=False, debug=False):
    """
    Enriched Post-Double Selection (EPDS).

    Step 1: PySR on (y ~ X) and (d ~ X) -- full sample
    Step 2: Extract exact terms PySR found
    Step 3: Build x* = all original linear terms + exact PySR terms
    Step 4: PDS-LASSO on x* -- full sample
    Step 5: OLS on full sample -> valid beta_hat

    Parameters
    ----------
    debug : bool
        If True, prints every step to stdout: Pareto fronts, chosen
        equations, extracted terms, dictionary composition, selected set.
        Use in notebooks for live inspection.
    log : bool
        If True, appends a structured record to EPDS_LOG.
    """
    try:
        from pysr import PySRRegressor
        import sympy as sp
    except ImportError:
        raise ImportError("PySR not installed. Run: pip install pysr")

    col_names = [f'x{i}' for i in range(p)]

    def _print_header(title):
        if debug:
            print(f"\n{'=' * 70}")
            print(f"  {title}")
            print(f"{'=' * 70}")

    def _print_step(msg):
        if debug:
            print(f"  → {msg}")

    _print_header("EPDS START")
    if debug:
        print(f"  n = {n}, p = {p}, pysr_iters = {pysr_iters}")

    # ============================================================
    # STEP 1: run PySR on (X, y) and (X, d)
    # ============================================================
    def run_pysr(X_in, y_in, label):
        _print_header(f"STEP 1{label}: PySR fit for {label}")
        model = PySRRegressor(
            niterations      = pysr_iters,
            binary_operators = ['+', '-', '*'],
            unary_operators  = ['square', 'log', 'sqrt'],
            maxsize          = 50,
            parsimony        = 0.005,
            model_selection  = 'best',
            procs            = 0,
            random_state     = 42,
            verbosity        = 0,
        )
        model.fit(X_in, y_in, variable_names=col_names)

        if debug:
            eqs = model.equations_
            print(f"\n  Pareto frontier ({len(eqs)} equations):")
            print(f"  {'idx':>3} {'complex':>7} {'loss':>12} {'score':>8}  equation")
            print(f"  {'-'*3} {'-'*7} {'-'*12} {'-'*8}  {'-'*40}")
            for idx, row in eqs.iterrows():
                eq_str = str(row['equation'])
                if len(eq_str) > 60:
                    eq_str = eq_str[:57] + '...'
                print(f"  {idx:>3} {row['complexity']:>7} "
                      f"{row['loss']:>12.6f} {row.get('score', 0):>8.4f}  {eq_str}")

            best_idx = _selected_idx(model, eqs)

            try:
                chosen_eq = model.sympy()
                print(f"\n  Chosen equation (model_selection='best'):")
                print(f"    raw:      {chosen_eq}")
                try:
                    chosen_expanded = sp.expand(chosen_eq)
                    if chosen_expanded != chosen_eq:
                        print(f"    expanded: {chosen_expanded}")
                except Exception:
                    pass
                print(f"    complexity = {eqs.loc[best_idx, 'complexity']}, "
                      f"loss = {eqs.loc[best_idx, 'loss']:.6f}")
            except Exception as e:
                print(f"  WARNING: could not extract sympy form: {e}")

        return model

    def model_selection_is_loss(model):
        return getattr(model, 'model_selection', 'best') == 'accuracy'

    def _selected_idx(model, eqs):
        try:
            chosen_sympy = model.sympy()
            for idx, row in eqs.iterrows():
                if str(row['equation']).strip() == str(chosen_sympy).strip():
                    return idx
        except Exception:
            pass
        return eqs['loss'].idxmin()

    model_y = run_pysr(X, y, label='Y')
    model_d = run_pysr(X, d, label='D')

    # ============================================================
    # STEP 2: extract exact terms (uses the robust extractor)
    # ============================================================
    _print_header("STEP 2: extract symbolic terms from PySR equations")

    def extract_features(model, label):
        cols = set()

        try:
            eq = model.sympy()
        except Exception as e:
            if debug:
                print(f"  [{label}] failed to extract sympy form: {e}")
            return []

        if debug:
            print(f"\n  [{label}] raw equation:    {eq}")
            try:
                eq_expanded = sp.expand(eq)
                print(f"  [{label}] expanded form:   {eq_expanded}")
            except Exception:
                eq_expanded = eq

        sym_to_idx = {sp.Symbol(f'x{i}'): i for i in range(p)}
        half = sp.Rational(1, 2)

        # ---- step A: capture log(x_i) anywhere in the tree ----
        for log_node in eq.atoms(sp.log):
            arg = log_node.args[0]
            if arg in sym_to_idx:                                       # log(x_i)
                cols.add(('log', sym_to_idx[arg], None))
            elif isinstance(arg, sp.Abs) and arg.args[0] in sym_to_idx: # log(|x_i|)
                cols.add(('log', sym_to_idx[arg.args[0]], None))
            elif isinstance(arg, sp.Pow):                               # log(x_i**k) — k even ⟹ log|x_i| up to constant
                base, exp = arg.args
                if base in sym_to_idx and exp.is_integer and exp > 0 and exp % 2 == 0:
                    cols.add(('log', sym_to_idx[base], None))

        # step B: sqrt
        for pow_node in eq.atoms(sp.Pow):
            base, exp = pow_node.args
            if exp == half:
                if base in sym_to_idx:
                    cols.add(('sqrt', sym_to_idx[base], None))
                elif isinstance(base, sp.Abs) and base.args[0] in sym_to_idx:
                    cols.add(('sqrt', sym_to_idx[base.args[0]], None))

        # step C: polynomial monomials
        try:
            eq_expanded = sp.expand(eq)
        except Exception:
            eq_expanded = eq

        for term in sp.Add.make_args(eq_expanded):
            coeff, sym_part = term.as_coeff_Mul()
            if sym_part == 1:
                continue

            var_powers = {}
            has_nonpoly = False

            for factor in sp.Mul.make_args(sym_part):
                if factor in sym_to_idx:
                    idx = sym_to_idx[factor]
                    var_powers[idx] = var_powers.get(idx, 0) + 1
                elif isinstance(factor, sp.Pow) and factor.args[0] in sym_to_idx:
                    base, exp = factor.args
                    if exp.is_integer and exp > 0:
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
            print(f"  [{label}] extracted {len(cols)} terms:")
            if not cols:
                print(f"    (none)")
            for term in cols:
                kind, i, j = term
                if kind == 'sq':     print(f"    x{i}^2")
                elif kind == 'int':  print(f"    x{i} * x{j}")
                elif kind == 'cubic':print(f"    x{i}^2 * x{j}")
                elif kind == 'log':  print(f"    log(x{i})")
                elif kind == 'sqrt': print(f"    sqrt(x{i})")

        return cols

    terms_y = extract_features(model_y, label='Y')
    terms_d = extract_features(model_d, label='D')

    all_terms = sorted(set(terms_y) | set(terms_d))
    if debug:
        print(f"\n  Union of terms: {len(all_terms)} "
              f"(Y-only: {len(set(terms_y) - set(terms_d))}, "
              f"D-only: {len(set(terms_d) - set(terms_y))}, "
              f"both: {len(set(terms_y) & set(terms_d))})")

    # ============================================================
    # STEP 3: build enriched feature dictionary
    # ============================================================
    _print_header("STEP 3: build enriched feature dictionary")

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

    def build_dict(X_in):
        parts = [X_in]
        nonlin = build_features(all_terms, X_in)
        if nonlin:
            parts.extend([f.reshape(-1, 1) if f.ndim == 1 else f
                          for f in nonlin])
        return np.column_stack(parts)

    X_star = build_dict(X)

    if debug:
        print(f"  original X:     shape ({n}, {p})")
        print(f"  added terms:    {len(all_terms)}")
        print(f"  enriched X*:    shape {X_star.shape}")
        print(f"  → dictionary went from {p} to {X_star.shape[1]} features")

    # ============================================================
    # STEP 4: PDS-LASSO on enriched dictionary
    # ============================================================
    _print_header("STEP 4: PDS-LASSO on enriched dictionary")

    p_star   = X_star.shape[1]
    lam_star = _theoretical_lambda(n, p_star)

    if debug:
        print(f"  p* = {p_star}, lambda* = {lam_star:.6f}")

    lasso_y2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, y)
    lasso_d2 = Lasso(alpha=lam_star, max_iter=10000).fit(X_star, d)

    S_y = set(np.where(lasso_y2.coef_ != 0)[0])
    S_d = set(np.where(lasso_d2.coef_ != 0)[0])
    S   = sorted(S_y | S_d)

    if debug:
        def _label_feature(idx):
            if idx < p:
                return f"x{idx}"
            term = all_terms[idx - p]
            kind, i, j = term
            if kind == 'sq':     return f"x{i}^2"
            elif kind == 'int':  return f"x{i}*x{j}"
            elif kind == 'cubic':return f"x{i}^2*x{j}"
            elif kind == 'log':  return f"log(x{i})"
            elif kind == 'sqrt': return f"sqrt(x{i})"

        print(f"\n  LASSO(Y ~ X*) selected {len(S_y)} features:")
        print(f"    {[_label_feature(i) for i in sorted(S_y)]}")
        print(f"\n  LASSO(D ~ X*) selected {len(S_d)} features:")
        print(f"    {[_label_feature(i) for i in sorted(S_d)]}")
        print(f"\n  Union (S_y ∪ S_d): {len(S)} features:")
        print(f"    {[_label_feature(i) for i in S]}")

    # ============================================================
    # log (structured record)
    # ============================================================
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

    # ============================================================
    # STEP 5: final OLS on full sample
    # ============================================================
    _print_header("STEP 5: final OLS on full sample")

    if len(S) == 0:
        if debug:
            print(f"  empty selection — falling back to y ~ d only")
        X_final = d.reshape(-1, 1)
    else:
        X_final = np.column_stack([d, X_star[:, S]])
        if debug:
            print(f"  regressing y on d + {len(S)} selected features")
            print(f"  final design matrix: shape {X_final.shape}")

    result = _ols_result(y, X_final)

    if debug:
        print(f"\n  RESULT:")
        print(f"    beta_hat = {result['beta_hat']:.6f}")
        print(f"    se       = {result['se']:.6f}")
        print(f"    95% CI   = [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
        _print_header("EPDS END")

    return result

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