"""
Improved cross-fitted SR-PDS, with per-fold recovery logging.

Coverage fix (unchanged from the previous version):
  1. Honest cross-fit variance. The treatment effect is the cross-fitted
     orthogonal-moment estimator and its standard error is the influence-
     function estimate (Chernozhukov et al., 2018), not a single HC1 OLS on the
     stacked out-of-fold residuals. This accounts for the fold structure and is
     the main coverage fix on the severe design.
  2. Per-fold recovery levers via ``cf_config`` (larger niterations, relaxed
     parsimony_d, more folds) so high-complexity terms are recovered on more
     folds.

New here: per-fold recovery.
  For each replication we record, per discovered term, the fraction of folds in
  which it was recovered (pre-LASSO) and kept (post-LASSO). A term found in 3 of
  5 folds is recorded as 0.6, not collapsed to 1.0 by a union over folds. These
  are returned as ``fold_pre_{y,d}`` / ``fold_post_{y,d}`` dicts keyed by the
  string form of the term, so downstream analysis can average them across reps.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from estimators import (
    PYSR_CONFIG,
    SR_PDS_LOG,
    _run_pysr,
    _extract_features,
    _build_nonlinear_cols,
    _theoretical_lambda,
)


CF_CONFIG = {
    'n_folds':      10,
    'niterations':  100,
    'parsimony_d':  0.0035,
    'parsimony_y':  None,    # None -> inherit PYSR_CONFIG['parsimony_y']
}


def _fit_fold_residuals(X_tr, y_tr, d_tr, X_te, y_te, d_te,
                        n_tr, p, col_names, cfg, debug=False, fold_tag=''):
    """Discover on the training portion, residualise the held-out portion.

    Returns (y_resid_te, d_resid_te, terms_y, terms_d, survivors) where terms_*
    are the terms PySR discovered on this fold and survivors are the discovered
    terms that this fold's enriched LASSO kept.
    """
    pars_y = cfg['parsimony_y'] or PYSR_CONFIG['parsimony_y']
    pars_d = cfg['parsimony_d'] or PYSR_CONFIG['parsimony_d']
    niter  = cfg['niterations'] or PYSR_CONFIG['niterations']

    try:
        mdl_y = _run_pysr(X_tr, y_tr, col_names, pars_y,
                          debug=debug, label=f'Y/{fold_tag}', niterations=niter)
        terms_y = _extract_features(mdl_y.sympy(), p, debug, 'Y')
    except Exception:
        terms_y = []
    try:
        mdl_d = _run_pysr(X_tr, d_tr, col_names, pars_d,
                          debug=debug, label=f'D/{fold_tag}', niterations=niter)
        terms_d = _extract_features(mdl_d.sympy(), p, debug, 'D')
    except Exception:
        terms_d = []

    terms_k = sorted(set(terms_y) | set(terms_d))

    parts_tr, parts_te = [X_tr], [X_te]
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

    y_resid = y_te - ly.predict(X_star_te)
    d_resid = d_te - ld.predict(X_star_te)

    S_k = set(np.where(ly.coef_ != 0)[0]) | set(np.where(ld.coef_ != 0)[0])
    survivors = [terms_k[i - p] for i in S_k
                 if i >= p and (i - p) < len(terms_k)]
    return y_resid, d_resid, terms_y, terms_d, survivors


def sr_pds_cf(X, d, y, n, p, n_folds=None,
              cf_config=None, log=False, debug=False, metadata=None):
    """Cross-fitted SR-PDS with honest cross-fit variance and per-fold recovery."""
    cfg = dict(CF_CONFIG)
    if cf_config:
        cfg.update(cf_config)
    if n_folds is not None:
        cfg['n_folds'] = n_folds

    k = max(2, min(cfg['n_folds'], n // 10))
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    col_names = [f'x{i}' for i in range(p)]

    y_tilde = np.zeros(n)
    d_tilde = np.zeros(n)
    pre_y, pre_d, post_y, post_d = [], [], [], []

    # per-fold recovery counters: how many folds discovered / kept each term
    fp_y, fp_d, fk_y, fk_d = Counter(), Counter(), Counter(), Counter()

    for fold_idx, (tr, te) in enumerate(kf.split(X)):
        yr, dr, t_y, t_d, surv = _fit_fold_residuals(
            X[tr], y[tr], d[tr], X[te], y[te], d[te],
            len(tr), p, col_names, cfg, debug=debug, fold_tag=f'f{fold_idx}')
        y_tilde[te] = yr
        d_tilde[te] = dr

        sty, std_, ssurv = set(t_y), set(t_d), set(surv)
        pre_y.extend(t_y); pre_d.extend(t_d)
        post_y.extend([t for t in surv if t in sty])
        post_d.extend([t for t in surv if t in std_])
        for t in sty:                fp_y[t] += 1
        for t in std_:               fp_d[t] += 1
        for t in (sty & ssurv):      fk_y[t] += 1
        for t in (std_ & ssurv):     fk_d[t] += 1

    # honest cross-fit moment estimate + influence-function SE
    Edd = np.mean(d_tilde ** 2)
    beta_hat = np.mean(d_tilde * y_tilde) / Edd
    psi = d_tilde * (y_tilde - beta_hat * d_tilde) / Edd
    se = np.sqrt(np.mean(psi ** 2) / n)

    # per-fold recovery as a fraction of folds (e.g. 3 of 5 -> 0.6)
    fold_pre_y = {str(t): c / k for t, c in fp_y.items()}
    fold_pre_d = {str(t): c / k for t, c in fp_d.items()}
    fold_post_y = {str(t): c / k for t, c in fk_y.items()}
    fold_post_d = {str(t): c / k for t, c in fk_d.items()}

    out = {
        'beta_hat': float(beta_hat),
        'se':       float(se),
        'ci_low':   float(beta_hat - 1.96 * se),
        'ci_high':  float(beta_hat + 1.96 * se),
        'sr_pds_variant':     'sr_pds_cf',
        'epds_variant':       'sr_pds_cf',
        'pre_lasso_terms_y':  sorted(set(pre_y)),
        'pre_lasso_terms_d':  sorted(set(pre_d)),
        'post_lasso_terms_y': sorted(set(post_y)),
        'post_lasso_terms_d': sorted(set(post_d)),
        'fold_pre_y':  fold_pre_y,    # term -> fraction of folds that DISCOVERED it
        'fold_pre_d':  fold_pre_d,
        'fold_post_y': fold_post_y,   # term -> fraction of folds that KEPT it
        'fold_post_d': fold_post_d,
        'n_folds': k,
        'term_coefs_y': {},
        'term_coefs_d': {},
    }

    if log:
        entry = {
            'variant': 'sr_pds_cf',
            'pre_lasso_terms_y':  out['pre_lasso_terms_y'],
            'pre_lasso_terms_d':  out['pre_lasso_terms_d'],
            'post_lasso_terms_y': out['post_lasso_terms_y'],
            'post_lasso_terms_d': out['post_lasso_terms_d'],
            'fold_pre_y':  fold_pre_y, 'fold_pre_d':  fold_pre_d,
            'fold_post_y': fold_post_y, 'fold_post_d': fold_post_d,
            'n_folds': k,
            'cf_config': {key: cfg[key] for key in
                          ('n_folds', 'niterations', 'parsimony_d', 'parsimony_y')},
        }
        if metadata:
            entry.update(metadata)
        SR_PDS_LOG.append(entry)

    return out
