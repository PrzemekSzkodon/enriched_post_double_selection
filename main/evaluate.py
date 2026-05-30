"""
evaluate.py
===========
Compute simulation metrics and recovery statistics.

Standard metrics (from beta_hat/CI results):
    bias, rmse, coverage, size, median, std

Recovery metrics (from EPDS_LOG):
    pre_recovery_y  : P(truth term found by PySR in y-equation)
    post_recovery_y : P(truth term survived LASSO in y-equation)
    pre_recovery_d  : P(truth term found by PySR in d-equation)
    post_recovery_d : P(truth term survived LASSO in d-equation)
    coef_bias_y     : mean(recovered coef - true coef) for y-terms
    coef_rmse_y     : rmse of recovered coefficient
"""

import numpy as np
import pandas as pd


# ============================================================
# standard metrics
# ============================================================

def evaluate(results_df, beta0):
    df = results_df.dropna(subset=['beta_hat'])
    if len(df) == 0:
        return {k: np.nan for k in
                ('bias','rmse','coverage','size','median','std','n_reps')}
    bias     = (df['beta_hat'] - beta0).mean()
    rmse     = np.sqrt(((df['beta_hat'] - beta0) ** 2).mean())
    coverage = ((df['ci_low'] < beta0) & (beta0 < df['ci_high'])).mean()
    size     = ((df['ci_low'] > 0) | (df['ci_high'] < 0)).mean()
    return {
        'bias'    : round(bias,     4),
        'rmse'    : round(rmse,     4),
        'coverage': round(coverage, 4),
        'size'    : round(size,     4),
        'median'  : round(df['beta_hat'].median(), 4),
        'std'     : round(df['beta_hat'].std(),    4),
        'n_reps'  : len(df),
    }


def evaluate_all(combined_df, beta0):
    rows = []
    for (dgp_key, est_key), group in combined_df.groupby(['dgp', 'estimator']):
        metrics              = evaluate(group, beta0)
        metrics['dgp']       = dgp_key
        metrics['estimator'] = est_key
        metrics['dgp_label'] = group['dgp_label'].iloc[0]
        metrics['est_label'] = group['est_label'].iloc[0]
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df[['dgp','dgp_label','estimator','est_label',
             'bias','rmse','coverage','size','std','n_reps']]
    return df.sort_values(['dgp','estimator']).reset_index(drop=True)


def summary_table(metrics_df, metric='rmse', round_digits=4):
    pivot = metrics_df.pivot(
        index='dgp_label', columns='est_label', values=metric
    ).round(round_digits)
    pivot.index.name   = 'DGP'
    pivot.columns.name = None
    return pivot


def print_summary(metrics_df, beta0):
    print(f"\nTrue ATE: {beta0}")
    print("=" * 80)
    for metric in ['bias', 'rmse', 'coverage', 'size']:
        print(f"\n{metric.upper()}")
        print("-" * 60)
        print(summary_table(metrics_df, metric).to_string())
    print("\n" + "=" * 80)


# ============================================================
# recovery metrics (from EPDS_LOG)
# ============================================================

def _term_label(term):
    kind, i, j = term
    if kind == 'sq':    return f'x{i}^2'
    if kind == 'int':   return f'x{i}*x{j}'
    if kind == 'cubic': return f'x{i}^2*x{j}'
    if kind == 'log':   return f'log|x{i}|'
    if kind == 'sqrt':  return f'sqrt|x{i}|'
    return str(term)


def evaluate_recovery(epds_log, dgp_registry):
    """
    Compute term-level recovery statistics from EPDS_LOG.

    Parameters
    ----------
    epds_log     : list of dicts -- contents of estimators.EPDS_LOG
                   each entry must have keys 'dgp', 'variant',
                   'pre_lasso_terms_y', 'post_lasso_terms_y',
                   'pre_lasso_terms_d', 'post_lasso_terms_d',
                   'term_coefs_y', 'term_coefs_d'
    dgp_registry : dict from dgp.py -- must have 'truth_terms_y', 'truth_terms_d',
                   'truth_coefs_y', 'truth_coefs_d' for each DGP key

    Returns
    -------
    pd.DataFrame with one row per (dgp, variant, equation, term, stage)
    Columns: dgp, dgp_label, variant, equation, term_label, stage,
             recovery_rate, mean_coef, std_coef, true_coef, coef_bias, coef_rmse
    """
    if not epds_log:
        return pd.DataFrame()

    rows = []

    for dgp_key, dgp_entry in dgp_registry.items():
        for eq_tag, truth_terms_key, truth_coefs_key, pre_key, post_key, coefs_key in [
            ('y', 'truth_terms_y', 'truth_coefs_y',
             'pre_lasso_terms_y', 'post_lasso_terms_y', 'term_coefs_y'),
            ('d', 'truth_terms_d', 'truth_coefs_d',
             'pre_lasso_terms_d', 'post_lasso_terms_d', 'term_coefs_d'),
        ]:
            truth_terms = dgp_entry.get(truth_terms_key, [])
            truth_coefs = dgp_entry.get(truth_coefs_key, {})

            if not truth_terms:
                continue

            # filter log for this DGP
            dgp_log = [e for e in epds_log if e.get('dgp') == dgp_key]
            if not dgp_log:
                continue

            # get unique variants
            variants = sorted(set(e.get('variant', 'epds') for e in dgp_log))

            for variant in variants:
                var_log = [e for e in dgp_log
                           if e.get('variant', 'epds') == variant]
                n_reps  = len(var_log)
                if n_reps == 0:
                    continue

                for term in truth_terms:
                    true_coef  = truth_coefs.get(term, np.nan)
                    term_lbl   = _term_label(term)

                    pre_found  = [int(term in (e.get(pre_key)  or []))
                                  for e in var_log]
                    post_found = [int(term in (e.get(post_key) or []))
                                  for e in var_log]
                    recovered_coefs = [
                        (e.get(coefs_key) or {}).get(term, np.nan)
                        for e in var_log
                    ]
                    valid_coefs = [c for c in recovered_coefs
                                   if not np.isnan(c)]

                    def _safe_stats(vals, true_val):
                        if not vals:
                            return np.nan, np.nan, np.nan, np.nan
                        arr  = np.array(vals)
                        mean = arr.mean()
                        std  = arr.std()
                        if np.isnan(true_val):
                            return mean, std, np.nan, np.nan
                        bias = mean - true_val
                        rmse = np.sqrt(((arr - true_val)**2).mean())
                        return mean, std, bias, rmse

                    mean_c, std_c, bias_c, rmse_c = _safe_stats(
                        valid_coefs, true_coef)

                    for stage, found in [('pre_lasso', pre_found),
                                         ('post_lasso', post_found)]:
                        rows.append({
                            'dgp'          : dgp_key,
                            'dgp_label'    : dgp_entry['label'],
                            'variant'      : variant,
                            'equation'     : eq_tag,
                            'term'         : term,
                            'term_label'   : term_lbl,
                            'stage'        : stage,
                            'recovery_rate': np.mean(found),
                            'n_reps'       : n_reps,
                            'true_coef'    : true_coef,
                            'mean_coef'    : mean_c if stage == 'post_lasso' else np.nan,
                            'std_coef'     : std_c  if stage == 'post_lasso' else np.nan,
                            'coef_bias'    : bias_c if stage == 'post_lasso' else np.nan,
                            'coef_rmse'    : rmse_c if stage == 'post_lasso' else np.nan,
                        })

    return pd.DataFrame(rows)


def recovery_summary_table(recovery_df, stage='post_lasso', equation='y'):
    """
    Pivot recovery rates: rows = (DGP, term), columns = variant.
    Useful for heatmap input.
    """
    sub = recovery_df[
        (recovery_df['stage'] == stage) &
        (recovery_df['equation'] == equation)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    sub['row_label'] = sub['dgp_label'] + ' | ' + sub['term_label']

    pivot = sub.pivot_table(
        index='row_label', columns='variant',
        values='recovery_rate', aggfunc='mean'
    ).round(3)
    pivot.index.name   = f'DGP | term ({equation}-eq, {stage})'
    pivot.columns.name = 'variant'
    return pivot


# ============================================================
# sanity check
# ============================================================
if __name__ == '__main__':
    from dgp import dgp1
    from estimators import pds_lasso, naive_ols
    from simulation import run_simulation

    beta0 = 0.5
    dfs   = []
    for fn, key, lbl, dgp_lbl in [
        (naive_ols,  'naive_ols',  'Naive OLS',  'Linear'),
        (pds_lasso,  'pds_lasso',  'PDS-LASSO',  'Linear'),
    ]:
        df = run_simulation(dgp1, fn, estimator_key=key, n_reps=20)
        df['dgp']='dgp1'; df['estimator']=key
        df['dgp_label']=dgp_lbl; df['est_label']=lbl; df['beta0']=beta0
        dfs.append(df)

    metrics = evaluate_all(pd.concat(dfs), beta0)
    print_summary(metrics, beta0)
