"""
Metrics and recovery statistics for the simulation study.

Two layers of evaluation:
  - standard metrics on beta_hat / CIs (evaluate, evaluate_all)
  - term-level recovery from the SR-PDS log (evaluate_recovery)

The recovery layer reads pickled log entries written by SR-PDS during
simulation. Each entry contains the variant ('sr_pds' or 'sr_pds_cf'),
the rep's DGP, and the pre/post-LASSO term sets PySR discovered.
"""

import numpy as np
import pandas as pd


# --- standard metrics ----------------------------------------------------

def evaluate(results_df, beta0):
    """Bias / RMSE / coverage / size / median / std for one (DGP, estimator) cell."""
    df = results_df.dropna(subset=['beta_hat'])
    if len(df) == 0:
        return {k: np.nan for k in
                ('bias', 'rmse', 'coverage', 'size', 'median', 'std', 'n_reps')}

    err      = df['beta_hat'] - beta0
    bias     = err.mean()
    rmse     = np.sqrt((err ** 2).mean())
    coverage = ((df['ci_low'] < beta0) & (beta0 < df['ci_high'])).mean()
    size     = ((df['ci_low'] > 0) | (df['ci_high'] < 0)).mean()

    return {
        'bias': round(bias, 4),
        'rmse': round(rmse, 4),
        'coverage': round(coverage, 4),
        'size': round(size, 4),
        'median': round(df['beta_hat'].median(), 4),
        'std': round(df['beta_hat'].std(), 4),
        'n_reps': len(df),
    }


def evaluate_all(combined_df, beta0):
    """Apply evaluate() to every (DGP, estimator) group; return a tidy DataFrame."""
    rows = []
    for (dgp_key, est_key), group in combined_df.groupby(['dgp', 'estimator']):
        m = evaluate(group, beta0)
        m['dgp'] = dgp_key
        m['estimator'] = est_key
        m['dgp_label'] = group['dgp_label'].iloc[0]
        m['est_label'] = group['est_label'].iloc[0]
        rows.append(m)

    cols = ['dgp', 'dgp_label', 'estimator', 'est_label',
            'bias', 'rmse', 'coverage', 'size', 'std', 'n_reps']
    return (pd.DataFrame(rows)[cols]
              .sort_values(['dgp', 'estimator'])
              .reset_index(drop=True))


def summary_table(metrics_df, metric='rmse', round_digits=4):
    """Pivot one metric to a (DGP x estimator) table for printing."""
    out = metrics_df.pivot(index='dgp_label', columns='est_label',
                           values=metric).round(round_digits)
    out.index.name = 'DGP'
    out.columns.name = None
    return out


def print_summary(metrics_df, beta0):
    print(f"\nTrue ATE: {beta0}")
    print("=" * 80)
    for m in ('bias', 'rmse', 'coverage', 'size'):
        print(f"\n{m.upper()}")
        print("-" * 60)
        print(summary_table(metrics_df, m).to_string())
    print("\n" + "=" * 80)


# --- term-level recovery -------------------------------------------------

def _term_label(term):
    kind, i, j = term
    if kind == 'sq':    return f'x{i}^2'
    if kind == 'int':   return f'x{i}*x{j}'
    if kind == 'cubic': return f'x{i}^2*x{j}'
    if kind == 'log':   return f'log|x{i}|'
    if kind == 'sqrt':  return f'sqrt|x{i}|'
    return str(term)


def _coef_stats(vals, truth):
    """Return (mean, std, bias_vs_truth, rmse_vs_truth). Empty -> NaNs."""
    if not vals:
        return np.nan, np.nan, np.nan, np.nan
    arr = np.array(vals)
    if np.isnan(truth):
        return arr.mean(), arr.std(), np.nan, np.nan
    return (arr.mean(),
            arr.std(),
            arr.mean() - truth,
            np.sqrt(((arr - truth) ** 2).mean()))


def evaluate_recovery(sr_pds_log, dgp_registry):
    """
    Term-level recovery stats from a list of SR-PDS log entries.

    For every (DGP, variant, equation, true_term) we compute:
        recovery_rate -- fraction of reps in which PySR found the term
        and at the post-LASSO stage, mean / bias / RMSE of the
        recovered OLS coefficient against the DGP's true value.

    Output is a long DataFrame with one row per (..., stage) so the
    'pre_lasso' and 'post_lasso' results can be plotted side by side.
    """
    if not sr_pds_log:
        return pd.DataFrame()

    # (eq_tag, truth_terms_key, truth_coefs_key, pre_key, post_key, coefs_key)
    equations = [
        ('y', 'truth_terms_y', 'truth_coefs_y',
         'pre_lasso_terms_y', 'post_lasso_terms_y', 'term_coefs_y'),
        ('d', 'truth_terms_d', 'truth_coefs_d',
         'pre_lasso_terms_d', 'post_lasso_terms_d', 'term_coefs_d'),
    ]

    rows = []
    for dgp_key, dgp_entry in dgp_registry.items():
        dgp_log = [e for e in sr_pds_log if e.get('dgp') == dgp_key]
        if not dgp_log:
            continue

        # there may be several variants per DGP (sr_pds, sr_pds_cf)
        variants = sorted({e.get('variant', 'sr_pds') for e in dgp_log})

        for eq_tag, t_terms_k, t_coefs_k, pre_k, post_k, coefs_k in equations:
            truth_terms = dgp_entry.get(t_terms_k, [])
            truth_coefs = dgp_entry.get(t_coefs_k, {})
            if not truth_terms:
                continue

            for variant in variants:
                var_log = [e for e in dgp_log
                           if e.get('variant', 'sr_pds') == variant]
                if not var_log:
                    continue
                n_reps = len(var_log)

                for term in truth_terms:
                    truth = truth_coefs.get(term, np.nan)
                    pre   = [int(term in (e.get(pre_k)  or [])) for e in var_log]
                    post  = [int(term in (e.get(post_k) or [])) for e in var_log]
                    coefs = [(e.get(coefs_k) or {}).get(term, np.nan)
                             for e in var_log]
                    coefs = [c for c in coefs if not np.isnan(c)]

                    mean_c, std_c, bias_c, rmse_c = _coef_stats(coefs, truth)

                    for stage, found in (('pre_lasso', pre),
                                         ('post_lasso', post)):
                        rows.append({
                            'dgp': dgp_key,
                            'dgp_label': dgp_entry['label'],
                            'variant': variant,
                            'equation': eq_tag,
                            'term': term,
                            'term_label': _term_label(term),
                            'stage': stage,
                            'recovery_rate': float(np.mean(found)),
                            'n_reps': n_reps,
                            'true_coef': truth,
                            # only report coef stats at the post-LASSO stage
                            'mean_coef': mean_c if stage == 'post_lasso' else np.nan,
                            'std_coef':  std_c  if stage == 'post_lasso' else np.nan,
                            'coef_bias': bias_c if stage == 'post_lasso' else np.nan,
                            'coef_rmse': rmse_c if stage == 'post_lasso' else np.nan,
                        })

    return pd.DataFrame(rows)


def recovery_summary_table(recovery_df, stage='post_lasso', equation='y'):
    """Pivot recovery rates to (DGP|term) x variant for heatmaps."""
    sub = recovery_df[(recovery_df['stage'] == stage)
                      & (recovery_df['equation'] == equation)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub['row_label'] = sub['dgp_label'] + ' | ' + sub['term_label']
    out = (sub.pivot_table(index='row_label', columns='variant',
                           values='recovery_rate', aggfunc='mean')
              .round(3))
    out.index.name = f'DGP | term ({equation}-eq, {stage})'
    out.columns.name = 'variant'
    return out


# --- quick smoke test ----------------------------------------------------

if __name__ == '__main__':
    from dgp import dgp1
    from estimators import pds_lasso
    from simulation import run_simulation

    df = run_simulation(dgp1, pds_lasso, estimator_key='pds_lasso', n_reps=20)
    df['dgp'] = 'dgp1'
    df['estimator'] = 'pds_lasso'
    df['dgp_label'] = 'Linear'
    df['est_label'] = 'PDS-LASSO'

    metrics = evaluate_all(df, beta0=0.5)
    print_summary(metrics, beta0=0.5)
