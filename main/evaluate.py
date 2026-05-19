"""
evaluate.py
===========
Compute simulation metrics from Monte Carlo results.

Metrics:
    bias     : E[beta_hat - beta0]
    rmse     : sqrt(E[(beta_hat - beta0)^2])
    coverage : P(beta0 in 95% CI)
    size     : P(reject H0: beta0=0 | beta0=0)  -- only meaningful when true beta0=0
    median   : median(beta_hat)
    std      : std(beta_hat)
"""

import numpy as np
import pandas as pd


def evaluate(results_df, beta0):
    """
    Compute metrics for a single (DGP, estimator) simulation result.

    Parameters
    ----------
    results_df : pd.DataFrame  output of run_simulation()
    beta0      : float         true ATE

    Returns
    -------
    dict of metric values
    """
    df = results_df.dropna(subset=['beta_hat'])

    bias     = (df['beta_hat'] - beta0).mean()
    rmse     = np.sqrt(((df['beta_hat'] - beta0) ** 2).mean())
    coverage = ((df['ci_low'] < beta0) & (beta0 < df['ci_high'])).mean()
    size     = ((df['ci_low'] > 0) | (df['ci_high'] < 0)).mean()
    median   = df['beta_hat'].median()
    std      = df['beta_hat'].std()
    n_reps   = len(df)

    return {
        'bias'     : round(bias,     4),
        'rmse'     : round(rmse,     4),
        'coverage' : round(coverage, 4),
        'size'     : round(size,     4),
        'median'   : round(median,   4),
        'std'      : round(std,      4),
        'n_reps'   : n_reps,
    }


def evaluate_all(combined_df, beta0):
    """
    Compute metrics for all (DGP, estimator) combinations
    from the combined results DataFrame.

    Parameters
    ----------
    combined_df : pd.DataFrame  output of run_all()
    beta0       : float         true ATE

    Returns
    -------
    pd.DataFrame with one row per (DGP, estimator) combination
    """
    rows = []

    for (dgp_key, est_key), group in combined_df.groupby(['dgp', 'estimator']):
        metrics = evaluate(group, beta0)
        metrics['dgp']       = dgp_key
        metrics['estimator'] = est_key
        metrics['dgp_label'] = group['dgp_label'].iloc[0]
        metrics['est_label'] = group['est_label'].iloc[0]
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df[['dgp', 'dgp_label', 'estimator', 'est_label',
             'bias', 'rmse', 'coverage', 'size', 'std', 'n_reps']]
    df = df.sort_values(['dgp', 'estimator']).reset_index(drop=True)

    return df


def summary_table(metrics_df, metric='rmse', round_digits=4):
    """
    Pivot metrics into a publication-ready table.
    Rows = DGPs, Columns = estimators.

    Parameters
    ----------
    metrics_df  : pd.DataFrame  output of evaluate_all()
    metric      : str           which metric to show
    round_digits: int           decimal places

    Returns
    -------
    pd.DataFrame pivot table
    """
    pivot = metrics_df.pivot(
        index   = 'dgp_label',
        columns = 'est_label',
        values  = metric,
    ).round(round_digits)

    pivot.index.name   = 'DGP'
    pivot.columns.name = None

    return pivot


def print_summary(metrics_df, beta0):
    """Print a clean summary of all metrics."""
    print(f"\nTrue ATE: {beta0}")
    print("=" * 80)

    for metric in ['bias', 'rmse', 'coverage', 'size']:
        print(f"\n{metric.upper()}")
        print("-" * 60)
        tbl = summary_table(metrics_df, metric=metric)
        print(tbl.to_string())

    print("\n" + "=" * 80)


# ============================================================
# sanity check
# ============================================================
if __name__ == '__main__':
    from dgp import dgp1
    from estimators import pds_lasso, naive_ols
    from simulation import run_simulation

    beta0 = 0.5

    results_pds   = run_simulation(dgp1, pds_lasso,  n_reps=50)
    results_naive = run_simulation(dgp1, naive_ols,  n_reps=50)

    results_pds['dgp']       = 'dgp1'
    results_pds['estimator'] = 'pds_lasso'
    results_pds['dgp_label'] = 'Linear sparse'
    results_pds['est_label'] = 'PDS-LASSO'
    results_pds['beta0']     = beta0

    results_naive['dgp']       = 'dgp1'
    results_naive['estimator'] = 'naive_ols'
    results_naive['dgp_label'] = 'Linear sparse'
    results_naive['est_label'] = 'Naive OLS'
    results_naive['beta0']     = beta0

    combined = pd.concat([results_pds, results_naive])
    metrics  = evaluate_all(combined, beta0)

    print_summary(metrics, beta0)