"""
simulation.py
=============
Monte Carlo simulation loop for the EPDS simulation study.

Key behaviours:
  - EPDS variants (requires_serial=True) always run with n_jobs=1
    because PySR uses its own multiprocessing internally and EPDS_LOG
    is a global list that is not fork-safe.
  - EPDS estimators receive a metadata dict {'dgp': key, 'rep': seed}
    so EPDS_LOG entries are automatically annotated for recovery analysis.
  - run_all saves EPDS_LOG to a pickle alongside the main CSV results.
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


# ============================================================
# single replication
# ============================================================

def _single_rep(dgp_fn, estimator_fn, n, p, s, beta0, seed,
                estimator_key=None):
    """
    Run one replication.  Returns a flat dict suitable for a DataFrame row.
    For EPDS variants the estimator_fn should already have metadata injected
    via a functools.partial or lambda.
    """
    try:
        X, d, y, beta0_actual = dgp_fn(n=n, p=p, s=s, beta0=beta0, seed=seed)
        result = estimator_fn(X, d, y, n, p)

        row = {
            'seed'   : seed,
            'failed' : False,
            'beta_hat': result['beta_hat'],
            'se'     : result['se'],
            'ci_low' : result['ci_low'],
            'ci_high': result['ci_high'],
        }

        # pull recovery metadata for EPDS variants
        for key in ('epds_variant',
                    'pre_lasso_terms_y', 'pre_lasso_terms_d',
                    'post_lasso_terms_y', 'post_lasso_terms_d',
                    'term_coefs_y', 'term_coefs_d',
                    'du_selected'):
            if key in result:
                row[key] = result[key]

        return row

    except Exception as e:
        return {
            'seed'   : seed,
            'failed' : True,
            'error'  : str(e)[:200],
            'beta_hat': np.nan, 'se': np.nan,
            'ci_low' : np.nan,  'ci_high': np.nan,
        }


# ============================================================
# run_simulation
# ============================================================

def run_simulation(
    dgp_fn,
    estimator_fn,
    estimator_key = None,
    n             = 500,
    p             = 50,
    s             = 6,
    beta0         = 0.5,
    n_reps        = 500,
    n_jobs        = 1,
    base_seed     = 0,
    dgp_key       = None,      # injected into EPDS metadata if provided
):
    """
    Run Monte Carlo simulation for one (DGP, estimator) combination.

    For EPDS variants, passes metadata={'dgp': dgp_key, 'rep': seed}
    so EPDS_LOG entries carry DGP and rep information.
    """
    from estimators import ESTIMATOR_REGISTRY

    # For EPDS variants, wrap the estimator to inject metadata
    is_epds = (estimator_key in ('epds', 'epds_du'))

    def _make_fn(seed):
        if is_epds and dgp_key is not None:
            meta = {'dgp': dgp_key, 'rep': seed}
            return lambda X, d, y, n, p: estimator_fn(
                X, d, y, n, p, log=True, metadata=meta)
        return estimator_fn

    seeds = list(range(base_seed, base_seed + n_reps))

    if n_jobs == 1:
        results = []
        for seed in tqdm(seeds, desc='reps', leave=False):
            fn  = _make_fn(seed)
            row = _single_rep(dgp_fn, fn, n, p, s, beta0, seed, estimator_key)
            results.append(row)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_rep)(dgp_fn, estimator_fn, n, p, s, beta0, seed, estimator_key)
            for seed in tqdm(seeds, desc='reps', leave=False)
        )

    df = pd.DataFrame(results)
    n_failed = df['failed'].sum()
    if n_failed > 0:
        print(f"  WARNING: {n_failed}/{n_reps} reps failed")

    return df


# ============================================================
# run_all
# ============================================================

def run_all(
    dgp_registry,
    estimator_registry,
    dgp_keys       = None,
    estimator_keys = None,
    n              = 500,
    p              = 50,
    s              = 6,
    beta0          = 0.5,
    n_reps         = 500,
    n_jobs         = 1,
    save_dir       = '../results',
):
    """
    Run all (DGP, estimator) combinations.

    EPDS variants always use n_jobs=1 (PySR is not fork-safe).
    Saves:
      - {save_dir}/{dgp}_{estimator}.csv  per combination
      - {save_dir}/all_results.csv        combined flat file
      - {save_dir}/epds_log.pkl           EPDS_LOG with dgp/rep annotations
    """
    from estimators import EPDS_LOG

    os.makedirs(save_dir, exist_ok=True)

    dgp_keys       = dgp_keys       or list(dgp_registry.keys())
    estimator_keys = estimator_keys or list(estimator_registry.keys())

    # clear EPDS_LOG before the full run so we only keep this run's entries
    EPDS_LOG.clear()

    all_dfs = []
    t_start = time.time()

    for dgp_key in dgp_keys:
        for est_key in estimator_keys:
            dgp_entry  = dgp_registry[dgp_key]
            est_entry  = estimator_registry[est_key]
            dgp_fn     = dgp_entry['fn']
            est_fn     = est_entry['fn']
            dgp_lbl    = dgp_entry['label']
            est_lbl    = est_entry['label']
            is_serial  = est_entry.get('requires_serial', False)
            eff_jobs   = 1 if is_serial else n_jobs

            elapsed = (time.time() - t_start) / 60
            print(f"\n[{elapsed:5.1f} min]  {dgp_lbl} × {est_lbl}"
                  f"  (n_jobs={eff_jobs})")

            df = run_simulation(
                dgp_fn, est_fn,
                estimator_key = est_key,
                n=n, p=p, s=s, beta0=beta0,
                n_reps=n_reps, n_jobs=eff_jobs,
                dgp_key=dgp_key,
            )

            df['dgp']       = dgp_key
            df['estimator'] = est_key
            df['dgp_label'] = dgp_lbl
            df['est_label'] = est_lbl
            df['beta0']     = beta0

            # serialise list/dict columns to string for CSV
            df_csv = df.copy()
            for col in ('pre_lasso_terms_y', 'pre_lasso_terms_d',
                        'post_lasso_terms_y', 'post_lasso_terms_d',
                        'term_coefs_y', 'term_coefs_d', 'du_selected'):
                if col in df_csv.columns:
                    df_csv[col] = df_csv[col].apply(str)

            fname = f"{save_dir}/{dgp_key}_{est_key}.csv"
            df_csv.to_csv(fname, index=False)

            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # save combined CSV (list columns as strings)
    combined_csv = combined.copy()
    for col in ('pre_lasso_terms_y', 'pre_lasso_terms_d',
                'post_lasso_terms_y', 'post_lasso_terms_d',
                'term_coefs_y', 'term_coefs_d', 'du_selected'):
        if col in combined_csv.columns:
            combined_csv[col] = combined_csv[col].apply(str)
    combined_csv.to_csv(f"{save_dir}/all_results.csv", index=False)
    print(f"\nAll results saved to {save_dir}/all_results.csv")

    # save EPDS_LOG as pickle (preserves lists, dicts, tuples)
    log_path = f"{save_dir}/epds_log.pkl"
    with open(log_path, 'wb') as f:
        pickle.dump(EPDS_LOG, f)
    print(f"EPDS log ({len(EPDS_LOG)} entries) saved to {log_path}")

    total_min = (time.time() - t_start) / 60
    print(f"\nTotal time: {total_min:.1f} min")

    return combined


# ============================================================
# sanity check
# ============================================================
if __name__ == '__main__':
    from dgp import dgp1
    from estimators import pds_lasso, naive_ols

    df = run_simulation(dgp1, pds_lasso, estimator_key='pds_lasso',
                        n=200, p=50, s=6, beta0=0.5, n_reps=5)
    print(df[['beta_hat', 'se', 'failed']].round(4))
