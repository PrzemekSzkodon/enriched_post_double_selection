"""
simulation.py
=============
Monte Carlo simulation loop for the EPDS simulation study.

Runs each (DGP, estimator) combination across n_reps replications
and saves results to CSV.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def _single_rep(dgp_fn, estimator_fn, n, p, s, beta0, seed):
    """Run one replication. Returns dict of results."""
    try:
        X, d, y, beta0 = dgp_fn(n=n, p=p, s=s, beta0=beta0, seed=seed)
        result = estimator_fn(X, d, y, n, p)
        result['seed']   = seed
        result['failed'] = False
        return result
    except Exception as e:
        return {
            'beta_hat' : np.nan,
            'se'       : np.nan,
            'ci_low'   : np.nan,
            'ci_high'  : np.nan,
            'seed'     : seed,
            'failed'   : True,
            'error'    : str(e),
        }


def run_simulation(
    dgp_fn,
    estimator_fn,
    n       = 500,
    p       = 50,
    s       = 5,
    beta0   = 0.5,
    n_reps  = 500,
    n_jobs  = 1,
    base_seed = 0,
):
    """
    Run Monte Carlo simulation.

    Parameters
    ----------
    dgp_fn       : callable  DGP function from dgp.py
    estimator_fn : callable  Estimator function from estimators.py
    n            : int       Sample size
    p            : int       Number of controls
    s            : int       True sparsity
    beta0        : float     True ATE
    n_reps       : int       Number of Monte Carlo replications
    n_jobs       : int       Parallel jobs (-1 = all cores)
    base_seed    : int       Starting seed (rep i uses seed base_seed + i)

    Returns
    -------
    pd.DataFrame with one row per replication
    """
    seeds = range(base_seed, base_seed + n_reps)

    if n_jobs == 1:
        results = [
            _single_rep(dgp_fn, estimator_fn, n, p, s, beta0, seed)
            for seed in tqdm(seeds, desc='Replications')
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_rep)(dgp_fn, estimator_fn, n, p, s, beta0, seed)
            for seed in tqdm(seeds, desc='Replications')
        )

    df = pd.DataFrame(results)

    n_failed = df['failed'].sum()
    if n_failed > 0:
        print(f"Warning: {n_failed}/{n_reps} replications failed")

    return df


def run_all(
    dgp_registry,
    estimator_registry,
    dgp_keys       = None,
    estimator_keys = None,
    n              = 500,
    p              = 50,
    s              = 5,
    beta0          = 0.5,
    n_reps         = 500,
    n_jobs         = 1,
    save_dir       = '../results',
):
    """
    Run all (DGP, estimator) combinations and save results.

    Parameters
    ----------
    dgp_registry       : dict  from dgp.py
    estimator_registry : dict  from estimators.py
    dgp_keys           : list  subset of DGPs to run (None = all)
    estimator_keys     : list  subset of estimators to run (None = all)
    save_dir           : str   directory to save CSV results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    dgp_keys       = dgp_keys       or list(dgp_registry.keys())
    estimator_keys = estimator_keys or list(estimator_registry.keys())

    all_results = []

    for dgp_key in dgp_keys:
        for est_key in estimator_keys:
            dgp_fn  = dgp_registry[dgp_key]['fn']
            est_fn  = estimator_registry[est_key]['fn']
            dgp_lbl = dgp_registry[dgp_key]['label']
            est_lbl = estimator_registry[est_key]['label']

            print(f"\nRunning: {dgp_lbl} × {est_lbl}")

            df = run_simulation(
                dgp_fn, est_fn,
                n=n, p=p, s=s, beta0=beta0,
                n_reps=n_reps, n_jobs=n_jobs,
            )

            df['dgp']       = dgp_key
            df['estimator'] = est_key
            df['dgp_label'] = dgp_lbl
            df['est_label'] = est_lbl
            df['beta0']     = beta0

            fname = f"{save_dir}/{dgp_key}_{est_key}.csv"
            df.to_csv(fname, index=False)
            print(f"  Saved to {fname}")

            all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(f"{save_dir}/all_results.csv", index=False)
    print(f"\nAll results saved to {save_dir}/all_results.csv")

    return combined


# ============================================================
# sanity check
# ============================================================
if __name__ == '__main__':
    from dgp import dgp1
    from estimators import pds_lasso

    print("Running 10 reps of DGP1 x PDS-LASSO as a quick check...\n")

    df = run_simulation(
        dgp_fn       = dgp1,
        estimator_fn = pds_lasso,
        n            = 500,
        p            = 50,
        s            = 5,
        beta0        = 0.5,
        n_reps       = 10,
    )

    print(df[['beta_hat', 'se', 'ci_low', 'ci_high', 'failed']].round(4))