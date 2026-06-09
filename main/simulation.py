"""
Monte Carlo driver for the simulation study.

Two entry points are typically called from a notebook:
    run_simulation       -- one (DGP, estimator) cell
    run_all              -- iterate over a grid of cells, save CSVs

The SR-PDS variants log their PySR discoveries to estimators.SR_PDS_LOG so
that downstream recovery analysis can compute per-term recovery rates.
This log is non-fork-safe, so SR-PDS jobs are forced to run serially even
when the rest of the simulation is parallel via joblib.

A small helper seed_robustness_check at the bottom runs SR-PDS multiple
times on one fixed dataset with different PySR seeds; it isn't part of
the main grid.
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def _single_rep(dgp_fn, estimator_fn, n, p, s, beta0, seed, estimator_key=None):
    """One Monte Carlo replication. Catches exceptions and records failure."""
    try:
        X, d, y, _ = dgp_fn(n=n, p=p, s=s, beta0=beta0, seed=seed)
        result = estimator_fn(X, d, y, n, p)

        row = {
            'seed': seed,
            'failed': False,
            'beta_hat': result['beta_hat'],
            'se': result['se'],
            'ci_low': result['ci_low'],
            'ci_high': result['ci_high'],
        }
        # carry through any SR-PDS recovery metadata that came back
        meta_keys = ('sr_pds_variant', 'epds_variant',
                     'pre_lasso_terms_y', 'pre_lasso_terms_d',
                     'post_lasso_terms_y', 'post_lasso_terms_d',
                     'term_coefs_y', 'term_coefs_d')
        for k in meta_keys:
            if k in result:
                row[k] = result[k]
        return row

    except Exception as e:
        # graceful: never crash the outer loop on a single bad rep
        return {
            'seed': seed, 'failed': True,
            'error': str(e)[:200],
            'beta_hat': np.nan, 'se': np.nan,
            'ci_low': np.nan,  'ci_high': np.nan,
        }


def run_simulation(dgp_fn, estimator_fn,
                   estimator_key=None,
                   n=500, p=50, s=6, beta0=0.5,
                   n_reps=500, n_jobs=1, base_seed=0,
                   dgp_key=None):
    """
    Monte Carlo loop for one (DGP, estimator).

    For SR-PDS variants the estimator is wrapped so that each rep records
    its own log entry tagged with {dgp, rep}. PySR is not fork-safe, so
    SR-PDS variants must be called with n_jobs=1.
    """
    is_sr_pds = estimator_key in ('sr_pds', 'sr_pds_cf', 'epds', 'epds_du')

    def _wrap(seed):
        # add per-rep metadata so we can attribute log entries later
        if is_sr_pds and dgp_key is not None:
            meta = {'dgp': dgp_key, 'rep': seed}
            return lambda X, d, y, n_, p_: estimator_fn(
                X, d, y, n_, p_, log=True, metadata=meta)
        return estimator_fn

    seeds = list(range(base_seed, base_seed + n_reps))

    if n_jobs == 1:
        results = []
        for seed in tqdm(seeds, desc='reps', leave=False):
            results.append(_single_rep(dgp_fn, _wrap(seed),
                                       n, p, s, beta0, seed, estimator_key))
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_rep)(dgp_fn, estimator_fn,
                                 n, p, s, beta0, seed, estimator_key)
            for seed in tqdm(seeds, desc='reps', leave=False)
        )

    df = pd.DataFrame(results)
    if df['failed'].sum() > 0:
        print(f"  WARNING: {df['failed'].sum()}/{n_reps} reps failed")
    return df


def run_all(dgp_registry, estimator_registry,
            dgp_keys=None, estimator_keys=None,
            n=500, p=50, s=6, beta0=0.5,
            n_reps=500, n_jobs=1,
            save_dir='../results',
            reps_overrides=None):
    """
    Iterate over (DGP x estimator) cells.

    reps_overrides : optional dict {estimator_key: n_reps_for_that_estimator}
                     useful for downweighting expensive cells (e.g. sr_pds_cf
                     can run with fewer reps than the rest).
    """
    from estimators import SR_PDS_LOG

    os.makedirs(save_dir, exist_ok=True)
    dgp_keys       = dgp_keys or list(dgp_registry.keys())
    estimator_keys = estimator_keys or list(estimator_registry.keys())
    reps_overrides = reps_overrides or {}

    SR_PDS_LOG.clear()

    all_dfs = []
    t_start = time.time()

    for dgp_key in dgp_keys:
        for est_key in estimator_keys:
            dgp_entry = dgp_registry[dgp_key]
            est_entry = estimator_registry[est_key]

            is_serial = est_entry.get('requires_serial', False)
            n_jobs_eff = 1 if is_serial else n_jobs
            reps_eff   = reps_overrides.get(est_key, n_reps)

            elapsed = (time.time() - t_start) / 60
            print(f"\n[{elapsed:5.1f} min]  "
                  f"{dgp_entry['label']} x {est_entry['label']}  "
                  f"(n_jobs={n_jobs_eff}, n_reps={reps_eff})")

            df = run_simulation(
                dgp_entry['fn'], est_entry['fn'],
                estimator_key=est_key,
                n=n, p=p, s=s, beta0=beta0,
                n_reps=reps_eff, n_jobs=n_jobs_eff,
                dgp_key=dgp_key,
            )

            df['dgp'] = dgp_key
            df['estimator'] = est_key
            df['dgp_label'] = dgp_entry['label']
            df['est_label'] = est_entry['label']
            df['beta0']     = beta0

            # serialise term-set columns for CSV (they're lists / dicts)
            list_cols = ('pre_lasso_terms_y', 'pre_lasso_terms_d',
                         'post_lasso_terms_y', 'post_lasso_terms_d',
                         'term_coefs_y', 'term_coefs_d')
            df_csv = df.copy()
            for c in list_cols:
                if c in df_csv.columns:
                    df_csv[c] = df_csv[c].apply(str)
            df_csv.to_csv(f"{save_dir}/{dgp_key}_{est_key}.csv", index=False)

            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = combined.copy()
    for c in ('pre_lasso_terms_y', 'pre_lasso_terms_d',
              'post_lasso_terms_y', 'post_lasso_terms_d',
              'term_coefs_y', 'term_coefs_d'):
        if c in combined_csv.columns:
            combined_csv[c] = combined_csv[c].apply(str)
    combined_csv.to_csv(f"{save_dir}/all_results.csv", index=False)
    print(f"\nAll results saved to {save_dir}/all_results.csv")

    log_path = f"{save_dir}/sr_pds_log.pkl"
    with open(log_path, 'wb') as f:
        pickle.dump(list(SR_PDS_LOG), f)
    print(f"SR-PDS log ({len(SR_PDS_LOG)} entries) saved to {log_path}")
    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")

    return combined


# --- seed-robustness helper ---------------------------------------------

def seed_robustness_check(dgp_fn, dgp_key='dgp6',
                          n=500, p=50, s=6, beta0=0.5,
                          data_seed=42, pysr_seeds=None,
                          save_dir=None):
    """
    Generate one dataset, run SR-PDS with several PySR seeds, see whether
    the discovered expressions and beta_hat are stable.

    Not part of the main grid -- call this once to back up the
    reproducibility claim in the dissertation.
    """
    from estimators import _sr_pds_core, PYSR_CONFIG

    if pysr_seeds is None:
        pysr_seeds = list(range(1, 11))

    X, d, y, _ = dgp_fn(n=n, p=p, s=s, beta0=beta0, seed=data_seed)
    col_names  = [f'x{i}' for i in range(p)]

    rows = []
    for seed in tqdm(pysr_seeds, desc='seeds'):
        # temporarily swap in this PySR seed
        original = PYSR_CONFIG['random_state']
        PYSR_CONFIG['random_state'] = seed
        try:
            res = _sr_pds_core(
                X_pysr_y=X, X_pysr_d=X, X_full=X,
                col_names_y=col_names, col_names_d=col_names,
                d=d, y=y, n=n, p=p,
                log=False, debug=False, variant='sr_pds',
            )
            row = {
                'pysr_seed': seed,
                'beta_hat': res['beta_hat'],
                'se': res['se'],
                'pre_lasso_terms_y':  tuple(res['pre_lasso_terms_y']),
                'pre_lasso_terms_d':  tuple(res['pre_lasso_terms_d']),
                'post_lasso_terms_y': tuple(res['post_lasso_terms_y']),
                'post_lasso_terms_d': tuple(res['post_lasso_terms_d']),
            }
        except Exception as e:
            row = {'pysr_seed': seed, 'beta_hat': np.nan,
                   'error': str(e)[:100]}
        finally:
            PYSR_CONFIG['random_state'] = original

        rows.append(row)

    out = pd.DataFrame(rows)
    out['dgp'] = dgp_key
    out['n']   = n
    out['beta0'] = beta0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_csv = out.copy()
        for c in out_csv.columns:
            if out_csv[c].dtype == object and len(out_csv[c]) > 0:
                out_csv[c] = out_csv[c].apply(
                    lambda x: str(x) if isinstance(x, (tuple, list)) else x)
        out_csv.to_csv(f"{save_dir}/seed_robustness.csv", index=False)
        print(f"Saved seed_robustness to {save_dir}/seed_robustness.csv")

    return out


if __name__ == '__main__':
    from dgp import dgp1
    from estimators import pds_lasso

    df = run_simulation(dgp1, pds_lasso, estimator_key='pds_lasso',
                        n=200, p=50, s=6, beta0=0.5, n_reps=5)
    print(df[['beta_hat', 'se', 'failed']].round(4))
