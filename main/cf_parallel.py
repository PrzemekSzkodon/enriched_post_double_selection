"""
Process-parallel driver for the cross-fit estimator.

The cross-fit reps are independent, so we run each in its own process with
joblib's loky backend. Process isolation (rather than forking) sidesteps the
two reasons the in-place simulation harness runs SR-PDS serially: PySR's Julia
backend is not fork-safe, and the shared ``SR_PDS_LOG`` global cannot be
written from parallel workers. Each worker instead *returns* its estimate and
its discovered terms, which the parent collects.

A per-replication heartbeat (count, rate, ETA) prints as each rep finishes, so
every design shows live progress, and it is visible that the reps complete
faster than wall-clock-serial would allow. Each worker pins its BLAS / Julia
thread count to 1 so rep-level parallelism does not oversubscribe the cores
against PySR's own threading.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _cf_one_rep(dgp_fn, seed, n, p, s, beta0, cf_config):
    for var in ('JULIA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = '1'

    from srpds_cf import sr_pds_cf  # imported inside the fresh worker process

    X, d, y, _ = dgp_fn(n=n, p=p, s=s, beta0=beta0, seed=seed)
    try:
        r = sr_pds_cf(X, d, y, n, p, cf_config=cf_config, log=False)
        return {
            'seed': seed, 'failed': False,
            'beta_hat': r['beta_hat'], 'se': r['se'],
            'ci_low': r['ci_low'], 'ci_high': r['ci_high'],
            'pre_lasso_terms_y':  r['pre_lasso_terms_y'],
            'pre_lasso_terms_d':  r['pre_lasso_terms_d'],
            'post_lasso_terms_y': r['post_lasso_terms_y'],
            'post_lasso_terms_d': r['post_lasso_terms_d'],
            'fold_pre_y':  r.get('fold_pre_y', {}),
            'fold_pre_d':  r.get('fold_pre_d', {}),
            'fold_post_y': r.get('fold_post_y', {}),
            'fold_post_d': r.get('fold_post_d', {}),
            'n_folds': r.get('n_folds'),
        }
    except Exception as e:
        return {'seed': seed, 'failed': True, 'error': str(e)[:200],
                'beta_hat': np.nan, 'se': np.nan,
                'ci_low': np.nan, 'ci_high': np.nan}


def _heartbeat(done, total, t0, desc):
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    left = (total - done) / rate / 60 if rate > 0 else float('nan')
    sys.stdout.write(
        f"\r  [{desc}] {done:>3}/{total} reps  "
        f"({rate*60:4.1f}/min, {left:4.1f} min left)   ")
    sys.stdout.flush()


def run_cf_parallel(dgp_fn, n_reps, n, p, s, beta0,
                    cf_config, n_jobs, base_seed=0, desc='CF'):
    """Run ``n_reps`` cross-fit replications in parallel; return a DataFrame.

    Prints a live heartbeat as each rep returns. One row per rep with beta_hat /
    CIs and the discovered term sets, seeded ``base_seed .. base_seed + n_reps -
    1`` so the draws match the serial harness.
    """
    seeds = list(range(base_seed, base_seed + n_reps))
    t0 = time.time()
    results = []

    # Preferred path: joblib >= 1.3 streams each result as a worker finishes it,
    # so the heartbeat is genuinely live (one tick per completed rep).
    try:
        gen = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
            delayed(_cf_one_rep)(dgp_fn, sd, n, p, s, beta0, cf_config)
            for sd in seeds)
        for r in gen:
            results.append(r)
            _heartbeat(len(results), n_reps, t0, desc)
        print()
        return pd.DataFrame(results)
    except TypeError:
        # older joblib: no return_as -> run in chunks so the heartbeat still moves
        pass

    chunk = max(1, n_jobs)
    for i in range(0, n_reps, chunk):
        batch = seeds[i:i + chunk]
        out = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_cf_one_rep)(dgp_fn, sd, n, p, s, beta0, cf_config)
            for sd in batch)
        results.extend(out)
        _heartbeat(len(results), n_reps, t0, desc)
    print()
    return pd.DataFrame(results)
