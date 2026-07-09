"""
Shared pipeline helpers used by both run scripts.

Keeping these in one place means ``run_from_scratch.py`` and ``run_remaining.py``
build the estimator set, run the trajectory and assemble the metric tables in
exactly the same way, so their outputs are directly comparable.
"""

from __future__ import annotations

import time
import pickle

import numpy as np
import pandas as pd

from estimators import ESTIMATOR_REGISTRY
from srpds_cf import sr_pds_cf as sr_pds_cf_improved
from srpds_poly import POLY_REGISTRY
from simulation import run_simulation
from evaluate import evaluate


def build_registry(include_poly=True, improved_cf=True):
    """Assemble the full estimator registry used in the paper.

    The improved cross-fit estimator (honest cross-fit variance, larger PySR
    budget) replaces the original ``sr_pds_cf`` entry when ``improved_cf`` is
    set; the polynomial-basis baselines are appended when ``include_poly``.
    """
    reg = dict(ESTIMATOR_REGISTRY)
    if improved_cf:
        reg['sr_pds_cf'] = {'fn': sr_pds_cf_improved, 'label': 'SR-PDS (CF)',
                            'requires_serial': True}
    if include_poly:
        reg.update(POLY_REGISTRY)
    return reg


def run_trajectory(dgp_registry, estimator_registry, est_keys, dgp_keys,
                   n_grid, n_reps, beta0, p, s, n_jobs, save_path=None):
    """Run a sample-size trajectory and return replication-level rows.

    Saves after every (n, dgp, estimator) cell when ``save_path`` is given, so
    an interrupted run can be resumed rather than lost.
    """
    rows = []
    if save_path is not None and save_path.exists():
        rows.append(pd.read_pickle(save_path))
        done = set(zip(rows[0]['n_grid'], rows[0]['dgp'], rows[0]['estimator']))
        print(f"resuming trajectory: {len(done)} cells already done")
    else:
        done = set()

    t0 = time.time()
    for n in n_grid:
        for dgp_key in dgp_keys:
            for est_key in est_keys:
                if (n, dgp_key, est_key) in done:
                    continue
                est = estimator_registry[est_key]
                print(f"[{(time.time()-t0)/60:6.1f}m]  n={n:>5}  "
                      f"{dgp_key}  {est_key}", flush=True)
                df = run_simulation(
                    dgp_registry[dgp_key]['fn'], est['fn'],
                    estimator_key=est_key, dgp_key=dgp_key,
                    n=n, p=p, s=s, beta0=beta0, n_reps=n_reps,
                    n_jobs=1 if est['requires_serial'] else n_jobs)
                df['dgp'] = dgp_key
                df['estimator'] = est_key
                df['dgp_label'] = dgp_registry[dgp_key]['label']
                df['est_label'] = est['label']
                df['beta0'] = beta0
                df['n_grid'] = n
                rows.append(df)
                if save_path is not None:
                    pd.concat(rows, ignore_index=True).to_pickle(save_path)
    return pd.concat(rows, ignore_index=True)


def metrics_from_reps(rep_df, beta0):
    """Collapse replication-level rows to one metric row per (dgp, estimator)."""
    out = []
    for (dgp_key, est_key), grp in rep_df.groupby(['dgp', 'estimator']):
        m = evaluate(grp, beta0)
        m.update({'dgp': dgp_key, 'estimator': est_key,
                  'dgp_label': grp['dgp_label'].iloc[0],
                  'est_label': grp['est_label'].iloc[0]})
        out.append(m)
    return pd.DataFrame(out)
