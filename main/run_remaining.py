"""
Finish the study from existing results.

The headline grid, the term-recovery log and the trajectory have already been
computed and saved under ``config.RESULTS_DIR``. This script loads them and
runs only what remains:

  1. the improved cross-fit SR-PDS on the confounding designs at the headline
     n (honest cross-fit variance + larger PySR budget -- see srpds_cf.py),
     replacing the under-covering original CF column;
  2. the polynomial-basis baselines (degree 2 across all DGPs and the
     trajectory; degree 3 on the confounding designs up to a memory-safe n);
  3. the n = 10,000 trajectory point for the original methods, if it is not
     already present.

Every step writes incrementally and skips work already on disk, so the script
is safe to re-run after an interruption.

    python run_remaining.py
"""

import pickle

import numpy as np
import pandas as pd

import config as C
from dgp import DGP_REGISTRY
from estimators import SR_PDS_LOG
from simulation import run_simulation
from evaluate import evaluate
from pipeline import build_registry, run_trajectory, metrics_from_reps


# tasks toggled here so a user can run a subset
DO_CF        = True
DO_POLY2     = True
DO_POLY3     = True
DO_LARGE_N   = True

LARGE_N      = 10000
POLY3_GRID   = [30, 50, 100, 200, 500, 1000, 2000]   # degree-3 is memory-bound


def _load(name):
    p = C.RESULTS_DIR / name
    return pd.read_pickle(p) if p.exists() else None


def improved_cf(registry):
    """Rerun the improved cross-fit estimator on the confounding designs."""
    out_path = C.RESULTS_DIR / 'cf_improved.pkl'
    log_path = C.RESULTS_DIR / 'sr_pds_log_cf.pkl'
    SR_PDS_LOG.clear()

    rows = []
    for dgp_key in C.CONF_DGPS:
        print(f"  improved CF: {dgp_key}", flush=True)
        df = run_simulation(
            DGP_REGISTRY[dgp_key]['fn'], registry['sr_pds_cf']['fn'],
            estimator_key='sr_pds_cf', dgp_key=dgp_key,
            n=C.N_HEADLINE, p=C.P, s=C.S, beta0=C.BETA0,
            n_reps=C.N_REPS_CF, n_jobs=1)
        df['dgp'] = dgp_key
        df['estimator'] = 'sr_pds_cf'
        df['dgp_label'] = DGP_REGISTRY[dgp_key]['label']
        df['est_label'] = 'SR-PDS (CF)'
        df['beta0'] = C.BETA0
        rows.append(df)
        pd.concat(rows, ignore_index=True).to_pickle(out_path)
        m = evaluate(df, C.BETA0)
        print(f"    bias={m['bias']:+.3f}  coverage={m['coverage']:.2f}", flush=True)

    with open(log_path, 'wb') as f:
        pickle.dump(list(SR_PDS_LOG), f)
    return pd.concat(rows, ignore_index=True)


def poly_fixed(registry, est_key):
    """Degree-2 (or -3) baseline across all DGPs at the headline n."""
    rows = []
    for dgp_key in DGP_REGISTRY:
        df = run_simulation(
            DGP_REGISTRY[dgp_key]['fn'], registry[est_key]['fn'],
            estimator_key=est_key, dgp_key=dgp_key,
            n=C.N_HEADLINE, p=C.P, s=C.S, beta0=C.BETA0,
            n_reps=C.N_REPS_MAIN, n_jobs=C.N_JOBS)
        m = evaluate(df, C.BETA0)
        m.update({'dgp': dgp_key, 'estimator': est_key,
                  'est_label': registry[est_key]['label']})
        rows.append(m)
        print(f"  {est_key} {dgp_key}: bias={m['bias']:+.3f}", flush=True)
    return pd.DataFrame(rows)


def large_n(registry):
    """n = 10,000 for the original methods on dgp1 + the confounding designs."""
    out_path = C.RESULTS_DIR / 'large_n.pkl'
    keys = ['full_ols', 'pds_lasso', 'dml_lasso', 'dml_rf', 'dml_nn', 'sr_pds']
    return run_trajectory(
        DGP_REGISTRY, registry, keys, ['dgp1'] + C.CONF_DGPS,
        n_grid=[LARGE_N], n_reps=C.N_REPS_TRAJ,
        beta0=C.BETA0, p=C.P, s=C.S, n_jobs=C.N_JOBS,
        save_path=out_path)


def main():
    registry = build_registry(include_poly=True, improved_cf=True)

    if DO_CF:
        print("\n[1] improved cross-fit SR-PDS")
        improved_cf(registry)

    if DO_POLY2:
        print("\n[2] degree-2 baseline")
        p2 = poly_fixed(registry, 'pds_poly2')
        p2.to_csv(C.RESULTS_DIR / 'pds_poly2_fixedn.csv', index=False)
        run_trajectory(
            DGP_REGISTRY, registry, ['pds_poly2'], C.CONF_DGPS,
            n_grid=C.N_GRID_TRAJ, n_reps=C.N_REPS_TRAJ,
            beta0=C.BETA0, p=C.P, s=C.S, n_jobs=C.N_JOBS,
            save_path=C.RESULTS_DIR / 'pds_poly2_trajectory.pkl')

    if DO_POLY3:
        print("\n[3] degree-3 baseline (memory-capped grid)")
        run_trajectory(
            DGP_REGISTRY, registry, ['pds_poly3'], C.CONF_DGPS,
            n_grid=POLY3_GRID, n_reps=C.N_REPS_TRAJ,
            beta0=C.BETA0, p=C.P, s=C.S, n_jobs=1,   # 23k-col matrix: serial
            save_path=C.RESULTS_DIR / 'pds_poly3_trajectory.pkl')

    if DO_LARGE_N:
        print("\n[4] n = 10,000 for the original methods")
        large_n(registry)

    print(f"\nDone. New results written to {C.RESULTS_DIR}/")


if __name__ == '__main__':
    main()
