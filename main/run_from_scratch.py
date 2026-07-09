"""
Reproduce the full simulation study from scratch.

Runs every method on every DGP at the headline sample size, the sample-size
trajectory on the three confounding designs, the polynomial-basis baselines,
the term-recovery log and the PySR reproducibility check, then writes the
metric tables and the trajectory figure.

This is the slow, complete path (several hours, dominated by the neural-network
nuisances and the PySR fits at large n). For day-to-day work after the first
run, use ``run_remaining.py``, which loads the saved results and only computes
what is missing.

    python run_from_scratch.py
"""

import pickle

import pandas as pd

import config as C
from dgp import DGP_REGISTRY
from estimators import SR_PDS_LOG
from simulation import run_all, seed_robustness_check
from evaluate import evaluate_all, evaluate_recovery, print_summary
from pipeline import build_registry, run_trajectory
from plots import plot_n_trajectory


def main():
    registry = build_registry(include_poly=True, improved_cf=True)
    dgp_keys = list(DGP_REGISTRY.keys())
    est_keys = list(registry.keys())

    # --- 1. headline fixed-n grid ---------------------------------------
    SR_PDS_LOG.clear()
    headline = run_all(
        DGP_REGISTRY, registry,
        dgp_keys=dgp_keys, estimator_keys=est_keys,
        n=C.N_HEADLINE, p=C.P, s=C.S, beta0=C.BETA0,
        n_reps=C.N_REPS_MAIN, n_jobs=C.N_JOBS,
        save_dir=str(C.RESULTS_DIR),
        reps_overrides={'sr_pds_cf': C.N_REPS_CF},
    )
    headline.to_pickle(C.RESULTS_DIR / 'headline.pkl')
    with open(C.RESULTS_DIR / 'sr_pds_log_headline.pkl', 'wb') as f:
        pickle.dump(list(SR_PDS_LOG), f)

    metrics = evaluate_all(headline, C.BETA0)
    metrics.to_csv(C.RESULTS_DIR / 'headline_metrics.csv', index=False)
    print_summary(metrics, C.BETA0)

    # --- 2. term recovery from the headline log -------------------------
    recovery = evaluate_recovery(list(SR_PDS_LOG), DGP_REGISTRY)
    recovery.to_csv(C.RESULTS_DIR / 'recovery.csv', index=False)

    # --- 3. sample-size trajectory (confounding designs) ----------------
    # the cross-fit variant is excluded from the trajectory (too expensive);
    # it is reported at the headline n only.
    traj_keys = [k for k in est_keys if k != 'sr_pds_cf']
    SR_PDS_LOG.clear()
    traj = run_trajectory(
        DGP_REGISTRY, registry, traj_keys, C.CONF_DGPS,
        n_grid=C.N_GRID_TRAJ, n_reps=C.N_REPS_TRAJ,
        beta0=C.BETA0, p=C.P, s=C.S, n_jobs=C.N_JOBS,
        save_path=C.RESULTS_DIR / 'trajectory.pkl',
    )
    traj.to_pickle(C.RESULTS_DIR / 'trajectory.pkl')

    # --- 4. reproducibility check ---------------------------------------
    seed_robustness_check(
        DGP_REGISTRY['dgp6']['fn'], dgp_key='dgp6',
        n=C.N_HEADLINE, p=C.P, s=C.S, beta0=C.BETA0,
        pysr_seeds=list(range(1, C.N_REPS_SEEDS + 1)),
        save_dir=str(C.RESULTS_DIR),
    )

    # --- 5. trajectory figure -------------------------------------------
    plot_n_trajectory(traj, beta0=C.BETA0, save=True)

    print(f"\nDone. Results in {C.RESULTS_DIR}/")


if __name__ == '__main__':
    main()
