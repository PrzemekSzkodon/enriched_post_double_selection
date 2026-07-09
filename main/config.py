"""
Central configuration for the simulation study.

Every run parameter lives here so the run scripts read rather than hard-code.
Sample sizes, replication counts, the trajectory grid and the output
directory are all set in one place.
"""

from pathlib import Path

# --- model / design ------------------------------------------------------
P        = 50      # number of candidate covariates
S        = 6       # sparsity of the linear part
BETA0    = 0.5     # true treatment effect

# --- replication counts --------------------------------------------------
N_HEADLINE   = 500    # sample size for the fixed-n tables
N_REPS_MAIN  = 500    # replications for most methods at the headline n
N_REPS_CF    = 100    # replications for the cross-fit variant (K-times slower)
N_REPS_TRAJ  = 25     # replications per point on the sample-size trajectory
N_REPS_SEEDS = 10     # PySR seeds for the reproducibility check

# --- sample-size trajectory ----------------------------------------------
N_GRID_TRAJ = [30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 2000, 5000]

# the three confounding designs are the ones plotted in the trajectory figure
CONF_DGPS = ['dgp6', 'dgp7', 'dgp8']

# --- parallelism ---------------------------------------------------------
N_JOBS = 8        # parallel workers; 10-core machine, headroom for OS + slow E-cores

# --- paths ---------------------------------------------------------------
RESULTS_DIR = Path('../results_full')
FIG_DIR     = Path('figures')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
