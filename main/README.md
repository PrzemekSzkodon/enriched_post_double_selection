# Symbolic Regression for Dictionary Discovery in Causal Inference

Simulation code for SR-PDS, an extension of Post-Double Selection (Belloni,
Chernozhukov and Hansen, 2014) that uses symbolic regression to discover
nonlinear basis terms before the LASSO selection step.

## Idea

PDS-LASSO is linear only in the columns it is given. Handed a basis that
already contains the relevant nonlinear terms it can represent nonlinear
nuisance functions; its real limitation is that it *selects* from a fixed
candidate set rather than *constructing* the basis. SR-PDS automates the basis
construction: PySR discovers candidate nonlinear transformations of the
covariates, appends them to the feature matrix, and standard PDS-LASSO then
runs on the enriched dictionary, returning a named set of controls that now
includes discovered nonlinear terms.

## Layout

```
dgp.py            eight data-generating processes + ground-truth terms
estimators.py     Full OLS, PDS-LASSO, DML (LASSO/RF/NN), SR-PDS, SR-PDS (CF)
srpds_cf.py       improved cross-fit SR-PDS (honest cross-fit variance)
srpds_poly.py     explicit polynomial-basis PDS baselines (degree 2 and 3)
simulation.py     Monte Carlo driver (run_simulation, run_all)
evaluate.py       bias / RMSE / coverage and term-recovery statistics
plots.py          figures (trajectory, coverage, recovery heatmaps)
config.py         all run parameters and paths in one place
pipeline.py       shared helpers (registry assembly, trajectory runner)
run_from_scratch.py   reproduce the whole study end to end
run_remaining.py      load saved results and run only the outstanding tasks
```

## Methods

| key          | method                                             |
|--------------|----------------------------------------------------|
| `full_ols`   | OLS on all covariates (falls back to PDS when p>=n) |
| `pds_lasso`  | Post-double selection, BCH lambda, HC1             |
| `dml_lasso`  | DML, LASSO nuisances, 5-fold cross-fitting         |
| `dml_rf`     | DML, random-forest nuisances                       |
| `dml_nn`     | DML, (64,32) MLP nuisances                         |
| `sr_pds`     | SR-PDS (same-data)                                 |
| `sr_pds_cf`  | SR-PDS, cross-fitted (honest cross-fit variance)   |
| `pds_poly2`  | PDS-LASSO on x + squares + pairwise interactions   |
| `pds_poly3`  | PDS-LASSO on the full degree-3 basis               |

## Cross-fit variance

`srpds_cf.py` estimates the treatment effect by the cross-fitted
orthogonal-moment estimator and forms its standard error from the influence
function (Chernozhukov et al., 2018), rather than running a single HC1 OLS on
the stacked out-of-fold residuals. This accounts for the fold structure and is
what restores honest coverage under sample-splitting. The search budget and
fold count are exposed via `CF_CONFIG` so per-fold recovery of high-complexity
terms (e.g. the cubic in the severe design) can be raised.

## Running

```bash
pip install -r requirements.txt          # PySR also needs a working Julia install
python run_from_scratch.py               # full reproduction (several hours)
python run_remaining.py                  # finish from saved results
```

Both scripts read every parameter from `config.py` and write to the directory
set there. Long runs save after each cell and skip completed work on restart,
so an interrupted run can simply be started again.
