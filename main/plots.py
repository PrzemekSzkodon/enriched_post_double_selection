"""
plots.py
========
Figures for the EPDS simulation study.

All functions save to figures/ directory and optionally display inline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

FIGDIR = '../figures'
os.makedirs(FIGDIR, exist_ok=True)

COLORS = {
    'Naive OLS'            : '#888780',
    'Full OLS (infeasible)': '#B4B2A9',
    'PDS-LASSO'            : '#534AB7',
    'DML-LASSO'            : '#1D9E75',
    'DML-NN'               : '#D85A30',
    'EPDS (ours)'          : '#BA7517',
}

STYLE = {
    'font.family'  : 'sans-serif',
    'font.size'    : 11,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'    : True,
    'grid.alpha'   : 0.3,
    'grid.linestyle': '--',
}


def _color(label):
    return COLORS.get(label, '#378ADD')


def _save(fig, fname, dpi=150):
    path = f"{FIGDIR}/{fname}"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")


# ============================================================
# 1. beta_hat distribution -- histogram per DGP
# ============================================================
def plot_distributions(combined_df, beta0, dgp_key, save=True):
    """
    Histogram of beta_hat distributions for each estimator
    for a single DGP.
    """
    plt.rcParams.update(STYLE)

    subset    = combined_df[combined_df['dgp'] == dgp_key]
    estimators = subset['est_label'].unique()
    dgp_label  = subset['dgp_label'].iloc[0]

    n_est = len(estimators)
    fig, axes = plt.subplots(1, n_est, figsize=(3.5 * n_est, 3.5),
                              sharey=True)
    if n_est == 1:
        axes = [axes]

    for ax, est in zip(axes, estimators):
        vals = subset[subset['est_label'] == est]['beta_hat'].dropna()
        ax.hist(vals, bins=30, color=_color(est), alpha=0.75,
                edgecolor='white', linewidth=0.5)
        ax.axvline(beta0, color='black', linewidth=1.5,
                   linestyle='--', label=f'True β₀={beta0}')
        ax.set_title(est, fontsize=10, fontweight='500')
        ax.set_xlabel('β̂₀')

    axes[0].set_ylabel('Count')
    fig.suptitle(f'Distribution of β̂₀ — {dgp_label}',
                 fontsize=12, fontweight='500', y=1.02)
    fig.tight_layout()

    if save:
        _save(fig, f'dist_{dgp_key}.png')
    return fig


# ============================================================
# 2. coverage bar chart -- all DGPs and estimators
# ============================================================
def plot_coverage(metrics_df, save=True):
    """
    Bar chart of 95% CI coverage for all (DGP, estimator) combinations.
    Dashed line at 0.95 nominal level.
    """
    plt.rcParams.update(STYLE)

    dgps      = metrics_df['dgp_label'].unique()
    estimators = metrics_df['est_label'].unique()
    x         = np.arange(len(dgps))
    width     = 0.8 / len(estimators)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, est in enumerate(estimators):
        vals = [
            metrics_df[(metrics_df['dgp_label'] == dgp) &
                       (metrics_df['est_label'] == est)]['coverage'].values
            for dgp in dgps
        ]
        vals = [v[0] if len(v) > 0 else np.nan for v in vals]
        offset = (i - len(estimators) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=est, color=_color(est), alpha=0.85)

    ax.axhline(0.95, color='black', linewidth=1.2,
               linestyle='--', label='Nominal 95%')
    ax.set_xticks(x)
    ax.set_xticklabels(dgps, fontsize=9)
    ax.set_ylabel('Coverage')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8, ncol=3, loc='lower right')
    ax.set_title('95% CI Coverage by DGP and Estimator',
                 fontweight='500')
    fig.tight_layout()

    if save:
        _save(fig, 'coverage.png')
    return fig


# ============================================================
# 3. RMSE heatmap -- DGPs × estimators
# ============================================================
def plot_rmse_heatmap(metrics_df, save=True):
    """Heatmap of RMSE -- lower is better."""
    from evaluate import summary_table
    plt.rcParams.update(STYLE)

    pivot = summary_table(metrics_df, metric='rmse')

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(pivot.values.astype(float),
                   cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30,
                       ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center',
                        va='center', fontsize=8,
                        color='white' if val > pivot.values.max() * 0.7
                        else 'black')

    plt.colorbar(im, ax=ax, label='RMSE')
    ax.set_title('RMSE by DGP and Estimator (lower = better)',
                 fontweight='500')
    fig.tight_layout()

    if save:
        _save(fig, 'rmse_heatmap.png')
    return fig


# ============================================================
# 4. bias plot -- all DGPs and estimators
# ============================================================
def plot_bias(metrics_df, save=True):
    """
    Dot plot of bias for all (DGP, estimator) combinations.
    Dashed line at 0 (unbiased).
    """
    plt.rcParams.update(STYLE)

    dgps       = metrics_df['dgp_label'].unique()
    estimators = metrics_df['est_label'].unique()
    x          = np.arange(len(dgps))
    width      = 0.8 / len(estimators)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, est in enumerate(estimators):
        vals = [
            metrics_df[(metrics_df['dgp_label'] == dgp) &
                       (metrics_df['est_label'] == est)]['bias'].values
            for dgp in dgps
        ]
        vals   = [v[0] if len(v) > 0 else np.nan for v in vals]
        offset = (i - len(estimators) / 2 + 0.5) * width
        ax.scatter(x + offset, vals, label=est,
                   color=_color(est), s=60, zorder=3)
        ax.vlines(x + offset, 0, vals,
                  color=_color(est), alpha=0.4, linewidth=1)

    ax.axhline(0, color='black', linewidth=1.2, linestyle='--',
               label='Zero bias')
    ax.set_xticks(x)
    ax.set_xticklabels(dgps, fontsize=9)
    ax.set_ylabel('Bias')
    ax.legend(fontsize=8, ncol=3)
    ax.set_title('Bias by DGP and Estimator',
                 fontweight='500')
    fig.tight_layout()

    if save:
        _save(fig, 'bias.png')
    return fig


# ============================================================
# 5. summary figure -- 2x2 panel
# ============================================================
def plot_summary_panel(metrics_df, save=True):
    """
    2x2 panel: bias, rmse, coverage, size.
    """
    plt.rcParams.update(STYLE)

    dgps       = metrics_df['dgp_label'].unique()
    estimators = metrics_df['est_label'].unique()
    x          = np.arange(len(dgps))
    width      = 0.8 / len(estimators)

    metrics_to_plot = [
        ('bias',     'Bias',       None,  0),
        ('rmse',     'RMSE',       None,  0),
        ('coverage', 'Coverage',   0.95,  None),
        ('size',     'Size',       0.05,  None),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, label, hline, vline) in zip(axes, metrics_to_plot):
        for i, est in enumerate(estimators):
            vals = [
                metrics_df[(metrics_df['dgp_label'] == dgp) &
                           (metrics_df['est_label'] == est)][metric].values
                for dgp in dgps
            ]
            vals   = [v[0] if len(v) > 0 else np.nan for v in vals]
            offset = (i - len(estimators) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9,
                   label=est, color=_color(est), alpha=0.85)

        if hline is not None:
            ax.axhline(hline, color='black', linewidth=1.2,
                       linestyle='--')
        if vline is not None:
            ax.axhline(vline, color='black', linewidth=1.2,
                       linestyle='--')

        ax.set_xticks(x)
        ax.set_xticklabels(dgps, fontsize=8, rotation=15, ha='right')
        ax.set_title(label, fontweight='500')

        if metric in ['coverage', 'size']:
            ax.yaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1))
            ax.set_ylim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(estimators), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Simulation Results — All DGPs and Estimators',
                 fontsize=13, fontweight='500')
    fig.tight_layout()

    if save:
        _save(fig, 'summary_panel.png')
    return fig


# ============================================================
# sanity check
# ============================================================
if __name__ == '__main__':
    import pandas as pd
    from dgp import dgp1, dgp2
    from estimators import pds_lasso, naive_ols
    from simulation import run_simulation
    from evaluate import evaluate_all

    beta0 = 0.5
    dfs   = []

    for dgp_fn, dgp_key, dgp_lbl in [
        (dgp1, 'dgp1', 'Linear sparse'),
        (dgp2, 'dgp2', 'Quadratic + linear'),
    ]:
        for est_fn, est_key, est_lbl in [
            (naive_ols,  'naive_ols',  'Naive OLS'),
            (pds_lasso,  'pds_lasso',  'PDS-LASSO'),
        ]:
            df = run_simulation(dgp_fn, est_fn, n_reps=50)
            df['dgp']       = dgp_key
            df['estimator'] = est_key
            df['dgp_label'] = dgp_lbl
            df['est_label'] = est_lbl
            df['beta0']     = beta0
            dfs.append(df)

    combined = pd.concat(dfs)
    metrics  = evaluate_all(combined, beta0)

    plot_coverage(metrics, save=True)
    plot_bias(metrics, save=True)
    plot_summary_panel(metrics, save=True)
    print("Plots saved to ../figures/")