"""
Figures for the simulation study.

By default everything is saved to ../figures/ at 150 dpi. The plot
functions take the standardised DataFrames produced by evaluate.py and
work off the est_label / dgp_label columns.

Estimator order and colours are kept consistent across plots via
EST_ORDER and COLORS below. Unknown labels get a fallback blue.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

FIGDIR = '../figures'
os.makedirs(FIGDIR, exist_ok=True)


# colour palette and display order. Add new methods here when extending.
COLORS = {
    'Full OLS':      '#B4B2A9',
    'PDS-LASSO':     '#534AB7',
    'DML-LASSO':     '#1D9E75',
    'DML-RF':        '#7A9E1D',
    'DML-NN':        '#D85A30',
    'SR-PDS':        '#BA7517',
    'SR-PDS (CF)':   '#185FA5',
}

EST_ORDER = [
    'Full OLS', 'PDS-LASSO',
    'DML-LASSO', 'DML-RF', 'DML-NN',
    'SR-PDS', 'SR-PDS (CF)',
]

STYLE = {
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# red-yellow-green colormap for recovery heatmaps (0 = miss, 1 = hit)
_RYG = LinearSegmentedColormap.from_list(
    'ryg', ['#E24B4A', '#FAC775', '#1D9E75'], N=256,
)


def _color(label):
    return COLORS.get(label, '#378ADD')


def _save(fig, fname, dpi=150):
    path = f"{FIGDIR}/{fname}"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {path}")


def _ordered_estimators(labels):
    present = set(labels)
    return [e for e in EST_ORDER if e in present]


def _term_label(term):
    """Pretty-print a (kind, i, j) term tuple."""
    kind, i, j = term
    if kind == 'sq':    return f'x{i}^2'
    if kind == 'int':   return f'x{i}*x{j}'
    if kind == 'cubic': return f'x{i}^2*x{j}'
    if kind == 'log':   return f'log|x{i}|'
    if kind == 'sqrt':  return f'sqrt|x{i}|'
    return str(term)


# --- distributions of beta_hat ------------------------------------------

def plot_distributions(combined_df, beta0, dgp_key, save=True):
    """Histograms of beta_hat for one DGP, one panel per estimator."""
    plt.rcParams.update(STYLE)
    sub = combined_df[combined_df['dgp'] == dgp_key]
    ests = _ordered_estimators(sub['est_label'].unique())
    dgp_lbl = sub['dgp_label'].iloc[0]

    fig, axes = plt.subplots(1, len(ests), figsize=(3.5 * len(ests), 3.5), sharey=True)
    if len(ests) == 1:
        axes = [axes]

    for ax, est in zip(axes, ests):
        vals = sub[sub['est_label'] == est]['beta_hat'].dropna()
        ax.hist(vals, bins=30, color=_color(est), alpha=0.75,
                edgecolor='white', linewidth=0.5)
        ax.axvline(beta0, color='black', linewidth=1.5, linestyle='--')
        ax.set_title(est, fontsize=9, fontweight='500')
        ax.set_xlabel(r'$\hat\beta$')

    axes[0].set_ylabel('Count')
    fig.suptitle(f'Distribution of beta_hat -- {dgp_lbl}',
                 fontsize=12, fontweight='500', y=1.02)
    fig.tight_layout()
    if save:
        _save(fig, f'dist_{dgp_key}.png')
    return fig


# back-compat alias used by older notebooks
plot_beta_hist = plot_distributions


# --- coverage -----------------------------------------------------------

def plot_coverage(metrics_df, save=True):
    plt.rcParams.update(STYLE)
    dgps  = list(metrics_df['dgp_label'].unique())
    ests  = _ordered_estimators(metrics_df['est_label'].unique())
    x     = np.arange(len(dgps))
    width = 0.8 / len(ests)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, est in enumerate(ests):
        vals = []
        for dgp in dgps:
            v = metrics_df[(metrics_df['dgp_label'] == dgp)
                           & (metrics_df['est_label'] == est)]['coverage'].values
            vals.append(v[0] if len(v) > 0 else np.nan)
        offset = (i - len(ests) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9,
               label=est, color=_color(est), alpha=0.85)

    ax.axhline(0.95, color='black', linewidth=1.2, linestyle='--', label='Nominal 95%')
    ax.set_xticks(x)
    ax.set_xticklabels(dgps, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Coverage')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=8, ncol=3, loc='lower right')
    ax.set_title('95% CI Coverage by DGP and Estimator', fontweight='500')
    fig.tight_layout()
    if save:
        _save(fig, 'coverage.png')
    return fig


# --- RMSE heatmap -------------------------------------------------------

def plot_rmse_heatmap(metrics_df, save=True):
    from evaluate import summary_table
    plt.rcParams.update(STYLE)

    pivot = summary_table(metrics_df, metric='rmse')

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), 3.5))
    im = ax.imshow(pivot.values.astype(float), cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # annotate cells; flip text colour against background for readability
    vmax = np.nanmax(pivot.values.astype(float))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not (isinstance(val, float) and np.isnan(val)):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=8,
                        color='white' if float(val) > 0.7 * vmax else 'black')

    plt.colorbar(im, ax=ax, label='RMSE')
    ax.set_title('RMSE -- lower is better', fontweight='500')
    fig.tight_layout()
    if save:
        _save(fig, 'rmse_heatmap.png')
    return fig


# --- bias dot plot ------------------------------------------------------

def plot_bias(metrics_df, save=True):
    plt.rcParams.update(STYLE)
    dgps  = list(metrics_df['dgp_label'].unique())
    ests  = _ordered_estimators(metrics_df['est_label'].unique())
    x     = np.arange(len(dgps))
    width = 0.8 / len(ests)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, est in enumerate(ests):
        vals = []
        for dgp in dgps:
            v = metrics_df[(metrics_df['dgp_label'] == dgp)
                           & (metrics_df['est_label'] == est)]['bias'].values
            vals.append(v[0] if len(v) > 0 else np.nan)
        offset = (i - len(ests) / 2 + 0.5) * width
        ax.scatter(x + offset, vals, label=est, color=_color(est), s=60, zorder=3)
        # stem-style drop lines to 0 to make the magnitude of bias easier to read
        ax.vlines(x + offset, 0, vals, color=_color(est), alpha=0.4, linewidth=1)

    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(dgps, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Bias')
    ax.legend(fontsize=8, ncol=3)
    ax.set_title('Bias by DGP and Estimator', fontweight='500')
    fig.tight_layout()
    if save:
        _save(fig, 'bias.png')
    return fig


# --- 2x2 summary panel --------------------------------------------------

def plot_summary_panel(metrics_df, save=True):
    plt.rcParams.update(STYLE)
    dgps  = list(metrics_df['dgp_label'].unique())
    ests  = _ordered_estimators(metrics_df['est_label'].unique())
    x     = np.arange(len(dgps))
    width = 0.8 / len(ests)

    # (metric, panel title, optional zero-line, optional reference hline)
    panels = [
        ('bias',     'Bias',     0,    None),
        ('rmse',     'RMSE',     None, None),
        ('coverage', 'Coverage', None, 0.95),
        ('size',     'Size',     None, 0.05),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, (metric, lbl, zero_line, hline) in zip(axes, panels):
        for i, est in enumerate(ests):
            vals = []
            for dgp in dgps:
                v = metrics_df[(metrics_df['dgp_label'] == dgp)
                               & (metrics_df['est_label'] == est)][metric].values
                vals.append(v[0] if len(v) > 0 else np.nan)
            offset = (i - len(ests) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9,
                   label=est, color=_color(est), alpha=0.85)

        if hline is not None:
            ax.axhline(hline, color='black', linewidth=1.2, linestyle='--')
        if zero_line is not None:
            ax.axhline(zero_line, color='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(dgps, fontsize=8, rotation=15, ha='right')
        ax.set_title(lbl, fontweight='500')
        if metric in ('coverage', 'size'):
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_ylim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=min(len(ests), 4), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Simulation Results -- All DGPs and Estimators',
                 fontsize=13, fontweight='500')
    fig.tight_layout()
    if save:
        _save(fig, 'summary_panel.png')
    return fig


# --- recovery heatmap ---------------------------------------------------

def plot_recovery_heatmap(recovery_df, equation='y', save=True):
    """
    Rows are (DGP, truth term); columns are (variant, stage). The cell
    value is the recovery rate. Colour: red 0, yellow 0.5, green 1.
    """
    plt.rcParams.update(STYLE)

    sub = recovery_df[recovery_df['equation'] == equation].copy()
    if sub.empty:
        print(f"  No recovery data for equation='{equation}'")
        return None

    # the 'col' label combines variant and stage in a multiline string so the
    # x-axis ticks read cleanly
    sub['col'] = sub['variant'] + '\n' + sub['stage'].str.replace('_lasso', '-LASSO')
    sub['row'] = sub['dgp_label'] + '\n' + sub['term_label']

    pivot = sub.pivot_table(index='row', columns='col',
                            values='recovery_rate', aggfunc='mean')

    # preserve DGP-then-term ordering on the rows
    row_order = []
    for dgp_lbl in sub['dgp_label'].unique():
        for t in sub[sub['dgp_label'] == dgp_lbl]['term_label'].unique():
            key = f"{dgp_lbl}\n{t}"
            if key in pivot.index:
                row_order.append(key)
    pivot = pivot.reindex([r for r in row_order if r in pivot.index])

    # order columns by (variant, stage) with pre before post -- not alphabetic
    def _col_key(col):
        variant, stage = col.split('\n')
        return (variant, 0 if stage == 'pre-LASSO' else 1)
    pivot = pivot[sorted(pivot.columns.tolist(), key=_col_key)]

    n_rows, n_cols = pivot.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.5), max(4, n_rows * 0.55)))
    im = ax.imshow(pivot.values.astype(float), cmap=_RYG,
                   vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(n_rows):
        for j in range(n_cols):
            val = pivot.values[i, j]
            if not np.isnan(val):
                txt_col = 'white' if val < 0.35 or val > 0.80 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=txt_col, fontweight='500')

    plt.colorbar(im, ax=ax, label='Recovery rate', shrink=0.6)
    eq_str = 'outcome (Y)' if equation == 'y' else 'treatment (D)'
    ax.set_title(f'Term recovery rates -- {eq_str} equation\n'
                 '(pre-LASSO = PySR found it; post-LASSO = survived selection)',
                 fontweight='500')
    fig.tight_layout()
    if save:
        _save(fig, f'recovery_heatmap_{equation}.png')
    return fig


# --- coefficient recovery -----------------------------------------------

def plot_coefficient_recovery(recovery_df, equation='y', save=True):
    """
    For each (DGP, truth term, variant) plot the recovered post-LASSO OLS
    coefficient (mean +/- std across reps), with the true value shown as
    a dashed horizontal mark.
    """
    plt.rcParams.update(STYLE)

    sub = recovery_df[
        (recovery_df['equation'] == equation)
        & (recovery_df['stage'] == 'post_lasso')
    ].dropna(subset=['mean_coef', 'true_coef']).copy()

    if sub.empty:
        print(f"  No coefficient recovery data for equation='{equation}'")
        return None

    # map internal variant keys to display labels (and hence to colours)
    variant_to_label = {
        'sr_pds': 'SR-PDS',
        'sr_pds_cf': 'SR-PDS (CF)',
        'epds': 'SR-PDS',           # back-compat with older runs
        'epds_du': 'SR-PDS (CF)',   # back-compat
    }

    dgps = sub['dgp_label'].unique()
    fig, axes = plt.subplots(len(dgps), 1,
                             figsize=(12, 2.8 * len(dgps)),
                             squeeze=False)

    for i, dgp_lbl in enumerate(dgps):
        ax = axes[i, 0]
        sub_dgp = sub[sub['dgp_label'] == dgp_lbl]
        terms = sub_dgp['term_label'].unique()
        variants = sorted(sub_dgp['variant'].unique())
        n_var = len(variants)
        x = np.arange(len(terms))
        width = 0.7 / n_var

        for j, var in enumerate(variants):
            sub_var = sub_dgp[sub_dgp['variant'] == var]
            means, stds, trues = [], [], []
            for t in terms:
                row = sub_var[sub_var['term_label'] == t]
                if len(row) > 0:
                    means.append(float(row['mean_coef'].iloc[0]))
                    std_val = row['std_coef'].iloc[0]
                    stds.append(float(std_val) if not np.isnan(std_val) else 0)
                    trues.append(float(row['true_coef'].iloc[0]))
                else:
                    means.append(np.nan)
                    stds.append(0)
                    trues.append(np.nan)

            offset = (j - n_var / 2 + 0.5) * width
            display_label = variant_to_label.get(var, var)
            ax.bar(x + offset, means, width * 0.9,
                   label=display_label, color=_color(display_label),
                   alpha=0.75,
                   yerr=stds, error_kw={'capsize': 3, 'linewidth': 0.8})

        # true coefficients overlaid as black dashed segments
        for k, true_val in enumerate(trues[:len(terms)]):
            if not np.isnan(true_val):
                ax.hlines(true_val, k - 0.4, k + 0.4,
                          color='black', linewidth=1.8, linestyle='--', zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(terms, fontsize=9)
        ax.set_ylabel('coefficient')
        ax.set_title(f'{dgp_lbl} -- post-LASSO coefficient vs truth (dashed)',
                     fontweight='500', fontsize=10)
        ax.axhline(0, color='gray', linewidth=0.4)
        if i == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    if save:
        _save(fig, f'coef_recovery_{equation}.png')
    return fig


# back-compat alias used by older notebooks
plot_coef_recovery = plot_coefficient_recovery


# --- n-trajectory -------------------------------------------------------

def plot_n_trajectory(traj_df, beta0, log_y=True, save=True):
    """
    Bias (mean +/- std band) and RMSE as a function of n, faceted by DGP.

    Assumes traj_df has the same columns the simulation harness produces
    (dgp, est_label, beta_hat, dgp_label) plus an 'n' column with the
    sample size for each row. Use either 'n' or 'n_grid' depending on
    how the notebook stored it.
    """
    plt.rcParams.update(STYLE)

    df = traj_df.copy()
    # the trajectory notebook stores the per-cell sample size as 'n_grid';
    # older runs called it 'n'. Standardise to 'n'.
    if 'n' not in df.columns and 'n_grid' in df.columns:
        df['n'] = df['n_grid']

    df['bias']   = df['beta_hat'] - beta0
    df['sq_err'] = df['bias'] ** 2

    agg = (df.groupby(['dgp', 'n', 'est_label'])
             .agg(bias_mean=('bias',  'mean'),
                  bias_std =('bias',  'std'),
                  rmse     =('sq_err', lambda s: np.sqrt(s.mean())))
             .reset_index())

    dgps = sorted(agg['dgp'].unique())
    ests = _ordered_estimators(agg['est_label'].unique())

    fig, axes = plt.subplots(len(dgps), 2,
                             figsize=(13, 2.8 * len(dgps)),
                             sharex=False, squeeze=False)

    for i, dgp in enumerate(dgps):
        sub = agg[agg['dgp'] == dgp]
        dgp_lbl = (traj_df[traj_df['dgp'] == dgp]['dgp_label'].iloc[0]
                   if 'dgp_label' in traj_df.columns else dgp)

        for est in ests:
            es = sub[sub['est_label'] == est].sort_values('n')
            if es.empty:
                continue
            c = _color(est)
            axes[i, 0].plot(es['n'], es['bias_mean'],
                            marker='o', markersize=4, linewidth=1.2,
                            label=est, color=c)
            axes[i, 0].fill_between(es['n'],
                                    es['bias_mean'] - es['bias_std'],
                                    es['bias_mean'] + es['bias_std'],
                                    color=c, alpha=0.12)
            axes[i, 1].plot(es['n'], es['rmse'],
                            marker='o', markersize=4, linewidth=1.2,
                            label=est, color=c)

        axes[i, 0].axhline(0, color='gray', linestyle='--', linewidth=0.5)
        axes[i, 0].set_xscale('log')
        axes[i, 1].set_xscale('log')
        if log_y:
            axes[i, 1].set_yscale('log')
        axes[i, 0].set_title(f'{dgp_lbl} -- bias (mean +/- std)', fontsize=10)
        axes[i, 1].set_title(f'{dgp_lbl} -- RMSE', fontsize=10)
        axes[i, 0].set_ylabel('bias')
        axes[i, 1].set_ylabel('RMSE')
        if i == 0:
            axes[i, 0].legend(loc='best', fontsize=7, framealpha=0.9)

    axes[-1, 0].set_xlabel('n (log scale)')
    axes[-1, 1].set_xlabel('n (log scale)')
    fig.tight_layout()
    if save:
        _save(fig, 'n_trajectory.png')
    return fig


# --- term recovery vs n -------------------------------------------------

def plot_recovery_vs_n(traj_df, dgp_registry, save=True):
    """
    Recovery rate (fraction of reps that found a given truth term) as a
    function of n, for SR-PDS variants. One panel per DGP that has a
    nontrivial y-equation truth.

    The trajectory CSVs store list-valued columns as strings; we eval()
    them back to tuples here.
    """
    plt.rcParams.update(STYLE)

    sr_df = traj_df[traj_df['estimator'].isin(['sr_pds', 'sr_pds_cf',
                                               'epds', 'epds_du'])].copy()
    if sr_df.empty:
        print("  No SR-PDS data in traj_df")
        return None

    if 'n' not in sr_df.columns and 'n_grid' in sr_df.columns:
        sr_df['n'] = sr_df['n_grid']

    dgps_with_truth = [k for k in sr_df['dgp'].unique()
                       if dgp_registry[k].get('truth_terms_y')]
    if not dgps_with_truth:
        return None

    fig, axes = plt.subplots(len(dgps_with_truth), 1,
                             figsize=(12, 2.8 * len(dgps_with_truth)),
                             squeeze=False)

    for i, dgp in enumerate(dgps_with_truth):
        ax = axes[i, 0]
        truth = dgp_registry[dgp]['truth_terms_y']
        sub = sr_df[sr_df['dgp'] == dgp].copy()
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(truth), 1)))

        for j, term in enumerate(truth):
            colour = cmap[j]
            lbl = _term_label(term)

            sub['_pre']  = sub['pre_lasso_terms_y'].apply(
                lambda v: int(term in (eval(v) if isinstance(v, str) else (v or []))))
            sub['_post'] = sub['post_lasso_terms_y'].apply(
                lambda v: int(term in (eval(v) if isinstance(v, str) else (v or []))))

            for col, marker, ls, alpha_v, suffix in [
                ('_pre',  'o', '--', 0.45, '(PySR)'),
                ('_post', 's', '-',  1.00, '(post-LASSO)'),
            ]:
                rate = (sub.groupby('n')[col].mean()
                           .reset_index().sort_values('n'))
                ax.plot(rate['n'], rate[col],
                        marker=marker, markersize=5, linestyle=ls,
                        alpha=alpha_v, color=colour,
                        label=f'{lbl} {suffix}')

        ax.set_title(f"{dgp_registry[dgp]['label']} -- term recovery rate vs n",
                     fontsize=10)
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('recovery rate')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        if i == len(dgps_with_truth) - 1:
            ax.set_xlabel('n (log scale)')

    fig.tight_layout()
    if save:
        _save(fig, 'recovery_vs_n.png')
    return fig


if __name__ == '__main__':
    print("plots.py loaded -- functions available:")
    for f in dir():
        if f.startswith('plot_'):
            print(f"  {f}")
