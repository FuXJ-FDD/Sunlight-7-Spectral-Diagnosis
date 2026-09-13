from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr
import geopandas as gpd
import matplotlib.patches as mpatches
import warnings
import seaborn as sns
from scipy import stats
from .core import PHI_COLUMNS

ARCH_ORDER=["Balanced Achievers","Economy–Innovation Driven","Planetary–Social Intermediate","Multidimensional Laggards"]


def _save(fig, out: Path, stem: str):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'{stem}.{ext}',bbox_inches='tight',dpi=300 if ext=='png' else None)
    plt.close(fig)


def figure1(scores: pd.DataFrame,out:Path):
    fig = plt.figure(figsize=(11, 8.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('A', loc='left', fontweight='bold')
    x_rank = scores['SDSN Index Rank']
    y_rank = scores['Ω Rank']
    diff = scores['ΔRank (Ω - SDSN)']
    rho, p_spearman = stats.spearmanr(x_rank, y_rank)
    lower_threshold = diff.nsmallest(5).iloc[-1]
    upper_threshold = diff.nlargest(5).iloc[-1]
    is_extreme_up = diff <= lower_threshold
    is_extreme_down = diff >= upper_threshold
    lower_bound = lower_threshold + 1
    upper_bound = upper_threshold - 1
    pt_size = 18
    pt_alpha = 0.7
    pt_edge = 'none'
    ref_line = np.array([0, 175])
    ax.plot(ref_line, ref_line, '--', lw=1.2, color='gray', label='1:1 Reference', zorder=1)
    ax.fill_between(ref_line, ref_line + lower_bound, ref_line + upper_bound,
                    color='gray', alpha=0.15, label='Normal variation zone', zorder=2)
    ax.scatter(x_rank[~(is_extreme_up | is_extreme_down)], y_rank[~(is_extreme_up | is_extreme_down)],
               s=pt_size, alpha=pt_alpha, color='#1f77b4', edgecolor=pt_edge, zorder=3, label='Normal shift')
    ax.scatter(x_rank[is_extreme_up], y_rank[is_extreme_up],
               s=pt_size + 20, alpha=pt_alpha, facecolors='none', edgecolors='#009E73',
               linewidths=1.5, marker='^', zorder=4, label='Largest upward shifts')
    ax.scatter(x_rank[is_extreme_down], y_rank[is_extreme_down],
               s=pt_size + 20, alpha=pt_alpha, facecolors='none', edgecolors='#D55E00',
               linewidths=1.5, marker='v', zorder=4, label='Largest downward shifts')
    ax.set_xlim(0, 175)
    ax.set_ylim(0, 175)
    ticks = [25, 50, 75, 100, 125, 150, 175]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.text(-0.03, -0.03, '0', transform=ax.transAxes, ha='center', va='center', fontsize=9)
    ax.set(
        xlabel='SDSN 2026 SDG Index Rank',
        ylabel=r'Nonlinear $\Omega_6$ Rank'
    )
    ax.grid(alpha=0.2)
    p_text = 'p < 0.001' if p_spearman < 0.001 else f'p = {p_spearman:.3f}'
    ax.text(
        0.97, 0.03,
        f'Spearman $\\rho$ = {rho:.3f}\n{p_text}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10
    )
    ax.legend(frameon=False, fontsize=8, loc='upper left')

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('B', loc='left', fontweight='bold')
    custom_bins = np.arange(-40, 36, 4)
    ax.hist(diff, bins=custom_bins, edgecolor='black', color='#1f77b4', alpha=0.8)
    ax.set_xticks(custom_bins)
    spaced_labels = [str(val) if i % 2 == 0 else "" for i, val in enumerate(custom_bins)]
    ax.set_xticklabels(spaced_labels, rotation=0, fontsize=9)
    ax.set(xlabel=r'Δ Rank ($\Omega_6$ Rank − SDSN Rank)', ylabel='Number of countries')
    ax.grid(axis='y', alpha=0.2)
    median_val = diff.median()
    ax.axvline(median_val, color='#333333', linestyle='--', linewidth=1.8, label=f'Median = {median_val:.1f}')
    ax.legend(frameon=False, fontsize=9, loc='upper right')

    ax = fig.add_subplot(gs[1, :])
    ax.set_title('C', loc='left', fontweight='bold')
    up = scores[scores['ΔRank (Ω - SDSN)'] <= lower_threshold].sort_values('ΔRank (Ω - SDSN)', ascending=True)
    down = scores[scores['ΔRank (Ω - SDSN)'] >= upper_threshold].sort_values('ΔRank (Ω - SDSN)', ascending=False)
    rows = pd.concat([up, down])
    n_up = len(up)
    n_down = len(down)
    y_positions = list(range(n_up + n_down, n_down, -1)) + list(range(n_down - 1, -1, -1))
    for yi, (_, r) in zip(y_positions, rows.iterrows()):
        ax.plot([r['SDSN Index Rank'], r['Ω Rank']], [yi, yi], lw=2, color='gray', alpha=0.6, zorder=1)
        ax.scatter(r['SDSN Index Rank'], yi, facecolors='white', edgecolors='black', s=45, zorder=3)
        ax.scatter(r['Ω Rank'], yi, facecolors='#1f77b4', edgecolors='none', s=45, zorder=3)
        ax.text(max(r['SDSN Index Rank'], r['Ω Rank']) + 2, yi, f"Δ{int(r['ΔRank (Ω - SDSN)']):+d}", va='center',
                fontsize=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(rows['Country'])
    ax.set_xlim(0, 140)
    ax.set_xticks(np.arange(0, 141, 20))
    ax.text(0, n_up + n_down + 0.8, ' Upward reranking', ha='left', va='center', fontweight='bold', fontsize=10)
    ax.text(0, n_down - 0.2, ' Downward reranking', ha='left', va='center', fontweight='bold', fontsize=10)
    ax.set_ylim(-1, n_up + n_down + 1.5)
    ax.scatter([], [], facecolors='white', edgecolors='black', s=45, label='SDSN Index Rank')
    ax.scatter([], [], facecolors='#1f77b4', edgecolors='none', s=45, label=r'$\Omega_6$ Rank')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.set_xlabel('Rank position')
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out, 'Figure1_rank_comparison')


def _world(scores, shp):
    world=gpd.read_file(shp).rename(columns={'SOC':'ISO3'})
    return world.merge(scores,on='ISO3',how='left')


def figure2(assignments:pd.DataFrame,out:Path,shp:Path):
    world = _world(assignments[['ISO3', 'Archetype']], shp)
    color_map = {
        "Balanced Achievers": "#2c7fb8",
        "Planetary–Social Intermediate": "#59a14f",
        "Economy–Innovation Driven": "#f28e2b",
        "Multidimensional Laggards": "#9e9e9e"
    }
    world['map_color'] = world['Archetype'].map(color_map).fillna('#eaeaea')
    fig, ax = plt.subplots(figsize=(12, 5.5))
    world.plot(color=world['map_color'], ax=ax, edgecolor='white', linewidth=0.1)
    ax.set_axis_off()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        small_countries = world[(world.geometry.area < 2.0) & (world['Archetype'].notna())].copy()
        if not small_countries.empty:
            centroids = small_countries.geometry.centroid
            centroids.plot(ax=ax, color=small_countries['map_color'],
                           markersize=18, edgecolor='.3', linewidth=0.4, zorder=3)
        if not small_countries.empty:
            centroids = small_countries.geometry.centroid
            centroids.plot(ax=ax, color=small_countries['map_color'],
                           markersize=18, edgecolor='.3', linewidth=0.4, zorder=3)
    minx, miny, maxx, maxy = world.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    legend_labels = [
        ("Balanced Achievers", "#2c7fb8"),
        ("Planetary–Social Intermediate", "#59a14f"),
        ("Economy–Innovation Driven", "#f28e2b"),
        ("Multidimensional Laggards", "#9e9e9e"),
        ("Not in sample / no data", "#eaeaea")
    ]
    handles = [mpatches.Patch(color=color, label=label) for label, color in legend_labels]
    ax.legend(handles=handles,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.05),
              ncol=3,
              frameon=False,
              fontsize=9)
    fig.tight_layout(pad=0)
    _save(fig, out, 'Figure2_global_archetype_distribution')


def figure3(assignments:pd.DataFrame,out:Path):
    fig, ax = plt.subplots(figsize=(10, 7.5));
    marks = ['o', 's', '^', 'D']
    n_dims = len(PHI_COLUMNS)
    x_coords = range(1, n_dims + 1)
    x_labels = [c.split()[0] for c in PHI_COLUMNS]
    for i, n in enumerate(ARCH_ORDER):
        s = assignments[assignments.Archetype == n]
        ax.plot(x_coords, s[PHI_COLUMNS].mean(), marker=marks[i], lw=2, label=f'{n} (n={len(s)})')
    ax.set_xticks(x_coords, x_labels)
    ax.set_ylim(0, 1);
    ax.set_ylabel('Mean dimension score');
    ax.grid(axis='y', alpha=.2);
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout();
    _save(fig, out, 'Figure3_mean_spectral_profiles')


def correlation_tables(scores:pd.DataFrame):
    n = len(PHI_COLUMNS)
    R = np.eye(n)
    P = np.zeros((n, n))
    for i, a in enumerate(PHI_COLUMNS):
        for j, b in enumerate(PHI_COLUMNS):
            if i != j: R[i, j], P[i, j] = pearsonr(scores[a], scores[b])
    idx = [c.split()[0] for c in PHI_COLUMNS]
    return pd.DataFrame(R, index=idx, columns=idx), pd.DataFrame(P, index=idx, columns=idx)


def figure4(scores:pd.DataFrame,out:Path):
    R, P = correlation_tables(scores)
    A = R.to_numpy()
    pv = P.to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(A, cmap='RdBu', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    n_dims = len(R)
    ax.set_xticks(range(n_dims), R.columns)
    ax.set_yticks(range(n_dims), R.index)
    for i in range(n_dims):
        for j in range(n_dims):
            t = f'{A[i, j]:.2f}';
            if i != j: t += '***' if pv[i, j] < .001 else '**' if pv[i, j] < .01 else '*' if pv[i, j] < .05 else ''
            ax.text(j, i, t, ha='center', va='center', fontsize=11, color='white' if abs(A[i, j]) > .55 else 'black')
    fig.colorbar(im, ax=ax, label='Pearson correlation (r)')
    fig.tight_layout();
    _save(fig, out, 'Figure4_correlation_matrix')


def figure5(assignments:pd.DataFrame,out:Path):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    markers = ['o', 's', '^', 'D']
    for m, n in zip(markers, ARCH_ORDER):
        s = assignments[assignments.Archetype == n]
        pts = ax.scatter(s[PHI_COLUMNS[1]], s[PHI_COLUMNS[0]], s=35, alpha=.75, marker=m, edgecolors='none',
                         label=f'{n} (n={len(s)})')
        base_color = pts.get_facecolor()[0]
        ax.scatter(s[PHI_COLUMNS[1]].mean(), s[PHI_COLUMNS[0]].mean(), s=150, marker=m, facecolor=base_color,
                   edgecolor='black', zorder=4)
    mean_phi1 = assignments[PHI_COLUMNS[0]].mean()
    mean_phi2 = assignments[PHI_COLUMNS[1]].mean()
    ax.axhline(mean_phi1, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=0, label='Overall Mean')
    ax.axvline(mean_phi2, color='gray', linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)
    ax.scatter([], [], s=150, marker='o', facecolor='gray', edgecolor='black', label='Group Mean (Centroid)')
    r, p = pearsonr(assignments[PHI_COLUMNS[1]], assignments[PHI_COLUMNS[0]])
    ax.text(.02, .98, f'Pearson r = {r:.2f}\np = {p:.3f}', transform=ax.transAxes, va='top', fontsize=11)
    ax.set(xlabel='Sustainable Prosperity (φ2)', ylabel='Planetary Health (φ1)')
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=11, markerscale=1.2, loc='best')
    fig.tight_layout()
    _save(fig, out, 'Figure5_phi2_phi1_scatter')


def figures6_7(assignments:pd.DataFrame,groups:pd.DataFrame,out:Path):
    d = assignments.merge(groups, on=['ISO3', 'Country'], how='left')
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    n_dims = len(PHI_COLUMNS)
    x_coords = range(1, n_dims + 1)
    x_labels = [c.split()[0] for c in PHI_COLUMNS]
    TITLE_SIZE = 14
    LABEL_SIZE = 13
    TICK_SIZE = 12
    LEGEND_SIZE = 11
    y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    income_order = ['High income', 'Upper-middle income', 'Lower-middle income', 'Low income']
    income_labels = {'High income': 'High-income','Upper-middle income': 'Upper-middle-income','Lower-middle income': 'Lower-middle-income','Low income': 'Low-income'}
    for g in income_order:
        s = d[d['Income group'] == g]
        axes[0].plot(x_coords, s[PHI_COLUMNS].mean(), marker='o', lw=2, label=f'{income_labels[g]} (n={len(s)})')
    axes[0].set_title('A', loc='left', fontweight='bold', fontsize=TITLE_SIZE)
    axes[0].set_xticks(x_coords)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks(y_ticks)
    axes[0].set_ylabel('Mean dimension score', fontsize=LABEL_SIZE)
    axes[0].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[0].grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False, fontsize=LEGEND_SIZE, loc='lower center', ncol=4)

    regions = ['OECD', 'E. Europe & C. Asia', 'East & South Asia', 'LAC', 'MENA', 'Sub-Saharan Africa', 'Oceania']
    for g in regions:
        s = d[d['SDR Region'] == g]
        axes[1].plot(x_coords, s[PHI_COLUMNS].mean(), marker='o', lw=1.6, label=f'{g} (n={len(s)})')
    axes[1].set_title('B', loc='left', fontweight='bold', fontsize=TITLE_SIZE)
    axes[1].set_xticks(x_coords)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks(y_ticks)
    axes[1].set_ylabel('Mean dimension score', fontsize=LABEL_SIZE)
    axes[1].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[1].grid(axis='y', alpha=.2)
    axes[1].legend(frameon=False, fontsize=LEGEND_SIZE, loc='lower center', ncol=4)

    for bloc, marker in [('G7/EU', 'o'), ('BRICS', 's'), ('ASEAN', '^')]:
        mask = d['Bloc membership'].fillna('').str.split(';').apply(lambda x: bloc in x)
        s = d[mask]
        axes[2].plot(x_coords, s[PHI_COLUMNS].mean(), marker=marker, lw=2.4, label=f'{bloc} (n={len(s)})')
    axes[2].set_title('C', loc='left', fontweight='bold', fontsize=TITLE_SIZE)
    axes[2].set_xticks(x_coords, x_labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks(y_ticks)
    axes[2].set_ylabel('Mean dimension score', fontsize=LABEL_SIZE)
    axes[2].tick_params(axis='both', labelsize=TICK_SIZE)
    axes[2].grid(axis='y', alpha=.2)
    axes[2].legend(frameon=False, fontsize=LEGEND_SIZE, loc='lower center', ncol=3)
    fig.tight_layout()
    _save(fig, out, 'Figure6_combined_profiles')


def supplementary_figures(scores,assignments,diagnostics,sensitivity,source,out,shp,theta4=.65):
    world = _world(scores[['ISO3'] + PHI_COLUMNS], shp)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    axes = axes.ravel()

    for i, c in enumerate(PHI_COLUMNS):
        world.plot(column=c, ax=axes[i], cmap='viridis', vmin=0, vmax=1,
                   missing_kwds={'color': '#eaeaea'}, edgecolor='.7', linewidth=.05)
        axes[i].set_title(f'{chr(65 + i)}  {c}', loc='left', fontweight='bold')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small_valid = world[(world.geometry.area < 2.0) & (world[c].notna())].copy()
            if not small_valid.empty:
                centroids = small_valid.geometry.centroid
                axes[i].scatter(centroids.x, centroids.y, c=small_valid[c],
                                cmap='viridis', vmin=0, vmax=1,
                                edgecolors='.3', linewidths=0.3, s=6, zorder=3)
        axes[i].set_axis_off()
        for j in range(len(PHI_COLUMNS), len(axes)):
            axes[j].set_axis_off()
    sm = ScalarMappable(norm=Normalize(0, 1), cmap='viridis')
    cax = fig.add_axes([0.35, 0.02, 0.30, 0.02])
    fig.colorbar(sm, cax=cax, orientation='horizontal', label='Dimension score')
    fig.tight_layout(rect=[0, .08, 1, .96], h_pad=3.0)
    _save(fig, out, 'FigureS1_global_dimension_maps_2col')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].plot(diagnostics.K, diagnostics.BIC, marker='o')
    axes[0].set_title('A', loc='left', fontweight='bold')
    axes[0].axvline(4, ls='--', color='gray')
    if 4 in diagnostics['K'].values:
        k4_bic = diagnostics.loc[diagnostics['K'] == 4, 'BIC'].values[0]
        axes[0].text(4.2, k4_bic, 'selected K = 4', va='center', ha='left', fontsize=10)
    axes[0].set(xlabel='Number of components (K)', ylabel='BIC')
    data = [assignments.loc[assignments.Archetype == n, 'Classification uncertainty'] for n in ARCH_ORDER]
    custom_labels = ["Balanced\nAchievers", "Economy-\nInnovation", "Planetary-\nSocial",
                     "Multidimensional\nLaggards"]
    axes[1].boxplot(data, labels=custom_labels, showfliers=True)
    axes[1].set_title('B', loc='left', fontweight='bold')
    mean_unc = assignments['Classification uncertainty'].mean()
    axes[1].axhline(mean_unc, ls='--', color='gray', label=f'Mean = {mean_unc:.3f}')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].set(ylabel='Classification uncertainty')
    axes[1].legend(frameon=False, fontsize=10, loc='upper right')
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].set(ylabel='Classification uncertainty')
    fig.tight_layout()
    _save(fig, out, 'FigureS2_model_diagnostics_and_uncertainty')

    x4 = scores['Institutional input x4'].dropna()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    custom_bins = np.arange(0, 1.05, 0.05)
    ax.hist(x4, bins=custom_bins, edgecolor='black', facecolor='#1f77b4', alpha=0.7)
    ticks_x = np.arange(0, 1.2, 0.2)
    ax.set_xticks(ticks_x)
    ax.set_xlim(0, 1.0)
    n_total = len(x4)
    median_val = x4.median()
    count_below = (x4 < theta4).sum()
    count_above = (x4 >= theta4).sum()
    ax.axvline(theta4, ls='--', lw=2, color='#1f77b4', label=f'Activation threshold $\\theta_4$ = {theta4:.2f}')
    ax.axvline(median_val, ls=':', lw=2.5, color='#1f77b4', label=f'Median = {median_val:.2f}')
    stats_text = f"n = {n_total}\nBelow $\\theta_4$: {count_below}\nAt/above $\\theta_4$: {count_above}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='square,pad=0.6', facecolor='white', alpha=0.9, edgecolor='.8'))
    ax.set(xlabel='Institutional input x4 = (G16+G17)/2', ylabel='Number of countries')
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    fig.tight_layout()
    _save(fig, out, 'FigureS3_institutional_threshold')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].bar(sensitivity['Scenario short'], sensitivity['Rank correlation with baseline'])
    axes[0].set_ylim(.98, 1.002)
    axes[0].set_title('A', loc='left', fontweight='bold')
    axes[0].set_ylabel('Spearman rank correlation')

    axes[1].bar(sensitivity['Scenario short'], sensitivity['Share of unchanged archetype assignments (%)'])
    axes[1].set_ylim(75, 101)
    axes[1].set_title('B', loc='left', fontweight='bold')
    axes[1].set_ylabel('Unchanged assignments (%)')

    fig.tight_layout()
    _save(fig, out, 'FigureS4_sensitivity_analysis')
