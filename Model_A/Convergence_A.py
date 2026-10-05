#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2026

Posterior convergence study for Model A.

Generates synthetic datasets of increasing size from the fixed parameters in
ReadyData_Synth_A.pkl, fits Model A to each, and tracks how the recovered
posteriors (mean, 95% HDI, sd) close in on the true values as the number of
trials grows.

Design:
    - Trials (stimulus level + group) are resampled with replacement from the
      experimental design, so any dataset size keeps the real mix of stimulus
      levels and groups, and sizes larger than the real dataset are possible.
    - Within a replicate the datasets are nested: the size-N dataset is the
      first N trials of one long synthetic run, so each larger dataset only
      adds trials to the smaller one. This isolates the effect of more data
      from the noise of drawing a brand new dataset at every size.
    - Each replicate uses a fresh synthetic run, to show fit-to-fit variability.
      Each replicate has its own random stream (spawned from SEED), so its
      data don't depend on how many replicates are run.
    - Coverage is a count over replicates x groups, so it needs many
      replicates: with 20, each size/parameter cell has 80 fits and a
      calibrated model lands within about +-0.05 of 0.95.

Runtime: roughly 7 min per 13k trials with the default sampler settings,
about 17 min per replicate (sum of SIZES ~32k trials) with chains run one
after another, so the defaults (20 replicates) take about 6 h with CORES = 1
and about a third of that with CORES = 4. Lower N_REPS / DRAWS for a quick look.

@author: vmschroe
"""

import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import pickle
import xarray as xr
from pathlib import Path
from scipy.stats import binom

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
DATA_DIR = MODEL_DIR / "Data_A"
PLOT_DIR = MODEL_DIR / "Plots_A" / "Convergence_Plots_A"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

#%% SETTINGS

SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000]   # total trials per dataset
N_REPS = 20                                         # independent synthetic runs
SHOW_REPS = 3                                       # replicates drawn individually in the per-replicate plots
DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4                                           # chains sampled in parallel; 1 = one after another
# Python 3.14's default 'forkserver' re-runs this script in every worker, so fork them instead
# ('fork' doesn't exist on Windows, which keeps its default)
MP_CTX = None if os.name == 'nt' else 'fork'
HDI_PROB = 0.95
SEED = 20260929

rep_rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(N_REPS)]

#%% LOAD DESIGN AND TRUE PARAMETERS

with open(DATA_DIR / "ReadyData_Synth_A.pkl", "rb") as f:
    data_dict = pickle.load(f)

design_cov_mat = data_dict['cov_mat']
design_grp_idx = data_dict['grp_idx']
params_fixed = data_dict['params_fixed']   # [gam_h, gam_l, beta0, beta1], one value per group
names_grp_idx = data_dict['names_grp_idx']
groups = sorted(names_grp_idx, key=names_grp_idx.get)   # ['left_uni','left_bi','right_uni','right_bi']

PARAM_NAMES = ['gam_h', 'gam_l', 'b0', 'b1', 'PSE', 'JND']

# true values, with PSE and JND computed the same way as in Build_Model_A.py
true_gam_h, true_gam_l, true_b0, true_b1 = [np.asarray(v, dtype=float) for v in params_fixed]
true_vals = {
    'gam_h': true_gam_h,
    'gam_l': true_gam_l,
    'b0': true_b0,
    'b1': true_b1,
    'PSE': (-true_b0 + np.log((1 - 2*true_gam_h) / (1 - 2*true_gam_l))) / true_b1,
    'JND': np.log(((3 - 4*true_gam_h)*(3 - 4*true_gam_l)) / ((1 - 4*true_gam_h)*(1 - 4*true_gam_l))) / (2*true_b1),
}

#%% HELPERS

# same generator as in Data_A/DataProcessing_A.py
def synth_generator_A(params_fixed, cov_mat, grp_idx, rng):
    [gam_h_fix, gam_l_fix, beta0_fix, beta1_fix] = params_fixed
    beta_mat = np.vstack([beta0_fix, beta1_fix])
    log_arg = (cov_mat @ beta_mat)
    psis_mat = gam_h_fix + (1-gam_h_fix-gam_l_fix)/( 1 + np.exp( - log_arg ))
    psis = psis_mat[np.arange(len(grp_idx)), grp_idx]
    synth_resp = binom.rvs(n=1, p=psis, random_state=rng)
    return synth_resp

def split_params(ds):
    """Map each name in PARAM_NAMES to a DataArray with a 'groups' dim,
    splitting beta_vec into b0 and b1. Works on posterior, rhat and ess datasets."""
    return {
        'gam_h': ds['gam_h'],
        'gam_l': ds['gam_l'],
        'b0': ds['beta_vec'].sel(betas='b0'),
        'b1': ds['beta_vec'].sel(betas='b1'),
        'PSE': ds['PSE'],
        'JND': ds['JND'],
    }

#%% GENERATE DATA AND FIT MODEL A AT EACH SIZE

rows = []
prop_rows = []      # observed response counts per stimulus level, instead of saving every response
post_samples = {}   # (rep, size) -> {param: array (n_samples, n_groups)}
var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND']

for rep in range(N_REPS):
    rng = rep_rngs[rep]
    # one long synthetic run per replicate; smaller datasets are its prefixes
    trial_draw = rng.integers(0, len(design_grp_idx), size=max(SIZES))
    rep_cov_mat = design_cov_mat[trial_draw]
    rep_grp_idx = design_grp_idx[trial_draw]
    rep_resp = synth_generator_A(params_fixed, rep_cov_mat, rep_grp_idx, rng)

    for size in SIZES:
        print(f"--- replicate {rep+1}/{N_REPS}, {size} trials ---")
        # Build_Model_A.py reads these globals
        cov_mat = rep_cov_mat[:size]
        grp_idx = rep_grp_idx[:size]
        obs_data = rep_resp[:size]

        counts = (pd.DataFrame({'group': np.asarray(groups)[grp_idx], 'stim': cov_mat[:, 1], 'resp': obs_data})
                  .groupby(['group', 'stim'])['resp'].agg(n_trials='size', n_high='sum').reset_index())
        counts['prop_high'] = counts['n_high'] / counts['n_trials']
        counts.insert(0, 'size', size)
        counts.insert(0, 'rep', rep)
        prop_rows.append(counts)

        exec(open(MODEL_DIR / "Build_Model_A.py").read())

        with model_A:
            trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CORES,
                              random_seed=rng, progressbar=True, mp_ctx=MP_CTX)

        post = split_params(trace.posterior[var_names])
        rhat = split_params(az.rhat(trace, var_names=var_names))
        ess = split_params(az.ess(trace, var_names=var_names))
        n_div = int(trace.sample_stats['diverging'].sum())
        grp_counts = np.bincount(grp_idx, minlength=len(groups))

        post_samples[(rep, size)] = {}
        for par in PARAM_NAMES:
            samps = post[par].stack(sample=('chain', 'draw')).transpose('sample', 'groups')
            post_samples[(rep, size)][par] = samps.values.astype(np.float32)
            for g_i, grp in enumerate(groups):
                s = samps.sel(groups=grp).values
                hdi_low, hdi_high = az.hdi(s, hdi_prob=HDI_PROB)
                truth = true_vals[par][g_i]
                rows.append({
                    'rep': rep, 'size': size, 'group': grp, 'group_trials': grp_counts[g_i],
                    'param': par, 'true': truth,
                    'mean': s.mean(), 'sd': s.std(), 'hdi_low': hdi_low, 'hdi_high': hdi_high,
                    'error': s.mean() - truth,
                    'covered': hdi_low <= truth <= hdi_high,
                    'r_hat': float(rhat[par].sel(groups=grp)),
                    'ess_bulk': float(ess[par].sel(groups=grp)),
                    'divergences': n_div,
                })

results = pd.DataFrame(rows)
obs_props = pd.concat(prop_rows, ignore_index=True)
print("FINISHED SAMPLING!")

with open(MODEL_DIR / "Results_Convergence_A.pkl", "wb") as f:
    pickle.dump({'results': results, 'post_samples': post_samples, 'obs_props': obs_props,
                 'groups': groups, 'true_vals': true_vals, 'sizes': SIZES, 'n_reps': N_REPS,
                 'hdi_prob': HDI_PROB, 'seed': SEED}, f)

#%% (optional) reload saved results instead of refitting

# with open(MODEL_DIR / "Results_Convergence_A.pkl", "rb") as f:
#     saved = pickle.load(f)
# results, post_samples, obs_props = saved['results'], saved['post_samples'], saved['obs_props']
# groups, true_vals, SIZES, N_REPS, HDI_PROB = saved['groups'], saved['true_vals'], saved['sizes'], saved['n_reps'], saved['hdi_prob']

#%% Sampling diagnostics: r_hat should be ~1 and there should be no divergences

diag = results.groupby('size').agg(max_r_hat=('r_hat', 'max'), min_ess=('ess_bulk', 'min'),
                                   divergences=('divergences', 'max'))
print(diag.to_string())

#%% Coverage: fraction of fits whose 95% HDI contains the true value (expect ~0.95)

coverage = results.pivot_table(index='param', columns='size', values='covered', aggfunc='mean')
n_cover = N_REPS * len(groups)   # fits per size/parameter cell
# central 95% range of the coverage of n_cover fits from a calibrated model
cover_lo, cover_hi = binom.ppf([0.025, 0.975], n_cover, HDI_PROB) / n_cover
print(coverage.loc[PARAM_NAMES].round(2).to_string())
print(f"{n_cover} fits per cell; a calibrated model gives {cover_lo:.2f}-{cover_hi:.2f} in 95% of cells")

#%% Plot colors: one per group, fixed order; blue ramp (light -> dark) for dataset size

CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GROUP_COLORS = dict(zip(groups, CATEGORICAL))
# one color per replicate where replicates share a panel; past 8 they would be indistinguishable, so use one color
# the per-replicate plots overlay only the first SHOW_REPS replicates; the summaries use all of them
SHOW_REPS = min(SHOW_REPS, N_REPS)
REP_COLORS = CATEGORICAL[:SHOW_REPS] if SHOW_REPS <= len(CATEGORICAL) else [CATEGORICAL[0]] * SHOW_REPS
SIZE_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SIZE_COLORS = dict(zip(SIZES, [SIZE_RAMP[i] for i in np.linspace(0, len(SIZE_RAMP) - 1, len(SIZES)).round().astype(int)]))

def size_axis(ax):
    # log x-axis limited to the fitted sizes, ticked at each size
    ax.set_xscale('log')
    ax.set_xlim(min(SIZES) / 1.4, max(SIZES) * 1.4)
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

def save_plot(fig, name):
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=200, bbox_inches='tight')

#%% Coverage vs dataset size, with the range a calibrated model would give

cover_grp = results.groupby(['param', 'group', 'size'])['covered'].mean()

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, constrained_layout=True, figsize=(12, 7))
for ax, par in zip(axes.ravel(), PARAM_NAMES):
    ax.axhspan(cover_lo, cover_hi, color='#52514e', alpha=0.15, linewidth=0,
               label=f'calibrated range ({n_cover} fits)')
    ax.axhline(HDI_PROB, linestyle='--', linewidth=1.5, color='#52514e', label=f'nominal {HDI_PROB:.2f}')
    for grp in groups:
        ax.plot(SIZES, cover_grp.loc[(par, grp)].loc[SIZES].values, marker='o', markersize=3, linewidth=1,
                alpha=0.6, color=GROUP_COLORS[grp], label=f'{grp} ({N_REPS} fits)')
    ax.plot(SIZES, coverage.loc[par, SIZES].values, marker='o', markersize=5, linewidth=2.5,
            color='#0b0b0b', label='all groups')
    size_axis(ax)
    ax.set_title(par)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel('number of trials (all groups)')
for ax in axes[:, 0]:
    ax.set_ylabel(f'fraction of {int(HDI_PROB*100)}% HDIs containing the truth')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right center')
fig.suptitle(f'Coverage of the {int(HDI_PROB*100)}% HDI vs dataset size ({N_REPS} replicates)', fontsize=14)
save_plot(fig, "coverage_vs_size_all_params")
plt.show()

#%% Posterior mean and 95% HDI vs dataset size, one figure per parameter

rep_offsets = np.exp(np.linspace(-0.08, 0.08, SHOW_REPS)) if SHOW_REPS > 1 else [1.0]   # spread reps on the log x-axis

for par in PARAM_NAMES:
    fig, axes = plt.subplots(2, 2, sharex=True, constrained_layout=True, figsize=(9, 6))
    for ax, grp in zip(axes.ravel(), groups):
        sub = results[(results['param'] == par) & (results['group'] == grp)]
        for rep in range(SHOW_REPS):
            r = sub[sub['rep'] == rep]
            ax.errorbar(r['size'] * rep_offsets[rep], r['mean'],
                        yerr=[r['mean'] - r['hdi_low'], r['hdi_high'] - r['mean']],
                        fmt='o', markersize=4, linewidth=1.5, capsize=0, color='#2a78d6',
                        label=f'posterior mean ± {int(HDI_PROB*100)}% HDI' if rep == 0 else None)
        ax.axhline(sub['true'].iloc[0], linestyle='--', linewidth=1.5, color='#52514e', label='true value')
        size_axis(ax)
        ax.set_title(grp)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('number of trials (all groups)')
    for ax in axes[:, 0]:
        ax.set_ylabel(par)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Recovery of {par} vs dataset size (first {SHOW_REPS} of {N_REPS} replicates)', fontsize=14)
    save_plot(fig, f"recovery_{par}_mean_hdi_vs_size")
    plt.show()

#%% Posterior sd vs dataset size (log-log); dashed line has the 1/sqrt(N) slope

sd_mean = results.groupby(['param', 'group', 'size'])['sd'].mean()
sizes_arr = np.array(SIZES, dtype=float)

fig, axes = plt.subplots(2, 3, sharex=True, constrained_layout=True, figsize=(12, 7))
for ax, par in zip(axes.ravel(), PARAM_NAMES):
    for grp in groups:
        ax.plot(sizes_arr, sd_mean.loc[(par, grp)].loc[SIZES].values, marker='o', markersize=4,
                linewidth=2, color=GROUP_COLORS[grp], label=grp)
    anchor = sd_mean.loc[par].xs(SIZES[0], level='size').mean()
    ax.plot(sizes_arr, anchor * np.sqrt(sizes_arr[0] / sizes_arr), linestyle='--', linewidth=1.5,
            color='#52514e', label=r'$\propto 1/\sqrt{N}$')
    ax.set_yscale('log')
    size_axis(ax)
    ax.set_title(par)
    ax.grid(alpha=0.3, which='both')
for ax in axes[1]:
    ax.set_xlabel('number of trials (all groups)')
for ax in axes[:, 0]:
    ax.set_ylabel('posterior sd (mean over reps)')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right center')
fig.suptitle('Posterior uncertainty vs dataset size', fontsize=14)
save_plot(fig, "posterior_sd_vs_size_all_params")
plt.show()

#%% Posterior densities, all replicates overlaid; darker = more trials

for par in PARAM_NAMES:
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(9, 6))
    for g_i, (ax, grp) in enumerate(zip(axes.ravel(), groups)):
        for size in SIZES:
            for rep in range(SHOW_REPS):
                vals = post_samples[(rep, size)][par][:, g_i]
                vals = vals[np.isfinite(vals)]
                az.plot_kde(vals, ax=ax, plot_kwargs={'color': SIZE_COLORS[size], 'linewidth': 1.5, 'alpha': 0.85},
                            label=f'{size}' if rep == 0 else None)
        ax.axvline(true_vals[par][g_i], linestyle='--', linewidth=1.5, color='#52514e', label='true value')
        ax.set_title(grp)
        ax.set_yticks([])
        ax.tick_params(labelsize=9)
        ax.set_xlabel(par)
        if ax.get_legend():
            ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='trials', loc='outside right center')
    fig.suptitle(f'Posterior of {par} as data grows (first {SHOW_REPS} of {N_REPS} replicates overlaid)', fontsize=14)
    save_plot(fig, f"posterior_density_{par}_by_size")
    plt.show()

#%% Psychometric curves: posterior median and pointwise 95% HDI band for every fit

stim_grid = np.linspace(design_cov_mat[:, 1].min(), design_cov_mat[:, 1].max(), 200)

def psych_curves(gam_h, gam_l, beta0, beta1, X):
    """Psychometric function with lapses for arrays of parameter samples.
    Returns shape (n_samples, len(X))."""
    gam_h, gam_l, beta0, beta1 = (np.atleast_1d(v)[:, None] for v in (gam_h, gam_l, beta0, beta1))
    return gam_h + (1 - gam_h - gam_l) / (1 + np.exp(-(beta0 + beta1 * X)))

true_curves = {grp: psych_curves(*(true_vals[p][g_i] for p in ['gam_h', 'gam_l', 'b0', 'b1']), stim_grid)[0]
               for g_i, grp in enumerate(groups)}

curve_bands = {}    # (rep, size, group) -> (median curve, hdi array (len(stim_grid), 2))
curve_rows = []
for (rep, size), samps in post_samples.items():
    for g_i, grp in enumerate(groups):
        curves = psych_curves(*(samps[p][:, g_i] for p in ['gam_h', 'gam_l', 'b0', 'b1']), stim_grid)
        med = np.median(curves, axis=0)
        hdi = az.hdi(curves[None], hdi_prob=HDI_PROB)   # (chain, draw, x) convention -> one band per x
        curve_bands[(rep, size, grp)] = (med, hdi)
        curve_rows.append({'rep': rep, 'size': size, 'group': grp,
                           'mean_hdi_width': np.mean(hdi[:, 1] - hdi[:, 0]),
                           'max_abs_error': np.max(np.abs(med - true_curves[grp])),
                           'truth_in_band': np.mean((hdi[:, 0] <= true_curves[grp]) & (true_curves[grp] <= hdi[:, 1]))})
curve_summary = pd.DataFrame(curve_rows)

#%% A. Curve and HDI band for each dataset size (rows) and group (columns), replicates overlaid

fig, axes = plt.subplots(len(SIZES), len(groups), sharex=True, sharey=True, constrained_layout=True,
                         figsize=(12, 2.2 * len(SIZES) + 0.8))
axes = np.atleast_2d(axes)
for r, size in enumerate(SIZES):
    for ax, grp in zip(axes[r], groups):
        for rep in range(SHOW_REPS):
            med, hdi = curve_bands[(rep, size, grp)]
            obs = obs_props[(obs_props['rep'] == rep) & (obs_props['size'] == size) & (obs_props['group'] == grp)]
            ax.fill_between(stim_grid, hdi[:, 0], hdi[:, 1], color=REP_COLORS[rep], alpha=0.15, linewidth=0)
            ax.plot(stim_grid, med, color=REP_COLORS[rep], linewidth=1.8)
            ax.scatter(obs['stim'], obs['prop_high'], s=12 + 60 * obs['n_trials'] / obs_props['n_trials'].max(),
                       color=REP_COLORS[rep], edgecolor='white', linewidth=0.8, zorder=3)
        ax.plot(stim_grid, true_curves[grp], linestyle='--', linewidth=1.5, color='#0b0b0b', zorder=4)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(grp)
    axes[r, 0].set_ylabel(f'{size} trials\nP("high")')
for ax in axes[-1]:
    ax.set_xlabel('stimulus (normalized)')
handles = [(matplotlib.patches.Patch(color=REP_COLORS[rep], alpha=0.3),
            matplotlib.lines.Line2D([], [], color=REP_COLORS[rep], linewidth=1.8)) for rep in range(SHOW_REPS)]
labels = [f'replicate {rep}' for rep in range(SHOW_REPS)]
handles += [matplotlib.lines.Line2D([], [], linestyle='--', linewidth=1.5, color='#0b0b0b'),
            matplotlib.lines.Line2D([], [], marker='o', linestyle='', color='#52514e')]
labels += ['true curve', 'observed proportion']
fig.legend(handles, labels, loc='outside lower center', ncol=len(labels))
fig.suptitle(f'Psychometric curve recovery by dataset size: posterior median and {int(HDI_PROB*100)}% HDI '
             f'(first {SHOW_REPS} of {N_REPS} replicates); dot area ∝ trials', fontsize=14)
save_plot(fig, "psych_curve_hdi_by_size")
plt.show()

#%% B. Deviation of the recovered curve from the true curve; smallest, middle and largest sizes (rows), replicates overlaid

sizes_show = sorted({SIZES[0], SIZES[len(SIZES) // 2], SIZES[-1]})

fig, axes = plt.subplots(len(sizes_show), len(groups), sharex=True, sharey=True, constrained_layout=True,
                         figsize=(14, 2.6 * len(sizes_show) + 0.8))
axes = np.atleast_2d(axes)
for r, size in enumerate(sizes_show):
    for ax, grp in zip(axes[r], groups):
        for rep in range(SHOW_REPS):
            med, hdi = curve_bands[(rep, size, grp)]
            ax.fill_between(stim_grid, hdi[:, 0] - true_curves[grp], hdi[:, 1] - true_curves[grp],
                            color=REP_COLORS[rep], alpha=0.15, linewidth=0)
            ax.plot(stim_grid, med - true_curves[grp], color=REP_COLORS[rep], linewidth=1.8)
        ax.axhline(0, linestyle='--', linewidth=1.5, color='#0b0b0b', zorder=4)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(grp)
    axes[r, 0].set_ylabel(f'{size} trials\nposterior − true')
for ax in axes[-1]:
    ax.set_xlabel('stimulus (normalized)')
handles = [(matplotlib.patches.Patch(color=REP_COLORS[rep], alpha=0.3),
            matplotlib.lines.Line2D([], [], color=REP_COLORS[rep], linewidth=1.8)) for rep in range(SHOW_REPS)]
labels = [f'replicate {rep}' for rep in range(SHOW_REPS)]
handles.append(matplotlib.lines.Line2D([], [], linestyle='--', linewidth=1.5, color='#0b0b0b'))
labels.append('true curve')
fig.legend(handles, labels, loc='outside lower center', ncol=len(labels))
fig.suptitle(f'Deviation of recovered P("high") from the true curve: posterior median and {int(HDI_PROB*100)}% HDI '
             f'(first {SHOW_REPS} of {N_REPS} replicates)', fontsize=14)
save_plot(fig, "psych_curve_deviation_from_truth")
plt.show()

#%% C. Curve-level convergence summaries vs dataset size (mean over replicates)

curve_means = curve_summary.groupby(['group', 'size'])[['mean_hdi_width', 'max_abs_error']].mean()

fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4.5))
for ax, col, title in zip(axes, ['mean_hdi_width', 'max_abs_error'],
                          [f'average {int(HDI_PROB*100)}% HDI width of the curve',
                           'largest |posterior median − true curve|']):
    for grp in groups:
        ax.plot(SIZES, curve_means.loc[grp].loc[SIZES, col].values, marker='o', markersize=4,
                linewidth=2, color=GROUP_COLORS[grp], label=grp)
    ax.set_yscale('log')
    size_axis(ax)
    ax.set_title(title)
    ax.set_xlabel('number of trials (all groups)')
    ax.grid(alpha=0.3, which='both')
axes[0].set_ylabel('P("high")')
axes[0].legend()
fig.suptitle('Psychometric curve convergence', fontsize=14)
save_plot(fig, "psych_curve_hdi_width_and_error_vs_size")
plt.show()


#%% Fraction of the stimulus range where the true curve lies inside the HDI band

print(curve_summary.pivot_table(index='group', columns='size', values='truth_in_band').round(2).to_string())
