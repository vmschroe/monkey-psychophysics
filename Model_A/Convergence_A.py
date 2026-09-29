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

Runtime: roughly 7 min per 13k trials with the default sampler settings, so
the defaults (sum of SIZES ~32k trials, 3 replicates) take about an hour.
Lower N_REPS / DRAWS for a quick look.

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

#%% SETTINGS

SIZES = [250, 500, 1000, 2000, 4000, 8000, 16000]   # total trials per dataset
N_REPS = 3                                          # independent synthetic runs
DRAWS = 1000
TUNE = 1000
CHAINS = 4
HDI_PROB = 0.95
SEED = 20260929

rng = np.random.default_rng(SEED)

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
post_samples = {}   # (rep, size) -> {param: array (n_samples, n_groups)}
var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND']

for rep in range(N_REPS):
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
        exec(open(MODEL_DIR / "Build_Model_A.py").read())

        with model_A:
            trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                              random_seed=rng, progressbar=True)

        post = split_params(trace.posterior[var_names])
        rhat = split_params(az.rhat(trace, var_names=var_names))
        ess = split_params(az.ess(trace, var_names=var_names))
        n_div = int(trace.sample_stats['diverging'].sum())
        grp_counts = np.bincount(grp_idx, minlength=len(groups))

        post_samples[(rep, size)] = {}
        for par in PARAM_NAMES:
            samps = post[par].stack(sample=('chain', 'draw')).transpose('sample', 'groups')
            post_samples[(rep, size)][par] = samps.values
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
print("FINISHED SAMPLING!")

with open(MODEL_DIR / "Results_Convergence_A.pkl", "wb") as f:
    pickle.dump({'results': results, 'post_samples': post_samples, 'groups': groups,
                 'true_vals': true_vals, 'sizes': SIZES, 'n_reps': N_REPS,
                 'hdi_prob': HDI_PROB, 'seed': SEED}, f)

#%% (optional) reload saved results instead of refitting

# with open(MODEL_DIR / "Results_Convergence_A.pkl", "rb") as f:
#     saved = pickle.load(f)
# results, post_samples, groups = saved['results'], saved['post_samples'], saved['groups']
# true_vals, SIZES, N_REPS, HDI_PROB = saved['true_vals'], saved['sizes'], saved['n_reps'], saved['hdi_prob']

#%% Sampling diagnostics: r_hat should be ~1 and there should be no divergences

diag = results.groupby('size').agg(max_r_hat=('r_hat', 'max'), min_ess=('ess_bulk', 'min'),
                                   divergences=('divergences', 'max'))
print(diag.to_string())

#%% Coverage: fraction of fits whose 95% HDI contains the true value (expect ~0.95)

coverage = results.pivot_table(index='param', columns='size', values='covered', aggfunc='mean')
print(coverage.loc[PARAM_NAMES].round(2).to_string())

#%% Plot colors: one per group, fixed order; blue ramp (light -> dark) for dataset size

GROUP_COLORS = dict(zip(groups, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
SIZE_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SIZE_COLORS = dict(zip(SIZES, [SIZE_RAMP[i] for i in np.linspace(0, len(SIZE_RAMP) - 1, len(SIZES)).round().astype(int)]))

def size_axis(ax):
    # log x-axis limited to the fitted sizes, ticked at each size
    ax.set_xscale('log')
    ax.set_xlim(min(SIZES) / 1.4, max(SIZES) * 1.4)
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

#%% Posterior mean and 95% HDI vs dataset size, one figure per parameter

rep_offsets = np.exp(np.linspace(-0.08, 0.08, N_REPS)) if N_REPS > 1 else [1.0]   # spread reps on the log x-axis

for par in PARAM_NAMES:
    fig, axes = plt.subplots(2, 2, sharex=True, constrained_layout=True, figsize=(9, 6))
    for ax, grp in zip(axes.ravel(), groups):
        sub = results[(results['param'] == par) & (results['group'] == grp)]
        for rep in range(N_REPS):
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
    fig.suptitle(f'Recovery of {par} vs dataset size', fontsize=14)
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
plt.show()

#%% Posterior densities for one replicate, darker = more trials

rep_show = 0

for par in PARAM_NAMES:
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(9, 6))
    for g_i, (ax, grp) in enumerate(zip(axes.ravel(), groups)):
        for size in SIZES:
            vals = post_samples[(rep_show, size)][par][:, g_i]
            vals = vals[np.isfinite(vals)]
            az.plot_kde(vals, ax=ax, plot_kwargs={'color': SIZE_COLORS[size], 'linewidth': 2},
                        label=f'{size}')
        ax.axvline(true_vals[par][g_i], linestyle='--', linewidth=1.5, color='#52514e', label='true value')
        ax.set_title(grp)
        ax.set_yticks([])
        ax.tick_params(labelsize=9)
        ax.set_xlabel(par)
        if ax.get_legend():
            ax.get_legend().remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='trials', loc='outside right center')
    fig.suptitle(f'Posterior of {par} as data grows (replicate {rep_show})', fontsize=14)
    plt.show()
