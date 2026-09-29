#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2026

Posterior convergence study for Model B (hierarchical over sessions).

Generates synthetic datasets from the fixed hyperparameters in
ReadyData_Synth_B.pkl, fits Model B to each, and tracks how the recovered
posteriors approach the true values. Dataset size is varied in two
separate sweeps:
    - 'sessions': the number of sessions grows, each session keeping the
      real average number of trials. This mainly shows recovery of the
      hyperparameters (group means and between-session sds).
    - 'trials': the number of sessions is fixed at the real 42 and the
      trials per session grow. This mainly shows recovery of the
      session-level parameters.

Two levels of true values are tracked:
    - hyperparameters (mu and sig of b0, b1, gam_h, gam_l for each group;
      the gamma ones on the logit scale used by Build_Model_B.py), plotted
      individually;
    - session-level parameters (gam_h, gam_l, b0, b1, PSE, JND for every
      session and group), summarized across sessions as RMSE of the
      posterior mean, mean HDI width and HDI coverage.

Design:
    - Every replicate draws new true session-level parameters from the
      fixed hyperparameters, as in Data_B/DataProcessing_B.py.
    - Groups cycle 0,1,2,3 within a session (the real sessions are balanced
      across groups) and each trial's stimulus is resampled from that
      group's stimuli in the real design.
    - Within a replicate the datasets of a sweep are nested: fewer sessions
      means the first n sessions, fewer trials means the first t trials of
      every session. Differences between sizes then come from the extra
      data rather than from a new random draw.

Notes on reading the results:
    - In the 'trials' sweep the hyperparameter posteriors can at best
      converge to the mean/sd of the 42 sessions actually drawn, not to
      the population values.
    - The true sig_b1 values are tiny (0.002-0.008) next to the prior on
      sig_b1 (Exponential with mean 4), so they may not be recoverable at
      these session counts.

Runtime: very roughly 1.5 min per 1000 trials per fit with the default
sampler settings, so the defaults (~74k trials per replicate over both
sweeps, 2 replicates) take about 4 hours. Results are saved after every
fit, so an interrupted run keeps the fits already done.

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
import time
import xarray as xr
from pathlib import Path

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
DATA_DIR = MODEL_DIR / "Data_B"
PLOT_DIR = MODEL_DIR / "Plots_B" / "Convergence_Plots_B"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = MODEL_DIR / "Results_Convergence_B.pkl"

#%% SETTINGS

SESSION_SWEEP = [5, 10, 20, 40, 80]     # number of sessions, each with TRIALS_PER_SESSION_FIXED trials
TRIALS_PER_SESSION_FIXED = 308          # real average trials per session
TRIALS_SWEEP = [20, 40, 80, 160, 320]   # trials per session (multiples of 4 keep groups balanced)
N_SESSIONS_FIXED = 42                   # real number of sessions
N_REPS = 2                              # independent synthetic runs of each sweep
DRAWS = 1000
TUNE = 1000
CHAINS = 4
HDI_PROB = 0.95
SEED = 20260930

rng = np.random.default_rng(SEED)

SWEEPS = {
    'sessions': {'sizes': SESSION_SWEEP,
                 'label': f'number of sessions ({TRIALS_PER_SESSION_FIXED} trials each)'},
    'trials': {'sizes': TRIALS_SWEEP,
               'label': f'trials per session ({N_SESSIONS_FIXED} sessions)'},
}

#%% LOAD DESIGN AND TRUE HYPERPARAMETERS

with open(DATA_DIR / "ReadyData_Synth_B.pkl", "rb") as f:
    data_dict = pickle.load(f)

design_cov_mat = data_dict['cov_mat']
design_grp_idx = data_dict['grp_idx']
hparams_fixed_B = data_dict['hparams_fixed_B']
names_grp_idx = data_dict['names_grp_idx']
groups = sorted(names_grp_idx, key=names_grp_idx.get)   # ['left_uni','left_bi','right_uni','right_bi']

# grp_idx = 2*hand + manual, as in Data_B/DataProcessing_B.py
GROUP_HAND_MANUAL = {grp: (['left', 'right'][g_i // 2], ['uni', 'bi'][g_i % 2]) for g_i, grp in enumerate(groups)}

# stimulus values each group received in the real design
stim_pool = {g_i: design_cov_mat[design_grp_idx == g_i, 1] for g_i in range(len(groups))}

# hyperparameter name -> (par_type, par_idx, hp_type) in hparams_fixed_B
HYPER_KEYS = {
    'mu_b0': ('beta', 'h0', 'mu'),
    'mu_b1': ('beta', 'l1', 'mu'),
    'mu_gam_h': ('gamma', 'h0', 'mu'),
    'mu_gam_l': ('gamma', 'l1', 'mu'),
    'sig_b0': ('beta', 'h0', 'sig'),
    'sig_b1': ('beta', 'l1', 'sig'),
    'sig_gam_h': ('gamma', 'h0', 'sig'),
    'sig_gam_l': ('gamma', 'l1', 'sig'),
}
HYPER_NAMES = list(HYPER_KEYS)
HYPER_FAMILIES = {'means': HYPER_NAMES[:4], 'sds': HYPER_NAMES[4:]}
SESSION_PARAMS = ['gam_h', 'gam_l', 'b0', 'b1', 'PSE', 'JND']

true_hyper = {name: np.array([float(hparams_fixed_B.loc[GROUP_HAND_MANUAL[grp] + key]) for grp in groups])
              for name, key in HYPER_KEYS.items()}

#%% HELPERS

def pse_jnd(gam_h, gam_l, b0, b1):
    # same formulas as Build_Model_B.py
    PSE = (-b0 + np.log((1 - 2*gam_h) / (1 - 2*gam_l))) / b1
    JND = np.log(((3 - 4*gam_h)*(3 - 4*gam_l)) / ((1 - 4*gam_h)*(1 - 4*gam_l))) / (2*b1)
    return PSE, JND

def sample_session_params(n_sessions, rng):
    """True session-level parameters drawn from the fixed hyperparameters, as in
    Data_B/DataProcessing_B.py. Returns {param: array (n_groups, n_sessions)}."""
    shape = (len(groups), n_sessions)
    hp = {name: vals[:, None] for name, vals in true_hyper.items()}
    tp = {
        'b0': hp['mu_b0'] + hp['sig_b0'] * rng.standard_normal(shape),
        'b1': hp['mu_b1'] + hp['sig_b1'] * rng.standard_normal(shape),
        'gam_h': 0.25 / (1 + np.exp(-(hp['sig_gam_h'] * rng.standard_normal(shape) + hp['mu_gam_h']))),
        'gam_l': 0.25 / (1 + np.exp(-(hp['sig_gam_l'] * rng.standard_normal(shape) + hp['mu_gam_l']))),
    }
    tp['PSE'], tp['JND'] = pse_jnd(tp['gam_h'], tp['gam_l'], tp['b0'], tp['b1'])
    return tp

def simulate_sessions(true_sess, trials_per_session, rng):
    """Stimulus, group and response for every trial of every session.
    Groups cycle 0,1,2,3 so any prefix of a session stays balanced across groups.
    Returns three arrays of shape (n_sessions, trials_per_session)."""
    n_sessions = true_sess['b0'].shape[1]
    n_cycles = -(-trials_per_session // len(groups))
    grp = np.tile(np.arange(len(groups)), (n_sessions, n_cycles))[:, :trials_per_session]
    stim = np.empty(grp.shape)
    for g_i in range(len(groups)):
        mask = grp == g_i
        stim[mask] = rng.choice(stim_pool[g_i], size=mask.sum())
    sess = np.broadcast_to(np.arange(n_sessions)[:, None], grp.shape)
    par = {p: true_sess[p][grp, sess] for p in ['gam_h', 'gam_l', 'b0', 'b1']}
    psi = par['gam_h'] + (1 - par['gam_h'] - par['gam_l']) / (1 + np.exp(-(par['b0'] + par['b1'] * stim)))
    resp = rng.binomial(1, psi)
    return stim, grp, resp

def split_hyper(ds):
    """Map each name in HYPER_NAMES to a DataArray with a 'groups' dim."""
    return {
        'mu_b0': ds['mu_betas'].sel(betas='b0'),
        'mu_b1': ds['mu_betas'].sel(betas='b1'),
        'sig_b0': ds['sig_betas'].sel(betas='b0'),
        'sig_b1': ds['sig_betas'].sel(betas='b1'),
        # Build_Model_B.py indexes the gamma hyperparameters with the 'betas' dim: b0 -> gam_h, b1 -> gam_l
        'mu_gam_h': ds['mu_gams'].sel(betas='b0'),
        'mu_gam_l': ds['mu_gams'].sel(betas='b1'),
        'sig_gam_h': ds['sig_gams'].sel(betas='b0'),
        'sig_gam_l': ds['sig_gams'].sel(betas='b1'),
    }

def split_session(ds):
    """Map each name in SESSION_PARAMS to a DataArray with 'groups' and 'sessions' dims."""
    return {
        'gam_h': ds['gam_h'],
        'gam_l': ds['gam_l'],
        'b0': ds['beta_vec'].sel(betas='b0'),
        'b1': ds['beta_vec'].sel(betas='b1'),
        'PSE': ds['PSE'],
        'JND': ds['JND'],
    }

HYPER_VARS = ['mu_betas', 'sig_betas', 'mu_gams', 'sig_gams']
SESSION_VARS = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND']
BUILD_MODEL_B = (MODEL_DIR / "Build_Model_B.py").read_text()

def fit_model_B(cov_mat, grp_idx, sess_idx, obs_data, n_sessions):
    # Build_Model_B.py reads these names; running it in its own namespace keeps it
    # from overwriting anything in this script
    ns = {'cov_mat': cov_mat, 'grp_idx': grp_idx, 'sess_idx': sess_idx, 'obs_data': obs_data,
          'sessions': [f's{i:02d}' for i in range(n_sessions)]}
    exec(BUILD_MODEL_B, ns)
    with ns['model_B']:
        return pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=1, random_seed=rng, progressbar=True)

hyper_rows = []
session_frames = []
hyper_samples = {}   # (sweep, rep, size) -> {hyper name: float32 array (n_samples, n_groups)}

def run_fit(sweep, rep, size, true_sess, stim, grp, resp, n_sessions, n_trials):
    """Fit Model B to the first n_sessions sessions and first n_trials trials of each,
    and record hyperparameter and session-level recovery."""
    t0 = time.time()
    print(f"--- {sweep} sweep, replicate {rep+1}/{N_REPS}, {n_sessions} sessions x {n_trials} trials ---")
    stim_d, grp_d, resp_d = (a[:n_sessions, :n_trials].ravel() for a in (stim, grp, resp))
    sess_d = np.repeat(np.arange(n_sessions), n_trials)
    cov_mat = np.column_stack([np.ones_like(stim_d), stim_d])
    trace = fit_model_B(cov_mat, grp_d, sess_d, resp_d, n_sessions)
    n_div = int(trace.sample_stats['diverging'].sum())

    # hyperparameters: one true value per group
    post = split_hyper(trace.posterior[HYPER_VARS])
    rhat = split_hyper(az.rhat(trace, var_names=HYPER_VARS))
    ess = split_hyper(az.ess(trace, var_names=HYPER_VARS))
    hyper_samples[(sweep, rep, size)] = {}
    for name in HYPER_NAMES:
        samps = post[name].stack(sample=('chain', 'draw')).transpose('sample', 'groups').values
        hyper_samples[(sweep, rep, size)][name] = samps.astype(np.float32)
        for g_i, grp_name in enumerate(groups):
            s = samps[:, g_i]
            hdi_low, hdi_high = az.hdi(s, hdi_prob=HDI_PROB)
            truth = true_hyper[name][g_i]
            hyper_rows.append({
                'sweep': sweep, 'rep': rep, 'size': size, 'n_sessions': n_sessions, 'n_trials': n_trials,
                'group': grp_name, 'param': name, 'true': truth,
                'mean': s.mean(), 'sd': s.std(), 'hdi_low': hdi_low, 'hdi_high': hdi_high,
                'error': s.mean() - truth, 'covered': hdi_low <= truth <= hdi_high,
                'r_hat': float(rhat[name].sel(groups=grp_name)),
                'ess_bulk': float(ess[name].sel(groups=grp_name)),
                'divergences': n_div,
            })

    # session level: posterior mean and HDI for every session, next to that session's true value
    post = split_session(trace.posterior[SESSION_VARS])
    hdi = split_session(az.hdi(trace, var_names=SESSION_VARS, hdi_prob=HDI_PROB))
    rhat = split_session(az.rhat(trace, var_names=SESSION_VARS))
    g_grid, s_grid = np.meshgrid(np.arange(len(groups)), np.arange(n_sessions), indexing='ij')
    for par in SESSION_PARAMS:
        order = ('groups', 'sessions')
        session_frames.append(pd.DataFrame({
            'sweep': sweep, 'rep': rep, 'size': size, 'param': par,
            'group': np.asarray(groups)[g_grid.ravel()], 'session': s_grid.ravel(),
            'true': true_sess[par][:, :n_sessions].ravel(),
            'mean': post[par].mean(('chain', 'draw')).transpose(*order).values.ravel(),
            'hdi_low': hdi[par].sel(hdi='lower').transpose(*order).values.ravel(),
            'hdi_high': hdi[par].sel(hdi='higher').transpose(*order).values.ravel(),
            'r_hat': rhat[par].transpose(*order).values.ravel(),
            'divergences': n_div,
        }))

    save_results()
    print(f"    done in {(time.time() - t0)/60:.1f} min, {n_div} divergences")

def collect_results():
    hyper_results = pd.DataFrame(hyper_rows)
    session_results = pd.concat(session_frames, ignore_index=True)
    session_results['error'] = session_results['mean'] - session_results['true']
    session_results['hdi_width'] = session_results['hdi_high'] - session_results['hdi_low']
    session_results['covered'] = ((session_results['hdi_low'] <= session_results['true'])
                                  & (session_results['true'] <= session_results['hdi_high']))
    return hyper_results, session_results

def save_results():
    hyper_results, session_results = collect_results()
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({'hyper_results': hyper_results, 'session_results': session_results,
                     'hyper_samples': hyper_samples, 'groups': groups, 'true_hyper': true_hyper,
                     'sweeps': SWEEPS, 'n_reps': N_REPS, 'hdi_prob': HDI_PROB, 'seed': SEED}, f)

#%% GENERATE DATA AND FIT MODEL B ALONG BOTH SWEEPS

run_start = time.time()
for rep in range(N_REPS):
    # sessions sweep: one set of sessions per replicate; smaller datasets use the first n sessions
    true_sess = sample_session_params(max(SESSION_SWEEP), rng)
    stim, grp, resp = simulate_sessions(true_sess, TRIALS_PER_SESSION_FIXED, rng)
    for n_sessions in SESSION_SWEEP:
        run_fit('sessions', rep, n_sessions, true_sess, stim, grp, resp, n_sessions, TRIALS_PER_SESSION_FIXED)

    # trials sweep: fixed sessions; smaller datasets use the first t trials of every session
    true_sess = sample_session_params(N_SESSIONS_FIXED, rng)
    stim, grp, resp = simulate_sessions(true_sess, max(TRIALS_SWEEP), rng)
    for n_trials in TRIALS_SWEEP:
        run_fit('trials', rep, n_trials, true_sess, stim, grp, resp, N_SESSIONS_FIXED, n_trials)

hyper_results, session_results = collect_results()
print(f"FINISHED SAMPLING! ({(time.time() - run_start)/3600:.1f} h)")

#%% (optional) reload saved results instead of refitting

# with open(RESULTS_PATH, "rb") as f:
#     saved = pickle.load(f)
# hyper_results, session_results, hyper_samples = saved['hyper_results'], saved['session_results'], saved['hyper_samples']
# groups, true_hyper, SWEEPS, N_REPS, HDI_PROB = saved['groups'], saved['true_hyper'], saved['sweeps'], saved['n_reps'], saved['hdi_prob']

#%% Sampling diagnostics: r_hat should be ~1 and there should be no divergences

diag = (hyper_results.groupby(['sweep', 'size'])
        .agg(max_r_hat_hyper=('r_hat', 'max'), min_ess_hyper=('ess_bulk', 'min'), divergences=('divergences', 'max'))
        .join(session_results.groupby(['sweep', 'size']).agg(max_r_hat_session=('r_hat', 'max'))))
print(diag.to_string())

#%% Hyperparameter coverage: fraction of fits whose 95% HDI contains the true value

for sweep in SWEEPS:
    cov = hyper_results[hyper_results['sweep'] == sweep].pivot_table(index='param', columns='size',
                                                                      values='covered', aggfunc='mean')
    print(f"\n{sweep} sweep\n" + cov.loc[HYPER_NAMES].round(2).to_string())

#%% Session-level summaries across sessions, for each fit

session_summary = (session_results.groupby(['sweep', 'rep', 'size', 'param', 'group'])
                   .agg(rmse=('error', lambda e: np.sqrt(np.mean(e**2))),
                        mean_hdi_width=('hdi_width', 'mean'),
                        coverage=('covered', 'mean'))
                   .reset_index())

for sweep in SWEEPS:
    cov = session_summary[session_summary['sweep'] == sweep].pivot_table(index='param', columns='size',
                                                                          values='coverage', aggfunc='mean')
    print(f"\n{sweep} sweep: session-level coverage\n" + cov.loc[SESSION_PARAMS].round(2).to_string())

#%% Plot colors and helpers

CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GROUP_COLORS = dict(zip(groups, CATEGORICAL))
SIZE_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

def size_colors(sizes):
    # blue ramp, light -> dark with dataset size
    idx = np.linspace(0, len(SIZE_RAMP) - 1, len(sizes)).round().astype(int)
    return dict(zip(sizes, [SIZE_RAMP[i] for i in idx]))

def size_axis(ax, sizes):
    # log x-axis limited to the fitted sizes, ticked at each size
    ax.set_xscale('log')
    ax.set_xlim(min(sizes) / 1.4, max(sizes) * 1.4)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

def save_plot(fig, name):
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=200, bbox_inches='tight')

rep_offsets = np.exp(np.linspace(-0.08, 0.08, N_REPS)) if N_REPS > 1 else [1.0]   # spread reps on the log x-axis

#%% Hyperparameter posterior mean and 95% HDI vs dataset size, for each sweep

for sweep, info in SWEEPS.items():
    sizes = info['sizes']
    for family, names in HYPER_FAMILIES.items():
        fig, axes = plt.subplots(len(names), len(groups), sharex=True, constrained_layout=True,
                                 figsize=(13, 2.4 * len(names) + 0.8))
        for r, name in enumerate(names):
            for ax, grp in zip(axes[r], groups):
                sub = hyper_results[(hyper_results['sweep'] == sweep) & (hyper_results['param'] == name)
                                    & (hyper_results['group'] == grp)]
                for rep in range(N_REPS):
                    d = sub[sub['rep'] == rep]
                    ax.errorbar(d['size'] * rep_offsets[rep], d['mean'],
                                yerr=[d['mean'] - d['hdi_low'], d['hdi_high'] - d['mean']],
                                fmt='o', markersize=4, linewidth=1.5, capsize=0, color='#2a78d6',
                                label=f'posterior mean ± {int(HDI_PROB*100)}% HDI' if rep == 0 else None)
                ax.axhline(true_hyper[name][groups.index(grp)], linestyle='--', linewidth=1.5, color='#52514e',
                           label='true value')
                size_axis(ax, sizes)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.set_title(grp)
            axes[r, 0].set_ylabel(name + (' (logit)' if 'gam' in name else ''))
        for ax in axes[-1]:
            ax.set_xlabel(info['label'])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2)
        fig.suptitle(f'Recovery of hyperparameter {family} vs {info["label"]}', fontsize=14)
        save_plot(fig, f"hyper_{family}_recovery_vs_{sweep}")
        plt.show()

#%% Hyperparameter posterior densities, replicates overlaid; darker = larger dataset

for sweep, info in SWEEPS.items():
    sizes = info['sizes']
    colors = size_colors(sizes)
    for family, names in HYPER_FAMILIES.items():
        fig, axes = plt.subplots(len(names), len(groups), constrained_layout=True,
                                 figsize=(13, 2.4 * len(names) + 0.8))
        for r, name in enumerate(names):
            for g_i, (ax, grp) in enumerate(zip(axes[r], groups)):
                for size in sizes:
                    for rep in range(N_REPS):
                        vals = hyper_samples[(sweep, rep, size)][name][:, g_i]
                        vals = vals[np.isfinite(vals)]
                        az.plot_kde(vals, ax=ax, plot_kwargs={'color': colors[size], 'linewidth': 1.5, 'alpha': 0.85},
                                    label=f'{size}' if rep == 0 else None)
                ax.axvline(true_hyper[name][g_i], linestyle='--', linewidth=1.5, color='#52514e', label='true value')
                ax.set_yticks([])
                ax.tick_params(labelsize=9)
                if ax.get_legend():
                    ax.get_legend().remove()
                if r == 0:
                    ax.set_title(grp)
            axes[r, 0].set_ylabel(name + (' (logit)' if 'gam' in name else ''))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title=info['label'].split(' (')[0], loc='outside right center')
        fig.suptitle(f'Posterior of hyperparameter {family} as data grows, {sweep} sweep '
                     f'({N_REPS} replicates overlaid)', fontsize=14)
        save_plot(fig, f"hyper_{family}_density_vs_{sweep}")
        plt.show()

#%% Session-level recovery: RMSE, HDI width and coverage across sessions vs dataset size

metrics = [('rmse', 'RMSE of posterior mean'), ('mean_hdi_width', f'mean {int(HDI_PROB*100)}% HDI width'),
           ('coverage', f'{int(HDI_PROB*100)}% HDI coverage')]

for sweep, info in SWEEPS.items():
    sizes = info['sizes']
    summ = (session_summary[session_summary['sweep'] == sweep]
            .groupby(['param', 'group', 'size'])[[m for m, _ in metrics]].mean())
    fig, axes = plt.subplots(len(SESSION_PARAMS), len(metrics), sharex=True, constrained_layout=True,
                             figsize=(12, 2.2 * len(SESSION_PARAMS) + 0.8))
    for r, par in enumerate(SESSION_PARAMS):
        for ax, (metric, title) in zip(axes[r], metrics):
            for grp in groups:
                ax.plot(sizes, summ.loc[(par, grp)].loc[sizes, metric].values, marker='o', markersize=4,
                        linewidth=2, color=GROUP_COLORS[grp], label=grp)
            if metric == 'coverage':
                ax.axhline(HDI_PROB, linestyle='--', linewidth=1.5, color='#52514e', label=f'nominal {HDI_PROB}')
                ax.set_ylim(0, 1.05)
            else:
                ax.set_yscale('log')
            size_axis(ax, sizes)
            ax.grid(alpha=0.3, which='both')
            if r == 0:
                ax.set_title(title)
        axes[r, 0].set_ylabel(par)
    for ax in axes[-1]:
        ax.set_xlabel(info['label'])
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(labels))
    fig.suptitle(f'Session-level parameter recovery across sessions, {sweep} sweep (mean over replicates)',
                 fontsize=14)
    save_plot(fig, f"session_params_error_width_coverage_vs_{sweep}")
    plt.show()
