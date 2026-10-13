#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 2026

Fit Model C (hierarchical over sessions, one group per hand, distractor
amplitude as a covariate) to the real data.

Model C assumes the distractor only shifts the psychometric curve, by
b2 * x_dist on bimanual trials: uni and bi trials of a session share their
slope and lapse rates. So the PSE shift (bimanual - unimanual) of a session is
-b2 * x_dist / b1 standardized units, and the JND and lapse rates are the same
in both conditions. The posterior predictive check by hand and condition
shows how well these assumptions fit the data.

Sections:
    - fit and sampling diagnostics
    - posterior predictive check by hand and condition
    - distractor effect: hand-level mu_b2, the PSE shift of every session
      against its distractor amplitude, and the shift of a typical session
    - comparison with Model B (LOO, and the session PSE shifts)

@author: vmschroe
"""

import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import os
import pickle
import time
from pathlib import Path

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
DATA_DIR = MODEL_DIR / "Data_C"

FIT_PLOTS_DIR = MODEL_DIR / "Plots_C" / "Fit_Plots_C"
SAVE_FIGS = True

def save_fig(fig, name):
    if SAVE_FIGS:
        FIT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIT_PLOTS_DIR / name, dpi=200, bbox_inches='tight')

#%% SETTINGS

DRAWS = 5000   # as for Model B's real-data fit
TUNE = 1000
CHAINS = 4
CORES = 4
# Python 3.14's default 'forkserver' re-runs this script in every worker, so fork them instead
# ('fork' doesn't exist on Windows, which keeps its default)
MP_CTX = None if os.name == 'nt' else 'fork'
SEED = 20261013

#%% LOAD DATA AND BUILD MODEL C

with open(DATA_DIR / "ReadyData_Sirius_C.pkl", "rb") as f:
    data_dict = pickle.load(f)

cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
sess_idx = data_dict['sess_idx']
sess_x_dist = data_dict['sess_x_dist']
obs_data = data_dict['resp']
sessions = list(data_dict['dates_sess_idx'])
x_mu, x_sig = data_dict['x_mu'], data_dict['x_sig']

exec(open(MODEL_DIR / "Build_Model_C.py").read())

groups = coords['groups']                        # hands: ['left', 'right']
condition = (cov_mat[:, 2] > 0).astype(int)      # 0 = unimanual, 1 = bimanual
amp = np.round(cov_mat[:, 1] * x_sig + x_mu).astype(int)   # stimulus amplitude in micrometres
levels = np.unique(amp)
sess_amp = np.round(sess_x_dist * x_sig).astype(int)       # each session's distractor amplitude (um)

#%% Sample from posteriors

sampling_start = time.time()
with model_C:
    trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CORES, mp_ctx=MP_CTX, random_seed=SEED,
                      progressbar=True, idata_kwargs={"log_likelihood": True})
sampling_minutes = (time.time() - sampling_start) / 60
print(f"FINISHED SAMPLING! ({sampling_minutes:.0f} min)")

#%% Sampling diagnostics

HYPER_VARS = ['mu_betas', 'sig_betas', 'mu_gams', 'sig_gams']
SESSION_VARS = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND']
result_df = az.summary(trace, var_names=HYPER_VARS + SESSION_VARS, hdi_prob=0.95)
full_summary = az.summary(trace, kind='diagnostics')
print(f"divergences: {int(trace.sample_stats['diverging'].sum())}, "
      f"max tree depth: {int(trace.sample_stats['tree_depth'].max())}")
print(f"all {len(full_summary)} variables: max r_hat {full_summary['r_hat'].max():.3f}, "
      f"min bulk ESS {full_summary['ess_bulk'].min():.0f}, min tail ESS {full_summary['ess_tail'].min():.0f}")
print("hyperparameters:")
print(az.summary(trace, var_names=HYPER_VARS, hdi_prob=0.95).round(3).to_string())   # 95% HDIs, as in the report
post = trace.posterior

#%% Trace plots: session parameters of the left hand in two sessions, and the hyperparameters of b2 and gam_h

CURVE_SESSION = sessions[10]   # the session whose curves and joint posteriors are shown (as for Model B)
amp_32_session = sessions[int(np.flatnonzero(sess_amp == sess_amp.max())[0])]   # first session with the strongest distractor
print(f"trace / pair / curve session: {CURVE_SESSION} (distractor {sess_amp[sessions.index(CURVE_SESSION)]} um); "
      f"second trace session: {amp_32_session} (distractor {sess_amp[sessions.index(amp_32_session)]} um)")

axes = az.plot_trace(trace, var_names=['gam_h', 'gam_l', 'beta_vec'],
                     coords={'groups': ['left'], 'sessions': [CURVE_SESSION, amp_32_session]},
                     compact=False, backend_kwargs={"constrained_layout": True})
save_fig(axes.ravel()[0].figure, 'trace_session_params_left.png')
plt.show()

axes = az.plot_trace(trace, var_names=['mu_betas', 'sig_betas'], coords={'groups': ['left'], 'betas': ['b2']},
                     compact=False, backend_kwargs={"constrained_layout": True})
save_fig(axes.ravel()[0].figure, 'trace_b2_hyper_left.png')
plt.show()

axes = az.plot_trace(trace, var_names=['gam_h', 'mu_gams', 'sig_gams'],
                     coords={'groups': ['left'], 'lapses': ['gam_h'], 'sessions': [CURVE_SESSION]},
                     compact=False, backend_kwargs={"constrained_layout": True})
save_fig(axes.ravel()[0].figure, 'trace_gam_h_hyper_left.png')
plt.show()

#%% Joint posteriors of one session's parameters, for each hand

for hand in groups:
    axes = az.plot_pair(trace, var_names=['gam_h', 'gam_l', 'beta_vec'],
                        coords={'groups': [hand], 'sessions': [CURVE_SESSION]}, kind='kde', marginals=True)
    save_fig(axes.ravel()[0].figure, f'pair_{hand}_{CURVE_SESSION}.png')
    plt.show()

axes = az.plot_pair(trace, var_names=['gam_h', 'mu_gams', 'sig_gams'],
                    coords={'groups': ['left'], 'lapses': ['gam_h'], 'sessions': [CURVE_SESSION]},
                    kind='kde', marginals=True)
save_fig(axes.ravel()[0].figure, 'pair_gam_h_hyper_left.png')
plt.show()

#%% Fitted psychometric curves of one session, unimanual and bimanual, with the observed proportions

COND_COLORS = {'uni': '#2a78d6', 'bi': '#e34948'}
COND_NAMES = {'uni': 'Unimanual', 'bi': 'Bimanual'}

def psychfunc(gh, gl, b0, b1, b2, x, d):
    """Model C's psychometric function; parameter arrays broadcast against the stimulus grid x."""
    return gh + (1 - gh - gl) / (1 + np.exp(-(b0 + b1 * x + b2 * d)))

s_i = sessions.index(CURVE_SESSION)
xfit = np.linspace(cov_mat[:, 1].min(), cov_mat[:, 1].max(), 300)
for g_i, hand in enumerate(groups):
    par = {p: post[p].sel(groups=hand, sessions=CURVE_SESSION).values.reshape(-1, 1) for p in ['gam_h', 'gam_l']}
    for b in ['b0', 'b1', 'b2']:
        par[b] = post['beta_vec'].sel(groups=hand, sessions=CURVE_SESSION, betas=b).values.reshape(-1, 1)
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6.5, 4.5))
    for c_i, cond in enumerate(['uni', 'bi']):
        d = sess_x_dist[s_i] * c_i
        y = psychfunc(par['gam_h'], par['gam_l'], par['b0'], par['b1'], par['b2'], xfit, d)   # (draws, grid)
        lo, hi = az.hdi(y[None], hdi_prob=0.95).T
        y_mean = psychfunc(*(par[p].mean() for p in ['gam_h', 'gam_l', 'b0', 'b1', 'b2']), xfit, d)
        in_cell = (sess_idx == s_i) & (grp_idx == g_i) & (condition == c_i)
        props = pd.Series(obs_data[in_cell]).groupby(amp[in_cell]).mean()
        ax.plot(xfit * x_sig + x_mu, y_mean, color=COND_COLORS[cond], label=f'{COND_NAMES[cond]}: curve at posterior means')
        ax.fill_between(xfit * x_sig + x_mu, lo, hi, color=COND_COLORS[cond], alpha=0.25, linewidth=0, label='95% HDI')
        ax.plot(props.index, props.values, 'o', color=COND_COLORS[cond], label=f'{COND_NAMES[cond]}: observed')
        print(f"{hand} {cond}, session {CURVE_SESSION}: {int(in_cell.sum())} trials, "
              f"{int(pd.Series(amp[in_cell]).value_counts().min())}-{int(pd.Series(amp[in_cell]).value_counts().max())} per amplitude")
    ax.axhline(0.5, color='#b9b8b3', linewidth=0.5)
    ax.axvline(28, linestyle='--', linewidth=1.5, color='green', label='trained threshold')
    ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
    ax.set_ylabel('P(response = "high")')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'{hand.capitalize()} hand, session {CURVE_SESSION} (distractor {sess_amp[s_i]} $\\mu m$)')
    save_fig(fig, f'curves_{hand}_{CURVE_SESSION}.png')
    plt.show()

#%% Posterior predictive check: observed vs predicted P("high") by hand and condition, sessions pooled

with model_C:
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=SEED)
    trace.extend(pm.sample_prior_predictive(draws=3000, random_seed=SEED))   # for the prior vs posterior plots
y_rep = trace.posterior_predictive['resp'].values.reshape(-1, len(obs_data)).astype(np.int8)   # (draws, trials)

ppc_rows = []
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(9, 6))
for r, (c_i, cond) in enumerate([(0, 'uni'), (1, 'bi')]):
    for ax, (g_i, hand) in zip(axes[r], enumerate(groups)):
        cell = (grp_idx == g_i) & (condition == c_i)
        obs = np.array([obs_data[cell & (amp == a)].mean() for a in levels])
        rep = np.array([y_rep[:, cell & (amp == a)].mean(1) for a in levels])   # (levels, draws)
        lo, med, hi = np.quantile(rep, [0.025, 0.5, 0.975], axis=1)
        for a, o, l_, m_, h_ in zip(levels, obs, lo, med, hi):
            ppc_rows.append({'hand': hand, 'condition': cond, 'amp': a, 'observed': o, 'median': m_,
                             'low': l_, 'high': h_, 'inside': l_ <= o <= h_})
        ax.vlines(levels, lo, hi, color='#2a78d6', linewidth=6, alpha=0.35, label='posterior predictive 95% interval')
        ax.plot(levels, med, '_', color='#2a78d6', markersize=12, markeredgewidth=2, label='posterior predictive median')
        ax.plot(levels, obs, 'o', color='#0b0b0b', markersize=5, label='observed')
        ax.axvline(28, linestyle='--', linewidth=1, color='#52514e')
        ax.set_title(f'{hand}_{cond}')
        ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
for ax in axes[:, 0]:
    ax.set_ylabel('P(response = "high")')
axes[0, 0].legend(fontsize=8, loc='upper left')
fig.suptitle('Model C posterior predictive check: all sessions pooled', fontsize=13)
save_fig(fig, 'ppc_pooled_by_stimulus.png')
plt.show()

ppc_table = pd.DataFrame(ppc_rows)
print(f"{int(ppc_table['inside'].sum())} of {len(ppc_table)} hand x condition x amplitude proportions inside "
      f"their 95% interval")
print("outside:")
print(ppc_table[~ppc_table['inside']].round(3).to_string(index=False))

#%% Posterior predictive check within single sessions

# Each session x hand x condition x amplitude cell (about 10 trials): is the observed count of "high"
# inside its 95% posterior predictive interval, and where does it fall in the predictive distribution
# (randomized PIT, uniform on (0, 1) for a calibrated model)?
pit_rng = np.random.default_rng(SEED)
cell_trials = pd.DataFrame({'s': sess_idx, 'g': grp_idx, 'c': condition, 'a': amp, 'i': np.arange(len(obs_data))}
                           ).groupby(['s', 'g', 'c', 'a'])['i'].apply(np.array)
cell_rows = []
for (s_, g, c, a), idx in cell_trials.items():
    k_obs = obs_data[idx].sum()
    k_rep = y_rep[:, idx].sum(1)
    low, high = np.quantile(k_rep, [0.025, 0.975])
    pit = np.mean(k_rep < k_obs) + pit_rng.uniform() * np.mean(k_rep == k_obs)
    cell_rows.append({'session': s_, 'hand': groups[g], 'condition': ['uni', 'bi'][c], 'amp': a,
                      'n_trials': len(idx), 'k_obs': k_obs, 'inside': low <= k_obs <= high, 'pit': pit})
session_cells = pd.DataFrame(cell_rows)

fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(11, 4))
axes[0].hist(session_cells['pit'], bins=20, range=(0, 1), color='#2a78d6', alpha=0.7, density=True)
axes[0].axhline(1, linestyle='--', color='#52514e', label='calibrated model')
axes[0].set_xlabel('PIT of the observed count')
axes[0].set_ylabel('density')
axes[0].set_title(f'all {len(session_cells)} session × hand × condition × amplitude cells')
axes[0].legend(fontsize='small')
inside_by_amp = session_cells.groupby(['hand', 'condition', 'amp'])['inside'].mean().unstack(['hand', 'condition'])
CELL_STYLES = {('left', 'uni'): ('#2a78d6', '-'), ('left', 'bi'): ('#2a78d6', '--'),
               ('right', 'uni'): ('#eb6834', '-'), ('right', 'bi'): ('#eb6834', '--')}
for (hand, cond), (color, ls) in CELL_STYLES.items():
    axes[1].plot(inside_by_amp.index, inside_by_amp[(hand, cond)], marker='o', color=color, linestyle=ls,
                 label=f'{hand}_{cond}')
axes[1].axhline(0.95, linestyle=':', color='#52514e', label='nominal 0.95')
axes[1].axvline(28, linestyle='--', linewidth=1, color='green')
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel(r'stimulus amplitude ($\mu m$)')
axes[1].set_ylabel('fraction of sessions inside 95% interval')
axes[1].legend(fontsize='small', ncol=2)
axes[1].grid(alpha=0.3)
fig.suptitle('Model C posterior predictive check within single sessions', fontsize=13)
save_fig(fig, 'ppc_session_cells.png')
plt.show()

print(f"single sessions: {int(session_cells['inside'].sum())} of {len(session_cells)} cells inside their 95% interval "
      f"({session_cells['inside'].mean():.1%})")
print(f"PIT < 0.025: {np.mean(session_cells['pit'] < 0.025):.1%}, PIT > 0.975: {np.mean(session_cells['pit'] > 0.975):.1%} "
      f"(2.5% each if calibrated)")
print("fraction of sessions inside, by amplitude, hand and condition:")
print(inside_by_amp.round(2).to_string())

#%% Does the shift scale with amplitude as assumed? Sessions grouped by distractor amplitude

# For the sessions at each of the most used distractor amplitudes, the bimanual - unimanual difference in
# the proportion of "high" responses (over all stimuli) is a model-free measure of the shift. Model C
# predicts it from b2 * d alone; compare the observed difference with its posterior predictive distribution.
AMP_GROUPS = [6, 12, 24, 32]
amp_rows = []
trial_amp = sess_amp[sess_idx]   # distractor amplitude of each trial's session
fig, axes = plt.subplots(len(groups), len(AMP_GROUPS), sharex=True, sharey=True, constrained_layout=True,
                         figsize=(14, 6.5))
for r, (g_i, hand) in enumerate(enumerate(groups)):
    for ax, a_d in zip(axes[r], AMP_GROUPS):
        in_grp = (grp_idx == g_i) & (trial_amp == a_d)
        uni, bi = in_grp & (condition == 0), in_grp & (condition == 1)
        diff_obs = obs_data[bi].mean() - obs_data[uni].mean()
        diff_rep = y_rep[:, bi].mean(1) - y_rep[:, uni].mean(1)
        lo, hi = np.quantile(diff_rep, [0.025, 0.975])
        amp_rows.append({'hand': hand, 'distractor': a_d, 'n_sessions': int((sess_amp == a_d).sum()),
                         'observed': diff_obs, 'predicted': np.median(diff_rep), 'low': lo, 'high': hi,
                         'inside': lo <= diff_obs <= hi, 'p_rep_below_obs': np.mean(diff_rep < diff_obs)})
        for cond_mask, cond in [(uni, 'uni'), (bi, 'bi')]:
            obs_p = np.array([obs_data[cond_mask & (amp == a)].mean() for a in levels])
            rep_p = np.array([y_rep[:, cond_mask & (amp == a)].mean(1) for a in levels])
            lo_p, med_p, hi_p = np.quantile(rep_p, [0.025, 0.5, 0.975], axis=1)
            ax.vlines(levels, lo_p, hi_p, color=COND_COLORS[cond], linewidth=5, alpha=0.3)
            ax.plot(levels, med_p, '_', color=COND_COLORS[cond], markersize=10, markeredgewidth=2)
            ax.plot(levels, obs_p, 'o', color=COND_COLORS[cond], markersize=4, markeredgecolor='#0b0b0b',
                    markeredgewidth=0.5, label=f'{COND_NAMES[cond]}: observed')
        ax.axvline(28, linestyle='--', linewidth=1, color='#52514e')
        ax.set_title(f'{hand}, {a_d} $\\mu m$ distractor ({int((sess_amp == a_d).sum())} sessions)', fontsize=10)
        ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
for ax in axes[:, 0]:
    ax.set_ylabel('P(response = "high")')
axes[0, 0].legend(fontsize=8, loc='upper left')
fig.suptitle('Observed proportions (dots) against the posterior predictive median and 95% interval (bars), '
             'by distractor amplitude', fontsize=12)
save_fig(fig, 'ppc_by_distractor_amplitude.png')
plt.show()

amp_check = pd.DataFrame(amp_rows)
print("bimanual - unimanual difference in P(high), observed vs posterior predictive, by distractor amplitude:")
print(amp_check.round(3).to_string(index=False))

#%% ArviZ posterior predictive check of the responses, all trials pooled

ax = az.plot_ppc(trace, num_pp_samples=100, random_seed=0)
save_fig(np.ravel(ax)[0].figure, 'ppc_arviz_resp.png')
plt.show()

#%% Distractor effect: hand-level mu_b2 and the PSE shift of a typical session at each amplitude

mu_b2 = post['mu_betas'].sel(betas='b2', drop=True)
print("hand-level distractor coefficient mu_b2 (logit units per unit of x_dist):")
for hand in groups:
    v = mu_b2.sel(groups=hand).values.ravel()
    lo, hi = az.hdi(v, hdi_prob=0.95)
    print(f"  {hand}: {v.mean():.2f} [{lo:.2f}, {hi:.2f}], P(>0) = {np.mean(v > 0):.3f}")

# Uni and bi share the lapse rates, so the PSE shift is -b2 * x_dist / b1 (standardized units).
# Typical session: the shift at the hand-level means, at each distractor amplitude used.
amp_grid = np.unique(sess_amp)
mu_b1 = post['mu_betas'].sel(betas='b1', drop=True)
pop_shift = {hand: np.stack([(-mu_b2.sel(groups=hand) * (a / x_sig) / mu_b1.sel(groups=hand)).values.ravel() * x_sig
                             for a in amp_grid], axis=1)   # (draws, amplitudes), micrometres
             for hand in groups}

# session level: PSE bimanual - unimanual in every session (micrometres)
sess_shift = (post['PSE'].sel(conditions='bi', drop=True) - post['PSE'].sel(conditions='uni', drop=True)) * x_sig

jitter = (np.arange(len(sessions)) % 5 - 2) * 0.25   # spreads sessions that share an amplitude
fig, axes = plt.subplots(1, len(groups), sharey=True, constrained_layout=True, figsize=(11, 4.5))
for ax, hand in zip(axes, groups):
    d = sess_shift.sel(groups=hand)
    d_mean = d.mean(('chain', 'draw')).values
    d_hdi = az.hdi(d.to_dataset(name='d'), hdi_prob=0.95)['d'].values   # (sessions, 2)
    ax.errorbar(sess_amp + jitter, d_mean, yerr=[d_mean - d_hdi[:, 0], d_hdi[:, 1] - d_mean], fmt='o',
                markersize=4, color='#2a78d6', elinewidth=1, alpha=0.8, label='session: mean, 95% HDI')
    lo, hi = az.hdi(pop_shift[hand][None], hdi_prob=0.95).T
    ax.fill_between(amp_grid, lo, hi, color='#e34948', alpha=0.2, linewidth=0)
    ax.plot(amp_grid, pop_shift[hand].mean(0), color='#e34948', linewidth=2, label='typical session (95% HDI)')
    ax.axhline(0, linestyle='--', linewidth=1, color='#52514e')
    ax.set_title(hand)
    ax.set_xlabel(r'distractor amplitude ($\mu m$)')
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r'PSE shift, bimanual − unimanual ($\mu m$)')
axes[0].legend(fontsize=8)
fig.suptitle('Model C: PSE shift against distractor amplitude', fontsize=13)
save_fig(fig, 'pse_shift_vs_amplitude.png')
plt.show()

effect_rows = []
for hand in groups:
    for a_i, a in enumerate(amp_grid):
        v = pop_shift[hand][:, a_i]
        lo, hi = az.hdi(v, hdi_prob=0.95)
        effect_rows.append({'hand': hand, 'amp': a, 'pop_mean': v.mean(), 'pop_hdi_low': lo, 'pop_hdi_high': hi})
    avg = sess_shift.sel(groups=hand).mean('sessions').values.ravel()
    lo, hi = az.hdi(avg, hdi_prob=0.95)
    print(f"{hand}: PSE shift averaged over the {len(sessions)} sessions {avg.mean():.2f} [{lo:.2f}, {hi:.2f}] um")
effect_summary = pd.DataFrame(effect_rows)
print("PSE shift of a typical session at each amplitude (um):")
print(effect_summary.round(2).to_string(index=False))

#%% Population level: prior vs posterior of a typical session's unimanual PSE, JND, lapse rates and mu_b2

def population_quantities(ds):
    """Unimanual PSE, JND and lapse rates (micrometres where applicable) of a typical session: the session
    parameters at the hand-level means. Works on trace.posterior or trace.prior; dims (chain, draw, groups)."""
    b0 = ds['mu_betas'].sel(betas='b0', drop=True)
    b1 = ds['mu_betas'].sel(betas='b1', drop=True)
    gh = 0.25 / (1 + np.exp(-ds['mu_gams'].sel(lapses='gam_h', drop=True)))
    gl = 0.25 / (1 + np.exp(-ds['mu_gams'].sel(lapses='gam_l', drop=True)))
    return {'PSE': (-b0 + np.log((1 - 2*gh) / (1 - 2*gl))) / b1 * x_sig + x_mu,
            'JND': np.log(((3 - 4*gh)*(3 - 4*gl)) / ((1 - 4*gh)*(1 - 4*gl))) / (2*b1) * x_sig,
            'gam_h': gh, 'gam_l': gl,
            'mu_b2': ds['mu_betas'].sel(betas='b2', drop=True)}

POP_LABELS = {'PSE': r'unimanual PSE ($\mu m$)', 'JND': r'JND ($\mu m$)', 'gam_h': r'$\gamma_h$',
              'gam_l': r'$\gamma_l$', 'mu_b2': r'$\mu_{\beta_2}$'}
FILE_LABELS = {'PSE': 'PSE', 'JND': 'JND', 'gam_h': 'gamh', 'gam_l': 'gaml', 'mu_b2': 'mub2'}

def finite(vals):
    vals = np.asarray(vals).reshape(-1)
    return vals[np.isfinite(vals)]

def plot_post_hdi(ax, vals, color, label, hdi_prob=0.95):
    """Posterior density with its HDI shaded and a dotted line at the posterior mean."""
    grid, dens = az.kde(vals)
    hdi_low, hdi_high = az.hdi(vals, hdi_prob=hdi_prob)
    in_hdi = (grid >= hdi_low) & (grid <= hdi_high)
    ax.plot(grid, dens, color=color, label=label)
    ax.fill_between(grid[in_hdi], dens[in_hdi], color=color, alpha=0.3, linewidth=0)
    ax.vlines(vals.mean(), 0, np.interp(vals.mean(), grid, dens), color=color, linestyles='dotted')

pop_post = population_quantities(trace.posterior)
pop_prior = population_quantities(trace.prior)
for q in pop_post:
    fig, axes = plt.subplots(1, len(groups), constrained_layout=True, figsize=(9, 3.5))
    for ax, hand in zip(axes, groups):
        post_vals = finite(pop_post[q].sel(groups=hand))
        prior_vals = finite(pop_prior[q].sel(groups=hand))
        # the prior is far wider than the posterior, so show it as a histogram over the posterior's range
        lo, hi = np.quantile(post_vals, [0.001, 0.999])
        bins = np.linspace(lo - (hi - lo), hi + (hi - lo), 60)
        ax.hist(prior_vals, bins=bins, histtype='step', color='#52514e', label='prior',
                weights=np.full(len(prior_vals), 1 / (len(prior_vals) * np.diff(bins)[0])))
        plot_post_hdi(ax, post_vals, '#2a78d6', 'posterior (95% HDI shaded)')
        if q == 'PSE':
            ax.axvline(28, linestyle='--', linewidth=1.5, color='green', label='trained threshold')
        if q == 'mu_b2':
            ax.axvline(0, linestyle='--', linewidth=1, color='#52514e')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_yticks([])
        ax.set_title(hand)
        ax.set_xlabel(POP_LABELS[q])
    axes[0].legend(fontsize='small')
    fig.suptitle(f"Typical session: {POP_LABELS[q]}", fontsize=13)
    save_fig(fig, f'population_{FILE_LABELS[q]}_prior_posterior.png')
    plt.show()

#%% Session level: estimates in every session (PSE unimanual and bimanual; JND and lapse rates, shared)

x = np.arange(len(sessions))
offset = 0.18   # small horizontal shift so the two conditions don't overlap
SESSION_QUANTITIES = {'PSE': (x_sig, x_mu, r'PSE ($\mu m$)'), 'JND': (x_sig, 0.0, r'JND ($\mu m$)'),
                      'gam_h': (1.0, 0.0, r'$\gamma_h$'), 'gam_l': (1.0, 0.0, r'$\gamma_l$')}

def errorbar_sessions(ax, vals, dx, color, label):
    """Posterior mean and 95% HDI of a (chain, draw, sessions) quantity at every session."""
    v_mean = vals.mean(('chain', 'draw')).values
    v_hdi = az.hdi(vals.to_dataset(name='v'), hdi_prob=0.95)['v'].values   # (sessions, 2)
    ax.errorbar(x + dx, v_mean, yerr=[v_mean - v_hdi[:, 0], v_hdi[:, 1] - v_mean], fmt='o', markersize=4,
                color=color, elinewidth=1.2, label=label)

for q, (scale, shift, label) in SESSION_QUANTITIES.items():
    fig, axes = plt.subplots(len(groups), 1, sharex=True, constrained_layout=True, figsize=(14, 7))
    for ax, hand in zip(axes, groups):
        if q == 'PSE':
            for cond, dx in [('uni', -offset), ('bi', offset)]:
                vals = post['PSE'].sel(groups=hand, conditions=cond, drop=True) * scale + shift
                errorbar_sessions(ax, vals, dx, COND_COLORS[cond], f'{COND_NAMES[cond]}: session (mean, 95% HDI)')
            ax.axhline(28, linestyle='--', linewidth=1.5, color='green', label='trained threshold')
        else:
            vals = post[q].sel(groups=hand, drop=True) * scale + shift
            errorbar_sessions(ax, vals, 0.0, '#2a78d6', 'session (mean, 95% HDI), both conditions')
            pop_low, pop_high = az.hdi(finite(pop_post[q].sel(groups=hand)), hdi_prob=0.95)
            ax.axhspan(pop_low, pop_high, color='#2a78d6', alpha=0.12, linewidth=0, label='typical session (95% HDI)')
        ax.set_title(hand)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(x, [f'{s} ({a})' for s, a in zip(sessions, sess_amp)], rotation=90, fontsize=8)
    axes[-1].set_xlabel(r'session (distractor amplitude, $\mu m$)')
    axes[0].legend(fontsize='small', ncol=3)
    fig.suptitle(f"{label} in each session (posterior mean and 95% HDI)", fontsize=14)
    save_fig(fig, f'{FILE_LABELS[q]}_by_session.png')
    plt.show()

#%% Session level: PSE shift (bimanual - unimanual) within every session

fig, axes = plt.subplots(len(groups), 1, sharex=True, constrained_layout=True, figsize=(14, 7))
for ax, hand in zip(axes, groups):
    d = sess_shift.sel(groups=hand)
    d_mean = d.mean(('chain', 'draw')).values
    d_hdi = az.hdi(d.to_dataset(name='d'), hdi_prob=0.95)['d'].values   # (sessions, 2)
    excludes_zero = (d_hdi[:, 0] > 0) | (d_hdi[:, 1] < 0)
    avg = finite(d.mean('sessions'))
    avg_low, avg_high = az.hdi(avg, hdi_prob=0.95)
    ax.axhspan(avg_low, avg_high, color='#eb6834', alpha=0.2, linewidth=0, label='average over sessions (95% HDI)')
    ax.axhline(avg.mean(), color='#eb6834', linewidth=1.5)
    ax.axhline(0, linestyle='--', linewidth=1, color='#52514e')
    ax.errorbar(x, d_mean, yerr=[d_mean - d_hdi[:, 0], d_hdi[:, 1] - d_mean], fmt='none', ecolor='#2a78d6', elinewidth=1.2)
    ax.plot(x[excludes_zero], d_mean[excludes_zero], 'o', color='#2a78d6', markersize=5, label='session: 95% HDI excludes 0')
    ax.plot(x[~excludes_zero], d_mean[~excludes_zero], 'o', markerfacecolor='white', markeredgecolor='#2a78d6',
            markersize=5, label='session: 95% HDI includes 0')
    ax.set_title(f"{hand}: {excludes_zero.sum()} of {len(sessions)} sessions with 95% HDI excluding 0")
    ax.set_ylabel('PSE shift, bimanual − unimanual\n' + r'($\mu m$)')
    ax.grid(alpha=0.3)
axes[-1].set_xticks(x, [f'{s} ({a})' for s, a in zip(sessions, sess_amp)], rotation=90, fontsize=8)
axes[-1].set_xlabel(r'session (distractor amplitude, $\mu m$)')
axes[0].legend(fontsize='small', ncol=3)
fig.suptitle("PSE shift within each session (posterior mean and 95% HDI)", fontsize=14)
save_fig(fig, 'effect_PSE_by_session.png')
plt.show()

#%% Compare with Model B: LOO and the session PSE shifts

LOO_results = az.loo(trace, pointwise=True)
print(f"Model C: elpd_loo {LOO_results.elpd_loo:.1f} (SE {LOO_results.se:.1f}), p_loo {LOO_results.p_loo:.1f}, "
      f"max Pareto k {np.max(LOO_results.pareto_k.values):.2f}, k > 0.7: {int(np.sum(LOO_results.pareto_k.values > 0.7))}")

with open(MODEL_DIR.parent / "Model_B" / "Results_B.pkl", "rb") as f:
    fit_results_B = pickle.load(f)   # written by Model_B/Main_Analysis_B.py
with open(MODEL_DIR.parent / "Model_B" / "Data_B" / "ReadyData_Sirius_B.pkl", "rb") as f:
    data_B = pickle.load(f)
assert np.array_equal(data_B['resp'], obs_data)   # same trials in the same order
loo_B = fit_results_B['az_loo_trace']
elpd_diff_i = LOO_results.loo_i.values - loo_B.loo_i.values
elpd_diff = elpd_diff_i.sum()
elpd_diff_se = np.sqrt(len(elpd_diff_i) * np.var(elpd_diff_i))
print(f"Model B: elpd_loo {loo_B.elpd_loo:.1f} (SE {loo_B.se:.1f}), p_loo {loo_B.p_loo:.1f}")
print(f"Model C - Model B: {elpd_diff:.1f} (SE {elpd_diff_se:.1f}), {elpd_diff / elpd_diff_se:.1f} SE")

# where the models differ: elpd difference summed by hand, condition and stimulus amplitude
diff_by_cell = (pd.DataFrame({'hand': np.array(groups)[grp_idx], 'condition': np.where(condition, 'bi', 'uni'),
                              'amp': amp, 'diff': elpd_diff_i})
                .pivot_table(index='amp', columns=['hand', 'condition'], values='diff', aggfunc='sum'))
print("elpd_loo(C) - elpd_loo(B) summed by amplitude, hand and condition:")
print(diff_by_cell.round(1).to_string())

# session PSE shifts: Model C against Model B (posterior means, micrometres)
summary_B = fit_results_B['az_summary_trace']['mean']
fig, axes = plt.subplots(1, len(groups), constrained_layout=True, figsize=(10, 4.5))
for ax, hand in zip(axes, groups):
    shift_B = np.array([(summary_B[f'PSE[{hand}_bi, {s}]'] - summary_B[f'PSE[{hand}_uni, {s}]']) * x_sig for s in sessions])
    shift_C = sess_shift.sel(groups=hand).mean(('chain', 'draw')).values
    sc = ax.scatter(shift_B, shift_C, c=sess_amp, cmap='Blues', vmin=0, vmax=sess_amp.max(), s=30,
                    edgecolors='#52514e', linewidths=0.5)
    lims = [min(shift_B.min(), shift_C.min()), max(shift_B.max(), shift_C.max())]
    ax.plot(lims, lims, linestyle='--', linewidth=1.5, color='#52514e')
    ax.set_title(f'{hand}: r = {np.corrcoef(shift_B, shift_C)[0, 1]:.2f}')
    ax.set_xlabel(r'Model B PSE shift ($\mu m$)')
    ax.grid(alpha=0.3)
    print(f"{hand}: session PSE shift, mean over sessions: B {shift_B.mean():.2f}, C {shift_C.mean():.2f} um; "
          f"sd across sessions: B {shift_B.std():.2f}, C {shift_C.std():.2f}")
axes[0].set_ylabel(r'Model C PSE shift ($\mu m$)')
fig.colorbar(sc, ax=axes, label=r'distractor amplitude ($\mu m$)')
fig.suptitle('Session PSE shifts (bimanual − unimanual), posterior means', fontsize=13)
save_fig(fig, 'pse_shift_C_vs_B.png')
plt.show()

#%% Numbers for Fit_C_Report.tex
# Everything quoted in the report that is not printed by an earlier cell.

def mean_hdi(vals, fmt='.2f'):
    vals = np.asarray(vals).ravel()
    lo, hi = az.hdi(vals, hdi_prob=0.95)
    return f"{vals.mean():{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"

print("--- Data and fit")
print(f"{len(obs_data)} trials, {len(sessions)} sessions; trials per hand x condition: "
      + ', '.join(f"{h}_{c}: {int(((grp_idx == g_i) & (condition == c_i)).sum())}"
                  for g_i, h in enumerate(groups) for c_i, c in enumerate(['uni', 'bi'])))
print(f"{CHAINS} chains x {DRAWS} draws after {TUNE} tuning steps, {sampling_minutes:.0f} min")
print(f"divergences {int(trace.sample_stats['diverging'].sum())} of {CHAINS * DRAWS} draws; "
      f"max r_hat {full_summary['r_hat'].max():.3f}, min bulk ESS {full_summary['ess_bulk'].min():.0f}")

print("--- Typical session (hand-level means), posterior mean [95% HDI]")
for hand in groups:
    b0, b1 = (post['mu_betas'].sel(betas=b, groups=hand).values for b in ['b0', 'b1'])
    gh = 0.25 / (1 + np.exp(-post['mu_gams'].sel(lapses='gam_h', groups=hand).values))
    gl = 0.25 / (1 + np.exp(-post['mu_gams'].sel(lapses='gam_l', groups=hand).values))
    pse_uni = (-b0 + np.log((1 - 2*gh) / (1 - 2*gl))) / b1 * x_sig + x_mu
    jnd = np.log(((3 - 4*gh)*(3 - 4*gl)) / ((1 - 4*gh)*(1 - 4*gl))) / (2*b1) * x_sig
    print(f"{hand}: unimanual PSE {mean_hdi(pse_uni)} um, JND {mean_hdi(jnd)} um, "
          f"gam_h {mean_hdi(gh, '.4f')}, gam_l {mean_hdi(gl, '.4f')}")
    v = post['mu_betas'].sel(betas='b2', groups=hand).values
    print(f"  mu_b2 posterior sd {v.std():.3f} (prior sd 1)")

print("--- Session level")
for hand in groups:
    d = sess_shift.sel(groups=hand)
    d_hdi = az.hdi(d.to_dataset(name='d'), hdi_prob=0.95)['d'].values
    print(f"{hand}: PSE shift HDI below 0 in {int((d_hdi[:, 1] < 0).sum())} of {len(sessions)} sessions, "
          f"posterior mean below 0 in {int((d.mean(('chain', 'draw')).values < 0).sum())}")
    jnd_sess = (post['JND'].sel(groups=hand) * x_sig).mean(('chain', 'draw')).values
    print(f"  session JNDs (posterior means): {jnd_sess.min():.2f}-{jnd_sess.max():.2f} um, mean {jnd_sess.mean():.2f}")

print("--- Typical session, posterior sd / prior sd")
for q in pop_post:
    print(f"{q}: " + ', '.join(f"{hand} {finite(pop_post[q].sel(groups=hand)).std() / finite(pop_prior[q].sel(groups=hand)).std():.3f}"
                               for hand in groups))

print("--- Session estimates across the 42 sessions (posterior means)")
for hand in groups:
    for cond in ['uni', 'bi']:
        v = (post['PSE'].sel(groups=hand, conditions=cond) * x_sig + x_mu).mean(('chain', 'draw')).values
        print(f"{hand} PSE {cond}: {v.min():.1f}-{v.max():.1f} um")
    for q in ['gam_h', 'gam_l']:
        v = post[q].sel(groups=hand).mean(('chain', 'draw')).values
        print(f"{hand} {q}: {v.min():.4f}-{v.max():.4f}")

print(f"--- Joint posteriors, session {CURVE_SESSION}: correlations of the session parameters")
for hand in groups:
    draws = {p: post[p].sel(groups=hand, sessions=CURVE_SESSION).values.ravel() for p in ['gam_h', 'gam_l']}
    for b in ['b0', 'b1', 'b2']:
        draws[b] = post['beta_vec'].sel(groups=hand, sessions=CURVE_SESSION, betas=b).values.ravel()
    print(f"{hand}:\n" + pd.DataFrame(draws).corr().round(2).to_string())
    print(f"  session PSE: uni {mean_hdi(post['PSE'].sel(groups=hand, conditions='uni', sessions=CURVE_SESSION) * x_sig + x_mu)}, "
          f"bi {mean_hdi(post['PSE'].sel(groups=hand, conditions='bi', sessions=CURVE_SESSION) * x_sig + x_mu)} um")
gh = post['gam_h'].sel(groups='left', sessions=CURVE_SESSION).values.ravel()
for name, v in [('mu_gam_h', post['mu_gams'].sel(groups='left', lapses='gam_h')),
                ('sig_gam_h', post['sig_gams'].sel(groups='left', lapses='gam_h'))]:
    print(f"left gam_h on {CURVE_SESSION} vs {name}: r = {np.corrcoef(gh, v.values.ravel())[0, 1]:.2f}")

print("--- Model B against Model C: session PSE shifts at the smallest and largest distractor (posterior means, um)")
for hand in groups:
    shift_B = np.array([(summary_B[f'PSE[{hand}_bi, {s}]'] - summary_B[f'PSE[{hand}_uni, {s}]']) * x_sig for s in sessions])
    shift_C = sess_shift.sel(groups=hand).mean(('chain', 'draw')).values
    for a in [amp_grid.min(), amp_grid.max()]:
        m = sess_amp == a
        print(f"{hand}, {a} um ({m.sum()} sessions): B mean {shift_B[m].mean():.2f} (range {shift_B[m].min():.2f} to "
              f"{shift_B[m].max():.2f}), C mean {shift_C[m].mean():.2f} (range {shift_C[m].min():.2f} to {shift_C[m].max():.2f})")

print("--- LOO by condition and amplitude range")
cells = pd.DataFrame({'condition': np.where(condition, 'bi', 'uni'), 'low': amp <= 12, 'diff': elpd_diff_i})
print(cells.groupby(['condition', 'low'])['diff'].sum().round(1).to_string())
print(f"total over 6 and 12 um: {cells.loc[cells['low'], 'diff'].sum():.1f} of {elpd_diff:.1f}")

#%% Save results

fit_results = {'az_summary_trace': result_df,
               'az_loo_trace': LOO_results,
               'effect_summary': effect_summary,
               'ppc_table': ppc_table,
               'session_cells': session_cells,
               'amp_check': amp_check}
with open(MODEL_DIR / "Results_C.pkl", "wb") as f:
    pickle.dump(fit_results, f)
