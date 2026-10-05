#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:41:58 2026

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
import ast
import xarray as xr
from pathlib import Path

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
DATA_DIR = MODEL_DIR / "Data_B"

# Figures used in Fit_B_Report.txt are saved here under the names the report uses
FIT_PLOTS_DIR = MODEL_DIR / "Plots_B" / "Fit_Plots_B"
SAVE_FIGS = True

def save_fig(fig, name):
    if SAVE_FIGS:
        FIT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIT_PLOTS_DIR / name, dpi=200, bbox_inches='tight')

#%% LOAD DATA (delete from this script later)
#load and unpack data
with open(DATA_DIR / "ReadyData_Sirius_B.pkl", "rb") as f:
    data_dict = pickle.load(f)
#%% 
sessions = list(data_dict['dates_sess_idx'])
cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
sess_idx = data_dict['sess_idx']

sess_summary = data_dict['session_summary']
#%%%

exec(open(MODEL_DIR / "Build_Model_B.py").read())


#%% Sample from posteriors

sampling_start = time.time()
with model_B:
    trace = pm.sample(draws = 5000, return_inferencedata=True, chains = 4, cores = 1, progressbar=True, idata_kwargs={"log_likelihood": True})   
sampling_minutes = (time.time() - sampling_start) / 60
print(f"FINISHED SAMPLING! ({sampling_minutes:.0f} min)")


#%% Look at r_hats and effective sample sizes

result_df = az.summary(trace, var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND', 'mu_betas', 'sig_betas', 'mu_gams', 'sig_gams'])

#result_df[ result_df['r_hat']>1 ]
    # r_hat = 1 and ess is large, so sampling was successful
#%% Look at traceplots

axes = az.plot_trace(trace, var_names=('gam_h', 'gam_l', 'beta_vec'), coords = {
    'groups': ['left_bi'],
    'betas': ["b0", "b1"], 
    'sessions':[sessions[5],sessions[10], '05-30']}, compact=False,  backend_kwargs={"constrained_layout": True})
save_fig(axes.ravel()[0].figure, 'trace_session_params_left_bi.png')
plt.show()


#%% plot joint posteriors

sess_choice = '06-14'

for grp_num, grp_choice in enumerate(coords['groups']):
     axes = az.plot_pair(trace, var_names=['gam_h', 'gam_l'
                                    ,'beta_vec'
                                    #,'PSE', 'JND'
                                    ], 
             coords = {'betas': ["b0", "b1"], 'groups': [grp_choice], 'sessions': [sess_choice]}, 
             kind = 'kde', marginals=True)
     save_fig(axes.ravel()[0].figure, f'pair_{grp_choice}_{sess_choice}.png')
     plt.show()
     

#%%

axes = az.plot_trace(trace, var_names=('gam_h', 'mu_gams', 'sig_gams'), coords = {
    'groups': ['left_bi'],
    'betas': ["b0"], 
    'sessions':['06-14'],}, compact=False,  backend_kwargs={"constrained_layout": True})
save_fig(axes.ravel()[0].figure, 'trace_gam_h_hyper_left_bi.png')
plt.show()


#%% weird joint posterior at 05-30, could it be from halfnormal in prior? z_sig_gams, sig_gams, z_gams trace?

axes = az.plot_pair(trace, var_names=['gam_h', 'mu_gams', 'sig_gams'], 
         coords = {
             'groups': ['left_bi'],
             'betas': ["b0"], 
             'sessions':['06-14'],},  kind = 'kde', marginals=True)
save_fig(axes.ravel()[0].figure, 'pair_gam_h_hyper_left_bi.png')
plt.show()

#%% Plot curves per session and group ??????????
sess = 10
sess_label = sessions[sess]

param_samps = trace.posterior[['beta_vec', 'gam_h', 'gam_l']]


gam_h_samps = {}
gam_l_samps = {}
beta_0_samps = {}
beta_1_samps = {}



for grp in ["left_bi","left_uni","right_bi","right_uni"]:
    gam_h_samps[grp] = param_samps['gam_h'].sel(groups = grp, sessions=sess_label).values.flatten()
    gam_l_samps[grp] = param_samps['gam_l'].sel(groups = grp, sessions=sess_label).values.flatten()
    beta_0_samps[grp] = param_samps['beta_vec'].sel(groups = grp, sessions=sess_label, betas='b0').values.flatten()
    beta_1_samps[grp] = param_samps['beta_vec'].sel(groups = grp, sessions=sess_label, betas='b1').values.flatten()


def psychfunc(params, X):
    """
    Psychometric function with lapses

    Parameters:
    params : [gamma, lambda_, beta0, beta1]
    X : Stimulus amplitude level

    Returns:
    probability of guess "high"

    """
    X = np.asarray(X)
    gam_h, gam_l, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gam_h + (1 - gam_h - gam_l) * logistic

#frquencies for each level
freq_df = pd.DataFrame({'stim': cov_mat[sess_idx==sess][:,1], 'grp_idx': grp_idx[sess_idx==sess], 'obs_data': obs_data[sess_idx==sess]})

freqs = pd.pivot_table(
    freq_df, 
    values='obs_data',
    index='stim',
    columns='grp_idx',
    aggfunc='mean'
)


xfit = np.linspace(-1.6,1.6,500)
y_samples = {}
hdis = {}
rec_params = {}
yrec = {}

for grp_i, grp in enumerate(['left_uni','left_bi','right_uni','right_bi']):
    y_samples[grp] = np.array([psychfunc([gam_h,gam_l,beta_0,beta_1], xfit) 
                          for gam_h,gam_l,beta_0,beta_1 in zip(
                              gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp])])
    hdis[grp] = az.hdi(y_samples[grp], hdi_prob=0.95)
    rec_params[grp] = np.mean(np.array([gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp]]), axis = 1)
    
    
    yrec[grp] = psychfunc(rec_params[grp], xfit)
 



x_old = [6,12,18,24,32,38,44,50]
x_mu = np.mean(x_old)
x_sig = np.std(x_old)


    
plt.figure()
plt.plot(xfit*x_sig+x_mu,yrec['left_uni'],label='Unimanual',color='blue')
plt.fill_between(xfit*x_sig+x_mu, hdis['left_uni'][:, 0], hdis['left_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[0]),label='Data', color = 'blue')
plt.plot(xfit*x_sig+x_mu,yrec['left_bi'],label='Bimanual',color='red')
plt.fill_between(xfit*x_sig+x_mu, hdis['left_bi'][:, 0], hdis['left_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[1]),label='Data', color = 'red')
plt.hlines(y= 0.5, xmin=0, xmax=51, colors='gray',lw=0.5)
plt.vlines(x=28, ymin=-0.05,ymax=1.05, label="trained threshold", color='green', linestyles='dashed')
plt.xlim(5,51)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'Stimulus Amplitude ($\mu m$)')
plt.ylabel('Prob[response = "high"]')
plt.legend(loc='upper left', fontsize=9.5)
plt.title("Left Hand Psychometric Curves, Session " + sess_label)
save_fig(plt.gcf(), f'curves_left_{sess_label}.png')
plt.show()     




plt.figure()
plt.plot(xfit*x_sig+x_mu,yrec['right_uni'],label='Unimanual',color='blue')
plt.fill_between(xfit*x_sig+x_mu, hdis['right_uni'][:, 0], hdis['right_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[2]),label='Data', color = 'blue')
plt.plot(xfit*x_sig+x_mu,yrec['right_bi'],label='Bimanual',color='red')
plt.fill_between(xfit*x_sig+x_mu, hdis['right_bi'][:, 0], hdis['right_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[3]),label='Data', color = 'red')
plt.hlines(y= 0.5, xmin=0, xmax=51, colors='gray',lw=0.5)
plt.vlines(x=28, ymin=-0.05,ymax=1.05, label="trained threshold", color='green', linestyles='dashed')
plt.xlim(5,51)
plt.ylim(-0.05, 1.05)
plt.xlabel(r'Stimulus Amplitude ($\mu m$)')
plt.ylabel('Prob[response = "high"]')
plt.legend(loc='upper left', fontsize=9.5)
plt.title("Right Hand Psychometric Curves, Session " + sess_label)
save_fig(plt.gcf(), f'curves_right_{sess_label}.png')
plt.show()     
     

#%%#%% sample priors and post pred


with model_B:
    pm.sample_posterior_predictive(trace,extend_inferencedata=True)
    prior = pm.sample_prior_predictive(samples=3000)
trace.extend(prior)

#%% Posterior predictive check: observed vs predicted P("high"), all sessions pooled

amp = np.round(cov_mat[:, 1] * x_sig + x_mu).astype(int)   # stimulus amplitude in micrometres
levels = np.unique(amp)
y_rep = trace.posterior_predictive['resp'].values.reshape(-1, len(obs_data)).astype(np.int8)   # (draws, trials)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(9, 6))
for ax, g in zip(axes.ravel(), ['left_uni', 'left_bi', 'right_uni', 'right_bi']):
    in_grp = grp_idx == coords['groups'].index(g)
    obs = np.array([obs_data[in_grp & (amp == a)].mean() for a in levels])
    rep = np.array([y_rep[:, in_grp & (amp == a)].mean(1) for a in levels])   # (levels, draws)
    lo, med, hi = np.quantile(rep, [0.025, 0.5, 0.975], axis=1)
    ax.vlines(levels, lo, hi, color='#2a78d6', linewidth=6, alpha=0.35, label='posterior predictive 95% interval')
    ax.plot(levels, med, '_', color='#2a78d6', markersize=12, markeredgewidth=2, label='posterior predictive median')
    ax.plot(levels, obs, 'o', color='#0b0b0b', markersize=5, label='observed')
    ax.axvline(28, linestyle='--', linewidth=1, color='#52514e')
    ax.set_title(g)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
for ax in axes[:, 0]:
    ax.set_ylabel('P(response = "high")')
axes[0, 0].legend(fontsize=8, loc='upper left')
fig.suptitle('Posterior predictive check: all sessions pooled', fontsize=13)
save_fig(fig, 'ppc_pooled_by_stimulus.png')
plt.show()

#%% Effect of the distractor (bimanual - unimanual): shared setup

# The goal for Model B is the effect of the distractor across sessions, looked at two ways:
#   session level:    bimanual - unimanual within each session, so anything that shifts both
#                     conditions on the same day cancels
#   population level: the effect for a typical session, from the hyperparameters
# Differences are taken per posterior draw, so their uncertainty is propagated exactly.
# Note: pooling all sessions' posteriors into one distribution per group (as this script used
# to) mixes session-to-session variation into the uncertainty and can hide a consistent effect.

groups = coords['groups']
HANDS = {'Left Hand': ('left_uni', 'left_bi'), 'Right Hand': ('right_uni', 'right_bi')}
QUANTITIES = ['PSE', 'JND', 'gam_h', 'gam_l']
QUANTITY_LABELS = {'PSE': r'PSE ($\mu m$)', 'JND': r'JND ($\mu m$)', 'gam_h': r'$\gamma_h$', 'gam_l': r'$\gamma_l$'}
# normalized stimulus units -> micrometres (x_sig, x_mu from the psychometric curve cell);
# differences are only scaled, never shifted
SCALE = {'PSE': x_sig, 'JND': x_sig, 'gam_h': 1.0, 'gam_l': 1.0}
SHIFT = {'PSE': x_mu, 'JND': 0.0, 'gam_h': 0.0, 'gam_l': 0.0}
COLORS = {'population': '#2a78d6', 'sessions': '#eb6834', 'prior': '#52514e'}
FILE_LABELS = {'PSE': 'PSE', 'JND': 'JND', 'gam_h': 'gamh', 'gam_l': 'gaml'}   # quantity names in figure files

def plot_post_hdi(ax, vals, color, label, hdi_prob=0.95):
    """Posterior density with its HDI shaded and a dotted line at the posterior mean."""
    grid, dens = az.kde(vals)
    hdi_low, hdi_high = az.hdi(vals, hdi_prob=hdi_prob)
    in_hdi = (grid >= hdi_low) & (grid <= hdi_high)
    ax.plot(grid, dens, color=color, label=label)
    ax.fill_between(grid[in_hdi], dens[in_hdi], color=color, alpha=0.3, linewidth=0)
    ax.vlines(vals.mean(), 0, np.interp(vals.mean(), grid, dens), color=color, linestyles='dotted')

def population_quantities(ds):
    """PSE, JND and lapse rates of a typical session: the psychometric parameters at the
    population centres (mu). Works on trace.posterior or trace.prior; dims (chain, draw, groups)."""
    b0 = ds['mu_betas'].sel(betas='b0', drop=True)
    b1 = ds['mu_betas'].sel(betas='b1', drop=True)
    # Build_Model_B.py indexes the gamma hyperparameters with the 'betas' dim: b0 -> gam_h, b1 -> gam_l
    gh = 0.25 / (1 + np.exp(-ds['mu_gams'].sel(betas='b0', drop=True)))
    gl = 0.25 / (1 + np.exp(-ds['mu_gams'].sel(betas='b1', drop=True)))
    pop = {'PSE': (-b0 + np.log((1 - 2*gh) / (1 - 2*gl))) / b1,
           'JND': np.log(((3 - 4*gh)*(3 - 4*gl)) / ((1 - 4*gh)*(1 - 4*gl))) / (2*b1),
           'gam_h': gh, 'gam_l': gl}
    return {q: v * SCALE[q] + SHIFT[q] for q, v in pop.items()}

def finite(vals):
    vals = np.asarray(vals).reshape(-1)
    return vals[np.isfinite(vals)]

pop_post = population_quantities(trace.posterior)
pop_prior = population_quantities(trace.prior)

# population-level difference, dims (chain, draw)
pop_diff = {(hand, q): pop_post[q].sel(groups=bi, drop=True) - pop_post[q].sel(groups=uni, drop=True)
            for hand, (uni, bi) in HANDS.items() for q in QUANTITIES}
# session-level difference within each session, dims (chain, draw, sessions)
sess_diff = {(hand, q): (trace.posterior[q].sel(groups=bi, drop=True)
                         - trace.posterior[q].sel(groups=uni, drop=True)) * SCALE[q]
             for hand, (uni, bi) in HANDS.items() for q in QUANTITIES}

#%% Population level: prior vs posterior of the typical-session PSE, JND and lapse rates

for q in QUANTITIES:
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(9, 6))
    for ax, g in zip(axes.ravel(), groups):
        post_vals = finite(pop_post[q].sel(groups=g))
        prior_vals = finite(pop_prior[q].sel(groups=g))
        # the prior is far wider than the posterior, so show it as a histogram over the posterior's range
        lo, hi = np.quantile(post_vals, [0.001, 0.999])
        bins = np.linspace(lo - (hi - lo), hi + (hi - lo), 60)
        ax.hist(prior_vals, bins=bins, histtype='step', color=COLORS['prior'], label='prior',
                weights=np.full(len(prior_vals), 1 / (len(prior_vals) * np.diff(bins)[0])))
        plot_post_hdi(ax, post_vals, COLORS['population'], 'posterior (95% HDI shaded)')
        if q == 'PSE':
            ax.axvline(28, linestyle="--", linewidth=1.5, label="trained threshold", color='green')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_yticks([])
        ax.set_title(g)
        ax.set_xlabel(QUANTITY_LABELS[q])
    axes[0, 0].legend(fontsize='small')
    fig.suptitle(f"Population level (typical session): {QUANTITY_LABELS[q]}", fontsize=14)
    save_fig(fig, f'population_{FILE_LABELS[q]}_prior_posterior.png')
    plt.show()

#%% Session level: bimanual - unimanual within every session (main result for Model B)

dates = list(trace.posterior['sessions'].values)
x = np.arange(len(dates))

for q in QUANTITIES:
    fig, axes = plt.subplots(len(HANDS), 1, sharex=True, constrained_layout=True, figsize=(14, 7))
    for ax, hand in zip(axes, HANDS):
        d = sess_diff[(hand, q)]
        d_mean = d.mean(('chain', 'draw')).values
        d_hdi = az.hdi(d.to_dataset(name='d'), hdi_prob=0.95)['d'].values   # (sessions, 2)
        excludes_zero = (d_hdi[:, 0] > 0) | (d_hdi[:, 1] < 0)
        avg = finite(d.mean('sessions'))
        avg_low, avg_high = az.hdi(avg, hdi_prob=0.95)
        ax.axhspan(avg_low, avg_high, color=COLORS['sessions'], alpha=0.2, linewidth=0,
                   label='average over sessions (95% HDI)')
        ax.axhline(avg.mean(), color=COLORS['sessions'], linewidth=1.5)
        ax.axhline(0, linestyle='--', linewidth=1, color=COLORS['prior'])
        ax.errorbar(x, d_mean, yerr=[d_mean - d_hdi[:, 0], d_hdi[:, 1] - d_mean], fmt='none',
                    ecolor=COLORS['population'], elinewidth=1.2)
        ax.plot(x[excludes_zero], d_mean[excludes_zero], 'o', color=COLORS['population'], markersize=5,
                label='session: 95% HDI excludes 0')
        ax.plot(x[~excludes_zero], d_mean[~excludes_zero], 'o', markerfacecolor='white',
                markeredgecolor=COLORS['population'], markersize=5, label='session: 95% HDI includes 0')
        ax.set_title(f"{hand}: {excludes_zero.sum()} of {len(dates)} sessions with 95% HDI excluding 0")
        ax.set_ylabel(f'bimanual − unimanual\n{QUANTITY_LABELS[q]}')
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(x, dates, rotation=90, fontsize=8)
    axes[-1].set_xlabel('session')
    axes[0].legend(fontsize='small', ncol=3)
    fig.suptitle(f"{QUANTITY_LABELS[q]}: bimanual − unimanual within each session (posterior mean and 95% HDI)",
                 fontsize=14)
    save_fig(fig, f'effect_{FILE_LABELS[q]}_by_session.png')
    plt.show()

#%% Session level: unimanual (blue) and bimanual (red) estimates in every session

COND_COLORS = {'Unimanual': '#2a78d6', 'Bimanual': '#e34948'}
offset = 0.18   # small horizontal shift so the two conditions don't overlap

for q in QUANTITIES:
    fig, axes = plt.subplots(len(HANDS), 1, sharex=True, constrained_layout=True, figsize=(14, 7))
    for ax, (hand, (uni, bi)) in zip(axes, HANDS.items()):
        for cond, grp, dx in [('Unimanual', uni, -offset), ('Bimanual', bi, offset)]:
            vals = trace.posterior[q].sel(groups=grp, drop=True) * SCALE[q] + SHIFT[q]
            v_mean = vals.mean(('chain', 'draw')).values
            v_hdi = az.hdi(vals.to_dataset(name='v'), hdi_prob=0.95)['v'].values   # (sessions, 2)
            pop_low, pop_high = az.hdi(finite(pop_post[q].sel(groups=grp)), hdi_prob=0.95)
            ax.axhspan(pop_low, pop_high, color=COND_COLORS[cond], alpha=0.12, linewidth=0,
                       label=f'{cond}: population, typical session (95% HDI)')
            ax.errorbar(x + dx, v_mean, yerr=[v_mean - v_hdi[:, 0], v_hdi[:, 1] - v_mean], fmt='o', markersize=4,
                        color=COND_COLORS[cond], elinewidth=1.2, label=f'{cond}: session (mean, 95% HDI)')
        if q == 'PSE':
            ax.axhline(28, linestyle='--', linewidth=1.5, color='green', label='trained threshold')
        ax.set_title(hand)
        ax.set_ylabel(QUANTITY_LABELS[q])
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(x, dates, rotation=90, fontsize=8)
    axes[-1].set_xlabel('session')
    axes[0].legend(fontsize='small', ncol=3)
    fig.suptitle(f"{QUANTITY_LABELS[q]}: unimanual and bimanual in each session (posterior mean and 95% HDI)",
                 fontsize=14)
    save_fig(fig, f'{FILE_LABELS[q]}_uni_bi_by_session.png')
    plt.show()

#%% Overall distractor effect: population level vs average over these sessions

fig, axes = plt.subplots(len(QUANTITIES), len(HANDS), constrained_layout=True, figsize=(12, 12))
for r, q in enumerate(QUANTITIES):
    for ax, hand in zip(axes[r], HANDS):
        pop_d = finite(pop_diff[(hand, q)])
        avg_d = finite(sess_diff[(hand, q)].mean('sessions'))
        plot_post_hdi(ax, pop_d, COLORS['population'], f'population (typical session), P(>0) = {np.mean(pop_d > 0):.2f}')
        plot_post_hdi(ax, avg_d, COLORS['sessions'], f'average over the {len(dates)} sessions, P(>0) = {np.mean(avg_d > 0):.2f}')
        ax.axvline(0, linestyle='--', linewidth=1, color=COLORS['prior'])
        ax.set_yticks([])
        ax.set_xlabel(f'bimanual − unimanual {QUANTITY_LABELS[q]}')
        ax.legend(fontsize='x-small', title='95% HDI shaded, mean dotted', title_fontsize='x-small')
        if r == 0:
            ax.set_title(hand)
fig.suptitle("Effect of the distractor (bimanual − unimanual)", fontsize=14)
save_fig(fig, 'effect_densities.png')
plt.show()

#%% Effect summary table

effect_rows = []
for hand in HANDS:
    for q in QUANTITIES:
        pop_d = finite(pop_diff[(hand, q)])
        avg_d = finite(sess_diff[(hand, q)].mean('sessions'))
        d_hdi = az.hdi(sess_diff[(hand, q)].to_dataset(name='d'), hdi_prob=0.95)['d'].values
        effect_rows.append({
            'hand': hand, 'quantity': q,
            'pop_mean': pop_d.mean(), 'pop_hdi_low': az.hdi(pop_d, hdi_prob=0.95)[0],
            'pop_hdi_high': az.hdi(pop_d, hdi_prob=0.95)[1], 'pop_P(>0)': np.mean(pop_d > 0),
            'sess_avg_mean': avg_d.mean(), 'sess_avg_hdi_low': az.hdi(avg_d, hdi_prob=0.95)[0],
            'sess_avg_hdi_high': az.hdi(avg_d, hdi_prob=0.95)[1], 'sess_avg_P(>0)': np.mean(avg_d > 0),
            'n_sessions_hdi_above_0': int((d_hdi[:, 0] > 0).sum()),
            'n_sessions_hdi_below_0': int((d_hdi[:, 1] < 0).sum()),
        })
effect_summary = pd.DataFrame(effect_rows)
print(effect_summary.round(3).to_string())

#%% Distractor effect against distractor amplitude (post hoc: amplitude is not in the model)

# Session-level bimanual - unimanual differences, regressed on distractor amplitude separately for
# every posterior draw, so the trend's uncertainty comes from the posterior.
dist_amp = sess_summary.set_index(sess_summary.index.astype(str))['dist_amp'].reindex(dates).values.astype(float)
X_amp = np.c_[np.ones_like(dist_amp), dist_amp]
amp_grid = np.linspace(dist_amp.min(), dist_amp.max(), 50)
jitter = (np.arange(len(dist_amp)) % 5 - 2) * 0.25   # spreads sessions that share an amplitude

fig, axes = plt.subplots(2, len(HANDS), constrained_layout=True, figsize=(11, 7))
for c, hand in enumerate(HANDS):
    for r, q in enumerate(['PSE', 'JND']):
        ax = axes[r, c]
        d = sess_diff[(hand, q)].stack(sample=('chain', 'draw')).transpose('sample', 'sessions').values
        d_mean = d.mean(0)
        d_hdi = az.hdi(d[None], hdi_prob=0.95)   # (sessions, 2)
        coef = np.linalg.lstsq(X_amp, d.T, rcond=None)[0]   # (intercept/slope, draws)
        slope = coef[1] * 10   # per 10 um of distractor
        slope_low, slope_high = az.hdi(slope, hdi_prob=0.95)
        ax.errorbar(dist_amp + jitter, d_mean, yerr=[d_mean - d_hdi[:, 0], d_hdi[:, 1] - d_mean], fmt='o',
                    markersize=4, color='#2a78d6', elinewidth=1, alpha=0.8, label='session: mean, 95% HDI')
        lines = coef[0][:, None] + coef[1][:, None] * amp_grid
        ax.fill_between(amp_grid, *np.quantile(lines, [0.025, 0.975], axis=0), color='#e34948', alpha=0.2, linewidth=0)
        ax.plot(amp_grid, lines.mean(0), color='#e34948', linewidth=2, label='linear trend (95% band)')
        ax.axhline(0, linestyle='--', linewidth=1, color=COLORS['prior'])
        ax.set_title(f'{hand}: slope {slope.mean():.2f} [{slope_low:.2f}, {slope_high:.2f}] per 10 $\\mu m$', fontsize=10)
        ax.set_ylabel(f'bimanual − unimanual\n{QUANTITY_LABELS[q]}')
        ax.grid(alpha=0.3)
        if r == 1:
            ax.set_xlabel(r'distractor amplitude ($\mu m$)')
axes[0, 0].legend(fontsize=8)
fig.suptitle('Distractor effect against distractor amplitude (post hoc; amplitude is not in the model)', fontsize=12)
save_fig(fig, 'effect_vs_distractor_amplitude.png')
plt.show()

#%%

ax = az.plot_ppc(trace, num_pp_samples=100, random_seed=0)
save_fig(np.ravel(ax)[0].figure, 'ppc_arviz_resp.png')
plt.show()

LOO_results = az.loo(trace)

fit_results = {'az_summary_trace': result_df,
               'az_loo_trace': LOO_results,
               'effect_summary': effect_summary}
#%% Numbers for Fit_B_Report.txt
# Everything quoted in the report that is not printed by an earlier cell. Section names follow the report.

def mean_hdi(vals, fmt='.2f'):
    vals = finite(vals)
    low, high = az.hdi(vals, hdi_prob=0.95)
    return f"{vals.mean():{fmt}} [{low:{fmt}}, {high:{fmt}}]"

def per_draw_ols(X, Y):
    """Least-squares coefficients for every posterior draw. X: (sessions, k), Y: (draws, sessions) -> (k, draws)."""
    return np.linalg.lstsq(X, Y.T, rcond=None)[0]

def draws_by_session(da):
    return da.stack(sample=('chain', 'draw')).transpose('sample', 'sessions').values

print("--- Data and fit")
print(f"{len(obs_data)} trials, {len(sessions)} sessions, {len(coords['groups'])} groups")
print(f"stimulus scale: PSE_um = {x_mu:.0f} + {x_sig:.2f} PSE, JND_um = {x_sig:.2f} JND")
print(f"{trace.posterior.sizes['chain']} chains x {trace.posterior.sizes['draw']} draws, "
      f"{trace.posterior.attrs.get('tuning_steps', '?')} tuning steps"
      + (f", {sampling_minutes:.0f} min" if 'sampling_minutes' in globals() else ""))

print("--- Sampling diagnostics")
full_summary = az.summary(trace, kind='diagnostics')
is_z_sig_b0 = full_summary.index.str.startswith('z_sig_b0')
rest = full_summary[~is_z_sig_b0]
print(f"divergences: {int(trace.sample_stats['diverging'].sum())}, "
      f"max tree depth: {int(trace.sample_stats['tree_depth'].max())}")
print(f"{len(full_summary)} parameters; all but z_sig_b0 ({is_z_sig_b0.sum()}): max r_hat {rest['r_hat'].max():.2f}, "
      f"min bulk ESS {rest['ess_bulk'].min():.0f}, min tail ESS {rest['ess_tail'].min():.0f}")
print(f"z_sig_b0: max r_hat {full_summary[is_z_sig_b0]['r_hat'].max():.2f}")
sig_b0_summary = full_summary.loc[full_summary.index.str.startswith('sig_betas[b0')]
print(f"sig_betas[b0]: max r_hat {sig_b0_summary['r_hat'].max():.2f}, min bulk ESS {sig_b0_summary['ess_bulk'].min():.0f}")

print("--- Posterior predictive checks, every session x group x amplitude cell")
cell_trials = pd.DataFrame({'s': sess_idx, 'g': grp_idx, 'a': amp, 'i': np.arange(len(obs_data))}
                           ).groupby(['s', 'g', 'a'])['i'].apply(np.array)
n_inside = 0
for idx in cell_trials:
    k_rep = y_rep[:, idx].sum(1)
    low, high = np.quantile(k_rep, [0.025, 0.975])
    n_inside += low <= obs_data[idx].sum() <= high
print(f"{n_inside} of {len(cell_trials)} cells inside their 95% posterior predictive interval "
      f"({n_inside / len(cell_trials):.1%})")

print("--- LOO: current priors vs earlier priors")
with open(MODEL_DIR / "Results_B_oldpriors.pkl", "rb") as f:
    fit_results_oldpriors = pickle.load(f)   # the fit with the earlier priors, before 2026-10-05
for label, loo in [('current', LOO_results), ('earlier', fit_results_oldpriors['az_loo_trace'])]:
    print(f"{label}: elpd_loo {loo.elpd_loo:.1f} (SE {loo.se:.1f}), p_loo {loo.p_loo:.0f}, "
          f"max Pareto k {np.max(loo.pareto_k.values):.2f}")

print("--- Population level (typical session), Table tab:population")
pop_table = pd.DataFrame({q: {g: mean_hdi(pop_post[q].sel(groups=g), '.3f' if q.startswith('gam') else '.2f')
                              for g in groups} for q in QUANTITIES})
print(pop_table.to_string())

print("--- Hyperparameters under the earlier and current priors: mean (sd), Table tab:hyper")
HYPER_ROWS = {'mu_b1': 'mu_betas[b1, {g}]', 'sig_b0': 'sig_betas[b0, {g}]', 'sig_b1': 'sig_betas[b1, {g}]',
              'mu_gam_h': 'mu_gams[b0, {g}]', 'mu_gam_l': 'mu_gams[b1, {g}]',
              'sig_gam_h': 'sig_gams[b0, {g}]', 'sig_gam_l': 'sig_gams[b1, {g}]'}
hyper_summaries = {'earlier': fit_results_oldpriors['az_summary_trace'],
                   'current': az.summary(trace, var_names=['mu_betas', 'sig_betas', 'mu_gams', 'sig_gams'])}
hyper_table = pd.DataFrame({(g, label): {row: f"{summ.loc[name.format(g=g), 'mean']:.2f} ({summ.loc[name.format(g=g), 'sd']:.2f})"
                                         for row, name in HYPER_ROWS.items()}
                            for g in groups for label, summ in hyper_summaries.items()})
print(hyper_table.to_string())
print(f"prior median of sig_gams: {np.median(trace.prior['sig_gams'].values):.2f}")
print(f"left_bi sig_b1: {mean_hdi(trace.posterior['sig_betas'].sel(betas='b1', groups='left_bi'))}")
jnd_left_bi = trace.posterior['JND'].sel(groups='left_bi').mean(('chain', 'draw')) * x_sig
print(f"left_bi JND: largest {float(jnd_left_bi.max()):.1f} um on {str(jnd_left_bi.idxmax('sessions').values)}, "
      f"median over sessions {float(jnd_left_bi.median()):.1f} um")

print("--- Distractor effect, per session")
for hand in HANDS:
    pse_mean = sess_diff[(hand, 'PSE')].mean(('chain', 'draw'))
    above = pse_mean.sessions.values[pse_mean.values >= 0]
    print(f"{hand}: PSE shift negative in {int((pse_mean < 0).sum())} of {len(dates)} sessions; "
          f"non-negative in {list(above)} ({np.round(pse_mean.sel(sessions=above).values, 2)} um)")
jnd_shift_left = sess_diff[('Left Hand', 'JND')].mean(('chain', 'draw'))
print(f"Left hand: largest JND shift {float(jnd_shift_left.max()):+.1f} um on {str(jnd_shift_left.idxmax('sessions').values)}")
for g in groups:
    for q in ['gam_h', 'gam_l']:
        m = trace.posterior[q].sel(groups=g).mean(('chain', 'draw')).values
        print(f"{g} {q}: session posterior means {m.min():.3f} to {m.max():.3f}")

print("--- Joint posteriors in session 06-14 (appendix)")
for g in groups:
    pars = {name: trace.posterior[v].sel(groups=g, sessions='06-14', **sel).values.ravel()
            for name, v, sel in [('gam_h', 'gam_h', {}), ('gam_l', 'gam_l', {}),
                                 ('b0', 'beta_vec', {'betas': 'b0'}), ('b1', 'beta_vec', {'betas': 'b1'})]}
    r = pd.DataFrame(pars).corr()
    pse_um = float(trace.posterior['PSE'].sel(groups=g, sessions='06-14').mean()) * x_sig + x_mu
    print(f"{g}: PSE {pse_um:.1f} um, r(b0,b1) {r.loc['b0', 'b1']:+.2f}, "
          f"max |r| lapse vs beta or lapse {r.loc[['gam_h', 'gam_l'], ['b0', 'b1']].abs().values.max():.2f}, "
          f"r(gam_h,gam_l) {r.loc['gam_h', 'gam_l']:+.2f}")
hyper_pars = pd.DataFrame({
    'gam_h': trace.posterior['gam_h'].sel(groups='left_bi', sessions='06-14').values.ravel(),
    'mu_gam_h': trace.posterior['mu_gams'].sel(groups='left_bi', betas='b0').values.ravel(),
    'sig_gam_h': trace.posterior['sig_gams'].sel(groups='left_bi', betas='b0').values.ravel()}).corr()
print(f"left_bi: r(gam_h, mu_gam_h) {hyper_pars.loc['gam_h', 'mu_gam_h']:.2f}, "
      f"r(gam_h, sig_gam_h) {hyper_pars.loc['gam_h', 'sig_gam_h']:.2f}")

print("--- Dependence on distractor amplitude")
print("schedule:", ', '.join(f"{a:.0f} um: {n} sessions" for a, n in zip(*np.unique(dist_amp, return_counts=True))))
print("sessions in order:", ' '.join(f"{a:.0f}" for a in dist_amp))
session_order = np.arange(len(dates), dtype=float)
print(f"correlation of amplitude and session order: r = {np.corrcoef(dist_amp, session_order)[0, 1]:.2f}")
print("mean PSE shift over the sessions at each amplitude (Table tab:amp):")
for hand in HANDS:
    d = draws_by_session(sess_diff[(hand, 'PSE')])
    print(f"  {hand}: " + '; '.join(f"{a:.0f} um ({(dist_amp == a).sum()}): {mean_hdi(d[:, dist_amp == a].mean(1), '.1f')}"
                                   for a in np.unique(dist_amp) if (dist_amp == a).sum() >= 5))
X_both = np.c_[np.ones_like(dist_amp), dist_amp, session_order]
print("regression on amplitude and session order (per 10 um, per 10 sessions):")
for hand, (uni, bi) in HANDS.items():
    for label, d in [('bimanual - unimanual', draws_by_session(sess_diff[(hand, 'PSE')])),
                     ('unimanual', draws_by_session(trace.posterior['PSE'].sel(groups=uni, drop=True)) * x_sig),
                     ('bimanual', draws_by_session(trace.posterior['PSE'].sel(groups=bi, drop=True)) * x_sig)]:
        coef = per_draw_ols(X_both, d)
        print(f"  {hand} PSE {label}: amplitude {mean_hdi(coef[1] * 10, '.1f')}, order {mean_hdi(coef[2] * 10, '.1f')}")

#%%
with open(MODEL_DIR / "Results_B.pkl","wb") as f:
    pickle.dump(fit_results, f)