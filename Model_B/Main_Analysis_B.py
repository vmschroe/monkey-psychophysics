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

with model_B:
    trace = pm.sample(draws = 5000, return_inferencedata=True, chains = 4, cores = 1, progressbar=True, idata_kwargs={"log_likelihood": True})   
print("FINISHED SAMPLING!")


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

ax = az.plot_ppc(trace, num_pp_samples=100)
save_fig(np.ravel(ax)[0].figure, 'ppc_arviz_resp.png')
plt.show()

LOO_results = az.loo(trace)

fit_results = {'az_summary_trace': result_df,
               'az_loo_trace': LOO_results,
               'effect_summary': effect_summary}
#%%
with open(MODEL_DIR / "Results_B.pkl","wb") as f:
    pickle.dump(fit_results, f)