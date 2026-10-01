#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:34:04 2026

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
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import uniform


#%% USING LOGIT_NORMAL for gamma params
nsamps = 10000

z_all = norm.rvs(size=nsamps*3).reshape((-1,3))
z_s = z_all[:,0]
z_mu = z_all[:,1]
z_sig = z_all[:,2]

mu_gam = 1.5*z_mu+(-3)
sig_gam = np.abs(0.5*z_sig)
gam_samps = 0.25*(1 / (1 + np.exp(-(sig_gam*z_s+mu_gam))))


plt.hist(gam_samps, density=True, bins=20)


#%% SLOPE AND LAPSE PRIORS, sampled from the actual model on the real design
# mu_b0 ~ N(0, 2), sig_b0 ~ HalfNormal(2), mu_b1 ~ LogNormal(log 4, 0.5), sig_b1 ~ Exp(mean 1),
# mu_gam ~ N(-3, 1.5), sig_gam ~ HalfNormal(0.5)

from pathlib import Path
from scipy.stats import lognorm

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()

with open(MODEL_DIR / "Data_B" / "ReadyData_Sirius_B.pkl", "rb") as f:
    data_dict = pickle.load(f)
sessions = list(data_dict['dates_sess_idx'])
cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']
sess_idx = data_dict['sess_idx']
exec(open(MODEL_DIR / "Build_Model_B.py").read())

with model_B:
    prior = pm.sample_prior_predictive(draws=4000, var_names=['mu_betas', 'sig_betas', 'beta_vec', 'mu_gams', 'sig_gams', 'gam_h', 'gam_l', 'PSE', 'JND'],
                                       random_seed=1).prior

mu_b1_samps = prior['mu_betas'].sel(betas='b1').values.ravel()
sig_b1_samps = prior['sig_betas'].sel(betas='b1').values.ravel()
b1_samps = prior['beta_vec'].sel(betas='b1').values.ravel()
jnd_samps = prior['JND'].values.ravel()
print(f"mu_b1 prior: median {np.median(mu_b1_samps):.2f}, 95% interval {np.round(np.percentile(mu_b1_samps, [2.5, 97.5]), 2)}, "
      f"P(mu_b1 > 8) = {np.mean(mu_b1_samps > 8):.3f}")
print(f"sig_b1 prior: mean {sig_b1_samps.mean():.2f}, 95% interval {np.round(np.percentile(sig_b1_samps, [2.5, 97.5]), 2)}")
print(f"session b1 prior: P(b1 < 0) = {np.mean(b1_samps < 0):.3f}")
print(f"session JND prior (finite, b1 > 0): median {np.median(jnd_samps[b1_samps > 0]):.2f}")


BLUE, GRAY = '#2a78d6', '#8a8984'
stim_levels = np.unique(np.round(cov_mat[:, 1], 3))

fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

ax = axes[0, 0]
grid = np.linspace(0, 16, 400)
ax.hist(mu_b1_samps, bins=np.linspace(0, 16, 65), density=True, color=BLUE, alpha=0.35, label='model samples')
ax.plot(grid, lognorm.pdf(grid, s=0.5, scale=4), color=BLUE, linewidth=2, label='LogNormal(log 4, 0.5)')
ax.plot([0, 8, 8], [1/8, 1/8, 0], color=GRAY, linewidth=2, linestyle='--', label='old Uniform[0, 8]')
ax.set_xlabel('$\\mu_{b_1}$')
ax.set_title('Group slope prior')
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.hist(b1_samps, bins=np.linspace(-4, 20, 97), density=True, color=BLUE, alpha=0.6)
ax.set_xlabel('$b_{1,gs}$')
ax.set_title(f'Session slope prior (P($b_1$ < 0) = {np.mean(b1_samps < 0):.3f})')

ax = axes[1, 0]
jnd_pos = jnd_samps[b1_samps > 0]
ax.hist(jnd_pos, bins=np.linspace(0, 2, 81), density=True, color=BLUE, alpha=0.6)
ax.set_xlabel('JND (standardized stimulus units)')
ax.set_title(f'Session JND prior ({np.mean(jnd_pos > 2):.1%} above 2 not shown)')

ax = axes[1, 1]
rng_curves = np.random.default_rng(2)
x = np.linspace(-2, 2, 200)
b0_s, b1_s = prior['beta_vec'].sel(betas='b0').values.ravel(), b1_samps
gh_s, gl_s = prior['gam_h'].values.ravel(), prior['gam_l'].values.ravel()
for k in rng_curves.choice(b1_s.size, 60, replace=False):
    psi = gh_s[k] + (1 - gh_s[k] - gl_s[k]) / (1 + np.exp(-(b0_s[k] + b1_s[k] * x)))
    ax.plot(x, psi, color=BLUE, linewidth=1, alpha=0.35)
for s in stim_levels:
    ax.axvline(s, color=GRAY, linewidth=0.8, linestyle=':')
ax.set_xlabel('standardized stimulus (dotted: stimulus levels)')
ax.set_ylabel('P(response = 1)')
ax.set_title('Prior psychometric curves (60 draws)')

for ax in axes.ravel()[:3]:
    ax.set_yticks([])
fig.suptitle('Model B prior on the slope', fontsize=14)
fig.savefig(MODEL_DIR / "Plots_B" / "Prior_Check_B_slope.png", dpi=200, bbox_inches='tight')
plt.show()


#%% Lapse-rate prior: typical lapse rate of a group and session-level lapse rates

typical_gam = 0.25 / (1 + np.exp(-prior['mu_gams'].values.ravel()))
typical_gam_old = 0.25 / (1 + np.exp(-norm.rvs(loc=-2.5, scale=1, size=typical_gam.size, random_state=3)))
session_gam = np.concatenate([prior['gam_h'].values.ravel(), prior['gam_l'].values.ravel()])
for name, s in [('typical gamma, N(-3, 1.5)', typical_gam), ('typical gamma, old N(-2.5, 1)', typical_gam_old),
                ('session gamma', session_gam)]:
    print(f"{name}: median {np.median(s):.4f}, 95% interval {np.round(np.percentile(s, [2.5, 97.5]), 4)}")

# day-to-day spread implied by the sig_gam prior: sessions of a group whose typical lapse rate is 0.05
sig_gam_samps = prior['sig_gams'].values.ravel()
mu_at_5pct = np.log(0.2 / 0.8)   # 0.25 * logit^-1(mu) = 0.05
z_sess = norm.rvs(size=sig_gam_samps.size, random_state=4)
gam_at_5pct = 0.25 / (1 + np.exp(-(mu_at_5pct + sig_gam_samps * z_sess)))
print(f"sig_gam prior: median {np.median(sig_gam_samps):.2f}, 95% interval {np.round(np.percentile(sig_gam_samps, [2.5, 97.5]), 2)}")
print(f"sessions at a typical lapse rate of 0.05: {np.mean((gam_at_5pct > 0.03) & (gam_at_5pct < 0.07)):.0%} within 0.03-0.07, "
      f"90% interval {np.round(np.percentile(gam_at_5pct, [5, 95]), 3)}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
bins = np.logspace(-5, np.log10(0.25), 61)
frac = lambda s: np.full(s.size, 1 / s.size)   # fraction of samples per bin; density=True would distort log-spaced bins

ax = axes[0]
ax.hist(typical_gam, bins=bins, weights=frac(typical_gam), color=BLUE, alpha=0.6, label='$\\mu_\\gamma$ ~ N(-3, 1.5)')
ax.hist(typical_gam_old, bins=bins, weights=frac(typical_gam_old), histtype='step', color=GRAY, linewidth=2, linestyle='--',
        label='old $\\mu_\\gamma$ ~ N(-2.5, 1)')
ax.set_title('Typical lapse rate of a group, 0.25 logit$^{-1}(\\mu_\\gamma)$')
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(session_gam, bins=bins, weights=frac(session_gam), color=BLUE, alpha=0.6)
ax.set_title('Session lapse rates $\\gamma_{h,gs}$, $\\gamma_{l,gs}$')

for ax in axes:
    ax.set_xscale('log')
    ax.set_xlabel('lapse rate (log scale; capped at 0.25 by the model)')
    ax.set_yticks([])
fig.suptitle('Model B prior on the lapse rates', fontsize=14)
fig.savefig(MODEL_DIR / "Plots_B" / "Prior_Check_B_lapse.png", dpi=200, bbox_inches='tight')
plt.show()


#%% Bias prior, seen through the PSE in raw amplitude units (28 is the category boundary)

with open(MODEL_DIR.parent / "Sirius_Data.pkl", "rb") as f:
    raw_stims = np.unique(pickle.load(f)['data']['stim_amp'])
amp_mu, amp_sd = raw_stims.mean(), raw_stims.std()   # same standardization as DataProcessing_B.py
to_amp = lambda x: amp_mu + amp_sd * x

mu_b0_samps = prior['mu_betas'].sel(betas='b0').values.ravel()
typical_pse = to_amp(-mu_b0_samps / mu_b1_samps)   # typical session of a group, ignoring lapses
session_pse = to_amp(prior['PSE'].values.ravel())
day_to_day_sd = amp_sd * prior['sig_betas'].sel(betas='b0').values.ravel() / mu_b1_samps
outside = lambda s: np.mean((s < raw_stims.min()) | (s > raw_stims.max()))
print(f"typical PSE: 90% interval {np.round(np.percentile(typical_pse, [5, 95]), 1)}, outside stimulus range {outside(typical_pse):.1%}")
print(f"session PSE: 90% interval {np.round(np.percentile(session_pse, [5, 95]), 1)}, outside stimulus range {outside(session_pse):.1%}")
print(f"day-to-day PSE sd (amplitude units): median {np.median(day_to_day_sd):.1f}, 95% up to {np.percentile(day_to_day_sd, 95):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
bins = np.linspace(-30, 86, 117)
for ax, s, title in [(axes[0], typical_pse, 'Typical PSE of a group'), (axes[1], session_pse, 'Session PSEs')]:
    ax.hist(s, bins=bins, weights=np.full(s.size, 1 / s.size), color=BLUE, alpha=0.6)
    for a in raw_stims:
        ax.axvline(a, color=GRAY, linewidth=0.8, linestyle=':')
    ax.set_title(f'{title} ({outside(s):.1%} outside the stimulus range)')
    ax.set_xlabel('stimulus amplitude (dotted: stimulus levels)')
    ax.set_yticks([])
fig.suptitle('Model B prior on the bias, through the PSE', fontsize=14)
fig.savefig(MODEL_DIR / "Plots_B" / "Prior_Check_B_bias.png", dpi=200, bbox_inches='tight')
plt.show()


#%%



#%% Defunct below


#%% DONT USE BETA ANYMORE! screws w hmc

def get_beta_params(mu_gh,sig_gh):
    nu_gh  = (mu_gh * (1.0 - mu_gh) / (sig_gh**2)) - 1.0  # <-- fixed *
    a_gh   = mu_gh * nu_gh
    b_gh   = (1.0 - mu_gh) * nu_gh
    return [a_gh,b_gh]

#%%

mu_tru = 0.2
sig_tru = 0.15
ab = get_beta_params(mu_tru,sig_tru)
samps_true_beta = beta.rvs(a=ab[0], b=ab[1], size=1000)

mus = beta.rvs(a=ab[0], b=ab[1], size=1000)
uvar = uniform.rvs(loc=0, scale=1, size=1000)
sigs = np.sqrt(mus*(1-mus)*uvar)
abweird = get_beta_params(mus,sigs)
samps_w = beta.rvs(a=abweird[0][:], b=abweird[1][:], size=1000)


#%% 

plt.hist(samps_true_beta, density=True)
plt.hist(samps_w, density=True) 
