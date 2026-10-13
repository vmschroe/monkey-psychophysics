#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 2026

Prior predictive check of Model C's distractor prior.

The priors on b0, b1 and the lapse rates are Model B's, checked in
Model_B/Check_Prior_B.py. New in Model C is b2, the coefficient of the
distractor covariate d (distractor amplitude / 14.63, 0 on unimanual trials):
    mu_b2 ~ N(0, 1), sig_b2 ~ HalfNormal(1).
Since unimanual and bimanual trials share the slope and lapse rates, the PSE
shift (bimanual - unimanual) of a session is -b2 * d / b1, in standardized
units. This script samples the prior from the model on the real design and
looks at the shifts it implies, in micrometres:
    - for a typical session (hand-level means) at each distractor amplitude;
    - between sessions at the same amplitude (sig_b2);
    - for the bimanual PSE itself, and the prior psychometric curves.

Every number and figure is used in Convergence_C_Report.tex.

@author: vmschroe
"""

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
PLOT_DIR = MODEL_DIR / "Plots_C" / "Convergence_Plots_C"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261013
N_DRAWS = 4000

#%% Sample the prior from the model on the real design

with open(MODEL_DIR / "Data_C" / "ReadyData_Sirius_C.pkl", "rb") as f:
    data_dict = pickle.load(f)
cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
sess_idx = data_dict['sess_idx']
sess_x_dist = data_dict['sess_x_dist']
obs_data = data_dict['resp']
sessions = list(data_dict['dates_sess_idx'])
x_mu, x_sig = data_dict['x_mu'], data_dict['x_sig']

exec(open(MODEL_DIR / "Build_Model_C.py").read())

with model_C:
    prior = pm.sample_prior_predictive(draws=N_DRAWS, random_seed=SEED,
                                       var_names=['mu_betas', 'sig_betas', 'beta_vec', 'gam_h', 'gam_l', 'PSE']).prior

AMPS = [6, 12, 24, 32]   # the most used distractor amplitudes (um)
STIM_RANGE = (6, 50)     # stimulus amplitudes (um)
mu_b1 = prior['mu_betas'].sel(betas='b1').values.ravel()
mu_b2 = prior['mu_betas'].sel(betas='b2').values.ravel()
sig_b2 = prior['sig_betas'].sel(betas='b2').values.ravel()

#%% Typical-session PSE shift at each amplitude, and its spread between sessions

print("Typical session: PSE shift -mu_b2 * d / mu_b1 (um)")
typical_shift = {}
for a in AMPS:
    d = a / x_sig
    typical_shift[a] = -mu_b2 * d / mu_b1 * x_sig
    v = typical_shift[a]
    print(f"  {a:2d} um: median |shift| {np.median(np.abs(v)):.1f}, 90% interval {np.round(np.percentile(v, [5, 95]), 1)}, "
          f"P(|shift| > 10 um) = {np.mean(np.abs(v) > 10):.3f}")

# sessions at the same amplitude differ in b2 by sig_b2, so in shift by sig_b2 * d / mu_b1
d32 = 32 / x_sig
spread32 = sig_b2 * d32 / mu_b1 * x_sig
print(f"Between-session sd of the shift at 32 um: median {np.median(spread32):.1f} um, 95% up to {np.percentile(spread32, 95):.1f} um")

#%% Session level: bimanual PSE and session shifts

pse = prior['PSE'] * x_sig + x_mu   # (chain, draw, groups, conditions, sessions)
outside = lambda v: np.mean((v < STIM_RANGE[0]) | (v > STIM_RANGE[1]))
pse_uni = pse.sel(conditions='uni').values.ravel()
pse_bi = pse.sel(conditions='bi').values.ravel()
shift = (pse.sel(conditions='bi') - pse.sel(conditions='uni'))   # (chain, draw, groups, sessions)
sess_amp = np.round(sess_x_dist * x_sig).astype(int)
shift_32 = shift.isel(sessions=np.flatnonzero(sess_amp == 32)).values.ravel()
shift_6 = shift.isel(sessions=np.flatnonzero(sess_amp == 6)).values.ravel()
print(f"session PSE outside the stimulus range {STIM_RANGE}: unimanual {outside(pse_uni):.1%}, bimanual {outside(pse_bi):.1%}")
print(f"session PSE shift, 6 um sessions: 90% interval {np.round(np.percentile(shift_6, [5, 95]), 1)} um")
print(f"session PSE shift, 32 um sessions: 90% interval {np.round(np.percentile(shift_32, [5, 95]), 1)} um")

#%% Figure: typical shifts, session shifts, prior curves

BLUE, RED, GRAY = '#2a78d6', '#e34948', '#8a8984'
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

ax = axes[0]
lo, med, hi = (np.array([np.percentile(typical_shift[a], q) for a in AMPS]) for q in [5, 50, 95])
lo25, hi75 = (np.array([np.percentile(typical_shift[a], q) for a in AMPS]) for q in [25, 75])
ax.fill_between(AMPS, lo, hi, color=BLUE, alpha=0.2, linewidth=0, label='90% interval')
ax.fill_between(AMPS, lo25, hi75, color=BLUE, alpha=0.35, linewidth=0, label='50% interval')
ax.plot(AMPS, med, color=BLUE, marker='o', label='median')
ax.axhline(0, linestyle='--', linewidth=1, color=GRAY)
ax.set_xlabel(r'distractor amplitude ($\mu m$)')
ax.set_ylabel(r'PSE shift, bimanual − unimanual ($\mu m$)')
ax.set_title('Typical session')
ax.legend(fontsize=9)

ax = axes[1]
bins = np.linspace(-40, 40, 81)
for v, color, label in [(shift_6, GRAY, '6 $\\mu m$ sessions'), (shift_32, BLUE, '32 $\\mu m$ sessions')]:
    ax.hist(v, bins=bins, weights=np.full(v.size, 1 / v.size), color=color, alpha=0.5, label=label)
ax.set_xlabel(r'session PSE shift ($\mu m$)')
ax.set_yticks([])
ax.set_title('Sessions')
ax.legend(fontsize=9)

ax = axes[2]
rng_curves = np.random.default_rng(2)
x = np.linspace(-1.6, 1.6, 200)
s_32 = int(np.flatnonzero(sess_amp == 32)[0])
par = {p: prior[p].isel(sessions=s_32, groups=0).values.ravel() for p in ['gam_h', 'gam_l']}
for b in ['b0', 'b1', 'b2']:
    par[b] = prior['beta_vec'].sel(betas=b).isel(sessions=s_32, groups=0).values.ravel()
for k in rng_curves.choice(par['b0'].size, 30, replace=False):
    for d, color in [(0.0, BLUE), (d32, RED)]:
        psi = par['gam_h'][k] + (1 - par['gam_h'][k] - par['gam_l'][k]) / (1 + np.exp(-(par['b0'][k] + par['b1'][k] * x + par['b2'][k] * d)))
        ax.plot(x * x_sig + x_mu, psi, color=color, linewidth=1, alpha=0.35)
ax.plot([], [], color=BLUE, label='unimanual')
ax.plot([], [], color=RED, label='bimanual (32 $\\mu m$)')
ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
ax.set_ylabel('P(response = "high")')
ax.set_title('Prior curves, one 32 $\\mu m$ session (30 draws)')
ax.legend(fontsize=9)

fig.suptitle('Model C prior on the distractor effect', fontsize=14)
fig.savefig(PLOT_DIR / "Prior_Check_C_distractor.png", dpi=200, bbox_inches='tight')
plt.show()
