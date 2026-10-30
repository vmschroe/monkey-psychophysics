#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

Steps 6 and 7 of the workflow for Model D: simulate a Bayesian ensemble from
the model on the real trial design, then compare it with the thresholds in
Summary_Stats_D.py.
    - Prior pushforward checks: summary functions of the parameters.
    - Prior predictive checks: summary statistics of the simulated responses.
A threshold passes if 0.1%-5% of the ensemble lies beyond it; less than 0.1%
means the prior is tighter than the threshold, which isn't a failure.

The real responses are never used: the model is built on zeros in their place,
and nothing from a real-data fit appears here.

Outputs, all written by this script and used in Workflow_D_Report.tex:
    - Prior_Check_D_Results.txt
    - Plots_D/Prior_Plots_D/prior_pushforward.png, prior_shift_by_amplitude.png,
      prior_predictive_curves.png, prior_predictive_stats.png

@author: vmschroe
"""

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
sys.path.insert(0, str(MODEL_DIR))
from Summary_Stats_D import (PUSHFORWARD_THRESHOLDS, PREDICTIVE_THRESHOLDS, prepare, summary_stats)

PLOT_DIR = MODEL_DIR / "Plots_D" / "Prior_Plots_D"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = MODEL_DIR / "Prior_Check_D_Results.txt"

SEED = 20261030
N_DRAWS = 1000
AMPS = [6, 12, 18, 24, 32]   # distractor amplitudes shown (um)
BLUE, RED, GRAY = '#2a78d6', '#e34948', '#8a8984'

lines = []
def report(text=''):
    print(text)
    lines.append(str(text))

#%% Step 6: simulate the ensemble on the real design

with open(MODEL_DIR / "Data_D" / "ReadyData_Sirius_D.pkl", "rb") as f:
    data_dict = pickle.load(f)
design = {k: v for k, v in data_dict.items() if k != 'resp'}
x_stim = design['x_stim']
cond_idx = design['cond_idx']
grp_idx = design['grp_idx']
sess_idx = design['sess_idx']
sess_x_dist = design['sess_x_dist']
obs_data = np.zeros_like(data_dict['resp'])   # only sets the shape; the responses stay unused
sessions = list(design['dates_sess_idx'])
x_mu, x_sig = design['x_mu'], design['x_sig']
del data_dict

exec(open(MODEL_DIR / "Build_Model_D.py").read())

with model_D:
    ensemble = pm.sample_prior_predictive(draws=N_DRAWS, random_seed=SEED)
prior = ensemble.prior
y_sim = ensemble.prior_predictive['resp'].values.reshape(N_DRAWS, -1)

#%% Step 5: summary functions of the parameters

def over_draws(da):
    return da.values.reshape(N_DRAWS, -1)

pse = prior['PSE'] * x_sig + x_mu
gam = {name: 0.25 / (1 + np.exp(-prior['mu_gams'].sel(lapses=name))) for name in ['gam_h', 'gam_l']}
mu_b1 = prior['mu_betas'].sel(betas='b1')
typical_shift = {a: -(prior['mu_b3'] + prior['mu_b2'] * a / x_sig) / mu_b1 * x_sig for a in AMPS}

pushforward = {
    'session PSE, unimanual (um)': over_draws(pse.sel(conditions='uni')),
    'session PSE, bimanual (um)': over_draws(pse.sel(conditions='bi')),
    'session JND (um)': over_draws(prior['JND'] * x_sig),
    'typical gam_h': over_draws(gam['gam_h']),
    'typical gam_l': over_draws(gam['gam_l']),
    'session gam_h': over_draws(prior['gam_h']),
    'session gam_l': over_draws(prior['gam_l']),
    'typical shift at 6 um (um)': over_draws(typical_shift[6]),
    'typical shift at 32 um (um)': over_draws(typical_shift[32]),
    'between-session sd of PSE (um)': over_draws(pse.sel(conditions='uni').std('sessions')),
}

#%% Step 7: tail probabilities beyond the thresholds

def tail_row(name, values, limits):
    lo, hi = limits
    values = values[np.isfinite(values)]
    beyond = np.zeros(values.size, bool)
    beyond |= values < lo if lo is not None else beyond
    beyond |= values > hi if hi is not None else beyond
    share = beyond.mean()
    verdict = 'too wide' if share > 0.05 else ('tighter' if share < 0.001 else 'ok')
    q05, q50, q95 = np.percentile(values, [5, 50, 95])
    return {'summary': name, 'threshold': f"({'' if lo is None else lo}, {'' if hi is None else hi})",
            'q05': round(q05, 3), 'median': round(q50, 3), 'q95': round(q95, 3),
            'beyond': f"{share:.2%}", 'verdict': verdict}

report(f"Model D prior checks: {N_DRAWS} draws, seed {SEED}, real design ({len(obs_data)} trials, {len(sessions)} sessions)")
report("\nPrior pushforward checks")
table_push = pd.DataFrame([tail_row(k, pushforward[k], v) for k, v in PUSHFORWARD_THRESHOLDS.items()])
report(table_push.to_string(index=False))

prep = prepare(design)
stats = pd.DataFrame([summary_stats(y, prep) for y in y_sim])

report("\nPrior predictive checks")
table_pred = pd.DataFrame([tail_row(k, stats[k].to_numpy(), v) for k, v in PREDICTIVE_THRESHOLDS.items()])
report(table_pred.to_string(index=False))

report("\nStatistics without thresholds, for the retrodictive checks: 5%, 50%, 95% of the ensemble")
for name in ['shift_intercept_left', 'shift_slope_left', 'shift_resid_sd_weak_left', 'shift_resid_sd_strong_left',
             'corr_uni_bi_left', 'corr_hands_uni', 'corr_hands_bi', 'p_repeat', 'accuracy_late_minus_early', 'target_side_bias_left']:
    report(f"  {name:28s} {np.round(np.nanpercentile(stats[name], [5, 50, 95]), 3)}")

# what the design can say about b3 apart from b2: session-level regression of delta on x_dist
X = np.column_stack([np.ones_like(sess_x_dist), sess_x_dist])
cov = np.linalg.inv(X.T @ X)
report("\nDesign: regression of the session shift delta on x_dist, per unit of sig_delta")
report(f"  sd of mu_b3 {np.sqrt(cov[0, 0]):.3f}, sd of mu_b2 {np.sqrt(cov[1, 1]):.3f}, correlation {cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]):.3f}")

RESULTS_PATH.write_text("\n".join(lines) + "\n")

#%% Figures

def threshold_lines(ax, limits):
    for v in limits:
        if v is not None:
            ax.axvline(v, color=RED, linewidth=1.5, linestyle='--')

def hist_panel(ax, values, limits, title, clip):
    values = values[np.isfinite(values)]
    ax.hist(np.clip(values, *clip), bins=60, range=clip, color=BLUE, alpha=0.75)
    threshold_lines(ax, limits)
    ax.set_xlim(*clip)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)

panels = [('session PSE, unimanual (um)', (-40, 100)), ('session PSE, bimanual (um)', (-40, 100)),
          ('session JND (um)', (0, 40)), ('session gam_h', (0, 0.25)),
          ('typical shift at 6 um (um)', (-40, 40)), ('typical shift at 32 um (um)', (-60, 60)),
          ('between-session sd of PSE (um)', (0, 40)), ('typical gam_h', (0, 0.25))]
fig, axes = plt.subplots(2, 4, figsize=(15, 6), constrained_layout=True)
for ax, (name, clip) in zip(axes.ravel(), panels):
    hist_panel(ax, pushforward[name], PUSHFORWARD_THRESHOLDS[name], name.replace('um', '$\\mu m$').replace('gam_h', '$\\gamma_h$'), clip)
fig.suptitle('Model D prior pushforward (dashed: thresholds; values past the axis are piled at its ends)', fontsize=13)
fig.savefig(PLOT_DIR / "prior_pushforward.png", dpi=200, bbox_inches='tight')
plt.show()

def ribbons(ax, xs, samples, color):
    # nested 10-90, 20-80, 30-70, 40-60% intervals and the median, as in the workflow's prior checks
    for lo, hi, alpha in [(10, 90, 0.15), (20, 80, 0.2), (30, 70, 0.25), (40, 60, 0.3)]:
        ax.fill_between(xs, np.nanpercentile(samples, lo, axis=0), np.nanpercentile(samples, hi, axis=0), color=color, alpha=alpha, linewidth=0)
    ax.plot(xs, np.nanmedian(samples, axis=0), color=color, marker='o', markersize=4, linewidth=2)

fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
ribbons(ax, AMPS, np.column_stack([typical_shift[a].sel(groups='left').values.ravel() for a in AMPS]), BLUE)
for v in (-22, 22):
    ax.axhline(v, color=RED, linestyle='--', linewidth=1.5)
ax.set_xlabel(r'distractor amplitude ($\mu m$)')
ax.set_ylabel(r'typical PSE shift, bimanual − unimanual ($\mu m$)')
ax.set_title('Prior on the typical shift, left hand\n(nested 20-80% ... ribbons: 80, 60, 40, 20% intervals)', fontsize=11)
fig.savefig(PLOT_DIR / "prior_shift_by_amplitude.png", dpi=200, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
for ax, hand in zip(axes, ['left', 'right']):
    for cond, color, label in [('uni', BLUE, 'unimanual'), ('bi', RED, 'bimanual')]:
        ribbons(ax, prep['stims'], stats[[f'p_high_{hand}_{cond}_{s}' for s in prep['stims']]].to_numpy(), color)
        ax.plot([], [], color=color, label=label)
    ax.axvline(28, color=GRAY, linewidth=1, linestyle=':')
    ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
    ax.set_title(f'{hand} hand', fontsize=11)
axes[0].set_ylabel('P(response = "high"), pooled over sessions')
axes[0].legend(fontsize=9)
fig.suptitle('Model D prior predictive curves (ribbons: 80, 60, 40, 20% intervals of the ensemble)', fontsize=13)
fig.savefig(PLOT_DIR / "prior_predictive_curves.png", dpi=200, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True)
stat_panels = [('accuracy', 'accuracy', (0.4, 1)), ('p_high_6um', 'largest P(high) at 6 $\\mu m$', (0, 1)),
               ('p_high_50um', 'smallest P(high) at 50 $\\mu m$', (0, 1)),
               ('mid_diff_bi_uni', 'bi − uni in P(high), 24 and 32 $\\mu m$', (-1, 1)),
               ('sess_sd_mid', 'between-session sd of P(high)', (0, 0.5))]
for ax, (name, title, clip) in zip(axes, stat_panels):
    hist_panel(ax, stats[name].to_numpy(), PREDICTIVE_THRESHOLDS[name], title, clip)
fig.suptitle('Model D prior predictive statistics (dashed: thresholds)', fontsize=13)
fig.savefig(PLOT_DIR / "prior_predictive_stats.png", dpi=200, bbox_inches='tight')
plt.show()
