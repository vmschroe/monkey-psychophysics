#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 2026

Within-session learning check for Model A.

Each session's trials are split in half chronologically. Model A (one
psychometric curve per group, sessions pooled) is then fitted separately to
all the first halves and to all the second halves, with the model and priors
unchanged (Build_Model_A.py), and the two posteriors are compared group by
group: PSE, JND, lapse rates and psychometric curves.

Split rules:
    - Halves are by trial count, using the trials kept in the analysis data
      (aborts, frequency manipulations and repeats already dropped).
    - Rows of Sirius_Data.pkl are in chronological order within each session.
      The one exception to increasing start_time is session 04-14, which was
      recorded as two files (A at 08:59, B at 09:42); its rows are A then B,
      so row order is still chronological there.
    - A session with an odd number of trials puts the extra trial in the
      second half.

The two fits use disjoint data, so the posterior of (second - first) for any
quantity is computed by pairing draws of the two independent posteriors.

Every figure is saved to Plots_Learning_A under the names used below; the
numbers are printed at the end and stored in Results_Learning_A.pkl.

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
    LEARN_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    LEARN_DIR = Path.cwd()
MODEL_DIR = LEARN_DIR.parent           # Model_A
DATA_DIR = MODEL_DIR / "Data_A"
PLOTS_DIR = LEARN_DIR / "Plots_Learning_A"
SAVE_FIGS = True

def save_fig(fig, name):
    if SAVE_FIGS:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / name, dpi=200, bbox_inches='tight')

SEED = 20261104
DRAWS = 5000
CHAINS = 4
CORES = 4
# Python 3.14's default 'forkserver' re-runs this script in every worker, so fork them instead
# ('fork' doesn't exist on Windows, which keeps its default)
MP_CTX = None if os.name == 'nt' else 'fork'

#%% Load data and split each session in half

with open(DATA_DIR / "ReadyData_Sirius_A.pkl", "rb") as f:
    data_dict = pickle.load(f)
cov_all = data_dict['cov_mat']
grp_all = data_dict['grp_idx']
resp_all = data_dict['resp']

with open(MODEL_DIR.parent / "Sirius_Data.pkl", "rb") as f:
    raw_data = pickle.load(f)
raw = raw_data['data']
assert np.array_equal(raw['response'].to_numpy(), resp_all)
assert np.array_equal(raw['group_idx'].to_numpy(), grp_all)
sess_all = raw['sess_idx'].to_numpy()
sessions = raw_data['session_summary'].sort_values('sess_idx').index.tolist()

# each session's trials are contiguous, and start_time increases within a session except at the
# known join of the two 04-14 recordings
assert np.all(np.diff(sess_all) >= 0)
restarts = [(s, i) for s, g in raw.groupby('sess_idx') for i in np.where(np.diff(g['start_time'].to_numpy()) <= 0)[0]]
assert [sessions[s] for s, _ in restarts] == ['04-14'], restarts

# position of each trial within its session, and the session's trial count
pos_in_sess = raw.groupby('sess_idx').cumcount().to_numpy()
n_in_sess = raw.groupby('sess_idx')['sess_idx'].transform('size').to_numpy()
in_first = pos_in_sess < n_in_sess // 2     # odd count: the extra trial goes to the second half

HALVES = ['first', 'second']
HALF_LABELS = {'first': 'First half', 'second': 'Second half'}
HALF_COLORS = {'first': '#2a78d6', 'second': '#eb6834'}
half_mask = {'first': in_first, 'second': ~in_first}

#%% Fit Model A to each half

traces = {}
sampling_minutes = {}
for h in HALVES:
    m = half_mask[h]
    cov_mat, grp_idx, obs_data = cov_all[m], grp_all[m], resp_all[m]
    exec(open(MODEL_DIR / "Build_Model_A.py").read())
    start = time.time()
    with model_A:
        traces[h] = pm.sample(draws=DRAWS, chains=CHAINS, cores=CORES, mp_ctx=MP_CTX, random_seed=SEED,
                              return_inferencedata=True, progressbar=True)
        pm.sample_posterior_predictive(traces[h], extend_inferencedata=True, random_seed=SEED)
    sampling_minutes[h] = (time.time() - start) / 60
    print(f"FINISHED {h} half ({sampling_minutes[h]:.1f} min)")
    print(az.summary(traces[h], var_names=['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND']).to_string())

groups = coords['groups']

#%% Shared setup

x_levels_um = np.unique(raw['stim_amp'])
x_mu, x_sig = np.mean(x_levels_um), np.std(x_levels_um)
amp_all = np.round(cov_all[:, 1] * x_sig + x_mu).astype(int)
levels = np.unique(amp_all)

QUANTITIES = ['PSE', 'JND', 'gam_h', 'gam_l']
QUANTITY_LABELS = {'PSE': r'PSE ($\mu m$)', 'JND': r'JND ($\mu m$)', 'gam_h': r'$\gamma_h$', 'gam_l': r'$\gamma_l$'}
FILE_LABELS = {'PSE': 'PSE', 'JND': 'JND', 'gam_h': 'gamh', 'gam_l': 'gaml'}
SCALE = {'PSE': x_sig, 'JND': x_sig, 'gam_h': 1.0, 'gam_l': 1.0}
SHIFT = {'PSE': x_mu, 'JND': 0.0, 'gam_h': 0.0, 'gam_l': 0.0}
GRAY = '#52514e'

def draws_of(h, g, q):
    """Posterior draws of quantity q for group g in half h, in micrometres for PSE and JND."""
    return traces[h].posterior[q].sel(groups=g).values.reshape(-1) * SCALE[q] + SHIFT[q]

def plot_post_hdi(ax, vals, color, label, hdi_prob=0.95):
    """Posterior density with its HDI shaded and a dotted line at the posterior mean."""
    grid, dens = az.kde(vals)
    lo, hi = az.hdi(vals, hdi_prob=hdi_prob)
    inside = (grid >= lo) & (grid <= hi)
    ax.plot(grid, dens, color=color, linewidth=2, label=label)
    ax.fill_between(grid[inside], dens[inside], color=color, alpha=0.3, linewidth=0)
    ax.vlines(vals.mean(), 0, np.interp(vals.mean(), grid, dens), color=color, linestyles='dotted')

def mean_hdi(vals, fmt='.2f'):
    lo, hi = az.hdi(vals, hdi_prob=0.95)
    return f"{vals.mean():{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"

def fmt_of(q):
    return '.3f' if q.startswith('gam') else '.2f'

def psychfunc(gam_h, gam_l, b0, b1, X):
    return gam_h + (1 - gam_h - gam_l) / (1 + np.exp(-(b0 + b1 * X)))

#%% Data only: proportion correct in each half, by group and session

correct = raw['correct'].to_numpy().astype(float)
acc = pd.DataFrame({'session': sess_all, 'group': np.array(groups)[grp_all],
                    'half': np.where(in_first, 'first', 'second'), 'correct': correct})
acc_by_group = acc.groupby(['group', 'half'])['correct'].mean().unstack('half').loc[groups]
acc_by_sess = acc.groupby(['session', 'half'])['correct'].mean().unstack('half')

fig, ax = plt.subplots(constrained_layout=True, figsize=(9, 4))
ax.plot(acc_by_sess.index, acc_by_sess['first'], 'o-', color=HALF_COLORS['first'], markersize=5,
        linewidth=1, label=HALF_LABELS['first'])
ax.plot(acc_by_sess.index, acc_by_sess['second'], 'o-', color=HALF_COLORS['second'], markersize=5,
        linewidth=1, label=HALF_LABELS['second'])
ax.set_xticks(acc_by_sess.index[::3])
ax.set_xticklabels([sessions[i] for i in acc_by_sess.index[::3]], rotation=45)
ax.set_xlabel('session')
ax.set_ylabel('proportion correct')
ax.grid(alpha=0.3)
ax.legend(fontsize='small')
ax.set_title('Proportion correct in each half of each session (all groups)')
save_fig(fig, 'accuracy_by_session.png')
plt.show()

#%% Psychometric curves: first vs second half, one panel per group

xfit = np.linspace(-1.6, 1.6, 400)
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(10, 7))
for ax, g in zip(axes.ravel(), groups):
    gi = groups.index(g)
    for h in HALVES:
        post = traces[h].posterior
        pars = [post[v].sel(groups=g).values.reshape(-1)[:, None] for v in ('gam_h', 'gam_l')]
        bet = [post['beta_vec'].sel(groups=g, betas=b).values.reshape(-1)[:, None] for b in ('b0', 'b1')]
        ys = psychfunc(*pars, *bet, xfit)
        hdi = az.hdi(ys[None], hdi_prob=0.95)
        ax.plot(xfit * x_sig + x_mu, ys.mean(0), color=HALF_COLORS[h], linewidth=2, label=HALF_LABELS[h])
        ax.fill_between(xfit * x_sig + x_mu, hdi[:, 0], hdi[:, 1], color=HALF_COLORS[h], alpha=0.25, linewidth=0)
        m = half_mask[h] & (grp_all == gi)
        obs = [resp_all[m & (amp_all == a)].mean() for a in levels]
        ax.plot(levels, obs, 'o', color=HALF_COLORS[h], markersize=6, markeredgecolor='white', markeredgewidth=1)
    ax.axhline(0.5, color=GRAY, linewidth=0.5)
    ax.axvline(28, linestyle='--', linewidth=1, color='green')
    ax.set_title(g)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
for ax in axes[:, 0]:
    ax.set_ylabel('P(response = "high")')
axes[0, 0].legend(fontsize='small', loc='upper left', title='mean curve, 95% HDI; dots = data', title_fontsize='x-small')
fig.suptitle('Model A psychometric curves: first and second half of each session', fontsize=13)
save_fig(fig, 'curves_halves.png')
plt.show()

#%% Posterior densities: first vs second half, every quantity and group

fig, axes = plt.subplots(len(QUANTITIES), len(groups), constrained_layout=True, figsize=(14, 11))
for r, q in enumerate(QUANTITIES):
    for ax, g in zip(axes[r], groups):
        for h in HALVES:
            plot_post_hdi(ax, draws_of(h, g, q), HALF_COLORS[h], HALF_LABELS[h])
        if q == 'PSE':
            ax.axvline(28, linestyle='--', linewidth=1, color='green')
        ax.set_yticks([])
        ax.set_xlabel(QUANTITY_LABELS[q])
        if r == 0:
            ax.set_title(g)
axes[0, 0].legend(fontsize='small', title='95% HDI shaded, mean dotted', title_fontsize='x-small')
fig.suptitle('Model A posteriors: first and second half of each session', fontsize=14)
save_fig(fig, 'posteriors_halves.png')
plt.show()

#%% Change within session (second - first), pairing draws of the two independent fits

change = {(g, q): draws_of('second', g, q) - draws_of('first', g, q) for g in groups for q in QUANTITIES}

fig, axes = plt.subplots(len(QUANTITIES), len(groups), constrained_layout=True, figsize=(14, 11))
for r, q in enumerate(QUANTITIES):
    for ax, g in zip(axes[r], groups):
        d = change[(g, q)]
        plot_post_hdi(ax, d, HALF_COLORS['first'], f'P(>0) = {np.mean(d > 0):.2f}')
        ax.axvline(0, linestyle='--', linewidth=1, color=GRAY)
        ax.set_yticks([])
        ax.set_xlabel(f'second − first {QUANTITY_LABELS[q]}')
        ax.legend(fontsize='small')
        if r == 0:
            ax.set_title(g)
fig.suptitle('Change within session (second half − first half), Model A', fontsize=14)
save_fig(fig, 'change_densities.png')
plt.show()

change_rows = []
for g in groups:
    for q in QUANTITIES:
        d = change[(g, q)]
        lo, hi = az.hdi(d, hdi_prob=0.95)
        change_rows.append({'group': g, 'quantity': q, 'mean': d.mean(), 'hdi_low': lo, 'hdi_high': hi,
                            'P(>0)': np.mean(d > 0)})
change_summary = pd.DataFrame(change_rows)

#%% Distractor effect (bimanual - unimanual) in each half

HANDS = {'Left Hand': ('left_uni', 'left_bi'), 'Right Hand': ('right_uni', 'right_bi')}
effect_rows = []
for h in HALVES:
    for hand, (uni, bi) in HANDS.items():
        for q in QUANTITIES:
            d = draws_of(h, bi, q) - draws_of(h, uni, q)
            lo, hi = az.hdi(d, hdi_prob=0.95)
            effect_rows.append({'half': h, 'hand': hand, 'quantity': q, 'mean': d.mean(), 'hdi_low': lo,
                                'hdi_high': hi, 'P(>0)': np.mean(d > 0)})
effect_summary = pd.DataFrame(effect_rows)

fig, axes = plt.subplots(1, len(QUANTITIES), constrained_layout=True, figsize=(14, 3.5))
for ax, q in zip(axes, QUANTITIES):
    sub = effect_summary[effect_summary['quantity'] == q]
    for j, hand in enumerate(HANDS):
        for k, h in enumerate(HALVES):
            row = sub[(sub['hand'] == hand) & (sub['half'] == h)].iloc[0]
            y = j + (0.15 if k == 0 else -0.15)
            ax.hlines(y, row['hdi_low'], row['hdi_high'], color=HALF_COLORS[h], linewidth=2)
            ax.plot(row['mean'], y, 'o', color=HALF_COLORS[h], markersize=8,
                    label=HALF_LABELS[h] if j == 0 else None)
    ax.axvline(0, linestyle='--', linewidth=1, color=GRAY)
    ax.set_yticks(range(len(HANDS)))
    ax.set_yticklabels(list(HANDS) if q == QUANTITIES[0] else [])
    ax.set_ylim(-0.6, len(HANDS) - 0.4)
    ax.set_xlabel(f'bimanual − unimanual {QUANTITY_LABELS[q]}')
    ax.grid(axis='x', alpha=0.3)
axes[0].legend(fontsize='small', loc='lower right', title='mean, 95% HDI', title_fontsize='x-small')
fig.suptitle('Effect of the distractor in each half of the session, Model A', fontsize=13)
save_fig(fig, 'effect_halves.png')
plt.show()

#%% Posterior predictive check in each half, sessions pooled

fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, constrained_layout=True, figsize=(15, 6))
n_inside = {}
for row_axes, h in zip(axes, HALVES):
    m = half_mask[h]
    grp_h, resp_h, amp_h = grp_all[m], resp_all[m], amp_all[m]
    y_rep = traces[h].posterior_predictive['resp'].values.reshape(-1, m.sum()).astype(np.int8)
    n_inside[h] = 0
    for ax, g in zip(row_axes, groups):
        in_grp = grp_h == groups.index(g)
        obs = np.array([resp_h[in_grp & (amp_h == a)].mean() for a in levels])
        rep = np.array([y_rep[:, in_grp & (amp_h == a)].mean(1) for a in levels])
        lo, med, hi = np.quantile(rep, [0.025, 0.5, 0.975], axis=1)
        n_inside[h] += int(np.sum((lo <= obs) & (obs <= hi)))
        ax.vlines(levels, lo, hi, color=HALF_COLORS[h], linewidth=6, alpha=0.35, label='predictive 95% interval')
        ax.plot(levels, med, '_', color=HALF_COLORS[h], markersize=12, markeredgewidth=2, label='predictive median')
        ax.plot(levels, obs, 'o', color='#0b0b0b', markersize=5, label='observed')
        ax.axvline(28, linestyle='--', linewidth=1, color=GRAY)
        ax.set_title(f'{g}, {h} half')
        ax.grid(alpha=0.3)
    row_axes[0].set_ylabel('P(response = "high")')
    row_axes[0].legend(fontsize=7, loc='upper left')
    del y_rep
for ax in axes[1]:
    ax.set_xlabel(r'stimulus amplitude ($\mu m$)')
fig.suptitle('Posterior predictive check in each half, sessions pooled', fontsize=13)
save_fig(fig, 'ppc_halves.png')
plt.show()

#%% Trace plots, one figure per half

for h in HALVES:
    axes = az.plot_trace(traces[h], var_names=('gam_h', 'gam_l', 'beta_vec'), compact=True, legend=True,
                         backend_kwargs={"constrained_layout": True, "figsize": (13, 7)})
    for ax_dens, ax_draws in axes:
        leg = ax_dens.get_legend()
        if leg is not None:
            handles, labels, title = leg.legend_handles, [t.get_text() for t in leg.get_texts()], leg.get_title().get_text()
            leg.remove()
            ax_draws.legend(handles, labels, title=title, loc='center left', bbox_to_anchor=(1.01, 0.5),
                            fontsize='small', title_fontsize='small')
    axes.ravel()[0].figure.suptitle(f'Model A trace plots, {h} half')
    save_fig(axes.ravel()[0].figure, f'trace_{h}.png')
    plt.show()

#%% Numbers

print("--- Data split")
print(f"{len(resp_all)} trials, {len(sessions)} sessions")
for h in HALVES:
    m = half_mask[h]
    print(f"{h} half: {m.sum()} trials; per group {dict(zip(groups, np.bincount(grp_all[m], minlength=4).tolist()))}")
print("trials per stimulus level (first, second):")
print(pd.crosstab(amp_all, np.where(in_first, 'first', 'second')).to_string())
print("proportion correct by group:")
print(acc_by_group.round(3).to_string())
d_acc = acc_by_sess['second'] - acc_by_sess['first']
print(f"per session, second - first proportion correct: mean {d_acc.mean():.3f}, "
      f"{int((d_acc > 0).sum())} of {len(d_acc)} sessions higher in the second half")

print("--- Sampling diagnostics")
for h in HALVES:
    t = traces[h]
    diag = az.summary(t, kind='diagnostics')
    print(f"{h}: {t.posterior.sizes['chain']} chains x {t.posterior.sizes['draw']} draws, "
          f"{sampling_minutes[h]:.1f} min; divergences {int(t.sample_stats['diverging'].sum())}, "
          f"max tree depth {int(t.sample_stats['tree_depth'].max())}; max r_hat {diag['r_hat'].max():.3f}, "
          f"min bulk ESS {diag['ess_bulk'].min():.0f}, min tail ESS {diag['ess_tail'].min():.0f}")

print("--- Group estimates, posterior mean [95% HDI]")
estimates = {}
for h in HALVES:
    estimates[h] = pd.DataFrame({q: {g: mean_hdi(draws_of(h, g, q), fmt_of(q)) for g in groups} for q in QUANTITIES})
    print(f"{h} half:")
    print(estimates[h].to_string())

print("--- Change within session, second - first")
for _, row in change_summary.iterrows():
    f = fmt_of(row['quantity'])
    print(f"{row['group']} {row['quantity']}: {row['mean']:{f}} [{row['hdi_low']:{f}}, {row['hdi_high']:{f}}], "
          f"P(>0) = {row['P(>0)']:.3f}")

print("--- Distractor effect (bimanual - unimanual) in each half")
for _, row in effect_summary.iterrows():
    f = fmt_of(row['quantity'])
    print(f"{row['half']} {row['hand']} {row['quantity']}: {row['mean']:{f}} [{row['hdi_low']:{f}}, "
          f"{row['hdi_high']:{f}}], P(>0) = {row['P(>0)']:.3f}")

print("--- Posterior predictive check")
for h in HALVES:
    print(f"{h} half: {n_inside[h]} of {len(groups) * len(levels)} group x amplitude proportions inside 95% interval")

#%% Save results

learning_results = {'az_summary': {h: az.summary(traces[h], var_names=['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND'])
                                   for h in HALVES},
                    'estimates': estimates,
                    'change_summary': change_summary,
                    'effect_summary': effect_summary,
                    'accuracy_by_group': acc_by_group,
                    'accuracy_by_session': acc_by_sess,
                    'in_first_half': in_first}
with open(LEARN_DIR / "Results_Learning_A.pkl", "wb") as f:
    pickle.dump(learning_results, f)
