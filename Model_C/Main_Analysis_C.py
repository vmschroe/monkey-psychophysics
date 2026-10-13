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

#%% Posterior predictive check: observed vs predicted P("high") by hand and condition, sessions pooled

with model_C:
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=SEED)
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

#%% Distractor effect: hand-level mu_b2 and the PSE shift of a typical session at each amplitude

post = trace.posterior
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
               'ppc_table': ppc_table}
with open(MODEL_DIR / "Results_C.pkl", "wb") as f:
    pickle.dump(fit_results, f)
