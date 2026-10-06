#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 19:04:20 2026

Fit Model A (one psychometric curve per group, all sessions pooled) to the real
data. Every figure and number in Fit_A_Report.txt is produced by this script;
figures are saved to Plots_A/Fit_Plots_A under the names the report uses.

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
DATA_DIR = MODEL_DIR / "Data_A"

# Figures used in Fit_A_Report.txt are saved here under the names the report uses
FIT_PLOTS_DIR = MODEL_DIR / "Plots_A" / "Fit_Plots_A"
SAVE_FIGS = True

def save_fig(fig, name):
    if SAVE_FIGS:
        FIT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIT_PLOTS_DIR / name, dpi=200, bbox_inches='tight')

SEED = 20261006
DRAWS = 5000
CHAINS = 4
CORES = 4
# Python 3.14's default 'forkserver' re-runs this script in every worker, so fork them instead
# ('fork' doesn't exist on Windows, which keeps its default)
MP_CTX = None if os.name == 'nt' else 'fork'

#%%
#load and unpack data
with open(DATA_DIR / "ReadyData_Sirius_A.pkl", "rb") as f:
    data_dict = pickle.load(f)

cov_mat = data_dict['cov_mat']
grp_idx = data_dict['grp_idx']
obs_data = data_dict['resp']

# Model A ignores sessions, but Model B's data file has the same trials in the same order plus the
# session of each trial; it is used only to check how well Model A fits single sessions
with open(MODEL_DIR.parent / "Model_B" / "Data_B" / "ReadyData_Sirius_B.pkl", "rb") as f:
    data_dict_B = pickle.load(f)
assert all(np.array_equal(data_dict[k], data_dict_B[k]) for k in ['cov_mat', 'grp_idx', 'resp'])
sess_idx = data_dict_B['sess_idx']
sessions = list(data_dict_B['dates_sess_idx'])

#%%

exec(open(MODEL_DIR / "Build_Model_A.py").read())

#%% Sample from posteriors

sampling_start = time.time()
with model_A:
    trace = pm.sample(draws=DRAWS, chains=CHAINS, cores=CORES, mp_ctx=MP_CTX, random_seed=SEED,
                      return_inferencedata=True, progressbar=True, idata_kwargs={"log_likelihood": True})
sampling_minutes = (time.time() - sampling_start) / 60
print(f"FINISHED SAMPLING! ({sampling_minutes:.1f} min)")


#%% Look at r_hats and effective sample sizes

result_df = az.summary(trace, var_names = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND'])
print(result_df.to_string())

#%% Look at traceplots (all groups overlaid)

axes = az.plot_trace(trace, var_names=('gam_h', 'gam_l', 'beta_vec'), compact=True, legend=True,
                     backend_kwargs={"constrained_layout": True, "figsize": (13, 7)})
# move each group legend to the right of its row, where it can't cover the curves or the title
for ax_dens, ax_draws in axes:
    leg = ax_dens.get_legend()
    if leg is not None:
        handles, labels, title = leg.legend_handles, [t.get_text() for t in leg.get_texts()], leg.get_title().get_text()
        leg.remove()
        ax_draws.legend(handles, labels, title=title, loc='center left', bbox_to_anchor=(1.01, 0.5),
                        fontsize='small', title_fontsize='small')
save_fig(axes.ravel()[0].figure, 'trace_all_groups.png')
plt.show()

#%% plot joint posteriors

for grp_num, grp_choice in enumerate(coords['groups']):
     axes = az.plot_pair(trace, var_names=['gam_h', 'gam_l'
                                    #,'beta_vec'
                                    ,'PSE', 'JND'
                                    ],
             coords = {'betas': ["b0", "b1"], 'groups': [grp_choice]},
             kind = 'kde', marginals=True)
     save_fig(axes.ravel()[0].figure, f'pair_{grp_choice}.png')
     plt.show()

#%%


param_samps = trace.posterior[['beta_vec', 'gam_h', 'gam_l']]

gam_h_samps = {}
gam_l_samps = {}
beta_0_samps = {}
beta_1_samps = {}


for grp in ["left_bi","left_uni","right_bi","right_uni"]:
    gam_h_samps[grp] = param_samps['gam_h'].sel(groups = grp).values.flatten()
    gam_l_samps[grp] = param_samps['gam_l'].sel(groups = grp).values.flatten()
    beta_0_samps[grp] = param_samps['beta_vec'].sel(groups = grp, betas='b0').values.flatten()
    beta_1_samps[grp] = param_samps['beta_vec'].sel(groups = grp, betas='b1').values.flatten()

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
freq_df = pd.DataFrame({'stim': cov_mat[:,1], 'grp_idx': grp_idx, 'obs_data': obs_data})
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
    # one curve per posterior draw, shape (draws, len(xfit))
    y_samples[grp] = psychfunc([v[:, None] for v in (gam_h_samps[grp], gam_l_samps[grp],
                                                      beta_0_samps[grp], beta_1_samps[grp])], xfit)
    hdis[grp] = az.hdi(y_samples[grp][None], hdi_prob=0.95)
    rec_params[grp] = np.mean(np.array([gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp]]), axis = 1)


    yrec[grp] = psychfunc(rec_params[grp], xfit)

#%%
x_old = [6,12,18,24,32,38,44,50]
x_mu = np.mean(x_old)
x_sig = np.std(x_old)

#%% Psychometric curves for each hand

for hand, (uni, bi), (uni_i, bi_i) in [('Left', ('left_uni', 'left_bi'), (0, 1)),
                                      ('Right', ('right_uni', 'right_bi'), (2, 3))]:
    plt.figure()
    plt.plot(xfit*x_sig+x_mu,yrec[uni],label='Unimanual',color='blue')
    plt.fill_between(xfit*x_sig+x_mu, hdis[uni][:, 0], hdis[uni][:, 1], color='blue', alpha=0.3, label='95% HDI')
    plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[uni_i]),label='Data', color = 'blue')
    plt.plot(xfit*x_sig+x_mu,yrec[bi],label='Bimanual',color='red')
    plt.fill_between(xfit*x_sig+x_mu, hdis[bi][:, 0], hdis[bi][:, 1], color='red', alpha=0.3, label='95% HDI')
    plt.scatter(np.array(freqs.index)*x_sig+x_mu,np.array(freqs[bi_i]),label='Data', color = 'red')
    plt.hlines(y= 0.5, xmin=0, xmax=51, colors='gray',lw=0.5)
    plt.vlines(x=28, ymin=-0.05,ymax=1.05, label="trained threshold", color='green', linestyles='dashed')
    plt.xlim(5,51)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(r'Stimulus Amplitude ($\mu m$)')
    plt.ylabel('Prob[response = "high"]')
    plt.legend(loc='upper left', fontsize=9.5)
    plt.title(f"{hand} Hand Psychometric Curves")
    save_fig(plt.gcf(), f'curves_{hand.lower()}.png')
    plt.show()


#%% sample priors and post pred

with model_A:
    pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=SEED)
    prior = pm.sample_prior_predictive(draws=3000, random_seed=SEED)
trace.extend(prior)

#%% Shared setup for the comparisons below

groups = coords['groups']
HANDS = {'Left Hand': ('left_uni', 'left_bi'), 'Right Hand': ('right_uni', 'right_bi')}
QUANTITIES = ['PSE', 'JND', 'gam_h', 'gam_l']
QUANTITY_LABELS = {'PSE': r'PSE ($\mu m$)', 'JND': r'JND ($\mu m$)', 'gam_h': r'$\gamma_h$', 'gam_l': r'$\gamma_l$'}
FILE_LABELS = {'PSE': 'PSE', 'JND': 'JND', 'gam_h': 'gamh', 'gam_l': 'gaml'}   # quantity names in figure files
# normalized stimulus units -> micrometres; differences are only scaled, never shifted
SCALE = {'PSE': x_sig, 'JND': x_sig, 'gam_h': 1.0, 'gam_l': 1.0}
SHIFT = {'PSE': x_mu, 'JND': 0.0, 'gam_h': 0.0, 'gam_l': 0.0}
COLORS = {'posterior': '#2a78d6', 'prior': '#52514e'}
COND_COLORS = {'Unimanual': '#2a78d6', 'Bimanual': '#e34948'}

def finite(vals):
    vals = np.asarray(vals).reshape(-1)
    return vals[np.isfinite(vals)]

def draws_of(group_name, q, ds=None):
    """Draws of quantity q for one group, in micrometres for PSE and JND."""
    ds = trace.posterior if ds is None else ds
    return finite(ds[q].sel(groups=group_name).values * SCALE[q] + SHIFT[q])

def plot_post_hdi(ax, vals, color, label, hdi_prob=0.95):
    """Posterior density with its HDI shaded and a dotted line at the posterior mean."""
    grid, dens = az.kde(vals)
    hdi_low, hdi_high = az.hdi(vals, hdi_prob=hdi_prob)
    in_hdi = (grid >= hdi_low) & (grid <= hdi_high)
    ax.plot(grid, dens, color=color, label=label)
    ax.fill_between(grid[in_hdi], dens[in_hdi], color=color, alpha=0.3, linewidth=0)
    ax.vlines(vals.mean(), 0, np.interp(vals.mean(), grid, dens), color=color, linestyles='dotted')

def mean_hdi(vals, fmt='.2f'):
    vals = finite(vals)
    low, high = az.hdi(vals, hdi_prob=0.95)
    return f"{vals.mean():{fmt}} [{low:{fmt}}, {high:{fmt}}]"

#%% compare prior and posteriors for parameters

for q in QUANTITIES:
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(9, 6))
    for ax, g in zip(axes.ravel(), groups):
        post_vals = draws_of(g, q)
        prior_vals = draws_of(g, q, trace.prior)
        # the prior is far wider than the posterior, so show it as a histogram over the posterior's range
        lo, hi = np.quantile(post_vals, [0.001, 0.999])
        bins = np.linspace(lo - (hi - lo), hi + (hi - lo), 60)
        ax.hist(prior_vals, bins=bins, histtype='step', color=COLORS['prior'], label='prior',
                weights=np.full(len(prior_vals), 1 / (len(prior_vals) * np.diff(bins)[0])))
        plot_post_hdi(ax, post_vals, COLORS['posterior'], 'posterior (95% HDI shaded)')
        if q == 'PSE':
            ax.axvline(28, linestyle="--", linewidth=1.5, label="trained threshold", color='green')
        ax.set_xlim(bins[0], bins[-1])
        ax.set_yticks([])
        ax.set_title(g)
        ax.set_xlabel(QUANTITY_LABELS[q])
    axes[0, 0].legend(fontsize='small')
    fig.suptitle(f"Prior and posterior: {QUANTITY_LABELS[q]}", fontsize=14)
    save_fig(fig, f'prior_post_{FILE_LABELS[q]}.png')
    plt.show()

#%% compare unimanual vs bimanual posteriors, every quantity and hand

fig, axes = plt.subplots(len(QUANTITIES), len(HANDS), constrained_layout=True, figsize=(11, 12))
for r, q in enumerate(QUANTITIES):
    for ax, (hand, (uni, bi)) in zip(axes[r], HANDS.items()):
        for cond, g in [('Unimanual', uni), ('Bimanual', bi)]:
            plot_post_hdi(ax, draws_of(g, q), COND_COLORS[cond], cond)
        if q == 'PSE':
            ax.axvline(28, linestyle="--", linewidth=1.5, label="trained threshold", color='green')
        ax.set_yticks([])
        ax.set_xlabel(QUANTITY_LABELS[q])
        if r == 0:
            ax.set_title(hand)
            ax.legend(fontsize='small', title='95% HDI shaded, mean dotted', title_fontsize='x-small')
fig.suptitle("Posterior estimates: unimanual and bimanual", fontsize=14)
save_fig(fig, 'uni_bi_posteriors.png')
plt.show()

#%% Effect of the distractor (bimanual - unimanual), computed per posterior draw

effect = {(hand, q): finite((trace.posterior[q].sel(groups=bi) - trace.posterior[q].sel(groups=uni)).values * SCALE[q])
          for hand, (uni, bi) in HANDS.items() for q in QUANTITIES}

fig, axes = plt.subplots(len(QUANTITIES), len(HANDS), constrained_layout=True, figsize=(11, 11))
for r, q in enumerate(QUANTITIES):
    for ax, hand in zip(axes[r], HANDS):
        d = effect[(hand, q)]
        plot_post_hdi(ax, d, COLORS['posterior'], f'P(>0) = {np.mean(d > 0):.2f}')
        ax.axvline(0, linestyle='--', linewidth=1, color=COLORS['prior'])
        ax.set_yticks([])
        ax.set_xlabel(f'bimanual − unimanual {QUANTITY_LABELS[q]}')
        ax.legend(fontsize='small', title='95% HDI shaded, mean dotted', title_fontsize='x-small')
        if r == 0:
            ax.set_title(hand)
fig.suptitle("Effect of the distractor (bimanual − unimanual), Model A", fontsize=14)
save_fig(fig, 'effect_densities.png')
plt.show()

effect_rows = []
for hand in HANDS:
    for q in QUANTITIES:
        d = effect[(hand, q)]
        low, high = az.hdi(d, hdi_prob=0.95)
        effect_rows.append({'hand': hand, 'quantity': q, 'mean': d.mean(), 'hdi_low': low, 'hdi_high': high,
                            'P(>0)': np.mean(d > 0)})
effect_summary = pd.DataFrame(effect_rows)
print(effect_summary.round(3).to_string())

#%% Posterior predictive check: observed vs predicted P("high"), all sessions pooled

amp = np.round(cov_mat[:, 1] * x_sig + x_mu).astype(int)   # stimulus amplitude in micrometres
levels = np.unique(amp)
y_rep = trace.posterior_predictive['resp'].values.reshape(-1, len(obs_data)).astype(np.int8)   # (draws, trials)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(9, 6))
n_pooled_inside = 0
for ax, g in zip(axes.ravel(), groups):
    in_grp = grp_idx == groups.index(g)
    obs = np.array([obs_data[in_grp & (amp == a)].mean() for a in levels])
    rep = np.array([y_rep[:, in_grp & (amp == a)].mean(1) for a in levels])   # (levels, draws)
    lo, med, hi = np.quantile(rep, [0.025, 0.5, 0.975], axis=1)
    n_pooled_inside += int(np.sum((lo <= obs) & (obs <= hi)))
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

#%% Posterior predictive check within single sessions

# Model A predicts the same curve in every session. If sessions really differ, the observed counts of
# "high" in each session x group x amplitude cell are more spread out than Model A predicts: too many
# cells fall outside their predictive intervals, and the PIT values pile up near 0 and 1.
pit_rng = np.random.default_rng(SEED)
cell_trials = pd.DataFrame({'s': sess_idx, 'g': grp_idx, 'a': amp, 'i': np.arange(len(obs_data))}
                           ).groupby(['s', 'g', 'a'])['i'].apply(np.array)
cell_rows = []
for (s, g, a), idx in cell_trials.items():
    k_obs = obs_data[idx].sum()
    k_rep = y_rep[:, idx].sum(1)
    low, high = np.quantile(k_rep, [0.025, 0.975])
    # randomized PIT, uniform on (0, 1) when the model is calibrated
    pit = np.mean(k_rep < k_obs) + pit_rng.uniform() * np.mean(k_rep == k_obs)
    cell_rows.append({'session': s, 'group': groups[g], 'amp': a, 'n_trials': len(idx), 'k_obs': k_obs,
                      'inside': low <= k_obs <= high, 'pit': pit})
session_cells = pd.DataFrame(cell_rows)

fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(11, 4))
axes[0].hist(session_cells['pit'], bins=20, range=(0, 1), color='#2a78d6', alpha=0.7, density=True)
axes[0].axhline(1, linestyle='--', color=COLORS['prior'], label='calibrated model')
axes[0].set_xlabel('PIT of the observed count')
axes[0].set_ylabel('density')
axes[0].set_title(f'all {len(session_cells)} session × group × amplitude cells')
axes[0].legend(fontsize='small')
inside_by_amp = session_cells.groupby(['group', 'amp'])['inside'].mean().unstack('group')
for g in groups:
    axes[1].plot(inside_by_amp.index, inside_by_amp[g], marker='o', label=g)
axes[1].axhline(0.95, linestyle='--', color=COLORS['prior'], label='nominal 0.95')
axes[1].axvline(28, linestyle='--', linewidth=1, color='green')
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel(r'stimulus amplitude ($\mu m$)')
axes[1].set_ylabel('fraction of sessions inside 95% interval')
axes[1].legend(fontsize='small')
axes[1].grid(alpha=0.3)
fig.suptitle('Posterior predictive check within single sessions (Model A pools the sessions)', fontsize=13)
save_fig(fig, 'ppc_session_cells.png')
plt.show()

#%% Compare with Model B: leave-one-out cross-validation and the distractor effect

ax = az.plot_ppc(trace, num_pp_samples=100, random_seed=0)
save_fig(np.ravel(ax)[0].figure, 'ppc_arviz_resp.png')
plt.show()

LOO_results = az.loo(trace, pointwise=True)

with open(MODEL_DIR.parent / "Model_B" / "Results_B.pkl", "rb") as f:
    fit_results_B = pickle.load(f)   # written by Model_B/Main_Analysis_B.py
loo_B = fit_results_B['az_loo_trace']
elpd_diff_i = loo_B.loo_i.values - LOO_results.loo_i.values   # same trials in the same order
elpd_diff = elpd_diff_i.sum()
elpd_diff_se = np.sqrt(len(elpd_diff_i) * np.var(elpd_diff_i))

# effect of the distractor: Model A (pooled) next to Model B (average over sessions, typical session)
effect_B = fit_results_B['effect_summary'].set_index(['hand', 'quantity'])
ESTIMATES = [('Model A (pooled)', '#0b0b0b'), ('Model B: average over sessions', '#eb6834'),
             ('Model B: typical session', '#2a78d6')]
fig, axes = plt.subplots(len(HANDS), len(QUANTITIES), constrained_layout=True, figsize=(13, 5))
for r, hand in enumerate(HANDS):
    for ax, q in zip(axes[r], QUANTITIES):
        a = effect_summary.set_index(['hand', 'quantity']).loc[(hand, q)]
        b = effect_B.loc[(hand, q)]
        rows = [(a['mean'], a['hdi_low'], a['hdi_high']),
                (b['sess_avg_mean'], b['sess_avg_hdi_low'], b['sess_avg_hdi_high']),
                (b['pop_mean'], b['pop_hdi_low'], b['pop_hdi_high'])]
        for y, ((m, low, high), (label, color)) in enumerate(zip(rows, ESTIMATES)):
            ax.errorbar(m, -y, xerr=[[m - low], [high - m]], fmt='o', color=color, capsize=3, label=label)
        ax.axvline(0, linestyle='--', linewidth=1, color=COLORS['prior'])
        ax.set_yticks([])
        ax.set_ylim(-2.6, 0.6)
        ax.set_xlabel(f'bimanual − unimanual\n{QUANTITY_LABELS[q]}')
        ax.grid(alpha=0.3, axis='x')
        if q == QUANTITIES[0]:
            ax.set_ylabel(hand)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=3)
fig.suptitle('Effect of the distractor: Model A and Model B (posterior mean and 95% HDI)', fontsize=13)
save_fig(fig, 'effect_A_vs_B.png')
plt.show()

#%% Numbers for Fit_A_Report.txt
# Everything quoted in the report that is not printed by an earlier cell.

print("--- Data and fit")
print(f"{len(obs_data)} trials, {len(sessions)} sessions; trials per group: "
      f"{dict(zip(groups, np.bincount(grp_idx).tolist()))}")
print(f"stimulus scale: PSE_um = {x_mu:.0f} + {x_sig:.2f} PSE, JND_um = {x_sig:.2f} JND")
print(f"{trace.posterior.sizes['chain']} chains x {trace.posterior.sizes['draw']} draws, "
      f"{trace.posterior.attrs.get('tuning_steps', '?')} tuning steps"
      + (f", {sampling_minutes:.1f} min" if 'sampling_minutes' in globals() else ""))

print("--- Sampling diagnostics")
full_summary = az.summary(trace, kind='diagnostics')
print(f"divergences: {int(trace.sample_stats['diverging'].sum())}, "
      f"max tree depth: {int(trace.sample_stats['tree_depth'].max())}")
print(f"{len(full_summary)} parameters: max r_hat {full_summary['r_hat'].max():.3f}, "
      f"min bulk ESS {full_summary['ess_bulk'].min():.0f}, min tail ESS {full_summary['ess_tail'].min():.0f}")

print("--- Group estimates, posterior mean [95% HDI]")
estimates = pd.DataFrame({q: {g: mean_hdi(draws_of(g, q), '.3f' if q.startswith('gam') else '.2f') for g in groups}
                          for q in QUANTITIES})
print(estimates.to_string())
print("beta0, beta1 (standardized units):")
print(pd.DataFrame({b: {g: mean_hdi(trace.posterior['beta_vec'].sel(groups=g, betas=b).values) for g in groups}
                    for b in ['b0', 'b1']}).to_string())
print("posterior sd / prior sd:")
print(pd.DataFrame({q: {g: draws_of(g, q).std() / draws_of(g, q, trace.prior).std() for g in groups}
                    for q in QUANTITIES}).round(3).to_string())

print("--- Distractor effect, Model A (Table)")
for _, row in effect_summary.iterrows():
    f = '.3f' if row['quantity'].startswith('gam') else '.2f'
    print(f"{row['hand']} {row['quantity']}: {row['mean']:{f}} [{row['hdi_low']:{f}}, {row['hdi_high']:{f}}], "
          f"P(>0) = {row['P(>0)']:.3f}")

print("--- Posterior predictive checks")
print(f"pooled: {n_pooled_inside} of {len(groups) * len(levels)} group x amplitude proportions inside their 95% interval")
print(f"single sessions: {session_cells['inside'].sum()} of {len(session_cells)} cells inside their 95% interval "
      f"({session_cells['inside'].mean():.1%})")
print(f"PIT < 0.025: {np.mean(session_cells['pit'] < 0.025):.1%}, PIT > 0.975: {np.mean(session_cells['pit'] > 0.975):.1%} "
      f"(2.5% each if calibrated)")
print("fraction of sessions inside, by group and amplitude:")
print(inside_by_amp.round(2).to_string())

print("--- LOO")
print(f"Model A: elpd_loo {LOO_results.elpd_loo:.1f} (SE {LOO_results.se:.1f}), p_loo {LOO_results.p_loo:.1f}, "
      f"max Pareto k {np.max(LOO_results.pareto_k.values):.2f}")
print(f"Model B: elpd_loo {loo_B.elpd_loo:.1f} (SE {loo_B.se:.1f}), p_loo {loo_B.p_loo:.0f}")
print(f"Model B - Model A: {elpd_diff:.1f} (SE {elpd_diff_se:.1f})")

print("--- Model A vs Model B, group values (Model B: average of the 42 session posterior means)")
summary_B = fit_results_B['az_summary_trace']
for q in QUANTITIES:
    b_vals = {g: summary_B.loc[summary_B.index.str.startswith(f'{q}[{g},'), 'mean'].mean() * SCALE[q] + SHIFT[q]
              for g in groups}
    f = '.3f' if q.startswith('gam') else '.2f'
    print(f"{q}: " + '; '.join(f"{g} A {draws_of(g, q).mean():{f}}, B {b_vals[g]:{f}}" for g in groups))
print("--- Distractor effect: Model A vs Model B average over sessions vs Model B typical session")
for hand in HANDS:
    for q in QUANTITIES:
        a = effect_summary.set_index(['hand', 'quantity']).loc[(hand, q)]
        b = effect_B.loc[(hand, q)]
        f = '.3f' if q.startswith('gam') else '.2f'
        print(f"{hand} {q}: A {a['mean']:{f}} [{a['hdi_low']:{f}}, {a['hdi_high']:{f}}] (width {a['hdi_high'] - a['hdi_low']:{f}}) | "
              f"B avg {b['sess_avg_mean']:{f}} [{b['sess_avg_hdi_low']:{f}}, {b['sess_avg_hdi_high']:{f}}] "
              f"(width {b['sess_avg_hdi_high'] - b['sess_avg_hdi_low']:{f}}) | "
              f"B pop {b['pop_mean']:{f}} [{b['pop_hdi_low']:{f}}, {b['pop_hdi_high']:{f}}] "
              f"(width {b['pop_hdi_high'] - b['pop_hdi_low']:{f}})")

#%%

fit_results = {'az_summary_trace': result_df,
               'az_loo_trace': LOO_results,
               'effect_summary': effect_summary}

with open(MODEL_DIR / "Results_A.pkl","wb") as f:
    pickle.dump(fit_results, f)
