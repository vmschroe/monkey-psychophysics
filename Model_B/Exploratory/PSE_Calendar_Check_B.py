#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is there a rhythm in the session PSEs? Two post hoc looks at Model B:

  1. The PSE against calendar date, raw and with each amplitude's mean level removed.
  2. A mixed posterior predictive check. Model B treats sessions as independent draws from their
     population, so each replicate draws hyperparameters from the posterior, NEW session parameters
     from them, and trials on the real design. Replicating from the fitted session parameters
     instead would copy any real pattern into the replicates and hide it.

Replicated data can't be refit with Model B thousands of times, so real and replicated data both go
through the same fast estimator: a maximum likelihood PSE for each session and group, with slope
and lapse rates fixed at the group's typical values.

The first run refits Model B and caches its hyperparameters in Hyper_Draws_B.pkl. The PSE draws come
from PSE_Within_Amplitude_B.py, so run that first. Nothing from Main_Analysis_B.py is touched.
"""

import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
from pathlib import Path
from scipy.signal import lombscargle

try:
    EXPLORATORY_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    EXPLORATORY_DIR = Path.cwd()
MODEL_DIR = EXPLORATORY_DIR.parent
DATA_DIR = MODEL_DIR / "Data_B"
HYPER_PATH = EXPLORATORY_DIR / "Hyper_Draws_B.pkl"
PLOTS_DIR = EXPLORATORY_DIR / "Plots" / "Calendar_Check_Plots_B"

N_REPS = 2000
SEED = 20261031
PERIODS = np.linspace(2.5, 60, 400)   # days
WINDOW = (15, 22)                     # days; around the ~19-day peak, chosen after seeing it

def sample_hyper_draws():
    """Model B exactly as in Main_Analysis_B.py, keeping only the hyperparameters."""
    with open(DATA_DIR / "ReadyData_Sirius_B.pkl", "rb") as f:
        data_dict = pickle.load(f)
    namespace = {'sessions': list(data_dict['dates_sess_idx']), 'cov_mat': data_dict['cov_mat'],
                 'grp_idx': data_dict['grp_idx'], 'obs_data': data_dict['resp'], 'sess_idx': data_dict['sess_idx']}
    exec(open(MODEL_DIR / "Build_Model_B.py").read(), namespace)
    with namespace['model_B']:
        trace = pm.sample(draws=5000, chains=4, cores=4, random_seed=20261030, progressbar=False,
                          return_inferencedata=True, compute_convergence_checks=False)
    print(f"divergences: {int(trace.sample_stats['diverging'].sum())}")
    hyper = {v: trace.posterior[v].stack(sample=('chain', 'draw')).transpose('sample', 'betas', 'groups').values
             for v in ['mu_betas', 'sig_betas', 'mu_gams', 'sig_gams']}
    with open(HYPER_PATH, "wb") as f:
        pickle.dump(hyper, f)

if __name__ == "__main__" and not HYPER_PATH.exists():   # guarded: chain processes re-import this file
    sample_hyper_draws()

if __name__ == "__main__":

#%% Load

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    with open(DATA_DIR / "ReadyData_Sirius_B.pkl", "rb") as f:
        data_dict = pickle.load(f)
    with open(EXPLORATORY_DIR / "PSE_Draws_B.pkl", "rb") as f:
        pse_xr = pickle.load(f)
    with open(HYPER_PATH, "rb") as f:
        hyper = pickle.load(f)
    with open(MODEL_DIR.parent / "Sirius_Data.pkl", "rb") as f:
        stims = np.unique(pickle.load(f)['data']['stim_amp'])
    x_mu, x_sig = np.mean(stims), np.std(stims)

    groups = list(pse_xr['groups'].values)
    dates = [str(d) for d in pse_xr['sessions'].values]
    sess_summary = data_dict['session_summary']
    dist_amp = sess_summary.set_index(sess_summary.index.astype(str))['dist_amp'].reindex(dates).values.astype(float)
    when = pd.to_datetime(['2000-' + d for d in dates])   # the data have no year; only the days between sessions matter
    day = (when - when[0]).days.to_numpy().astype(float)
    used = np.array([(dist_amp == a).sum() >= 2 for a in dist_amp])
    amps_used, days_used = dist_amp[used], day[used]

    pse = {g: pse_xr.sel(groups=g, drop=True).stack(sample=('chain', 'draw')).transpose('sample', 'sessions').values
              * x_sig + x_mu for g in groups}
    HANDS = {'Left Hand': ('left_uni', 'left_bi'), 'Right Hand': ('right_uni', 'right_bi')}
    CONDS = {'Unimanual': (0, '#2a78d6', 'o', -0.3), 'Bimanual': (1, '#e34948', 's', 0.3)}
    GRID, AXIS, MUTED, INK = '#e1e0d9', '#c3c2b7', '#898781', '#52514e'
    MU = r'$\mu m$'

    def style(ax):
        ax.grid(axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(AXIS)
        ax.tick_params(colors=INK, labelsize=8, length=0)

    def hdi95(d):
        return d.mean(0), az.hdi(d[None], hdi_prob=0.95)

#%% PSE against calendar date, raw and with each amplitude's mean level removed

    def within_amp(d):
        """Subtract each amplitude's mean over its sessions, separately for every draw."""
        d = d.copy()
        for a in np.unique(amps_used):
            d[:, amps_used == a] -= d[:, amps_used == a].mean(1, keepdims=True)
        return d

    for residual in [False, True]:
        fig, axes = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(14, 8.5),
                                 gridspec_kw={'height_ratios': [1, 2.4, 2.4]})
        keep = used if residual else np.ones(len(dates), bool)
        axes[0].plot(when, dist_amp, color=MUTED, linewidth=1, drawstyle='steps-post')
        axes[0].plot(when, dist_amp, 'o', color=INK, markersize=4)
        axes[0].set_yticks([6, 12, 24, 32])
        axes[0].set_ylabel(f'distractor\n({MU})')
        for ax, (hand, pair) in zip(axes[1:], HANDS.items()):
            for cond, (i, color, marker, dx) in CONDS.items():
                d = pse[pair[i]][:, used]
                d = within_amp(d) if residual else pse[pair[i]]
                m, h = hdi95(d)
                t = when[keep] + pd.Timedelta(days=dx)
                ax.plot(t, m, color=color, linewidth=1, alpha=0.45)
                ax.errorbar(t, m, yerr=[m - h[:, 0], h[:, 1] - m], fmt=marker, color=color, markersize=4.5,
                            elinewidth=1, label=cond)
            ax.axhline(0 if residual else 28, linestyle='--', linewidth=1, color=MUTED if residual else 'green',
                       label=None if residual else 'trained threshold')
            ax.set_ylabel(f"{hand}\n{'PSE − amplitude mean' if residual else 'PSE'} ({MU})")
        for ax in axes:
            style(ax)
        axes[-1].xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize=9, frameon=False)
        fig.suptitle(('PSE with each amplitude\'s mean removed' if residual else 'PSE') +
                     ' against calendar date (posterior mean and 95% HDI)', fontsize=13)
        fig.savefig(PLOTS_DIR / ('PSE_calendar_within_amplitude.png' if residual else 'PSE_calendar.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

#%% Fast PSE estimator, the same for real and replicated data

    x_levels = np.unique(np.round(data_dict['cov_mat'][:, 1], 6))
    level = np.searchsorted(x_levels, np.round(data_dict['cov_mat'][:, 1], 6))
    S, G, L = len(dates), len(groups), len(x_levels)
    n_cell, k_real = np.zeros((S, G, L)), np.zeros((S, G, L))
    np.add.at(n_cell, (data_dict['sess_idx'], data_dict['grp_idx'], level), 1)
    np.add.at(k_real, (data_dict['sess_idx'], data_dict['grp_idx'], level), data_dict['resp'])

    typical = {'b1': hyper['mu_betas'][:, 1].mean(0),
               'gh': (0.25 / (1 + np.exp(-hyper['mu_gams'][:, 0]))).mean(0),
               'gl': (0.25 / (1 + np.exp(-hyper['mu_gams'][:, 1]))).mean(0)}
    pse_grid = np.arange(-2.5, 2.5001, 0.01)   # normalized units, 0.15 um steps
    log_p, log_q = [], []
    for g in range(G):
        b1, gh, gl = typical['b1'][g], typical['gh'][g], typical['gl'][g]
        b0 = -pse_grid * b1 + np.log((1 - 2*gh) / (1 - 2*gl))
        psi = gh + (1 - gh - gl) / (1 + np.exp(-(b0[:, None] + b1 * x_levels[None, :])))
        log_p.append(np.log(psi))
        log_q.append(np.log(1 - psi))

    def estimate_pse(k):
        """Maximum likelihood PSE in um. k: (sessions, groups, levels) -> (sessions, groups)."""
        out = np.empty((S, G))
        for g in range(G):
            out[:, g] = pse_grid[np.argmax(k[:, g] @ log_p[g].T + (n_cell[:, g] - k[:, g]) @ log_q[g].T, axis=1)]
        return out * x_sig + x_mu

#%% Statistics

    W = 2 * np.pi / PERIODS
    in_window = (PERIODS >= WINDOW[0]) & (PERIODS <= WINDOW[1])
    i_pair, j_pair = np.triu_indices(used.sum(), 1)
    gap = days_used[j_pair] - days_used[i_pair]

    def statistics(est):
        y = est[used]
        for a in np.unique(amps_used):
            y[amps_used == a] -= y[amps_used == a].mean(0)
        mean4 = y.mean(1)
        power = lombscargle(days_used, mean4 - mean4.mean(), W, normalize=True)
        semivar = (mean4[i_pair] - mean4[j_pair]) ** 2 / 2
        band = (gap >= 15) & (gap <= 21)
        r = np.corrcoef(y.T)   # order: left_uni, left_bi, right_uni, right_bi
        shifts = [y[:, 1] - y[:, 0], y[:, 3] - y[:, 2]]
        return {'peak': power.max(), 'period': PERIODS[power.argmax()], 'window': power[in_window].max(),
                'semivar': semivar[band].mean() / mean4.var(), 'lag1': np.corrcoef(mean4[:-1], mean4[1:])[0, 1],
                'r_uni': r[0, 2], 'r_bi': r[1, 3], 'r_left': r[0, 1], 'r_right': r[2, 3],
                'sd_group': y.std(0).mean(), 'sd_mean4': mean4.std(),
                'peak_shift': max(lombscargle(days_used, s - s.mean(), W, normalize=True).max() for s in shifts)}

#%% Mixed replication: fresh independent sessions from the posterior hyperparameters

    def replicate(i):
        mu_b, sig_b = hyper['mu_betas'][i], hyper['sig_betas'][i]
        mu_g, sig_g = hyper['mu_gams'][i], hyper['sig_gams'][i]
        b0 = rng.normal(mu_b[0], sig_b[0], (S, G))
        b1 = rng.normal(mu_b[1], sig_b[1], (S, G))
        gh = 0.25 / (1 + np.exp(-rng.normal(mu_g[0], sig_g[0], (S, G))))
        gl = 0.25 / (1 + np.exp(-rng.normal(mu_g[1], sig_g[1], (S, G))))
        psi = gh[..., None] + (1 - gh - gl)[..., None] / (1 + np.exp(-(b0[..., None] + b1[..., None] * x_levels)))
        return rng.binomial(n_cell.astype(int), psi).astype(float)

    est_real = estimate_pse(k_real)
    observed = statistics(est_real.copy())
    draws = rng.choice(len(hyper['mu_betas']), N_REPS, replace=False)
    reps = pd.DataFrame([statistics(estimate_pse(replicate(i))) for i in draws])

    CHECKS = [('peak', 'peak power, mean of 4 PSEs', 'high'), ('window', f'peak power, {WINDOW[0]}–{WINDOW[1]} days (post hoc)', 'high'),
              ('semivar', 'semivariance at 15–21 days / variance\n(post hoc)', 'low'), ('lag1', 'lag-1 autocorrelation', 'high'),
              ('r_uni', 'left–right r, unimanual', 'high'), ('r_bi', 'left–right r, bimanual', 'high'),
              ('r_left', 'unimanual–bimanual r, left', 'high'), ('r_right', 'unimanual–bimanual r, right', 'high')]
    fig, axes = plt.subplots(2, 4, constrained_layout=True, figsize=(14, 6.5))
    rows = []
    for ax, (key, label, side) in zip(axes.ravel(), CHECKS):
        p = (reps[key] >= observed[key]).mean() if side == 'high' else (reps[key] <= observed[key]).mean()
        ax.hist(reps[key], bins=40, color='#2a78d6', alpha=0.75, edgecolor='#fcfcfb', linewidth=0.5)
        ax.axvline(observed[key], color='#0b0b0b', linewidth=2)
        ax.set_title(f'{label}\nobserved {observed[key]:.2f}, p = {p:.3f}', fontsize=9)
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color(AXIS)
        ax.tick_params(colors=INK, labelsize=8, length=0)
        rows.append({'statistic': key, 'observed': observed[key], 'rep_median': reps[key].median(),
                     'rep_2.5': reps[key].quantile(0.025), 'rep_97.5': reps[key].quantile(0.975), 'p': p})
    fig.suptitle(f'Mixed posterior predictive check of Model B: {N_REPS} replicates of independent sessions (blue), '
                 'real data (black line)', fontsize=12)
    fig.savefig(PLOTS_DIR / 'PSE_mixed_predictive_check.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

#%% Numbers

    pd.set_option('display.width', 200, 'display.max_columns', 30)
    means = np.column_stack([pse[g].mean(0) for g in groups])
    print("estimator vs Model B posterior means: r by group",
          [round(float(np.corrcoef(est_real[:, g], means[:, g])[0, 1]), 2) for g in range(G)],
          "; sd of estimator / sd of posterior means", np.round(est_real.std(0) / means.std(0), 2))
    print(pd.DataFrame(rows).round(3).to_string())
    print(f"observed peak period {observed['period']:.1f} days; replicates' most common peak periods:",
          reps['period'].round(-1).value_counts().head(4).to_dict())
    for key in ['sd_group', 'sd_mean4', 'peak_shift']:
        print(f"{key}: observed {observed[key]:.2f}, replicates median {reps[key].median():.2f}, "
              f"95% [{reps[key].quantile(0.025):.2f}, {reps[key].quantile(0.975):.2f}], "
              f"P(rep >= obs) {(reps[key] >= observed[key]).mean():.3f}")
