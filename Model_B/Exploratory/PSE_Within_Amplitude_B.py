#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
How the PSE changes from one session to the next among sessions at the same distractor amplitude.

Post hoc, like the amplitude section of Main_Analysis_B.py: neither amplitude nor session order is
in Model B. Holding amplitude fixed separates a change over repeated exposure from the effect of
amplitude itself, which are confounded in the raw schedule (r = 0.91).

Results_B.pkl keeps only summaries, so the first run refits Model B and caches the PSE draws in
PSE_Draws_B.pkl. Nothing written by Main_Analysis_B.py is touched.
"""

import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.stats import t as t_dist

try:
    EXPLORATORY_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    EXPLORATORY_DIR = Path.cwd()
MODEL_DIR = EXPLORATORY_DIR.parent
DATA_DIR = MODEL_DIR / "Data_B"
DRAWS_PATH = EXPLORATORY_DIR / "PSE_Draws_B.pkl"
PLOTS_DIR = EXPLORATORY_DIR / "Plots" / "Within_Amplitude_Plots_B"

TREND_MIN_SESSIONS = 5      # same cutoff as the amplitude table in Main_Analysis_B.py

def sample_pse_draws():
    """Model B exactly as in Main_Analysis_B.py, keeping only the PSE."""
    with open(DATA_DIR / "ReadyData_Sirius_B.pkl", "rb") as f:
        data_dict = pickle.load(f)
    namespace = {'sessions': list(data_dict['dates_sess_idx']), 'cov_mat': data_dict['cov_mat'],
                 'grp_idx': data_dict['grp_idx'], 'obs_data': data_dict['resp'], 'sess_idx': data_dict['sess_idx']}
    exec(open(MODEL_DIR / "Build_Model_B.py").read(), namespace)
    with namespace['model_B']:
        trace = pm.sample(draws=5000, chains=4, cores=4, random_seed=20261030, progressbar=False,
                          return_inferencedata=True, compute_convergence_checks=False)
    print(f"divergences: {int(trace.sample_stats['diverging'].sum())}")
    print(az.summary(trace, var_names=['PSE'], kind='diagnostics')[['ess_bulk', 'ess_tail', 'r_hat']].describe().round(2))
    pse = trace.posterior['PSE'].astype('float32')
    with open(DRAWS_PATH, "wb") as f:
        pickle.dump(pse, f)

if __name__ == "__main__" and not DRAWS_PATH.exists():   # guarded: chain processes re-import this file
    sample_pse_draws()

#%% Load: PSE in micrometres, dims (draws, sessions) for every group

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "ReadyData_Sirius_B.pkl", "rb") as f:
        data_dict = pickle.load(f)
    with open(DRAWS_PATH, "rb") as f:
        pse_xr = pickle.load(f)

    groups = list(pse_xr['groups'].values)
    dates = [str(d) for d in pse_xr['sessions'].values]
    sess_summary = data_dict['session_summary']
    dist_amp = sess_summary.set_index(sess_summary.index.astype(str))['dist_amp'].reindex(dates).values.astype(float)

    with open(MODEL_DIR.parent / "Sirius_Data.pkl", "rb") as f:
        stims = np.unique(pickle.load(f)['data']['stim_amp'])
    X_MU, X_SIG = np.mean(stims), np.std(stims)

    pse = {g: pse_xr.sel(groups=g, drop=True).stack(sample=('chain', 'draw')).transpose('sample', 'sessions').values
              * X_SIG + X_MU for g in groups}

    HANDS = {'Left Hand': ('left_uni', 'left_bi'), 'Right Hand': ('right_uni', 'right_bi')}
    CONDS = {'Unimanual': (0, '#2a78d6', 'o', -0.09), 'Bimanual': (1, '#e34948', 's', 0.09)}
    SHIFT_COLOR = '#52514e'
    GRID, AXIS, MUTED = '#e1e0d9', '#c3c2b7', '#898781'
    MU = r'$\mu m$'

    # sessions in order at every amplitude that was run more than once
    amps = [a for a in np.unique(dist_amp) if (dist_amp == a).sum() >= 2]
    amp_sess = {a: np.flatnonzero(dist_amp == a) for a in amps}
    single = [a for a in np.unique(dist_amp) if (dist_amp == a).sum() == 1]
    shift = {hand: pse[bi] - pse[uni] for hand, (uni, bi) in HANDS.items()}

    def hdi95(d):
        """Mean and 95% HDI of every column of a (draws, k) array."""
        return d.mean(0), az.hdi(d[None], hdi_prob=0.95)

    def trend(d):
        """Least-squares line through (nth session, value) for every draw. d: (draws, n) -> (intercept/slope, draws)."""
        X = np.c_[np.ones(d.shape[1]), np.arange(1, d.shape[1] + 1)]
        return np.linalg.lstsq(X, d.T, rcond=None)[0]

    def style(ax):
        ax.grid(axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(AXIS)
        ax.tick_params(colors='#52514e', labelsize=8, length=0)

    def mark_breaks(ax, idx, offset=0.5):
        """Dotted line wherever another amplitude ran between two sessions at this one."""
        for n in np.flatnonzero(np.diff(idx) > 1):
            ax.axvline(n + 1 + offset, color=MUTED, linewidth=0.8, linestyle=':')

    def panel_grid(ylabel, title):
        widths = [len(amp_sess[a]) + 1.5 for a in amps]
        fig, axes = plt.subplots(len(HANDS), len(amps), sharey=True, constrained_layout=True, figsize=(14, 7.2),
                                 gridspec_kw={'width_ratios': widths})
        for c, a in enumerate(amps):
            axes[0, c].set_title(f'{a:.0f} {MU} (n = {len(amp_sess[a])})', fontsize=10)
            axes[-1, c].set_xlabel(f'session at {a:.0f} {MU}', fontsize=9)
        for ax, hand in zip(axes[:, 0], HANDS):
            ax.set_ylabel(f'{hand}\n{ylabel}')
        for ax in axes.ravel():
            style(ax)
        fig.suptitle(title, fontsize=13)
        return fig, axes

    def plot_series(ax, d, color, marker, dx, label, show_trend):
        n = d.shape[1]
        x = np.arange(1, n + 1)
        m, h = hdi95(d)
        ax.plot(x + dx, m, color=color, linewidth=1, alpha=0.45)
        ax.errorbar(x + dx, m, yerr=[m - h[:, 0], h[:, 1] - m], fmt=marker, color=color, markersize=4.5,
                    elinewidth=1, label=label)
        if show_trend:
            coef = trend(d)
            lines = coef[0][:, None] + coef[1][:, None] * x
            ax.fill_between(x, *np.quantile(lines, [0.025, 0.975], axis=0), color=color, alpha=0.15, linewidth=0)
            ax.plot(x, lines.mean(0), color=color, linewidth=2)
        ax.set_xticks(x)
        ax.set_xlim(0.4, n + 0.6)

    def add_key(fig, axes):
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color=MUTED, linestyle=':', linewidth=1))
        labels.append('another amplitude ran in between')
        handles.append(plt.Line2D([], [], color=SHIFT_COLOR, linewidth=2))
        labels.append(f'linear trend, 95% band ({TREND_MIN_SESSIONS}+ sessions)')
        fig.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize=9, frameon=False)

#%% PSE at every consecutive session of each amplitude

    fig, axes = panel_grid(f'PSE ({MU})', 'PSE over consecutive sessions at the same distractor amplitude '
                                          '(posterior mean and 95% HDI)')
    for r, (hand, pair) in enumerate(HANDS.items()):
        for c, a in enumerate(amps):
            ax, idx = axes[r, c], amp_sess[a]
            for cond, (i, color, marker, dx) in CONDS.items():
                plot_series(ax, pse[pair[i]][:, idx], color, marker, dx, cond, len(idx) >= TREND_MIN_SESSIONS)
            ax.axhline(28, linestyle='--', linewidth=1, color='green', label='trained threshold')
            mark_breaks(ax, idx)
    add_key(fig, axes)
    fig.savefig(PLOTS_DIR / 'PSE_within_amplitude.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

#%% PSE shift (bimanual - unimanual) at every consecutive session of each amplitude

    fig, axes = panel_grid(f'bimanual − unimanual PSE ({MU})',
                           'Distractor PSE shift over consecutive sessions at the same distractor amplitude '
                           '(posterior mean and 95% HDI)')
    for r, hand in enumerate(HANDS):
        for c, a in enumerate(amps):
            ax, idx = axes[r, c], amp_sess[a]
            plot_series(ax, shift[hand][:, idx], SHIFT_COLOR, 'o', 0, 'bimanual − unimanual', len(idx) >= TREND_MIN_SESSIONS)
            ax.axhline(0, linestyle='--', linewidth=1, color=MUTED)
            mark_breaks(ax, idx)
    add_key(fig, axes)
    fig.savefig(PLOTS_DIR / 'PSE_shift_within_amplitude.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

#%% Change from each session to the next one at the same amplitude

    fig, axes = panel_grid(f'change in PSE ({MU})',
                           'Change in PSE from each session to the next at the same distractor amplitude '
                           '(posterior mean and 95% HDI)')
    step_rows = []
    for r, (hand, pair) in enumerate(HANDS.items()):
        for c, a in enumerate(amps):
            ax, idx = axes[r, c], amp_sess[a]
            for cond, (i, color, marker, dx) in CONDS.items():
                d = np.diff(pse[pair[i]][:, idx], axis=1)
                m, h = hdi95(d)
                x = np.arange(1, len(idx))
                ax.errorbar(x + dx, m, yerr=[m - h[:, 0], h[:, 1] - m], fmt=marker, color=color, markersize=4.5,
                            elinewidth=1, label=cond)
                step_rows += [{'hand': hand, 'cond': cond, 'amp': a, 'from': dates[idx[k]], 'to': dates[idx[k + 1]],
                               'interrupted': idx[k + 1] - idx[k] > 1, 'mean': m[k], 'low': h[k, 0], 'high': h[k, 1]}
                              for k in range(len(idx) - 1)]
            ax.axhline(0, linestyle='--', linewidth=1, color=MUTED)
            ax.set_xticks(np.arange(1, len(idx)), [f'{n}→{n + 1}' for n in range(1, len(idx))], fontsize=7)
            ax.set_xlim(0.4, len(idx) - 0.4)
            mark_breaks(ax, idx, offset=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, linestyle=':', linewidth=1))
    labels.append('another amplitude ran between these two sessions')
    fig.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize=9, frameon=False)
    fig.savefig(PLOTS_DIR / 'PSE_step_change_within_amplitude.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    steps = pd.DataFrame(step_rows)
    steps['excludes_0'] = (steps['low'] > 0) | (steps['high'] < 0)

#%% Slope per consecutive session: one amplitude at a time, and pooled across amplitudes

    trend_amps = [a for a in amps if len(amp_sess[a]) >= TREND_MIN_SESSIONS]
    used = np.concatenate([amp_sess[a] for a in amps])
    nth = np.zeros(len(dates))
    for a in amps:
        nth[amp_sess[a]] = np.arange(1, len(amp_sess[a]) + 1)
    X_pool = np.c_[np.vstack([dist_amp[used] == a for a in amps]).T.astype(float), nth[used]]

    def pooled_slope(d):
        """Common slope in the nth session at an amplitude, with a separate level for every amplitude."""
        return np.linalg.lstsq(X_pool, d[:, used].T, rcond=None)[0][-1]

    def classical_ci(y, X):
        """OLS slope and 95% interval on the posterior means, so session-to-session scatter counts as noise."""
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ b
        dof = len(y) - X.shape[1]
        se = np.sqrt(res @ res / dof * np.linalg.inv(X.T @ X)[-1, -1])
        return b[-1], b[-1] - t_dist.ppf(0.975, dof) * se, b[-1] + t_dist.ppf(0.975, dof) * se

    measures = {'Unimanual PSE': lambda hand: pse[HANDS[hand][0]], 'Bimanual PSE': lambda hand: pse[HANDS[hand][1]],
                'Bimanual − unimanual': lambda hand: shift[hand]}
    measure_colors = dict(zip(measures, ['#2a78d6', '#e34948', SHIFT_COLOR]))
    row_labels = [f'{a:.0f} {MU} ({len(amp_sess[a])})' for a in trend_amps] + [f'pooled ({len(used)})']

    fig, axes = plt.subplots(len(HANDS), len(measures), sharex=True, sharey=True, constrained_layout=True, figsize=(11, 6))
    slope_rows = []
    for r, hand in enumerate(HANDS):
        for ax, (name, get) in zip(axes[r], measures.items()):
            d = get(hand)
            slopes = [trend(d[:, amp_sess[a]])[1] for a in trend_amps] + [pooled_slope(d)]
            for y, (s, label) in enumerate(zip(slopes, trend_amps + ['pooled'])):
                low, high = az.hdi(s, hdi_prob=0.95)
                is_pool = label == 'pooled'
                ax.errorbar(s.mean(), -y, xerr=[[s.mean() - low], [high - s.mean()]], fmt='D' if is_pool else 'o',
                            color=measure_colors[name], markersize=6 if is_pool else 4.5, elinewidth=2 if is_pool else 1.2)
                if is_pool:
                    y_means, X_c = d[:, used].mean(0), X_pool
                else:
                    y_means = d[:, amp_sess[label]].mean(0)
                    X_c = np.c_[np.ones(len(y_means)), np.arange(1, len(y_means) + 1)]
                c_mean, c_low, c_high = classical_ci(y_means, X_c)
                slope_rows.append({'hand': hand, 'measure': name, 'amp': label, 'mean': s.mean(), 'low': low, 'high': high,
                                   'ols_low': c_low, 'ols_high': c_high})
            ax.axvline(0, linestyle='--', linewidth=1, color=MUTED)
            ax.axhline(-len(trend_amps) + 0.5, color=AXIS, linewidth=0.8)
            ax.grid(axis='x', color=GRID, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color(AXIS)
            ax.tick_params(colors='#52514e', labelsize=9, length=0)
            ax.set_yticks(-np.arange(len(row_labels)), row_labels)
            if r == 0:
                ax.set_title(name, fontsize=10)
            else:
                ax.set_xlabel(f'{MU} per consecutive session')
        axes[r, 0].set_ylabel(hand)
    fig.suptitle('Trend across consecutive sessions at the same distractor amplitude (posterior mean and 95% HDI)', fontsize=12)
    fig.savefig(PLOTS_DIR / 'PSE_slope_within_amplitude.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    slopes_df = pd.DataFrame(slope_rows)

#%% Numbers

    pd.set_option('display.width', 200, 'display.max_rows', 300, 'display.max_columns', 30)
    print(f"stimulus scale: PSE_um = {X_MU:.2f} + {X_SIG:.2f} PSE")
    print("amplitudes run once, left out:", [f'{a:.0f}' for a in single])
    print(pd.DataFrame({a: pd.Series(dates)[amp_sess[a]].reset_index(drop=True) for a in amps}).to_string())
    print(slopes_df.round(2).to_string())
    print(f"steps with 95% HDI excluding 0: {steps['excludes_0'].sum()} of {len(steps)}")
    print(steps[steps['excludes_0']].round(2).to_string())
    print(steps.groupby('interrupted')['mean'].agg(['size', 'mean', lambda s: s.abs().mean()]).round(2))
    print("slope per session at 6 um without 04-10 (177 trials, the shortest session):")
    for g in groups:
        s = trend(pse[g][:, amp_sess[6.0][:-1]])[1]
        print(f"  {g}: {s.mean():.2f} [{az.hdi(s, hdi_prob=0.95)[0]:.2f}, {az.hdi(s, hdi_prob=0.95)[1]:.2f}]")
    print("slope per session at 24 um over sessions 1-5, before the July sessions:")
    for g in groups:
        s = trend(pse[g][:, amp_sess[24.0][:5]])[1]
        print(f"  {g}: {s.mean():.2f} [{az.hdi(s, hdi_prob=0.95)[0]:.2f}, {az.hdi(s, hdi_prob=0.95)[1]:.2f}]")
    means = pd.DataFrame({g: pse[g].mean(0) for g in groups})
    print("across sessions, correlation of uni and bi posterior-mean PSE after removing each amplitude's mean:")
    resid = means[dist_amp != 0].sub(means.groupby(dist_amp).transform('mean'))[np.isin(dist_amp, amps)]
    for hand, (uni, bi) in HANDS.items():
        print(f"  {hand}: r = {resid[uni].corr(resid[bi]):.2f}")
