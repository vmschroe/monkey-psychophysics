#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The PSE against how recently the monkey was exposed to the session's distractor amplitude: days
since the first session at that amplitude, and days since the previous session at it.

Post hoc: neither is in Model B. Uses the PSE draws cached by PSE_Within_Amplitude_B.py, so run
that first. Only sessions in the dataset count as exposure, and sessions are dated MM-DD, all
within one year.
"""

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from pathlib import Path
from scipy.stats import t as t_dist

try:
    EXPLORATORY_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    EXPLORATORY_DIR = Path.cwd()
MODEL_DIR = EXPLORATORY_DIR.parent
DATA_DIR = MODEL_DIR / "Data_B"
PLOTS_DIR = EXPLORATORY_DIR / "Plots" / "Exposure_Days_Plots_B"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TREND_MIN_POINTS = 5   # same cutoff as PSE_Within_Amplitude_B.py
TREND_MIN_SPAN = 5     # days; a line through a narrower span would only be extrapolation
SLOPE_DAYS = 10        # slopes are reported per this many days
SUMMARY_MIN_SPAN = 15  # days; narrower amplitudes stay in the printed table but not the slope figure

#%% Load: PSE in micrometres, dims (draws, sessions) for every group

with open(DATA_DIR / "ReadyData_Sirius_B.pkl", "rb") as f:
    data_dict = pickle.load(f)
with open(EXPLORATORY_DIR / "PSE_Draws_B.pkl", "rb") as f:
    pse_xr = pickle.load(f)
with open(MODEL_DIR.parent / "Sirius_Data.pkl", "rb") as f:
    stims = np.unique(pickle.load(f)['data']['stim_amp'])
x_mu, x_sig = np.mean(stims), np.std(stims)

groups = list(pse_xr['groups'].values)
dates = [str(d) for d in pse_xr['sessions'].values]
sess_summary = data_dict['session_summary']
dist_amp = sess_summary.set_index(sess_summary.index.astype(str))['dist_amp'].reindex(dates).values.astype(float)
pse = {g: pse_xr.sel(groups=g, drop=True).stack(sample=('chain', 'draw')).transpose('sample', 'sessions').values
          * x_sig + x_mu for g in groups}

HANDS = {'Left Hand': ('left_uni', 'left_bi'), 'Right Hand': ('right_uni', 'right_bi')}
shift = {hand: pse[bi] - pse[uni] for hand, (uni, bi) in HANDS.items()}
CONDS = {'Unimanual': (0, '#2a78d6', 'o', -1), 'Bimanual': (1, '#e34948', 's', 1)}
SHIFT_COLOR = '#52514e'
GRID, AXIS, MUTED = '#e1e0d9', '#c3c2b7', '#898781'
MU = r'$\mu m$'

#%% Days since the first and the previous session at the same amplitude

day = (pd.to_datetime(['2000-' + d for d in dates]) - pd.Timestamp('2000-01-01')).days.to_numpy().astype(float)
days_first, days_last = np.full(len(dates), np.nan), np.full(len(dates), np.nan)
for a in np.unique(dist_amp):
    idx = np.flatnonzero(dist_amp == a)
    days_first[idx] = day[idx] - day[idx[0]]
    days_last[idx[1:]] = np.diff(day[idx])
schedule = pd.DataFrame({'date': dates, 'amp': dist_amp, 'days_first': days_first, 'days_last': days_last})

AXES_DEFS = {'first': (days_first, 'days since first session at this amplitude', 'days since first exposure',
                       'days_since_first'),
             'last': (days_last, 'days since previous session at this amplitude', 'days since previous exposure',
                      'days_since_last')}

#%% Helpers

def hdi95(d):
    """Mean and 95% HDI of every column of a (draws, k) array."""
    return d.mean(0), az.hdi(d[None], hdi_prob=0.95)

def trend(x, d):
    """Least-squares line for every draw. x: (n,), d: (draws, n) -> (intercept/slope, draws)."""
    return np.linalg.lstsq(np.c_[np.ones(len(x)), x], d.T, rcond=None)[0]

def style(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(AXIS)
    ax.tick_params(colors='#52514e', labelsize=8, length=0)

def panel_grid(amp_idx, ylabel, title):
    amps = list(amp_idx)
    fig, axes = plt.subplots(len(HANDS), len(amps), sharey=True, constrained_layout=True, figsize=(14, 7.2),
                             gridspec_kw={'width_ratios': [len(amp_idx[a]) + 1.5 for a in amps]})
    for c, a in enumerate(amps):
        axes[0, c].set_title(f'{a:.0f} {MU} (n = {len(amp_idx[a])})', fontsize=10)
    for ax, hand in zip(axes[:, 0], HANDS):
        ax.set_ylabel(f'{hand}\n{ylabel}')
    for ax in axes.ravel():
        style(ax)
    fig.suptitle(title, fontsize=13)
    return fig, axes

def has_trend(x):
    return len(x) >= TREND_MIN_POINTS and x.max() - x.min() >= TREND_MIN_SPAN

def plot_series(ax, x, d, color, marker, side, label):
    span = max(x.max() - x.min(), 1)
    m, h = hdi95(d)
    ax.errorbar(x + side * 0.012 * span, m, yerr=[m - h[:, 0], h[:, 1] - m], fmt=marker, color=color, markersize=4.5,
                elinewidth=1, label=label)
    if has_trend(x):
        coef = trend(x, d)
        grid = np.linspace(x.min(), x.max(), 50)
        lines = coef[0][:, None] + coef[1][:, None] * grid
        ax.fill_between(grid, *np.quantile(lines, [0.025, 0.975], axis=0), color=color, alpha=0.15, linewidth=0)
        ax.plot(grid, lines.mean(0), color=color, linewidth=2)
    ax.set_xlim(x.min() - 0.08 * span, x.max() + 0.08 * span)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

def add_key(fig, axes):
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=SHIFT_COLOR, linewidth=2))
    labels.append(f'linear trend, 95% band ({TREND_MIN_POINTS}+ points over {TREND_MIN_SPAN}+ days)')
    fig.legend(handles, labels, loc='outside lower center', ncol=len(labels), fontsize=9, frameon=False)

def classical_ci(y, X):
    """OLS slope and 95% interval on the posterior means, so session-to-session scatter counts as noise."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    dof = len(y) - X.shape[1]
    se = np.sqrt(res @ res / dof * np.linalg.inv(X.T @ X)[-1, -1])
    return b[-1], b[-1] - t_dist.ppf(0.975, dof) * se, b[-1] + t_dist.ppf(0.975, dof) * se

def level_design(idx, *covariates):
    """A separate level for every amplitude, then the covariates, for the sessions in idx."""
    levels = np.unique(dist_amp[idx])
    return np.c_[np.vstack([dist_amp[idx] == a for a in levels]).T.astype(float), *covariates]

measures = {'Unimanual PSE': lambda hand: pse[HANDS[hand][0]], 'Bimanual PSE': lambda hand: pse[HANDS[hand][1]],
            'Bimanual − unimanual': lambda hand: shift[hand]}
measure_colors = dict(zip(measures, ['#2a78d6', '#e34948', SHIFT_COLOR]))

#%% PSE, its shift and the slopes against each axis

slope_tables = {}
for key, (xvals, xlabel, short, file_label) in AXES_DEFS.items():
    # amplitudes with at least two sessions on this axis; a defined x is needed for every point
    amp_idx = {a: np.flatnonzero((dist_amp == a) & np.isfinite(xvals)) for a in np.unique(dist_amp)}
    amp_idx = {a: i for a, i in amp_idx.items() if len(i) >= 2}

    fig, axes = panel_grid(amp_idx, f'PSE ({MU})', f'PSE against {short} (posterior mean and 95% HDI)')
    for r, pair in enumerate(HANDS.values()):
        for c, (a, idx) in enumerate(amp_idx.items()):
            ax = axes[r, c]
            for cond, (i, color, marker, side) in CONDS.items():
                plot_series(ax, xvals[idx], pse[pair[i]][:, idx], color, marker, side, cond)
            ax.axhline(28, linestyle='--', linewidth=1, color='green', label='trained threshold')
            if r == 1:
                ax.set_xlabel('days', fontsize=9)
    add_key(fig, axes)
    fig.savefig(PLOTS_DIR / f'PSE_vs_{file_label}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, axes = panel_grid(amp_idx, f'bimanual − unimanual PSE ({MU})',
                           f'Distractor PSE shift against {short} (posterior mean and 95% HDI)')
    for r, hand in enumerate(HANDS):
        for c, (a, idx) in enumerate(amp_idx.items()):
            ax = axes[r, c]
            plot_series(ax, xvals[idx], shift[hand][:, idx], SHIFT_COLOR, 'o', 0, 'bimanual − unimanual')
            ax.axhline(0, linestyle='--', linewidth=1, color=MUTED)
            if r == 1:
                ax.set_xlabel('days', fontsize=9)
    add_key(fig, axes)
    fig.savefig(PLOTS_DIR / f'PSE_shift_vs_{file_label}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    trend_amps = [a for a, i in amp_idx.items() if has_trend(xvals[i])]
    used = np.concatenate(list(amp_idx.values()))
    X_pool = level_design(used, xvals[used])
    shown = [a for a in trend_amps if np.ptp(xvals[amp_idx[a]]) >= SUMMARY_MIN_SPAN] + ['pooled']
    row_labels = [f'{a:.0f} {MU} ({len(amp_idx[a])})' for a in shown[:-1]] + [f'pooled ({len(used)})']
    fig, axes = plt.subplots(len(HANDS), len(measures), sharex=True, sharey=True, constrained_layout=True, figsize=(11, 6))
    rows = []
    for r, hand in enumerate(HANDS):
        for ax, (name, get) in zip(axes[r], measures.items()):
            d = get(hand)
            slopes = [trend(xvals[amp_idx[a]], d[:, amp_idx[a]])[1] * SLOPE_DAYS for a in trend_amps]
            slopes.append(np.linalg.lstsq(X_pool, d[:, used].T, rcond=None)[0][-1] * SLOPE_DAYS)
            for s, label in zip(slopes, trend_amps + ['pooled']):
                low, high = az.hdi(s, hdi_prob=0.95)
                is_pool = label == 'pooled'
                if label in shown:
                    ax.errorbar(s.mean(), -shown.index(label), xerr=[[s.mean() - low], [high - s.mean()]],
                                fmt='D' if is_pool else 'o', color=measure_colors[name],
                                markersize=6 if is_pool else 4.5, elinewidth=2 if is_pool else 1.2)
                idx = used if is_pool else amp_idx[label]
                X = X_pool if is_pool else np.c_[np.ones(len(idx)), xvals[idx]]
                _, c_low, c_high = classical_ci(d[:, idx].mean(0), X)
                rows.append({'hand': hand, 'measure': name, 'amp': label, 'mean': s.mean(), 'low': low, 'high': high,
                             'ols_low': c_low * SLOPE_DAYS, 'ols_high': c_high * SLOPE_DAYS})
            ax.axvline(0, linestyle='--', linewidth=1, color=MUTED)
            ax.axhline(-len(shown) + 1.5, color=AXIS, linewidth=0.8)
            ax.grid(axis='x', color=GRID, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.spines['bottom'].set_color(AXIS)
            ax.tick_params(colors='#52514e', labelsize=9, length=0)
            ax.set_yticks(-np.arange(len(row_labels)), row_labels)
            if r == 0:
                ax.set_title(name, fontsize=10)
            else:
                ax.set_xlabel(f'{MU} per {SLOPE_DAYS} days')
        axes[r, 0].set_ylabel(hand)
    fig.suptitle(f'Trend against {short} (posterior mean and 95% HDI)', fontsize=12)
    fig.savefig(PLOTS_DIR / f'PSE_slope_vs_{file_label}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    slope_tables[key] = pd.DataFrame(rows)

#%% Numbers

pd.set_option('display.width', 200, 'display.max_rows', 300, 'display.max_columns', 30)
print(schedule.to_string())
for key, table in slope_tables.items():
    print(f"--- slopes per {SLOPE_DAYS} days, {key}")
    print(table.round(2).to_string())

has_last = np.flatnonzero(np.isfinite(days_last))
has_last = has_last[[(dist_amp == dist_amp[i]).sum() >= 3 for i in has_last]]
print("correlation of days since first and previous session (pooled sessions):",
      round(float(np.corrcoef(days_first[has_last], days_last[has_last])[0, 1]), 2))
nth = np.zeros(len(dates))
for a in np.unique(dist_amp):
    nth[dist_amp == a] = np.arange(1, (dist_amp == a).sum() + 1)
print("correlation of days since first and nth session within amplitude:",
      {f'{a:.0f}': round(float(np.corrcoef(days_first[dist_amp == a], nth[dist_amp == a])[0, 1]), 2)
       for a in [6, 12, 24, 32]})

def print_both(idx, title):
    """Both axes in one regression, with a level for each amplitude: posterior HDI, then the OLS interval."""
    print(title)
    for hand in HANDS:
        for name, get in measures.items():
            d = get(hand)[:, idx]
            parts = []
            for label, x, other in [('first', days_first[idx], days_last[idx]), ('previous', days_last[idx], days_first[idx])]:
                X = level_design(idx, other, x)
                s = np.linalg.lstsq(X, d.T, rcond=None)[0][-1] * SLOPE_DAYS
                low, high = az.hdi(s, hdi_prob=0.95)
                _, c_low, c_high = classical_ci(d.mean(0), X)
                parts.append(f"{label} {s.mean():.2f} [{low:.2f}, {high:.2f}] ols [{c_low * SLOPE_DAYS:.2f}, {c_high * SLOPE_DAYS:.2f}]")
            print(f"  {hand} {name}: " + ', '.join(parts))

long_gap = days_last > 20
keep = has_last[~long_gap[has_last]]
print_both(has_last, f"both axes in one regression, per {SLOPE_DAYS} days:")
print_both(keep, f"the same without the gaps over 20 days {[str(d) for d in np.array(dates)[long_gap]]}:")

print("pooled slope against days since previous, without those gaps:")
X_keep = level_design(keep, days_last[keep])
for hand in HANDS:
    for name, get in measures.items():
        s = np.linalg.lstsq(X_keep, get(hand)[:, keep].T, rcond=None)[0][-1] * SLOPE_DAYS
        low, high = az.hdi(s, hdi_prob=0.95)
        print(f"  {hand} {name}: {s.mean():.2f} [{low:.2f}, {high:.2f}]")
