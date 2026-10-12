#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 2026

Posterior convergence study for Model C (hierarchical over sessions, one
group per hand, distractor amplitude as a covariate).

Generates synthetic datasets from the true hyperparameters in
ReadyData_Synth_C.pkl (derived from Model B's real-data fit, see
SynthData_C.py), fits Model C to each, and tracks how the recovered
posteriors approach the true values. Dataset size is varied in two
separate sweeps, as in Convergence_B.py:
    - 'sessions': the number of sessions grows, each session keeping the
      real average number of trials. This mainly shows recovery of the
      hyperparameters (hand means and between-session sds).
    - 'trials': the number of sessions is fixed at the real 42 and the
      trials per session grow. This mainly shows recovery of the
      session-level parameters.

Two levels of true values are tracked:
    - hyperparameters (mu and sig of b0, b1, b2, gam_h, gam_l for each hand;
      the gamma ones on the logit scale used by Build_Model_C.py), plotted
      individually;
    - session-level parameters (gam_h, gam_l, b0, b1, b2, PSE on uni and on
      bi trials, JND for every session and hand), summarized across sessions
      as RMSE of the posterior mean, mean HDI width and HDI coverage.

Design:
    - Every replicate draws new true session-level parameters from the
      fixed hyperparameters (SynthData_C.sample_session_params).
    - Every session gets one distractor amplitude, drawn from the real
      sessions' amplitudes, as each real session uses a single amplitude.
    - Trials cycle through the 4 cells hand x (uni, bi) within a session
      (the real sessions are balanced across them) and each trial's stimulus
      is resampled from that cell's stimuli in the real design.
    - Within a replicate the datasets of a sweep are nested: fewer sessions
      means the first n sessions, fewer trials means the first t trials of
      every session. Differences between sizes then come from the extra
      data rather than from a new random draw.

Notes on reading the results:
    - In the 'trials' sweep the hyperparameter posteriors can at best
      converge to the mean/sd of the 42 sessions actually drawn, not to
      the population values.
    - b2 of a session is informed only by that session's uni-vs-bi shift at
      its single distractor amplitude, so with few sessions mu_b2 and
      sig_b2 also depend on how spread out the drawn amplitudes are.

Runtime: about 0.7 h for the defaults with the 4 chains run in parallel
(0.5-6.4 min per fit, growing with dataset size). Results are saved after
every fit, so an interrupted run keeps the fits already done.

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
import sys
import time
from pathlib import Path

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
DATA_DIR = MODEL_DIR / "Data_C"
PLOT_DIR = MODEL_DIR / "Plots_C" / "Convergence_Plots_C"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = MODEL_DIR / "Results_Convergence_C.pkl"

sys.path.insert(0, str(MODEL_DIR))   # SynthData_C.py, also when run interactively from elsewhere
from SynthData_C import HYPER_NAMES, sample_session_params, response_probs

#%% SETTINGS

SESSION_SWEEP = [5, 10, 20, 40, 80]     # number of sessions, each with TRIALS_PER_SESSION_FIXED trials
TRIALS_PER_SESSION_FIXED = 308          # real average trials per session
TRIALS_SWEEP = [20, 40, 80, 160, 320]   # trials per session (multiples of 4 keep cells balanced)
N_SESSIONS_FIXED = 42                   # real number of sessions
N_REPS = 2                              # independent synthetic runs of each sweep
DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4                               # chains sampled in parallel; 1 = one after another
# Python 3.14's default 'forkserver' re-runs this script in every worker, so fork them instead
# ('fork' doesn't exist on Windows, which keeps its default)
MP_CTX = None if os.name == 'nt' else 'fork'
HDI_PROB = 0.95
SEED = 20261013

rng = np.random.default_rng(SEED)

SWEEPS = {
    'sessions': {'sizes': SESSION_SWEEP,
                 'label': f'number of sessions ({TRIALS_PER_SESSION_FIXED} trials each)'},
    'trials': {'sizes': TRIALS_SWEEP,
               'label': f'trials per session ({N_SESSIONS_FIXED} sessions)'},
}

#%% LOAD DESIGN AND TRUE HYPERPARAMETERS

with open(DATA_DIR / "ReadyData_Synth_C.pkl", "rb") as f:
    data_dict = pickle.load(f)

design_cov_mat = data_dict['cov_mat']
design_grp_idx = data_dict['grp_idx']
real_sess_x_dist = data_dict['sess_x_dist']
hparams_fixed_C = data_dict['hparams_fixed_C']
names_grp_idx = data_dict['names_grp_idx']
groups = sorted(names_grp_idx, key=names_grp_idx.get)   # ['left', 'right']
CONDITIONS = ['uni', 'bi']

# cells cycle hand x condition: cell = 2*hand + condition
CELLS = [(g_i, c_i) for g_i in range(len(groups)) for c_i in range(len(CONDITIONS))]
design_cond = (design_cov_mat[:, 2] > 0).astype(int)
stim_pool = {(g_i, c_i): design_cov_mat[(design_grp_idx == g_i) & (design_cond == c_i), 1] for g_i, c_i in CELLS}

HYPER_FAMILIES = {'means': [n for n in HYPER_NAMES if n.startswith('mu_')],
                  'sds': [n for n in HYPER_NAMES if n.startswith('sig_')]}
SESSION_PARAMS = ['gam_h', 'gam_l', 'b0', 'b1', 'b2', 'PSE_uni', 'PSE_bi', 'JND']

true_hyper = {name: hparams_fixed_C.loc[name, groups].to_numpy() for name in HYPER_NAMES}

#%% HELPERS

def pse_jnd(tp, sess_x_dist):
    # same formulas as Build_Model_C.py; PSE at x_dist = 0 (uni) and at the session's x_dist (bi)
    log_ratio = np.log((1 - 2*tp['gam_h']) / (1 - 2*tp['gam_l']))
    PSE_uni = (-tp['b0'] + log_ratio) / tp['b1']
    PSE_bi = (-(tp['b0'] + tp['b2'] * sess_x_dist) + log_ratio) / tp['b1']
    JND = np.log(((3 - 4*tp['gam_h'])*(3 - 4*tp['gam_l'])) / ((1 - 4*tp['gam_h'])*(1 - 4*tp['gam_l']))) / (2*tp['b1'])
    return PSE_uni, PSE_bi, JND

def sample_sessions(n_sessions, rng):
    """True session-level parameters and each session's distractor value.
    Returns ({param: array (n_groups, n_sessions)}, sess_x_dist (n_sessions,))."""
    tp = sample_session_params(hparams_fixed_C, n_sessions, rng)
    sess_x_dist = rng.choice(real_sess_x_dist, size=n_sessions)
    tp['PSE_uni'], tp['PSE_bi'], tp['JND'] = pse_jnd(tp, sess_x_dist)
    return tp, sess_x_dist

def simulate_sessions(true_sess, sess_x_dist, trials_per_session, rng):
    """Covariates, hand and response for every trial of every session.
    Cells cycle hand x condition so any prefix of a session stays balanced.
    Returns cov (n_sessions, trials_per_session, 3), and hand and resp (n_sessions, trials_per_session)."""
    n_sessions = len(sess_x_dist)
    n_cycles = -(-trials_per_session // len(CELLS))
    cell = np.tile(np.arange(len(CELLS)), (n_sessions, n_cycles))[:, :trials_per_session]
    hand, cond = cell // len(CONDITIONS), cell % len(CONDITIONS)
    stim = np.empty(cell.shape)
    for c, key in enumerate(CELLS):
        mask = cell == c
        stim[mask] = rng.choice(stim_pool[key], size=mask.sum())
    sess = np.broadcast_to(np.arange(n_sessions)[:, None], cell.shape)
    cov = np.stack([np.ones(cell.shape), stim, cond * sess_x_dist[:, None]], axis=-1)
    psi = response_probs(true_sess, cov.reshape(-1, 3), hand.ravel(), sess.ravel()).reshape(cell.shape)
    resp = rng.binomial(1, psi)
    return cov, hand, resp

def split_hyper(ds):
    """Map each name in HYPER_NAMES to a DataArray with a 'groups' dim."""
    out = {}
    for b in ['b0', 'b1', 'b2']:
        out[f'mu_{b}'] = ds['mu_betas'].sel(betas=b)
        out[f'sig_{b}'] = ds['sig_betas'].sel(betas=b)
    for lapse in ['gam_h', 'gam_l']:
        out[f'mu_{lapse}'] = ds['mu_gams'].sel(lapses=lapse)
        out[f'sig_{lapse}'] = ds['sig_gams'].sel(lapses=lapse)
    return out

def split_session(ds):
    """Map each name in SESSION_PARAMS to a DataArray with 'groups' and 'sessions' dims."""
    return {
        'gam_h': ds['gam_h'],
        'gam_l': ds['gam_l'],
        'b0': ds['beta_vec'].sel(betas='b0'),
        'b1': ds['beta_vec'].sel(betas='b1'),
        'b2': ds['beta_vec'].sel(betas='b2'),
        'PSE_uni': ds['PSE'].sel(conditions='uni'),
        'PSE_bi': ds['PSE'].sel(conditions='bi'),
        'JND': ds['JND'],
    }

HYPER_VARS = ['mu_betas', 'sig_betas', 'mu_gams', 'sig_gams']
SESSION_VARS = ['beta_vec', 'gam_h', 'gam_l', 'PSE', 'JND']
BUILD_MODEL_C = (MODEL_DIR / "Build_Model_C.py").read_text()

def fit_model_C(cov_mat, grp_idx, sess_idx, sess_x_dist, obs_data):
    # Build_Model_C.py reads these names; running it in its own namespace keeps it
    # from overwriting anything in this script
    ns = {'cov_mat': cov_mat, 'grp_idx': grp_idx, 'sess_idx': sess_idx, 'sess_x_dist': sess_x_dist,
          'obs_data': obs_data, 'sessions': [f's{i:02d}' for i in range(len(sess_x_dist))]}
    exec(BUILD_MODEL_C, ns)
    with ns['model_C']:
        return pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CORES, mp_ctx=MP_CTX,
                         random_seed=int(rng.integers(2**31)), progressbar=True)

hyper_rows = []
session_frames = []
hyper_samples = {}   # (sweep, rep, size) -> {hyper name: float32 array (n_samples, n_groups)}

def run_fit(sweep, rep, size, true_sess, sess_x_dist, cov, hand, resp, n_sessions, n_trials):
    """Fit Model C to the first n_sessions sessions and first n_trials trials of each,
    and record hyperparameter and session-level recovery."""
    t0 = time.time()
    print(f"--- {sweep} sweep, replicate {rep+1}/{N_REPS}, {n_sessions} sessions x {n_trials} trials ---")
    cov_d = cov[:n_sessions, :n_trials].reshape(-1, 3)
    hand_d, resp_d = (a[:n_sessions, :n_trials].ravel() for a in (hand, resp))
    sess_d = np.repeat(np.arange(n_sessions), n_trials)
    trace = fit_model_C(cov_d, hand_d, sess_d, sess_x_dist[:n_sessions], resp_d)
    n_div = int(trace.sample_stats['diverging'].sum())

    # hyperparameters: one true value per hand
    post = split_hyper(trace.posterior[HYPER_VARS])
    rhat = split_hyper(az.rhat(trace, var_names=HYPER_VARS))
    ess = split_hyper(az.ess(trace, var_names=HYPER_VARS))
    hyper_samples[(sweep, rep, size)] = {}
    for name in HYPER_NAMES:
        samps = post[name].stack(sample=('chain', 'draw')).transpose('sample', 'groups').values
        hyper_samples[(sweep, rep, size)][name] = samps.astype(np.float32)
        for g_i, grp_name in enumerate(groups):
            s = samps[:, g_i]
            hdi_low, hdi_high = az.hdi(s, hdi_prob=HDI_PROB)
            truth = true_hyper[name][g_i]
            hyper_rows.append({
                'sweep': sweep, 'rep': rep, 'size': size, 'n_sessions': n_sessions, 'n_trials': n_trials,
                'group': grp_name, 'param': name, 'true': truth,
                'mean': s.mean(), 'sd': s.std(), 'hdi_low': hdi_low, 'hdi_high': hdi_high,
                'error': s.mean() - truth, 'covered': hdi_low <= truth <= hdi_high,
                'r_hat': float(rhat[name].sel(groups=grp_name)),
                'ess_bulk': float(ess[name].sel(groups=grp_name)),
                'divergences': n_div,
            })

    # session level: posterior mean and HDI for every session, next to that session's true value
    post = split_session(trace.posterior[SESSION_VARS])
    hdi = split_session(az.hdi(trace, var_names=SESSION_VARS, hdi_prob=HDI_PROB))
    rhat = split_session(az.rhat(trace, var_names=SESSION_VARS))
    g_grid, s_grid = np.meshgrid(np.arange(len(groups)), np.arange(n_sessions), indexing='ij')
    for par in SESSION_PARAMS:
        order = ('groups', 'sessions')
        session_frames.append(pd.DataFrame({
            'sweep': sweep, 'rep': rep, 'size': size, 'param': par,
            'group': np.asarray(groups)[g_grid.ravel()], 'session': s_grid.ravel(),
            'x_dist': sess_x_dist[s_grid.ravel()],
            'true': true_sess[par][:, :n_sessions].ravel(),
            'mean': post[par].mean(('chain', 'draw')).transpose(*order).values.ravel(),
            'hdi_low': hdi[par].sel(hdi='lower').transpose(*order).values.ravel(),
            'hdi_high': hdi[par].sel(hdi='higher').transpose(*order).values.ravel(),
            'r_hat': rhat[par].transpose(*order).values.ravel(),
            'divergences': n_div,
        }))

    save_results()
    print(f"    done in {(time.time() - t0)/60:.1f} min, {n_div} divergences")

def collect_results():
    hyper_results = pd.DataFrame(hyper_rows)
    session_results = pd.concat(session_frames, ignore_index=True)
    session_results['error'] = session_results['mean'] - session_results['true']
    session_results['hdi_width'] = session_results['hdi_high'] - session_results['hdi_low']
    session_results['covered'] = ((session_results['hdi_low'] <= session_results['true'])
                                  & (session_results['true'] <= session_results['hdi_high']))
    return hyper_results, session_results

def save_results():
    hyper_results, session_results = collect_results()
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({'hyper_results': hyper_results, 'session_results': session_results,
                     'hyper_samples': hyper_samples, 'groups': groups, 'true_hyper': true_hyper,
                     'sweeps': SWEEPS, 'n_reps': N_REPS, 'hdi_prob': HDI_PROB, 'seed': SEED}, f)

#%% GENERATE DATA AND FIT MODEL C ALONG BOTH SWEEPS

run_start = time.time()
for rep in range(N_REPS):
    # sessions sweep: one set of sessions per replicate; smaller datasets use the first n sessions
    true_sess, sess_x_dist = sample_sessions(max(SESSION_SWEEP), rng)
    cov, hand, resp = simulate_sessions(true_sess, sess_x_dist, TRIALS_PER_SESSION_FIXED, rng)
    for n_sessions in SESSION_SWEEP:
        run_fit('sessions', rep, n_sessions, true_sess, sess_x_dist, cov, hand, resp, n_sessions, TRIALS_PER_SESSION_FIXED)

    # trials sweep: fixed sessions; smaller datasets use the first t trials of every session
    true_sess, sess_x_dist = sample_sessions(N_SESSIONS_FIXED, rng)
    cov, hand, resp = simulate_sessions(true_sess, sess_x_dist, max(TRIALS_SWEEP), rng)
    for n_trials in TRIALS_SWEEP:
        run_fit('trials', rep, n_trials, true_sess, sess_x_dist, cov, hand, resp, N_SESSIONS_FIXED, n_trials)

hyper_results, session_results = collect_results()
print(f"FINISHED SAMPLING! ({(time.time() - run_start)/3600:.1f} h)")

#%% (optional) reload saved results instead of refitting

# with open(RESULTS_PATH, "rb") as f:
#     saved = pickle.load(f)
# hyper_results, session_results, hyper_samples = saved['hyper_results'], saved['session_results'], saved['hyper_samples']
# groups, true_hyper, SWEEPS, N_REPS, HDI_PROB = saved['groups'], saved['true_hyper'], saved['sweeps'], saved['n_reps'], saved['hdi_prob']

#%% Sampling diagnostics: r_hat should be ~1 and there should be no divergences

fits = hyper_results.drop_duplicates(['sweep', 'rep', 'size'])
diag = (hyper_results.groupby(['sweep', 'size'])
        .agg(max_r_hat_hyper=('r_hat', 'max'), min_ess_hyper=('ess_bulk', 'min'))
        .join(fits.groupby(['sweep', 'size'])['divergences'].sum())
        .join(session_results.groupby(['sweep', 'size']).agg(max_r_hat_session=('r_hat', 'max'))))
print(diag.to_string())

#%% Hyperparameter coverage: fraction of fits whose 95% HDI contains the true value

for sweep in SWEEPS:
    cov_tab = hyper_results[hyper_results['sweep'] == sweep].pivot_table(index='param', columns='size',
                                                                          values='covered', aggfunc='mean')
    print(f"\n{sweep} sweep\n" + cov_tab.loc[HYPER_NAMES].round(2).to_string())

#%% Session-level summaries across sessions, for each fit

session_summary = (session_results.groupby(['sweep', 'rep', 'size', 'param', 'group'])
                   .agg(rmse=('error', lambda e: np.sqrt(np.mean(e**2))),
                        mean_hdi_width=('hdi_width', 'mean'),
                        coverage=('covered', 'mean'))
                   .reset_index())

for sweep in SWEEPS:
    cov_tab = session_summary[session_summary['sweep'] == sweep].pivot_table(index='param', columns='size',
                                                                              values='coverage', aggfunc='mean')
    print(f"\n{sweep} sweep: session-level coverage\n" + cov_tab.loc[SESSION_PARAMS].round(2).to_string())

#%% Plot colors and helpers

CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
GROUP_COLORS = dict(zip(groups, CATEGORICAL))
SIZE_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']

def size_colors(sizes):
    # blue ramp, light -> dark with dataset size
    idx = np.linspace(0, len(SIZE_RAMP) - 1, len(sizes)).round().astype(int)
    return dict(zip(sizes, [SIZE_RAMP[i] for i in idx]))

def size_axis(ax, sizes):
    # log x-axis limited to the fitted sizes, ticked at each size
    ax.set_xscale('log')
    ax.set_xlim(min(sizes) / 1.4, max(sizes) * 1.4)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

def save_plot(fig, name):
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=200, bbox_inches='tight')

rep_offsets = np.exp(np.linspace(-0.08, 0.08, N_REPS)) if N_REPS > 1 else [1.0]   # spread reps on the log x-axis

#%% Hyperparameter posterior mean and 95% HDI vs dataset size, for each sweep

for sweep, info in SWEEPS.items():
    sizes = info['sizes']
    for family, names in HYPER_FAMILIES.items():
        fig, axes = plt.subplots(len(names), len(groups), sharex=True, constrained_layout=True,
                                 figsize=(9, 2.2 * len(names) + 0.8))
        for r, name in enumerate(names):
            for ax, grp in zip(axes[r], groups):
                sub = hyper_results[(hyper_results['sweep'] == sweep) & (hyper_results['param'] == name)
                                    & (hyper_results['group'] == grp)]
                for rep in range(N_REPS):
                    d = sub[sub['rep'] == rep]
                    ax.errorbar(d['size'] * rep_offsets[rep], d['mean'],
                                yerr=[d['mean'] - d['hdi_low'], d['hdi_high'] - d['mean']],
                                fmt='o', markersize=4, linewidth=1.5, capsize=0, color='#2a78d6',
                                label=f'posterior mean ± {int(HDI_PROB*100)}% HDI' if rep == 0 else None)
                ax.axhline(true_hyper[name][groups.index(grp)], linestyle='--', linewidth=1.5, color='#52514e',
                           label='true value')
                size_axis(ax, sizes)
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.set_title(grp)
            axes[r, 0].set_ylabel(name + (' (logit)' if 'gam' in name else ''))
        for ax in axes[-1]:
            ax.set_xlabel(info['label'])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2)
        fig.suptitle(f'Recovery of hyperparameter {family} vs {info["label"]}', fontsize=14)
        save_plot(fig, f"hyper_{family}_recovery_vs_{sweep}")
        plt.show()

#%% Hyperparameter posterior densities, replicates overlaid; darker = larger dataset

for sweep, info in SWEEPS.items():
    sizes = info['sizes']
    colors = size_colors(sizes)
    for family, names in HYPER_FAMILIES.items():
        fig, axes = plt.subplots(len(names), len(groups), constrained_layout=True,
                                 figsize=(9, 2.2 * len(names) + 0.8))
        for r, name in enumerate(names):
            for g_i, (ax, grp) in enumerate(zip(axes[r], groups)):
                for size in sizes:
                    for rep in range(N_REPS):
                        vals = hyper_samples[(sweep, rep, size)][name][:, g_i]
                        vals = vals[np.isfinite(vals)]
                        az.plot_kde(vals, ax=ax, plot_kwargs={'color': colors[size], 'linewidth': 1.5, 'alpha': 0.85},
                                    label=f'{size}' if rep == 0 else None)
                ax.axvline(true_hyper[name][g_i], linestyle='--', linewidth=1.5, color='#52514e', label='true value')
                ax.set_yticks([])
                ax.tick_params(labelsize=9)
                if ax.get_legend():
                    ax.get_legend().remove()
                if r == 0:
                    ax.set_title(grp)
            axes[r, 0].set_ylabel(name + (' (logit)' if 'gam' in name else ''))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title=info['label'].split(' (')[0], loc='outside right center')
        fig.suptitle(f'Posterior of hyperparameter {family} as data grows, {sweep} sweep '
                     f'({N_REPS} replicates overlaid)', fontsize=14)
        save_plot(fig, f"hyper_{family}_density_vs_{sweep}")
        plt.show()

#%% Session-level recovery: RMSE, HDI width and coverage across sessions vs dataset size

metrics = [('rmse', 'RMSE of posterior mean'), ('mean_hdi_width', f'mean {int(HDI_PROB*100)}% HDI width'),
           ('coverage', f'{int(HDI_PROB*100)}% HDI coverage')]

for sweep, info in SWEEPS.items():
    sizes = info['sizes']
    summ = (session_summary[session_summary['sweep'] == sweep]
            .groupby(['param', 'group', 'size'])[[m for m, _ in metrics]].mean())
    fig, axes = plt.subplots(len(SESSION_PARAMS), len(metrics), sharex=True, constrained_layout=True,
                             figsize=(12, 2.0 * len(SESSION_PARAMS) + 0.8))
    for r, par in enumerate(SESSION_PARAMS):
        for ax, (metric, title) in zip(axes[r], metrics):
            for grp in groups:
                ax.plot(sizes, summ.loc[(par, grp)].loc[sizes, metric].values, marker='o', markersize=4,
                        linewidth=2, color=GROUP_COLORS[grp], label=grp)
            if metric == 'coverage':
                ax.axhline(HDI_PROB, linestyle='--', linewidth=1.5, color='#52514e', label=f'nominal {HDI_PROB}')
                ax.set_ylim(0, 1.05)
            else:
                ax.set_yscale('log')
            size_axis(ax, sizes)
            ax.grid(alpha=0.3, which='both')
            if r == 0:
                ax.set_title(title)
        axes[r, 0].set_ylabel(par)
    for ax in axes[-1]:
        ax.set_xlabel(info['label'])
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(labels))
    fig.suptitle(f'Session-level parameter recovery across sessions, {sweep} sweep (mean over replicates)',
                 fontsize=14)
    save_plot(fig, f"session_params_error_width_coverage_vs_{sweep}")
    plt.show()

#%% b2 by session: recovered vs true, coloured by the session's distractor value (largest dataset of each sweep)

fig, axes = plt.subplots(len(SWEEPS), len(groups), constrained_layout=True, figsize=(9, 8))
for r, (sweep, info) in enumerate(SWEEPS.items()):
    size = max(info['sizes'])
    for ax, grp in zip(axes[r], groups):
        d = session_results[(session_results['sweep'] == sweep) & (session_results['size'] == size)
                            & (session_results['param'] == 'b2') & (session_results['group'] == grp)]
        ax.errorbar(d['true'], d['mean'], yerr=[d['mean'] - d['hdi_low'], d['hdi_high'] - d['mean']],
                    fmt='none', linewidth=1, color='#b9b8b3', zorder=1)
        sc = ax.scatter(d['true'], d['mean'], c=d['x_dist'], cmap='Blues', vmin=0, vmax=real_sess_x_dist.max(),
                        s=24, edgecolors='#52514e', linewidths=0.5, zorder=2)
        lims = [min(d['true'].min(), d['hdi_low'].min()), max(d['true'].max(), d['hdi_high'].max())]
        ax.plot(lims, lims, linestyle='--', linewidth=1.5, color='#52514e')
        ax.set_title(f'{grp}, {sweep} sweep, size {size}')
        ax.set_xlabel('true b2')
        ax.grid(alpha=0.3)
    axes[r, 0].set_ylabel(f'posterior mean ± {int(HDI_PROB*100)}% HDI')
fig.colorbar(sc, ax=axes, label="session's x_dist")
fig.suptitle(f'Session-level b2 recovery ({N_REPS} replicates pooled)', fontsize=14)
save_plot(fig, "session_b2_true_vs_recovered")
plt.show()


#%% Numbers for Convergence_C_Report.txt
# Everything quoted in the report that is not printed by an earlier cell. Works on freshly fitted or
# reloaded results.

from scipy.stats import norm, halfnorm

inv_logit = lambda v: 1 / (1 + np.exp(-v))
x_sig = data_dict['x_sig']

def rmse(e):
    return np.sqrt(np.mean(e**2))

print("--- Design")
print(f"x_dist = dist_amp / {x_sig:.2f}; real sessions' x_dist {real_sess_x_dist.min():.2f}-{real_sess_x_dist.max():.2f}, "
      f"distinct amplitudes {sorted(np.unique(np.round(real_sess_x_dist * x_sig).astype(int)).tolist())}")
amp_counts = pd.Series(np.round(real_sess_x_dist * x_sig).astype(int)).value_counts().sort_index()
print("real sessions per distractor amplitude:", amp_counts.to_dict())
print(f"real data: {len(real_sess_x_dist)} sessions, {len(design_grp_idx)} trials")

print("--- Priors on b2 (Build_Model_C.py), exact")
print(f"mu_b2 ~ N(0, 1): 95% interval [{norm.ppf(0.025):.2f}, {norm.ppf(0.975):.2f}]; "
      f"shift mu_b2 * x_dist at the largest x_dist ({real_sess_x_dist.max():.2f}): 95% within +-{1.96 * real_sess_x_dist.max():.2f} logits")
print(f"sig_b2 ~ HalfNormal(1): median {halfnorm.median():.2f}, 95% point {halfnorm.ppf(0.95):.2f}")

print("--- True hyperparameters (from ReadyData_Synth_C.pkl)")
print(pd.DataFrame(true_hyper, index=groups).T.round(3).to_string())
for lapse in ['gam_h', 'gam_l']:
    print(f"typical {lapse} = 0.25 invlogit(mu): " + ', '.join(f"{g} {0.25 * inv_logit(true_hyper[f'mu_{lapse}'][g_i]):.4f}"
                                                            for g_i, g in enumerate(groups)))

print("--- Sampling diagnostics (divergences summed over replicates)")
print(diag.round(3).to_string())
print(f"{len(fits)} fits: max r_hat (hyper) {hyper_results['r_hat'].max():.3f}, "
      f"max r_hat (session) {session_results['r_hat'].max():.3f}, min ESS {hyper_results['ess_bulk'].min():.0f}, "
      f"{int(fits['divergences'].sum())} divergences")

print("--- Hyperparameters at the realistic sizes: replicate-averaged mean [HDI], true, coverage")
for sweep, size in [('sessions', 40), ('sessions', 80), ('trials', 320)]:
    d = hyper_results[(hyper_results['sweep'] == sweep) & (hyper_results['size'] == size)]
    t = d.groupby(['param', 'group']).agg(true=('true', 'first'), mean=('mean', 'mean'), hdi_low=('hdi_low', 'mean'),
                                          hdi_high=('hdi_high', 'mean'), covered=('covered', 'mean'))
    t['hdi_width'] = t['hdi_high'] - t['hdi_low']
    print(f"{sweep} sweep, size {size}:")
    print(t.loc[HYPER_NAMES].round(3).to_string())

print("--- Hyperparameter HDI width by size, averaged over hands and replicates")
for sweep in SWEEPS:
    d = hyper_results[hyper_results['sweep'] == sweep]
    print(f"{sweep} sweep:\n" + d.assign(width=d['hdi_high'] - d['hdi_low'])
          .pivot_table(index='param', columns='size', values='width').loc[HYPER_NAMES].round(2).to_string())

print("--- Session-level parameters, pooled over replicates: RMSE / mean HDI width / coverage")
pooled = session_results.groupby(['sweep', 'size', 'param', 'group']).agg(
    rmse=('error', rmse), mean_hdi_width=('hdi_width', 'mean'), coverage=('covered', 'mean'))
for sweep, size in [('trials', 320), ('sessions', 40)]:
    print(f"{sweep} sweep, size {size}:")
    print(pooled.loc[(sweep, size)].loc[SESSION_PARAMS].round(3).to_string())

print("--- Session-level b2 by the session's distractor value (trials sweep, 320 trials, pooled)")
d = session_results[(session_results['sweep'] == 'trials') & (session_results['size'] == max(TRIALS_SWEEP))
                    & (session_results['param'] == 'b2')]
d = d.assign(amp=np.round(d['x_dist'] * x_sig).astype(int))
by_amp = d.groupby('amp').agg(n=('error', 'size'), rmse=('error', rmse), mean_hdi_width=('hdi_width', 'mean'),
                              coverage=('covered', 'mean'))
print(by_amp.round(3).to_string())
print(f"corr(x_dist, b2 HDI width) = {np.corrcoef(d['x_dist'], d['hdi_width'])[0, 1]:.2f}")
# shrinkage: slope of posterior mean on true value (1 = no shrinkage)
for grp in groups:
    g = d[d['group'] == grp]
    print(f"{grp}: slope of posterior mean on true b2 = {np.polyfit(g['true'], g['mean'], 1)[0]:.2f}")
