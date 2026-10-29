#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 2026

Model recovery study for Model C (distractor as a covariate) against Models A
(pooled over sessions) and B (hierarchical over sessions, distractor as
separate groups): can PSIS-LOO tell which of the three generated a dataset?

Each replicate draws three fresh synthetic datasets on the real trial design
(the same 12937 trials in the same order in ReadyData_Sirius_B.pkl and
ReadyData_Sirius_C.pkl), with true values taken from the fits to the real
data so the synthetic data resemble it:
    - data A: responses from Model A's group parameters, the same for every
      session; true values are the posterior means of Model A's real-data fit
      (Model_A/Results_A.pkl), as in Model_B/Model_Recov_BA.py;
    - data B: new session-level parameters drawn from Model B's
      hyperparameters (posterior means of Model_B/Results_B.pkl), then
      responses, as in Model_B/Model_Recov_BA.py;
    - data C: new session-level parameters drawn from Model C's
      hyperparameters (posterior means of Results_C.pkl, from
      Main_Analysis_C.py), then responses, as in SynthData_C.py.
These true values only generate data; the priors of all models are unchanged.
All three models are fitted to all three datasets and compared with PSIS-LOO.
elpd_diff_XY is elpd_loo(X) - elpd_loo(Y) on the same data, with its standard
error dse_XY computed from the pointwise differences, as az.compare does.
The model with the highest elpd_loo is counted as the one selected.

Outputs, all written by this script:
    - Results_Model_Recov_C.pkl: one row per (replicate, dataset) with all
      three models' LOO results and sampling diagnostics;
    - Model_Recov_C_Results.txt: settings, per-replicate table, selection
      counts and diagnostics;
    - Plots_C/Model_Recov_Plots_C/elpd_diff_by_replicate.png.

Runtime: about 1 min for a Model A fit, 3 min for a Model B fit and 3 min for
a Model C fit with the 4 chains run in parallel, so about 20 min per replicate
and 3.5 h for the default 10 replicates. Results are saved after every
replicate, so an interrupted run keeps the replicates already done.

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

import sys
sys.path.insert(0, str(MODEL_DIR))
from SynthData_C import GROUPS as HANDS, HYPER_NAMES as HYPER_NAMES_C, sample_session_params, simulate_resp

MODEL_A_DIR = MODEL_DIR.parent / "Model_A"
MODEL_B_DIR = MODEL_DIR.parent / "Model_B"
PLOT_DIR = MODEL_DIR / "Plots_C" / "Model_Recov_Plots_C"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = MODEL_DIR / "Results_Model_Recov_C.pkl"
SUMMARY_PATH = MODEL_DIR / "Model_Recov_C_Results.txt"

#%% SETTINGS

N_REPS = 10
DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4
# Python 3.14's default 'forkserver' re-runs this script in every worker, so fork them instead
# ('fork' doesn't exist on Windows, which keeps its default)
MP_CTX = None if os.name == 'nt' else 'fork'
SEED = 20261028

MODEL_KEYS = ['A', 'B', 'C']
PAIRS = [('C', 'A'), ('C', 'B'), ('B', 'A')]   # elpd_diff_XY = elpd(X) - elpd(Y)

rng = np.random.default_rng(SEED)

#%% LOAD DESIGNS AND TRUE PARAMETERS

# Models A and B: four groups (hand x uni/bi), cov_mat = (1, x_stim)
with open(MODEL_B_DIR / "Data_B" / "ReadyData_Sirius_B.pkl", "rb") as f:
    data_AB = pickle.load(f)
# Model C: two groups (hand), cov_mat = (1, x_stim, x_dist)
with open(MODEL_DIR / "Data_C" / "ReadyData_Sirius_C.pkl", "rb") as f:
    data_C = pickle.load(f)

# the three models are compared trial by trial, so the trials must be the same and in the same order
assert np.array_equal(data_AB['resp'], data_C['resp'])
assert np.array_equal(data_AB['sess_idx'], data_C['sess_idx'])
assert np.allclose(data_AB['cov_mat'][:, 1], data_C['cov_mat'][:, 1])
assert list(data_AB['dates_sess_idx'].items()) == list(data_C['dates_sess_idx'].items())

sess_idx = data_C['sess_idx']
sessions = list(data_C['dates_sess_idx'])
n_sessions = len(sessions)
n_trials = len(sess_idx)
groups_AB = sorted(data_AB['names_grp_idx'], key=data_AB['names_grp_idx'].get)   # ['left_uni','left_bi','right_uni','right_bi']

with open(MODEL_A_DIR / "Results_A.pkl", "rb") as f:
    real_fit_A = pickle.load(f)
with open(MODEL_B_DIR / "Results_B.pkl", "rb") as f:
    real_fit_B = pickle.load(f)
with open(MODEL_DIR / "Results_C.pkl", "rb") as f:
    real_fit_C = pickle.load(f)
means_A, means_B, means_C = (fit['az_summary_trace']['mean'] for fit in (real_fit_A, real_fit_B, real_fit_C))

def posterior_means(means, var, beta=None, groups=groups_AB):
    """Posterior means of one variable from a real-data fit, in the order of groups."""
    return np.array([means[f"{var}[{beta + ', ' if beta else ''}{grp}]"] for grp in groups])

# Model A truth: one value per group
true_A = {'b0': posterior_means(means_A, 'beta_vec', 'b0'),
          'b1': posterior_means(means_A, 'beta_vec', 'b1'),
          'gam_h': posterior_means(means_A, 'gam_h'),
          'gam_l': posterior_means(means_A, 'gam_l')}

# Model B truth: hyperparameters per group, gamma ones on the logit scale.
# Build_Model_B.py indexes the gamma hyperparameters with the 'betas' dim: b0 -> gam_h, b1 -> gam_l
true_B = {'mu_b0': posterior_means(means_B, 'mu_betas', 'b0'),
          'sig_b0': posterior_means(means_B, 'sig_betas', 'b0'),
          'mu_b1': posterior_means(means_B, 'mu_betas', 'b1'),
          'sig_b1': posterior_means(means_B, 'sig_betas', 'b1'),
          'mu_gam_h': posterior_means(means_B, 'mu_gams', 'b0'),
          'sig_gam_h': posterior_means(means_B, 'sig_gams', 'b0'),
          'mu_gam_l': posterior_means(means_B, 'mu_gams', 'b1'),
          'sig_gam_l': posterior_means(means_B, 'sig_gams', 'b1')}

# Model C truth: hyperparameters per hand, gamma ones on the logit scale, in SynthData_C.py's layout
true_C = pd.DataFrame(
    {hand: {f'{stat}_{beta}': means_C[f'{stat}_betas[{beta}, {hand}]'] for stat in ('mu', 'sig') for beta in ('b0', 'b1', 'b2')}
           | {f'{stat}_{gam}': means_C[f'{stat}_gams[{gam}, {hand}]'] for stat in ('mu', 'sig') for gam in ('gam_h', 'gam_l')}
     for hand in HANDS}).loc[HYPER_NAMES_C, HANDS]

#%% DATA GENERATORS

def psychometric(gam_h, gam_l, b0, b1, stim):
    return gam_h + (1 - gam_h - gam_l) / (1 + np.exp(-(b0 + b1 * stim)))

def simulate_A(rng):
    """Responses from Model A's true group parameters, as in Model_Recov_BA.py."""
    grp_idx = data_AB['grp_idx']
    psi = psychometric(*(true_A[p][grp_idx] for p in ['gam_h', 'gam_l', 'b0', 'b1']), data_AB['cov_mat'][:, 1])
    return rng.binomial(1, psi)

def simulate_B(rng):
    """New session-level parameters from Model B's true hyperparameters, then
    responses, as in Model_Recov_BA.py. Session parameters are (n_groups, n_sessions)."""
    shape = (len(groups_AB), n_sessions)
    hp = {name: vals[:, None] for name, vals in true_B.items()}
    sp = {
        'b0': hp['mu_b0'] + hp['sig_b0'] * rng.standard_normal(shape),
        'b1': hp['mu_b1'] + hp['sig_b1'] * rng.standard_normal(shape),
        'gam_h': 0.25 / (1 + np.exp(-(hp['sig_gam_h'] * rng.standard_normal(shape) + hp['mu_gam_h']))),
        'gam_l': 0.25 / (1 + np.exp(-(hp['sig_gam_l'] * rng.standard_normal(shape) + hp['mu_gam_l']))),
    }
    grp_idx = data_AB['grp_idx']
    psi = psychometric(*(sp[p][grp_idx, sess_idx] for p in ['gam_h', 'gam_l', 'b0', 'b1']), data_AB['cov_mat'][:, 1])
    return rng.binomial(1, psi)

def simulate_C(rng):
    """New session-level parameters from Model C's true hyperparameters, then responses (SynthData_C.py)."""
    sp = sample_session_params(true_C, n_sessions, rng)
    return simulate_resp(sp, data_C['cov_mat'], data_C['grp_idx'], sess_idx, rng)

SIMULATORS = {'A': simulate_A, 'B': simulate_B, 'C': simulate_C}

#%% BUILD ALL THREE MODELS

# The build scripts read these names; running each in its own namespace keeps them
# from overwriting anything here. The design is shared, so each model is built once
# and only the responses are swapped between fits.
def build(path, model_name, design):
    ns = {'cov_mat': design['cov_mat'], 'grp_idx': design['grp_idx'], 'sess_idx': sess_idx,
          'sessions': sessions, 'obs_data': np.zeros(n_trials, dtype=int)}
    if 'sess_x_dist' in design:
        ns['sess_x_dist'] = design['sess_x_dist']
    exec(path.read_text(), ns)
    return ns[model_name]

MODELS = {'A': build(MODEL_A_DIR / "Build_Model_A.py", 'model_A', data_AB),
          'B': build(MODEL_B_DIR / "Build_Model_B.py", 'model_B', data_AB),
          'C': build(MODEL_DIR / "Build_Model_C.py", 'model_C', data_C)}

def fit_and_loo(model_key, obs):
    """Fit one model to one dataset; return its pointwise LOO and sampling diagnostics.
    The trace itself is dropped: its log likelihood is ~400 MB."""
    model = MODELS[model_key]
    with model:
        model.set_data("obs_data", obs)
        trace = pm.sample(draws=DRAWS, tune=TUNE, chains=CHAINS, cores=CORES, mp_ctx=MP_CTX,
                          random_seed=int(rng.integers(2**31)), progressbar=True,
                          idata_kwargs={"log_likelihood": True})
    loo = az.loo(trace, pointwise=True)
    # Model B's z_sig_* only enter through abs(), so their sign is unidentified and chains can settle on
    # opposite signs without affecting anything else (see Model_Recov_BA_Report). Check the deterministics
    # (sig_betas, sig_gams, beta_vec, ...) in their place. Model C's z_sig_* are HalfNormal, so they stay in.
    unidentified_sign = (lambda name: name.startswith('z_sig_')) if model_key == 'B' else (lambda name: False)
    diag_vars = ([rv.name for rv in model.free_RVs if not unidentified_sign(rv.name)]
                 + [d.name for d in model.deterministics])
    diag = {'max_r_hat': float(az.rhat(trace, var_names=diag_vars).to_array().max()),
            'min_ess_bulk': float(az.ess(trace, var_names=diag_vars).to_array().min()),
            'divergences': int(trace.sample_stats['diverging'].sum())}
    return loo, diag

def elpd_diff(loo_x, loo_y):
    """elpd_loo(X) - elpd_loo(Y) and its standard error from the pointwise differences."""
    diff_i = loo_x.loo_i.values - loo_y.loo_i.values   # same trials in the same order
    return diff_i.sum(), np.sqrt(len(diff_i) * np.var(diff_i))

#%% FIT ALL THREE MODELS TO ALL THREE DATASETS IN EVERY REPLICATE

rows = []

def save_results():
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({'results': pd.DataFrame(rows), 'groups_AB': groups_AB, 'hands': HANDS,
                     'true_A': true_A, 'true_B': true_B, 'true_C': true_C, 'n_reps': N_REPS,
                     'draws': DRAWS, 'tune': TUNE, 'chains': CHAINS, 'seed': SEED}, f)

run_start = time.time()
for rep in range(N_REPS):
    data = {data_key: SIMULATORS[data_key](rng) for data_key in MODEL_KEYS}
    for data_key, obs in data.items():
        row = {'rep': rep, 'true_model': data_key, 'mean_resp': obs.mean()}
        loo = {}
        for model_key in MODEL_KEYS:
            t0 = time.time()
            print(f"--- replicate {rep+1}/{N_REPS}: model {model_key} on data {data_key} ---")
            loo[model_key], diag = fit_and_loo(model_key, obs)
            row.update({f'elpd_{model_key}': loo[model_key].elpd_loo, f'se_{model_key}': loo[model_key].se,
                        f'p_loo_{model_key}': loo[model_key].p_loo,
                        f'max_pareto_k_{model_key}': float(loo[model_key].pareto_k.max())})
            row.update({f'{k}_{model_key}': v for k, v in diag.items()})
            row[f'minutes_{model_key}'] = (time.time() - t0) / 60
            print(f"    done in {(time.time() - t0)/60:.1f} min, {diag['divergences']} divergences, "
                  f"max r_hat {diag['max_r_hat']:.3f}")
        for x, y in PAIRS:
            row[f'elpd_diff_{x}{y}'], row[f'dse_{x}{y}'] = elpd_diff(loo[x], loo[y])
        row['selected'] = max(MODEL_KEYS, key=lambda m: loo[m].elpd_loo)
        rows.append(row)
    save_results()
    print(f"=== replicate {rep+1}/{N_REPS} saved ({(time.time() - run_start)/3600:.2f} h so far) ===")

results = pd.DataFrame(rows)
print(f"FINISHED SAMPLING! ({(time.time() - run_start)/3600:.1f} h)")

#%% (optional) reload saved results instead of refitting

# with open(RESULTS_PATH, "rb") as f:
#     saved = pickle.load(f)
# results, N_REPS = saved['results'], saved['n_reps']

#%% Summary tables, written to Model_Recov_C_Results.txt

for x, y in PAIRS:
    results[f'z_{x}{y}'] = results[f'elpd_diff_{x}{y}'] / results[f'dse_{x}{y}']
selection = pd.crosstab(results['true_model'], results['selected']).reindex(
    index=MODEL_KEYS, columns=MODEL_KEYS, fill_value=0)
selection.index.name, selection.columns.name = 'generating model', 'selected model'

per_rep = results[['rep', 'true_model'] + [f'elpd_{m}' for m in MODEL_KEYS] + [f'p_loo_{m}' for m in MODEL_KEYS]
                  + [f'{q}_{x}{y}' for x, y in PAIRS for q in ('elpd_diff', 'dse', 'z')] + ['selected']].copy()
per_rep['rep'] += 1
per_rep = per_rep.sort_values(['true_model', 'rep'])

z_summary = results.groupby('true_model')[[f'z_{x}{y}' for x, y in PAIRS]].agg(['min', 'median', 'max'])
z_summary.index.name = 'generating model'

diag_cols = [f'{d}_{m}' for d in ['max_r_hat', 'min_ess_bulk', 'divergences', 'max_pareto_k'] for m in MODEL_KEYS]
diag_summary = results.groupby('true_model')[diag_cols].agg(
    {c: ('min' if c.startswith('min_ess') else 'max') for c in diag_cols}).T
diag_summary.columns.name = 'generating model'

true_table = "\n\n".join([
    "Model A (data A), group parameters\n" + pd.DataFrame(true_A, index=groups_AB).T.to_string(float_format=lambda v: f"{v:.3f}"),
    "Model B (data B), hyperparameters\n" + pd.DataFrame(true_B, index=groups_AB).T.to_string(float_format=lambda v: f"{v:.3f}"),
    "Model C (data C), hyperparameters\n" + true_C.to_string(float_format=lambda v: f"{v:.3f}")])

summary = "\n\n".join([
    "Model recovery study, Model C (distractor as covariate) against Models A (pooled) and B (hierarchical)",
    f"Produced by Model_Recov_C.py: {len(results) // len(MODEL_KEYS)} replicates, seed {SEED}, "
    f"{CHAINS} chains x {DRAWS} draws ({TUNE} tune). Each replicate draws a fresh dataset from each "
    f"model on the real design ({n_trials} trials, {n_sessions} sessions); all three models are fit to each.",
    "True values (posterior means of the real-data fits; gamma hyperparameters on the logit scale)\n\n" + true_table,
    "elpd_diff_XY = elpd_loo(X) - elpd_loo(Y); dse_XY = its standard error; z_XY = elpd_diff_XY / dse_XY.\n"
    "Positive elpd_diff_XY favours X. 'selected' is the model with the highest elpd_loo.",
    "Selection counts\n" + selection.to_string(),
    "z across replicates\n" + z_summary.round(2).to_string(),
    "Per replicate\n" + per_rep.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
    "Sampling and LOO diagnostics (worst across replicates; _A / _B / _C = fitted model). r_hat and ESS\n"
    "leave out Model B's z_sig_*, whose sign is unidentified.\n"
    + diag_summary.to_string(float_format=lambda v: f"{v:.3f}"),
])
SUMMARY_PATH.write_text(summary + "\n")
print(summary)

#%% Plot: elpd_diff C - A (top) and C - B (bottom) with +-2 dse for every replicate, one column per generating model

MODEL_COLORS = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a'}

fig, axes = plt.subplots(2, len(MODEL_KEYS), constrained_layout=True, figsize=(14, 7.5))
for row_axes, (x, y) in zip(axes, PAIRS[:2]):
    for ax, true_model in zip(row_axes, MODEL_KEYS):
        d = results[results['true_model'] == true_model].sort_values('rep')
        ax.errorbar(d['rep'] + 1, d[f'elpd_diff_{x}{y}'], yerr=2 * d[f'dse_{x}{y}'], fmt='o', markersize=6,
                    linewidth=1.5, capsize=0, color=MODEL_COLORS[true_model])
        ax.axhline(0, linestyle='--', linewidth=1.5, color='#52514e')
        ax.set_xticks(d['rep'] + 1)
        ax.grid(alpha=0.3)
    row_axes[0].set_ylabel(f'elpd_loo({x}) − elpd_loo({y})\n(> 0 favours {x})')
for ax, true_model in zip(axes[0], MODEL_KEYS):
    d = results[results['true_model'] == true_model]
    n_correct = int((d['selected'] == true_model).sum())
    ax.set_title(f'Data generated by Model {true_model}:\n{n_correct}/{len(d)} select {true_model}')
for ax in axes[1]:
    ax.set_xlabel('replicate')
# legend in neutral ink: the panel colours differ, the encoding is the same
handles = [matplotlib.lines.Line2D([], [], marker='o', markersize=6, linewidth=1.5, color='#52514e'),
           matplotlib.lines.Line2D([], [], linestyle='--', linewidth=1.5, color='#52514e')]
fig.legend(handles, ['elpd_diff ± 2 dse', 'no difference'], loc='outside lower center', ncol=2)
fig.suptitle('Model recovery: LOO comparison of Model C with Models A and B', fontsize=14)
fig.savefig(PLOT_DIR / "elpd_diff_by_replicate.png", dpi=200, bbox_inches='tight')
plt.show()

#%% Numbers for Model_Recov_C_Report.tex
# Everything quoted in the report that Model_Recov_C_Results.txt and the plot do not show. Works on freshly
# fitted or reloaded results.

for x, y in PAIRS:
    results[f'z_{x}{y}'] = results[f'elpd_diff_{x}{y}'] / results[f'dse_{x}{y}']
for x, y in PAIRS:
    print(f"--- elpd_diff ({x} - {y}) and dse by generating model")
    print(results.groupby('true_model')[[f'elpd_diff_{x}{y}', f'dse_{x}{y}', f'z_{x}{y}']]
          .agg(['min', 'median', 'max']).round(2).to_string())
    for true_model in MODEL_KEYS:
        d = results[results['true_model'] == true_model]
        z = d[f'z_{x}{y}']
        print(f"data {true_model}: z > 2 for {x}: {int((z > 2).sum())}, z < -2 for {y}: {int((z < -2).sum())}, "
              f"|z| <= 2: {int((z.abs() <= 2).sum())} of {len(d)}")
print("--- p_loo by fitted and generating model (min-max)")
for m in MODEL_KEYS:
    print(f"Model {m}: " + '; '.join(f"data {t} {d[f'p_loo_{m}'].min():.1f}-{d[f'p_loo_{m}'].max():.1f}"
                                     for t, d in results.groupby('true_model')))
print("--- LOO reliability")
k_cols = [f'max_pareto_k_{m}' for m in MODEL_KEYS]
print(f"max Pareto k over all fits: {results[k_cols].max().max():.3f}; fits with k > 0.7: {int((results[k_cols] > 0.7).sum().sum())}")
print("--- Sampling")
print("max r_hat: " + ', '.join(f"{m} {results[f'max_r_hat_{m}'].max():.3f}" for m in MODEL_KEYS)
      + "; min bulk ESS: " + ', '.join(f"{m} {results[f'min_ess_bulk_{m}'].min():.0f}" for m in MODEL_KEYS))
print("divergences, total and fits with any: " + ', '.join(
    f"{m} {int(results[f'divergences_{m}'].sum())} / {int((results[f'divergences_{m}'] > 0).sum())}" for m in MODEL_KEYS))
print("mean minutes per fit:", {m: round(float(results[f'minutes_{m}'].mean()), 1) for m in MODEL_KEYS})
print("--- Real data, for comparison (Results_A.pkl, Results_B.pkl, Results_C.pkl)")
loo_real = {'A': real_fit_A['az_loo_trace'], 'B': real_fit_B['az_loo_trace'], 'C': real_fit_C['az_loo_trace']}
print("real data: " + '; '.join(f"elpd_loo {m} {loo_real[m].elpd_loo:.1f}, p_loo {loo_real[m].p_loo:.1f}" for m in MODEL_KEYS))
for x, y in PAIRS:
    real_diff, real_dse = elpd_diff(loo_real[x], loo_real[y])
    print(f"real {x} - {y}: {real_diff:.1f} (dse {real_dse:.1f}, z {real_diff / real_dse:.2f}); replicates: "
          + '; '.join(f"data {t} {d[f'elpd_diff_{x}{y}'].min():.1f} to {d[f'elpd_diff_{x}{y}'].max():.1f}"
                      for t, d in results.groupby('true_model')))
