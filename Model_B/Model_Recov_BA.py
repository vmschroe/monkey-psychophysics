#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:09:04 2026

Model recovery study for Models A (pooled over sessions) and B (hierarchical
over sessions): can PSIS-LOO tell which model generated a dataset?

Each replicate draws two fresh synthetic datasets on the real trial design
(stimuli, groups and sessions of ReadyData_Sirius_B.pkl), with true values
taken from the fits to the real data so the synthetic data resemble it:
    - data A: responses from Model A's group parameters (b0, b1, gam_h,
      gam_l), the same for every session; true values are the posterior means
      of Model A's real-data fit (Results_A.pkl, from Main_Analysis_A.py);
    - data B: new session-level parameters drawn from Model B's
      hyperparameters, as in DataProcessing_B.py, then responses from those;
      true hyperparameters are the posterior means of Model B's real-data fit
      (Results_B.pkl, from Main_Analysis_B.py).
These true values only generate data; the priors of both models are unchanged.
Both models are fitted to both datasets and compared with PSIS-LOO.
elpd_diff is elpd_loo(B) - elpd_loo(A) on the same data, with its standard
error dse computed from the pointwise differences, as az.compare does.
The model with the higher elpd_loo is counted as the one selected.

Outputs, all written by this script:
    - Results_Model_Recov_BA.pkl: one row per (replicate, dataset) with both
      models' LOO results and sampling diagnostics;
    - Model_Recov_Results.txt: settings, per-replicate table, selection
      counts and diagnostics;
    - Plots_B/Model_Recov_Plots_BA/elpd_diff_by_replicate.png.

Runtime: about 1 min for a Model A fit and 3 min for a Model B fit with the
4 chains run in parallel, so about 8.5 min per replicate and 1.4 h for the
default 10 replicates. Results are saved
after every replicate, so an interrupted run keeps the replicates already done.

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
DATA_DIR = MODEL_DIR / "Data_B"
MODEL_A_DIR = MODEL_DIR.parent / "Model_A"
PLOT_DIR = MODEL_DIR / "Plots_B" / "Model_Recov_Plots_BA"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = MODEL_DIR / "Results_Model_Recov_BA.pkl"
SUMMARY_PATH = MODEL_DIR / "Model_Recov_Results.txt"

#%% SETTINGS

N_REPS = 10
DRAWS = 1000
TUNE = 1000
CHAINS = 4
CORES = 4
# Python 3.14's default 'forkserver' re-runs this script in every worker, so fork them instead
# ('fork' doesn't exist on Windows, which keeps its default)
MP_CTX = None if os.name == 'nt' else 'fork'
SEED = 20261012

rng = np.random.default_rng(SEED)

#%% LOAD DESIGN AND TRUE PARAMETERS

with open(DATA_DIR / "ReadyData_Sirius_B.pkl", "rb") as f:
    real_data = pickle.load(f)

cov_mat = real_data['cov_mat']
grp_idx = real_data['grp_idx']
sess_idx = real_data['sess_idx']
sessions = list(real_data['dates_sess_idx'])
n_sessions = len(sessions)
names_grp_idx = real_data['names_grp_idx']
groups = sorted(names_grp_idx, key=names_grp_idx.get)   # ['left_uni','left_bi','right_uni','right_bi']

with open(MODEL_A_DIR / "Results_A.pkl", "rb") as f:
    real_fit_A = pickle.load(f)['az_summary_trace']['mean']
with open(MODEL_DIR / "Results_B.pkl", "rb") as f:
    real_fit_B = pickle.load(f)['az_summary_trace']['mean']

def posterior_means(fit, var, beta=None):
    """Posterior means of one variable from a real-data fit, in the order of groups."""
    return np.array([fit[f"{var}[{beta + ', ' if beta else ''}{grp}]"] for grp in groups])

# Model A truth: one value per group
true_A = {'b0': posterior_means(real_fit_A, 'beta_vec', 'b0'),
          'b1': posterior_means(real_fit_A, 'beta_vec', 'b1'),
          'gam_h': posterior_means(real_fit_A, 'gam_h'),
          'gam_l': posterior_means(real_fit_A, 'gam_l')}

# Model B truth: hyperparameters per group, gamma ones on the logit scale.
# Build_Model_B.py indexes the gamma hyperparameters with the 'betas' dim: b0 -> gam_h, b1 -> gam_l
true_hyper = {'mu_b0': posterior_means(real_fit_B, 'mu_betas', 'b0'),
              'sig_b0': posterior_means(real_fit_B, 'sig_betas', 'b0'),
              'mu_b1': posterior_means(real_fit_B, 'mu_betas', 'b1'),
              'sig_b1': posterior_means(real_fit_B, 'sig_betas', 'b1'),
              'mu_gam_h': posterior_means(real_fit_B, 'mu_gams', 'b0'),
              'sig_gam_h': posterior_means(real_fit_B, 'sig_gams', 'b0'),
              'mu_gam_l': posterior_means(real_fit_B, 'mu_gams', 'b1'),
              'sig_gam_l': posterior_means(real_fit_B, 'sig_gams', 'b1')}

#%% DATA GENERATORS

def psychometric(gam_h, gam_l, b0, b1, stim):
    return gam_h + (1 - gam_h - gam_l) / (1 + np.exp(-(b0 + b1 * stim)))

def simulate_A(rng):
    """Responses from Model A's true group parameters, as in DataProcessing_A.py."""
    psi = psychometric(*(true_A[p][grp_idx] for p in ['gam_h', 'gam_l', 'b0', 'b1']), cov_mat[:, 1])
    return rng.binomial(1, psi)

def simulate_B(rng):
    """New session-level parameters from Model B's true hyperparameters, then
    responses, as in DataProcessing_B.py. Session parameters are (n_groups, n_sessions)."""
    shape = (len(groups), n_sessions)
    hp = {name: vals[:, None] for name, vals in true_hyper.items()}
    sp = {
        'b0': hp['mu_b0'] + hp['sig_b0'] * rng.standard_normal(shape),
        'b1': hp['mu_b1'] + hp['sig_b1'] * rng.standard_normal(shape),
        'gam_h': 0.25 / (1 + np.exp(-(hp['sig_gam_h'] * rng.standard_normal(shape) + hp['mu_gam_h']))),
        'gam_l': 0.25 / (1 + np.exp(-(hp['sig_gam_l'] * rng.standard_normal(shape) + hp['mu_gam_l']))),
    }
    psi = psychometric(*(sp[p][grp_idx, sess_idx] for p in ['gam_h', 'gam_l', 'b0', 'b1']), cov_mat[:, 1])
    return rng.binomial(1, psi)

#%% BUILD BOTH MODELS

# The build scripts read these names; running each in its own namespace keeps them
# from overwriting anything here. The design is shared, so each model is built once
# and only the responses are swapped between fits.
def build(path, model_name):
    ns = {'cov_mat': cov_mat, 'grp_idx': grp_idx, 'sess_idx': sess_idx, 'sessions': sessions,
          'obs_data': np.zeros(len(grp_idx), dtype=int)}
    exec(path.read_text(), ns)
    return ns[model_name]

MODELS = {'A': build(MODEL_A_DIR / "Build_Model_A.py", 'model_A'),
          'B': build(MODEL_DIR / "Build_Model_B.py", 'model_B')}

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
    # opposite signs (r_hat ~1.5-1.7) without affecting anything else. Check the deterministics
    # (sig_betas, sig_gams, beta_vec, ...) in their place.
    diag_vars = ([rv.name for rv in model.free_RVs if not rv.name.startswith('z_sig_')]
                 + [d.name for d in model.deterministics])
    diag = {'max_r_hat': float(az.rhat(trace, var_names=diag_vars).to_array().max()),
            'min_ess_bulk': float(az.ess(trace, var_names=diag_vars).to_array().min()),
            'divergences': int(trace.sample_stats['diverging'].sum())}
    # kept to show the sign flips: NaN for Model A, which has no z_sig_*
    sign_vars = [rv.name for rv in model.free_RVs if rv.name.startswith('z_sig_')]
    diag['max_r_hat_z_sig'] = float(az.rhat(trace, var_names=sign_vars).to_array().max()) if sign_vars else np.nan
    return loo, diag

#%% FIT BOTH MODELS TO BOTH DATASETS IN EVERY REPLICATE

rows = []

def save_results():
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({'results': pd.DataFrame(rows), 'groups': groups, 'true_hyper': true_hyper,
                     'true_A': true_A, 'n_reps': N_REPS,
                     'draws': DRAWS, 'tune': TUNE, 'chains': CHAINS, 'seed': SEED}, f)

run_start = time.time()
for rep in range(N_REPS):
    data = {'A': simulate_A(rng), 'B': simulate_B(rng)}
    for data_key, obs in data.items():
        row = {'rep': rep, 'true_model': data_key, 'mean_resp': obs.mean()}
        loo = {}
        for model_key in MODELS:
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
        diff_i = loo['B'].loo_i.values - loo['A'].loo_i.values   # same trials in the same order
        row['elpd_diff'] = diff_i.sum()                          # B - A
        row['dse'] = np.sqrt(len(diff_i) * np.var(diff_i))
        row['selected'] = 'B' if row['elpd_diff'] > 0 else 'A'
        rows.append(row)
    save_results()

results = pd.DataFrame(rows)
print(f"FINISHED SAMPLING! ({(time.time() - run_start)/3600:.1f} h)")

#%% (optional) reload saved results instead of refitting

# with open(RESULTS_PATH, "rb") as f:
#     saved = pickle.load(f)
# results, N_REPS = saved['results'], saved['n_reps']

#%% Summary tables, written to Model_Recov_Results.txt

results['z'] = results['elpd_diff'] / results['dse']
selection = pd.crosstab(results['true_model'], results['selected']).reindex(
    index=['A', 'B'], columns=['A', 'B'], fill_value=0)
selection.index.name, selection.columns.name = 'generating model', 'selected model'

per_rep = results[['rep', 'true_model', 'elpd_A', 'elpd_B', 'p_loo_A', 'p_loo_B',
                   'elpd_diff', 'dse', 'z', 'selected']].copy()
per_rep['rep'] += 1
per_rep = per_rep.sort_values(['true_model', 'rep'])

z_summary = results.groupby('true_model')['z'].agg(['min', 'median', 'max'])
z_summary.index.name = 'generating model'

diag_cols = ([f'{d}_{m}' for d in ['max_r_hat', 'min_ess_bulk', 'divergences', 'max_pareto_k'] for m in MODELS]
             + ['max_r_hat_z_sig_B'])
diag_summary = results.groupby('true_model')[diag_cols].agg(
    {c: ('min' if c.startswith('min_ess') else 'max') for c in diag_cols}).T
diag_summary.columns.name = 'generating model'

true_table = pd.concat({'Model A (data A)': pd.DataFrame(true_A, index=groups).T,
                        'Model B (data B)': pd.DataFrame(true_hyper, index=groups).T})

summary = "\n\n".join([
    "Model recovery study, Models A (pooled) and B (hierarchical over sessions)",
    f"Produced by Model_Recov_BA.py: {len(results) // 2} replicates, seed {SEED}, "
    f"{CHAINS} chains x {DRAWS} draws ({TUNE} tune). Each replicate draws a fresh dataset from each "
    f"model on the real design ({len(grp_idx)} trials, {n_sessions} sessions).",
    "True values (posterior means of the real-data fits; gamma hyperparameters on the logit scale)\n"
    + true_table.to_string(float_format=lambda v: f"{v:.3f}"),
    "elpd_diff = elpd_loo(B) - elpd_loo(A); dse = its standard error; z = elpd_diff / dse.\n"
    "Positive elpd_diff favours Model B. 'selected' is the model with the higher elpd_loo.",
    "Selection counts\n" + selection.to_string(),
    "z = elpd_diff / dse across replicates\n" + z_summary.round(2).to_string(),
    "Per replicate\n" + per_rep.to_string(index=False, float_format=lambda v: f"{v:.2f}"),
    "Sampling and LOO diagnostics (worst across replicates; _A / _B = fitted model). r_hat and ESS\n"
    "leave out Model B's z_sig_*, whose sign is unidentified; max_r_hat_z_sig_B shows them separately.\n"
    + diag_summary.to_string(float_format=lambda v: f"{v:.3f}"),
])
SUMMARY_PATH.write_text(summary + "\n")
print(summary)

#%% Plot: elpd_diff (B - A) with +-2 dse for every replicate, one panel per generating model

MODEL_COLORS = {'A': '#2a78d6', 'B': '#eb6834'}

fig, axes = plt.subplots(1, 2, sharey=False, constrained_layout=True, figsize=(11, 4.2))
for ax, true_model in zip(axes, ['A', 'B']):
    d = results[results['true_model'] == true_model].sort_values('rep')
    ax.errorbar(d['rep'] + 1, d['elpd_diff'], yerr=2 * d['dse'], fmt='o', markersize=6,
                linewidth=1.5, capsize=0, color=MODEL_COLORS[true_model])
    ax.axhline(0, linestyle='--', linewidth=1.5, color='#52514e')
    n_correct = int((d['selected'] == true_model).sum())
    ax.set_title(f'Data generated by Model {true_model}: {n_correct}/{len(d)} select {true_model}')
    ax.set_xlabel('replicate')
    ax.set_xticks(d['rep'] + 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('elpd_loo(B) − elpd_loo(A)\n(> 0 favours B)')
# legend in neutral ink: the panel colours differ, the encoding is the same
handles = [matplotlib.lines.Line2D([], [], marker='o', markersize=6, linewidth=1.5, color='#52514e'),
           matplotlib.lines.Line2D([], [], linestyle='--', linewidth=1.5, color='#52514e')]
fig.legend(handles, ['elpd_diff ± 2 dse', 'no difference'], loc='outside lower center', ncol=2)
fig.suptitle('Model recovery: LOO comparison of Models A and B', fontsize=14)
fig.savefig(PLOT_DIR / "elpd_diff_by_replicate.png", dpi=200, bbox_inches='tight')
plt.show()

#%% Numbers for Model_Recov_BA_Report.txt
# Everything quoted in the report that Model_Recov_Results.txt and the plot do not show. Works on freshly
# fitted or reloaded results.

results['z'] = results['elpd_diff'] / results['dse']
print("--- elpd_diff (B - A) and dse by generating model")
print(results.groupby('true_model')[['elpd_diff', 'dse', 'z']].agg(['min', 'median', 'max']).round(2).to_string())
for true_model in ['A', 'B']:
    d = results[results['true_model'] == true_model]
    print(f"data {true_model}: |z| > 2 in favour of A: {int((d['z'] < -2).sum())}, in favour of B: {int((d['z'] > 2).sum())}, "
          f"|z| <= 2: {int((d['z'].abs() <= 2).sum())} of {len(d)}")
print("--- p_loo by fitted and generating model (min-max)")
for m in MODELS:
    print(f"Model {m}: " + '; '.join(f"data {t} {d[f'p_loo_{m}'].min():.1f}-{d[f'p_loo_{m}'].max():.1f}"
                                     for t, d in results.groupby('true_model')))
print("--- LOO reliability")
k_cols = [f'max_pareto_k_{m}' for m in MODELS]
print(f"max Pareto k over all fits: {results[k_cols].max().max():.3f}; fits with k > 0.7: {int((results[k_cols] > 0.7).sum().sum())}")
print("--- Sampling")
print(f"max r_hat (z_sig excluded): A {results['max_r_hat_A'].max():.3f}, B {results['max_r_hat_B'].max():.3f}; "
      f"min bulk ESS: A {results['min_ess_bulk_A'].min():.0f}, B {results['min_ess_bulk_B'].min():.0f}; "
      f"divergences: {int(results[['divergences_A', 'divergences_B']].sum().sum())}")
print("max r_hat of Model B's z_sig_* by generating model:",
      results.groupby('true_model')['max_r_hat_z_sig_B'].agg(['min', 'max']).round(3).to_dict('index'))
print("mean minutes per fit:", {m: round(float(results[f'minutes_{m}'].mean()), 1) for m in MODELS})
print("--- Real data, for comparison (Results_A.pkl, Results_B.pkl)")
with open(MODEL_A_DIR / "Results_A.pkl", "rb") as f:
    loo_real_A = pickle.load(f)['az_loo_trace']
with open(MODEL_DIR / "Results_B.pkl", "rb") as f:
    loo_real_B = pickle.load(f)['az_loo_trace']
diff_i = loo_real_B.loo_i.values - loo_real_A.loo_i.values
real_diff, real_dse = diff_i.sum(), np.sqrt(len(diff_i) * np.var(diff_i))
print(f"real data: elpd_loo A {loo_real_A.elpd_loo:.1f}, B {loo_real_B.elpd_loo:.1f}; p_loo A {loo_real_A.p_loo:.1f}, "
      f"B {loo_real_B.p_loo:.1f}; elpd_diff {real_diff:.1f}, dse {real_dse:.1f}, z {real_diff / real_dse:.2f}")
for true_model in ['A', 'B']:
    d = results[results['true_model'] == true_model]
    print(f"real elpd_diff vs data-{true_model} replicates: range {d['elpd_diff'].min():.1f} to {d['elpd_diff'].max():.1f}")
