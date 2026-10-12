#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 2026

Format the real data for Model C (hierarchical over sessions, one group per
hand, distractor amplitude as a covariate).

Covariate matrix columns:
    - 1        (intercept, b0)
    - x_stim   (stim_amp normalized as in Models A and B, b1)
    - x_dist   (dist_amp divided by the same sd as x_stim but not centered,
                so 0 = no distractor (unimanual trials), b2)

Every session uses a single distractor amplitude on its bimanual trials;
sess_x_dist holds that value of x_dist for each session.

Also generates the synthetic dataset ReadyData_Synth_C.pkl on the real design,
with true values derived from Model B's real-data fit (see SynthData_C.py).

@author: vmschroe
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from SynthData_C import GROUPS, true_hyper_from_B, sample_session_params, response_probs

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
DATA_DIR = MODEL_DIR / "Data_C"
DATA_DIR.mkdir(exist_ok=True)

#%% Load raw data

with open(MODEL_DIR.parent / "Sirius_Data.pkl", "rb") as f:
    DataDict = pickle.load(f)

Raw_DataFrame = DataDict['data']
sess_sum = DataDict['session_summary']

#%% Format experimental data

stims = np.array(Raw_DataFrame['stim_amp'])
dists = np.array(Raw_DataFrame['dist_amp'])

# normalize stim amps, as in Models A and B
x_old = np.unique(stims)
x_mu = np.mean(x_old)
x_sig = np.std(x_old)
stims_normed = (stims - x_mu) / x_sig

# distractor amps on the same scale, not centered: 0 stays "no distractor"
dists_normed = dists / x_sig

# construct covariate matrix
cov_mat = np.vstack([np.full_like(stims_normed, 1.0), stims_normed, dists_normed]).T

# hand group indices
grp_idx = np.array(Raw_DataFrame['side_idx'])
names_grp_idx = {'left': 0, 'right': 1}
assert all((Raw_DataFrame['side'] == name).eq(grp_idx == i).all() for name, i in names_grp_idx.items())

# session indices
sess_idx = np.array(Raw_DataFrame['sess_idx'])
dates_sess_idx = dict(zip(sess_sum.index.tolist(), sess_sum['sess_idx'].tolist()))

# distractor covariate of each session's bimanual trials, in session order
assert (Raw_DataFrame[Raw_DataFrame['dist_amp'] > 0].groupby('sess_idx')['dist_amp'].nunique() == 1).all()
sess_x_dist = sess_sum.sort_values('sess_idx')['dist_amp'].to_numpy() / x_sig

# binary responses
resp = np.array(Raw_DataFrame['response'])

#%% Save processed data

ReadyData_Sirius_C = {'cov_mat': cov_mat,
                      'grp_idx': grp_idx,
                      'sess_idx': sess_idx,
                      'sess_x_dist': sess_x_dist,
                      'resp': resp,
                      'names_grp_idx': names_grp_idx,
                      'dates_sess_idx': dates_sess_idx,
                      'session_summary': sess_sum,
                      'x_mu': x_mu,
                      'x_sig': x_sig}

with open(DATA_DIR / "ReadyData_Sirius_C.pkl", "wb") as f:
    pickle.dump(ReadyData_Sirius_C, f)

#%% Generate synthetic dataset

SEED = 20261012
rng = np.random.default_rng(SEED)

with open(MODEL_DIR.parent / "Model_B" / "Results_B.pkl", "rb") as f:
    means_B = pickle.load(f)['az_summary_trace']['mean']

sessions = list(dates_sess_idx)
hparams_fixed_C = true_hyper_from_B(means_B, sessions, sess_x_dist)
synth_session_params = sample_session_params(hparams_fixed_C, len(sessions), rng)
synth_psi = response_probs(synth_session_params, cov_mat, grp_idx, sess_idx)
synth_resp = rng.binomial(1, synth_psi)

print("True hyperparameters (gamma ones on the logit scale)")
print(hparams_fixed_C.round(3).to_string())

#%% Sanity checks: synthetic vs real response rates

condition = np.where(cov_mat[:, 2] > 0, 'bi', 'uni')
rates = pd.DataFrame({'hand': np.array(GROUPS)[grp_idx], 'condition': condition, 'stim_amp': stims,
                      'real': resp, 'synthetic': synth_resp, 'synthetic psi': synth_psi})
print("\nP(respond 'high') by hand and condition")
print(rates.groupby(['hand', 'condition'])[['real', 'synthetic', 'synthetic psi']].mean().round(3).to_string())
print("\nP(respond 'high') by stimulus amplitude, hand and condition")
print(rates.pivot_table(index='stim_amp', columns=['hand', 'condition'], values=['real', 'synthetic'])
      .round(2).to_string())

#%% Save synthetic dataset

ReadyData_Synth_C = {'cov_mat': cov_mat,
                     'grp_idx': grp_idx,
                     'sess_idx': sess_idx,
                     'sess_x_dist': sess_x_dist,
                     'resp': synth_resp,
                     'names_grp_idx': names_grp_idx,
                     'dates_sess_idx': dates_sess_idx,
                     'session_summary': sess_sum,
                     'x_mu': x_mu,
                     'x_sig': x_sig,
                     'hparams_fixed_C': hparams_fixed_C,
                     'synth_session_params': synth_session_params,
                     'seed': SEED}

with open(DATA_DIR / "ReadyData_Synth_C.pkl", "wb") as f:
    pickle.dump(ReadyData_Synth_C, f)
