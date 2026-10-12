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

@author: vmschroe
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

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
