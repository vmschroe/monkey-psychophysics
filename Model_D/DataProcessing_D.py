#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 2026

Format the real data for Model D (hierarchical over sessions, one group per
hand, lapse rates per condition, PSE shift with a distractor-presence term).
This is the observational space of Step 2 of the workflow.

Each trial gives one binary response y (1 = "high"), with the metadata:
    - x_stim       (stim_amp normalized as in Models A, B and C)
    - cond_idx     (0 = unimanual, 1 = bimanual)
    - grp_idx      (hand: 0 = left, 1 = right)
    - sess_idx     (session)
    - trial_pos    (order within the session, among the kept trials)
    - high_right   (1 if the "high" target is on the right of the screen)
The last two are not in the model; they are used by the summary statistics.

Every session uses a single distractor amplitude on its bimanual trials;
sess_x_dist holds it for each session, on the x_stim scale but not centered.

Model D's calibration uses an ensemble simulated from its prior, so unlike
Models B and C no synthetic dataset is derived from an earlier fit here.

@author: vmschroe
"""

import ast
import numpy as np
import pickle
from pathlib import Path

try:
    MODEL_DIR = Path(__file__).resolve().parent
except NameError:  # __file__ isn't set when cells are run interactively
    MODEL_DIR = Path.cwd()
DATA_DIR = MODEL_DIR / "Data_D"
DATA_DIR.mkdir(exist_ok=True)

#%% Load raw data

with open(MODEL_DIR.parent / "Sirius_Data.pkl", "rb") as f:
    DataDict = pickle.load(f)

Raw_DataFrame = DataDict['data'].reset_index(drop=True)
sess_sum = DataDict['session_summary']

#%% Format experimental data

stims = np.array(Raw_DataFrame['stim_amp'])
dists = np.array(Raw_DataFrame['dist_amp'])

# normalize stim amps, as in Models A, B and C
x_old = np.unique(stims)
x_mu = np.mean(x_old)
x_sig = np.std(x_old)
x_stim = (stims - x_mu) / x_sig

cond_idx = (dists > 0).astype(int)
names_cond_idx = {'uni': 0, 'bi': 1}

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

# order within the session (a date with two recording files keeps file order)
assert Raw_DataFrame['sess_idx'].is_monotonic_increasing
trial_pos = Raw_DataFrame.groupby('sess_idx').cumcount().to_numpy()

# screen side of the "high" target: target IDs 3 and 4 are "high", 1 and 2 "low"
first_is_high = Raw_DataFrame['choice1_ID'].isin([3, 4]).to_numpy()
assert (first_is_high != Raw_DataFrame['choice2_ID'].isin([3, 4]).to_numpy()).all()
pos_first = Raw_DataFrame['choice1_Position'].map(lambda s: ast.literal_eval(s)[0]).to_numpy()
high_right = (np.where(first_is_high, pos_first, -pos_first) > 0).astype(int)

# binary responses
resp = np.array(Raw_DataFrame['response'])
assert (resp == Raw_DataFrame['selectedChoiceTarget_ID'].isin([3, 4]).to_numpy().astype(int)).all()

#%% Save processed data

ReadyData_Sirius_D = {'x_stim': x_stim,
                      'stim_amp': stims,
                      'cond_idx': cond_idx,
                      'grp_idx': grp_idx,
                      'sess_idx': sess_idx,
                      'sess_x_dist': sess_x_dist,
                      'trial_pos': trial_pos,
                      'high_right': high_right,
                      'resp': resp,
                      'names_cond_idx': names_cond_idx,
                      'names_grp_idx': names_grp_idx,
                      'dates_sess_idx': dates_sess_idx,
                      'session_summary': sess_sum,
                      'x_mu': x_mu,
                      'x_sig': x_sig}

with open(DATA_DIR / "ReadyData_Sirius_D.pkl", "wb") as f:
    pickle.dump(ReadyData_Sirius_D, f)
