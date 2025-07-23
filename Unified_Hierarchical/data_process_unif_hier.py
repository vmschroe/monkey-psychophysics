# -*- coding: utf-8 -*-
"""
Data processing for unified hierarchical model with distractability

Works but could be better

Could use class here to simplify:
    Specify data groupings to index (side, session)
    Specify covariates to include (1, x_stim, x_distract)
    
Created on Wed Jul 23 12:11:46 2025

@author: schro
"""

import pickle
import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
import xarray as xr
import sys
import numpy as np

#%% load data

with open(r"Sirius_DFS.pkl", "rb") as f:
    data = pickle.load(f)  
    
#%% process data
x_orig = np.array([6, 12, 18, 24, 32, 38, 44, 50])
amp_mu = np.mean(x_orig)
amp_sig = np.std(x_orig)


sessions = list(data.keys())



#%%

sess_cov_mats_left = []
sess_obs_data_left = []
Nt_left = []
sess_cov_mats_right = []
sess_obs_data_right = []
Nt_right = []

sess_distAMP= []




for sess in sessions:
    sessdf = data[sess].copy(deep=True)[['stimSIDE', 'lowORhighGUESS','stimAMP', 'distAMP']]
    
    sessdf['stimAMP'] = (sessdf['stimAMP']-amp_mu)/amp_sig
    sessdf['distAMP'] = (sessdf['distAMP'])/amp_sig
    
    sess_distAMP.append(np.max(sessdf['distAMP']))
    
    left_arr = np.array(sessdf[sessdf['stimSIDE']=='left'][['lowORhighGUESS','stimAMP', 'distAMP']])
    left_resp = np.array(sessdf[sessdf['stimSIDE']=='left']['lowORhighGUESS'])
    sess_obs_data_left.append(left_resp)
    left_arr[:,0] = 1
    sess_cov_mats_left.append(left_arr)
    Nt_left.append(left_resp.shape[0])
    
    right_arr = np.array(sessdf[sessdf['stimSIDE']=='right'][['lowORhighGUESS','stimAMP', 'distAMP']])
    right_resp = np.array(sessdf[sessdf['stimSIDE']=='right']['lowORhighGUESS'])
    sess_obs_data_right.append(right_resp)
    right_arr[:,0] = 1
    sess_cov_mats_right.append(right_arr)
    Nt_right.append(right_resp.shape[0])

Nt_left = np.array(Nt_left)
Nt_right = np.array(Nt_right)
sess_distAMP = np.array(sess_distAMP)


#%%
def setup_concatenated_data(session_covariate_matrices, session_obs_data, Nt):
    """
    Convert per-session data to concatenated format for PyMC
    
    Parameters:
    - session_covariate_matrices: list of arrays, each with shape (Nt[sess], 3)
    - session_obs_data: list of arrays, each with shape (Nt[sess],)
    """
    
    # Concatenate all covariate matrices
    covariate_matrix = np.vstack(session_covariate_matrices)
    
    # Concatenate all observation data
    obs_data = np.concatenate(session_obs_data)
    
    # Create session index array
    sessions_idx = np.concatenate([
        np.full(Nt[sess], sess) for sess in range(len(Nt))
    ])
    
    print(f"Covariate matrix shape: {covariate_matrix.shape}")
    print(f"Observations shape: {obs_data.shape}")
    print(f"Sessions index shape: {sessions_idx.shape}")
    print(f"Sessions index range: {sessions_idx.min()} to {sessions_idx.max()}")
    
    return covariate_matrix, obs_data, sessions_idx



# Set up data structures
cov_mat_left, obs_data_left, sessions_idx_left = setup_concatenated_data(
    sess_cov_mats_left, sess_obs_data_left, Nt_left
)

cov_mat_right, obs_data_right, sessions_idx_right = setup_concatenated_data(
    sess_cov_mats_right, sess_obs_data_right, Nt_right
)


sides_idx = np.concatenate([np.full(sum(Nt_left), 0), np.full(sum(Nt_right), 1)])
obs_data = np.concatenate([obs_data_left, obs_data_right])
sessions_idx = np.concatenate([sessions_idx_left, sessions_idx_right])

cov_mat = np.vstack([cov_mat_left,cov_mat_right])

#%%

processed_data = {
    'side_index': sides_idx,
    'obs_data': obs_data,
    'sessions_idx': sessions_idx,
    'sess_distAMP': sess_distAMP,
    'cov_mat': cov_mat,
    'sessions': sessions
    }

#%%



with open('data_unif_hier.pkl', 'wb') as f:
    pickle.dump(processed_data, f)




