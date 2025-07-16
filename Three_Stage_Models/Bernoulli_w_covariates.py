# -*- coding: utf-8 -*-
"""
Processing Data

Created on Tue Jul 15 16:44:00 2025

@author: schro
"""


#%% packages
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
lefts = []
rights = []

for sess in sessions:
    sessdf = data[sess].copy(deep=True)[['stimSIDE', 'lowORhighGUESS','stimAMP', 'distAMP']]
    
    sessdf['stimAMP'] = (sessdf['stimAMP']-amp_mu)/amp_sig
    sessdf['distAMP'] = (sessdf['distAMP'])/amp_sig
    left_arr = np.array(sessdf[sessdf['stimSIDE']=='left'][['lowORhighGUESS','stimAMP', 'distAMP']])
    right_arr = np.array(sessdf[sessdf['stimSIDE']=='right'][['lowORhighGUESS','stimAMP', 'distAMP']])
    lefts.append(left_arr)
    rights.append(right_arr)


df = pd.DataFrame( {'Session': sessions,
            'Left': lefts,
            'Right': rights})
df.set_index("Session", inplace=True)

#%% try just left hand
leftdf = df['Left']



coords = {'sessions': sessions, 'betas' : ["b0","b1","b2"]}

with pm.Model(coords=coords) as left_hier_nolapse_model:
    mu_betas = pm.Normal("mu_betas", mu=0, sigma=100, dims='betas')
    sig_betas = pm.Exponential("sig_betas", 10, dims='betas')
    beta_vec = pm.Normal("beta_vec", mu=mu_betas, sigma = sig_betas, dims=('sessions', 'betas'))
    logit_p = pm.Deterministic('logit_p', covariate_matrix[sessions_idx] @ beta_vec[sessions_idx])
    p = pm.Deterministic('p', pm.math.invlogit(logit_p))
    pm.Bernoulli("resp", p = p, observed = obs_data)
    
#%%

# Example data generation (replace with your actual data)
np.random.seed(42)
sessions = list(range(43))  # 43 sessions
Nt = np.random.randint(20, 100, 43)  # Random trial counts between 20-100 per session

# Method 1: Concatenated arrays (recommended for PyMC)
def setup_concatenated_data(session_covariate_matrices, session_obs_data):
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
    
    return covariate_matrix, obs_data, sessions_idx

# Example data generation for demonstration
session_covariate_matrices = []
session_obs_data = []

for sess in range(43):
    # Generate random covariate matrix for this session
    X = np.random.randn(Nt[sess], 3)
    X[:, 0] = 1  # intercept column
    session_covariate_matrices.append(X)
    
    # Generate random binary observations
    y = np.random.binomial(1, 0.5, Nt[sess])
    session_obs_data.append(y)

# Set up data structures
covariate_matrix, obs_data, sessions_idx = setup_concatenated_data(
    session_covariate_matrices, session_obs_data
)

print(f"Covariate matrix shape: {covariate_matrix.shape}")
print(f"Observations shape: {obs_data.shape}")
print(f"Sessions index shape: {sessions_idx.shape}")
print(f"Sessions index range: {sessions_idx.min()} to {sessions_idx.max()}")

# PyMC Model
coords = {
    'sessions': sessions, 
    'betas': ["b0", "b1", "b2"],
    'trials': range(len(obs_data))  # Add trials dimension
}

with pm.Model(coords=coords) as left_hier_nolapse_model:
    # Hyperpriors for session-level parameters
    mu_betas = pm.Normal("mu_betas", mu=0, sigma=2, dims='betas')
    sig_betas = pm.Exponential("sig_betas", 1, dims='betas')
    
    # Session-specific coefficients
    beta_vec = pm.Normal("beta_vec", 
                        mu=mu_betas, 
                        sigma=sig_betas, 
                        dims=('sessions', 'betas'))
    
    # Linear predictor using matrix multiplication
    # covariate_matrix[sessions_idx] selects the right rows
    # beta_vec[sessions_idx] selects the right session parameters
    logit_p = pm.Deterministic('logit_p', 
                              pm.math.sum(covariate_matrix * beta_vec[sessions_idx], axis=1),
                              dims='trials')
    
    # Probability
    p = pm.Deterministic('p', pm.math.invlogit(logit_p), dims='trials')
    
    # Likelihood
    resp = pm.Bernoulli("resp", p=p, observed=obs_data, dims='trials')

print("Model created successfully!")
print(f"Total trials across all sessions: {len(obs_data)}")
