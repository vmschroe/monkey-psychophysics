# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 17:17:36 2025

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
import aesara.tensor as at
import numpy as np

#%% PyMC Model
coords = {
    'sides': ['left','right'],
    'sessions': sessions, 
    'betas': ["b0", "b1", "b2"],
    'trials': range(len(obs_data))  # Add trials dimension
}

with pm.Model(coords=coords) as hier_nolapse_model:
    pm.MutableData("cov_mat", cov_mat, dims=("trials", "betas"))
    pm.MutableData("sessions_idx", sessions_idx, dims="trials")
    pm.MutableData("sides_idx", sides_idx, dims="trials")

    # Hyperpriors for session-level parameters
    global_mu = pm.Normal("global_mu", mu=[0,4,0], sigma=1, dims='betas')
    global_sig = pm.Exponential("global_sig", lam = 3, dims='betas')
    mu_betas = pm.Normal("mu_betas", mu=global_mu, sigma=global_sig, dims=('sides', 'betas'))
    sig_betas = pm.Exponential("sig_betas", lam = 3, dims=('sides', 'betas'))
    
    
    # Unconstrained for b0 and b2
    beta_0 = pm.Normal("beta_0", mu=mu_betas[:,0], sigma=sig_betas[:,0], dims=('sessions','sides'))
    
    # Constrained to be nonnegative for b1 (bc psi must be nondecreasing)
    beta_1 = pm.TruncatedNormal("beta_1", mu=mu_betas[:,1], sigma=sig_betas[:,1], lower = 0, dims=('sessions','sides'))

    
    beta_2 = pm.Normal("beta_2", mu=mu_betas[:,2], sigma=sig_betas[:,2], dims=('sessions','sides'))
   

   
    # Combine into full beta_vec with shape (sessions, betas)
    beta_vec = pm.Deterministic(
        "beta_vec",
        pm.math.stack([beta_0, beta_1, beta_2], axis=2),
        dims=('sessions','sides', 'betas')
    )
    
    
    # Linear predictor using matrix multiplication
    
    logit_p = pm.Deterministic(
        'logit_p',
        pm.math.sum(cov_mat * beta_vec[sessions_idx, sides_idx], axis=1),
        dims='trials'
    )
    
    # Probability
    p = pm.Deterministic('p', pm.math.invlogit(logit_p), dims='trials')
    
    # Likelihood
    resp = pm.Bernoulli("resp", p=p, observed=obs_data, dims='trials')

print("Model created successfully!")
print(f"Total trials across all sessions: {len(obs_data)}")