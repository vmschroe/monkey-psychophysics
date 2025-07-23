# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 19:58:28 2025

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

#%% load data

with open(r"Sirius_DFS.pkl", "rb") as f:
    data = pickle.load(f)  
    
#%% process data
x_orig = np.array([6, 12, 18, 24, 32, 38, 44, 50])
amp_mu = np.mean(x_orig)
amp_sig = np.std(x_orig)


sessions = list(data.keys())

sess_cov_mats_left = []
sess_obs_data_left = []
Nt_left = []
sess_cov_mats_right = []
sess_obs_data_right = []
Nt_right = []

for sess in sessions:
    sessdf = data[sess].copy(deep=True)[['stimSIDE', 'lowORhighGUESS','stimAMP', 'distAMP']]
    
    sessdf['stimAMP'] = (sessdf['stimAMP']-amp_mu)/amp_sig
    sessdf['distAMP'] = (sessdf['distAMP'])/amp_sig
    
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

#%%

#sessions = list(range(43))  # 43 sessions
sessions = list(data.keys())

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


#%%

pm.model_to_graphviz(hier_nolapse_model)
#%%

with hier_nolapse_model:
    prior_checks = pm.sample_prior_predictive()
    
#%%
glob = 'global_mu'
prior_glob = np.array(prior_checks.prior[glob])[0]
for i in range(3):
    plt.hist(prior_glob[:,i])
    plt.title(f'{glob} beta {i}')
    plt.show()
    
plt.hist(np.array(prior_checks.prior['mu_betas'])[0,:,0,1])
#%%
with hier_nolapse_model:
    trace = pm.sample(return_inferencedata=True, progressbar=True, idata_kwargs={"log_likelihood": True})
# if divergences, nuts_sampler_kwargs={'target_accept': 0.9}

#%%
summry = az.summary(trace, var_names = ['global_mu','global_sig','mu_betas','sig_betas'])
summry[(summry['ess_bulk']<400) | (summry['ess_tail']<400) | (summry['r_hat']>1)]

#%%
az.plot_trace(trace, var_names=['sig_betas'], compact=False)

#%%
with hier_nolapse_model:
    post_pred = pm.sample_posterior_predictive(trace, extend_inferencedata=True)
    
#%%

with hier_nolapse_model:
    trace.extend(pm.sample_prior_predictive())



#%%
az.plot_dist_comparison(trace, var_names = ['mu_betas'], kind={'latent':['posterior'], 'observed':['prior_predictive']})

az.plot_dist_comparison(trace, var_names = ['sig_betas'], kind={'latent':['posterior'], 'observed':['prior_predictive']})


#%%
var1 = np.array(trace.posterior.stack(sample=['chain', 'draw'])['global_mu'][2])
var2 = np.array(trace.posterior.stack(sample=['chain', 'draw'])['global_sig'][2])


divergent = np.array(trace.sample_stats['diverging'].stack(sample=['chain', 'draw']))
predivergent = np.roll(divergent, -1)
prepredivergent = np.roll(divergent, -2)

plt.scatter(var1, var2, color = "blue", alpha = 0.2)
plt.scatter(var1[prepredivergent], var2[prepredivergent],  color = "yellow", alpha = 0.2)
plt.scatter(var1[predivergent], var2[predivergent],  color = "orange", alpha = 0.2)
plt.scatter(var1[divergent], var2[divergent],  color = "red", alpha = 0.2)

plt.show()


#%%

p_test_pred = trace.posterior_predictive['resp'].mean(dim=["chain", "draw"])
y_test_pred = (p_test_pred >= 0.5).astype("int").to_numpy()
y_true = trace.observed_data['resp'].to_numpy()

y_naive = np.sign(cov_mat[:,1])*0.5+0.5

print(f"accuracy of model = {np.mean(y_true==y_test_pred)}")
print(f"accuracy of naive = {np.mean(y_true==y_naive)}")
