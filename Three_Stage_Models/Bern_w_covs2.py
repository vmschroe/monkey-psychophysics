# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 22:01:57 2025

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

#%% PyMC Model
coords = {
    'sessions': sessions, 
    'betas': ["b0", "b1", "b2"],
    'trials': range(len(obs_data_left))  # Add trials dimension
}

with pm.Model(coords=coords) as left_hier_nolapse_model2:
    pm.MutableData("cov_mat_left", cov_mat_left, dims=("trials", "betas"))
    pm.MutableData("sessions_idx_left", sessions_idx_left, dims="trials")
    # Hyperpriors for session-level parameters
    mu_betas = pm.Normal("mu_betas", mu=0, sigma=1, dims='betas')
    sig_betas = pm.Exponential("sig_betas", 3, dims='betas')
    
    
    # Unconstrained for b0 and b2
    beta_0 = pm.Normal("beta_0", mu=mu_betas[0], sigma=sig_betas[0], dims="sessions")
    
    beta_2 = pm.Normal("beta_2", mu=mu_betas[2], sigma=sig_betas[2], dims="sessions")
    # z_beta_2 = pm.Normal("z_beta_2", mu=0, sigma=1, dims="sessions")
    # beta_2 = pm.Deterministic("beta_2",  mu_betas[2] + z_beta_2 * sig_betas[2], dims="sessions")


    # Constrained to be nonnegative for b1
    beta_1 = pm.TruncatedNormal("beta_1", mu=mu_betas[1], sigma=sig_betas[1], lower = 0, dims="sessions")

    # Combine into full beta_vec with shape (sessions, betas)
    beta_vec = pm.Deterministic(
        "beta_vec",
        pm.math.stack([beta_0, beta_1, beta_2], axis=1),
        dims=("sessions", "betas")
    )
    
    
    # Linear predictor using matrix multiplication
    
    logit_p = pm.Deterministic(
        'logit_p',
        pm.math.sum(cov_mat_left * beta_vec[sessions_idx_left], axis=1),
        dims='trials'
    )
    
    # Probability
    p = pm.Deterministic('p', pm.math.invlogit(logit_p), dims='trials')
    
    # Likelihood
    resp = pm.Bernoulli("resp", p=p, observed=obs_data_left, dims='trials')

print("Model created successfully!")
print(f"Total trials across all sessions: {len(obs_data_left)}")

#%% 
pm.model_to_graphviz(left_hier_nolapse_model2)

#%% left hand run

with left_hier_nolapse_model2:
    left_trace2 = pm.sample(return_inferencedata=True, progressbar=True, idata_kwargs={"log_likelihood": True}, nuts_sampler_kwargs={'target_accept': 0.9})
    
    
#%%
with left_hier_nolapse_model2:
    post_pred = pm.sample_posterior_predictive(left_trace2, extend_inferencedata=True)
    

#%%

p_test_pred = left_trace2.posterior_predictive['resp'].mean(dim=["chain", "draw"])
y_test_pred = (p_test_pred >= 0.5).astype("int").to_numpy()
y_true = left_trace2.observed_data['resp'].to_numpy()

y_naive = np.sign(cov_mat_left[:,1])*0.5+0.5

print(f"accuracy of model = {np.mean(y_true==y_test_pred)}")
print(f"accuracy of naive = {np.mean(y_true==y_naive)}")




#%% plot curves
beta_post_samps = np.array(left_trace2.posterior['beta_vec']).reshape(-1,43,3)

sess = 30

sess_beta_samps = beta_post_samps[:,sess,:] #(4000,3)



sess_cov_mat = sess_cov_mats_left[sess] # shape (135,3)


xrange = [np.min(sess_cov_mat[:,1]), np.max(sess_cov_mat[:,1])]
dist_level = np.max(sess_cov_mat[:,2])

x_stim_vals = np.linspace(*xrange,100).reshape(-1,1)
x_unim = np.hstack([np.ones_like(x_stim_vals),x_stim_vals, np.zeros_like(x_stim_vals)])
x_bim = np.hstack([np.ones_like(x_stim_vals),x_stim_vals, np.full_like(x_stim_vals, dist_level)])

y_unim_samps = 1 / (1 + np.exp(-sess_beta_samps@x_unim.T)) # shape (500,100)
y_unim_HDI = az.hdi(y_unim_samps,hdi_prob=0.95).T
y_unim_mean = np.mean(y_unim_samps, axis=0)
y_bim_samps = 1 / (1 + np.exp(-sess_beta_samps@x_bim.T))
y_bim_HDI = az.hdi(y_bim_samps,hdi_prob=0.95).T
y_bim_mean = np.mean(y_bim_samps, axis=0)


plt.plot(x_stim_vals[:,0], y_unim_mean, 'b')
plt.fill_between(x_stim_vals[:,0], y_unim_HDI[0], y_unim_HDI[1], alpha = 0.1, color='blue')

plt.plot(x_stim_vals[:,0], y_bim_mean, 'r')
plt.fill_between(x_stim_vals[:,0], y_bim_HDI[0], y_bim_HDI[1], alpha = 0.1, color='red')

plt.show()

#%%


az.plot_pair(
    left_trace2,
    var_names=['mu_betas','sig_betas'], 
    marginals=True)

#%%


with left_hier_nolapse_model2:
    left_trace2.extend(pm.sample_prior_predictive())
    
#%%

az.plot_posterior(left_trace2,var_names = ['mu_betas', 'sig_betas'])
hp_post = az.summary(left_trace2,var_names = ['mu_betas', 'sig_betas'])[['mean','sd']]

az.plot_dist_comparison(left_trace2,var_names = ['sig_betas'], coords={'betas' : ['b2']}, kind={'latent':['posterior'], 'observed':['prior_predictive']})
az.plot_dist_comparison(left_trace2,var_names = ['beta_vec'], kind={'latent':['posterior'], 'observed':['prior_predictive']}, combine_dims={'sessions'})

az.plot_dist_comparison(left_trace2,var_names = ['mu_betas'], kind={'latent':['posterior'], 'observed':['prior_predictive']})


#%%
idx=2
var1 = np.array(left_trace2.posterior.stack(sample=['chain', 'draw'])['mu_betas'][idx])
var2 = np.array(left_trace2.posterior.stack(sample=['chain', 'draw'])['sig_betas'][idx])


divergent = np.array(left_trace2.sample_stats['diverging'].stack(sample=['chain', 'draw']))
predivergent = np.roll(np.array(left_trace2.sample_stats['diverging'].stack(sample=['chain', 'draw'])), -1)
prepredivergent = np.roll(np.array(left_trace2.sample_stats['diverging'].stack(sample=['chain', 'draw'])), -2)

plt.scatter(var1, var2, color = "blue", alpha = 0.2)
plt.scatter(var1[prepredivergent], var2[prepredivergent],  color = "yellow", alpha = 0.4)
plt.scatter(var1[predivergent], var2[predivergent],  color = "orange", alpha = 0.4)
plt.scatter(var1[divergent], var2[divergent],  color = "red", alpha = 0.4)

plt.show()

#%%


def pairplot_divergence(trace, ax=None, divergence=True, color="C3", divergence_color="C2"):
    theta = trace.get_values(varname="theta", combine=True)[:, 0]
    logtau = trace.get_values(varname="tau_log__", combine=True)
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(theta, logtau, "o", color=color, alpha=0.5)
    if divergence:
        divergent = trace["diverging"]
        ax.plot(theta[divergent], logtau[divergent], "o", color=divergence_color)
    ax.set_xlabel("theta[0]")
    ax.set_ylabel("log(tau)")
    ax.set_title("scatter plot between log(tau) and theta[0]")
    return ax


#%%
sessname = '03-30-S1'
b_idx = 'b2'

az.plot_trace(left_trace2, coords={'betas' : ['b2']}, var_names=['sig_betas'])


