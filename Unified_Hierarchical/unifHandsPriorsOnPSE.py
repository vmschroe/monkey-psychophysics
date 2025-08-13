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

#%%

with open(r"Unified_Hierarchical/desc_hyper_priors.pkl", "rb") as f:
    desc_hyper_priors = pickle.load(f)  
    
arr = np.array([desc_hyper_priors[col].to_list() for col in desc_hyper_priors.columns]).astype(float)
hparams = np.transpose(arr, (1, 0, 2))  # shape: (3, 2, 2)


#%%
with open(r"Unified_Hierarchical/data_unif_hier.pkl", "rb") as f:
    data_unif_hier = pickle.load(f)  


sides_idx = data_unif_hier['side_index']
obs_data = data_unif_hier['obs_data']
sessions_idx = data_unif_hier['sessions_idx']
sess_distAMP = data_unif_hier['sess_distAMP']
cov_mat = data_unif_hier['cov_mat']
sessions = data_unif_hier['sessions']

x_orig = np.array([6, 12, 18, 24, 32, 38, 44, 50])
amp_mu = np.mean(x_orig)
amp_sig = np.std(x_orig)
x_new = (x_orig-amp_mu)/amp_sig

#%% PyMC Model
coords = {
    'sides': ['left','right'],
    'sessions': sessions, 
    'betas': ["b0", "b1", "b2"],
    'trials': range(len(obs_data))  # Add trials dimension
}

with pm.Model(coords=coords) as hier_model:
    pm.MutableData("cov_mat", cov_mat, dims=("trials", "betas"))
    pm.MutableData("sessions_idx", sessions_idx, dims="trials")
    pm.MutableData("sides_idx", sides_idx, dims="trials")
    pm.MutableData("sess_distAMP", sess_distAMP, dims="sessions")

    # Hyperpriors
    # PSE unimanual
    A_glob_pse_uni = pm.Gamma("A_glob_pse_uni", alpha = hparams[0,1,0], beta = 1)
    B_glob_pse_uni = pm.Gamma("B_glob_pse_uni", alpha = hparams[0,1,1], beta = 1)
    A_hand_pse_uni = pm.Gamma("A_hand_pse_uni", alpha = pm.math.clip(A_glob_pse_uni, 1e-4, 1e4), beta = 1, dims = 'sides')
    B_hand_pse_uni = pm.Gamma("B_hand_pse_uni", alpha = pm.math.clip(B_glob_pse_uni, 1e-4, 1e4), beta = 1, dims = 'sides')
    
    z_pse_uni = pm.Beta("z_pse_uni", alpha = pm.math.clip(A_hand_pse_uni[:, None], 1e-4, 1e4), beta = pm.math.clip(B_hand_pse_uni[:, None], 1e-4, 1e4), dims = ('sides', 'sessions'))
    PSE_uni = pm.Deterministic('PSE_uni', hparams[0,0,0]+z_pse_uni*(hparams[0,0,1]-hparams[0,0,0]), dims = ('sides', 'sessions'))
    
    
    # PSE bimanual
    A_glob_pse_bi = pm.Gamma("A_glob_pse_bi", alpha = hparams[0,1,0], beta = 1)
    B_glob_pse_bi = pm.Gamma("B_glob_pse_bi", alpha = hparams[0,1,1], beta = 1)
    A_hand_pse_bi = pm.Gamma("A_hand_pse_bi", alpha = pm.math.clip(A_glob_pse_bi, 1e-4, 1e4), beta = 1, dims = 'sides')
    B_hand_pse_bi = pm.Gamma("B_hand_pse_bi", alpha = pm.math.clip(B_glob_pse_bi, 1e-4, 1e4), beta = 1, dims = 'sides')
    
    z_pse_bi = pm.Beta("z_pse_bi", alpha = pm.math.clip(A_hand_pse_bi[:, None], 1e-4, 1e4), beta = pm.math.clip(B_hand_pse_bi[:, None], 1e-4, 1e4), dims = ('sides', 'sessions'))
    PSE_bi = pm.Deterministic('PSE_bi', hparams[0,0,0]+z_pse_bi*(hparams[0,0,1]-hparams[0,0,0]), dims = ('sides', 'sessions'))
    
    # JND
    A_glob_jnd = pm.Gamma("A_glob_jnd", alpha = hparams[1,1,0], beta = 1)
    B_glob_jnd = pm.Gamma("B_glob_jnd", alpha = hparams[1,1,1], beta = 1)
    A_hand_jnd = pm.Gamma("A_hand_jnd", alpha = pm.math.clip(A_glob_jnd, 1e-4, 1e4), beta = 1, dims = 'sides')
    B_hand_jnd = pm.Gamma("B_hand_jnd", alpha = pm.math.clip(B_glob_jnd, 1e-4, 1e4), beta = 1, dims = 'sides')
    
    z_jnd = pm.Beta("z_jnd", alpha = pm.math.clip(A_hand_jnd[:, None], 1e-4, 1e4), beta = pm.math.clip(B_hand_jnd[:, None], 1e-4, 1e4), dims = ('sides', 'sessions'))
    JND = pm.Deterministic('JND', hparams[1,0,0]+z_jnd*(hparams[1,0,1]-hparams[1,0,0]), dims = ('sides', 'sessions'))
    
    # gammas
    # gam_h_uni
    A_glob_gam_h_uni = pm.Gamma("A_glob_gam_h_uni", alpha = hparams[2,1,0], beta = 1)
    B_glob_gam_h_uni = pm.Gamma("B_glob_gam_h_uni", alpha = hparams[2,1,1], beta = 1)
    A_hand_gam_h_uni = pm.Gamma("A_hand_gam_h_uni", alpha = pm.math.clip(A_glob_gam_h_uni, 1e-4, 1e4), beta = 1, dims = 'sides')
    B_hand_gam_h_uni = pm.Gamma("B_hand_gam_h_uni", alpha = pm.math.clip(B_glob_gam_h_uni, 1e-4, 1e4), beta = 1, dims = 'sides')
    
    z_gam_h_uni = pm.Beta("z_gam_h_uni", alpha = pm.math.clip(A_hand_gam_h_uni[:, None], 1e-4, 1e4), beta = pm.math.clip(B_hand_gam_h_uni[:, None], 1e-4, 1e4), dims = ('sides', 'sessions'))
    gam_h_uni = pm.Deterministic('gam_h_uni', hparams[2,0,0]+z_gam_h_uni*(hparams[2,0,1]-hparams[2,0,0]), dims = ('sides', 'sessions'))
    
    # gam_h_bi
    A_glob_gam_h_bi = pm.Gamma("A_glob_gam_h_bi", alpha = hparams[2,1,0], beta = 1)
    B_glob_gam_h_bi = pm.Gamma("B_glob_gam_h_bi", alpha = hparams[2,1,1], beta = 1)
    A_hand_gam_h_bi = pm.Gamma("A_hand_gam_h_bi", alpha = pm.math.clip(A_glob_gam_h_bi, 1e-4, 1e4), beta = 1, dims = 'sides')
    B_hand_gam_h_bi = pm.Gamma("B_hand_gam_h_bi", alpha = pm.math.clip(B_glob_gam_h_bi, 1e-4, 1e4), beta = 1, dims = 'sides')
    
    z_gam_h_bi = pm.Beta("z_gam_h_bi", alpha = pm.math.clip(A_hand_gam_h_bi[:, None], 1e-4, 1e4), beta = pm.math.clip(B_hand_gam_h_bi[:, None], 1e-4, 1e4), dims = ('sides', 'sessions'))
    gam_h_bi = pm.Deterministic('gam_h_bi', hparams[2,0,0]+z_gam_h_bi*(hparams[2,0,1]-hparams[2,0,0]), dims = ('sides', 'sessions'))
    
    # gam_l_uni
    A_glob_gam_l_uni = pm.Gamma("A_glob_gam_l_uni", alpha = hparams[2,1,0], beta = 1)
    B_glob_gam_l_uni = pm.Gamma("B_glob_gam_l_uni", alpha = hparams[2,1,1], beta = 1)
    A_hand_gam_l_uni = pm.Gamma("A_hand_gam_l_uni", alpha = pm.math.clip(A_glob_gam_l_uni, 1e-4, 1e4), beta = 1, dims = 'sides')
    B_hand_gam_l_uni = pm.Gamma("B_hand_gam_l_uni", alpha = pm.math.clip(B_glob_gam_l_uni, 1e-4, 1e4), beta = 1, dims = 'sides')
    
    z_gam_l_uni = pm.Beta("z_gam_l_uni", alpha = pm.math.clip(A_hand_gam_l_uni[:, None], 1e-4, 1e4), beta = pm.math.clip(B_hand_gam_l_uni[:, None], 1e-4, 1e4), dims = ('sides', 'sessions'))
    gam_l_uni = pm.Deterministic('gam_l_uni', hparams[2,0,0]+z_gam_l_uni*(hparams[2,0,1]-hparams[2,0,0]), dims = ('sides', 'sessions'))
    
    # gam_l_bi
    A_glob_gam_l_bi = pm.Gamma("A_glob_gam_l_bi", alpha = hparams[2,1,0], beta = 1)
    B_glob_gam_l_bi = pm.Gamma("B_glob_gam_l_bi", alpha = hparams[2,1,1], beta = 1)
    A_hand_gam_l_bi = pm.Gamma("A_hand_gam_l_bi", alpha = pm.math.clip(A_glob_gam_l_bi, 1e-4, 1e4), beta = 1, dims = 'sides')
    B_hand_gam_l_bi = pm.Gamma("B_hand_gam_l_bi", alpha = pm.math.clip(B_glob_gam_l_bi, 1e-4, 1e4), beta = 1, dims = 'sides')
    
    z_gam_l_bi = pm.Beta("z_gam_l_bi", alpha = pm.math.clip(A_hand_gam_l_bi[:, None], 1e-4, 1e4), beta = pm.math.clip(B_hand_gam_l_bi[:, None], 1e-4, 1e4), dims = ('sides', 'sessions'))
    gam_l_bi = pm.Deterministic('gam_l_bi', hparams[2,0,0]+z_gam_l_bi*(hparams[2,0,1]-hparams[2,0,0]), dims = ('sides', 'sessions'))
    
    
    # betas
    beta_0 = pm.Deterministic("beta_0", pm.math.log(3)/JND, dims = ('sides', 'sessions'))
    beta_1 = pm.Deterministic("beta_1", -beta_0*PSE_uni, dims = ('sides', 'sessions'))
    beta_2 = pm.Deterministic("beta_2", (-beta_1*PSE_bi-beta_0)/sess_distAMP[None, :]  , dims = ('sides', 'sessions'))
    beta_vec = pm.Deterministic(
        "beta_vec",
        pm.math.stack([beta_0, beta_1, beta_2], axis=2),
        dims=('sides','sessions', 'betas'))
    
    
    # lapses
    delta_h = pm.Deterministic("delta_h", (gam_h_bi-gam_h_uni)/sess_distAMP[None, :]  , dims = ('sides', 'sessions'))
    delta_l = pm.Deterministic("delta_l", (gam_l_bi-gam_l_uni)/sess_distAMP[None, :]  , dims = ('sides', 'sessions'))

    gam_h_vec = pm.Deterministic(
        "gam_h_vec",
        pm.math.stack([gam_h_uni, pm.math.zeros_like(gam_h_uni), delta_h], axis=2),
        dims=('sides','sessions','betas'))
    
    gam_l_vec = pm.Deterministic(
        "gam_l_vec",
        pm.math.stack([gam_l_uni, pm.math.zeros_like(gam_l_uni), delta_l], axis=2),
        dims=('sides','sessions','betas'))
   
    # Matrix multiplication
    logistic_arg = pm.Deterministic(
        'logistic_arg',
        pm.math.sum(cov_mat * beta_vec[sides_idx, sessions_idx], axis=1),
        dims='trials')
    
    gam_h =  pm.Deterministic(
        'gam_h',
        pm.math.sum(cov_mat * gam_h_vec[sides_idx, sessions_idx], axis=1),
        dims='trials')
    
    gam_l =  pm.Deterministic(
        'gam_l',
        pm.math.sum(cov_mat * gam_l_vec[sides_idx, sessions_idx], axis=1),
        dims='trials')
    
    # Probability
    p = pm.Deterministic('p', gam_h + (1 - gam_h - gam_l)*pm.math.invlogit(logistic_arg), dims='trials')
    
    # # Likelihood
    resp = pm.Bernoulli("resp", p=pm.math.clip(p,1e-4,1-1e-4), observed=obs_data, dims='trials')

print("Model created successfully!")
print(f"Total trials across all sessions: {len(obs_data)}")




#%%
pm.model_to_graphviz(hier_model)

#%%

with hier_model:
    print(cov_mat.shape)
    print(beta_vec[sides_idx, sessions_idx, :].eval().shape)
    print(gam_l_vec[sides_idx, sessions_idx, :].eval().shape)
    print("beta_vec shape:", beta_vec.eval().shape)
    print("beta_selected shape:", beta_vec[sides_idx, sessions_idx].eval().shape)
    print("cov_mat shape:", cov_mat.shape)
#%%

with hier_model:
    prior_checks = pm.sample_prior_predictive()

# with hier_model:
#     prior_checks = pm.sample_prior_predictive(var_names=['gam_h','gam_l','p','logistic_arg', 'beta_vec'])  # only sample p


# np.isnan(np.array(prior_checks.prior['p']))

prior_checks.prior_predictive['resp']

#%%

prior_pred_responses = np.mean(np.array(prior_checks.prior_predictive['resp']), axis = 1)[0]

az.plot_dist_comparison(prior_checks, var_names = ['gam_h_uni'], coords = {'sessions': [sessions[0], sessions[5]]})

#%%

with hier_model:
    hier_trace = pm.sample(return_inferencedata=True, cores=1, progressbar=True, idata_kwargs={"log_likelihood": True})
    
    
#%%




with hier_model:
    hier_trace.extend(pm.sample_prior_predictive())

    


#%%

with hier_model:
    post_pred = pm.sample_posterior_predictive(hier_trace, extend_inferencedata=True)

#%%
hand = 0
sess = 0
rec_params = {}

for desc in ['beta_0','beta_1', 'beta_2','gam_h_uni','gam_h_bi','gam_l_uni','gam_l_bi']:
    arr = np.array(hier_trace.posterior[desc])[0,:,hand,sess]
    rec_params[desc] = [np.mean(arr), np.std(arr)]
    
    
#%%


p_test_pred = hier_trace.posterior_predictive['resp'].mean(dim=["chain", "draw"])
y_test_pred = (p_test_pred >= 0.5).astype("int").to_numpy()
y_true = hier_trace.observed_data['resp'].to_numpy()

y_naive = np.sign(cov_mat[:,1])*0.5+0.5

print(f"accuracy of model = {np.mean(y_true==y_test_pred)}")
print(f"accuracy of naive = {np.mean(y_true==y_naive)}")

#%%

bad_pred = ((y_true==y_test_pred)==False)

bad_sess = sessions_idx[bad_pred]
bad_sides = sides_idx[bad_pred]
bad_stim = cov_mat[bad_pred,1]
bad_dist = cov_mat[bad_pred,2]

#issues with dist=0, so unimanual trials
#specifically gam_l_uni